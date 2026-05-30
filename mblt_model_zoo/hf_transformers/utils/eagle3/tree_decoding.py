"""Tree-decoding primitives for Mobilint EAGLE-3 generation."""

from __future__ import annotations

from typing import Any, Optional, Protocol, TypeAlias

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from ..cache_utils import MobilintEagle3Cache


class MobilintEagle3GenerationProtocol(Protocol):
    """Protocol for models that expose the EAGLE-3 generation contract."""

    eagle3_base_model: Any
    eagle3_draft_model: Any
    eagle3_fc_projector: Any


PosteriorResult: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
UpdateInputsResult: TypeAlias = tuple[
    torch.LongTensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    bool,
]


def _normalize_probs_or_fallback_uniform(probs: torch.Tensor) -> torch.Tensor:
    """Normalize a probability tensor and guard against zero/NaN denominators.

    Args:
        probs: 1D probability-like tensor of dtype float.

    Returns:
        A normalized probability tensor whose sum is 1.0.
    """
    total = probs.sum()
    if not torch.isfinite(total) or total <= 0:
        if probs.numel() == 0:
            return probs
        return torch.full_like(probs, 1.0 / float(probs.numel()))
    normalized = probs / total
    if not torch.isfinite(normalized).all():
        return torch.full_like(probs, 1.0 / float(max(1, probs.numel())))
    return normalized


def _apply_logits_processor(
    logits: torch.Tensor,
    logits_processor: Optional[LogitsProcessorList],
) -> torch.Tensor:
    """Apply logits processors/warpers to logits with shape-safe handling.

    Args:
        logits: 1D or 2D logits tensor.
        logits_processor: Optional HF logits processor list.

    Returns:
        Processed logits tensor with the same shape as input.
    """
    if logits_processor is None:
        return logits
    if logits.ndim == 1:
        return logits_processor(None, logits.unsqueeze(0))[0]
    return logits_processor(None, logits)


def _commit_accept_tokens_to_base(cache: MobilintEagle3Cache) -> None:
    """Commit accepted tokens into base cache length and clear pending state."""
    if cache.accept_tokens is None:
        return
    accepted_prefix_length = int(cache.accept_tokens.shape[1])
    if accepted_prefix_length > 0:
        cache.update_base_seen_tokens(accepted_prefix_length)
    cache.accept_tokens = None


def prepare_logits_processor(
    temperature: float = 0.0,
    repetition_penalty: float = 0.0,
    top_p: float = 0.0,
    top_k: int = 0,
) -> Optional[LogitsProcessorList]:
    """Build a minimal logits processor list for EAGLE-3 generation."""
    processor_list = LogitsProcessorList()
    if temperature <= 1e-5:
        return None
    if temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list or None


def softmax_topk_cpu_torch(
    logits: torch.Tensor,
    k: int,
    logits_processor: Optional[LogitsProcessorList] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k probabilities and indices from logits.

    Args:
        logits: Logits tensor with float-like dtype.
        k: Number of top candidates.
        logits_processor: Optional HF logits processor list.

    Returns:
        Tuple of ``(probs, topk_indices)`` where:
          - ``probs``: float tensor.
          - ``topk_indices``: ``torch.long`` tensor.
    """
    x = logits.float()
    processed_logits = _apply_logits_processor(x, logits_processor)
    topk_vals, topk_idx = torch.topk(processed_logits, k, dim=-1, largest=True, sorted=True)
    max_val = processed_logits.max(dim=-1, keepdim=True).values
    denom = torch.exp(processed_logits - max_val).sum(dim=-1, keepdim=True)
    probs = torch.exp(topk_vals - max_val) / denom
    return probs, topk_idx


@torch.no_grad()
def initialize_tree(
    input_ids: torch.LongTensor,
    model: MobilintEagle3GenerationProtocol,
    cache: MobilintEagle3Cache,
    logits_processor: Optional[LogitsProcessorList],
    *,
    remaining_tokens: Optional[int] = None,
    count_npu_time: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prefill base model once and initialize the first draft tree."""
    base_seq_length = cache.get_base_seq_length()
    prompt_delta_input_ids = input_ids[:, base_seq_length:]
    if prompt_delta_input_ids.shape[1] == 0:
        raise ValueError(
            "EAGLE-3 generate received empty prompt delta. "
            "When reusing `past_key_values`, provide at least one new input token."
        )

    outputs, logits = model.eagle3_base_model(
        prompt_delta_input_ids,
        cache=cache,
        output_orig=True,
        requires_all_features_logits=False,
        count_npu_time=count_npu_time,
    )
    cache.update_base_seen_tokens(prompt_delta_input_ids.shape[1])

    if logits_processor is not None:
        probabilities, indices = softmax_topk_cpu_torch(logits, 10, logits_processor=logits_processor)
        sampled_idx = torch.multinomial(probabilities, 1)
        token = indices.gather(dim=1, index=sampled_idx)
    else:
        token = torch.argmax(logits, dim=-1, keepdim=True)

    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    hidden_states = outputs["hidden_states"][0].contiguous()
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.eagle3_draft_model.topk_generate(
        hidden_states,
        input_ids=input_ids,
        cache=cache,
        logits_processor=logits_processor,
        max_draft_tokens=None if remaining_tokens is None else max(1, remaining_tokens - 1),
        count_npu_time=count_npu_time,
    )
    cache.pending_draft_tokens = draft_tokens
    cache.retrieve_indices = retrieve_indices
    cache.tree_mask = tree_mask
    cache.tree_position_ids = tree_position_ids
    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits


@torch.no_grad()
def tree_decoding(
    model: MobilintEagle3GenerationProtocol,
    cache: MobilintEagle3Cache,
    tree_candidates: torch.LongTensor,
    input_ids: torch.LongTensor,
    retrieve_indices: torch.LongTensor,
    tree_position_ids: torch.LongTensor,
    *,
    count_npu_time: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run base-model tree decoding for the current draft tree."""
    if cache.accept_tokens is not None:
        tree_position_ids = tree_position_ids + cache.accept_tokens.shape[1]
        accept_position_ids = torch.arange(cache.accept_tokens.shape[1], device=tree_position_ids.device)
        tree_position_ids = torch.cat((accept_position_ids, tree_position_ids), dim=0)
        tree_candidates = torch.cat((cache.accept_tokens, tree_candidates), dim=1)
        accepted_prefix_length = cache.accept_tokens.shape[1]
    else:
        accepted_prefix_length = 0

    position_ids = tree_position_ids + input_ids.shape[1] - accepted_prefix_length
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)

    outputs, tree_logits = model.eagle3_base_model(
        tree_candidates,
        cache=cache,
        output_orig=True,
        position_ids=position_ids,
        requires_all_features_logits=True,
        count_npu_time=count_npu_time,
    )

    if cache.accept_tokens is not None:
        _commit_accept_tokens_to_base(cache)

    hidden_state = outputs["hidden_states"][0]
    del retrieve_indices
    tree_logits = tree_logits[:, accepted_prefix_length:]
    logits = tree_logits[0]
    return logits, hidden_state[:, accepted_prefix_length:]


@torch.no_grad()
def evaluate_posterior(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    logits_processor: Optional[LogitsProcessorList],
    retrieve_indices: torch.Tensor,
) -> PosteriorResult:
    """Choose the best accepted branch and the next sampling distribution.

    Returns:
        Tuple of ``(best_candidate, accepted_draft_count, sample_p, sampled_indices)``.
        ``best_candidate`` and ``accepted_draft_count`` are ``torch.long`` scalars.
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    def select_token_logits(source: torch.Tensor, index: torch.Tensor | int) -> torch.Tensor:
        selected = source[index]
        if selected.ndim == 0:
            return source
        if selected.ndim > 1:
            return selected[0]
        return selected

    if logits_processor is None:
        path_positions = retrieve_indices[:, :-1].to(logits.device)
        safe_positions = path_positions.clamp_min(0)
        path_logits = logits[safe_positions]
        greedy_tokens = torch.argmax(path_logits, dim=-1)
        candidate_targets = candidates[:, 1:].to(logits.device)
        valid_mask = (path_positions >= 0) & (candidate_targets >= 0)
        posterior_mask = ((candidate_targets == greedy_tokens) & valid_mask).int()
        accepted_draft_counts = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
        accepted_draft_count = accepted_draft_counts.max()
        if accepted_draft_count == 0:
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(accepted_draft_counts).to(torch.long)

        leaf_position = retrieve_indices[best_candidate, accepted_draft_count].to(logits.device)
        if 0 <= int(leaf_position.item()) < logits.shape[0]:
            sample_p = logits[leaf_position]
        else:
            sample_index = torch.clamp(accepted_draft_count, max=path_logits.shape[1] - 1)
            sample_p = path_logits[best_candidate, sample_index]
        return best_candidate, accepted_draft_count, sample_p, None

    accepted_candidate_length = 1
    accept_prefix = candidates[0][:1]
    best_candidate = 0
    retrieve_idx = retrieve_indices[0, :1]
    sampled_indices: Optional[torch.Tensor] = None
    adjusted = False

    for idx in range(1, candidates.shape[1]):
        if idx != accepted_candidate_length:
            break
        matching = (candidates[:, :accepted_candidate_length] == accept_prefix).all(dim=1)
        topk_probs, topk_indices = softmax_topk_cpu_torch(
            select_token_logits(logits, retrieve_idx),
            10,
            logits_processor=logits_processor,
        )
        sampled_indices = topk_indices
        seen_tokens: set[int] = set()
        for candidate_idx in range(candidates.shape[0]):
            if not matching[candidate_idx]:
                continue
            token = candidates[candidate_idx, idx].item()
            if token in seen_tokens or token == -1:
                continue
            seen_tokens.add(token)
            mask = topk_indices == token
            token_prob = topk_probs[mask.nonzero(as_tuple=True)[0].item()] if mask.any() else 0.0
            if torch.rand((), device=topk_probs.device) <= token_prob:
                accept_prefix = torch.cat((accept_prefix, candidates[candidate_idx, idx][None]), dim=0)
                accepted_candidate_length += 1
                best_candidate = candidate_idx
                retrieve_idx = retrieve_indices[candidate_idx, idx]
                break
            if mask.any():
                topk_probs[mask.nonzero(as_tuple=True)[0].item()] = 0
                topk_probs = _normalize_probs_or_fallback_uniform(topk_probs)
                adjusted = True

    if adjusted and accepted_candidate_length != candidates.shape[1]:
        sample_p = topk_probs
    else:
        sample_logits = select_token_logits(logits, retrieve_indices[best_candidate, accepted_candidate_length - 1])
        sample_p, sampled_indices = softmax_topk_cpu_torch(
            sample_logits,
            10,
            logits_processor=logits_processor,
        )

    accepted_draft_count = max(0, accepted_candidate_length - 1)
    return torch.tensor(best_candidate), torch.tensor(accepted_draft_count), sample_p, sampled_indices


@torch.no_grad()
def update_inference_inputs(
    input_ids: torch.LongTensor,
    candidates: torch.Tensor,
    best_candidate: torch.Tensor,
    accepted_draft_count: torch.Tensor,
    retrieve_indices: torch.Tensor,
    logits_processor: Optional[LogitsProcessorList],
    new_token_count: int,
    model: MobilintEagle3GenerationProtocol,
    cache: MobilintEagle3Cache,
    hidden_state_new: torch.Tensor,
    sample_p: torch.Tensor,
    sampled_indices: Optional[torch.Tensor],
    *,
    remaining_tokens: Optional[int] = None,
    eos_token_id: Optional[int | list[int]] = None,
    count_npu_time: bool = False,
) -> tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
    """Advance accepted tokens and build the next draft tree.

    Args:
        input_ids: ``torch.long`` prompt+generated token IDs.
        candidates: Candidate token tree tensor (token IDs are expected to be integer-like).
        sample_p: Sampling distribution tensor (float).

    Returns:
        Updated generation state tuple.
    """
    accepted_draft_count_int = int(accepted_draft_count.item())
    best_candidate_int = int(best_candidate.item())
    accepted_candidate_length = accepted_draft_count_int + 1
    accepted = candidates[None, best_candidate_int, :accepted_candidate_length].to(input_ids.device)
    if remaining_tokens is not None:
        accepted = accepted[:, : max(0, remaining_tokens)]
    should_stop = accepted.numel() == 0
    if eos_token_id is not None and accepted.numel() > 0:
        eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
        eos_positions = [index for index, token in enumerate(accepted[0].tolist()) if token in eos_ids]
        if eos_positions:
            accepted = accepted[:, : eos_positions[0] + 1]
            should_stop = True
    input_ids = torch.cat([input_ids, accepted], dim=-1)
    cache.accept_tokens = accepted
    new_token_count += int(accepted.shape[1])
    if should_stop or (remaining_tokens is not None and int(accepted.shape[1]) >= remaining_tokens):
        _commit_accept_tokens_to_base(cache)
        return (
            input_ids,
            torch.empty(0, dtype=torch.long, device=input_ids.device),
            retrieve_indices,
            torch.empty(0),
            torch.empty(0),
            new_token_count,
            True,
        )

    retrieved_hidden_state = hidden_state_new[:, retrieve_indices]
    accepted_hidden_state = retrieved_hidden_state[:, best_candidate_int, : int(accepted.shape[1])]

    if logits_processor is not None:
        assert sampled_indices is not None
        sampled_idx = torch.multinomial(sample_p, 1)
        token = sampled_indices[None, sampled_idx]
    else:
        token = torch.argmax(sample_p, dim=-1, keepdim=True)[None]

    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.eagle3_draft_model.topk_generate(
        accepted_hidden_state,
        input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
        cache=cache,
        logits_processor=logits_processor,
        max_draft_tokens=None if remaining_tokens is None else max(1, remaining_tokens - int(accepted.shape[1]) - 1),
        count_npu_time=count_npu_time,
    )
    cache.pending_draft_tokens = draft_tokens
    cache.retrieve_indices = retrieve_indices
    cache.tree_mask = tree_mask
    cache.tree_position_ids = tree_position_ids
    return input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token_count, False
