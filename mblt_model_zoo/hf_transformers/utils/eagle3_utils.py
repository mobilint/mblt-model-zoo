"""Utilities for Mobilint EAGLE-3 runtime and tree decoding."""

from __future__ import annotations

import random
from typing import Any, Optional

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from .cache_utils import MobilintEagle3Cache


def load_embedding_override(path: str) -> torch.Tensor:
    """Load an embedding override from a torch tensor or state dict file."""
    data = torch.load(path, map_location="cpu")
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, dict):
        if "weight" in data:
            return data["weight"]
        return next(iter(data.values()))
    raise ValueError(f"Unsupported embedding override payload in {path!r}")


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
    """Return top-k probabilities and indices from logits."""
    x = logits.float()
    topk_vals, topk_idx = torch.topk(x, k, dim=-1, largest=True, sorted=True)
    if logits_processor is not None:
        topk_vals = logits_processor(None, topk_vals)
    max_val = x.max(dim=-1, keepdim=True).values
    denom = torch.exp(x - max_val).sum(dim=-1, keepdim=True)
    probs = torch.exp(topk_vals - max_val) / denom
    return probs, topk_idx


@torch.no_grad()
def initialize_tree(
    input_ids: torch.LongTensor,
    model: Any,
    cache: MobilintEagle3Cache,
    logits_processor: Optional[LogitsProcessorList],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prefill base model once and initialize the first draft tree."""
    outputs, logits = model.eagle3_base_model(
        input_ids[:, cache.get_base_seq_length() :],
        cache=cache,
        output_orig=True,
        requires_all_features_logits=False,
    )
    cache.update_base_seen_tokens(input_ids.shape[1] - cache.get_base_seq_length())

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
    )
    cache.pending_draft_tokens = draft_tokens
    cache.retrieve_indices = retrieve_indices
    cache.tree_mask = tree_mask
    cache.tree_position_ids = tree_position_ids
    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits


@torch.no_grad()
def tree_decoding(
    model: Any,
    cache: MobilintEagle3Cache,
    tree_candidates: torch.LongTensor,
    input_ids: torch.LongTensor,
    retrieve_indices: torch.LongTensor,
    tree_position_ids: torch.LongTensor,
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
    )

    if cache.accept_tokens is not None:
        cache.update_base_seen_tokens(accepted_prefix_length)
        cache.accept_tokens = None

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Choose the best accepted branch and the next sampling distribution."""
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    def select_token_logits(source: torch.Tensor, index: torch.Tensor | int) -> torch.Tensor:
        """Select a token distribution without collapsing it to a scalar."""
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
        accepted_lengths = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
        accept_length = accepted_lengths.max()
        if accept_length == 0:
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(accepted_lengths).to(torch.long)
        sample_p = path_logits[best_candidate, accept_length]
        return best_candidate, accept_length, sample_p, None

    accept_length = 1
    accept_prefix = candidates[0][:1]
    best_candidate = 0
    retrieve_idx = retrieve_indices[0, :1]
    sampled_indices: Optional[torch.Tensor] = None
    adjusted = False

    for idx in range(1, candidates.shape[1]):
        if idx != accept_length:
            break
        matching = (candidates[:, :accept_length] == accept_prefix).all(dim=1)
        topk_probs, topk_indices = softmax_topk_cpu_torch(
            select_token_logits(logits, retrieve_idx),
            10,
            logits_processor=logits_processor,
        )
        sampled_indices = topk_indices
        seen_tokens: list[int] = []
        for candidate_idx in range(candidates.shape[0]):
            if not matching[candidate_idx]:
                continue
            token = candidates[candidate_idx, idx].item()
            if token in seen_tokens or token == -1:
                continue
            seen_tokens.append(token)
            mask = topk_indices == token
            token_prob = topk_probs[mask.nonzero(as_tuple=True)[0].item()] if mask.any() else 0.0
            if random.random() <= token_prob:
                accept_prefix = torch.cat((accept_prefix, candidates[candidate_idx, idx][None]), dim=0)
                accept_length += 1
                best_candidate = candidate_idx
                retrieve_idx = retrieve_indices[candidate_idx, idx]
                break
            if mask.any():
                topk_probs[mask.nonzero(as_tuple=True)[0].item()] = 0
                topk_probs = topk_probs / topk_probs.sum()
                adjusted = True

    if adjusted and accept_length != candidates.shape[1]:
        sample_p = topk_probs
    else:
        sample_logits = select_token_logits(logits, retrieve_indices[best_candidate, accept_length - 1])
        sample_p, sampled_indices = softmax_topk_cpu_torch(
            sample_logits,
            10,
            logits_processor=logits_processor,
        )

    return torch.tensor(best_candidate), torch.tensor(accept_length - 1), sample_p, sampled_indices


@torch.no_grad()
def update_inference_inputs(
    input_ids: torch.LongTensor,
    candidates: torch.Tensor,
    best_candidate: torch.Tensor,
    accept_length: torch.Tensor,
    retrieve_indices: torch.Tensor,
    logits_processor: Optional[LogitsProcessorList],
    new_token_count: int,
    model: Any,
    cache: MobilintEagle3Cache,
    hidden_state_new: torch.Tensor,
    sample_p: torch.Tensor,
    sampled_indices: Optional[torch.Tensor],
) -> tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Advance accepted tokens and build the next draft tree."""
    accept_length_int = int(accept_length.item())
    best_candidate_int = int(best_candidate.item())
    accepted = candidates[None, best_candidate_int, : accept_length_int + 1].to(input_ids.device)
    input_ids = torch.cat([input_ids, accepted], dim=-1)
    cache.accept_tokens = accepted

    retrieved_hidden_state = hidden_state_new[:, retrieve_indices]
    accepted_hidden_state = retrieved_hidden_state[:, best_candidate_int, : accept_length_int + 1]

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
    )
    cache.pending_draft_tokens = draft_tokens
    cache.retrieve_indices = retrieve_indices
    cache.tree_mask = tree_mask
    cache.tree_position_ids = tree_position_ids
    new_token_count += accept_length_int + 1
    return input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token_count
