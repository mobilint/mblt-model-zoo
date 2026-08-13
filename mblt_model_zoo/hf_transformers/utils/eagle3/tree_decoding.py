"""Tree-decoding primitives for Mobilint EAGLE-3 generation."""

from __future__ import annotations

import os
from typing import Any, Literal, Optional, Protocol, TypeAlias

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.utils import logging

from ..cache_utils import MobilintEagle3Cache

logger = logging.get_logger(__name__)

SoftmaxTopkMode: TypeAlias = Literal["auto", "full", "sliced"]
_VALID_SOFTMAX_TOPK_MODES: tuple[SoftmaxTopkMode, ...] = ("auto", "full", "sliced")
_SOFTMAX_TOPK_MODE_ENV_VAR = "MBLT_EAGLE3_SOFTMAX_TOPK_MODE"

_SLICED_DEPRECATION_MESSAGE = (
    "EAGLE-3 softmax_topk_cpu_torch: %r mode is deprecated. It applies the logits processor "
    "list on top of an arbitrary top-N slice, which does not implement HF nucleus sampling "
    "when a TopPLogitsWarper is present without a TopKLogitsWarper. Prefer %r (default) or "
    "%r for HF-equivalent behavior."
)


def _read_softmax_topk_mode_from_env() -> SoftmaxTopkMode:
    """Read the mode from the environment, defaulting to ``auto`` when unset or invalid."""
    raw = os.environ.get(_SOFTMAX_TOPK_MODE_ENV_VAR, "").strip().lower()
    if raw in _VALID_SOFTMAX_TOPK_MODES:
        return raw  # type: ignore[return-value]
    return "auto"


# Runtime-selectable mode for ``softmax_topk_cpu_torch``.
#
# ``auto`` (default) dispatches per-call:
#   * If a ``TopKLogitsWarper`` is present, slice to top-K first and apply the processor list
#     on that slice — mathematically identical to the full-vocab HF path whenever the standard
#     Temperature/TopK/TopP warpers are used, *except* when a boundary tie pushes part of the
#     active support outside the slice; in that case we fall back to the full-vocab path so
#     the denominator matches HF's strict-less-than TopK filter exactly.
#   * Otherwise, take the full-vocab path so that a bare ``TopPLogitsWarper`` still determines
#     its nucleus from the whole distribution.
# ``full`` forces the full-vocab path unconditionally (manual override).
# ``sliced`` is a deprecated back-compat mode that always renormalizes over a top-``max_return_k``
# slice; it is retained only for A/B reproducibility.
#
# The value is initialized from ``MBLT_EAGLE3_SOFTMAX_TOPK_MODE`` at module import; call
# :func:`set_softmax_topk_mode` to override it programmatically.
SOFTMAX_TOPK_MODE: SoftmaxTopkMode = _read_softmax_topk_mode_from_env()

_last_logged_softmax_topk_mode: Optional[SoftmaxTopkMode] = None

if SOFTMAX_TOPK_MODE == "sliced":
    logger.warning(_SLICED_DEPRECATION_MESSAGE, "sliced", "auto", "full")


def set_softmax_topk_mode(mode: SoftmaxTopkMode) -> None:
    """Override the softmax top-k mode used by :func:`softmax_topk_cpu_torch`.

    Args:
        mode: One of ``"auto"`` (default; dispatches per-call based on the processor list),
            ``"full"`` (force full-vocab softmax), or ``"sliced"`` (deprecated legacy top-N
            renormalization).

    Raises:
        ValueError: If ``mode`` is not a supported value.
    """
    global SOFTMAX_TOPK_MODE
    if mode not in _VALID_SOFTMAX_TOPK_MODES:
        raise ValueError(
            f"Unsupported softmax top-k mode {mode!r}; expected one of {_VALID_SOFTMAX_TOPK_MODES}."
        )
    if mode == "sliced":
        logger.warning(_SLICED_DEPRECATION_MESSAGE, "sliced", "auto", "full")
    SOFTMAX_TOPK_MODE = mode


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
    """Build a minimal logits processor list for EAGLE-3 generation.

    Ordering matches Hugging Face's ``_get_logits_processor``/``_get_logits_warper`` contract
    (repetition penalty → temperature → top-k → top-p) so that :func:`softmax_topk_cpu_torch`
    can safely apply the list on top of a top-k slice when a ``TopKLogitsWarper`` is present.
    """
    processor_list = LogitsProcessorList()
    if temperature <= 1e-5:
        return None
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    return processor_list or None


def _extract_top_k_from_processor(
    logits_processor: Optional[LogitsProcessorList],
) -> Optional[int]:
    """Return the ``top_k`` value from the first ``TopKLogitsWarper`` in the list, or ``None``."""
    if logits_processor is None:
        return None
    for warper in logits_processor:
        if isinstance(warper, TopKLogitsWarper):
            return int(warper.top_k)
    return None


def _softmax_topk_full_vocab(
    logits: torch.Tensor,
    max_return_k: int,
    logits_processor: Optional[LogitsProcessorList],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the processor list on full-vocab logits, softmax with a full-vocab denominator."""
    x = logits.float()
    processed = _apply_logits_processor(x, logits_processor)
    return_k = min(int(max_return_k), processed.shape[-1])
    max_val = processed.max(dim=-1, keepdim=True).values
    denom = torch.exp(processed - max_val).sum(dim=-1, keepdim=True)
    topk_vals, topk_idx = torch.topk(processed, return_k, dim=-1, largest=True, sorted=True)
    probs = torch.exp(topk_vals - max_val) / denom
    return probs, topk_idx


def _softmax_topk_sliced_by_top_k(
    logits: torch.Tensor,
    max_return_k: int,
    logits_processor: Optional[LogitsProcessorList],
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice to top-K first, apply the processor list on the slice, softmax over the slice.

    Mathematically equivalent to :func:`_softmax_topk_full_vocab` for HF-ordered
    Temperature/TopK/TopP warpers because entries outside the K slice would be set to
    ``-inf`` by ``TopKLogitsWarper`` and contribute zero to the softmax denominator —
    *except* when boundary ties push part of the active support outside the slice. HF's
    ``TopKLogitsWarper`` uses a strict-less-than filter (``scores < kth_threshold``), so
    every entry whose logit equals the k-th threshold survives, while ``torch.topk``
    picks a fixed count and drops arbitrary tied entries at the boundary. When such a
    boundary tie is detected we fall back to :func:`_softmax_topk_full_vocab` so the
    denominator and returned candidate probabilities match HF exactly.
    """
    x = logits.float()
    vocab = x.shape[-1]
    slice_size = min(max(int(top_k), int(max_return_k)), vocab)
    return_k = min(int(max_return_k), vocab)
    topk_vals, topk_idx = torch.topk(x, slice_size, dim=-1, largest=True, sorted=True)
    # Detect boundary ties: HF's ``TopKLogitsWarper`` keeps every entry ``>=`` the k-th
    # threshold, but ``torch.topk`` picks exactly ``slice_size`` entries. When more full-vocab
    # entries share the threshold value than fit inside the slice, part of the active support
    # falls outside the slice and the sliced softmax denominator diverges from the full-vocab
    # one. The check is a single ``>=`` compare + reduce over the vocab dimension, which is far
    # cheaper than the full-vocab ``exp`` we take on fallback.
    if slice_size < vocab:
        threshold = topk_vals[..., -1:]
        total_at_or_above = (x >= threshold).sum(dim=-1)
        if bool((total_at_or_above > slice_size).any().item()):
            return _softmax_topk_full_vocab(logits, max_return_k, logits_processor)
    processed_slice = _apply_logits_processor(topk_vals, logits_processor)
    max_val = processed_slice.max(dim=-1, keepdim=True).values
    exp_slice = torch.exp(processed_slice - max_val)
    denom = exp_slice.sum(dim=-1, keepdim=True)
    probs_slice = exp_slice / denom
    if slice_size == return_k:
        return probs_slice, topk_idx
    _, order = torch.topk(processed_slice, return_k, dim=-1, largest=True, sorted=True)
    probs = probs_slice.gather(-1, order)
    idx = topk_idx.gather(-1, order)
    return probs, idx


def _softmax_topk_legacy_sliced(
    logits: torch.Tensor,
    max_return_k: int,
    logits_processor: Optional[LogitsProcessorList],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy top-``max_return_k`` slice + renormalization (retained for A/B reproducibility)."""
    x = logits.float()
    return_k = min(int(max_return_k), x.shape[-1])
    topk_vals, topk_idx = torch.topk(x, return_k, dim=-1, largest=True, sorted=True)
    processed_vals = _apply_logits_processor(topk_vals, logits_processor)
    max_val = processed_vals.max(dim=-1, keepdim=True).values
    exp_vals = torch.exp(processed_vals - max_val)
    probs = exp_vals / exp_vals.sum(dim=-1, keepdim=True)
    return probs, topk_idx


def softmax_topk_cpu_torch(
    logits: torch.Tensor,
    max_return_k: int = 10,
    *,
    logits_processor: Optional[LogitsProcessorList] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the top-``max_return_k`` ``(probs, indices)`` for candidate matching.

    **This function is intended for candidate matching only, NOT for sampling.** In the
    default ``"auto"`` / ``"full"`` paths the returned probabilities sum to less than 1
    when the returned slice does not cover the entire active support of the processed
    distribution, so passing them directly to ``torch.multinomial`` would silently
    renormalize over the returned slice and assign zero mass to every token outside it.
    Use :func:`_sample_next_token_from_processor` to sample the root or next token from
    the full-vocab processed distribution.

    ``max_return_k`` is a return-slice size (for downstream candidate matching), *not* a math
    slice for the softmax. The mathematical path is chosen by :data:`SOFTMAX_TOPK_MODE`:

    - ``"auto"`` (default): dispatch per call.
        * If ``logits_processor`` contains a ``TopKLogitsWarper``, slice the raw logits to the
          declared top-K first and apply the processor list on that slice. HF ordering
          (Temperature → TopK → TopP) makes the slice mathematically identical to full-vocab
          softmax while skipping the full-vocab ``exp`` — *except* when boundary ties would
          push part of the active support outside the slice, in which case the sliced path
          transparently falls back to the full-vocab path so the denominator matches HF's
          strict-less-than TopK filter exactly.
        * Otherwise (no TopK; TopP or bare Temperature only), take the full-vocab path so that
          ``TopPLogitsWarper`` determines its nucleus from the whole distribution.
    - ``"full"``: force full-vocab softmax and processor application. Manual override.
    - ``"sliced"`` (deprecated): unconditionally slice to top-``max_return_k`` first and
      renormalize over that slice. Retained for A/B reproducibility only; incorrect for
      ``TopPLogitsWarper`` without a companion ``TopKLogitsWarper``.

    Args:
        logits: Float-like logits tensor.
        max_return_k: Number of top candidates to return. Defaults to ``10``.
        logits_processor: Optional HF logits processor list.

    Returns:
        Tuple ``(probs, indices)`` where ``probs`` is a float tensor and ``indices`` is a
        ``torch.long`` tensor.
    """
    global _last_logged_softmax_topk_mode
    mode = SOFTMAX_TOPK_MODE
    if mode != _last_logged_softmax_topk_mode:
        logger.info("EAGLE-3 softmax_topk_cpu_torch mode = %r", mode)
        _last_logged_softmax_topk_mode = mode

    if mode == "full":
        return _softmax_topk_full_vocab(logits, max_return_k, logits_processor)
    if mode == "sliced":
        return _softmax_topk_legacy_sliced(logits, max_return_k, logits_processor)

    top_k = _extract_top_k_from_processor(logits_processor)
    if top_k is not None:
        return _softmax_topk_sliced_by_top_k(logits, max_return_k, logits_processor, top_k)
    return _softmax_topk_full_vocab(logits, max_return_k, logits_processor)


def _sample_next_token_from_processor(
    logits: torch.Tensor,
    logits_processor: LogitsProcessorList,
) -> torch.LongTensor:
    """Sample a single token from the full-vocab HF-processed distribution.

    This is the sampling counterpart to :func:`softmax_topk_cpu_torch`. The latter returns a
    top-N slice whose masses do not sum to 1, so feeding it to ``torch.multinomial`` would
    renormalize over just the returned slice and give every token outside the slice zero
    probability. That silently diverges from Hugging Face's ``LogitsProcessorList`` +
    ``softmax`` + ``multinomial`` contract whenever the effective active support extends
    past the return slice (temperature-only, or ``top_k`` larger than the return slice).

    Args:
        logits: Float-like logits of shape ``(vocab,)`` or ``(batch, vocab)``.
        logits_processor: HF logits processor list; must not be ``None``.

    Returns:
        ``torch.long`` sampled token. Shape is ``(1,)`` for 1D input or ``(batch, 1)`` for
        2D input.
    """
    input_ndim = logits.ndim
    x = logits.float()
    x2d = x.unsqueeze(0) if input_ndim == 1 else x
    processed = _apply_logits_processor(x2d, logits_processor)
    probs = torch.softmax(processed, dim=-1)
    token = torch.multinomial(probs, 1)
    if input_ndim == 1:
        return token.squeeze(0).to(torch.long)
    return token.to(torch.long)


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
    # Support both HF-style delta-only inputs and full-sequence inputs when cache is reused.
    if input_ids.shape[1] > base_seq_length:
        prompt_delta_input_ids = input_ids[:, base_seq_length:]
    else:
        prompt_delta_input_ids = input_ids
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
        token = _sample_next_token_from_processor(logits, logits_processor)
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

    The next-token sampling schema encoded in the returned tuple is:

    - Greedy (``logits_processor is None``): ``sample_p`` holds the raw logits row at the
      leaf position (or the fallback node), ``sampled_indices`` is ``None``. Callers
      argmax on ``sample_p``.
    - Sampling clean-accept (``logits_processor`` present, no rejection ever triggered):
      ``sample_p`` holds the raw next-root logits row, ``sampled_indices`` is ``None``.
      Callers sample from the full processed distribution via
      :func:`_sample_next_token_from_processor`.
    - Sampling rejection-adjusted (``logits_processor`` present, at least one draft token
      was rejected mid-tree): ``sample_p`` is the top-N rejection-renormalized probability
      vector and ``sampled_indices`` is the aligned top-N token id vector. Callers sample
      from that top-N distribution. This is a partial approximation of true rejection
      sampling (the redistribution stays within the top-N support), retained for
      compatibility until a full-vocab rejection-sampling algorithm lands.

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
        # Argmax once over the full tree (n_tree_nodes, vocab) instead of the fancy-indexed
        # (n_cand, depth-1, vocab) view; the latter materializes a 30 MB copy per iteration
        # for Qwen3-4B (~152k vocab) and dominates greedy-path CPU cost.
        greedy_tokens_per_node = torch.argmax(logits, dim=-1)
        greedy_tokens = greedy_tokens_per_node[safe_positions]
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
            sample_index = torch.clamp(accepted_draft_count, max=safe_positions.shape[1] - 1)
            fallback_node = safe_positions[best_candidate, sample_index]
            sample_p = logits[fallback_node]
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
        # Rejection-adjusted mid-tree stop: sample from the top-N renormalized distribution
        # (partial approximation of full rejection sampling). ``sampled_indices`` was
        # captured inside the loop and stays aligned with ``topk_probs``.
        sample_p = topk_probs
    else:
        # Clean-accept (no rejection ever triggered): hand the raw next-root logits row
        # back to the caller so sampling happens over the full-vocab processed
        # distribution, matching HF's ``LogitsProcessorList`` + ``softmax`` + ``multinomial``
        # contract instead of renormalizing over the top-N candidate-matching slice.
        sample_p = select_token_logits(logits, retrieve_indices[best_candidate, accepted_candidate_length - 1])
        sampled_indices = None

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
        if sampled_indices is None:
            # Clean-accept path: sample from the full-vocab HF-processed distribution.
            # ``sample_p`` is the raw next-root logits row here (see ``evaluate_posterior``).
            token = _sample_next_token_from_processor(sample_p, logits_processor)[None]
        else:
            # Rejection-adjusted path: fall back to the top-N renormalized distribution
            # produced by ``evaluate_posterior``. Partial approximation until a full-vocab
            # rejection-sampling algorithm lands.
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
