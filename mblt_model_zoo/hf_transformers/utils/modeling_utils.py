import logging
import math
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union, cast

import numpy as np
import qbruntime
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

from ...utils.npu_backend import MobilintNPUBackend
from ..utils.cache_utils import MobilintCache
from .base_utils import PretrainedOnlyMixin
from .configuration_utils import MobilintConfigMixin, MobilintEncoderDecoderConfigMixin

logger = logging.getLogger(__name__)


_MXQ_DYNAMIC_AXIS_SENTINEL = -1
_MXQ_TOKEN_AXIS_INDEX = -2


class MobilintModelMixin(PretrainedOnlyMixin, PreTrainedModel):
    npu_backend_prefix: Literal["", "encoder_", "decoder_", "base_", "draft_", "fc_"] = ""
    _DEFAULT_PREFILL_CHUNK_SIZE = 128

    def __init__(self, config: Union[MobilintConfigMixin, MobilintEncoderDecoderConfigMixin], *args, **kwargs):
        no_launch = kwargs.pop("no_launch", False)

        super().__init__(config, *args, **kwargs)

        if TYPE_CHECKING:
            self.config = config

        assert self.config.name_or_path is not None, "config.name_or_path is None!"

        # Used for benchmark
        self.npu_time = None
        self.reset_npu_timing()

        self.npu_backend: MobilintNPUBackend = self.config.__getattribute__(self.npu_backend_prefix + "npu_backend")
        self.npu_backend.name_or_path = self.config.name_or_path
        revision = getattr(self.config, "revision", None)
        if revision:
            self.npu_backend.revision = revision
        commit_hash = getattr(self.config, "_commit_hash", None)
        if commit_hash:
            self.npu_backend._commit_hash = commit_hash
        self.npu_backend.create()
        if not no_launch:
            self.launch()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        embedding_weight_path = kwargs.pop("embedding_weight", None)
        revision = kwargs.get("revision", None)

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        if embedding_weight_path:
            cls._inject_custom_embeddings(model, embedding_weight_path)

        if hasattr(model, "npu_backend"):
            if revision is not None:
                setattr(model.npu_backend, "revision", revision)
            commit_hash = getattr(getattr(model, "config", None), "_commit_hash", None)
            if commit_hash:
                setattr(model.npu_backend, "_commit_hash", commit_hash)

        return model

    @staticmethod
    def _inject_custom_embeddings(model: PreTrainedModel, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Custom embedding path not found: {path}")

        print(f"[Mobilint] Loading custom embeddings from: {path}")

        custom_data = torch.load(path, map_location="cpu")

        # Handle dict (state_dict) vs Tensor
        if isinstance(custom_data, dict):
            # Try to find common keys for weights
            if "weight" in custom_data:
                new_weight = custom_data["weight"]
            else:
                # If ambiguous, take the first value
                new_weight = next(iter(custom_data.values()))
        elif isinstance(custom_data, torch.Tensor):
            new_weight = custom_data
        else:
            raise ValueError(f"Unsupported data format in {path}. Expected dict or Tensor.")

        input_embeddings: nn.Embedding = cast(nn.Embedding, model.get_input_embeddings())

        original_vocab_size = input_embeddings.weight.shape[0]
        new_vocab_size = new_weight.shape[0]
        embed_dim = input_embeddings.weight.shape[1]

        if new_weight.shape[1] != embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch! Model expects {embed_dim}, but file has {new_weight.shape[1]}"
            )

        if original_vocab_size != new_vocab_size:
            raise ValueError(f"Vocab size mismatch! Model expects {original_vocab_size}, but file has {new_vocab_size}")

        with torch.no_grad():
            input_embeddings.weight.data = new_weight.to(
                device=input_embeddings.weight.device, dtype=input_embeddings.weight.dtype
            )

        print("[Mobilint] Custom embeddings successfully injected.")

    def launch(self):
        self.npu_backend.launch()

    def dispose(self):
        self.npu_backend.dispose()

    def get_mxq_model(self):
        return self.npu_backend.mxq_model

    def reset_npu_timing(self) -> None:
        """Reset aggregate NPU timing counters used by TPS benchmarks."""
        self.npu_timing = {
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "prefill_calls": 0,
            "decode_calls": 0,
        }

    def get_npu_timing(self) -> dict[str, float | int]:
        """Return aggregate NPU timing counters used by TPS benchmarks."""
        timing = getattr(self, "npu_timing", None)
        if not isinstance(timing, dict):
            self.reset_npu_timing()
            timing = self.npu_timing
        return dict(timing)

    def _record_npu_timing(self, phase: Literal["prefill", "decode"], elapsed: float) -> None:
        """Accumulate one MXQ inference elapsed time without retaining per-token records."""
        timing = getattr(self, "npu_timing", None)
        if not isinstance(timing, dict):
            self.reset_npu_timing()
            timing = self.npu_timing
        timing[f"{phase}_time"] = float(timing.get(f"{phase}_time", 0.0)) + float(elapsed)
        timing[f"{phase}_calls"] = int(timing.get(f"{phase}_calls", 0)) + 1

    def _warn_last_only_slow_path_once(self) -> None:
        """Emit a one-shot WARNING when a call enters the last-only MXQ Path 3.

        Path 3 runs one MXQ infer per kept position, so a manual
        ``.forward()`` with the default ``logits_to_keep=0`` on a
        last-only compiled model is dramatically slower than the
        HF-generate case (which sets ``logits_to_keep=1``). Docstrings
        alone don't surface this; a stderr warning does. Fires at most
        once per instance so long training / eval loops don't spam.
        """
        if getattr(self, "_mxq_last_only_slow_path_warned", False):
            return
        logger.warning(
            "Non-default `logits_to_keep` on a last-only MXQ build triggers a "
            "fallback that runs one MXQ infer per kept position and is "
            "dramatically slower than the default on long prefills. Pass "
            "`logits_to_keep=1` for last-token workloads; HF `.generate()` "
            "already sets this automatically, so this warning does not "
            "indicate a bug in generation. This message fires once per "
            "model instance."
        )
        self._mxq_last_only_slow_path_warned = True

    def _mxq_supports_all_logits(self) -> bool:
        """Return True when the compiled MXQ emits logits for every input token.

        Some MXQ builds compile the LM head with a dynamic token axis, so a
        single ``mxq_model.infer`` call returns per-position logits instead
        of only the last-token logit. We probe this by inspecting the first
        entry of :py:meth:`qbruntime.Model.get_model_output_shape`: the
        token axis lives at ``_MXQ_TOKEN_AXIS_INDEX`` (the second-to-last
        dimension, matching both ``(batch, seq_len, vocab)`` and
        ``(batch, 1, seq_len, vocab)`` layouts we see in practice), and a
        value equal to ``_MXQ_DYNAMIC_AXIS_SENTINEL`` marks it as dynamic.
        The result is cached on the instance because the shape is fixed
        for the lifetime of the loaded model.
        """
        cached = getattr(self, "_mxq_all_logits_cached", None)
        if cached is not None:
            return cached
        supports = False
        try:
            output_shapes = self.npu_backend.mxq_model.get_model_output_shape()
            if output_shapes:
                first_shape = tuple(output_shapes[0])
                if (
                    len(first_shape) >= abs(_MXQ_TOKEN_AXIS_INDEX)
                    and int(first_shape[_MXQ_TOKEN_AXIS_INDEX]) == _MXQ_DYNAMIC_AXIS_SENTINEL
                ):
                    supports = True
        except (AttributeError, qbruntime.QbRuntimeError):
            # AttributeError: backend or ``get_model_output_shape`` missing.
            # QbRuntimeError: backend refused the probe. Only backend-specific
            # probe failures are swallowed; any other exception is a real bug
            # and should propagate.
            supports = False
        self._mxq_all_logits_cached = supports
        return supports

    def _mxq_static_vocab_size(self) -> int:
        """Return the compiled MXQ vocab dimension (last output-shape axis).

        Symmetric with :meth:`_mxq_supports_all_logits`: probes the backend
        once and caches the result on the instance because the compiled
        vocab dim is fixed for the lifetime of the model. Path 2 in
        ``_llm_forward_batch`` needs this value to synthesize placeholder
        rows for zero-length batch items and would otherwise call
        ``mxq_model.get_model_output_shape()`` on every batched forward.

        Unlike the boolean probe, there is no safe fallback for an unknown
        vocab size: the same ``(AttributeError, qbruntime.QbRuntimeError)``
        probe-failure classes recognized by :meth:`_mxq_supports_all_logits`
        propagate through here rather than being swallowed into a sentinel
        — callers see the underlying backend error, and nothing is cached
        so a later successful probe can still be recorded. Any other
        exception is a real bug and also propagates.
        """
        cached = getattr(self, "_mxq_static_vocab_cached", None)
        if cached is not None:
            return cached
        output_shapes = self.npu_backend.mxq_model.get_model_output_shape()
        vocab = int(output_shapes[0][-1])
        assert vocab > 0, (
            "MXQ vocab dim must be static (>0); got %d (mblt_model_zoo assumes the LM head vocab axis is compiled statically even when the token axis is dynamic)"
            % vocab
        )
        self._mxq_static_vocab_cached = vocab
        return vocab

    @staticmethod
    def _is_last_only_selector(
        logits_to_keep: Union[int, torch.Tensor], seq_len: int
    ) -> bool:
        """Return True when ``logits_to_keep`` semantically selects only the last position.

        Matches ``logits_to_keep == 1`` (the int shortcut) and its tensor
        equivalents: a single-element tensor holding ``seq_len - 1`` or
        ``-1``. HF callers that build the selector programmatically as
        ``torch.tensor([seq_len - 1])`` are asking for the same last-token
        logit as ``logits_to_keep=1`` and should hit Path 1 identically
        rather than probing the dynamic axis and taking Path 2/3.
        """
        if isinstance(logits_to_keep, int) and logits_to_keep == 1:
            return True
        if isinstance(logits_to_keep, torch.Tensor) and logits_to_keep.numel() == 1:
            idx = int(logits_to_keep.flatten()[0].item())
            return idx == seq_len - 1 or idx == -1
        return False

    @staticmethod
    def _wrap_and_validate_indices(indices: list[int], seq_len: int) -> list[int]:
        """Apply HF-style negative-wrap and range-check to a list of position indices.

        HF's ``hidden_states[:, tensor, :]`` fancy-indexing wraps negatives
        (``-1 -> seq_len-1``) and raises on out-of-range values. We mirror
        that: silently dropping out-of-range indices is a footgun because
        the caller cannot tell whether their intent was satisfied.
        """
        wrapped: list[int] = []
        for raw_index in indices:
            idx = int(raw_index)
            if idx < 0:
                idx = seq_len + idx
            if not 0 <= idx < seq_len:
                raise IndexError(
                    f"logits_to_keep index {int(raw_index)} out of range for seq_len={seq_len}"
                )
            wrapped.append(idx)
        return wrapped

    @classmethod
    def _output_positions_for_logits_to_keep(
        cls,
        logits_to_keep: Union[int, torch.Tensor],
        seq_len: int,
    ) -> list[int]:
        """Position indices to place in the final logits output, in HF order.

        Mirrors HF's ``hidden_states[:, logits_to_keep, :]`` semantics for
        tensor inputs: caller-supplied order is preserved and duplicates
        are kept (a repeated index picks the same position twice).

        Integer semantics: ``0`` keeps every position (HF keep-all);
        ``n > 0`` keeps the last ``n`` positions (clamped to ``seq_len``,
        so ``n >= seq_len`` also keeps every position). Negative ints are
        rejected with :class:`ValueError`: silently mapping them to
        keep-all is a footgun (the caller probably meant ``torch.tensor``
        negative-wrap indexing), and HF itself does not accept them.
        """
        if isinstance(logits_to_keep, torch.Tensor):
            raw = logits_to_keep.detach().to("cpu").flatten().tolist()
            return cls._wrap_and_validate_indices(raw, seq_len)
        n = int(logits_to_keep)
        if n < 0:
            raise ValueError(
                "logits_to_keep must be a non-negative int (0 keeps all positions, "
                "n>0 keeps last n) or a torch.Tensor of positions; got %d" % n
            )
        if n == 0 or n >= seq_len:
            return list(range(seq_len))
        return list(range(seq_len - n, seq_len))

    @classmethod
    def _walk_positions_for_logits_to_keep(
        cls,
        logits_to_keep: Union[int, torch.Tensor],
        seq_len: int,
    ) -> list[int]:
        """Unique kept positions in ascending order, for KV-cursor walks.

        The Path 3 fallback interleaves prefill and size-1 infer calls
        while advancing a monotonic cursor through the sequence, so it
        needs a sorted, deduped set of positions even when the caller's
        tensor contains duplicates or arrives out of order. Assembly of
        the final output tensor uses :meth:`_output_positions_for_logits_to_keep`
        instead so duplicates and caller order survive.
        """
        return sorted(set(cls._output_positions_for_logits_to_keep(logits_to_keep, seq_len)))

    def mxq_forward(
        self,
        input: torch.Tensor,
    ):
        input_numpy = input.type(torch.float32).cpu().numpy()

        result = self.npu_backend.mxq_model.infer([input_numpy])
        assert result is not None, "mxq infer result is None!"

        output = torch.tensor(result[0], dtype=input.dtype, device=input.device)

        return output

    def _run_chunked_logits_to_keep(
        self,
        *,
        do_infer: Callable[[int, int], np.ndarray],
        seq_len: int,
        prefill_chunk_size: int,
        logits_to_keep: Union[int, torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Shared 3-path dispatch for HF-style ``logits_to_keep`` over a chunked MXQ.

        Callers provide a ``do_infer(start, end)`` closure that runs one
        MXQ infer over the ``[start, end)`` slice of their own input(s)
        (advancing the KV cache) and returns the raw logits ndarray. This
        helper handles the three execution paths uniformly:

        1. ``logits_to_keep == 1``: fast path; the caller's chunk size is
           used as-is and only the last chunk's logits are returned.
        2. MXQ has a dynamic token axis (see
           :meth:`_mxq_supports_all_logits`): keep the caller's chunk size
           and concatenate per-chunk outputs, then slice to the requested
           positions along the token axis (``ndim - 2``).
        3. Otherwise (last-only MXQ + non-default request): a fallback that
           prefills the non-kept prefix with normal chunks (throwing away
           logits, only advancing the KV cache) and runs a size-1 infer at
           each kept position to capture its logits.

        The returned tensor still carries the leading batch axis produced
        by ``do_infer``; single-input callers typically apply
        ``.squeeze(0)`` afterwards, dual-input callers leave it as-is.
        """
        is_default_keep = self._is_last_only_selector(logits_to_keep, seq_len)
        supports_all = False if is_default_keep else self._mxq_supports_all_logits()

        # Path 1 & 2: chunked pass with caller's prefill_chunk_size. Path 2
        # additionally collects every chunk's output so the concatenation
        # covers the full input.
        if is_default_keep or supports_all:
            num_of_chunks = math.ceil(seq_len / prefill_chunk_size)
            assert num_of_chunks > 0, "num_of_chunks is not positive! num_of_chunks: %d" % num_of_chunks

            per_chunk_logits: list[np.ndarray] = []
            logits_ndarray: Optional[np.ndarray] = None
            for i in range(num_of_chunks):
                start_index = i * prefill_chunk_size
                end_index = min(start_index + prefill_chunk_size, seq_len)
                logits_ndarray = do_infer(start_index, end_index)
                if supports_all:
                    per_chunk_logits.append(logits_ndarray)

            if supports_all:
                logits_ndarray = np.concatenate(per_chunk_logits, axis=-2)
                # HF fancy-indexing preserves caller order and duplicates;
                # ``np.take`` matches that when we pass the output positions
                # directly without dedup/sort.
                output_positions = self._output_positions_for_logits_to_keep(logits_to_keep, seq_len)
                token_axis = logits_ndarray.ndim - 2
                logits_ndarray = np.take(logits_ndarray, output_positions, axis=token_axis)

            assert logits_ndarray is not None
            return torch.tensor(logits_ndarray, dtype=dtype, device=device)

        # Path 3: fallback. Interleave normal chunks (for positions the
        # caller does not care about) with size-1 infer calls (one per kept
        # position). The KV cache advances monotonically through every
        # position of the input, so subsequent decode steps see the same
        # cache state as the other two paths would leave behind.
        #
        # ``walk_positions`` is sorted-unique so the cursor can advance
        # monotonically; ``output_positions`` preserves caller order and
        # duplicates so the final tensor matches HF fancy-indexing.
        self._warn_last_only_slow_path_once()
        walk_positions = self._walk_positions_for_logits_to_keep(logits_to_keep, seq_len)
        output_positions = self._output_positions_for_logits_to_keep(logits_to_keep, seq_len)
        per_position_logits: dict[int, np.ndarray] = {}
        cursor = 0
        for p in walk_positions:
            while cursor < p:
                end_index = min(cursor + prefill_chunk_size, p)
                do_infer(cursor, end_index)
                cursor = end_index
            per_position_logits[p] = do_infer(cursor, cursor + 1)
            cursor += 1
        while cursor < seq_len:
            end_index = min(cursor + prefill_chunk_size, seq_len)
            do_infer(cursor, end_index)
            cursor = end_index

        if not output_positions:
            # Empty selection (empty tensor input): the KV cache has already
            # been advanced through the entire input by the trailing loop
            # above. Return a (1, 0, 0) placeholder rather than calling
            # np.concatenate on an empty list. The leading 1 matches the
            # batch axis that do_infer would have produced, so single-input
            # callers can still ``.squeeze(0)`` down to (0, 0).
            return torch.zeros((1, 0, 0), dtype=dtype, device=device)

        logits_ndarray = np.concatenate(
            [per_position_logits[p] for p in output_positions], axis=-2
        )
        return torch.tensor(logits_ndarray, dtype=dtype, device=device)

    def llm_forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[MobilintCache],
        cache_position: torch.Tensor,
        prefill_chunk_size: Optional[int] = None,
        count_npu_time: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 1,
    ):
        """Chunked MXQ prefill / decode with HF-style ``logits_to_keep``.

        ``logits_to_keep`` follows the ``transformers`` semantics: ``1``
        (default) returns only the last-token logits; ``0`` returns every
        position; ``>1`` returns the last N positions; a ``torch.Tensor``
        selects specific positions.

        The three execution paths (fast path, dynamic-axis, fallback) are
        implemented once in :meth:`_run_chunked_logits_to_keep`; this
        wrapper just builds a single-input ``do_infer`` closure that
        advances the KV cache and forwards to the shared helper. When an
        ``attention_mask`` is provided we dispatch to
        :meth:`_llm_forward_batch`, whose padded rows are filled with
        ``-inf`` so downstream softmax leaves padded positions at zero
        probability.

        Performance: on last-only MXQ (:meth:`_mxq_supports_all_logits`
        returns False), ``logits_to_keep=0`` triggers the Path 3 fallback,
        which runs one size-1 infer per input token and is dramatically
        slower than ``logits_to_keep=1`` on long prefills. Callers using
        ``.generate()`` are safe because HF sets ``logits_to_keep=1``
        automatically; callers doing manual ``.forward()`` for perplexity
        eval / logit collection should know this cost is inherent to
        last-only compiled models. A runtime WARNING fires once per model
        instance on the first Path 3 entry to surface the cost.
        """
        resolved_prefill_chunk_size = self.resolve_prefill_chunk_size(prefill_chunk_size)
        self.npu_time = 0.0 if count_npu_time else None

        if attention_mask is not None:
            self._validate_batch_cache(past_key_values, attention_mask.shape[0])
            return self._llm_forward_batch(
                inputs_embeds,
                attention_mask,
                past_key_values,
                resolved_prefill_chunk_size,
                count_npu_time=count_npu_time,
                logits_to_keep=logits_to_keep,
            )

        inputs_embeds_numpy = inputs_embeds.type(torch.float32).cpu().numpy()
        if inputs_embeds_numpy.ndim == 3:
            inputs_embeds_numpy = np.expand_dims(inputs_embeds_numpy, 1)  # (batch, 1, seqlen, hidden_size)

        seq_len = inputs_embeds_numpy.shape[2]
        mxq_model = self.npu_backend.mxq_model
        initial_cache_size = 0 if past_key_values is None else past_key_values.get_seq_length()
        timing_phase: Literal["prefill", "decode"] = "prefill" if initial_cache_size == 0 else "decode"

        def _do_infer(start: int, end: int) -> np.ndarray:
            cache_size = 0 if past_key_values is None else past_key_values.get_seq_length()
            chunk = inputs_embeds_numpy[:, :, start:end, :]
            if count_npu_time:
                t0 = time.perf_counter()
                result = mxq_model.infer([chunk], None, cache_size)
                elapsed = time.perf_counter() - t0
                assert self.npu_time is not None
                self.npu_time += elapsed
                self._record_npu_timing(timing_phase, elapsed)
            else:
                result = mxq_model.infer([chunk], None, cache_size)
            assert result is not None, "mxq infer result is None!"
            if past_key_values is not None:
                past_key_values.update_cache_position(cache_position[start:end])
            return result[0]

        logits = self._run_chunked_logits_to_keep(
            do_infer=_do_infer,
            seq_len=seq_len,
            prefill_chunk_size=resolved_prefill_chunk_size,
            logits_to_keep=logits_to_keep,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        return logits.squeeze(0)

    def resolve_batched_attention_mask(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Return the attention mask used by batched LLM inference."""
        configured_batch_size = max(1, getattr(self.config, "max_batch_size", 1))
        if configured_batch_size <= 1:
            return None
        if attention_mask is not None:
            return attention_mask
        return torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=inputs_embeds.device,
        )

    def _llm_forward_batch(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[MobilintCache],
        prefill_chunk_size: int = 0,
        count_npu_time: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 1,
    ):
        """Batched sibling of :meth:`llm_forward` with the same 3-path dispatch.

        Paths 2 and 3 produce a variable number of kept positions per
        batch item and stack them into ``(batch, max_keep_len, vocab)``.
        Padded rows are filled with ``-inf`` (not 0.0) so downstream
        softmax assigns exactly zero probability to padded slots; callers
        that don't also track a padding mask (e.g. generation) would
        otherwise see real vocab mass leak onto padded positions.

        Performance: on last-only MXQ (:meth:`_mxq_supports_all_logits`
        returns False), ``logits_to_keep=0`` walks the batch cursor one
        token at a time through kept positions, so long prefills are
        dramatically slower than ``logits_to_keep=1``. ``.generate()``
        avoids this because HF sets ``logits_to_keep=1`` automatically;
        manual ``.forward()`` callers doing perplexity eval / logit
        collection incur this cost inherently on last-only builds. A
        runtime WARNING fires once per model instance on the first Path
        3 entry to surface the cost.
        """
        debug_enabled = logger.isEnabledFor(logging.DEBUG)
        batch_size = attention_mask.shape[0]

        if attention_mask.shape == inputs_embeds.shape[:-1]:
            attention_mask_bool = cast(torch.BoolTensor, attention_mask.type(torch.bool))
            inputs_embeds_masked = [inputs_embeds[i, attention_mask_bool[i, :], :] for i in range(batch_size)]
            sequence_lengths = cast(list[int], attention_mask.sum(dim=1).tolist())
        else:
            assert inputs_embeds.shape[1] == 1
            inputs_embeds_masked = [inputs_embeds[i, :, :] for i in range(batch_size)]
            sequence_lengths = [1 for _ in range(batch_size)]

        if debug_enabled:
            logger.debug(
                "[BATCH-LLM][START] inputs_embeds=%s attention_mask=%s batch_size=%d sequence_lengths=%s",
                tuple(inputs_embeds.shape),
                tuple(attention_mask.shape),
                batch_size,
                sequence_lengths,
            )

        max_sequence_length = max(sequence_lengths)
        mxq_model = self.npu_backend.mxq_model
        if prefill_chunk_size == 0:
            prefill_chunk_size = mxq_model.get_input_buffer_info()[0].max_width
        assert prefill_chunk_size > 0, (
            "prefill_chunk_size should be a positive number! prefill_chunk_size: %d" % prefill_chunk_size
        )

        # int==1 always hits Path 1 (the existing per-item cursor code handles
        # mixed lengths correctly and this is the HF-generate default). For a
        # scalar tensor selector we additionally require every item to share the
        # same length, because a single value can only uniformly represent
        # "last position" when there is only one length in the batch.
        if isinstance(logits_to_keep, int) and logits_to_keep == 1:
            is_default_keep = True
        else:
            lens_equal = all(sl == sequence_lengths[0] for sl in sequence_lengths)
            is_default_keep = lens_equal and self._is_last_only_selector(
                logits_to_keep, sequence_lengths[0]
            )
        supports_all = False if is_default_keep else self._mxq_supports_all_logits()

        # Path 2 uses the output positions directly to assemble the final
        # per-item tensor (HF order, duplicates preserved). Path 3 also
        # needs a sorted-unique variant to drive the shared cursor.
        output_positions_by_item: dict[int, list[int]] = {}
        walk_positions_by_item: dict[int, list[int]] = {}
        if not is_default_keep:
            for j in range(batch_size):
                output_positions_by_item[j] = self._output_positions_for_logits_to_keep(
                    logits_to_keep, sequence_lengths[j]
                )
                walk_positions_by_item[j] = sorted(set(output_positions_by_item[j]))

        def _run_batch_infer(
            cache_ids_l: list[int],
            sequence_lengths_chunks_l: list[int],
            cache_sizes_chunks_l: list[int],
            inputs_embeds_chunks_l: list[torch.Tensor],
        ) -> tuple[np.ndarray, tuple[int, ...]]:
            inputs_embeds_concat_l = torch.concat(inputs_embeds_chunks_l, dim=0).unsqueeze(0)
            inputs_embeds_numpy_l: np.ndarray = inputs_embeds_concat_l.type(torch.float32).cpu().numpy()
            if inputs_embeds_numpy_l.ndim == 3:
                inputs_embeds_numpy_l = np.expand_dims(inputs_embeds_numpy_l, 1)
            batch_params_l = [
                qbruntime.BatchParam(
                    sequence_length=sequence_lengths_chunks_l[k],
                    cache_size=cache_sizes_chunks_l[k],
                    cache_id=cache_ids_l[k],
                )
                for k in range(len(cache_ids_l))
            ]
            if count_npu_time:
                t0 = time.perf_counter()
                result_l = mxq_model.infer([inputs_embeds_numpy_l], None, 0, batch_params_l)
                assert self.npu_time is not None
                elapsed = time.perf_counter() - t0
                self.npu_time += elapsed
                phase: Literal["prefill", "decode"] = (
                    "prefill"
                    if max_sequence_length > 1 or any(cache_size == 0 for cache_size in cache_sizes_chunks_l)
                    else "decode"
                )
                self._record_npu_timing(phase, elapsed)
            else:
                result_l = mxq_model.infer([inputs_embeds_numpy_l], None, 0, batch_params_l)
            assert result_l is not None, "mxq infer result is None!"
            return result_l[0], inputs_embeds_numpy_l.shape

        # ------------------------------------------------------------------
        # Path 1: default fast path (logits_to_keep == 1). Preserved
        # verbatim including the existing debug instrumentation so cache
        # contract regressions stay visible.
        # ------------------------------------------------------------------
        if is_default_keep:
            zero_length_rows = [
                row_index for row_index, sequence_length in enumerate(sequence_lengths) if sequence_length == 0
            ]
            if zero_length_rows:
                raise ValueError(
                    "Batched LLM inputs contain empty sequences after applying attention_mask. "
                    f"Zero-length rows: {zero_length_rows}"
                )
            num_of_chunks = math.ceil(max_sequence_length / prefill_chunk_size)
            logits_dict: dict[int, torch.Tensor] = {}

            for i in range(num_of_chunks):
                start_index = i * prefill_chunk_size

                sequence_lengths_chunks: list[int] = []
                cache_sizes_chunks: list[int] = []
                cache_ids: list[int] = []
                prefill_masks: list[bool] = []
                inputs_embeds_chunks: list[torch.Tensor] = []
                seen_tokens: dict[int, int] = {}

                for j in range(batch_size):
                    end_index = min(start_index + prefill_chunk_size, sequence_lengths[j])
                    if start_index < sequence_lengths[j] and end_index <= sequence_lengths[j]:
                        sequence_lengths_chunks.append(end_index - start_index)
                        cache_sizes_chunks.append(
                            past_key_values.get_seq_length(j) if past_key_values is not None else 0
                        )
                        cache_ids.append(j)
                        prefill_masks.append(end_index < inputs_embeds_masked[j].shape[0])
                        inputs_embeds_chunks.append(inputs_embeds_masked[j][start_index:end_index, :])
                        seen_tokens[j] = end_index - start_index

                if len(inputs_embeds_chunks) == 0:
                    continue

                raw_output, inputs_embeds_numpy_shape = _run_batch_infer(
                    cache_ids, sequence_lengths_chunks, cache_sizes_chunks, inputs_embeds_chunks
                )
                logits_chunks = cast(
                    torch.FloatTensor,
                    torch.tensor(raw_output, dtype=inputs_embeds.dtype, device=inputs_embeds.device).reshape(
                        [len(cache_ids), 1, -1]
                    ),
                )

                if debug_enabled:
                    batch_seq_sum = sum(sequence_lengths_chunks)
                    logger.debug(
                        (
                            "[BATCH-LLM][CHUNK %d/%d] start=%d active=%d ids=%s seq_chunks=%s "
                            "cache_sizes=%s prefill=%s input_shape=%s batch_seq_sum=%d"
                        ),
                        i + 1,
                        num_of_chunks,
                        start_index,
                        len(cache_ids),
                        cache_ids,
                        sequence_lengths_chunks,
                        cache_sizes_chunks,
                        prefill_masks,
                        tuple(inputs_embeds_numpy_shape),
                        batch_seq_sum,
                    )
                    logger.debug(
                        "[BATCH-LLM][CHUNK %d/%d][BATCH_PARAMS] %s",
                        i + 1,
                        num_of_chunks,
                        [
                            {
                                "sequence_length": sequence_lengths_chunks[k],
                                "cache_size": cache_sizes_chunks[k],
                                "cache_id": cache_ids[k],
                            }
                            for k in range(len(cache_ids))
                        ],
                    )
                    first_input_chunk = inputs_embeds_chunks[0]
                    input_batch_same = all(
                        chunk.shape == first_input_chunk.shape and torch.equal(chunk, first_input_chunk)
                        for chunk in inputs_embeds_chunks[1:]
                    )
                    logger.debug(
                        "[BATCH-LLM][CHUNK %d/%d] input_batch_same=%s",
                        i + 1,
                        num_of_chunks,
                        input_batch_same,
                    )
                    first_logits = logits_chunks[0]
                    result_batch_same = all(torch.equal(chunk, first_logits) for chunk in logits_chunks[1:])
                    next_tokens = logits_chunks[:, -1, :].argmax(dim=-1).tolist()
                    logger.debug(
                        "[BATCH-LLM][CHUNK %d/%d] result_batch_same=%s next_tokens=%s",
                        i + 1,
                        num_of_chunks,
                        result_batch_same,
                        next_tokens,
                    )
                    if not result_batch_same and len(cache_ids) > 1:
                        logits_vectors = logits_chunks[:, -1, :]
                        cosine_similarity = torch.nn.functional.cosine_similarity(
                            logits_vectors.unsqueeze(1),
                            logits_vectors.unsqueeze(0),
                            dim=-1,
                        )
                        max_abs_diff = torch.abs(
                            logits_vectors.unsqueeze(1) - logits_vectors.unsqueeze(0)
                        ).amax(dim=-1)
                        cosine_rows = [
                            " ".join(
                                f"{float(cosine_similarity[row, col]):7.3f}"
                                for col in range(cosine_similarity.shape[1])
                            )
                            for row in range(cosine_similarity.shape[0])
                        ]
                        max_abs_diff_rows = [
                            " ".join(
                                f"{float(max_abs_diff[row, col]):7.3f}" for col in range(max_abs_diff.shape[1])
                            )
                            for row in range(max_abs_diff.shape[0])
                        ]
                        logger.debug(
                            "[BATCH-LLM][CHUNK %d/%d][COSINE_SIM]\ncache_ids=%s\n%s\n[MAX_ABS_DIFF]\n%s",
                            i + 1,
                            num_of_chunks,
                            cache_ids,
                            "\n".join(cosine_rows),
                            "\n".join(max_abs_diff_rows),
                        )
                    logger.debug(
                        "[BATCH-LLM][CHUNK %d/%d] infer_output_shape=%s",
                        i + 1,
                        num_of_chunks,
                        tuple(logits_chunks.shape),
                    )

                for j, prefill_mask in enumerate(prefill_masks):
                    if prefill_mask is False:
                        cache_id = cache_ids[j]
                        logits_dict[cache_id] = logits_chunks[j, :, :].clone()

                if past_key_values is not None:
                    past_key_values.update_seen_tokens(seen_tokens)

            if debug_enabled:
                missing_ids = [cache_id for cache_id in range(batch_size) if cache_id not in logits_dict]
                logger.debug(
                    "[BATCH-LLM][END] logits_dict_keys=%s missing_ids=%s",
                    sorted(list(logits_dict.keys())),
                    missing_ids,
                )

            logits_list = [logits_dict[cache_id] for cache_id in range(batch_size)]
            stacked = cast(torch.FloatTensor, torch.stack(logits_list, dim=0))
            if debug_enabled:
                logger.debug("[BATCH-LLM][END] output_shape=%s", tuple(stacked.shape))
            return stacked

        # ------------------------------------------------------------------
        # Path 2: dynamic-output MXQ. The compiled model already produces
        # per-position logits, so we keep the caller's chunk size and split
        # each chunk's flat output by per-item offsets. Per-item logits are
        # concatenated, sliced by that item's kept positions, right-padded
        # to the longest kept sequence, and stacked.
        # ------------------------------------------------------------------
        if supports_all:
            num_of_chunks = math.ceil(max_sequence_length / prefill_chunk_size)
            all_logits_by_item: dict[int, list[np.ndarray]] = {j: [] for j in range(batch_size)}

            for i in range(num_of_chunks):
                start_index = i * prefill_chunk_size

                sequence_lengths_chunks = []
                cache_sizes_chunks = []
                cache_ids = []
                inputs_embeds_chunks = []
                seen_tokens = {}

                for j in range(batch_size):
                    end_index = min(start_index + prefill_chunk_size, sequence_lengths[j])
                    if start_index < sequence_lengths[j] and end_index <= sequence_lengths[j]:
                        sequence_lengths_chunks.append(end_index - start_index)
                        cache_sizes_chunks.append(
                            past_key_values.get_seq_length(j) if past_key_values is not None else 0
                        )
                        cache_ids.append(j)
                        inputs_embeds_chunks.append(inputs_embeds_masked[j][start_index:end_index, :])
                        seen_tokens[j] = end_index - start_index

                if len(inputs_embeds_chunks) == 0:
                    continue

                raw_output, _ = _run_batch_infer(
                    cache_ids, sequence_lengths_chunks, cache_sizes_chunks, inputs_embeds_chunks
                )

                total_tokens = sum(sequence_lengths_chunks)
                flat_output = np.asarray(raw_output).reshape(total_tokens, -1)
                offset = 0
                for k, cache_id in enumerate(cache_ids):
                    len_k = sequence_lengths_chunks[k]
                    all_logits_by_item[cache_id].append(flat_output[offset : offset + len_k, :].copy())
                    offset += len_k

                if debug_enabled:
                    logger.debug(
                        "[BATCH-LLM][DYNAMIC CHUNK %d/%d] start=%d ids=%s seq_chunks=%s flat_output_shape=%s",
                        i + 1,
                        num_of_chunks,
                        start_index,
                        cache_ids,
                        sequence_lengths_chunks,
                        tuple(flat_output.shape),
                    )

                if past_key_values is not None:
                    past_key_values.update_seen_tokens(seen_tokens)

            mxq_vocab_size = self._mxq_static_vocab_size()
            per_item_sliced: list[torch.Tensor] = []
            vocab_size = 0
            for j in range(batch_size):
                if all_logits_by_item[j]:
                    item_all = np.concatenate(all_logits_by_item[j], axis=0)
                else:
                    # Zero-length row (attention_mask entirely zero): no chunk
                    # contributed. Synthesize a (0, vocab) placeholder so the
                    # padding pass below can right-fill with -inf, symmetric
                    # with the Path 3 fallback.
                    item_all = np.empty((0, mxq_vocab_size), dtype=np.float32)
                positions_j = output_positions_by_item[j]
                sliced = item_all[positions_j, :] if positions_j else item_all[0:0, :]
                vocab_size = max(vocab_size, sliced.shape[-1])
                per_item_sliced.append(
                    torch.tensor(sliced, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                )

            max_keep_len = max((item.shape[0] for item in per_item_sliced), default=0)
            if max_keep_len == 0:
                # No batch item had any kept positions (e.g. empty-tensor
                # selector). Return the (batch, 0, 0) shape contract explicitly
                # so we don't rely on the torch.full((0, 0), -inf) idiom in the
                # padding loop silently degenerating to a no-op when future
                # refactors change how max_keep_len / vocab_size are computed.
                return cast(
                    torch.FloatTensor,
                    torch.zeros(
                        (batch_size, 0, 0),
                        dtype=inputs_embeds.dtype,
                        device=inputs_embeds.device,
                    ),
                )
            padded: list[torch.Tensor] = []
            for item in per_item_sliced:
                if item.shape[0] < max_keep_len:
                    # -inf so downstream softmax assigns 0 probability to padded
                    # positions; using 0.0 would spread mass over real vocab rows.
                    pad = torch.full(
                        (max_keep_len - item.shape[0], vocab_size),
                        float("-inf"),
                        dtype=item.dtype,
                        device=item.device,
                    )
                    item = torch.cat([item, pad], dim=0)
                padded.append(item)
            stacked = cast(torch.FloatTensor, torch.stack(padded, dim=0))
            if debug_enabled:
                logger.debug("[BATCH-LLM][END] output_shape=%s", tuple(stacked.shape))
            return stacked

        # ------------------------------------------------------------------
        # Path 3: fallback for last-only MXQ with a non-default request.
        # We advance a shared cursor across the batch and pick the chunk
        # size from the smallest next-needed kept position across only
        # those items that are still active AND still have pending kept
        # positions. Items whose kept-set is exhausted (or empty) no
        # longer force a size-1 walk across the rest of the batch — a
        # single late kept position in one item used to drag every other
        # item into per-step infer calls. ``walk_positions_by_item`` is
        # sorted-unique so a per-item pointer tracks the next needed
        # position in O(1); duplicates and caller order are restored from
        # ``output_positions_by_item`` when the final tensor is assembled.
        # ------------------------------------------------------------------
        self._warn_last_only_slow_path_once()
        pointer_by_item: dict[int, int] = {j: 0 for j in range(batch_size)}
        per_item_kept_logits: dict[int, dict[int, torch.Tensor]] = {j: {} for j in range(batch_size)}
        cursor = 0
        while cursor < max_sequence_length:
            min_next_needed: Optional[int] = None
            for j in range(batch_size):
                if cursor >= sequence_lengths[j]:
                    continue
                ptr = pointer_by_item[j]
                positions_j = walk_positions_by_item[j]
                if ptr < len(positions_j):
                    nn = positions_j[ptr]
                    if min_next_needed is None or nn < min_next_needed:
                        min_next_needed = nn

            if min_next_needed is None:
                chunk = prefill_chunk_size
            elif min_next_needed == cursor:
                chunk = 1
            else:
                chunk = min(prefill_chunk_size, min_next_needed - cursor)

            sequence_lengths_chunks = []
            cache_sizes_chunks = []
            cache_ids = []
            inputs_embeds_chunks = []
            seen_tokens = {}

            for j in range(batch_size):
                end_j = min(cursor + chunk, sequence_lengths[j])
                if cursor < sequence_lengths[j] and end_j <= sequence_lengths[j]:
                    sequence_lengths_chunks.append(end_j - cursor)
                    cache_sizes_chunks.append(
                        past_key_values.get_seq_length(j) if past_key_values is not None else 0
                    )
                    cache_ids.append(j)
                    inputs_embeds_chunks.append(inputs_embeds_masked[j][cursor:end_j, :])
                    seen_tokens[j] = end_j - cursor

            if not inputs_embeds_chunks:
                cursor += chunk
                continue

            raw_output, _ = _run_batch_infer(
                cache_ids, sequence_lengths_chunks, cache_sizes_chunks, inputs_embeds_chunks
            )
            logits_chunks = cast(
                torch.FloatTensor,
                torch.tensor(raw_output, dtype=inputs_embeds.dtype, device=inputs_embeds.device).reshape(
                    [len(cache_ids), 1, -1]
                ),
            )

            if debug_enabled:
                logger.debug(
                    "[BATCH-LLM][FALLBACK] cursor=%d chunk=%d ids=%s seq_chunks=%s cache_sizes=%s",
                    cursor,
                    chunk,
                    cache_ids,
                    sequence_lengths_chunks,
                    cache_sizes_chunks,
                )

            if chunk == 1:
                for k, cache_id in enumerate(cache_ids):
                    ptr = pointer_by_item[cache_id]
                    positions_j = walk_positions_by_item[cache_id]
                    if ptr < len(positions_j) and positions_j[ptr] == cursor:
                        per_item_kept_logits[cache_id][cursor] = logits_chunks[k, :, :].clone()
                        pointer_by_item[cache_id] = ptr + 1

            if past_key_values is not None:
                past_key_values.update_seen_tokens(seen_tokens)

            cursor += chunk

        per_item_final: list[torch.Tensor] = []
        vocab_size = 0
        for j in range(batch_size):
            # per_item_kept_logits is keyed by (unique) walk positions; the
            # final tensor uses output positions so duplicate indices in the
            # caller's tensor pick the same walk-position tensor twice.
            positions_j = output_positions_by_item[j]
            if positions_j:
                item_logits = torch.cat(
                    [per_item_kept_logits[j][p] for p in positions_j], dim=0
                )
                vocab_size = max(vocab_size, item_logits.shape[-1])
            else:
                item_logits = torch.empty(
                    (0, 0), dtype=inputs_embeds.dtype, device=inputs_embeds.device
                )
            per_item_final.append(item_logits)

        max_keep_len = max((item.shape[0] for item in per_item_final), default=0)
        if max_keep_len == 0:
            # No batch item had any kept positions (e.g. empty-tensor selector).
            # Return the (batch, 0, 0) shape contract explicitly so we don't
            # rely on the torch.full((0, 0), -inf) idiom in the padding loop
            # silently degenerating to a no-op when future refactors change
            # how max_keep_len / vocab_size are computed.
            return cast(
                torch.FloatTensor,
                torch.zeros(
                    (batch_size, 0, 0),
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                ),
            )
        padded_final: list[torch.Tensor] = []
        for item in per_item_final:
            if item.shape[0] < max_keep_len:
                pad_rows = max_keep_len - item.shape[0]
                pad_cols = vocab_size if item.numel() == 0 else item.shape[-1]
                if item.numel() == 0:
                    item = torch.empty(
                        (0, pad_cols), dtype=inputs_embeds.dtype, device=inputs_embeds.device
                    )
                # -inf so downstream softmax assigns 0 probability to padded
                # positions; using 0.0 would spread mass over real vocab rows.
                pad = torch.full(
                    (pad_rows, pad_cols),
                    float("-inf"),
                    dtype=item.dtype,
                    device=item.device,
                )
                item = torch.cat([item, pad], dim=0)
            padded_final.append(item)
        stacked = cast(torch.FloatTensor, torch.stack(padded_final, dim=0))
        if debug_enabled:
            logger.debug("[BATCH-LLM][END] output_shape=%s", tuple(stacked.shape))
        return stacked

    @staticmethod
    def _validate_batch_cache(past_key_values: Optional[MobilintCache], batch_size: int) -> None:
        if past_key_values is None:
            return

        cache_batch_size = getattr(past_key_values, "batch_size", 1)
        if cache_batch_size < batch_size:
            raise ValueError(
                "Batch cache size is too small: "
                f"past_key_values.batch_size={cache_batch_size}, input batch_size={batch_size}. "
                "Create MobilintCache with a batch size greater than or equal to the batched request."
            )

    def resolve_prefill_chunk_size(self, prefill_chunk_size: Optional[int]) -> int:
        explicit_prefill_chunk_size = self._coerce_positive_int(prefill_chunk_size)
        if explicit_prefill_chunk_size is not None:
            return explicit_prefill_chunk_size

        config_value = self._get_config_prefill_chunk_size()
        config_prefill_chunk_size = self._coerce_positive_int(config_value)
        if config_prefill_chunk_size is not None:
            return config_prefill_chunk_size

        return self._DEFAULT_PREFILL_CHUNK_SIZE

    def _get_config_prefill_chunk_size(self) -> Any:
        config_value = getattr(self.config, "npu_prefill_chunk_size", None)
        if isinstance(config_value, dict):
            core_mode = getattr(self.npu_backend, "core_mode", None)
            if core_mode is None:
                return None
            return config_value.get(core_mode)
        return config_value

    @staticmethod
    def _coerce_positive_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, float) and value.is_integer():
            return int(value) if value > 0 else None
        if isinstance(value, str):
            try:
                parsed = int(value.strip())
            except ValueError:
                return None
            return parsed if parsed > 0 else None
        return None

    def decoder_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        past_key_values: Optional[MobilintCache],
        cache_position: torch.Tensor,
    ):
        hidden_states_numpy = hidden_states.type(torch.float32).cpu().numpy()
        encoder_hidden_states_numpy = encoder_hidden_states.type(torch.float32).cpu().numpy()

        mxq_model = self.npu_backend.mxq_model

        cache_size = 0 if past_key_values is None else past_key_values.get_seq_length()

        result = mxq_model.infer([hidden_states_numpy, encoder_hidden_states_numpy], None, cache_size)
        assert result is not None, "mxq infer result is None!"
        logits_ndarray = result[0]

        if past_key_values is not None:
            past_key_values.update_cache_position(cache_position)

        logits = torch.tensor(logits_ndarray, dtype=hidden_states.dtype, device=hidden_states.device)

        return logits


class MobilintEagle3ModelMixin(MobilintModelMixin):
    """Base Mobilint model mixin for EAGLE-3 child backends.

    EAGLE-3 models are composed from base, draft, and optional projection MXQ
    child models. This mixin exists as the shared extension point for those
    child backends so future LLM architectures can reuse the same EAGLE-3
    runtime contracts without inheriting from a Qwen2-specific class.
    """
