"""Multi-slot NPU batched-infer dispatcher.

Centralizes the routing that :meth:`MobilintModelMixin._llm_forward_batch`
previously reimplemented inside a per-call closure. Callers hand the dispatcher
a flat list of batch rows plus their per-row inputs; the dispatcher groups
rows by owning ``qbruntime.Model`` slot, submits one blocking ``.infer`` per
group (in parallel via a :class:`~concurrent.futures.ThreadPoolExecutor` when
there is more than one group), and merges the outputs back into caller order.

Layout resolution is done once — the compiled MXQ has one output convention
across every slot, so the merge reads ``backend.output_layout`` instead of
guessing per dispatch. When the compile-time probe is ambiguous the dispatcher
inspects an unambiguous runtime group and pins the answer via
:meth:`MobilintNPUBackend._set_output_layout` for the remainder of the process.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Callable, List, Literal, Optional, Tuple, cast

import numpy as np
import qbruntime
import torch

if TYPE_CHECKING:
    from ...utils.npu_backend import MobilintNPUBackend
    from .cache_utils import MobilintCache


logger = logging.getLogger(__name__)


PhaseName = Literal["prefill", "decode"]
LayoutName = Literal["n_items", "n_tokens"]


class MultiSlotDispatcher:
    """Owns the multi-slot topology for a :class:`MobilintNPUBackend`.

    Instances hold a reference to the backend and read live topology from it
    (``mxq_models``, ``k_per_model``, ``output_layout``), so the dispatcher
    stays valid across create / launch / dispose cycles that reshape the
    backend's slot layout.
    """

    def __init__(self, backend: "MobilintNPUBackend") -> None:
        self.backend = backend

    # ------------------------------------------------------------------
    # Topology accessors
    # ------------------------------------------------------------------

    @property
    def mxq_models(self) -> List[qbruntime.Model]:
        return self.backend.mxq_models

    @property
    def k_per_model(self) -> int:
        return max(1, int(getattr(self.backend, "k_per_model", 1) or 1))

    @property
    def n_slots(self) -> int:
        return len(self.backend.mxq_models)

    def slot_of(
        self,
        row: int,
        past_key_values: Optional["MobilintCache"] = None,
    ) -> Tuple[int, int]:
        """Return ``(model_idx, local_cache_id)`` for a flat batch row.

        Multi-slot dispatch routes each flat row to its owning Model via
        ``(row // K, row % K)``. When ``past_key_values`` exposes ``slot_of``
        (a :class:`MobilintCache`) it is the source of truth — it carries the
        cache's own ``N`` and ``K``. Otherwise (e.g. ``use_cache=False`` on a
        decoder-only forward with labels) we derive the mapping from the
        backend's compiled ``k_per_model``.
        """
        if past_key_values is not None and hasattr(past_key_values, "slot_of"):
            return past_key_values.slot_of(int(row))
        return divmod(int(row), self.k_per_model)

    def _validate_cache_topology(self, past_key_values: Optional["MobilintCache"]) -> None:
        """Reject a caller-supplied cache whose ``(N, K, model handles)`` disagree with this backend.

        Same aggregate capacity is not the same as compatible routing.
        :meth:`MobilintModelMixin._validate_batch_cache` guards the aggregate
        ``n_models * k_per_model`` axis, but a cache built with ``(N=1, K=2)``
        has the same capacity as a backend running ``(N=2, K=1)`` — routing a
        row through ``cache.slot_of`` then indexes ``backend.mxq_models[0]``
        twice and leaves ``mxq_models[1]`` idle while the second row is sent
        to slot 0 with ``local_cache_id=1``, an invalid inference on a
        ``K==1`` Model. Reject that upfront by comparing the topology axes
        directly.

        Model-handle identity is the second half: a cache built for a prior
        backend that has since been disposed and recreated carries stale
        :class:`qbruntime.Model` references. ``infer`` on those handles would
        target either freed HBM or a wholly unrelated Model. Compare the
        cache's ``mxq_models`` against the backend's current slot list
        element-by-element (``is`` identity) so a rebuilt backend surfaces
        the mismatch here rather than as an opaque qbruntime error.
        """
        if past_key_values is None or not hasattr(past_key_values, "slot_of"):
            return
        cache_n = int(getattr(past_key_values, "n_models", 1) or 1)
        cache_k = int(getattr(past_key_values, "k_per_model", 1) or 1)
        if cache_n != self.n_slots:
            raise ValueError(
                f"MobilintCache topology mismatch: cache.n_models={cache_n} "
                f"but backend n_slots={self.n_slots}. Rebuild the cache after "
                f"changing backend max_batch_size."
            )
        if cache_k != self.k_per_model:
            raise ValueError(
                f"MobilintCache topology mismatch: cache.k_per_model={cache_k} "
                f"but backend k_per_model={self.k_per_model}. The compiled MXQ "
                f"batch axis must match the cache's per-model capacity."
            )
        cache_models = getattr(past_key_values, "mxq_models", None)
        if cache_models is not None:
            backend_models = self.mxq_models
            cache_models_list = list(cache_models)
            if len(cache_models_list) != len(backend_models) or any(
                cm is not bm for cm, bm in zip(cache_models_list, backend_models)
            ):
                raise ValueError(
                    "MobilintCache Model-handle identity mismatch: cache slots do not "
                    "reference the currently loaded backend slots. The backend was "
                    "likely disposed and recreated after the cache was built; rebuild "
                    "the cache to bind to the new slot handles."
                )

    def assert_single_slot(self, caller: str, remediation: str) -> None:
        """Raise ``NotImplementedError`` if the backend has grown beyond ``N == 1``.

        Cross-attention decoders (BLIP text head, and similar encoder-decoder
        stacks) issue one blocking ``mxq_model.infer`` on slot 0 with no
        cross-slot routing or beam-cache reorder. Keep the invariant enforced
        in a single place so future callers inherit the check.
        """
        n = self.n_slots
        if n > 1:
            raise NotImplementedError(
                f"{caller} does not support multi-slot sw-batch dispatch "
                f"(backend launched N={n} slots). {remediation}"
            )

    # ------------------------------------------------------------------
    # Batched infer entry point
    # ------------------------------------------------------------------

    def dispatch(
        self,
        cache_ids: List[int],
        sequence_lengths: List[int],
        cache_sizes: List[int],
        inputs_embeds_chunks: List[torch.Tensor],
        *,
        max_sequence_length: int,
        pack_extra_inputs: Optional[Callable[..., List[np.ndarray]]] = None,
        past_key_values: Optional["MobilintCache"] = None,
        batched_input_expand_dims: bool = True,
        chunk_start: int = 0,
        count_npu_time: bool = False,
        phase_override: Optional[PhaseName] = None,
        record_npu_time: Optional[Callable[[PhaseName, float], None]] = None,
        debug_enabled: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """Dispatch a batched infer across the backend's Model slots.

        Args:
            cache_ids: Flat batch rows (caller-visible cache slots).
            sequence_lengths: Per-row sequence length for this dispatch.
            cache_sizes: Per-row KV cache size at the start of the dispatch.
            inputs_embeds_chunks: Per-row ``(seq_len_k, hidden)`` embed tensors.
            max_sequence_length: Batch-wide max sequence length; used by the
                phase heuristic to distinguish prefill (any row has seq_len > 1
                or a fresh cache) from pure decode.
            pack_extra_inputs: Optional hook that returns additional ndarray
                inputs to concatenate after ``inputs_embeds`` in the compiled
                MXQ input list. Receives the caller-visible flat ``cache_ids``
                (not local slot ids) so deepstack-style side inputs slice
                against the same row identity the caller used.
            past_key_values: Cache used to resolve ``slot_of`` when present.
            batched_input_expand_dims: When ``True`` (LLM default) the packed
                embeds are unsqueezed to rank 4 before ``mxq_model.infer``.
                Multi-input decoders that ship rank-3 compiled inputs (e.g.
                Qwen3-VL) override this to ``False``.
            chunk_start: Passed through to ``pack_extra_inputs``.
            count_npu_time: Time the NPU dispatch and forward the total to
                ``record_npu_time``. Single-group calls report elapsed;
                multi-group calls report wall time so parallel work is not
                double-counted.
            phase_override: Force the timing phase. Path 3 uses this because
                size-1 capture chunks would otherwise be misclassified as
                prefill by the auto-heuristic.
            record_npu_time: Called with ``(phase, elapsed)`` when
                ``count_npu_time`` is ``True``.
            debug_enabled: Emit the ``[BATCH-LLM][PARALLEL]`` debug line.

        Returns:
            A ``(merged_output, first_group_shape)`` tuple. ``merged_output``
            is the concatenated ``(n_items, vocab)`` or ``(n_tokens, vocab)``
            ndarray in caller row order (choice determined by
            ``backend.output_layout``); ``first_group_shape`` is the first
            group's ``inputs_embeds_numpy.shape`` retained for debug logging.
        """
        # Topology + identity check: reject a caller-supplied cache whose
        # ``(N, K)`` disagrees with the backend or whose ``mxq_models``
        # reference stale slot handles. Runs once per dispatch, before any
        # routing decision that would otherwise trust ``cache.slot_of``.
        self._validate_cache_topology(past_key_values)

        mxq_models = self.mxq_models
        n_backend_slots = self.n_slots
        n_items = len(cache_ids)

        # Cacheless capacity guard: when the caller omits ``past_key_values``
        # (e.g. ``use_cache=False`` on an evaluation forward), ``slot_of``
        # falls back to ``divmod(row, k_per_model)``. That routing has no
        # awareness of ``n_slots``, so a request beyond ``N * K`` would
        # otherwise surface as an ``IndexError`` on ``mxq_models[model_idx]``
        # (multi-slot) or a low-level qbruntime error when ``cache_ids``
        # exceed the compiled ``K`` on slot 0 (single-slot). Fail fast with a
        # message that points at both the request size and the backend
        # capacity. The with-cache branch is validated by
        # :meth:`MobilintModelMixin._validate_batch_cache` against the cache's
        # own capacity, so we skip this check when a cache is supplied.
        if past_key_values is None:
            capacity = n_backend_slots * self.k_per_model
            max_row = max(cache_ids) if cache_ids else -1
            if n_items > capacity or max_row >= capacity:
                raise ValueError(
                    "Cacheless batched dispatch exceeds backend capacity: "
                    f"request has n_items={n_items} and max row_id={max_row}; "
                    f"backend N*K = {n_backend_slots} * {self.k_per_model} = {capacity}. "
                    "Either supply a MobilintCache with matching capacity, or "
                    "reduce the batched request to at most N*K rows."
                )

        # Multi-slot dispatch routes each flat row to its owning Model via
        # ``slot_of``. A single-slot backend keeps the flat cache_id
        # passthrough on slot 0.
        if n_backend_slots > 1:
            buckets: dict[int, list[Tuple[int, int]]] = {}
            for k, flat_row in enumerate(cache_ids):
                model_idx, local_cache_id = self.slot_of(flat_row, past_key_values)
                buckets.setdefault(model_idx, []).append((k, local_cache_id))
            groups: List[Tuple[int, List[int], List[int]]] = [
                (
                    m,
                    [entry[0] for entry in buckets[m]],
                    [entry[1] for entry in buckets[m]],
                )
                for m in sorted(buckets.keys())
            ]
        else:
            groups = [(0, list(range(len(cache_ids))), list(cache_ids))]

        def _classify_phase() -> PhaseName:
            # Path 3 fallback advances a shared cursor across the batch and
            # its size-1 captures run at cursor > 0 with all cache_sizes > 0.
            # The auto-heuristic latches on batch-wide max_sequence_length
            # (not the current chunk size), so Path 3 passes an explicit
            # phase_override; Paths 1 and 2 leave it None.
            return (
                "prefill"
                if max_sequence_length > 1 or any(cs == 0 for cs in cache_sizes)
                else "decode"
            )

        def _build_group_payload(
            group_items: List[int],
            group_local_ids: List[int],
        ) -> Tuple[Tuple[int, ...], List[np.ndarray], List[qbruntime.BatchParam], List[int]]:
            g_orig_ids = [cache_ids[k] for k in group_items]
            g_seq_lens = [sequence_lengths[k] for k in group_items]
            g_cache_sizes = [cache_sizes[k] for k in group_items]
            g_embeds = [inputs_embeds_chunks[k] for k in group_items]
            inputs_embeds_concat = torch.concat(g_embeds, dim=0).unsqueeze(0)
            inputs_embeds_numpy = inputs_embeds_concat.type(torch.float32).cpu().numpy()
            if batched_input_expand_dims and inputs_embeds_numpy.ndim == 3:
                inputs_embeds_numpy = np.expand_dims(inputs_embeds_numpy, 1)
            infer_inputs: List[np.ndarray] = [inputs_embeds_numpy]
            if pack_extra_inputs is not None:
                # ``cache_ids`` here is the caller-visible flat-row list so
                # the extras hook can slice per-item side inputs (rope,
                # deepstack) after we route to different Model slots.
                extras = pack_extra_inputs(
                    chunk_start=chunk_start,
                    sequence_lengths_chunks=g_seq_lens,
                    cache_ids=g_orig_ids,
                )
                infer_inputs.extend(extras)
            # ``BatchParam.cache_id`` is the LOCAL slot id inside the target
            # Model (``0..k_per_model - 1``), not the flat batch row.
            batch_params = [
                qbruntime.BatchParam(
                    sequence_length=g_seq_lens[k],
                    cache_size=g_cache_sizes[k],
                    cache_id=group_local_ids[k],
                )
                for k in range(len(group_items))
            ]
            return inputs_embeds_numpy.shape, infer_inputs, batch_params, g_seq_lens

        # Single-group fast path: keep the pre-refactor behavior verbatim
        # (one blocking infer call, no thread pool overhead). Covers N == 1
        # backends and multi-slot backends whose batch lands on one Model.
        if len(groups) == 1:
            m_idx, group_items, group_local_ids = groups[0]
            input_shape, infer_inputs, batch_params, _seq_lens = _build_group_payload(
                group_items, group_local_ids
            )
            if count_npu_time:
                t0 = time.perf_counter()
                result = mxq_models[m_idx].infer(infer_inputs, None, 0, batch_params)
                elapsed = time.perf_counter() - t0
                phase: PhaseName = phase_override or _classify_phase()
                if record_npu_time is not None:
                    record_npu_time(phase, elapsed)
            else:
                result = mxq_models[m_idx].infer(infer_inputs, None, 0, batch_params)
            if result is None:
                raise RuntimeError("mxq infer result is None!")
            return result[0], input_shape

        # Multi-group parallel dispatch: one blocking ``.infer`` per Model
        # slot dispatched from its own thread. ``qbruntime.Model.infer``
        # releases the GIL for the duration of the NPU call so wall time
        # drops to roughly ``max(group_elapsed)`` on independent slots.
        per_group_payload: List[
            Tuple[Tuple[int, ...], List[np.ndarray], List[qbruntime.BatchParam], List[int]]
        ] = [_build_group_payload(g[1], g[2]) for g in groups]

        group_raw: List[Optional[np.ndarray]] = [None] * len(groups)
        group_elapsed: List[float] = [0.0] * len(groups)

        def _run(gi: int) -> None:
            m_idx = groups[gi][0]
            _shape, infer_inputs, batch_params, _seq_lens = per_group_payload[gi]
            t0 = time.perf_counter()
            result = mxq_models[m_idx].infer(infer_inputs, None, 0, batch_params)
            group_elapsed[gi] = time.perf_counter() - t0
            if result is None:
                raise RuntimeError("mxq infer result is None!")
            group_raw[gi] = np.asarray(result[0])

        t_wall_0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            futures = [executor.submit(_run, gi) for gi in range(len(groups))]
            for f in futures:
                # ``.result()`` re-raises worker exceptions here so the caller
                # sees the underlying qbruntime error rather than a silently
                # missing group_raw slot.
                f.result()
        wall_elapsed = time.perf_counter() - t_wall_0

        if count_npu_time:
            phase = phase_override or _classify_phase()
            if record_npu_time is not None:
                # Aggregate the wall time (not the sum of group times):
                # parallel work on independent slots overlaps, and doubling
                # it would inflate TPS-facing counters.
                record_npu_time(phase, wall_elapsed)

        if debug_enabled:
            logger.debug(
                "[BATCH-LLM][PARALLEL] n_groups=%d wall=%.6fs group_elapsed=%s "
                "model_indices=%s group_item_counts=%s",
                len(groups),
                wall_elapsed,
                [f"{e:.6f}" for e in group_elapsed],
                [g[0] for g in groups],
                [len(g[1]) for g in groups],
            )

        group_raw_arrs: List[np.ndarray] = [cast(np.ndarray, arr) for arr in group_raw]
        group_seq_lens: List[List[int]] = [payload[3] for payload in per_group_payload]
        merged = self._merge_group_outputs(
            group_raw_arrs,
            groups,
            group_seq_lens,
            sequence_lengths,
            n_items,
        )

        # Return the first group's shape so the ``[BATCH-LLM]`` debug lines in
        # the caller keep printing something meaningful without synthesizing a
        # fake shape across heterogeneous groups.
        return merged, per_group_payload[0][0]

    # ------------------------------------------------------------------
    # Merge helpers
    # ------------------------------------------------------------------

    def _merge_group_outputs(
        self,
        group_raw: List[np.ndarray],
        groups: List[Tuple[int, List[int], List[int]]],
        group_seq_lens: List[List[int]],
        sequence_lengths: List[int],
        n_items: int,
    ) -> np.ndarray:
        """Merge per-group outputs into a single row-ordered ndarray.

        Reads ``backend.output_layout``; falls back to inspecting an
        unambiguous group and pins the answer on the backend when the
        compile-time probe was ambiguous. The old shape-inference-per-group
        branch is intentionally gone: a first group of size-1 rows (decode
        step) collapses ``first_n_items == first_n_tokens`` and used to
        default silently to layout A, truncating longer prompts in later
        groups.

        Belt-and-suspenders: the cached layout is cross-checked against an
        unambiguous group's actual row count. If they disagree — which
        happens when the compile-time probe locked ``"n_tokens"`` on a
        ``K > 1`` MXQ whose compiled batch axis was reported dynamic at
        position ``-2`` — the runtime observation wins and the backend
        cache is overwritten via :meth:`MobilintNPUBackend._set_output_layout`.
        """
        layout = self.backend.output_layout
        observed = self._observe_layout_from_groups(group_raw, groups, group_seq_lens)
        if observed is not None:
            if layout is None:
                layout = observed
                self.backend._set_output_layout(layout)
            elif layout != observed:
                logger.debug(
                    "output_layout override: probe pinned %r but runtime observed %r; "
                    "re-pinning backend cache",
                    layout,
                    observed,
                )
                layout = observed
                self.backend._set_output_layout(layout)
        elif layout is None:
            raise RuntimeError(
                "Cannot resolve output_layout: every non-empty group has "
                "n_rows == n_items == n_tokens (typically an all-decode batch). "
                "Re-issue a dispatch that includes at least one row with seq_len > 1."
            )

        # Find the first non-empty group so we can size the merged buffer.
        first_arr: Optional[np.ndarray] = None
        for arr in group_raw:
            if arr.size > 0:
                first_arr = arr
                break
        if first_arr is None:
            raise RuntimeError("every group returned an empty output")
        vocab = int(first_arr.shape[-1])

        if layout == "n_items":
            merged = np.empty((n_items, vocab), dtype=first_arr.dtype)
            for gi, (_m, group_items, _local_ids) in enumerate(groups):
                g_flat = group_raw[gi].reshape(-1, vocab)
                for r, item_idx in enumerate(group_items):
                    merged[item_idx] = g_flat[r]
            return merged

        # ``"n_tokens"`` — per-token flat rows, indexed by caller-order offsets.
        total_tokens = sum(sequence_lengths)
        offsets = [0] * n_items
        running = 0
        for k in range(n_items):
            offsets[k] = running
            running += sequence_lengths[k]
        merged = np.empty((total_tokens, vocab), dtype=first_arr.dtype)
        for gi, (_m, group_items, _local_ids) in enumerate(groups):
            g_flat = group_raw[gi].reshape(-1, vocab)
            g_off = 0
            for item_idx in group_items:
                len_k = sequence_lengths[item_idx]
                merged[offsets[item_idx] : offsets[item_idx] + len_k] = g_flat[g_off : g_off + len_k]
                g_off += len_k
        return merged

    @staticmethod
    def _observe_layout_from_groups(
        group_raw: List[np.ndarray],
        groups: List[Tuple[int, List[int], List[int]]],
        group_seq_lens: List[List[int]],
    ) -> Optional[LayoutName]:
        """Inspect groups until one is unambiguous, then return that layout.

        A group is ambiguous when its row count equals both ``n_items`` and
        ``n_tokens`` (typically all rows have ``seq_len == 1``). Skip those
        and read the next; when every non-empty group is ambiguous, return
        ``None`` so the caller can decide whether to trust a cached layout
        or raise.
        """
        for gi, arr in enumerate(group_raw):
            if arr.size == 0:
                continue
            vocab = int(arr.shape[-1])
            n_rows = arr.size // vocab
            n_group_items = len(groups[gi][1])
            n_group_tokens = sum(group_seq_lens[gi])
            if n_rows == n_group_items and n_rows != n_group_tokens:
                return "n_items"
            if n_rows == n_group_tokens and n_rows != n_group_items:
                return "n_tokens"
            if n_rows != n_group_items and n_rows != n_group_tokens:
                raise RuntimeError(
                    "Unexpected group MXQ output row count "
                    f"{n_rows} (vocab={vocab}) for active={n_group_items}, "
                    f"total_tokens={n_group_tokens}"
                )
        return None

    @classmethod
    def _resolve_layout_from_groups(
        cls,
        group_raw: List[np.ndarray],
        groups: List[Tuple[int, List[int], List[int]]],
        group_seq_lens: List[List[int]],
    ) -> LayoutName:
        """Inspect groups until one is unambiguous, then return that layout.

        Raises when every non-empty group is ambiguous.
        """
        observed = cls._observe_layout_from_groups(group_raw, groups, group_seq_lens)
        if observed is None:
            raise RuntimeError(
                "Cannot resolve output_layout: every non-empty group has "
                "n_rows == n_items == n_tokens (typically an all-decode batch). "
                "Re-issue a dispatch that includes at least one row with seq_len > 1."
            )
        return observed
