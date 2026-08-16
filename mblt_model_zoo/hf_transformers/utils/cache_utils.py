import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import qbruntime
import torch
from transformers.cache_utils import Cache

try:
    from transformers.cache_utils import CacheLayerMixin
except ImportError:
    # transformers < 4.54 compat shim: CacheLayerMixin was introduced in transformers 4.54.0.
    # This stub only satisfies subclassing at class-definition time so this module can import
    # cleanly for GPU-only workflows (e.g. running text-generation benchmarks on transformers
    # 4.53.x, which is required by some third-party custom modeling code such as EXAONE-3.5).
    # Any actual MobilintCache/MobilintLayer runtime path relies on the real CacheLayerMixin
    # contract and requires transformers>=4.54; instantiating those on the stub will fail
    # downstream. That is intentional — the stub must not silently masquerade as the real class.
    class CacheLayerMixin:  # type: ignore[no-redef]
        """Compat stub for transformers<4.54; MobilintCache paths require the real class."""

        pass


def is_whisper_beam_debug_trace_enabled() -> bool:
    """Return whether Whisper beam-cache debug tracing is enabled."""
    return bool(os.environ.get("MBLT_WHISPER_BEAM_DEBUG_TRACE"))


def append_whisper_beam_debug_event(event: dict[str, Any]) -> None:
    """Append one Whisper beam-cache debug event when tracing is enabled."""
    trace_path = os.environ.get("MBLT_WHISPER_BEAM_DEBUG_TRACE")
    if not trace_path:
        return
    payload = {"time_s": time.time(), **event}
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


class MobilintLayer(CacheLayerMixin):
    is_sliding = False

    def __init__(self, mxq_model: qbruntime.Model, cache_id: int = 0):
        self.mxq_model = mxq_model
        self.cache_id = cache_id
        self._seen_tokens = 0
        self.buffer: list[bytes] = []
        self.buffer_seq_length: Optional[int] = None

    def lazy_initialization(self, key_states: torch.Tensor):
        raise NotImplementedError("lazy_initialization is not implemented")

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: Optional[dict[str, Any]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("update is not implemented")

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    def get_seq_length(self, cache_position=None) -> int:
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        return self.mxq_model.get_input_buffer_info()[0].max_cache_size

    def set_seq_length(self, seq_length: int) -> None:
        """Set the cached sequence length for an in-memory cache snapshot."""
        if seq_length < 0:
            raise ValueError(f"seq_length must be non-negative, got {seq_length}")
        self._seen_tokens = seq_length

    def fake_prefill(self, seq_length: int) -> None:
        """Mark this layer as prefilled without restoring KV cache memory.

        This helper is intended for NPU decode TPS benchmarks where qbruntime
        receives the requested cache size directly and the actual KV payload is
        not needed for measuring decode compute cost.
        """
        self.reset()
        self.set_seq_length(seq_length)

    def reset(self) -> None:
        self._seen_tokens = 0
        self.buffer = []
        self.buffer_seq_length = None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError("reorder_cache is not implemented")

    def update_cache_position(self, cache_position: torch.Tensor):
        self._seen_tokens += cache_position.numel()

    def update_seen_tokens(self, num_new_seen_tokens: int):
        self._seen_tokens += num_new_seen_tokens

    def dump_cache_memory(self):
        self.buffer = self.mxq_model.dump_cache_memory(self.cache_id)
        self.buffer_seq_length = self.get_seq_length()

    def load_cache_memory(self):
        if self.buffer_seq_length is not None:
            self.set_seq_length(self.buffer_seq_length)
        if self.get_seq_length() > 0:
            self.mxq_model.load_cache_memory(self.buffer, self.cache_id)

    def dump_cache_memory_to(self, cache_dir: str):
        self.mxq_model.dump_cache_memory_to(cache_dir, self.cache_id)
        seq_path = Path(cache_dir) / "seq_length.txt"
        seq_path.write_text(f"{self.get_seq_length()}\n", encoding="utf-8")

    def load_cache_memory_from(self, cache_dir: str):
        self.reset()
        seq_path = Path(cache_dir) / "seq_length.txt"
        if seq_path.exists():
            seq_length = int(seq_path.read_text(encoding="utf-8").strip())
        else:
            seq_length = 0
        self.set_seq_length(seq_length)
        self.mxq_model.load_cache_memory_from(cache_dir, self.cache_id)

    def copy(self) -> "MobilintLayer":
        copied = MobilintLayer(self.mxq_model, self.cache_id)
        copied._seen_tokens = self._seen_tokens
        copied.buffer = copy.deepcopy(self.buffer)
        copied.buffer_seq_length = self.buffer_seq_length
        return copied


class MobilintCache(Cache):
    def __init__(
        self,
        mxq_models: Union[List[qbruntime.Model], qbruntime.Model],
        per_model_batch: int = 1,
        *,
        batch_size: Optional[int] = None,
    ):
        """Create a cache with fresh logical cursors across one or more runtime models.

        The cache dualizes KV state along ``(model_idx, cache_id)``: it holds
        ``N = len(mxq_models)`` Model handles with ``K = per_model_batch``
        cache slots each, laid out as flat rows so external callers can keep
        the historical row-index API. Row ``i`` maps to
        ``(i // K, i % K)``; helpers :meth:`slot_of`, :meth:`model_of`, and
        :meth:`group_by_model` expose that routing without leaking the slot
        concept into upstream dispatch.

        qbruntime selects the readable KV prefix from the ``cache_size`` passed
        to inference; fresh layers therefore start at sequence length zero,
        causing the next inference to overwrite cache entries from the
        beginning rather than making a previous request's KV prefix
        addressable.

        Args:
            mxq_models: One :class:`qbruntime.Model` or a list of them. A
                single Model is promoted to a length-1 list for backward
                compatibility.
            per_model_batch: Cache slots per Model (``K``). Total rows =
                ``N * K``.
            batch_size: Legacy keyword-only alias for ``per_model_batch``.
                Only accepted when the cache hosts a single Model (``N = 1``)
                so that ``MobilintCache(model, batch_size=K)`` keeps working;
                rejected with a multi-Model list because the legacy alias
                would silently double the total capacity and pin the first
                ``K`` rows entirely to slot 0.

        Raises:
            TypeError: If both ``per_model_batch`` and legacy ``batch_size``
                are provided, or if the legacy ``batch_size`` is combined
                with a multi-Model list.
            ValueError: If ``mxq_models`` is an empty list.
        """
        if isinstance(mxq_models, list):
            models_list: List[qbruntime.Model] = list(mxq_models)
        else:
            models_list = [mxq_models]

        if not models_list:
            raise ValueError("mxq_models must contain at least one Model")

        if batch_size is not None:
            if per_model_batch != 1:
                raise TypeError(
                    "Pass either per_model_batch or the legacy batch_size, not both"
                )
            if len(models_list) > 1:
                raise TypeError(
                    "MobilintCache legacy batch_size= is a single-Model (N=1) shim; "
                    f"pass per_model_batch=K instead (got N={len(models_list)} Models). "
                    "The legacy alias would misroute rows across slots and defeat "
                    "multi-slot dispatch."
                )
            per_model_batch = int(batch_size)

        self.mxq_models: List[qbruntime.Model] = models_list
        self.k_per_model: int = max(1, int(per_model_batch))
        self.n_models: int = len(self.mxq_models)
        self.batch_size: int = self.n_models * self.k_per_model

        self.layers: list[MobilintLayer] = [
            MobilintLayer(self.mxq_models[model_idx], cache_id)
            for model_idx in range(self.n_models)
            for cache_id in range(self.k_per_model)
        ]
        self.layer_classes = MobilintLayer

        self.num_hidden_layers = 1
        self.cache_processor = None

    @property
    def mxq_model(self) -> Optional[qbruntime.Model]:
        """First Model handle for callers written against the pre-multi-Model API."""
        return self.mxq_models[0] if self.mxq_models else None

    def slot_of(self, row: int) -> Tuple[int, int]:
        """Return ``(model_idx, local_cache_id)`` for a flat row index."""
        row = int(row)
        if row < 0 or row >= len(self.layers):
            raise IndexError(
                f"row {row} out of range for cache with {len(self.layers)} rows "
                f"(N={self.n_models}, K={self.k_per_model})"
            )
        return divmod(row, self.k_per_model)

    def model_of(self, row: int) -> qbruntime.Model:
        """Return the :class:`qbruntime.Model` that owns the KV state for ``row``."""
        model_idx, _ = self.slot_of(row)
        return self.mxq_models[model_idx]

    def group_by_model(self, rows: Iterable[int]) -> Dict[int, List[Tuple[int, int]]]:
        """Group flat rows by owning Model.

        Args:
            rows: Iterable of flat row indices.

        Returns:
            Mapping ``model_idx -> [(row, local_cache_id), ...]`` in the input
            order, so upstream dispatch can issue one blocking
            :meth:`qbruntime.Model.infer` per Model without inspecting slots.
        """
        grouped: Dict[int, List[Tuple[int, int]]] = {}
        for row in rows:
            model_idx, local_cache_id = self.slot_of(int(row))
            grouped.setdefault(model_idx, []).append((int(row), local_cache_id))
        return grouped

    def get_seq_length(self, index: int = 0) -> int:
        return self.layers[index].get_seq_length()

    def set_seq_length(self, sequence_lengths: Union[dict[int, int], int], index: int = 0) -> None:
        """Set cached sequence lengths for one cache entry or a batch of entries."""
        if isinstance(sequence_lengths, int):
            self.layers[index].set_seq_length(sequence_lengths)
            return
        for cache_id, seq_len in sequence_lengths.items():
            self.layers[cache_id].set_seq_length(seq_len)

    def fake_prefill(self, sequence_lengths: Union[dict[int, int], int], index: int = 0) -> None:
        """Mark one or more cache entries as prefilled without loading cache memory.

        Args:
            sequence_lengths: Single sequence length or per-cache-id sequence
                lengths to expose via ``get_seq_length()``. A single sequence length
                is applied to every cache entry in the batch.
            index: Unused compatibility argument for scalar sequence lengths.

        Raises:
            ValueError: If any sequence length is negative.
        """
        if isinstance(sequence_lengths, int):
            for layer in self.layers:
                layer.fake_prefill(sequence_lengths)
            return
        for cache_id, seq_len in sequence_lengths.items():
            self.layers[cache_id].fake_prefill(seq_len)

    def update_cache_position(self, cache_position: torch.Tensor, index: int = 0):
        self.layers[index].update_cache_position(cache_position)

    def update_seen_tokens(self, sequence_lengths: Union[dict[int, int], int], index: int = 0):
        if isinstance(sequence_lengths, int):
            self.layers[index].update_seen_tokens(sequence_lengths)
            return
        for cache_id, seq_len in sequence_lengths.items():
            self.layers[cache_id].update_seen_tokens(seq_len)

    def dump_cache_memory(self):
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].dump_cache_memory()

    def load_cache_memory(self):
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].load_cache_memory()

    def dump_cache_memory_to(self, cache_dir: str, index: int = 0):
        self.layers[index].dump_cache_memory_to(cache_dir)

    def load_cache_memory_from(self, cache_dir: str, index: int = 0):
        self.layers[index].load_cache_memory_from(cache_dir)

    def reset(self) -> None:
        """Reset all cache entries in this Mobilint cache."""
        for layer in self.layers:
            layer.reset()

    def ensure_batch_size(self, batch_size: int) -> None:
        """Grow logical cache entries so batched generation can track each active row.

        Growth beyond ``N * K`` is only supported on the legacy single-Model
        hardware-batch path (``N == 1``); multi-Model caches must be sized
        upfront because slot count is fixed by the backend.
        """
        batch_size = max(1, int(batch_size))
        if batch_size <= self.batch_size:
            return
        if self.n_models != 1:
            raise ValueError(
                f"cannot grow multi-Model cache beyond {self.batch_size} rows "
                f"(N={self.n_models}, K={self.k_per_model}); allocate a larger cache upfront"
            )
        only_model = self.mxq_models[0]
        for cache_id in range(self.batch_size, batch_size):
            self.layers.append(MobilintLayer(only_model, cache_id))
        self.batch_size = batch_size
        self.k_per_model = batch_size

    def copy(self):
        copied = MobilintCache(list(self.mxq_models), per_model_batch=self.k_per_model)
        if len(copied.layers) < self.batch_size:
            copied.ensure_batch_size(self.batch_size)
        for i in range(len(self.layers)):
            copied.layers[i] = self.layers[i].copy()
        return copied


def _resolve_language_model_candidate(model: Any) -> Optional[Any]:
    """Return the nested language model commonly used by VLM wrappers."""
    nested_model = getattr(model, "model", None)
    if nested_model is not None:
        language_model = getattr(nested_model, "language_model", None)
        if language_model is not None:
            return language_model
    return getattr(model, "language_model", None)


def _call_maybe_getter(obj: Any, name: str) -> Optional[Any]:
    """Return an attribute value, calling it when it is a zero-argument getter."""
    candidate = getattr(obj, name, None)
    if candidate is None:
        return None
    if callable(candidate):
        try:
            return candidate()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
    return candidate


def resolve_multi_slot_backend(model: Any) -> Optional[Any]:
    """Return the Mobilint NPU backend that hosts one or more Model slots for ``model``.

    Walks ``model`` and its nested language model (the two shapes current VLM
    wrappers use — ``model.language_model`` and ``model.model.language_model``)
    looking for an ``npu_backend`` that exposes a non-empty ``mxq_models``
    list. Returns ``None`` for models that do not expose a Mobilint backend
    (e.g. unit-test stubs that only carry a bare ``get_cache_mxq_model``).
    """
    for candidate in (model, _resolve_language_model_candidate(model)):
        if candidate is None:
            continue
        backend = getattr(candidate, "npu_backend", None)
        if backend is None:
            continue
        if getattr(backend, "mxq_models", None):
            return backend
    return None


def resolve_cache_mxq_model(model: Any) -> Optional[qbruntime.Model]:
    """Return the single ``qbruntime.Model`` used by the legacy cache path.

    Falls back through ``get_cache_mxq_model``, then ``get_mxq_model`` on both
    ``model`` and its nested language model, so wrappers that override
    ``get_cache_mxq_model`` to delegate into ``language_model`` continue to work.
    """
    for candidate in (model, _resolve_language_model_candidate(model)):
        if candidate is None:
            continue
        for getter_name in ("get_cache_mxq_model", "get_mxq_model"):
            mxq_model = _call_maybe_getter(candidate, getter_name)
            if mxq_model is not None:
                return mxq_model
    return None


def build_mobilint_cache_from_model(
    model: Any,
    batch_size: int,
    *,
    cache_cls: Type[MobilintCache] = MobilintCache,
    **cache_kwargs: Any,
) -> MobilintCache:
    """Build a Mobilint cache routed across every ``qbruntime.Model`` slot the backend hosts.

    Uses the multi-slot ``cache_cls(mxq_models, per_model_batch=K)`` signature
    when the model exposes a multi-slot :class:`MobilintNPUBackend`, so
    :meth:`MobilintCache.slot_of` routes each flat row to its owning Model.
    Falls back to ``cache_cls(mxq_model, batch_size=batch_size)`` when the
    backend cannot be resolved (unit-test stubs and single-Model wrappers
    without a discoverable backend).

    Growing beyond ``n_models * k_per_model`` is only supported on the legacy
    single-Model hardware-batch path (``ensure_batch_size`` raises otherwise);
    multi-slot caches must be sized upfront by the backend.
    """
    backend = resolve_multi_slot_backend(model)
    if backend is None:
        mxq_model = resolve_cache_mxq_model(model)
        if mxq_model is None:
            raise RuntimeError(
                "Cannot build MobilintCache: no Mobilint NPU backend or "
                "get_cache_mxq_model resolver on this model."
            )
        return cache_cls(mxq_model, batch_size=batch_size, **cache_kwargs)

    mxq_models = list(getattr(backend, "mxq_models", []) or [])
    if not mxq_models:
        raise RuntimeError("Mobilint NPU backend has no loaded Model slots.")
    k_per_model = int(getattr(backend, "k_per_model", 1) or 1)
    cache = cache_cls(mxq_models, per_model_batch=k_per_model, **cache_kwargs)
    if batch_size > cache.batch_size:
        # ensure_batch_size raises for multi-Model caches (N > 1); the legacy
        # single-Model path can still grow here.
        cache.ensure_batch_size(batch_size)
    return cache


def cache_matches_backend_topology(cache: Any, model: Any) -> bool:
    """Return True when ``cache`` was built for ``model``'s current backend topology.

    Cache reuse across a dispose+recreate cycle must invalidate on any change to
    ``(mxq_models, k_per_model)`` because :meth:`MobilintCache.slot_of` and
    :meth:`MobilintCache.model_of` bake that routing in at construction time. Two
    backends with the same aggregate row capacity (e.g. ``N=2, K=2`` vs
    ``N=4, K=1``) hand out incompatible ``(model_idx, cache_id)`` pairs, so a
    reuse guard that only checks ``batch_size`` would silently misroute rows.

    Legacy single-Model fallback (no discoverable multi-slot backend): the cache
    was built via ``cache_cls(mxq_model, batch_size=B)`` and may have grown via
    :meth:`MobilintCache.ensure_batch_size`, so only the Model handle identity
    is compared and ``k_per_model`` growth is left to the caller's capacity check.
    """
    if not isinstance(cache, MobilintCache):
        return False

    backend = resolve_multi_slot_backend(model)
    if backend is not None:
        backend_mxq_models = list(getattr(backend, "mxq_models", []) or [])
        backend_k_per_model = int(getattr(backend, "k_per_model", 1) or 1)
        if cache.k_per_model != backend_k_per_model:
            return False
    else:
        fallback_mxq_model = resolve_cache_mxq_model(model)
        if fallback_mxq_model is None:
            return False
        backend_mxq_models = [fallback_mxq_model]

    if len(cache.mxq_models) != len(backend_mxq_models):
        return False
    for cached_model, backend_model in zip(cache.mxq_models, backend_mxq_models):
        if cached_model is not backend_model:
            return False
    return True


class MobilintBeamCache(MobilintCache):
    """Mobilint beam cache tracked by token histories instead of KV snapshots.

    qbruntime owns one active KV cache. This class tracks the token history for
    each logical beam and the token history currently represented by the active
    qbruntime cache. Callers can compare a target beam history with the active
    history, skip the common prefix, and forward only the suffix with the proper
    cache position.

    The beam-cache dispatch path is ``N == 1`` only: it issues one blocking
    ``mxq_model.infer`` on the single tracked slot with no cross-slot routing or
    beam-cache reorder. The ``mxq_models`` argument alone cannot detect the
    owning-backend topology when a caller passes ``get_cache_mxq_model()`` (which
    returns slot 0 as a single :class:`qbruntime.Model`) even though the backend
    launched ``N > 1`` slots. Callers that own a Mobilint NPU backend should pass
    ``n_slots=backend.dispatcher.n_slots`` (or equivalent) so the invariant is
    enforced at construction time rather than deep in generation. When
    ``n_slots`` is not supplied, only the existing multi-Model list guard runs
    to preserve backward compatibility for unit-test stubs.
    """

    def __init__(
        self,
        mxq_models: Union[List[qbruntime.Model], qbruntime.Model],
        batch_size: int = 1,
        *,
        n_slots: Optional[int] = None,
    ) -> None:
        if isinstance(mxq_models, list) and len(mxq_models) > 1:
            raise NotImplementedError(
                "MobilintBeamCache does not support multi-Model dispatch (N > 1); "
                "beam search keeps N=1 (encoder-decoder) — use MobilintCache for N > 1"
            )
        if n_slots is not None and int(n_slots) > 1:
            raise NotImplementedError(
                "MobilintBeamCache does not support multi-slot dispatch "
                f"(owning backend launched N={int(n_slots)} slots); beam search "
                "keeps N=1. Cap the owning backend's max_batch_size at K so it "
                "keeps a single slot, or compile a batched (K>1) MXQ so slot-0 "
                "hardware batching serves the load — use MobilintCache for N > 1."
            )
        super().__init__(mxq_models=mxq_models, batch_size=batch_size)
        self._beam_token_histories: list[list[int]] = [[] for _ in range(self.batch_size)]
        self._beam_source_indices: list[int | None] = [None for _ in range(self.batch_size)]
        self._active_token_history: list[int] = []
        self._active_source_index: int | None = None
        self._beam_seq_lengths: list[int] = [0 for _ in range(self.batch_size)]

    def reset(self) -> None:
        """Reset active qbruntime cache bookkeeping and clear beam token histories."""
        super().reset()
        self._beam_token_histories = [[] for _ in range(self.batch_size)]
        self._beam_source_indices = [None for _ in range(self.batch_size)]
        self._active_token_history = []
        self._active_source_index = None
        self._beam_seq_lengths = [0 for _ in range(self.batch_size)]

    def matches_live_topology(
        self,
        expected_mxq_model: qbruntime.Model,
        n_slots: Optional[int] = None,
    ) -> bool:
        """Return True when this cache can be reused for the current backend topology.

        The beam-cache contract is ``N == 1``: one active qbruntime cache and one
        Model handle. Callers reuse an existing beam cache via :meth:`reset`, but
        the constructor-time guard does not re-run on that path. This helper
        performs the same invariant check so ``_get_cache`` implementations can
        rebuild rather than reset when either (a) the owning backend now hosts
        more than one slot (breaking the ``N == 1`` contract) or (b) the stored
        Model handle no longer matches the current slot 0 (the backend was
        disposed and re-created with a fresh Model).
        """
        if n_slots is not None and int(n_slots) > 1:
            return False
        cached_models = getattr(self, "mxq_models", None)
        if not cached_models:
            return False
        if len(cached_models) != 1:
            return False
        return cached_models[0] is expected_mxq_model

    def ensure_batch_size(self, batch_size: int) -> None:
        """Grow logical beam token storage for beam-expanded generation."""
        previous_batch_size = self.batch_size
        super().ensure_batch_size(batch_size)
        if self.batch_size <= previous_batch_size:
            return
        self._beam_token_histories.extend([[] for _ in range(self.batch_size - previous_batch_size)])
        self._beam_source_indices.extend([None for _ in range(self.batch_size - previous_batch_size)])
        self._beam_seq_lengths.extend([0 for _ in range(self.batch_size - previous_batch_size)])

    def get_seq_length(self, index: int = 0) -> int:
        """Return the stored sequence length for one logical beam."""
        self.ensure_batch_size(index + 1)
        return self._beam_seq_lengths[index]

    def set_seq_length(self, sequence_lengths: Union[dict[int, int], int], index: int = 0) -> None:
        """Set stored sequence lengths for one or more logical beams."""
        if isinstance(sequence_lengths, int):
            self.ensure_batch_size(index + 1)
            if sequence_lengths < 0:
                raise ValueError(f"seq_length must be non-negative, got {sequence_lengths}")
            self._beam_seq_lengths[index] = sequence_lengths
            self._beam_token_histories[index] = self._beam_token_histories[index][:sequence_lengths]
            self.layers[index].set_seq_length(sequence_lengths)
            return
        if sequence_lengths:
            self.ensure_batch_size(max(sequence_lengths) + 1)
        for beam_id, seq_len in sequence_lengths.items():
            if seq_len < 0:
                raise ValueError(f"seq_length must be non-negative, got {seq_len}")
            self._beam_seq_lengths[beam_id] = seq_len
            self._beam_token_histories[beam_id] = self._beam_token_histories[beam_id][:seq_len]
            self.layers[beam_id].set_seq_length(seq_len)

    def update_cache_position(self, cache_position: torch.Tensor, index: int = 0) -> None:
        """Update one logical beam length after its active qbruntime cache advances."""
        self.ensure_batch_size(index + 1)
        self._beam_seq_lengths[index] += int(cache_position.numel())
        self.layers[index].set_seq_length(self._beam_seq_lengths[index])

    def build_target_tokens(self, beam_index: int, input_ids: torch.Tensor) -> list[int]:
        """Return the target token history for one beam after appending new ids."""
        self.ensure_batch_size(beam_index + 1)
        new_tokens = self._tensor_to_token_list(input_ids)
        return [*self._beam_token_histories[beam_index], *new_tokens]

    def get_beam_tokens(self, beam_index: int) -> list[int]:
        """Return a copy of the stored token history for one logical beam."""
        self.ensure_batch_size(beam_index + 1)
        return list(self._beam_token_histories[beam_index])

    def get_beam_source_index(self, beam_index: int) -> int | None:
        """Return the source row identity stored for one logical beam."""
        self.ensure_batch_size(beam_index + 1)
        return self._beam_source_indices[beam_index]

    def set_beam_source_indices(self, source_indices: Sequence[int | None]) -> None:
        """Store source row identities for logical beams when they are first resolved."""
        self.ensure_batch_size(len(source_indices))
        for beam_index, source_index in enumerate(source_indices):
            self._beam_source_indices[beam_index] = None if source_index is None else int(source_index)

    def get_active_tokens(self) -> list[int]:
        """Return a copy of the token history represented by the active qbruntime cache."""
        return list(self._active_token_history)

    def get_active_source_index(self) -> int | None:
        """Return the source row represented by the active qbruntime cache."""
        return self._active_source_index

    def get_common_prefix_length(self, target_tokens: Sequence[int], source_index: int | None = None) -> int:
        """Return how many target tokens already match the active qbruntime cache."""
        if source_index is not None and self._active_source_index != int(source_index):
            return 0
        prefix_length = 0
        for active_token, target_token in zip(self._active_token_history, target_tokens):
            if active_token != target_token:
                break
            prefix_length += 1
        return prefix_length

    def commit_beam_tokens(self, beam_index: int, target_tokens: Sequence[int]) -> None:
        """Store the completed target history for one logical beam."""
        self.ensure_batch_size(beam_index + 1)
        token_history = [int(token) for token in target_tokens]
        self._beam_token_histories[beam_index] = token_history
        self._beam_seq_lengths[beam_index] = len(token_history)
        self.layers[beam_index].set_seq_length(len(token_history))

    def commit_active_tokens(self, target_tokens: Sequence[int], source_index: int | None = None) -> None:
        """Record which token history is now represented by active qbruntime cache memory."""
        self._active_token_history = [int(token) for token in target_tokens]
        self._active_source_index = None if source_index is None else int(source_index)
        self.layers[0].set_seq_length(len(self._active_token_history))

    def _tensor_to_token_list(self, input_ids: torch.Tensor) -> list[int]:
        """Convert a one-row token tensor to a flat Python token list."""
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor")
        if input_ids.ndim == 0:
            input_ids = input_ids.reshape(1)
        return [int(token) for token in input_ids.reshape(-1).detach().cpu().tolist()]

    def reorder_cache(self, beam_idx: torch.LongTensor) -> "MobilintBeamCache":
        """Reorder application-level beam token histories in HF beam order."""
        beam_idx = self._validate_beam_indices(beam_idx)
        trace_enabled = is_whisper_beam_debug_trace_enabled()

        if trace_enabled:
            append_whisper_beam_debug_event(
                {
                    "event": "cache_reorder_before",
                    "beam_idx": [int(index) for index in beam_idx.cpu().tolist()],
                    "beam_token_histories": [list(tokens) for tokens in self._beam_token_histories],
                    "beam_source_indices": list(self._beam_source_indices),
                    "beam_seq_lengths": list(self._beam_seq_lengths),
                    "active_token_history": list(self._active_token_history),
                    "active_source_index": self._active_source_index,
                }
            )

        if torch.equal(beam_idx.cpu(), torch.arange(int(beam_idx.numel()), dtype=torch.long)):
            if trace_enabled:
                append_whisper_beam_debug_event(
                    {
                        "event": "cache_reorder_identity",
                        "beam_idx": [int(index) for index in beam_idx.cpu().tolist()],
                        "active_token_history": list(self._active_token_history),
                        "active_source_index": self._active_source_index,
                    }
                )
            return self

        old_token_histories = [list(tokens) for tokens in self._beam_token_histories]
        old_source_indices = list(self._beam_source_indices)
        old_seq_lengths = list(self._beam_seq_lengths)
        beam_indices = [int(index) for index in beam_idx.cpu().tolist()]
        self._beam_token_histories = [list(old_token_histories[index]) for index in beam_indices]
        self._beam_source_indices = [old_source_indices[index] for index in beam_indices]
        self._beam_seq_lengths = [old_seq_lengths[index] for index in beam_indices]
        for beam_id, seq_length in enumerate(self._beam_seq_lengths):
            self.layers[beam_id].set_seq_length(seq_length)
        if trace_enabled:
            append_whisper_beam_debug_event(
                {
                    "event": "cache_reorder_after",
                    "beam_idx": beam_indices,
                    "beam_token_histories": [list(tokens) for tokens in self._beam_token_histories],
                    "beam_source_indices": list(self._beam_source_indices),
                    "beam_seq_lengths": list(self._beam_seq_lengths),
                    "active_token_history": list(self._active_token_history),
                    "active_source_index": self._active_source_index,
                }
            )
        return self

    def _validate_beam_indices(self, beam_idx: torch.LongTensor) -> torch.LongTensor:
        """Validate beam indices before reordering token histories."""
        if not isinstance(beam_idx, torch.Tensor):
            raise TypeError("beam_idx must be a torch.Tensor")
        if beam_idx.ndim != 1:
            raise ValueError(f"beam_idx must be rank 1, got shape {tuple(beam_idx.shape)}")

        beam_idx = beam_idx.to(dtype=torch.long)
        self.ensure_batch_size(int(beam_idx.numel()))
        if beam_idx.numel() > 0 and (int(beam_idx.min()) < 0 or int(beam_idx.max()) >= int(beam_idx.numel())):
            raise ValueError(f"beam_idx contains out-of-range values for {int(beam_idx.numel())} beams")
        return beam_idx

    def copy(self) -> "MobilintBeamCache":
        """Return a copy preserving application-level beam token histories."""
        copied = self.__class__(list(self.mxq_models), batch_size=self.k_per_model)
        if len(copied.layers) < self.batch_size:
            copied.ensure_batch_size(self.batch_size)
        for i in range(len(self.layers)):
            copied.layers[i] = self.layers[i].copy()
        copied._beam_token_histories = [list(tokens) for tokens in self._beam_token_histories]
        copied._beam_source_indices = list(self._beam_source_indices)
        copied._active_token_history = list(self._active_token_history)
        copied._active_source_index = self._active_source_index
        copied._beam_seq_lengths = list(self._beam_seq_lengths)
        return copied


class MobilintWhisperCache(MobilintBeamCache):
    """Whisper cache using token-history beam replay."""

    def __init__(
        self,
        mxq_models: Union[List[qbruntime.Model], qbruntime.Model],
        batch_size: int = 1,
        *,
        n_slots: Optional[int] = None,
    ) -> None:
        super().__init__(mxq_models=mxq_models, batch_size=batch_size, n_slots=n_slots)
        self._encoder_source_count: int | None = None

    def reset(self) -> None:
        """Reset beam cache state and forget the current encoder source grouping."""
        super().reset()
        self._encoder_source_count = None

    def set_encoder_source_count(self, source_count: int) -> None:
        """Record the original audio batch size before Hugging Face beam expansion."""
        source_count = int(source_count)
        if source_count < 1:
            raise ValueError(f"source_count must be positive, got {source_count}")
        self._encoder_source_count = source_count

    def get_encoder_source_count(self) -> int | None:
        """Return the original encoder source count when it is known."""
        return self._encoder_source_count

    def copy(self) -> "MobilintWhisperCache":
        """Return a copy preserving Whisper encoder source grouping metadata."""
        copied = super().copy()
        assert isinstance(copied, MobilintWhisperCache)
        copied._encoder_source_count = self._encoder_source_count
        return copied


class MobilintDeepStackCache(MobilintCache):
    """Mobilint KV cache carrying Qwen3-VL deepstack decoder inputs.

    Qwen3-VL text MXQ uses token embeddings and a dense deepstack tensor as decoder inputs.
    This cache keeps the KV sequence length in ``MobilintCache`` while providing the matching
    deepstack chunk for each decoder invocation. Fake prefill stores only the requested sequence
    length and lazily serves zero deepstack chunks for synthetic decode TPS measurements.

    The constructor mirrors :class:`MobilintCache`'s dual signature so a multi-slot backend can
    build the deepstack cache through :func:`build_mobilint_cache_from_model` with
    ``per_model_batch=K`` while the legacy single-Model path keeps its ``batch_size=B`` keyword.
    """

    def __init__(
        self,
        mxq_models: Union[List[qbruntime.Model], qbruntime.Model],
        per_model_batch: int = 1,
        *,
        num_deepstack_layers: int = 0,
        hidden_size: int = 0,
        batch_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            mxq_models=mxq_models,
            per_model_batch=per_model_batch,
            batch_size=batch_size,
        )
        if num_deepstack_layers < 0:
            raise ValueError(f"num_deepstack_layers must be non-negative, got {num_deepstack_layers}")
        if hidden_size < 0:
            raise ValueError(f"hidden_size must be non-negative, got {hidden_size}")
        self.num_deepstack_layers = int(num_deepstack_layers)
        self.hidden_size = int(hidden_size)
        self._deepstack_tensor: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Reset KV sequence length and clear any per-call deepstack tensor."""
        for layer in self.layers:
            layer.reset()
        self._deepstack_tensor = None

    def fake_prefill(self, sequence_lengths: Union[dict[int, int], int], index: int = 0) -> None:
        """Mark the cache as fake-prefilled and clear real deepstack payloads."""
        super().fake_prefill(sequence_lengths, index=index)
        self._deepstack_tensor = None

    def set_deepstack_tensor(self, deepstack_tensor: torch.Tensor) -> None:
        """Set the deepstack tensor for the current decoder forward call.

        Args:
            deepstack_tensor: Dense tensor with shape ``(layers, seq_len, hidden_size)``.

        Raises:
            ValueError: If the tensor rank or configured dimensions do not match.
        """
        if deepstack_tensor.ndim != 3:
            raise ValueError(f"Expected deepstack tensor rank 3, got shape {tuple(deepstack_tensor.shape)}")
        if int(deepstack_tensor.shape[0]) != self.num_deepstack_layers:
            raise ValueError(
                "Deepstack layer count mismatch: "
                f"{int(deepstack_tensor.shape[0])} vs {self.num_deepstack_layers}"
            )
        if int(deepstack_tensor.shape[2]) != self.hidden_size:
            raise ValueError(
                f"Deepstack hidden size mismatch: {int(deepstack_tensor.shape[2])} vs {self.hidden_size}"
            )
        self._deepstack_tensor = deepstack_tensor

    def get_deepstack_chunk(
        self,
        start_index: int,
        end_index: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the deepstack input chunk for the current decoder chunk.

        Args:
            start_index: Inclusive local token offset in the current forward call.
            end_index: Exclusive local token offset in the current forward call.
            device: Device for a lazily-created fake chunk.
            dtype: Dtype for a lazily-created fake chunk.

        Returns:
            Tensor with shape ``(layers, end_index - start_index, hidden_size)``.
        """
        if start_index < 0 or end_index < start_index:
            raise ValueError(f"Invalid deepstack chunk range: {start_index}:{end_index}")

        if self._deepstack_tensor is not None and end_index <= int(self._deepstack_tensor.shape[1]):
            return self._deepstack_tensor[:, start_index:end_index, :].to(device=device, dtype=dtype)

        chunk_len = end_index - start_index
        return torch.zeros(
            (self.num_deepstack_layers, chunk_len, self.hidden_size),
            dtype=dtype,
            device=device,
        )

    def copy(self) -> "MobilintDeepStackCache":
        """Return a copy preserving KV state and the current deepstack tensor."""
        copied = MobilintDeepStackCache(
            list(self.mxq_models),
            per_model_batch=self.k_per_model,
            num_deepstack_layers=self.num_deepstack_layers,
            hidden_size=self.hidden_size,
        )
        if len(copied.layers) < self.batch_size:
            copied.ensure_batch_size(self.batch_size)
        for i in range(len(self.layers)):
            copied.layers[i] = self.layers[i].copy()
        copied._deepstack_tensor = None if self._deepstack_tensor is None else self._deepstack_tensor.clone()
        return copied


class MobilintEagle3Cache(Cache):
    """Mobilint cache for EAGLE-3 speculative decoding.

    This cache carries both base and draft MXQ cache states plus the mutable tree
    decoding state that upstream EAGLE-3 stores on the model instance.
    """

    def __init__(
        self,
        base_mxq_model: qbruntime.Model,
        draft_mxq_model: qbruntime.Model,
    ) -> None:
        self.base_mxq_model = base_mxq_model
        self.draft_mxq_model = draft_mxq_model
        self.base_layer = MobilintLayer(base_mxq_model, 0)
        self.draft_layer = MobilintLayer(draft_mxq_model, 0)
        self.layers = [self.base_layer]
        self.layer_classes = MobilintLayer
        self.num_hidden_layers = 1
        self.cache_processor = None
        self.accept_tokens: Optional[torch.LongTensor] = None
        self.tree_mask: Optional[torch.Tensor] = None
        self.retrieve_indices: Optional[torch.LongTensor] = None
        self.tree_position_ids: Optional[torch.LongTensor] = None
        self.pending_draft_tokens: Optional[torch.LongTensor] = None

    def get_seq_length(self, index: int = 0) -> int:
        del index
        return self.base_layer.get_seq_length()

    def get_base_seq_length(self) -> int:
        return self.base_layer.get_seq_length()

    def get_draft_seq_length(self) -> int:
        return self.draft_layer.get_seq_length()

    def set_seq_length(self, sequence_length: int, index: int = 0) -> None:
        del index
        self.base_layer.set_seq_length(sequence_length)

    def set_base_seq_length(self, sequence_length: int) -> None:
        self.base_layer.set_seq_length(sequence_length)

    def set_draft_seq_length(self, sequence_length: int) -> None:
        self.draft_layer.set_seq_length(sequence_length)

    def sync_draft_seq_length_to_base(self) -> None:
        """Align the draft cache length with the committed base cache length."""
        self.draft_layer.set_seq_length(self.get_base_seq_length())

    def update_cache_position(self, cache_position: torch.Tensor, index: int = 0) -> None:
        del index
        self.base_layer.update_cache_position(cache_position)

    def update_base_seen_tokens(self, num_new_seen_tokens: int) -> None:
        self.base_layer.update_seen_tokens(num_new_seen_tokens)

    def update_draft_seen_tokens(self, num_new_seen_tokens: int) -> None:
        self.draft_layer.update_seen_tokens(num_new_seen_tokens)

    def fake_prefill(self, sequence_length: int, index: int = 0) -> None:
        del index
        self.base_layer.fake_prefill(sequence_length)
        self.draft_layer.fake_prefill(sequence_length)
        self.clear_tree_state()

    def clear_tree_state(self) -> None:
        """Drop speculative decoding metadata while preserving KV cache state."""
        self.accept_tokens = None
        self.tree_mask = None
        self.retrieve_indices = None
        self.tree_position_ids = None
        self.pending_draft_tokens = None

    def reset(self) -> None:
        self.base_layer.reset()
        self.draft_layer.reset()
        self.clear_tree_state()

    def dump_cache_memory(self) -> None:
        self.base_layer.dump_cache_memory()
        self.draft_layer.dump_cache_memory()

    def load_cache_memory(self) -> None:
        self.base_layer.load_cache_memory()
        self.draft_layer.load_cache_memory()

    def dump_cache_memory_to(self, cache_dir: str, index: int = 0) -> None:
        del index
        base_dir = Path(cache_dir) / "base"
        draft_dir = Path(cache_dir) / "draft"
        base_dir.mkdir(parents=True, exist_ok=True)
        draft_dir.mkdir(parents=True, exist_ok=True)
        self.base_layer.dump_cache_memory_to(str(base_dir))
        self.draft_layer.dump_cache_memory_to(str(draft_dir))

    def load_cache_memory_from(self, cache_dir: str, index: int = 0) -> None:
        del index
        self.reset()
        self.base_layer.load_cache_memory_from(str(Path(cache_dir) / "base"))
        self.draft_layer.load_cache_memory_from(str(Path(cache_dir) / "draft"))

    def copy(self) -> "MobilintEagle3Cache":
        """Return a copy preserving committed KV cache state only.

        Speculative tree metadata is intentionally not copied because it is
        per-generation-call transient state and should be reconstructed by
        ``initialize_tree``.
        """
        copied = MobilintEagle3Cache(self.base_mxq_model, self.draft_mxq_model)
        copied.base_layer = self.base_layer.copy()
        copied.draft_layer = self.draft_layer.copy()
        copied.layers = [copied.base_layer]
        copied.clear_tree_state()
        return copied
