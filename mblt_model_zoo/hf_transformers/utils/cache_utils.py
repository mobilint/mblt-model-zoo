import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import qbruntime
import torch
from transformers.cache_utils import Cache, CacheLayerMixin


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
    def __init__(self, mxq_model: qbruntime.Model, batch_size: int = 1):
        self.mxq_model = mxq_model
        self.batch_size = max(1, batch_size)

        self.layers: list[MobilintLayer] = [
            MobilintLayer(self.mxq_model, cache_id) for cache_id in range(self.batch_size)
        ]
        self.layer_classes = MobilintLayer

        self.num_hidden_layers = 1
        self.cache_processor = None

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
        """Grow logical cache entries so batched generation can track each active row."""
        batch_size = max(1, int(batch_size))
        if batch_size <= self.batch_size:
            return
        for cache_id in range(self.batch_size, batch_size):
            self.layers.append(MobilintLayer(self.mxq_model, cache_id))
        self.batch_size = batch_size

    def copy(self):
        copied = MobilintCache(self.mxq_model, batch_size=self.batch_size)
        for i in range(self.batch_size):
            copied.layers[i] = self.layers[i].copy()
        return copied


class MobilintBeamCache(MobilintCache):
    """Mobilint beam cache tracked by token histories instead of KV snapshots.

    qbruntime owns one active KV cache. This class tracks the token history for
    each logical beam and the token history currently represented by the active
    qbruntime cache. Callers can compare a target beam history with the active
    history, skip the common prefix, and forward only the suffix with the proper
    cache position.
    """

    def __init__(self, mxq_model: qbruntime.Model, batch_size: int = 1) -> None:
        super().__init__(mxq_model=mxq_model, batch_size=batch_size)
        self._beam_token_histories: list[list[int]] = [[] for _ in range(self.batch_size)]
        self._active_token_history: list[int] = []
        self._active_source_index: int | None = None
        self._beam_seq_lengths: list[int] = [0 for _ in range(self.batch_size)]

    def reset(self) -> None:
        """Reset active qbruntime cache bookkeeping and clear beam token histories."""
        super().reset()
        self._beam_token_histories = [[] for _ in range(self.batch_size)]
        self._active_token_history = []
        self._active_source_index = None
        self._beam_seq_lengths = [0 for _ in range(self.batch_size)]

    def ensure_batch_size(self, batch_size: int) -> None:
        """Grow logical beam token storage for beam-expanded generation."""
        previous_batch_size = self.batch_size
        super().ensure_batch_size(batch_size)
        if self.batch_size <= previous_batch_size:
            return
        self._beam_token_histories.extend([[] for _ in range(self.batch_size - previous_batch_size)])
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
        old_seq_lengths = list(self._beam_seq_lengths)
        beam_indices = [int(index) for index in beam_idx.cpu().tolist()]
        self._beam_token_histories = [list(old_token_histories[index]) for index in beam_indices]
        self._beam_seq_lengths = [old_seq_lengths[index] for index in beam_indices]
        for beam_id, seq_length in enumerate(self._beam_seq_lengths):
            self.layers[beam_id].set_seq_length(seq_length)
        if trace_enabled:
            append_whisper_beam_debug_event(
                {
                    "event": "cache_reorder_after",
                    "beam_idx": beam_indices,
                    "beam_token_histories": [list(tokens) for tokens in self._beam_token_histories],
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
        copied = self.__class__(self.mxq_model, batch_size=self.batch_size)
        for i in range(self.batch_size):
            copied.layers[i] = self.layers[i].copy()
        copied._beam_token_histories = [list(tokens) for tokens in self._beam_token_histories]
        copied._active_token_history = list(self._active_token_history)
        copied._active_source_index = self._active_source_index
        copied._beam_seq_lengths = list(self._beam_seq_lengths)
        return copied


class MobilintWhisperCache(MobilintBeamCache):
    """Whisper cache using token-history beam replay."""

    def __init__(self, mxq_model: qbruntime.Model, batch_size: int = 1) -> None:
        super().__init__(mxq_model=mxq_model, batch_size=batch_size)
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
    """

    def __init__(
        self,
        mxq_model: qbruntime.Model,
        batch_size: int = 1,
        num_deepstack_layers: int = 0,
        hidden_size: int = 0,
    ) -> None:
        super().__init__(mxq_model=mxq_model, batch_size=batch_size)
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
            self.mxq_model,
            batch_size=self.batch_size,
            num_deepstack_layers=self.num_deepstack_layers,
            hidden_size=self.hidden_size,
        )
        for i in range(self.batch_size):
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
