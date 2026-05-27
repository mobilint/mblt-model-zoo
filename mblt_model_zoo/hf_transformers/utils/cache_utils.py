import copy
from pathlib import Path
from typing import Any, List, Optional, Union

import qbruntime
import torch
from transformers.cache_utils import Cache, CacheLayerMixin


class MobilintLayer(CacheLayerMixin):
    is_sliding = False

    def __init__(self, mxq_model: qbruntime.Model, cache_id: int = 0):
        self.mxq_model = mxq_model
        self.cache_id = cache_id
        self._seen_tokens = 0
        self.buffer: List[bytes] = []
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

    def copy(self):
        copied = MobilintCache(self.mxq_model, batch_size=self.batch_size)
        for i in range(self.batch_size):
            copied.layers[i] = self.layers[i].copy()
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
        self.accept_tokens = None
        self.tree_mask = None
        self.retrieve_indices = None
        self.tree_position_ids = None
        self.pending_draft_tokens = None

    def reset(self) -> None:
        self.base_layer.reset()
        self.draft_layer.reset()
        self.accept_tokens = None
        self.tree_mask = None
        self.retrieve_indices = None
        self.tree_position_ids = None
        self.pending_draft_tokens = None

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
        copied = MobilintEagle3Cache(self.base_mxq_model, self.draft_mxq_model)
        copied.base_layer = self.base_layer.copy()
        copied.draft_layer = self.draft_layer.copy()
        copied.layers = [copied.base_layer]
        copied.accept_tokens = None if self.accept_tokens is None else self.accept_tokens.clone()
        copied.tree_mask = None if self.tree_mask is None else self.tree_mask.clone()
        copied.retrieve_indices = None if self.retrieve_indices is None else self.retrieve_indices.clone()
        copied.tree_position_ids = None if self.tree_position_ids is None else self.tree_position_ids.clone()
        copied.pending_draft_tokens = (
            None if self.pending_draft_tokens is None else self.pending_draft_tokens.clone()
        )
        return copied
