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

    def reset(self) -> None:
        self._seen_tokens = 0
        self.buffer = []

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError("reorder_cache is not implemented")

    def update_cache_position(self, cache_position: torch.Tensor):
        self._seen_tokens += cache_position.numel()

    def update_seen_tokens(self, num_new_seen_tokens: int):
        self._seen_tokens += num_new_seen_tokens

    def dump_cache_memory(self):
        self.buffer = self.mxq_model.dump_cache_memory(self.cache_id)

    def load_cache_memory(self):
        if self.get_seq_length() > 0:
            self.mxq_model.reset_cache_memory()
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
        self._seen_tokens = seq_length
        self.mxq_model.load_cache_memory_from(cache_dir, self.cache_id)

    def copy(self) -> "MobilintLayer":
        copied = MobilintLayer(self.mxq_model, self.cache_id)
        copied._seen_tokens = self._seen_tokens
        copied.buffer = copy.deepcopy(self.buffer)
        return copied


class MobilintCache(Cache):
    def __init__(self, mxq_model: qbruntime.Model, batch_size: int = 1):
        self.mxq_model = mxq_model
        self.batch_size = max(1, batch_size)

        self.layers: list[MobilintLayer] = [MobilintLayer(self.mxq_model, cache_id) for cache_id in range(self.batch_size)]
        self.layer_classes = MobilintLayer

        self.num_hidden_layers = 1
        self.cache_processor = None

    def get_seq_length(self, index: int = 0) -> int:
        return self.layers[index].get_seq_length()

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
