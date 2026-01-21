import math
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from transformers.modeling_utils import PreTrainedModel

from ..utils.cache_utils import MobilintCache
from .base_utils import PretrainedOnlyMixin
from .configuration_utils import MobilintConfigMixin


class MobilintModelMixin(PretrainedOnlyMixin, PreTrainedModel):
    def __init__(self, config: MobilintConfigMixin, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        if TYPE_CHECKING:
            self.config = config
    
    def launch(self):
        self.config.npu_backend.launch()
    
    def dispose(self):
        self.config.npu_backend.dispose()
    
    def get_cache_mxq_model(self):
        return self.config.npu_backend.mxq_model

    def llm_forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[MobilintCache],
        cache_position: torch.Tensor,
        chunk_size: int = 128,
    ):
        inputs_embeds_numpy = inputs_embeds.type(torch.float32).cpu().numpy()

        if inputs_embeds_numpy.ndim == 3:
            inputs_embeds_numpy = np.expand_dims(inputs_embeds_numpy, 1)  # (batch, 1, seqlen, hidden_size)

        assert chunk_size > 0, "chunk_size should be a positive number! chunk_size: %d" % chunk_size
        num_of_chunks = math.ceil(inputs_embeds_numpy.shape[2] / chunk_size)

        mxq_model = self.config.npu_backend.mxq_model
        
        for i in range(num_of_chunks):
            start_index = i * chunk_size
            end_index = min(start_index + chunk_size, inputs_embeds_numpy.shape[2])
            cache_size = (0 if past_key_values is None else past_key_values.get_seq_length())

            result = mxq_model.infer([inputs_embeds_numpy[:, :, start_index:end_index, :]], None, cache_size)
            assert result is not None, "mxq infer result is None!"
            logits_ndarray = result[0]

            if past_key_values is not None:
                past_key_values.update_cache_position(cache_position[start_index:end_index])

        logits = torch.tensor(logits_ndarray, dtype=torch.float32, device=self.device).squeeze(0)
        
        return logits