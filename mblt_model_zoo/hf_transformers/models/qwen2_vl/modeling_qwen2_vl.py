import inspect
from functools import lru_cache
from typing import Any, Union, cast

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import (
    AutoModel,
    AutoModelForImageTextToText,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
)
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, can_return_tuple, logging

from ...utils.base_utils import PretrainedOnlyMixin
from ...utils.cache_utils import MobilintCache
from ...utils.generation_utils import (
    MobilintGenerationMixin,
    build_loss_kwargs_dynamic,
    mirror_output_fields,
    pop_loss_only_kwargs,
    upstream_positional_params,
    with_mobilint_generation_signature,
)
from ...utils.modeling_utils import MobilintModelMixin
from .configuration_qwen2_vl import (
    MobilintQwen2VLConfig,
    MobilintQwen2VLTextConfig,
    MobilintQwen2VLVisionConfig,
)

logger = logging.get_logger(__name__)


@lru_cache(maxsize=1)
def _upstream_qwen2_vl_uses_structured_vision_outputs() -> bool:
    """Check whether the installed Transformers expects ``visual()`` to return a model output.

    Returns:
        ``True`` when the installed upstream ``Qwen2VLModel.get_image_features`` reads
        ``vision_outputs.pooler_output``. ``False`` for older releases that expect ``visual()`` to
        return a raw tensor.
    """
    get_image_features = inspect.unwrap(Qwen2VLModel.get_image_features)
    code = getattr(get_image_features, "__code__", None)
    if code is not None:
        return "pooler_output" in code.co_names

    try:
        return "pooler_output" in inspect.getsource(get_image_features)
    except OSError:
        return True


class MobilintQwen2VLPreTrainedModel(PreTrainedModel):
    config: MobilintQwen2VLConfig
    base_model_prefix = "model"
    input_modalities = ("image", "video", "text")


class MobilintQwen2VisionTransformerPretrainedModel(MobilintModelMixin, MobilintQwen2VLPreTrainedModel):
    config: MobilintQwen2VLVisionConfig
    input_modalities = ("image", "video")

    @property
    def dtype(self) -> torch.dtype:
        """Expose the MXQ vision input dtype expected by upstream Qwen2-VL helpers."""
        return torch.float32

    @property
    def spatial_merge_size(self) -> int:
        """Expose the merge factor expected by upstream Qwen2-VL helpers."""
        return int(self.config.spatial_merge_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BaseModelOutputWithPooling]:
        """Run the compiled vision encoder and adapt to HF's vision output contract.

        Args:
            hidden_states: Processor-produced image or video patch tensor.
            grid_thw: Temporal, height, and width grid metadata for each image or video.
            **kwargs: Additional Transformers kwargs. ``return_dict`` falls back to
                ``self.config.return_dict`` when the installed upstream expects structured outputs.

        Returns:
            By default, a value matching the installed upstream Qwen2-VL contract:
            ``BaseModelOutputWithPooling`` on newer Transformers and a raw tensor on older ones.
            The Mobilint backend does not expose per-patch hidden states or attentions, so those
            structured fields are returned as ``None`` when present.
        """
        return_dict = kwargs.pop("return_dict", None)
        if return_dict is None and _upstream_qwen2_vl_uses_structured_vision_outputs():
            return_dict = self.config.return_dict
        del kwargs
        merged_hidden_states = self._encode_images(hidden_states, grid_thw)
        structured_outputs = BaseModelOutputWithPooling(
            last_hidden_state=None,
            pooler_output=merged_hidden_states,
            hidden_states=None,
            attentions=None,
        )
        if return_dict is True:
            return structured_outputs
        if _upstream_qwen2_vl_uses_structured_vision_outputs():
            if return_dict is False:
                return structured_outputs.to_tuple()
            return structured_outputs
        return merged_hidden_states

    def _preprocess_image_tokens(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Convert one Qwen2-VL image patch sequence to the MXQ vision input layout."""
        gt, gh, gw = grid_thw.tolist()

        c = 3
        pt = 2
        mh = 2
        mw = 2
        gh2 = gh // 2
        gw2 = gw // 2
        ph = pw = int((hidden_states.shape[-1] // (pt * c)) ** 0.5)

        expected_tokens = gt * gh2 * gw2 * mh * mw
        expected_hidden = c * pt * ph * pw
        if hidden_states.shape[0] != expected_tokens:
            raise ValueError(
                f"Unexpected pixel token count for Qwen2-VL vision input: {hidden_states.shape[0]} vs {expected_tokens}"
            )
        if hidden_states.shape[1] != expected_hidden:
            raise ValueError(
                f"Unexpected pixel hidden size for Qwen2-VL vision input: {hidden_states.shape[1]} vs {expected_hidden}"
            )

        hidden_states = hidden_states.view(gt, gh2, gw2, mh, mw, c, pt, ph, pw)
        hidden_states = hidden_states.permute(0, 1, 2, 7, 3, 4, 8, 6, 5).contiguous()
        return hidden_states.view(gt, gh2 * gw2 * ph, mh * mw * pw, pt * c).squeeze(0)

    def _split_hidden_states_by_grid(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Split flattened processor image tokens according to `grid_thw` rows."""
        offset = 0
        chunks: list[torch.Tensor] = []
        for grid in grid_thw:
            gt, gh, gw = grid.tolist()
            token_count = int(gt * gh * gw)
            chunks.append(hidden_states[offset : offset + token_count])
            offset += token_count
        if offset != int(hidden_states.shape[0]):
            raise ValueError(f"Unexpected total Qwen2-VL pixel token count: {hidden_states.shape[0]} vs {offset}")
        return chunks

    def _flatten_encoder_output(
        self,
        output: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        """Normalize Qwen2-VL MXQ vision output to `(total_image_tokens, hidden_size)`."""
        if output.ndim >= 3 and int(output.shape[0]) == batch_size:
            return output.reshape(-1, int(output.shape[-1]))
        if output.ndim >= 3 and int(output.shape[0]) == 1:
            return output.squeeze(0).reshape(-1, int(output.shape[-1]))
        if output.ndim == 2:
            return output
        raise ValueError(f"Unexpected Qwen2-VL vision output shape: {tuple(output.shape)}")

    def _encode_images(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Run Qwen2-VL vision encoding with core-mode-specific batch handling."""
        chunks = self._split_hidden_states_by_grid(hidden_states, grid_thw)
        mxq_inputs = [self._preprocess_image_tokens(chunk, grid) for chunk, grid in zip(chunks, grid_thw)]
        npu_backend = getattr(self, "npu_backend", None)
        core_mode = getattr(npu_backend, "core_mode", getattr(self.config, "core_mode", "single"))
        if core_mode == "multi" and len(mxq_inputs) > 1:
            batched_inputs = torch.stack(mxq_inputs, dim=0)
            return self._flatten_encoder_output(self.mxq_forward(batched_inputs), batch_size=len(mxq_inputs))

        outputs = [self._flatten_encoder_output(self.mxq_forward(mxq_input), batch_size=1) for mxq_input in mxq_inputs]
        return torch.cat(outputs, dim=0)


class MobilintQwen2VLTextModel(MobilintModelMixin, MobilintGenerationMixin, MobilintQwen2VLPreTrainedModel):
    config: MobilintQwen2VLTextConfig
    input_modalities = ("text",)

    def __init__(self, config: MobilintQwen2VLTextConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: Union[torch.LongTensor, None] = None,
        attention_mask: Union[torch.Tensor, None] = None,
        position_ids: Union[torch.LongTensor, None] = None,
        past_key_values: Union[MobilintCache, None] = None,
        inputs_embeds: Union[torch.FloatTensor, None] = None,
        use_cache: Union[bool, None] = None,
        output_attentions: Union[bool, None] = None,
        output_hidden_states: Union[bool, None] = None,
        return_dict: Union[bool, None] = None,
        cache_position: Union[torch.LongTensor, None] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        npu_prefill_chunk_size: Union[int, None] = None,
        count_npu_time: bool = False,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = self._get_cache("", 0, 0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        assert inputs_embeds is not None

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = cast(
                torch.LongTensor,
                torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device),
            )

        if output_attentions:
            logger.warning("output_attentions is not supported.")

        if output_hidden_states:
            logger.warning("output_hidden_states is not supported.")

        logits = self.llm_forward(
            inputs_embeds,
            past_key_values,
            cache_position,
            npu_prefill_chunk_size,
            count_npu_time=count_npu_time,
            logits_to_keep=logits_to_keep,
        )

        if not return_dict:
            return tuple(v for v in [logits, past_key_values, None, None] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=cast(torch.FloatTensor, logits),
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )


class MobilintQwen2VLModel(PretrainedOnlyMixin, MobilintQwen2VLPreTrainedModel, Qwen2VLModel):
    def __init__(self, config: MobilintQwen2VLConfig, *args, **kwargs):
        MobilintQwen2VLPreTrainedModel.__init__(self, config, *args, **kwargs)
        self.visual = MobilintQwen2VisionTransformerPretrainedModel._from_config(
            config.vision_config, _internal_call=True
        )
        self.language_model = MobilintQwen2VLTextModel._from_config(
            config.text_config,
            _internal_call=True,
        )
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()


class MobilintQwen2VLForConditionalGeneration(
    PretrainedOnlyMixin,
    MobilintQwen2VLPreTrainedModel,
    MobilintGenerationMixin,
    Qwen2VLForConditionalGeneration,
):
    def __init__(self, config: MobilintQwen2VLConfig, *args, **kwargs):
        PretrainedOnlyMixin.__init__(self, config, *args, **kwargs)

        self.model = MobilintQwen2VLModel(config, _internal_call=True)
        # lm_head is done in self.model
        # So we just replace self.lm_head with identity module
        self.lm_head = nn.Identity()

    def get_cache_mxq_model(self):
        return self.model.language_model.get_mxq_model()

    @with_mobilint_generation_signature(
        Qwen2VLForConditionalGeneration.prepare_inputs_for_generation,
        "count_npu_time",
        "npu_prefill_chunk_size",
    )
    def prepare_inputs_for_generation(
        self,
        *args: object,
        count_npu_time: bool = False,
        npu_prefill_chunk_size: int | None = None,
        **kwargs: object,
    ):
        """Prepare generation inputs while preserving Mobilint timing kwargs.

        Args:
            *args: Positional arguments forwarded to the upstream Qwen2-VL generation helper.
            count_npu_time: Whether Mobilint decoder NPU time should be accumulated.
            npu_prefill_chunk_size: Optional prefill chunk size forwarded to Mobilint generation.
            **kwargs: Keyword arguments forwarded to the upstream Qwen2-VL generation helper.

        Returns:
            Model inputs for a generation step.
        """
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        model_inputs["count_npu_time"] = count_npu_time
        if npu_prefill_chunk_size is not None:
            model_inputs["npu_prefill_chunk_size"] = npu_prefill_chunk_size
        return model_inputs

    @with_mobilint_generation_signature(Qwen2VLForConditionalGeneration.forward, "count_npu_time")
    @can_return_tuple
    def forward(
        self,
        *args: Any,
        count_npu_time: bool = False,
        **kwargs: Any,
    ) -> Union[tuple, Qwen2VLCausalLMOutputWithPast]:
        """Route ``logits_to_keep`` to the Mobilint text decoder.

        Upstream ``Qwen2VLForConditionalGeneration.forward`` extracts ``logits_to_keep``
        as a named argument and performs its own final slice on the text model output,
        which bypasses the Mobilint decoder's position selection. To keep that decoder
        in charge of picking positions, we pop ``logits_to_keep`` here and thread it
        into ``self.model`` via kwargs (upstream ``Qwen2VLModel.forward`` forwards its
        own ``**kwargs`` to the text model). All other arguments follow the upstream
        signature by way of ``@with_mobilint_generation_signature``, so upstream
        additions such as ``position_ids`` continue to pass through unchanged.

        Tuple mode: ``@can_return_tuple`` strips ``return_dict`` from kwargs before
        the wrapper body runs (so ``self.model`` never returns a tuple) and converts
        the assembled ``Qwen2VLCausalLMOutputWithPast`` back to a tuple when
        ``return_dict=False`` was requested — matching the upstream forward's
        contract.

        Dynamic adaptation:
            * Loss kwargs are built via :func:`build_loss_kwargs_dynamic`, so
              upstream additions like ``num_items_in_batch`` / ``shift_labels``
              flow through when the loss function accepts them.
            * The returned ``Qwen2VLCausalLMOutputWithPast`` is assembled by
              :func:`mirror_output_fields`, so new output fields (e.g. a future
              ``image_hidden_states``) are mirrored from the upstream model
              output automatically instead of requiring wrapper edits.

        Performance: the default ``logits_to_keep=0`` (keep-all) matches HF but on
        last-only MXQ triggers a size-1 infer per input token. ``.generate()`` is
        safe (HF passes ``logits_to_keep=1``); manual ``.forward()`` callers doing
        perplexity eval / logit collection inherit this cost on last-only builds.
        """
        positional_params = upstream_positional_params(Qwen2VLForConditionalGeneration.forward)
        if len(args) > len(positional_params):
            raise TypeError(
                f"forward() takes at most {len(positional_params)} positional arguments "
                f"but {len(args)} were given"
            )
        for name, value in zip(positional_params, args):
            if name in kwargs:
                raise TypeError(f"forward() got multiple values for argument {name!r}")
            kwargs[name] = value

        labels = kwargs.pop("labels", None)
        logits_to_keep = kwargs.pop("logits_to_keep", 0)
        # Loss-only kwargs (``num_items_in_batch``, ``shift_labels``) must be
        # stripped BEFORE ``self.model`` is called: upstream ``Qwen2VLModel``
        # forwards its own ``**kwargs`` to ``self.language_model``, and
        # ``MobilintQwen2VLTextModel.forward`` declares no ``**kwargs`` sink,
        # so leaking these into that call would raise ``TypeError``.
        loss_only_kwargs = pop_loss_only_kwargs(kwargs)

        outputs = self.model(
            logits_to_keep=logits_to_keep,
            count_npu_time=count_npu_time,
            **kwargs,
        )

        # The Mobilint text decoder already returns logits sliced to the requested
        # positions and ``self.lm_head`` is ``nn.Identity``, so skip the upstream
        # ``hidden_states[:, slice_indices, :]`` step.
        logits = cast(torch.FloatTensor, self.lm_head(outputs.last_hidden_state))

        loss = None
        if labels is not None:
            loss = self.loss_function(
                **build_loss_kwargs_dynamic(
                    self.loss_function,
                    logits=logits,
                    labels=labels,
                    vocab_size=self.config.text_config.vocab_size,
                    upstream_kwargs=loss_only_kwargs,
                )
            )

        return mirror_output_fields(
            Qwen2VLCausalLMOutputWithPast,
            outputs,
            loss=loss,
            logits=logits,
        )


AutoModel.register(MobilintQwen2VLConfig, MobilintQwen2VLForConditionalGeneration)
AutoModelForImageTextToText.register(MobilintQwen2VLConfig, MobilintQwen2VLForConditionalGeneration)
