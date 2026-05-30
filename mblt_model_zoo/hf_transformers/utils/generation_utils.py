import inspect
from abc import ABC
from functools import wraps
from typing import Any, Callable, Dict, Optional, cast

import qbruntime
import torch
import torch.nn as nn
from transformers import GenerationConfig, GenerationMixin, PreTrainedModel
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

from ..utils.cache_utils import MobilintCache, MobilintEagle3Cache
from ..utils.modeling_utils import MobilintModelMixin

logger = logging.get_logger(__name__)

_EAGLE3_GENERATE_IGNORED_ARGS_MSG = {
    "attention_mask": "attention_mask is not supported and will be ignored.",
    "min_new_tokens": "min_new_tokens is not supported and will be ignored.",
    "pad_token_id": "pad_token_id is not supported and will be ignored.",
    "prefill_chunk_size": "prefill_chunk_size is not supported by EAGLE-3 generate and will be ignored.",
    "cache_position": "cache_position is not supported and will be ignored.",
}


def llm_eagle3_forward(
    model: Any,
    *,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[MobilintEagle3Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    count_npu_time: bool = False,
    output_hidden_states: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    **kwargs: Any,
) -> CausalLMOutputWithPast:
    """Run the shared EAGLE-3 forward path for causal LM wrappers."""
    if output_attentions:
        logger.warning("output_attentions is not supported.")
    if output_hidden_states:
        logger.warning("output_hidden_states is not supported.")
    if cache_position is not None:
        logger.warning("cache_position is not supported.")
    if kwargs:
        logger.warning("Unsupported forward kwargs are ignored: %s" % ", ".join(sorted(kwargs)))
    if use_cache is False:
        raise ValueError("EAGLE-3 models require use_cache=True.")
    if past_key_values is None:
        past_key_values = model._get_cache("mobilint-eagle3", 1, 0)
    if not isinstance(past_key_values, MobilintEagle3Cache):
        raise TypeError("past_key_values must be MobilintEagle3Cache for EAGLE-3 models.")

    outputs, logits = model.eagle3_base_model(
        input_ids=input_ids,
        cache=past_key_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        output_orig=True,
        requires_all_features_logits=False,
        count_npu_time=count_npu_time,
    )
    loss = None
    if labels is not None:
        loss = model.loss_function(logits=logits, labels=labels, vocab_size=model.config.vocab_size)
    return CausalLMOutputWithPast(
        loss=loss,
        logits=cast(torch.FloatTensor, logits),
        past_key_values=past_key_values,
        hidden_states=None if outputs is None else tuple(outputs["hidden_states"]),
        attentions=None,
    )


def with_mobilint_generation_signature(wrapped: Callable, *extra_keyword_names: str) -> Callable:
    """Preserve an upstream generation hook signature while adding Mobilint kwargs.

    Args:
        wrapped: Upstream callable whose public signature should be preserved.
        *extra_keyword_names: Keyword-only parameters to append before ``**kwargs``.

    Returns:
        Decorator that applies ``functools.wraps`` and exposes an augmented signature.
    """

    def decorator(wrapper: Callable) -> Callable:
        wrapper_signature = inspect.signature(wrapper)
        wrapper = wraps(wrapped)(wrapper)
        signature = inspect.signature(wrapped)
        parameters = list(signature.parameters.values())
        existing = set(signature.parameters)
        insert_at = next(
            (idx for idx, parameter in enumerate(parameters) if parameter.kind == inspect.Parameter.VAR_KEYWORD),
            len(parameters),
        )
        for name in extra_keyword_names:
            if name in existing:
                continue
            extra_parameter = wrapper_signature.parameters.get(name)
            default = False if extra_parameter is None else extra_parameter.default
            annotation = bool if extra_parameter is None else extra_parameter.annotation
            parameters.insert(
                insert_at,
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation),
            )
            insert_at += 1
        wrapper.__signature__ = signature.replace(parameters=parameters)
        return wrapper

    return decorator


class MobilintGenerationMixin(ABC, GenerationMixin):
    @with_mobilint_generation_signature(
        GenerationMixin.prepare_inputs_for_generation,
        "count_npu_time",
        "prefill_chunk_size",
    )
    def prepare_inputs_for_generation(
        self,
        *args: Any,
        count_npu_time: bool = False,
        prefill_chunk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare generation inputs while preserving Mobilint benchmark kwargs.

        Args:
            *args: Positional arguments forwarded to the upstream generation helper.
            count_npu_time: Whether Mobilint decoder NPU time should be accumulated.
            prefill_chunk_size: Optional Mobilint prefill chunk size.
            **kwargs: Keyword arguments forwarded to the upstream generation helper.

        Returns:
            Prepared model inputs for the next generation step.
        """
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        model_inputs["count_npu_time"] = count_npu_time
        model_inputs["prefill_chunk_size"] = prefill_chunk_size
        return model_inputs

    def get_cache_mxq_model(self) -> qbruntime.Model:
        if isinstance(self, MobilintModelMixin):
            return self.get_mxq_model()
        else:
            raise TypeError("mxq_model for cache not found! Class: %s" % self.__class__.__name__)

    # Function arguments changed for transformers>=4.56.0
    # args contain device and model_kwargs in transformers<4.56.0
    # args contain only model_kwargs in transformers>=4.56.0
    def _get_cache(self, cache_implementation: str, batch_size: int, max_cache_len: int, *args) -> MobilintCache:
        configured_batch_size = max(1, getattr(self.config, "max_batch_size", 1))
        if not hasattr(self, "_cache"):
            self._cache = MobilintCache(self.get_cache_mxq_model(), batch_size=configured_batch_size)
        elif getattr(self._cache, "batch_size", 1) != configured_batch_size:
            self._cache = MobilintCache(self.get_cache_mxq_model(), batch_size=configured_batch_size)
        else:
            self._cache.reset()

        return self._cache

    # Function arguments changed for transformers>=4.56.0
    # args contain device in transformers<4.56.0
    # args empty in transformers>=4.56.0
    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        assistant_model: "PreTrainedModel",
        batch_size: int,
        max_cache_length: int,
        *args,
    ) -> bool:
        super()._prepare_cache_for_generation(
            generation_config,
            model_kwargs,
            assistant_model,  # type: ignore
            batch_size,
            max_cache_length,
            *args,
        )

        cache_name = "past_key_values"

        if model_kwargs.get(cache_name, None) is None:
            return False
        elif isinstance(model_kwargs[cache_name], MobilintCache):
            return True
        else:
            model_kwargs[cache_name] = self._get_cache("mobilint", batch_size, max_cache_length, *args, model_kwargs)
            return True


class MobilintEagle3GenerationMixin(ABC, GenerationMixin):
    """Custom generation mixin for Mobilint EAGLE-3 models."""

    @property
    def eagle3_model(self) -> Any:
        """Return the object that owns EAGLE-3 child modules."""
        nested_model = getattr(self, "_modules", {}).get("model")
        if nested_model is not None and "eagle3_base_model" not in getattr(self, "_modules", {}):
            return nested_model
        return self

    def _resolve_eagle3_child(self, attribute_name: str, *fallback_names: str) -> Any:
        """Resolve a direct or nested EAGLE-3 child module."""
        candidate_names = (attribute_name, *fallback_names)
        direct_modules = getattr(self, "_modules", {})
        for candidate_name in candidate_names:
            if candidate_name in direct_modules:
                return direct_modules[candidate_name]

        for candidate_name in candidate_names:
            direct_child = self.__dict__.get(candidate_name, None)
            if direct_child is not None:
                return direct_child

        nested_model = getattr(self, "_modules", {}).get("model")
        if nested_model is not None:
            nested_modules = getattr(nested_model, "_modules", {})
            for candidate_name in candidate_names:
                if candidate_name in nested_modules:
                    return nested_modules[candidate_name]
                nested_child = getattr(nested_model, candidate_name, None)
                if nested_child is not None:
                    return nested_child

        raise AttributeError(f"EAGLE-3 child module {attribute_name!r} not found")

    @property
    def eagle3_base_model(self) -> Any:
        """Return the base LLM MXQ child model."""
        return self._resolve_eagle3_child("eagle3_base_model", "base_model")

    @property
    def eagle3_draft_model(self) -> Any:
        """Return the draft LLM MXQ child model."""
        return self._resolve_eagle3_child("eagle3_draft_model", "draft_model")

    @property
    def eagle3_fc_projector(self) -> Any:
        """Return the optional FC projector MXQ child model."""
        return self._resolve_eagle3_child("eagle3_fc_projector", "fc_projector")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[str], *model_args: object, **kwargs: object):
        """Load an EAGLE-3 model and inject role-specific embedding overrides."""
        legacy_embedding_weight = kwargs.pop("embedding_weight", None)
        base_embedding_weight = kwargs.pop("base_embedding_weight", None)
        draft_embedding_weight = kwargs.pop("draft_embedding_weight", None)
        if legacy_embedding_weight is not None:
            raise ValueError(
                "`embedding_weight` is not supported for EAGLE-3 models. "
                "Use `base_embedding_weight` and/or `draft_embedding_weight`."
            )
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if base_embedding_weight is not None:
            model._inject_embedding_override(model.eagle3_base_model.embed_tokens, base_embedding_weight, "base")
        if draft_embedding_weight is not None:
            model._inject_embedding_override(model.eagle3_draft_model.embed_tokens, draft_embedding_weight, "draft")
        return model

    @staticmethod
    def _inject_embedding_override(embedding: nn.Embedding, path: str, role: str) -> None:
        """Load and validate a role-specific embedding override."""
        from ..utils.eagle3.eagle3_utils import load_embedding_override

        weight = load_embedding_override(path)
        if weight.ndim != 2:
            raise ValueError(f"{role} embedding override must be rank 2, got shape {tuple(weight.shape)}")
        expected_shape = tuple(embedding.weight.shape)
        if tuple(weight.shape) != expected_shape:
            raise ValueError(
                f"{role} embedding override shape mismatch: expected {expected_shape}, got {tuple(weight.shape)}"
            )
        with torch.no_grad():
            embedding.weight.data = weight.to(device=embedding.weight.device, dtype=embedding.weight.dtype)

    def get_input_embeddings(self) -> nn.Module:
        """Return the base model input embedding layer."""
        return self.eagle3_base_model.get_input_embeddings()

    def get_cache_mxq_models(self) -> tuple[qbruntime.Model, qbruntime.Model]:
        """Return MXQ models used by the base and draft cache layers."""
        return self.eagle3_base_model.get_mxq_model(), self.eagle3_draft_model.get_mxq_model()

    def reset_npu_timing(self) -> None:
        """Reset aggregate NPU timing counters for all EAGLE-3 child backends."""
        self._require_eagle3_components()
        for child in (self.eagle3_base_model, self.eagle3_draft_model, self.eagle3_fc_projector):
            child.reset_npu_timing()

    def get_npu_timing(self) -> dict[str, float | int]:
        """Return aggregate NPU timing counters across base, draft, and FC backends."""
        self._require_eagle3_components()
        aggregate: dict[str, float | int] = {
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "prefill_calls": 0,
            "decode_calls": 0,
        }
        for child in (self.eagle3_base_model, self.eagle3_draft_model, self.eagle3_fc_projector):
            timing = child.get_npu_timing()
            aggregate["prefill_time"] = float(aggregate["prefill_time"]) + float(timing.get("prefill_time", 0.0))
            aggregate["decode_time"] = float(aggregate["decode_time"]) + float(timing.get("decode_time", 0.0))
            aggregate["prefill_calls"] = int(aggregate["prefill_calls"]) + int(timing.get("prefill_calls", 0))
            aggregate["decode_calls"] = int(aggregate["decode_calls"]) + int(timing.get("decode_calls", 0))
        return aggregate

    def _require_eagle3_components(self) -> None:
        """Ensure base/draft/fc child backends are all mounted."""
        missing: list[str] = []
        for name in ("eagle3_base_model", "eagle3_draft_model", "eagle3_fc_projector"):
            try:
                _ = getattr(self, name)
            except AttributeError:
                missing.append(name)
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"EAGLE-3 requires all child backends (base/draft/fc). Missing: {joined}")

    def _get_cache(self, cache_implementation: str, batch_size: int, max_cache_len: int, *args) -> MobilintEagle3Cache:
        del cache_implementation, batch_size, max_cache_len, args
        if not hasattr(self, "_cache"):
            base_mxq_model, draft_mxq_model = self.get_cache_mxq_models()
            self._cache = MobilintEagle3Cache(base_mxq_model, draft_mxq_model)
        else:
            self._cache.reset()
        return self._cache

    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        assistant_model: "PreTrainedModel",
        batch_size: int,
        max_cache_length: int,
        *args,
    ) -> bool:
        del generation_config, assistant_model, batch_size, max_cache_length, args
        cache_name = "past_key_values"
        if model_kwargs.get(cache_name) is None:
            model_kwargs[cache_name] = self._get_cache("mobilint-eagle3", 1, 0)
        elif not isinstance(model_kwargs[cache_name], MobilintEagle3Cache):
            model_kwargs[cache_name] = self._get_cache("mobilint-eagle3", 1, 0)
        else:
            model_kwargs[cache_name].reset()
        return True

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[MobilintEagle3Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        count_npu_time: bool = False,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Run the shared EAGLE-3 base forward path."""
        return llm_eagle3_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            count_npu_time=count_npu_time,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            **kwargs,
        )

    def _validate_eagle3_generate_request(
        self,
        *,
        output_scores: bool,
        output_hidden_states: bool,
        output_attentions: bool,
        num_beams: int,
        assistant_model: Optional[PreTrainedModel],
        use_cache: Optional[bool],
        synced_gpus: Optional[bool],
        logits_processor_arg: Any,
        negative_prompt_ids: Optional[torch.Tensor],
        negative_prompt_attention_mask: Optional[torch.Tensor],
        kwargs: dict[str, Any],
    ) -> None:
        """Validate unsupported options for EAGLE-3 generation.

        Policy:
        - 일부 인자는 warning 후 무시한다 (``generate`` 본문에서 처리).
        - 동작 의미를 바꾸는 인자는 ``NotImplementedError``로 즉시 실패한다.
        """
        if output_scores:
            logger.warning("output_scores is not supported.")
        if output_hidden_states:
            logger.warning("output_hidden_states is not supported.")
        if output_attentions:
            logger.warning("output_attentions is not supported.")
        if num_beams != 1:
            raise NotImplementedError("EAGLE-3 models do not support beam search.")
        if assistant_model is not None:
            raise NotImplementedError("EAGLE-3 models do not support HF assistant_model mixing.")
        if use_cache is False:
            raise NotImplementedError("EAGLE-3 models require use_cache=True.")
        if synced_gpus not in (None, False):
            raise NotImplementedError("EAGLE-3 models do not support synced_gpus generation.")
        if logits_processor_arg not in (None, []):
            raise NotImplementedError("EAGLE-3 models do not support custom logits_processor yet.")
        if negative_prompt_ids is not None or negative_prompt_attention_mask is not None:
            raise NotImplementedError("EAGLE-3 models do not support negative prompts.")
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise NotImplementedError(f"Unsupported generate kwargs for EAGLE-3 models: {unsupported}")

    def _resolve_eagle3_generation_config(
        self,
        generation_config: Optional[GenerationConfig],
        *,
        max_new_tokens: Optional[int],
        do_sample: Optional[bool],
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
    ) -> tuple[GenerationConfig, int, Optional[float], Optional[float], int]:
        """Resolve generation config values used by the EAGLE-3 loop."""
        generation_config = self.generation_config if generation_config is None else generation_config
        resolved_max_new_tokens = int(
            max_new_tokens if max_new_tokens is not None else generation_config.max_new_tokens
        )
        resolved_do_sample = bool(
            do_sample if do_sample is not None else getattr(generation_config, "do_sample", False)
        )
        resolved_temperature = (
            temperature if temperature is not None else getattr(generation_config, "temperature", None)
        )
        resolved_top_p = top_p if top_p is not None else getattr(generation_config, "top_p", None)
        raw_top_k = top_k if top_k is not None else getattr(generation_config, "top_k", 0)
        resolved_top_k = int(raw_top_k) if raw_top_k is not None else 0
        if not resolved_do_sample:
            resolved_temperature = 0.0
        num_assistant_tokens = int(getattr(generation_config, "num_assistant_tokens", 64))
        self.eagle3_draft_model.max_draft_tokens = max(1, num_assistant_tokens - 1)
        return generation_config, resolved_max_new_tokens, resolved_temperature, resolved_top_p, resolved_top_k

    def _prepare_eagle3_cache(self, past_key_values: Optional[MobilintEagle3Cache]) -> MobilintEagle3Cache:
        """Resolve and normalize the EAGLE-3 cache for a generation call."""
        self._require_eagle3_components()
        cache = past_key_values
        if cache is None:
            cache = self._get_cache("mobilint-eagle3", 1, 0)
            cache.reset()
        elif not isinstance(cache, MobilintEagle3Cache):
            raise TypeError("past_key_values must be MobilintEagle3Cache for EAGLE-3 models.")
        cache.clear_tree_state()
        cache.sync_draft_seq_length_to_base()
        return cache

    @staticmethod
    def _eagle3_stopping_scores_adapter(
        logits: torch.Tensor,
        *,
        generated: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Adapt EAGLE-3 logits to HF ``StoppingCriteriaList`` score contract.

        HF stopping criteria receives ``scores`` shaped as ``[batch, vocab]``.
        EAGLE-3 tree decoding can return various intermediate layouts, so this
        helper normalizes them to a stable 2D tensor.
        """
        if logits.ndim == 0:
            return logits.view(1, 1).to(dtype=torch.float32)
        if logits.ndim == 1:
            return logits.unsqueeze(0).to(dtype=torch.float32)
        if logits.ndim == 2:
            # Prefer last candidate distribution to match current-step semantics.
            return logits[-1:, :].to(dtype=torch.float32)
        # Fallback for unexpected higher-rank tensors.
        flattened = logits.reshape(-1, logits.shape[-1])
        if flattened.shape[0] == 0:
            return torch.zeros((generated.shape[0], 1), device=generated.device, dtype=torch.float32)
        return flattened[-1:, :].to(dtype=torch.float32)

    @torch.no_grad()
    @with_mobilint_generation_signature(
        GenerationMixin.generate,
        "count_npu_time",
        "prefill_chunk_size",
    )
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[MobilintEagle3Cache] = None,
        generation_config: Optional[GenerationConfig] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        streamer: Optional[Any] = None,
        return_dict_in_generate: bool = False,
        output_scores: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        num_beams: int = 1,
        assistant_model: Optional[PreTrainedModel] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        stopping_criteria: Optional[StoppingCriteriaList | list[Any]] = None,
        min_new_tokens: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int | list[int]] = None,
        count_npu_time: bool = False,
        prefill_chunk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor | GenerateDecoderOnlyOutput:
        """Generate tokens with the Mobilint EAGLE-3 decoding loop.

        Compatibility policy:
        - Ignored-with-warning: ``attention_mask``, ``min_new_tokens``,
          ``pad_token_id``, ``prefill_chunk_size``, ``cache_position``.
        - Hard error: beam search, ``assistant_model``, ``use_cache=False``,
          custom ``logits_processor``, negative prompts, unknown kwargs.
        """
        from ..utils.eagle3.decoding import (
            evaluate_posterior,
            initialize_tree,
            prepare_logits_processor,
            tree_decoding,
            update_inference_inputs,
        )

        if attention_mask is not None:
            logger.warning(_EAGLE3_GENERATE_IGNORED_ARGS_MSG["attention_mask"])
        if min_new_tokens is not None:
            logger.warning(_EAGLE3_GENERATE_IGNORED_ARGS_MSG["min_new_tokens"])
        if pad_token_id is not None:
            logger.warning(_EAGLE3_GENERATE_IGNORED_ARGS_MSG["pad_token_id"])
        if prefill_chunk_size is not None:
            logger.warning(_EAGLE3_GENERATE_IGNORED_ARGS_MSG["prefill_chunk_size"])
        if cache_position is not None:
            logger.warning(_EAGLE3_GENERATE_IGNORED_ARGS_MSG["cache_position"])

        logits_processor_arg = kwargs.pop("logits_processor", None)
        synced_gpus = kwargs.pop("synced_gpus", None)
        negative_prompt_ids = kwargs.pop("negative_prompt_ids", None)
        negative_prompt_attention_mask = kwargs.pop("negative_prompt_attention_mask", None)
        self._validate_eagle3_generate_request(
            output_scores=output_scores,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            num_beams=num_beams,
            assistant_model=assistant_model,
            use_cache=use_cache,
            synced_gpus=synced_gpus,
            logits_processor_arg=logits_processor_arg,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            kwargs=kwargs,
        )

        input_ids = input_ids if input_ids is not None else inputs
        if input_ids is None:
            raise ValueError("`generate` requires `input_ids` or `inputs`.")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise NotImplementedError("EAGLE-3 models only support batch size 1.")

        generation_config, max_tokens, resolved_temperature, resolved_top_p, resolved_top_k = (
            self._resolve_eagle3_generation_config(
                generation_config,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        )
        cache = self._prepare_eagle3_cache(past_key_values)
        logits_processor = prepare_logits_processor(
            temperature=0.0 if resolved_temperature is None else resolved_temperature,
            top_p=0.0 if resolved_top_p is None else resolved_top_p,
            top_k=resolved_top_k,
        )

        generated = input_ids.clone()
        eos_token_id = eos_token_id if eos_token_id is not None else generation_config.eos_token_id
        if stopping_criteria is None:
            stopping_criteria_list = StoppingCriteriaList()
        elif isinstance(stopping_criteria, StoppingCriteriaList):
            stopping_criteria_list = stopping_criteria
        else:
            stopping_criteria_list = StoppingCriteriaList(stopping_criteria)
        if streamer is not None:
            streamer.put(generated[0].detach().cpu())

        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _ = initialize_tree(
            generated,
            self,
            cache,
            logits_processor,
            remaining_tokens=max_tokens,
            count_npu_time=count_npu_time,
        )
        new_token_count = 0
        acceptance_steps = 0
        acceptance_tokens_sum = 0
        acceptance_ratio_sum = 0.0

        while new_token_count < max_tokens:
            remaining_tokens = max_tokens - new_token_count
            logits, hidden_state_new = tree_decoding(
                self,
                cache,
                draft_tokens.to(generated.device),
                generated,
                retrieve_indices,
                tree_position_ids,
                count_npu_time=count_npu_time,
            )
            padding = torch.full((1, 1), -1, dtype=torch.long, device=generated.device)
            padded_draft_tokens = torch.cat((draft_tokens.to(generated.device), padding), dim=1)
            candidates = padded_draft_tokens[0, retrieve_indices].contiguous()
            best_candidate, accepted_draft_count, sample_p, sampled_indices = evaluate_posterior(
                logits,
                candidates,
                logits_processor,
                retrieve_indices,
            )
            candidate_width = int(candidates.shape[-1]) if candidates.ndim >= 2 else 1
            candidate_draft_tokens = max(1, candidate_width - 1)
            accepted_tokens = max(0, int(accepted_draft_count))
            acceptance_steps += 1
            acceptance_tokens_sum += accepted_tokens
            acceptance_ratio_sum += float(accepted_tokens) / float(candidate_draft_tokens)
            prev_len = generated.shape[1]
            generated, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token_count, should_stop = (
                update_inference_inputs(
                    generated,
                    candidates,
                    best_candidate,
                    accepted_draft_count,
                    retrieve_indices,
                    logits_processor,
                    new_token_count,
                    self,
                    cache,
                    hidden_state_new,
                    sample_p,
                    sampled_indices,
                    remaining_tokens=remaining_tokens,
                    eos_token_id=eos_token_id,
                    count_npu_time=count_npu_time,
                )
            )
            if streamer is not None:
                for token_id in generated[0, prev_len:]:
                    streamer.put(token_id.unsqueeze(0))
            stopping_scores = self._eagle3_stopping_scores_adapter(logits, generated=generated)
            if stopping_criteria_list(generated, stopping_scores):
                break
            if should_stop:
                break

        if streamer is not None:
            streamer.end()
        acceptance_avg = (float(acceptance_tokens_sum) / float(acceptance_steps)) if acceptance_steps > 0 else 0.0
        acceptance_ratio = (acceptance_ratio_sum / float(acceptance_steps)) if acceptance_steps > 0 else 0.0
        self._last_eagle3_acceptance_stats = {
            "steps": int(acceptance_steps),
            "accepted_tokens_sum": int(acceptance_tokens_sum),
            "accepted_tokens_avg": float(acceptance_avg),
            "acceptance_ratio": float(acceptance_ratio),
        }
        cache.clear_tree_state()
        cache.sync_draft_seq_length_to_base()
        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(sequences=generated, past_key_values=cache)
        return generated
