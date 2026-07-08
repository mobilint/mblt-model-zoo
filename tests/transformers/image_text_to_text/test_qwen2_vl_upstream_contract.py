"""Upstream contract: ``Qwen2VLModel.forward`` must not slice its own output.

``MobilintQwen2VLForConditionalGeneration.forward`` pops ``logits_to_keep``
from kwargs, threads it into ``self.model`` (which forwards it via ``**kwargs``
to the Mobilint text decoder), and then SKIPS the upstream
``hidden_states[:, slice_indices, :]`` step because the Mobilint decoder
already returns logits selected to the requested positions.

That contract holds only as long as upstream ``Qwen2VLModel.forward`` itself
does not do its own position selection. If a future HF transformers release
moves the slice from ``Qwen2VLForConditionalGeneration.forward`` into
``Qwen2VLModel.forward``, our wrapper would silently return pre-sliced values
(KV would still be correct — only the logits would be wrong), and the existing
``_RecordingModel``-based tests could not detect the drift.

These tests pin that contract via source inspection plus a functional check
that ``logits_to_keep`` flows through unmodified to ``self.language_model``.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLModel,
    Qwen2VLModelOutputWithPast,
)

pytestmark = pytest.mark.upstream_contract


def _forward_source() -> str:
    return inspect.getsource(Qwen2VLModel.forward)


class TestQwen2VLModelForwardSourceContract:
    """Fail loudly if the upstream source acquires its own logits slicing."""

    def test_no_slice_indices_local(self) -> None:
        # ``slice_indices = slice(-logits_to_keep, None) ...`` is the exact
        # pattern used in ``Qwen2VLForConditionalGeneration.forward``. It must
        # not migrate into ``Qwen2VLModel.forward``.
        assert "slice_indices" not in _forward_source()

    def test_no_hidden_states_bracket_slice(self) -> None:
        source = _forward_source()
        # Explicit ``hidden_states[:, ..., :]`` / ``last_hidden_state[:, ..., :]``
        # position selection would bypass the Mobilint decoder.
        assert not re.search(r"hidden_states\s*\[\s*:\s*,", source)
        assert not re.search(r"last_hidden_state\s*\[\s*:\s*,", source)

    def test_no_gather_along_seq_dim(self) -> None:
        # A ``.gather(1, indices)`` on the decoder output would be another way
        # to implement position selection at the model level.
        assert not re.search(r"\.gather\(\s*1\s*,", _forward_source())

    def test_logits_to_keep_not_a_named_parameter(self) -> None:
        # If upstream promotes ``logits_to_keep`` to a named parameter on
        # ``Qwen2VLModel.forward``, it stops flowing through ``**kwargs`` to
        # ``self.language_model`` and is very likely paired with local slicing.
        params = inspect.signature(Qwen2VLModel.forward).parameters
        assert "logits_to_keep" not in params

    def test_kwargs_are_forwarded_to_language_model(self) -> None:
        # Our wrapper depends on ``**kwargs`` reaching ``self.language_model``.
        source = _forward_source()
        # Match ``self.language_model(...)`` and ensure ``**kwargs`` appears in
        # the same call.
        match = re.search(r"self\.language_model\((?P<args>.*?)\)", source, re.DOTALL)
        assert match is not None, "expected a self.language_model(...) call in upstream forward"
        assert "**kwargs" in match.group("args")


# ---------------------------------------------------------------------------
# Functional check: logits_to_keep reaches self.language_model unmodified.
# ---------------------------------------------------------------------------


class _RecordingLanguageModel:
    """Stand-in that captures the kwargs forwarded by ``Qwen2VLModel.forward``."""

    def __init__(self, vocab_size: int = 5, hidden_size: int = 4) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.received: dict = {}

    def __call__(self, **kwargs):
        self.received = kwargs
        inputs_embeds = kwargs.get("inputs_embeds")
        assert inputs_embeds is not None, "test setup must pass inputs_embeds"
        seq_len = inputs_embeds.shape[1]
        return SimpleNamespace(
            last_hidden_state=torch.zeros(1, seq_len, self.hidden_size),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


def _make_bare_qwen2_vl_model() -> Qwen2VLModel:
    model = Qwen2VLModel.__new__(Qwen2VLModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=True,
        return_dict=True,
    )
    model.language_model = _RecordingLanguageModel()
    model.rope_deltas = torch.zeros(1, dtype=torch.long)
    return model


class TestQwen2VLModelForwardKwargsPassthrough:
    def test_logits_to_keep_reaches_language_model_unmodified(self) -> None:
        model = _make_bare_qwen2_vl_model()
        seq_len = 6
        inputs_embeds = torch.zeros(1, seq_len, model.language_model.hidden_size)
        # Provide ``position_ids`` explicitly so upstream skips the rope-index
        # computation branch (which requires ``get_rope_index`` internals).
        position_ids = torch.zeros(3, 1, seq_len, dtype=torch.long)
        sentinel = torch.tensor([1, 3, 5])

        model.forward(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            logits_to_keep=sentinel,
        )

        assert "logits_to_keep" in model.language_model.received
        forwarded = model.language_model.received["logits_to_keep"]
        assert torch.is_tensor(forwarded)
        assert torch.equal(forwarded, sentinel)

    def test_int_logits_to_keep_reaches_language_model_unmodified(self) -> None:
        model = _make_bare_qwen2_vl_model()
        seq_len = 4
        inputs_embeds = torch.zeros(1, seq_len, model.language_model.hidden_size)
        position_ids = torch.zeros(3, 1, seq_len, dtype=torch.long)

        model.forward(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            logits_to_keep=2,
        )

        assert model.language_model.received.get("logits_to_keep") == 2


# ---------------------------------------------------------------------------
# Safety net for dynamic output/loss adaptation.
#
# The Mobilint wrapper builds its returned ``Qwen2VLCausalLMOutputWithPast``
# via ``mirror_output_fields`` and its loss kwargs via
# ``build_loss_kwargs_dynamic``. Those helpers pick up new upstream additions
# automatically, but they can only mirror fields present on the source model
# output and can only supply loss kwargs whose value source lives in
# ``LOSS_KWARG_SUPPLY_NAMES``. The two checks below catch the drift cases the
# dynamic path cannot cover on its own.
# ---------------------------------------------------------------------------


class TestQwen2VLCausalLMOutputFieldCoverage:
    """Every CausalLM field must be reachable via override or model-output mirror."""

    def test_uncovered_causal_lm_fields_have_defaults(self) -> None:
        overrides = {"loss", "logits"}
        model_output_fields = {field.name for field in dataclasses.fields(Qwen2VLModelOutputWithPast)}
        uncovered = [
            field
            for field in dataclasses.fields(Qwen2VLCausalLMOutputWithPast)
            if field.name not in overrides and field.name not in model_output_fields
        ]
        missing_default = [
            field.name
            for field in uncovered
            if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
        ]
        assert not missing_default, (
            "Qwen2VLCausalLMOutputWithPast added a required field that "
            "mirror_output_fields cannot fill: "
            f"{missing_default}. The Mobilint wrapper must supply this field "
            "as an explicit override, or upstream must add it to "
            "Qwen2VLModelOutputWithPast so the mirror picks it up."
        )


class TestQwen2VLLossFunctionKwargsSupply:
    """Every required loss parameter must be in the supply pool."""

    def test_required_loss_params_are_in_supply_pool(self) -> None:
        from transformers.loss.loss_utils import ForCausalLMLoss

        from mblt_model_zoo.hf_transformers.utils.generation_utils import (
            LOSS_KWARG_SUPPLY_NAMES,
        )

        signature = inspect.signature(ForCausalLMLoss)
        required_named = [
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and parameter.default is inspect.Parameter.empty
        ]
        missing = [name for name in required_named if name not in LOSS_KWARG_SUPPLY_NAMES]
        assert not missing, (
            "ForCausalLMLoss requires parameters that build_loss_kwargs_dynamic "
            f"does not know how to supply: {missing}. Extend "
            "LOSS_KWARG_SUPPLY_NAMES and the ``supply`` dict in "
            "build_loss_kwargs_dynamic together to cover them."
        )
