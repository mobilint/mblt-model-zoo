"""Upstream contract: ``Qwen3VLModel.forward`` must not slice its own output.

``MobilintQwen3VLForConditionalGeneration.forward`` pops ``logits_to_keep``
from kwargs, threads it into ``self.model`` (which forwards it via ``**kwargs``
to the Mobilint text decoder), and then SKIPS the upstream
``hidden_states[:, slice_indices, :]`` step because the Mobilint decoder
already returns logits selected to the requested positions.

That contract holds only as long as upstream ``Qwen3VLModel.forward`` itself
does not do its own position selection. If a future HF transformers release
moves the slice from ``Qwen3VLForConditionalGeneration.forward`` into
``Qwen3VLModel.forward``, our wrapper would silently return pre-sliced values
(KV would still be correct — only the logits would be wrong), and the existing
``_RecordingModel``-based tests could not detect the drift.

These tests pin that contract via source inspection.
"""

from __future__ import annotations

import inspect
import re

import pytest

from tests.transformers.image_text_to_text.qwen3_vl_compat import (
    skip_if_transformers_lacks_qwen3_vl_support,
)

skip_if_transformers_lacks_qwen3_vl_support()

from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel  # noqa: E402

pytestmark = pytest.mark.upstream_contract


def _forward_source() -> str:
    return inspect.getsource(Qwen3VLModel.forward)


class TestQwen3VLModelForwardSourceContract:
    """Fail loudly if the upstream source acquires its own logits slicing."""

    def test_no_slice_indices_local(self) -> None:
        # ``slice_indices = slice(-logits_to_keep, None) ...`` is the exact
        # pattern used in ``Qwen3VLForConditionalGeneration.forward``. It must
        # not migrate into ``Qwen3VLModel.forward``.
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
        # ``Qwen3VLModel.forward``, it stops flowing through ``**kwargs`` to
        # ``self.language_model`` and is very likely paired with local slicing.
        params = inspect.signature(Qwen3VLModel.forward).parameters
        assert "logits_to_keep" not in params

    def test_kwargs_are_forwarded_to_language_model(self) -> None:
        # Our wrapper depends on ``**kwargs`` reaching ``self.language_model``.
        source = _forward_source()
        match = re.search(r"self\.language_model\((?P<args>.*?)\)", source, re.DOTALL)
        assert match is not None, "expected a self.language_model(...) call in upstream forward"
        assert "**kwargs" in match.group("args")
