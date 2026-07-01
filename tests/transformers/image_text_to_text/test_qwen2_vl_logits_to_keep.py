"""Regression tests for the Qwen2-VL ``forward`` wrapper's ``logits_to_keep`` routing.

The Qwen2-VL text decoder now honors ``logits_to_keep`` inside its own
``llm_forward``. That only works if
``MobilintQwen2VLForConditionalGeneration.forward`` pops ``logits_to_keep``
from kwargs and threads it into ``self.model`` — otherwise upstream extracts
the same named argument and performs its own final slice on the decoder
output, silently bypassing the Mobilint decoder's position selection.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from mblt_model_zoo.hf_transformers.models.qwen2_vl.modeling_qwen2_vl import (
    MobilintQwen2VLForConditionalGeneration,
)


class _RecordingModel:
    """Stand-in for ``self.model`` used to observe forwarded kwargs."""

    def __init__(self, vocab_size: int, kept_len: int) -> None:
        self.vocab_size = vocab_size
        self.kept_len = kept_len
        self.received: dict = {}

    def __call__(self, **kwargs):
        self.received = kwargs
        last_hidden_state = torch.arange(
            self.kept_len * self.vocab_size, dtype=torch.float32
        ).reshape(1, self.kept_len, self.vocab_size)
        return SimpleNamespace(
            last_hidden_state=last_hidden_state,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            rope_deltas=None,
        )


def _make_wrapper(kept_len: int = 3, vocab_size: int = 5):
    wrapper = MobilintQwen2VLForConditionalGeneration.__new__(
        MobilintQwen2VLForConditionalGeneration
    )
    torch.nn.Module.__init__(wrapper)
    wrapper.config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=vocab_size))
    wrapper.model = _RecordingModel(vocab_size=vocab_size, kept_len=kept_len)
    wrapper.lm_head = torch.nn.Identity()
    return wrapper


class TestQwen2VLForwardLogitsToKeep:
    def test_forward_pops_logits_to_keep_and_threads_to_model(self) -> None:
        wrapper = _make_wrapper(kept_len=2)
        input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

        output = wrapper.forward(input_ids=input_ids, logits_to_keep=torch.tensor([1, 3]))

        assert output.logits.shape == (1, 2, wrapper.config.text_config.vocab_size)
        assert "logits_to_keep" in wrapper.model.received
        forwarded = wrapper.model.received["logits_to_keep"]
        assert torch.is_tensor(forwarded)
        assert forwarded.tolist() == [1, 3]

    def test_forward_removes_labels_and_logits_to_keep_before_forwarding(self) -> None:
        wrapper = _make_wrapper(kept_len=4, vocab_size=5)
        input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        labels = torch.zeros_like(input_ids)

        wrapper.forward(
            input_ids=input_ids,
            labels=labels,
            logits_to_keep=1,
        )

        # ``labels`` is consumed for the loss computation and must not leak to
        # the model call, and ``logits_to_keep`` must be popped so upstream's
        # own re-slicing cannot bypass the Mobilint decoder.
        assert "labels" not in wrapper.model.received
        # ``logits_to_keep`` is passed once, as an explicit keyword.
        assert wrapper.model.received["logits_to_keep"] == 1

    def test_forward_defaults_to_keep_all_when_kwarg_absent(self) -> None:
        wrapper = _make_wrapper(kept_len=4)
        input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

        wrapper.forward(input_ids=input_ids)

        # The wrapper's default is ``0`` (keep every position).
        assert wrapper.model.received.get("logits_to_keep") == 0

    def test_forward_maps_positional_args_to_upstream_signature(self) -> None:
        wrapper = _make_wrapper(kept_len=1)
        input_ids = torch.tensor([[7, 8, 9]], dtype=torch.long)

        wrapper.forward(input_ids)

        assert "input_ids" in wrapper.model.received
        assert torch.equal(wrapper.model.received["input_ids"], input_ids)

    def test_forward_computes_loss_when_labels_are_provided(self) -> None:
        wrapper = _make_wrapper(kept_len=4, vocab_size=5)
        input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        labels = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

        output = wrapper.forward(input_ids=input_ids, labels=labels)

        assert output.loss is not None
        assert output.loss.dim() == 0
