"""Regression tests for the Qwen3-VL text decoder ``logits_to_keep`` path.

Covers the three inference branches the Qwen3-VL text decoder implements
inside its own ``llm_forward`` (mirroring the shared core), plus the outer
``MobilintQwen3VLForConditionalGeneration.forward`` wrapper that has to pop
``logits_to_keep`` from kwargs and thread it into the text model.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
import torch

from tests.transformers.image_text_to_text.qwen3_vl_compat import (
    skip_if_transformers_lacks_qwen3_vl_support,
)

skip_if_transformers_lacks_qwen3_vl_support()

from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
    MobilintQwen3VLForConditionalGeneration,
    MobilintQwen3VLTextModel,
)


# ---------------------------------------------------------------------------
# Fake MXQ backends for the dual-input Qwen3-VL text decoder
#
# These are intentionally NOT unified with ``tests/transformers/_fake_mxq.py``:
# the Qwen3-VL decoder feeds two input buffers (inputs_embeds and deepstack
# embeds) and its ``infer`` signature has no ``batch_params`` argument, so the
# fakes exercise a different code path than the shared LLM core.
# ---------------------------------------------------------------------------


class _StaticLastOnlyMxq:
    """MXQ stub for the qwen3_vl decoder that emits only last-token logits."""

    def __init__(self, vocab_size: int = 5):
        self.vocab_size = vocab_size
        self.calls: list[dict] = []

    def get_model_output_shape(self):
        return [(1, 1, self.vocab_size)]

    def infer(self, inputs, _extra, cache_size):
        inputs_chunk = np.asarray(inputs[0])
        deepstack_chunk = np.asarray(inputs[1])
        self.calls.append(
            {
                "inputs_shape": tuple(inputs_chunk.shape),
                "deepstack_shape": tuple(deepstack_chunk.shape),
                "cache_size": int(cache_size),
            }
        )
        return [np.full((1, 1, self.vocab_size), fill_value=float(cache_size), dtype=np.float32)]


class _DynamicAxisMxq:
    """MXQ stub for the qwen3_vl decoder that emits per-position logits."""

    def __init__(self, vocab_size: int = 5):
        self.vocab_size = vocab_size
        self.calls: list[dict] = []

    def get_model_output_shape(self):
        return [(1, -1, self.vocab_size)]

    def infer(self, inputs, _extra, cache_size):
        inputs_chunk = np.asarray(inputs[0])
        chunk_len = int(inputs_chunk.shape[1])
        base = int(cache_size) * self.vocab_size
        values = (np.arange(chunk_len * self.vocab_size, dtype=np.float32) + base).reshape(
            1, chunk_len, self.vocab_size
        )
        self.calls.append(
            {
                "inputs_shape": tuple(inputs_chunk.shape),
                "cache_size": int(cache_size),
                "chunk_len": chunk_len,
            }
        )
        return [values]


class _FakeBackend:
    def __init__(self, mxq_model):
        self.mxq_model = mxq_model


class _BareQwen3VLTextModel(MobilintQwen3VLTextModel):
    """Instantiate ``MobilintQwen3VLTextModel`` without booting the NPU backend.

    Only the attributes ``llm_forward`` needs are populated; ``get_mxq_model``
    is redirected to the fake backend via ``self.npu_backend``.
    """

    def __init__(self, mxq_model, *, vocab_size: int = 5, hidden_size: int = 3):
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            pad_token_id=0,
            npu_prefill_chunk_size=None,
        )
        self.npu_backend = _FakeBackend(mxq_model)
        self.num_deepstack_layers = 0
        self.npu_time = None

    def get_mxq_model(self):  # noqa: D401 - trivial passthrough
        return self.npu_backend.mxq_model


# ---------------------------------------------------------------------------
# Direct llm_forward branches
# ---------------------------------------------------------------------------


class TestQwen3VLTextDecoderLogitsToKeep:
    def _run(
        self,
        mxq,
        *,
        seq_len: int,
        hidden_size: int = 3,
        logits_to_keep,
        prefill_chunk_size: Optional[int] = None,
    ):
        model = _BareQwen3VLTextModel(mxq, hidden_size=hidden_size)
        inputs_embeds = torch.arange(seq_len * hidden_size, dtype=torch.float32).reshape(
            1, seq_len, hidden_size
        )
        cache_position = torch.arange(seq_len)
        logits = model.llm_forward(
            inputs_embeds=inputs_embeds,
            deepstack_visual_embeds=None,
            visual_pos_masks=None,
            past_key_values=None,
            cache_position=cache_position,
            prefill_chunk_size=prefill_chunk_size,
            logits_to_keep=logits_to_keep,
        )
        return model, logits

    def test_default_keep_returns_last_token_logits(self) -> None:
        mxq = _StaticLastOnlyMxq(vocab_size=5)
        _model, logits = self._run(mxq, seq_len=6, logits_to_keep=1, prefill_chunk_size=3)

        chunk_seqs = [c["inputs_shape"][1] for c in mxq.calls]
        assert chunk_seqs == [3, 3]
        # No squeeze on this path: (batch=1, 1, vocab).
        assert logits.shape == (1, 1, mxq.vocab_size)

    def test_dynamic_axis_keep_all_returns_every_position(self) -> None:
        mxq = _DynamicAxisMxq(vocab_size=5)
        model, logits = self._run(mxq, seq_len=6, logits_to_keep=0, prefill_chunk_size=3)

        assert model._mxq_supports_all_logits() is True
        assert logits.shape == (1, 6, mxq.vocab_size)

        # Concatenation should be in call order, one chunk per prefill_chunk_size window.
        # Without a caller cache, ``_do_infer`` uses ``start_index`` as cache_size so
        # the NPU sees prior chunks' KV state — the fake mirrors that by offsetting.
        expected_chunks = []
        running_cache_size = 0
        for chunk_len in (3, 3):
            base = running_cache_size * mxq.vocab_size
            expected_chunks.append(
                (np.arange(chunk_len * mxq.vocab_size, dtype=np.float32) + base).reshape(
                    1, chunk_len, mxq.vocab_size
                )
            )
            running_cache_size += chunk_len
        expected = np.concatenate(expected_chunks, axis=1)
        np.testing.assert_allclose(logits.numpy(), expected)

    def test_dynamic_axis_last_n_slices_kept_positions(self) -> None:
        mxq = _DynamicAxisMxq(vocab_size=5)
        _model, logits = self._run(mxq, seq_len=6, logits_to_keep=2, prefill_chunk_size=3)
        assert logits.shape == (1, 2, mxq.vocab_size)

    def test_dynamic_axis_tensor_indices_pick_out_positions(self) -> None:
        mxq = _DynamicAxisMxq(vocab_size=5)
        indices = torch.tensor([0, 3])
        _model, logits = self._run(
            mxq, seq_len=6, logits_to_keep=indices, prefill_chunk_size=3
        )
        assert logits.shape == (1, 2, mxq.vocab_size)

    def test_fallback_interleaves_size_one_infer_for_kept_positions(self) -> None:
        mxq = _StaticLastOnlyMxq(vocab_size=5)
        indices = torch.tensor([2, 5])
        _model, logits = self._run(
            mxq, seq_len=6, logits_to_keep=indices, prefill_chunk_size=4
        )

        chunk_seqs = [c["inputs_shape"][1] for c in mxq.calls]
        # Prefix stride is clamped to the next kept position (like the shared core).
        assert chunk_seqs == [2, 1, 2, 1]
        # No ``.squeeze(0)`` on this path: shape is (1, kept_len, vocab).
        assert logits.shape == (1, len(indices), mxq.vocab_size)

    def test_fallback_empty_tensor_returns_empty_logits(self) -> None:
        """An empty ``logits_to_keep`` tensor must not raise on np.concatenate."""
        mxq = _StaticLastOnlyMxq(vocab_size=5)
        _model, logits = self._run(
            mxq,
            seq_len=6,
            logits_to_keep=torch.tensor([], dtype=torch.long),
            prefill_chunk_size=4,
        )
        # KV prefix still walks the whole sequence in normal-sized chunks.
        chunk_seqs = [c["inputs_shape"][1] for c in mxq.calls]
        assert chunk_seqs == [4, 2]
        # Batch dim preserved (no ``.squeeze(0)`` on this path).
        assert logits.shape == (1, 0, 0)

    def test_fallback_out_of_range_indices_raise_indexerror(self) -> None:
        """HF fancy-indexing raises on out-of-range indices; the VL decoder inherits this."""
        mxq = _StaticLastOnlyMxq(vocab_size=5)
        with pytest.raises(IndexError, match=r"100.*seq_len=6"):
            self._run(
                mxq,
                seq_len=6,
                logits_to_keep=torch.tensor([100, -100]),
                prefill_chunk_size=4,
            )

    def test_deepstack_chunk_shape_matches_input_chunk(self) -> None:
        mxq = _StaticLastOnlyMxq(vocab_size=5)
        model = _BareQwen3VLTextModel(mxq, hidden_size=3)
        model.num_deepstack_layers = 2

        seq_len = 4
        inputs_embeds = torch.zeros(1, seq_len, 3)
        cache_position = torch.arange(seq_len)

        model.llm_forward(
            inputs_embeds=inputs_embeds,
            deepstack_visual_embeds=None,
            visual_pos_masks=None,
            past_key_values=None,
            cache_position=cache_position,
            prefill_chunk_size=2,
            logits_to_keep=1,
        )
        # Deepstack chunk is built from a zero tensor of shape (num_layers, seq, hidden)
        # sliced by the same [start:end] window as the inputs chunk. Each call
        # sees (2, chunk_len, 3).
        assert [c["deepstack_shape"] for c in mxq.calls] == [(2, 2, 3), (2, 2, 3)]


# ---------------------------------------------------------------------------
# Outer ``MobilintQwen3VLForConditionalGeneration.forward`` wrapper
# ---------------------------------------------------------------------------


class _ModelOutput:
    """Subscriptable, attribute-accessible output mimicking ``Qwen3VLModelOutput``."""

    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state
        self.past_key_values = None
        self.hidden_states = None
        self.attentions = None
        self.rope_deltas = None

    def __getitem__(self, index):  # noqa: D401 - matches HF ModelOutput contract
        return self.last_hidden_state if index == 0 else None


class _RecordingModel:
    """Stand-in for ``self.model`` used to observe forwarded kwargs."""

    def __init__(self, vocab_size: int, kept_len: int, hidden_size: int) -> None:
        self.vocab_size = vocab_size
        self.kept_len = kept_len
        self.hidden_size = hidden_size
        self.received: dict = {}

    def __call__(self, **kwargs):
        self.received = kwargs
        last_hidden_state = torch.arange(
            self.kept_len * self.vocab_size, dtype=torch.float32
        ).reshape(1, self.kept_len, self.vocab_size)
        return _ModelOutput(last_hidden_state)


class TestQwen3VLForConditionalGenerationForward:
    def _make_wrapper(self, kept_len: int = 3, vocab_size: int = 5, hidden_size: int = 4):
        wrapper = MobilintQwen3VLForConditionalGeneration.__new__(
            MobilintQwen3VLForConditionalGeneration
        )
        torch.nn.Module.__init__(wrapper)
        wrapper.config = SimpleNamespace(
            text_config=SimpleNamespace(vocab_size=vocab_size),
            return_dict=True,
        )
        wrapper.model = _RecordingModel(
            vocab_size=vocab_size, kept_len=kept_len, hidden_size=hidden_size
        )
        wrapper.lm_head = torch.nn.Identity()
        return wrapper

    def test_forward_pops_logits_to_keep_and_threads_to_model(self) -> None:
        wrapper = self._make_wrapper(kept_len=2)
        input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

        output = wrapper.forward(input_ids=input_ids, logits_to_keep=torch.tensor([1, 3]))

        # The wrapper must forward the raw ``logits_to_keep`` value to
        # ``self.model``; upstream would otherwise re-slice the decoder output.
        assert isinstance(output.logits, torch.Tensor)
        assert output.logits.shape == (1, 2, wrapper.config.text_config.vocab_size)
        assert "logits_to_keep" in wrapper.model.received
        forwarded = wrapper.model.received["logits_to_keep"]
        assert torch.is_tensor(forwarded)
        assert forwarded.tolist() == [1, 3]
        # ``labels`` and ``logits_to_keep`` must not leak into the model kwargs.
        assert "labels" not in wrapper.model.received

    def test_forward_defaults_to_keep_all_when_kwarg_absent(self) -> None:
        wrapper = self._make_wrapper(kept_len=4)
        input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

        wrapper.forward(input_ids=input_ids)

        # The wrapper's default is ``0`` (keep every position), matching the
        # underlying text model's default.
        assert wrapper.model.received.get("logits_to_keep") == 0

    def test_forward_maps_positional_args_to_upstream_signature(self) -> None:
        wrapper = self._make_wrapper(kept_len=1)
        input_ids = torch.tensor([[7, 8, 9]], dtype=torch.long)

        # Pass input_ids positionally so we exercise the upstream-parameter
        # remapping done by the wrapper.
        wrapper.forward(input_ids)

        assert "input_ids" in wrapper.model.received
        assert torch.equal(wrapper.model.received["input_ids"], input_ids)

    def test_forward_rejects_duplicate_positional_and_keyword_arg(self) -> None:
        wrapper = self._make_wrapper(kept_len=1)
        input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

        with pytest.raises(TypeError, match="multiple values for argument 'input_ids'"):
            wrapper.forward(input_ids, input_ids=input_ids)

    def test_forward_strips_loss_only_kwargs_before_model_call(self) -> None:
        """Loss-only kwargs must not leak into ``self.model``.

        Upstream ``Qwen3VLModel.forward`` forwards its ``**kwargs`` to
        ``self.language_model``. Even though ``MobilintQwen3VLTextModel``
        accepts ``**kwargs``, loss-only kwargs conceptually belong to loss
        computation only and must be stripped for parity with Qwen2-VL.
        """
        wrapper = self._make_wrapper(kept_len=4, vocab_size=5)
        input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        labels = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

        output = wrapper.forward(
            input_ids=input_ids,
            labels=labels,
            num_items_in_batch=2,
            shift_labels=torch.tensor([[1, 2, 3, -100]], dtype=torch.long),
        )

        assert "num_items_in_batch" not in wrapper.model.received
        assert "shift_labels" not in wrapper.model.received
        # Loss still computes normally with the stripped loss-only kwargs.
        assert output.loss is not None

    def test_forward_return_dict_false_returns_tuple(self) -> None:
        """``return_dict=False`` must not leak into ``self.model`` and must yield a tuple.

        Without ``@can_return_tuple``, ``return_dict`` would flow through to
        ``self.model`` and the upstream ``Qwen3VLModel`` contract would return a
        tuple whose ``.last_hidden_state`` dereference crashes the wrapper.
        """
        wrapper = self._make_wrapper(kept_len=2, vocab_size=5)
        input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

        output = wrapper.forward(input_ids=input_ids, return_dict=False)

        assert isinstance(output, tuple)
        # ``return_dict`` must be stripped before ``self.model`` is called.
        assert "return_dict" not in wrapper.model.received
        # Logits are the first non-None field of Qwen3VLCausalLMOutputWithPast
        # (loss is None because no labels).
        assert output[0].shape == (1, 2, 5)

    def test_forward_rejects_too_many_positional_args(self) -> None:
        """Extra positional args beyond upstream's signature must raise TypeError.

        ``zip`` silently drops extras, so without an explicit length check a caller
        passing too many positional args would have those extras discarded rather
        than surfaced. Mirror CPython's own message so the failure reads naturally.
        """
        from mblt_model_zoo.hf_transformers.utils.generation_utils import (
            upstream_positional_params,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )

        wrapper = self._make_wrapper(kept_len=1)
        param_count = len(upstream_positional_params(Qwen3VLForConditionalGeneration.forward))
        extra_args = tuple(object() for _ in range(param_count + 1))

        with pytest.raises(
            TypeError,
            match=rf"forward\(\) takes at most {param_count} positional arguments "
            rf"but {param_count + 1} were given",
        ):
            wrapper.forward(*extra_args)
