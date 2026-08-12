"""Unit tests for TokenIteratorStreamer token-id collection used by tps measure --print-output."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from mblt_model_zoo.hf_transformers.utils.benchmark_utils import TokenIteratorStreamer


def _make_streamer(collect: bool) -> TokenIteratorStreamer:
    tokenizer = MagicMock()
    tokenizer.decode = MagicMock(return_value="")
    return TokenIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        collect_token_ids=collect,
    )


def test_streamer_collects_token_ids_after_prompt():
    streamer = _make_streamer(collect=True)

    # First put is the prompt (skipped by skip_prompt=True).
    streamer.put(torch.tensor([[1, 2, 3, 4]]))
    # Subsequent puts are decoded tokens.
    streamer.put(torch.tensor([[100]]))
    streamer.put(torch.tensor([[200]]))
    streamer.put(torch.tensor([[300]]))
    streamer.end()

    assert streamer.token_ids == [100, 200, 300]


def test_streamer_default_does_not_collect_token_ids():
    streamer = _make_streamer(collect=False)

    streamer.put(torch.tensor([[1, 2, 3, 4]]))  # prompt
    streamer.put(torch.tensor([[100]]))
    streamer.put(torch.tensor([[200]]))
    streamer.end()

    assert streamer.token_ids == []


def test_streamer_collects_1d_tensor_after_prompt():
    streamer = _make_streamer(collect=True)

    streamer.put(torch.tensor([1, 2, 3, 4]))  # prompt (rank-1)
    streamer.put(torch.tensor([100]))
    streamer.put(torch.tensor([200]))
    streamer.end()

    assert streamer.token_ids == [100, 200]
