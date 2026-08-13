"""Unit tests for the ``tps measure --print-output`` footer format.

The footer separates the TTFT sample from decode tokens using the same convention as
``decode_tps = (decoded_tokens - 1) / decode_duration`` so the diagnostic and the
reported throughput agree.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

from mblt_model_zoo.cli import tps as tps_cli


class _StubTokenizer:
    """Minimal tokenizer that returns deterministic strings for either decode mode."""

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        marker = "clean" if skip_special_tokens else "raw"
        return f"<{marker}:{len(token_ids)}>"


def _build_fixtures(token_ids: Sequence[int]) -> tuple[SimpleNamespace, SimpleNamespace]:
    pipeline = SimpleNamespace(tokenizer=_StubTokenizer())
    run = SimpleNamespace(generated_token_ids=list(token_ids))
    return pipeline, run


def _captured_footer(capsys) -> str:
    lines = capsys.readouterr().out.splitlines()
    footer = [line for line in lines if line.startswith("--- token count:")]
    assert footer, "footer line not emitted"
    return footer[-1]


def test_print_output_footer_full_length_run(capsys):
    """A full-length run with 33 emitted tokens reports 32 decode tokens + 1 TTFT sample."""
    pipeline, run = _build_fixtures(list(range(33)))

    tps_cli._print_generated_output(pipeline, [run], decode_budget=32)

    assert (
        _captured_footer(capsys)
        == "--- token count: 32 decode tokens (+ 1 TTFT sample = 33 emitted; --decode 32 max) ---"
    )


def test_print_output_footer_eos_early_termination(capsys):
    """EOS-terminated runs report the decode count as ``emitted - 1`` against the full budget."""
    pipeline, run = _build_fixtures(list(range(5)))

    tps_cli._print_generated_output(pipeline, [run], decode_budget=200)

    assert (
        _captured_footer(capsys)
        == "--- token count: 4 decode tokens (+ 1 TTFT sample = 5 emitted; --decode 200 max) ---"
    )


def test_print_output_footer_no_tokens_emits_unavailable(capsys):
    """Empty ``generated_token_ids`` short-circuits before the footer is rendered."""
    pipeline, run = _build_fixtures([])

    tps_cli._print_generated_output(pipeline, [run], decode_budget=32)

    output = capsys.readouterr().out
    assert "unavailable" in output
    assert "token count:" not in output


def test_print_output_footer_single_token_reports_zero_decode(capsys):
    """A single emitted token counts as a TTFT sample only, matching decode_tps convention."""
    pipeline, run = _build_fixtures([7])

    tps_cli._print_generated_output(pipeline, [run], decode_budget=32)

    assert (
        _captured_footer(capsys)
        == "--- token count: 0 decode tokens (+ 1 TTFT sample = 1 emitted; --decode 32 max) ---"
    )
