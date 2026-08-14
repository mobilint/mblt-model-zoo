import sys
from pathlib import Path

import pytest

_TRANSFORMERS_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "transformers"
if str(_TRANSFORMERS_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_TRANSFORMERS_BENCHMARK_DIR))

from benchmark.transformers import search_npu_prefill_chunk_size as chunk_search  # noqa: E402


def test_include_private_flag_defaults_false_and_forwards_to_list_default_model_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify --include-private threads through the shared default model listing helper."""
    observed: list[bool] = []

    class _StopAfterListing(SystemExit):
        pass

    def fake_list_default_model_ids(task, *, include_private=False):  # type: ignore[no-untyped-def]
        observed.append(bool(include_private))
        raise _StopAfterListing("stop after include_private forwarding")

    monkeypatch.setattr(chunk_search, "list_default_model_ids", fake_list_default_model_ids)

    default_args = chunk_search.build_arg_parser().parse_args(["--output-dir", str(tmp_path)])
    assert default_args.include_private is False
    private_args = chunk_search.build_arg_parser().parse_args(["--include-private", "--output-dir", str(tmp_path)])
    assert private_args.include_private is True

    for cli_args in (
        ["--output-dir", str(tmp_path)],
        ["--include-private", "--output-dir", str(tmp_path)],
    ):
        with pytest.raises(_StopAfterListing):
            chunk_search.main(cli_args)

    assert observed == [False, True]
