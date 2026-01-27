import pytest

from mblt_model_zoo.cli.main import build_parser


def test_cli_tps_sweep_range_parsing():
    parser = build_parser()
    args = parser.parse_args(
        [
            "tps",
            "sweep",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct",
            "--prefill-range",
            "1:3:1",
            "--decode-range",
            "2,4,2",
            "--no-plot",
        ]
    )
    assert args.prefill_range == (1, 3, 1)
    assert args.decode_range == (2, 4, 2)
    assert args.plot is None


@pytest.mark.parametrize("spec", ["", "1", "1:2", "1:2:0", "2:1:1", "a:b:c"])
def test_cli_tps_invalid_range_exits(spec: str):
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "tps",
                "sweep",
                "--model",
                "mobilint/Llama-3.2-1B-Instruct",
                "--prefill-range",
                spec,
                "--no-plot",
            ]
        )
    assert excinfo.value.code == 2
