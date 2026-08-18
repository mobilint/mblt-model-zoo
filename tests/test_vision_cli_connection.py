"""Compatibility coverage for Model Zoo Vision CLI delegation."""

from __future__ import annotations

from mblt_vision.cli.compile import _run_compile
from mblt_vision.cli.predict import _cmd_predict
from mblt_vision.cli.val import _cmd_val

from mblt_model_zoo.cli.main import build_parser


def test_model_zoo_vision_commands_use_standalone_handlers() -> None:
    """Keep Model Zoo command parsing connected to the standalone CLI."""

    parser = build_parser()

    predict_args = parser.parse_args(["predict", "--source", "image.jpg", "--model", "resnet50"])
    val_args = parser.parse_args(["val", "--model", "resnet50"])
    compile_args = parser.parse_args(["compile", "--model-cls", "resnet50", "--target-device", "aries-rb"])

    assert predict_args._handler is _cmd_predict
    assert val_args._handler is _cmd_val
    assert compile_args._handler is _run_compile
