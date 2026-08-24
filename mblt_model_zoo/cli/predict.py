"""Compatibility exports for the standalone Vision prediction command."""

from mblt_vision.cli.predict import _cmd_predict, add_predict_parser

__all__ = ["_cmd_predict", "add_predict_parser"]
