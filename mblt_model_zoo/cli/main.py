from __future__ import annotations

import argparse
from typing import Sequence

from .tps import add_tps_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mblt-model-zoo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_tps_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args._handler(args))
