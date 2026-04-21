"""Compatibility helpers for delegating to the upstream Transformers CLI."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Sequence

import transformers

from .chat import register_mobilint_models

TRANSFORMERS_CLI_COMMANDS = frozenset(
    {
        "add-fast-image-processor",
        "add-new-model-like",
        "chat",
        "convert",
        "download",
        "env",
        "run",
        "serve",
        "version",
    }
)


def is_transformers_cli_command(argv: Sequence[str]) -> bool:
    """Return whether the argv targets an upstream Transformers CLI command."""
    return len(argv) > 1 and argv[1] in TRANSFORMERS_CLI_COMMANDS


def dispatch_transformers_cli(argv: Sequence[str]) -> int:
    """Delegate the active command to the installed Transformers CLI."""
    _maybe_register_mobilint_chat_model(argv)

    module_name = "transformers.cli.transformers" if _has_module("transformers.cli.transformers") else (
        "transformers.commands.transformers_cli"
    )
    cli_module = importlib.import_module(module_name)

    original_argv = sys.argv[:]
    try:
        sys.argv = list(argv)
        cli_module.main()
    except SystemExit as exc:
        return _normalize_exit_code(exc.code)
    finally:
        sys.argv = original_argv

    return 0


def _has_module(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        return False
    return True


def _maybe_register_mobilint_chat_model(argv: Sequence[str]) -> None:
    if len(argv) <= 2 or argv[1] != "chat":
        return

    model_name_or_path_or_address = _extract_chat_model_name(argv[2:])
    if model_name_or_path_or_address is None or _looks_like_remote_endpoint(model_name_or_path_or_address):
        return

    register_mobilint_models(
        SimpleNamespace(model_name_or_path_or_address=model_name_or_path_or_address),
        transformers,
    )


def _extract_chat_model_name(argv: Sequence[str]) -> str | None:
    for token in argv:
        if token == "--":
            return None
        if token.startswith("-"):
            continue
        return token
    return None


def _looks_like_remote_endpoint(value: str) -> bool:
    return value.startswith(("http://", "https://", "localhost"))


def _normalize_exit_code(code: object) -> int:
    if isinstance(code, int):
        return code
    if code is None:
        return 0
    return 1
