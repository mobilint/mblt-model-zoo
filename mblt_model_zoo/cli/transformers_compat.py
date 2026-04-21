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
    _prepare_transformers_cli(argv)

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


def _prepare_transformers_cli(argv: Sequence[str]) -> None:
    _maybe_register_mobilint_chat_model(argv)
    if len(argv) > 1 and argv[1] == "serve":
        _install_transformers_serve_registration_hook()


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


def _install_transformers_serve_registration_hook() -> None:
    serve_module_name = _get_transformers_serve_module_name()
    serve_module = importlib.import_module(serve_module_name)
    serve_cls = serve_module.Serve if hasattr(serve_module, "Serve") else serve_module.ServeCommand

    if hasattr(serve_cls, "_load_model_and_data_processor"):
        _install_registration_wrapper(
            target_cls=serve_cls,
            method_name="_load_model_and_data_processor",
            extra_transformers=getattr(serve_module, "transformers", transformers),
        )
        return

    if _has_module("transformers.cli.serving.model_manager"):
        model_manager_module = importlib.import_module("transformers.cli.serving.model_manager")
        _install_registration_wrapper(
            target_cls=model_manager_module.ModelManager,
            method_name="load_model_and_processor",
            extra_transformers=getattr(model_manager_module, "transformers", transformers),
        )


def _install_registration_wrapper(target_cls: type, method_name: str, extra_transformers: object) -> None:
    marker_attr = f"_mblt_registration_hook_installed_for_{method_name}"
    if getattr(target_cls, marker_attr, False):
        return

    original_load = getattr(target_cls, method_name)

    def _wrapped_load(self, model_id_and_revision: str, *args, **kwargs):
        _register_mobilint_model_for_modules(model_id_and_revision, extra_transformers)
        return original_load(self, model_id_and_revision, *args, **kwargs)

    setattr(target_cls, method_name, _wrapped_load)
    setattr(target_cls, marker_attr, True)


def _register_mobilint_model_for_modules(model_id_and_revision: str, extra_transformers: object) -> None:
    model_name_or_path_or_address = model_id_and_revision.split("@", 1)[0]
    args = SimpleNamespace(model_name_or_path_or_address=model_name_or_path_or_address)
    register_mobilint_models(args, transformers)
    if extra_transformers is not transformers:
        register_mobilint_models(args, extra_transformers)


def _get_transformers_serve_module_name() -> str:
    if _has_module("transformers.cli.serve"):
        return "transformers.cli.serve"
    if _has_module("transformers.commands.serving"):
        return "transformers.commands.serving"
    return "transformers.commands.serve"


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
