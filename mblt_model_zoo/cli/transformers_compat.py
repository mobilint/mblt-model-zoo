"""Compatibility helpers for delegating to the upstream Transformers CLI."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import transformers
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id

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

    module_name = (
        "transformers.cli.transformers"
        if _has_module("transformers.cli.transformers")
        else ("transformers.commands.transformers_cli")
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
    if _should_install_transformers_serve_registration_hook(argv):
        _install_transformers_serve_registration_hook()


def _should_install_transformers_serve_registration_hook(argv: Sequence[str]) -> bool:
    """Return whether delegated command execution may load models through `ServeCommand`."""
    if len(argv) <= 1:
        return False
    if argv[1] == "serve":
        return True
    if argv[1] == "chat":
        return _chat_uses_transformers_serve_backend()
    return False


def _chat_uses_transformers_serve_backend() -> bool:
    """Return whether delegated chat uses the legacy local `ServeCommand` backend."""
    return _has_module_spec("transformers.commands.chat") and not _has_module_spec("transformers.cli.chat")


def _has_module_spec(module_name: str) -> bool:
    """Return whether the Python import machinery can resolve the module name."""
    return importlib.util.find_spec(module_name) is not None


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
        trust_remote_code = _get_loader_trust_remote_code(self)
        _register_mobilint_model_for_modules(
            model_id_and_revision,
            extra_transformers,
            trust_remote_code=trust_remote_code,
        )
        return original_load(self, model_id_and_revision, *args, **kwargs)

    setattr(target_cls, method_name, _wrapped_load)
    setattr(target_cls, marker_attr, True)


def _get_loader_trust_remote_code(loader_owner: object) -> bool:
    """Return the effective `trust_remote_code` value for the active serve loader."""
    candidate_sources = [
        getattr(loader_owner, "args", None),
        loader_owner,
        getattr(loader_owner, "_model_manager", None),
    ]
    for source in candidate_sources:
        if source is None:
            continue
        trust_remote_code = getattr(source, "trust_remote_code", None)
        if isinstance(trust_remote_code, bool):
            return trust_remote_code
    return False


def _register_mobilint_model_for_modules(
    model_id_and_revision: str,
    extra_transformers: object,
    trust_remote_code: bool = False,
) -> None:
    model_name_or_path_or_address, model_revision = _split_model_id_and_revision(model_id_and_revision)
    args = SimpleNamespace(
        model_name_or_path_or_address=model_name_or_path_or_address,
        model_revision=model_revision,
        trust_remote_code=trust_remote_code,
    )
    register_mobilint_models(args, transformers)
    if extra_transformers is not transformers:
        register_mobilint_models(args, extra_transformers)


def _get_transformers_serve_module_name() -> str:
    if _has_module("transformers.cli.serve"):
        return "transformers.cli.serve"
    if _has_module("transformers.commands.serving"):
        return "transformers.commands.serving"
    return "transformers.commands.serve"


def _split_model_id_and_revision(model_id_and_revision: str) -> tuple[str, str | None]:
    """Split Hub and canonicalized local refs into a model reference plus revision."""
    if "@" not in model_id_and_revision:
        return model_id_and_revision, None
    if _existing_local_model_path(model_id_and_revision) is not None:
        return model_id_and_revision, None

    model_name_or_path_or_address, separator, model_revision = model_id_and_revision.rpartition("@")
    if not separator or not model_name_or_path_or_address or not model_revision:
        return model_id_and_revision, None

    if _existing_local_model_path(model_name_or_path_or_address) is not None:
        return model_name_or_path_or_address, model_revision
    if _looks_like_local_model_path(model_id_and_revision):
        return model_id_and_revision, None
    if _is_valid_hub_model_id(model_name_or_path_or_address):
        return model_name_or_path_or_address, model_revision
    return model_id_and_revision, None


def _existing_local_model_path(value: str) -> Path | None:
    """Return the expanded local path when it exists on disk."""
    try:
        path = Path(value).expanduser()
    except (OSError, RuntimeError, ValueError):
        return None
    return path if path.exists() else None


def _is_valid_hub_model_id(value: str) -> bool:
    """Return whether the value is a valid Hugging Face repo ID."""
    try:
        validate_repo_id(value)
    except HFValidationError:
        return False
    return True


def _looks_like_local_model_path(value: str) -> bool:
    """Return whether the value clearly points to a local filesystem path."""
    if value.startswith(("/", "./", "../", "~/", ".\\", "..\\", "~\\", "\\\\", "//")):
        return True
    if value.endswith(("/", "\\")):
        return True
    if "\\" in value or _looks_like_windows_drive_path(value):
        return True
    return value.count("/") > 1


def _looks_like_windows_drive_path(value: str) -> bool:
    """Return whether the value looks like a Windows drive-qualified path."""
    return len(value) >= 2 and value[0].isalpha() and value[1] == ":"


def _normalize_exit_code(code: object) -> int:
    if isinstance(code, int):
        return code
    if code is None:
        return 0
    return 1
