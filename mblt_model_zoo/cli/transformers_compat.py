"""Compatibility helpers for delegating to the upstream Transformers CLI."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
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

CHAT_OPTIONS_WITH_REQUIRED_VALUE = frozenset(
    {
        "--model_name_or_path",
        "--model-name-or-path",
        "--model_revision",
        "--model-revision",
        "--user",
        "--system_prompt",
        "--system-prompt",
        "--save_folder",
        "--save-folder",
        "--examples_path",
        "--examples-path",
        "--generation_config",
        "--generation-config",
        "--device",
        "--dtype",
        "--torch_dtype",
        "--torch-dtype",
        "--attn_implementation",
        "--attn-implementation",
        "--bnb_4bit_quant_type",
        "--bnb-4bit-quant-type",
        "--host",
        "--port",
    }
)
CHAT_OPTIONS_WITH_OPTIONAL_VALUE = frozenset(
    {
        "--verbose",
        "--trust_remote_code",
        "--trust-remote-code",
        "--load_in_8bit",
        "--load-in-8bit",
        "--load_in_4bit",
        "--load-in-4bit",
        "--use_bnb_nested_quant",
        "--use-bnb-nested-quant",
    }
)
CHAT_MODEL_OPTIONS = frozenset({"--model_name_or_path", "--model-name-or-path"})
CHAT_MODEL_REVISION_OPTIONS = frozenset({"--model_revision", "--model-revision"})
CHAT_BOOLEAN_OPTION_VALUES = frozenset({"0", "1", "false", "true", "no", "yes", "off", "on"})


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
    _maybe_register_mobilint_chat_model(argv)
    if len(argv) > 1 and argv[1] == "serve":
        _install_transformers_serve_registration_hook()


def _should_register_mobilint_chat_model() -> bool:
    """Return whether delegated chat still needs local Mobilint registration."""
    return not _has_module("transformers.cli.chat")


def _maybe_register_mobilint_chat_model(argv: Sequence[str]) -> None:
    """Register local Mobilint chat models before delegating to the upstream CLI."""
    if len(argv) <= 2 or argv[1] != "chat":
        return
    if not _should_register_mobilint_chat_model():
        return

    model_name_or_path_or_address, is_remote, model_revision = _extract_chat_model_registration_target(argv[2:])
    if model_name_or_path_or_address is None or is_remote:
        return
    model_name_or_path_or_address, inline_model_revision = _split_model_id_and_revision(model_name_or_path_or_address)
    if model_revision is None:
        model_revision = inline_model_revision

    register_mobilint_models(
        SimpleNamespace(
            model_name_or_path_or_address=model_name_or_path_or_address,
            model_revision=model_revision,
        ),
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
    model_name_or_path_or_address, model_revision = _split_model_id_and_revision(model_id_and_revision)
    args = SimpleNamespace(
        model_name_or_path_or_address=model_name_or_path_or_address,
        model_revision=model_revision,
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
    """Split inline revisions only for clear Hub-style model identifiers."""
    model_name_or_path_or_address, separator, model_revision = model_id_and_revision.partition("@")
    if not separator:
        return model_name_or_path_or_address, None
    if not _looks_like_hub_model_id_with_revision(
        model_id_and_revision,
        model_name_or_path_or_address,
        model_revision,
    ):
        return model_id_and_revision, None
    return model_name_or_path_or_address, model_revision


def _looks_like_hub_model_id_with_revision(
    model_id_and_revision: str,
    model_name_or_path_or_address: str,
    model_revision: str,
) -> bool:
    """Return whether the value clearly matches Hub `model_id@revision` syntax."""
    if not model_name_or_path_or_address or not model_revision:
        return False
    if _looks_like_local_model_path(model_id_and_revision):
        return False
    if model_name_or_path_or_address.count("/") > 1:
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
    try:
        return Path(value).expanduser().exists()
    except (OSError, RuntimeError, ValueError):
        return False


def _looks_like_windows_drive_path(value: str) -> bool:
    """Return whether the value looks like a Windows drive-qualified path."""
    return len(value) >= 2 and value[0].isalpha() and value[1] == ":"


def _extract_chat_model_name(argv: Sequence[str]) -> str | None:
    return _extract_chat_model_name_and_remote(argv)[0]


def _extract_chat_model_name_and_remote(argv: Sequence[str]) -> tuple[str | None, bool]:
    """Return the chat model identifier and whether the command targets a remote endpoint."""
    model_name_or_path_or_address, is_remote, _ = _extract_chat_model_registration_target(argv)
    return model_name_or_path_or_address, is_remote


def _extract_chat_model_registration_target(argv: Sequence[str]) -> tuple[str | None, bool, str | None]:
    """Return the chat model identifier, remote-target flag, and requested revision."""
    index = 0
    model_name_or_path: str | None = None
    model_revision: str | None = None
    positional_tokens: list[str] = []
    ambiguous_option_value_tokens: list[str] = []

    while index < len(argv):
        token = argv[index]
        if token == "--":
            positional_tokens.extend(argv[index + 1 :])
            break
        if token.startswith("--"):
            option_name, separator, option_value = token.partition("=")
            if option_name in CHAT_MODEL_OPTIONS and separator:
                model_name_or_path = option_value
                index += 1
                continue
            if option_name in CHAT_MODEL_REVISION_OPTIONS and separator:
                model_revision = option_value
                index += 1
                continue
            if option_name in CHAT_OPTIONS_WITH_REQUIRED_VALUE and separator:
                if option_name in CHAT_MODEL_OPTIONS:
                    model_name_or_path = option_value
                elif option_name in CHAT_MODEL_REVISION_OPTIONS:
                    model_revision = option_value
                index += 1
                continue
            if option_name in CHAT_OPTIONS_WITH_REQUIRED_VALUE:
                if index + 1 >= len(argv):
                    break
                if option_name in CHAT_MODEL_OPTIONS:
                    model_name_or_path = argv[index + 1]
                elif option_name in CHAT_MODEL_REVISION_OPTIONS:
                    model_revision = argv[index + 1]
                index += 2
                continue
            if option_name in CHAT_OPTIONS_WITH_OPTIONAL_VALUE:
                if separator:
                    index += 1
                    continue
                if index + 1 < len(argv) and argv[index + 1].lower() in CHAT_BOOLEAN_OPTION_VALUES:
                    index += 2
                    continue
                index += 1
                continue
            if index + 1 < len(argv) and not argv[index + 1].startswith("-"):
                ambiguous_option_value_tokens.append(argv[index + 1])
                index += 2
                continue
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        positional_tokens.append(token)
        index += 1

    resolved_positionals = _resolve_chat_model_positionals(
        positional_tokens,
        ambiguous_option_value_tokens,
        model_name_or_path,
    )
    if resolved_positionals:
        model_name_or_path = resolved_positionals[0]

    is_remote = False
    if resolved_positionals:
        is_remote = _looks_like_remote_endpoint(resolved_positionals[0])
    if len(resolved_positionals) > 1 and _looks_like_remote_endpoint(resolved_positionals[1]):
        is_remote = True

    return model_name_or_path, is_remote, model_revision


def _resolve_chat_model_positionals(
    positional_tokens: Sequence[str],
    ambiguous_option_value_tokens: Sequence[str],
    explicit_model_name_or_path: str | None,
) -> list[str]:
    """Resolve positionals after filtering values that followed unknown long options."""
    if positional_tokens:
        if ambiguous_option_value_tokens and _should_prefix_ambiguous_chat_token(
            ambiguous_option_value_tokens[0],
            positional_tokens,
        ):
            return [ambiguous_option_value_tokens[0], *positional_tokens]
        return list(positional_tokens)
    if ambiguous_option_value_tokens and explicit_model_name_or_path is None:
        return [ambiguous_option_value_tokens[0]]
    return []


def _should_prefix_ambiguous_chat_token(ambiguous_token: str, positional_tokens: Sequence[str]) -> bool:
    """Return whether an ambiguous unknown-option value is more likely to be a positional chat token."""
    first_positional = positional_tokens[0]
    if _looks_like_remote_endpoint(ambiguous_token) or _looks_like_remote_endpoint(first_positional):
        return True
    return _looks_like_generation_flag(first_positional)


def _looks_like_generation_flag(value: str) -> bool:
    """Return whether a positional token matches the `key=value` generate-flags form."""
    return "=" in value and not value.startswith("=")


def _looks_like_remote_endpoint(value: str) -> bool:
    return value.startswith(("http://", "https://", "localhost"))


def _normalize_exit_code(code: object) -> int:
    if isinstance(code, int):
        return code
    if code is None:
        return 0
    return 1
