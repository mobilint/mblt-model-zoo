"""Regression tests for Transformers CLI compatibility glue."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import pytest

from mblt_model_zoo.cli import transformers_compat

cli_main_module = importlib.import_module("mblt_model_zoo.cli.main")


def test_main_delegates_transformers_cli_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the delegated exit code for upstream Transformers commands."""
    monkeypatch.setattr(cli_main_module, "is_transformers_cli_command", lambda argv: True)
    monkeypatch.setattr(cli_main_module, "dispatch_transformers_cli", lambda argv: 7)
    monkeypatch.setattr(
        cli_main_module,
        "build_parser",
        lambda: pytest.fail("build_parser should not be called for delegated Transformers commands"),
    )
    monkeypatch.setattr("sys.argv", ["mblt-model-zoo", "chat", "--help"])

    assert cli_main_module.main() == 7


def test_registers_mobilint_chat_model_for_local_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register Mobilint chat models before delegating local chat commands."""
    calls: list[str] = []

    def _fake_register(args: Any, transformers_module: Any) -> None:
        calls.append(args.model_name_or_path_or_address)

    monkeypatch.setattr(transformers_compat, "register_mobilint_models", _fake_register)

    transformers_compat._maybe_register_mobilint_chat_model(
        ["mblt-model-zoo", "chat", "mobilint/Llama-3.2-1B-Instruct"]
    )

    assert calls == ["mobilint/Llama-3.2-1B-Instruct"]


@pytest.mark.parametrize(
    ("argv", "expected_model"),
    [
        (
            ["mblt-model-zoo", "chat", "--user", "alice", "mobilint/Llama-3.2-1B-Instruct"],
            "mobilint/Llama-3.2-1B-Instruct",
        ),
        (
            ["mblt-model-zoo", "chat", "--user=alice", "mobilint/Llama-3.2-1B-Instruct"],
            "mobilint/Llama-3.2-1B-Instruct",
        ),
        (
            ["mblt-model-zoo", "chat", "--dtype", "float16", "mobilint/Llama-3.2-1B-Instruct"],
            "mobilint/Llama-3.2-1B-Instruct",
        ),
        (
            ["mblt-model-zoo", "chat", "--model-name-or-path", "mobilint/Llama-3.2-1B-Instruct"],
            "mobilint/Llama-3.2-1B-Instruct",
        ),
        (
            [
                "mblt-model-zoo",
                "chat",
                "--model-name-or-path",
                "ignored/by-positional",
                "mobilint/Llama-3.2-1B-Instruct",
            ],
            "mobilint/Llama-3.2-1B-Instruct",
        ),
        (
            [
                "mblt-model-zoo",
                "chat",
                "--model-name-or-path=mobilint/Llama-3.2-1B-Instruct",
            ],
            "mobilint/Llama-3.2-1B-Instruct",
        ),
    ],
)
def test_extract_chat_model_name_skips_option_values(argv: list[str], expected_model: str) -> None:
    """Ignore chat option values while preserving explicit model options as fallback."""
    assert transformers_compat._extract_chat_model_name(argv[2:]) == expected_model


@pytest.mark.parametrize(
    "argv",
    [
        ["mblt-model-zoo", "chat", "http://localhost:8000", "mobilint/Llama-3.2-1B-Instruct"],
        ["mblt-model-zoo", "chat", "https://example.invalid/v1", "mobilint/Llama-3.2-1B-Instruct"],
        ["mblt-model-zoo", "chat", "localhost:8000", "mobilint/Llama-3.2-1B-Instruct"],
    ],
)
def test_skips_mobilint_chat_registration_for_remote_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    """Avoid local model registration when chat connects to a remote endpoint."""
    calls: list[str] = []

    def _fake_register(args: Any, transformers_module: Any) -> None:
        calls.append(args.model_name_or_path_or_address)

    monkeypatch.setattr(transformers_compat, "register_mobilint_models", _fake_register)

    transformers_compat._maybe_register_mobilint_chat_model(argv)

    assert calls == []


@pytest.mark.parametrize(
    "argv",
    [
        ["mblt-model-zoo", "chat", "gpt-4o", "https://api.openai.com/v1"],
        ["mblt-model-zoo", "chat", "mobilint/Llama-3.2-1B-Instruct", "https://api.openai.com/v1"],
    ],
)
def test_skips_mobilint_chat_registration_for_v5_remote_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    """Avoid local model registration when chat uses the v5 model-id then remote-endpoint flow."""
    calls: list[str] = []

    def _fake_register(args: Any, transformers_module: Any) -> None:
        calls.append(args.model_name_or_path_or_address)

    monkeypatch.setattr(transformers_compat, "register_mobilint_models", _fake_register)

    transformers_compat._maybe_register_mobilint_chat_model(argv)

    assert calls == []


def test_install_transformers_serve_registration_hook_wraps_loader_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Register Mobilint serve models on demand without double-wrapping the loader."""

    class _FakeServe:
        _mblt_registration_hook_installed = False

        def _load_model_and_data_processor(self, model_id_and_revision: str) -> tuple[str, str]:
            return ("loaded", model_id_and_revision)

    fake_serve_module = ModuleType("transformers.cli.serve")
    fake_serve_module.Serve = _FakeServe

    calls: list[tuple[str, str | None]] = []

    def _fake_register(args: Any, transformers_module: Any) -> None:
        calls.append((args.model_name_or_path_or_address, getattr(args, "model_revision", None)))

    monkeypatch.setattr(transformers_compat, "register_mobilint_models", _fake_register)
    monkeypatch.setattr(
        transformers_compat,
        "_has_module",
        lambda module_name: module_name == "transformers.cli.serve",
    )
    monkeypatch.setattr(
        transformers_compat.importlib,
        "import_module",
        lambda module_name: fake_serve_module,
    )

    transformers_compat._install_transformers_serve_registration_hook()
    wrapped_loader = _FakeServe._load_model_and_data_processor
    transformers_compat._install_transformers_serve_registration_hook()

    service = _FakeServe()
    result = service._load_model_and_data_processor("mobilint/Llama-3.2-1B-Instruct@main")

    assert wrapped_loader is _FakeServe._load_model_and_data_processor
    assert calls == [("mobilint/Llama-3.2-1B-Instruct", "main")]
    assert result == ("loaded", "mobilint/Llama-3.2-1B-Instruct@main")


@pytest.mark.parametrize(
    ("argv", "expect_hook"),
    [
        (["mblt-model-zoo", "chat", "mobilint/Llama-3.2-1B-Instruct"], False),
        (["mblt-model-zoo", "env"], False),
        (["mblt-model-zoo", "version"], False),
        (["mblt-model-zoo", "serve", "mobilint/Llama-3.2-1B-Instruct"], True),
    ],
)
def test_prepare_transformers_cli_installs_serve_hook_only_for_serve(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expect_hook: bool,
) -> None:
    """Only touch the serving stack when delegating the serve subcommand."""
    install_calls = 0

    def _fake_install_hook() -> None:
        nonlocal install_calls
        install_calls += 1

    monkeypatch.setattr(transformers_compat, "_maybe_register_mobilint_chat_model", lambda argv: None)
    monkeypatch.setattr(transformers_compat, "_install_transformers_serve_registration_hook", _fake_install_hook)

    transformers_compat._prepare_transformers_cli(argv)

    assert install_calls == int(expect_hook)


def test_install_transformers_serve_registration_hook_registers_separate_serve_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Register Mobilint architectures on both top-level and serve-local Transformers modules."""

    class _FakeServe:
        _mblt_registration_hook_installed = False

        def _load_model_and_data_processor(self, model_id_and_revision: str) -> tuple[str, str]:
            return ("loaded", model_id_and_revision)

    fake_serve_module = ModuleType("transformers.commands.serving")
    fake_serve_module.ServeCommand = _FakeServe
    fake_serve_module.transformers = object()

    calls: list[tuple[str, str | None, Any]] = []

    def _fake_register(args: Any, transformers_module: Any) -> None:
        calls.append((args.model_name_or_path_or_address, getattr(args, "model_revision", None), transformers_module))

    monkeypatch.setattr(transformers_compat, "register_mobilint_models", _fake_register)
    monkeypatch.setattr(
        transformers_compat,
        "_has_module",
        lambda module_name: module_name == "transformers.commands.serving",
    )
    monkeypatch.setattr(
        transformers_compat.importlib,
        "import_module",
        lambda module_name: fake_serve_module,
    )

    transformers_compat._install_transformers_serve_registration_hook()
    service = _FakeServe()
    result = service._load_model_and_data_processor("mobilint/Llama-3.2-1B-Instruct@main")

    assert result == ("loaded", "mobilint/Llama-3.2-1B-Instruct@main")
    assert calls == [
        ("mobilint/Llama-3.2-1B-Instruct", "main", transformers_compat.transformers),
        ("mobilint/Llama-3.2-1B-Instruct", "main", fake_serve_module.transformers),
    ]


def test_install_transformers_serve_registration_hook_wraps_v55_model_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Register Mobilint architectures for the Transformers 5.5+ serving stack."""

    class _FakeServe:
        pass

    class _FakeModelManager:
        _mblt_registration_hook_installed_for_load_model_and_processor = False

        def load_model_and_processor(self, model_id_and_revision: str, *args, **kwargs) -> tuple[str, str]:
            return ("loaded", model_id_and_revision)

    fake_serve_module = ModuleType("transformers.cli.serve")
    fake_serve_module.Serve = _FakeServe
    fake_model_manager_module = ModuleType("transformers.cli.serving.model_manager")
    fake_model_manager_module.ModelManager = _FakeModelManager
    fake_model_manager_module.transformers = object()

    calls: list[tuple[str, str | None, Any]] = []

    def _fake_register(args: Any, transformers_module: Any) -> None:
        calls.append((args.model_name_or_path_or_address, getattr(args, "model_revision", None), transformers_module))

    monkeypatch.setattr(transformers_compat, "register_mobilint_models", _fake_register)
    monkeypatch.setattr(
        transformers_compat,
        "_has_module",
        lambda module_name: module_name in {"transformers.cli.serve", "transformers.cli.serving.model_manager"},
    )

    def _fake_import_module(module_name: str) -> ModuleType:
        if module_name == "transformers.cli.serve":
            return fake_serve_module
        if module_name == "transformers.cli.serving.model_manager":
            return fake_model_manager_module
        raise AssertionError(f"unexpected module import: {module_name}")

    monkeypatch.setattr(transformers_compat.importlib, "import_module", _fake_import_module)

    transformers_compat._install_transformers_serve_registration_hook()
    manager = _FakeModelManager()
    result = manager.load_model_and_processor("mobilint/Llama-3.2-1B-Instruct@main")

    assert result == ("loaded", "mobilint/Llama-3.2-1B-Instruct@main")
    assert calls == [
        ("mobilint/Llama-3.2-1B-Instruct", "main", transformers_compat.transformers),
        ("mobilint/Llama-3.2-1B-Instruct", "main", fake_model_manager_module.transformers),
    ]


def test_register_mobilint_model_for_modules_preserves_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the requested revision when registering serve models."""
    calls: list[tuple[str, str | None, Any]] = []
    extra_transformers = object()

    def _fake_register(args: Any, transformers_module: Any) -> None:
        calls.append((args.model_name_or_path_or_address, getattr(args, "model_revision", None), transformers_module))

    monkeypatch.setattr(transformers_compat, "register_mobilint_models", _fake_register)

    transformers_compat._register_mobilint_model_for_modules("mobilint/demo-model@dev", extra_transformers)

    assert calls == [
        ("mobilint/demo-model", "dev", transformers_compat.transformers),
        ("mobilint/demo-model", "dev", extra_transformers),
    ]


def test_dispatch_transformers_cli_prefers_v5_entrypoint_and_restores_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the v5 CLI module when available and restore sys.argv afterwards."""
    seen_argv: list[str] = []
    fake_cli_module = ModuleType("transformers.cli.transformers")

    def _fake_main() -> None:
        import sys

        seen_argv.extend(sys.argv)
        raise SystemExit(3)

    fake_cli_module.main = _fake_main  # type: ignore[attr-defined]

    monkeypatch.setattr(transformers_compat, "_prepare_transformers_cli", lambda argv: None)
    monkeypatch.setattr(
        transformers_compat,
        "_has_module",
        lambda module_name: module_name == "transformers.cli.transformers",
    )
    monkeypatch.setattr(
        transformers_compat.importlib,
        "import_module",
        lambda module_name: fake_cli_module,
    )
    monkeypatch.setattr("sys.argv", ["python", "-m", "pytest"])

    exit_code = transformers_compat.dispatch_transformers_cli(["mblt-model-zoo", "chat", "--help"])

    assert exit_code == 3
    assert seen_argv == ["mblt-model-zoo", "chat", "--help"]
    assert __import__("sys").argv == ["python", "-m", "pytest"]
