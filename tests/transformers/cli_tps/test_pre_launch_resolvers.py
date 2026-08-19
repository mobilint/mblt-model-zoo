"""Pre-launch batch-size and core-mode resolvers for the TPS CLI.

Every ``args.batch_size`` / ``args.core_mode`` read in
``mblt_model_zoo/cli/tps.py`` routes through one of four canonical resolvers:

* :func:`_resolve_effective_batch_size_pre_launch` — before pipeline
  construction (CLI → config → 1).
* :func:`_resolve_effective_core_mode_pre_launch` — before pipeline
  construction (CLI → config → None).
* :func:`_resolve_cli_batch_size` — post-launch measurement.
* :func:`_resolve_effective_qwen3_vl_batch` — post-launch Qwen3-VL guard.

These tests cover the pre-launch pair and the ``_default_single_target_cores_for_args``
consumer that was the site of the original "K=1 MXQ with config.max_batch_size > 1
pinned all slots to ``0:0`` -> Model_NotAlive" bug (PR review r3813611879),
plus regression coverage that the two post-launch resolvers were not touched
and a static grep guard that no fresh direct read of ``args.batch_size`` or
``args.core_mode`` has landed outside these resolver bodies.
"""

from __future__ import annotations

import argparse
import ast
import importlib
from pathlib import Path

from mblt_model_zoo.cli import tps as tps_cli
from mblt_model_zoo.cli.main import build_parser


class _StubConfig:
    """Minimal stand-in for a Transformers config object with optional sub-configs."""

    def __init__(self, max_batch_size=None, core_mode=None, text_config=None, vision_config=None):
        if max_batch_size is not None:
            self.max_batch_size = max_batch_size
        if core_mode is not None:
            self.core_mode = core_mode
        if text_config is not None:
            self.text_config = text_config
        if vision_config is not None:
            self.vision_config = vision_config


def _stub_autoconfig(monkeypatch, config) -> None:
    """Patch ``AutoConfig.from_pretrained`` so resolvers see ``config`` without any load."""
    auto_config = importlib.import_module("transformers").AutoConfig
    monkeypatch.setattr(auto_config, "from_pretrained", lambda *a, **kw: config)


def _parse_tps_measure(*extra: str) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(
        [
            "tps",
            "measure",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct-Batch16",
            *extra,
        ]
    )


def _parse_tps_measure_vlm(*extra: str) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(
        [
            "tps",
            "measure",
            "--task",
            "image-text-to-text",
            "--model",
            "mobilint/Qwen3-VL-Batch16",
            *extra,
        ]
    )


# ---------------------------------------------------------------------------
# _resolve_effective_batch_size_pre_launch: priority CLI -> config -> 1.
# ---------------------------------------------------------------------------


def test_batch_resolver_returns_one_when_no_signals(monkeypatch):
    """(a) args.batch_size=None, config.max_batch_size=1 -> resolver returns 1."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=1))

    args = _parse_tps_measure()

    assert (
        tps_cli._resolve_effective_batch_size_pre_launch(
            args,
            model=args.model,
            task=args.task,
            trust_remote_code=args.trust_remote_code,
            revision=None,
        )
        == 1
    )


def test_batch_resolver_reads_config_when_cli_unset(monkeypatch):
    """(b) args.batch_size=None, config.max_batch_size=16 -> resolver returns 16 (TODAY's fix)."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=16))

    args = _parse_tps_measure()

    assert (
        tps_cli._resolve_effective_batch_size_pre_launch(
            args,
            model=args.model,
            task=args.task,
            trust_remote_code=args.trust_remote_code,
            revision=None,
        )
        == 16
    )


def test_batch_resolver_cli_wins_over_config(monkeypatch):
    """(c) args.batch_size=32, config.max_batch_size=1 -> resolver returns 32 (CLI wins)."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=1))

    args = _parse_tps_measure("--batch-size", "32")

    assert (
        tps_cli._resolve_effective_batch_size_pre_launch(
            args,
            model=args.model,
            task=args.task,
            trust_remote_code=args.trust_remote_code,
            revision=None,
        )
        == 32
    )


def test_batch_resolver_cli_batch_1_pins_scalar_even_when_config_is_batched(monkeypatch):
    """(d) args.batch_size=1, config.max_batch_size=16 -> resolver returns 1 (explicit scalar wins)."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=16))

    args = _parse_tps_measure("--batch-size", "1")

    assert (
        tps_cli._resolve_effective_batch_size_pre_launch(
            args,
            model=args.model,
            task=args.task,
            trust_remote_code=args.trust_remote_code,
            revision=None,
        )
        == 1
    )


def test_batch_resolver_reads_vlm_text_subconfig(monkeypatch):
    """(e) VLM task with config.text_config.max_batch_size -> resolver walks sub-config."""
    _stub_autoconfig(monkeypatch, _StubConfig(text_config=_StubConfig(max_batch_size=8)))

    args = _parse_tps_measure_vlm()

    assert (
        tps_cli._resolve_effective_batch_size_pre_launch(
            args,
            model=args.model,
            task=args.task,
            trust_remote_code=args.trust_remote_code,
            revision=None,
        )
        == 8
    )


def test_batch_resolver_returns_one_when_autoconfig_fails(monkeypatch):
    """A config-load failure falls through to 1 without raising (fault-tolerance)."""
    auto_config = importlib.import_module("transformers").AutoConfig

    def _fail(*a, **kw):
        raise OSError("simulated offline")

    monkeypatch.setattr(auto_config, "from_pretrained", _fail)

    args = _parse_tps_measure()

    assert (
        tps_cli._resolve_effective_batch_size_pre_launch(
            args,
            model=args.model,
            task=args.task,
            trust_remote_code=args.trust_remote_code,
            revision=None,
        )
        == 1
    )


def test_batch_resolver_returns_one_when_model_missing():
    """No model string -> skip the config probe and return 1."""
    args = _parse_tps_measure()

    assert (
        tps_cli._resolve_effective_batch_size_pre_launch(
            args,
            model=None,
            task=args.task,
            trust_remote_code=args.trust_remote_code,
            revision=None,
        )
        == 1
    )


# ---------------------------------------------------------------------------
# _default_single_target_cores_for_args: consumer of the batch resolver.
# The whole point of this refactor — see PR review r3813611879.
# ---------------------------------------------------------------------------


def test_default_single_target_cores_returns_00_when_no_signals(monkeypatch):
    """(a) No batching -> pin ("0:0",) so single-mode has a default target."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=1))

    args = _parse_tps_measure()

    assert tps_cli._default_single_target_cores_for_args(args) == ("0:0",)


def test_default_single_target_cores_skips_00_on_config_batch_gt_1(monkeypatch):
    """(b) TODAY's bug: config.max_batch_size=16 without --batch-size must skip the "0:0" pin.

    Under sw-batch a K=1 MXQ with config.max_batch_size=16 fans out to N=16
    slots; pinning every slot to "0:0" collapses them onto one core and
    triggers Model_NotAlive at launch. The pre-launch batch resolver must
    detect this and return None so qbruntime uses every available core.
    """
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=16))

    args = _parse_tps_measure()

    assert tps_cli._default_single_target_cores_for_args(args) is None


def test_default_single_target_cores_skips_00_on_cli_batch_gt_1(monkeypatch):
    """(c) --batch-size 32 with config.max_batch_size=1 still skips the "0:0" pin (regression)."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=1))

    args = _parse_tps_measure("--batch-size", "32")

    assert tps_cli._default_single_target_cores_for_args(args) is None


def test_default_single_target_cores_returns_00_when_cli_batch_1_over_config(monkeypatch):
    """(d) --batch-size 1 wins over config.max_batch_size=16 -> pin ("0:0",)."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=16))

    args = _parse_tps_measure("--batch-size", "1")

    assert tps_cli._default_single_target_cores_for_args(args) == ("0:0",)


def test_default_single_target_cores_skips_00_on_dev_no_list(monkeypatch):
    """List-shaped --dev-no still skips the sentinel to avoid device-set mismatch warnings."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=1))

    args = _parse_tps_measure("--dev-no", "0,1")

    assert tps_cli._default_single_target_cores_for_args(args) is None


# ---------------------------------------------------------------------------
# _resolve_effective_core_mode_pre_launch: priority CLI -> config -> None.
# ---------------------------------------------------------------------------


def test_core_mode_resolver_returns_none_when_no_signals(monkeypatch):
    """No CLI flag and no config core_mode -> return None (caller decides default)."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=1))

    args = _parse_tps_measure()

    assert (
        tps_cli._resolve_effective_core_mode_pre_launch(
            args,
            model=args.model,
            task=args.task,
            trust_remote_code=args.trust_remote_code,
            revision=None,
        )
        is None
    )


def test_core_mode_resolver_cli_wins_over_config(monkeypatch):
    """--core-mode single overrides even a config that ships global4."""
    _stub_autoconfig(monkeypatch, _StubConfig(core_mode="global4"))

    args = _parse_tps_measure("--core-mode", "single")

    assert (
        tps_cli._resolve_effective_core_mode_pre_launch(
            args,
            model=args.model,
            task=args.task,
            trust_remote_code=args.trust_remote_code,
            revision=None,
        )
        == "single"
    )


def test_core_mode_resolver_reads_config_when_cli_unset(monkeypatch):
    """CLI unset -> pick up config.core_mode."""
    _stub_autoconfig(monkeypatch, _StubConfig(core_mode="global4"))

    args = _parse_tps_measure()

    assert (
        tps_cli._resolve_effective_core_mode_pre_launch(
            args,
            model=args.model,
            task=args.task,
            trust_remote_code=args.trust_remote_code,
            revision=None,
        )
        == "global4"
    )


def test_core_mode_resolver_reads_vlm_text_subconfig(monkeypatch):
    """VLM task walks text_config.core_mode when top-level is absent (Qwen3-VL Batch16)."""
    _stub_autoconfig(monkeypatch, _StubConfig(text_config=_StubConfig(core_mode="global4")))

    args = _parse_tps_measure_vlm()

    assert (
        tps_cli._resolve_effective_core_mode_pre_launch(
            args,
            model=args.model,
            task=args.task,
            trust_remote_code=args.trust_remote_code,
            revision=None,
        )
        == "global4"
    )


def test_core_mode_resolver_returns_none_when_autoconfig_fails(monkeypatch):
    """A config-load failure returns None (matches _probe_config_max_batch_size discipline)."""
    auto_config = importlib.import_module("transformers").AutoConfig

    def _fail(*a, **kw):
        raise OSError("simulated offline")

    monkeypatch.setattr(auto_config, "from_pretrained", _fail)

    args = _parse_tps_measure()

    assert (
        tps_cli._resolve_effective_core_mode_pre_launch(
            args,
            model=args.model,
            task=args.task,
            trust_remote_code=args.trust_remote_code,
            revision=None,
        )
        is None
    )


# ---------------------------------------------------------------------------
# _resolve_effective_llm_core_mode: base-flag fallback now routes through
# the pre-launch resolver so a config-declared core_mode surfaces at
# pre-launch with an honest ``flag_label``.
# ---------------------------------------------------------------------------


def test_llm_core_mode_resolver_flag_label_is_config_when_cli_unset(monkeypatch):
    """No CLI flag + config.core_mode='global4' -> flag_label='release config core_mode'."""
    _stub_autoconfig(monkeypatch, _StubConfig(core_mode="global4"))

    args = _parse_tps_measure()

    effective, flag_label, args_attr = tps_cli._resolve_effective_llm_core_mode(args, is_eagle3=False)

    assert effective == "global4"
    assert flag_label == "release config core_mode"
    assert args_attr == "core_mode"


def test_llm_core_mode_resolver_flag_label_is_dashcore_when_cli_set(monkeypatch):
    """--core-mode set -> flag_label='--core-mode' regardless of config."""
    _stub_autoconfig(monkeypatch, _StubConfig(core_mode="global4"))

    args = _parse_tps_measure("--core-mode", "global8")

    effective, flag_label, args_attr = tps_cli._resolve_effective_llm_core_mode(args, is_eagle3=False)

    assert effective == "global8"
    assert flag_label == "--core-mode"
    assert args_attr == "core_mode"


# ---------------------------------------------------------------------------
# Regression: existing post-launch resolvers keep their previous priority.
# The refactor added two new pre-launch resolvers; it must not have changed
# the two post-launch helpers' behavior.
# ---------------------------------------------------------------------------


class _FakePipelineWithMaxBatch:
    """Return a pipeline whose model.config.max_batch_size resolves to ``value``."""

    def __init__(self, value):
        cfg = _StubConfig(max_batch_size=value) if value is not None else _StubConfig()
        self.model = argparse.Namespace(config=cfg)


def test_resolve_cli_batch_size_unchanged_cli_wins():
    """Post-launch resolver: explicit --batch-size overrides model config."""
    args = _parse_tps_measure("--batch-size", "8")
    pipeline = _FakePipelineWithMaxBatch(4)

    assert tps_cli._resolve_cli_batch_size(args, pipeline) == 8


def test_resolve_cli_batch_size_unchanged_config_fallback():
    """Post-launch resolver: when CLI is unset, fall back to pipeline model.config."""
    args = _parse_tps_measure()
    pipeline = _FakePipelineWithMaxBatch(4)

    assert tps_cli._resolve_cli_batch_size(args, pipeline) == 4


def test_resolve_effective_qwen3_vl_batch_unchanged_backend_wins():
    """Qwen3-VL post-launch resolver: backend.max_batch_size beats pipeline config."""
    args = _parse_tps_measure_vlm()
    backend = argparse.Namespace(max_batch_size=16)
    pipeline = _FakePipelineWithMaxBatch(4)

    assert tps_cli._resolve_effective_qwen3_vl_batch(args, pipeline, backend) == 16


def test_resolve_effective_qwen3_vl_batch_unchanged_cli_wins():
    """Qwen3-VL post-launch resolver: --batch-size beats backend and config."""
    args = _parse_tps_measure_vlm("--batch-size", "2")
    backend = argparse.Namespace(max_batch_size=16)
    pipeline = _FakePipelineWithMaxBatch(4)

    assert tps_cli._resolve_effective_qwen3_vl_batch(args, pipeline, backend) == 2


# ---------------------------------------------------------------------------
# Static grep guard: nobody has landed a fresh CLI-only reader outside the
# resolver bodies. This closes the cycle — if a fourth-repeat "add config
# fallback" comment can even be typed, this test must have been silenced.
# ---------------------------------------------------------------------------


_TPS_PATH = Path(tps_cli.__file__)

# Resolver bodies and raw pass-through call sites that intentionally read
# ``args.batch_size`` or ``args.core_mode``. Every occurrence must be one of
# these; a fresh reader added outside these functions represents a new
# CLI-only site and must fail this guard.
_ALLOWED_READERS = frozenset(
    {
        "_resolve_cli_batch_size",
        "_resolve_effective_qwen3_vl_batch",
        "_resolve_effective_batch_size_pre_launch",
        "_resolve_effective_core_mode_pre_launch",
        "_resolve_effective_llm_core_mode",
        # Raw pass-through: kwarg-only reads passed straight into _build_pipeline
        # so the Mobilint config layer normalizes downstream.
        "_run_text_measure",
        "_run_vlm_measure",
        "_run_text_sweep",
        "_run_vlm_sweep",
    }
)

_TRACKED_ATTRS = frozenset({"batch_size", "core_mode"})


class _AttrReaderCollector(ast.NodeVisitor):
    """Collect real-code reads of ``args.batch_size`` / ``args.core_mode`` and their enclosing def."""

    def __init__(self) -> None:
        self.function_stack: list[str] = []
        self.offenders: list[tuple[int, str, str, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 - stdlib API
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802 - stdlib API
        self._visit_function(node)

    def _visit_function(self, node) -> None:
        self.function_stack.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self.function_stack.pop()

    def _current_def(self) -> str:
        return self.function_stack[-1] if self.function_stack else "<module>"

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802 - stdlib API
        # Match ``args.batch_size`` / ``args.core_mode`` (Name -> Attribute).
        if isinstance(node.value, ast.Name) and node.value.id == "args" and node.attr in _TRACKED_ATTRS:
            self.offenders.append((node.lineno, self._current_def(), "args." + node.attr, "attribute"))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - stdlib API
        # Match ``getattr(args, "batch_size", ...)`` / ``getattr(args, "core_mode", ...)``.
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "args"
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and node.args[1].value in _TRACKED_ATTRS
        ):
            self.offenders.append(
                (node.lineno, self._current_def(), f"getattr(args, {node.args[1].value!r}, ...)", "getattr")
            )
        self.generic_visit(node)


def test_no_direct_args_reads_outside_resolvers():
    """Static guard: every ``args.batch_size`` / ``args.core_mode`` read lives in an allowed function.

    Rationale: three "CLI-only reader missing config aggregate" review
    comments landed in the same week for tps.py before this refactor. The
    canonical way to resolve either signal is now the pre-launch resolvers
    or the pre-existing post-launch resolvers listed in ``_ALLOWED_READERS``.
    Anything else — a fresh direct read — is exactly the class of bug this
    refactor was designed to prevent. If this test fails, add the new call
    site to ``_ALLOWED_READERS`` only after confirming it truly is a raw
    pass-through with no CLI-vs-config decision to make; otherwise route
    the read through one of the resolvers instead.

    Uses ``ast`` so docstring and comment mentions of ``args.batch_size`` do
    not trip the guard.
    """
    source = _TPS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    collector = _AttrReaderCollector()
    collector.visit(tree)

    offenders = [
        (lineno, def_name, expr)
        for lineno, def_name, expr, _ in collector.offenders
        if def_name not in _ALLOWED_READERS
    ]

    assert not offenders, (
        "Direct args.batch_size/args.core_mode reads outside the allowed "
        "resolver / raw-pass-through sites:\n"
        + "\n".join(f"  L{lineno} in {name}: {expr}" for lineno, name, expr in offenders)
    )
