"""Guard: batched-MXQ TPS runs must use --core-mode single.

Mirrors the ``batch benchmark only supports --core-mode single`` enforcement
in ``benchmark/transformers/benchmark_text_generation_models.py`` and
``benchmark_image_text_to_text_models.py`` so a user running
``mblt-model-zoo tps ... --core-mode global8 --batch-size 16`` on a batched
MXQ receives the same friendly ``SystemExit`` instead of a low-level
backend error mid-launch (or silently wrong throughput).

Two paths keep the guard honest:

* **Pre-launch fast path** — a locally-resolvable ``--mxq-path`` (or
  ``--base-mxq-path`` / ``--text-mxq-path``) is probed via
  :func:`_probe_mxq_artifact_k` and rejected before pipeline construction.

* **Post-launch verification** — when no local artifact probe is available
  the pre-launch guard defers, and
  :func:`_verify_batched_mxq_core_mode_post_launch` reads the true
  ``k_per_model`` off the launched NPU backend and applies the same
  rejection logic. Under the sw-batch contract ``config.max_batch_size``
  is the aggregate ``N * K`` capacity and cannot classify ``K`` on its own,
  so a pre-launch config-only fallback used to misclassify both a ``K == 1``
  release with ``N > 1`` sw-batch (falsely rejected) and a ``K > 1`` release
  with a batch-1 config (silently allowed through).

Config resolution is stubbed via ``AutoConfig.from_pretrained`` so no real
MXQ or hardware is required.
"""

from __future__ import annotations

import argparse
import importlib
from types import SimpleNamespace

import pytest

from mblt_model_zoo.cli import tps as tps_cli
from mblt_model_zoo.cli.main import build_parser


class _StubConfig:
    """Minimal stand-in for a Transformers config object with an optional text_config."""

    def __init__(self, max_batch_size=None, text_config=None, vision_config=None):
        if max_batch_size is not None:
            self.max_batch_size = max_batch_size
        if text_config is not None:
            self.text_config = text_config
        if vision_config is not None:
            self.vision_config = vision_config


def _stub_autoconfig(monkeypatch, config) -> None:
    """Patch ``AutoConfig.from_pretrained`` so the guard sees ``config`` without any load."""
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


def _parse_tps_sweep(*extra: str) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(
        [
            "tps",
            "sweep",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct-Batch16",
            "--no-plot",
            *extra,
        ]
    )


def _stub_artifact_probe(monkeypatch, k_by_path: dict[str, int | None] | None = None) -> None:
    """Patch :func:`_probe_mxq_artifact_k` to return ``k`` for the given path.

    Bypasses ``qbruntime.Model`` entirely so no hardware or MXQ file is
    required. ``k_by_path=None`` (or an unknown path) returns ``None``,
    forcing the guard to defer to post-launch verification.
    """
    mapping = dict(k_by_path or {})
    monkeypatch.setattr(tps_cli, "_probe_mxq_artifact_k", lambda path: mapping.get(path))


# ---------------------------------------------------------------------------
# Pre-launch fast path: a locally resolvable ``--mxq-path`` override rejects
# (or auto-pins single) based on the artifact's compiled ``K``.
# ---------------------------------------------------------------------------


def test_measure_accepts_single_core_mode_on_batched_mxq(monkeypatch):
    """``--core-mode single`` short-circuits before any probe."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=16))

    args = _parse_tps_measure("--core-mode", "single", "--batch-size", "16")

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert args.core_mode == "single"
    assert not hasattr(args, "_batched_mxq_guard_ctx")


def test_measure_overrides_config_batch_when_mxq_path_probes_k1(monkeypatch):
    """A Batch16 release overridden with a locally-probed K=1 MXQ must not be rejected."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=16))
    _stub_artifact_probe(monkeypatch, {"/tmp/override-k1.mxq": 1})

    args = _parse_tps_measure("--core-mode", "global4", "--mxq-path", "/tmp/override-k1.mxq")

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert args.core_mode == "global4"
    assert not hasattr(args, "_batched_mxq_guard_ctx")


def test_measure_rejects_when_mxq_path_probes_k_gt_1_on_batch1_config(monkeypatch):
    """A batch-1 release overridden with a locally-probed K=4 MXQ must be rejected under global4."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=1))
    _stub_artifact_probe(monkeypatch, {"/tmp/override-k4.mxq": 4})

    args = _parse_tps_measure("--core-mode", "global4", "--mxq-path", "/tmp/override-k4.mxq")

    with pytest.raises(SystemExit) as excinfo:
        tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    message = str(excinfo.value)
    assert "batched MXQ only supports --core-mode single" in message
    assert "artifact K=4" in message
    assert "global4" in message


def test_measure_forces_single_when_unspecified_core_mode_on_probed_batched_artifact(monkeypatch):
    """Even without --core-mode, a probed K>1 override pins core_mode=single."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=1))
    _stub_artifact_probe(monkeypatch, {"/tmp/override-k8.mxq": 8})

    args = _parse_tps_measure("--mxq-path", "/tmp/override-k8.mxq")
    assert args.core_mode is None

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert args.core_mode == "single"
    assert not hasattr(args, "_batched_mxq_guard_ctx")


def test_measure_prefers_base_mxq_path_for_eagle3_over_bare_mxq_path(monkeypatch):
    """EAGLE-3 --base-mxq-path takes precedence: the base MXQ is the LLM that governs batching."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=1))
    _stub_artifact_probe(
        monkeypatch,
        {"/tmp/bare-k1.mxq": 1, "/tmp/base-k4.mxq": 4},
    )

    args = _parse_tps_measure(
        "--core-mode",
        "global4",
        "--mxq-path",
        "/tmp/bare-k1.mxq",
        "--base-mxq-path",
        "/tmp/base-k4.mxq",
    )

    with pytest.raises(SystemExit) as excinfo:
        tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert "artifact K=4" in str(excinfo.value)


def test_measure_vlm_prefers_text_mxq_path_over_bare_mxq_path(monkeypatch):
    """VLM --text-mxq-path takes precedence over --mxq-path for the batched-guard check."""
    _stub_autoconfig(monkeypatch, _StubConfig(text_config=_StubConfig(max_batch_size=1)))
    _stub_artifact_probe(
        monkeypatch,
        {"/tmp/bare.mxq": 8, "/tmp/text-k1.mxq": 1},
    )

    args = _parse_tps_measure(
        "--task",
        "image-text-to-text",
        "--core-mode",
        "global4",
        "--mxq-path",
        "/tmp/bare.mxq",
        "--text-mxq-path",
        "/tmp/text-k1.mxq",
    )

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert args.core_mode == "global4"
    assert not hasattr(args, "_batched_mxq_guard_ctx")


def test_select_llm_mxq_override_precedence():
    """Precedence: text_mxq_path (VLM) > base_mxq_path > mxq_path; None when no override."""
    parser = build_parser()

    # No override at all.
    args = parser.parse_args(["tps", "measure", "--model", "m"])
    assert tps_cli._select_llm_mxq_override(args) is None

    # Plain --mxq-path only.
    args = parser.parse_args(["tps", "measure", "--model", "m", "--mxq-path", "/a.mxq"])
    assert tps_cli._select_llm_mxq_override(args) == "/a.mxq"

    # --base-mxq-path wins over --mxq-path for EAGLE-3-shaped releases.
    args = parser.parse_args(["tps", "measure", "--model", "m", "--mxq-path", "/a.mxq", "--base-mxq-path", "/base.mxq"])
    assert tps_cli._select_llm_mxq_override(args) == "/base.mxq"

    # VLM: --text-mxq-path wins.
    args = parser.parse_args(
        [
            "tps",
            "measure",
            "--task",
            "image-text-to-text",
            "--model",
            "m",
            "--mxq-path",
            "/a.mxq",
            "--text-mxq-path",
            "/text.mxq",
        ]
    )
    assert tps_cli._select_llm_mxq_override(args) == "/text.mxq"


def test_probe_mxq_artifact_k_returns_none_for_non_local_path(tmp_path):
    """The probe short-circuits for a non-existent path so no ``qbruntime.Model`` is instantiated."""
    missing = tmp_path / "does_not_exist.mxq"
    assert tps_cli._probe_mxq_artifact_k(str(missing)) is None
    assert tps_cli._probe_mxq_artifact_k("") is None


# ---------------------------------------------------------------------------
# Deferral: without a locally-resolvable artifact probe, the pre-launch
# guard must NOT reject on config alone. It stashes a
# :class:`_BatchedMxqGuardContext` for the post-launch verifier to consume.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("core_mode", ["global4", "global8"])
def test_measure_defers_non_single_core_mode_when_no_local_override(monkeypatch, core_mode):
    """No --mxq-path override + batched config no longer rejects at CLI time (defers)."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=16))

    args = _parse_tps_measure("--core-mode", core_mode, "--batch-size", "16")

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert args.core_mode == core_mode
    ctx = getattr(args, "_batched_mxq_guard_ctx", None)
    assert ctx is not None
    assert ctx.effective_core_mode == core_mode
    assert ctx.flag_label == "--core-mode"
    assert ctx.args_attr == "core_mode"


def test_measure_defers_when_core_mode_unspecified_on_batched_mxq(monkeypatch):
    """Config-only knowledge cannot classify K under sw-batch — no auto-pin, defer instead."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=16))

    args = _parse_tps_measure("--batch-size", "16")
    assert args.core_mode is None

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    # Effective mode is None, so post-launch verifier will short-circuit; no auto-pin.
    assert args.core_mode is None


def test_measure_defers_on_non_batch_config(monkeypatch):
    """Non-batch config (max_batch_size == 1) with --core-mode global8 defers cleanly.

    Post-launch will observe ``k_per_model == 1`` and accept the mode — this is
    the sw-batch regression the pre-launch config fallback used to mishandle.
    """
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=1))

    args = _parse_tps_measure("--core-mode", "global8", "--batch-size", "4")

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert args.core_mode == "global8"


def test_measure_defers_when_config_missing_max_batch_size(monkeypatch):
    """A config without ``max_batch_size`` behaves like the deferral path — no CLI-time rejection."""
    _stub_autoconfig(monkeypatch, _StubConfig())

    args = _parse_tps_measure("--core-mode", "global4")

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert args.core_mode == "global4"


def test_measure_defers_when_autoconfig_fails(monkeypatch):
    """AutoConfig failures still fall through — pipeline construction surfaces any real error."""
    auto_config = importlib.import_module("transformers").AutoConfig

    def _fail(*a, **kw):
        raise OSError("simulated network failure")

    monkeypatch.setattr(auto_config, "from_pretrained", _fail)

    args = _parse_tps_measure("--core-mode", "global4")

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert args.core_mode == "global4"


def test_measure_defers_when_artifact_probe_returns_none(monkeypatch):
    """A non-local override falls through to post-launch verification, not the config fallback."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=16))
    _stub_artifact_probe(monkeypatch, {})  # every path returns None

    args = _parse_tps_measure("--core-mode", "global4", "--mxq-path", "hf://mobilint/Some-Non-Local")

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    ctx = getattr(args, "_batched_mxq_guard_ctx", None)
    assert ctx is not None
    assert ctx.effective_core_mode == "global4"


def test_sweep_defers_non_single_core_mode_when_no_local_override(monkeypatch):
    """``tps sweep`` uses the same guard; deferral applies to the sweep command as well."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=16))

    args = _parse_tps_sweep("--core-mode", "global8", "--batch-size", "16")

    # Do not run the full sweep; call the guard directly, matching how ``tps measure`` deferral is asserted.
    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert args.core_mode == "global8"
    assert getattr(args, "_batched_mxq_guard_ctx", None) is not None


def test_probe_config_max_batch_size_uses_task_specific_vlm_vision_config(monkeypatch):
    """The probe helper still forwards ``task`` to the candidate resolver (used by Qwen3-VL guard)."""
    config = _StubConfig(vision_config=_StubConfig(max_batch_size=2))
    _stub_autoconfig(monkeypatch, config)

    assert (
        tps_cli._probe_config_max_batch_size(
            "mobilint/Qwen3-VL-Batch2",
            trust_remote_code=True,
            revision=None,
            task="image-text-to-text",
        )
        == 2
    )


# ---------------------------------------------------------------------------
# Role-specific core-mode overrides: --text-core-mode (VLM) and
# --base-core-mode (EAGLE-3) get resolved to the appropriate LLM-role flag
# so the deferred guard context names the flag the user actually passed.
# ---------------------------------------------------------------------------


class _StubEagle3Config(_StubConfig):
    """Stub config that ``_is_eagle3_config`` classifies as an EAGLE-3 release."""

    model_type = "eagle3-qwen3"


def test_measure_vlm_defers_text_core_mode_when_bare_core_mode_unset(monkeypatch):
    """A batched text MXQ under --text-core-mode global4 defers with the role-specific flag label."""
    _stub_autoconfig(monkeypatch, _StubConfig(text_config=_StubConfig(max_batch_size=4)))

    args = _parse_tps_measure(
        "--task",
        "image-text-to-text",
        "--text-core-mode",
        "global4",
        "--batch-size",
        "4",
    )
    assert args.core_mode is None

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    ctx = getattr(args, "_batched_mxq_guard_ctx", None)
    assert ctx is not None
    assert ctx.effective_core_mode == "global4"
    assert ctx.flag_label == "--text-core-mode"
    assert ctx.args_attr == "text_core_mode"


def test_measure_eagle3_defers_base_core_mode_when_bare_core_mode_unset(monkeypatch):
    """A batched base MXQ under --base-core-mode global4 defers with the base flag label."""
    _stub_autoconfig(monkeypatch, _StubEagle3Config(max_batch_size=16))

    args = _parse_tps_measure("--base-core-mode", "global4", "--batch-size", "16")
    assert args.core_mode is None

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    ctx = getattr(args, "_batched_mxq_guard_ctx", None)
    assert ctx is not None
    assert ctx.effective_core_mode == "global4"
    assert ctx.flag_label == "--base-core-mode"
    assert ctx.args_attr == "base_core_mode"


def test_measure_vlm_allows_text_core_mode_on_non_batch_text_mxq_via_artifact_probe(monkeypatch):
    """VLM text MXQ probed to K==1 keeps its --text-core-mode global4 (fast-path regression)."""
    _stub_autoconfig(monkeypatch, _StubConfig(text_config=_StubConfig(max_batch_size=1)))
    _stub_artifact_probe(monkeypatch, {"/tmp/text-k1.mxq": 1})

    args = _parse_tps_measure(
        "--task",
        "image-text-to-text",
        "--text-core-mode",
        "global4",
        "--text-mxq-path",
        "/tmp/text-k1.mxq",
    )

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    assert args.text_core_mode == "global4"


def test_measure_eagle3_prefers_base_core_mode_over_shared_core_mode_via_artifact(monkeypatch):
    """When both --core-mode and --base-core-mode are set, the deferred ctx names --base-core-mode."""
    _stub_autoconfig(monkeypatch, _StubEagle3Config(max_batch_size=16))

    args = _parse_tps_measure(
        "--core-mode",
        "single",
        "--base-core-mode",
        "global4",
        "--batch-size",
        "16",
    )

    tps_cli._enforce_batched_mxq_core_mode_constraint(args)

    ctx = getattr(args, "_batched_mxq_guard_ctx", None)
    assert ctx is not None
    assert ctx.flag_label == "--base-core-mode"
    assert ctx.effective_core_mode == "global4"


# ---------------------------------------------------------------------------
# Post-launch verification: reads ``k_per_model`` off the LLM backend and
# hard-fails when the caller explicitly requested a non-single core mode.
# ---------------------------------------------------------------------------


class _FakeBackend:
    """Minimal ``MobilintNPUBackend`` stand-in exposing ``k_per_model`` and ``core_mode``."""

    def __init__(self, k: int, core_mode: str | None = None) -> None:
        self.k_per_model = k
        self.core_mode = core_mode


def _fake_pipeline_with_llm_backend(k: int, core_mode: str | None = None) -> SimpleNamespace:
    """Return a pipeline whose top-level ``.model.npu_backend`` mimics a plain LLM release."""
    model = SimpleNamespace(npu_backend=_FakeBackend(k, core_mode=core_mode))
    return SimpleNamespace(model=model)


def _fake_vlm_pipeline_with_text_backend(k: int, core_mode: str | None = None) -> SimpleNamespace:
    """Return a pipeline whose ``.model.model.language_model.npu_backend`` mimics Qwen3-VL."""
    language_model = SimpleNamespace(npu_backend=_FakeBackend(k, core_mode=core_mode))
    inner = SimpleNamespace(language_model=language_model)
    return SimpleNamespace(model=SimpleNamespace(model=inner))


def _fake_eagle3_pipeline_with_base_backend(k: int, core_mode: str | None = None) -> SimpleNamespace:
    """Return a pipeline whose ``.model.eagle3_base_model.npu_backend`` mimics an EAGLE-3 release."""
    eagle3_base_model = SimpleNamespace(npu_backend=_FakeBackend(k, core_mode=core_mode))
    return SimpleNamespace(model=SimpleNamespace(eagle3_base_model=eagle3_base_model))


def _fake_blip_pipeline_with_bert_backend(k: int, core_mode: str | None = None) -> SimpleNamespace:
    """Return a pipeline whose ``.model.text_decoder.bert.npu_backend`` mimics a BLIP release."""
    bert = SimpleNamespace(npu_backend=_FakeBackend(k, core_mode=core_mode))
    text_decoder = SimpleNamespace(bert=bert)
    return SimpleNamespace(model=SimpleNamespace(text_decoder=text_decoder))


def test_post_launch_rejects_k_gt_1_on_non_single_core_mode(monkeypatch):
    """K=4 backend under --core-mode global4 (deferred) raises the same friendly SystemExit."""
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=16))

    args = _parse_tps_measure("--core-mode", "global4", "--batch-size", "16")
    tps_cli._enforce_batched_mxq_core_mode_constraint(args)
    assert getattr(args, "_batched_mxq_guard_ctx", None) is not None

    pipeline = _fake_pipeline_with_llm_backend(k=4)
    with pytest.raises(SystemExit) as excinfo:
        tps_cli._verify_batched_mxq_core_mode_post_launch(pipeline, args)

    message = str(excinfo.value)
    assert "batched MXQ only supports --core-mode single" in message
    assert "artifact K=4" in message
    assert "--core-mode='global4'" in message


def test_post_launch_accepts_k_eq_1_under_non_single_core_mode_regression(monkeypatch):
    """K=1 (non-batch sw-batch release) under --core-mode global4 must NOT raise post-launch.

    This is the sw-batch regression the pre-launch config fallback used to
    mishandle: a ``K == 1`` release with ``config.max_batch_size = 8`` was
    treated as batched and rejected for ``global4``. Post-launch the true
    ``k_per_model = 1`` classifies the artifact correctly.
    """
    _stub_autoconfig(monkeypatch, _StubConfig(max_batch_size=8))

    args = _parse_tps_measure("--core-mode", "global4", "--batch-size", "8")
    tps_cli._enforce_batched_mxq_core_mode_constraint(args)
    assert getattr(args, "_batched_mxq_guard_ctx", None) is not None

    pipeline = _fake_pipeline_with_llm_backend(k=1)
    tps_cli._verify_batched_mxq_core_mode_post_launch(pipeline, args)


def test_post_launch_skips_when_no_deferred_context():
    """No ctx (single mode short-circuited or override probe already ran) => noop."""
    args = argparse.Namespace()
    pipeline = _fake_pipeline_with_llm_backend(k=4)

    tps_cli._verify_batched_mxq_core_mode_post_launch(pipeline, args)


def test_post_launch_skips_when_ctx_and_backend_both_lack_core_mode():
    """Deferred ctx with ``effective_core_mode=None`` and no backend ``core_mode`` is a noop.

    When the CLI omitted the role-specific flag and the loaded backend also
    fails to expose a resolvable ``core_mode`` string (unusual, but covers
    non-Mobilint-shaped stand-ins), the guard has no authoritative mode to
    check against and must fall through rather than false-positive.
    """
    args = argparse.Namespace()
    args._batched_mxq_guard_ctx = tps_cli._BatchedMxqGuardContext(
        effective_core_mode=None,
        flag_label="--core-mode",
        args_attr="core_mode",
        model="mobilint/example",
    )
    pipeline = _fake_pipeline_with_llm_backend(k=4)  # backend.core_mode defaults to None

    tps_cli._verify_batched_mxq_core_mode_post_launch(pipeline, args)


def test_post_launch_falls_back_to_backend_core_mode_when_ctx_effective_is_none():
    """CLI omitted --core-mode but the release ships global4 in config: reject at post-launch.

    Regression: PR #109 review comment r3793175258. The pre-launch guard
    resolves ``effective_core_mode`` from the CLI flags alone; when the
    user does not pass ``--core-mode``/``--text-core-mode``/``--base-core-mode``
    the deferred ctx carries ``effective_core_mode=None``. Previously the
    post-launch verifier returned early on that None, silently allowing a
    Qwen3-VL Batch16 release (``text_config.core_mode = 'global4'``) with a
    Hub-only ``--text-mxq-path`` override to run its batched (K>1) MXQ under
    global4. The fallback now reads the loaded backend's actual
    ``core_mode`` and applies the same rejection.
    """
    args = argparse.Namespace()
    args._batched_mxq_guard_ctx = tps_cli._BatchedMxqGuardContext(
        effective_core_mode=None,
        flag_label="--core-mode",
        args_attr="core_mode",
        model="mobilint/example",
    )
    pipeline = _fake_pipeline_with_llm_backend(k=4, core_mode="global4")

    with pytest.raises(SystemExit) as excinfo:
        tps_cli._verify_batched_mxq_core_mode_post_launch(pipeline, args)

    message = str(excinfo.value)
    assert "batched MXQ only supports --core-mode single" in message
    assert "artifact K=4" in message
    assert "backend core_mode='global4'" in message


def test_post_launch_accepts_backend_core_mode_single_when_ctx_effective_is_none():
    """CLI omitted --core-mode and the release ships ``single``: no rejection even at K>1.

    Safety guard for legitimately-single releases: when the fallback reads
    the backend's ``core_mode`` it must still short-circuit for the
    ``single`` case rather than treating "unspecified CLI" as an escalation.
    """
    args = argparse.Namespace()
    args._batched_mxq_guard_ctx = tps_cli._BatchedMxqGuardContext(
        effective_core_mode=None,
        flag_label="--core-mode",
        args_attr="core_mode",
        model="mobilint/example",
    )
    pipeline = _fake_pipeline_with_llm_backend(k=4, core_mode="single")

    tps_cli._verify_batched_mxq_core_mode_post_launch(pipeline, args)


def test_post_launch_skips_when_backend_missing():
    """A non-Mobilint pipeline (no ``npu_backend`` found) skips silently."""
    args = argparse.Namespace()
    args._batched_mxq_guard_ctx = tps_cli._BatchedMxqGuardContext(
        effective_core_mode="global4",
        flag_label="--core-mode",
        args_attr="core_mode",
        model="mobilint/example",
    )
    pipeline = SimpleNamespace(model=SimpleNamespace())  # no npu_backend anywhere

    tps_cli._verify_batched_mxq_core_mode_post_launch(pipeline, args)


def test_post_launch_reads_vlm_language_model_backend(monkeypatch):
    """VLM ctx resolves to ``pipeline.model.model.language_model.npu_backend`` (Qwen3-VL layout)."""
    _stub_autoconfig(monkeypatch, _StubConfig(text_config=_StubConfig(max_batch_size=1)))

    args = _parse_tps_measure(
        "--task",
        "image-text-to-text",
        "--text-core-mode",
        "global4",
        "--batch-size",
        "4",
    )
    tps_cli._enforce_batched_mxq_core_mode_constraint(args)
    assert getattr(args, "_batched_mxq_guard_ctx", None) is not None

    pipeline = _fake_vlm_pipeline_with_text_backend(k=4)
    with pytest.raises(SystemExit) as excinfo:
        tps_cli._verify_batched_mxq_core_mode_post_launch(pipeline, args)

    message = str(excinfo.value)
    assert "artifact K=4" in message
    assert "--text-core-mode='global4'" in message


def test_post_launch_reads_eagle3_base_backend(monkeypatch):
    """EAGLE-3 ctx resolves to ``pipeline.model.eagle3_base_model.npu_backend``."""
    _stub_autoconfig(monkeypatch, _StubEagle3Config(max_batch_size=16))

    args = _parse_tps_measure("--base-core-mode", "global4", "--batch-size", "16")
    tps_cli._enforce_batched_mxq_core_mode_constraint(args)
    assert getattr(args, "_batched_mxq_guard_ctx", None) is not None

    pipeline = _fake_eagle3_pipeline_with_base_backend(k=4)
    with pytest.raises(SystemExit) as excinfo:
        tps_cli._verify_batched_mxq_core_mode_post_launch(pipeline, args)

    message = str(excinfo.value)
    assert "artifact K=4" in message
    assert "--base-core-mode='global4'" in message


def test_resolve_llm_npu_backend_finds_blip_bert_backend():
    """BLIP's LLM lives at ``pipeline.model.text_decoder.bert.npu_backend``."""
    pipeline = _fake_blip_pipeline_with_bert_backend(k=2)

    backend = tps_cli._resolve_llm_npu_backend(pipeline.model)

    assert backend is not None
    assert backend.k_per_model == 2


def test_resolve_llm_npu_backend_prefers_eagle3_base_over_top_level():
    """When both ``eagle3_base_model.npu_backend`` and ``npu_backend`` exist, prefer the base."""
    top_level = _FakeBackend(k=1)
    base_backend = _FakeBackend(k=4)
    eagle3_base_model = SimpleNamespace(npu_backend=base_backend)
    model = SimpleNamespace(eagle3_base_model=eagle3_base_model, npu_backend=top_level)

    backend = tps_cli._resolve_llm_npu_backend(model)

    assert backend is base_backend


def test_resolve_llm_npu_backend_none_when_model_is_none():
    assert tps_cli._resolve_llm_npu_backend(None) is None
