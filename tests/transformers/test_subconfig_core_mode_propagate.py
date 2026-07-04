"""Tests for base->subconfig core-mode propagation across CLI and benchmark entry points."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

_TRANSFORMERS_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "transformers"
if str(_TRANSFORMERS_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_TRANSFORMERS_BENCHMARK_DIR))

from benchmark.transformers import benchmark_automatic_speech_recognition_models as asr_bench  # noqa: E402
from benchmark.transformers import benchmark_image_text_to_text_models as vlm_bench  # noqa: E402
from mblt_model_zoo.cli import tps as tps_cli  # noqa: E402
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (  # noqa: E402
    apply_subconfig_core_mode_model_kwargs,
    coalesce_subconfig,
)


def test_coalesce_subconfig_prefers_override() -> None:
    """Verify explicit subconfig value wins over base fallback."""
    assert coalesce_subconfig("global4", "single") == "global4"


def test_coalesce_subconfig_falls_back_to_base_when_missing() -> None:
    """Verify base value is used when the subconfig value is None."""
    assert coalesce_subconfig(None, "single") == "single"


def test_coalesce_subconfig_keeps_none_when_both_missing() -> None:
    """Verify neither override nor base results in None."""
    assert coalesce_subconfig(None, None) is None


def test_apply_subconfig_core_mode_propagates_base_to_all_prefixes() -> None:
    """Verify base core mode fills in every subconfig when no override is set."""
    kwargs = apply_subconfig_core_mode_model_kwargs(
        {},
        ("vision", "text"),
        "single",
    )
    assert kwargs == {
        "vision_core_mode": "single",
        "vision_target_cores": ["0:0"],
        "text_core_mode": "single",
        "text_target_cores": ["0:0"],
    }


def test_apply_subconfig_core_mode_lets_one_subconfig_override() -> None:
    """Verify a per-subconfig core mode replaces the base value for that prefix only."""
    kwargs = apply_subconfig_core_mode_model_kwargs(
        {},
        ("vision", "text"),
        "single",
        subconfig_core_modes={"vision": "global8"},
    )
    assert kwargs == {
        "vision_core_mode": "global8",
        "vision_target_clusters": [0, 1],
        "text_core_mode": "single",
        "text_target_cores": ["0:0"],
    }


def test_apply_subconfig_core_mode_without_base_keeps_none() -> None:
    """Verify omitting base and subconfig values leaves everything unset."""
    kwargs = apply_subconfig_core_mode_model_kwargs(
        {},
        ("vision", "text"),
        None,
    )
    assert kwargs == {}


def test_apply_subconfig_core_mode_supports_encoder_decoder() -> None:
    """Verify encoder-decoder prefixes get their base fallback and overrides."""
    kwargs = apply_subconfig_core_mode_model_kwargs(
        {},
        ("encoder", "decoder"),
        "single",
        subconfig_core_modes={"decoder": "global4"},
    )
    assert kwargs["encoder_core_mode"] == "single"
    assert kwargs["decoder_core_mode"] == "global4"
    assert kwargs["encoder_target_cores"] == ["0:0"]
    assert kwargs["decoder_target_clusters"] == [0]


def test_apply_subconfig_core_mode_forwards_target_overrides() -> None:
    """Verify target-cores and target-clusters follow the same fallback rule."""
    kwargs = apply_subconfig_core_mode_model_kwargs(
        {},
        ("vision", "text"),
        "single",
        base_target_cores=["0:1"],
        subconfig_target_cores={"text": ["0:2"]},
    )
    assert kwargs["vision_target_cores"] == ["0:1"]
    assert kwargs["text_target_cores"] == ["0:2"]


def test_apply_subconfig_core_mode_forwards_mxq_paths() -> None:
    """Verify per-subconfig mxq paths override the base fallback."""
    kwargs = apply_subconfig_core_mode_model_kwargs(
        {},
        ("vision", "text"),
        None,
        base_mxq_path="/base.mxq",
        subconfig_mxq_paths={"vision": "/vision.mxq"},
    )
    assert kwargs["vision_mxq_path"] == "/vision.mxq"
    assert kwargs["text_mxq_path"] == "/base.mxq"


def test_vlm_benchmark_helper_uses_vision_override() -> None:
    """Verify VLM benchmark helper honors vision override with text fallback."""
    kwargs = vlm_bench._apply_vlm_core_mode_model_kwargs(
        {},
        "single",
        vision_core_mode="global8",
    )
    assert kwargs["vision_core_mode"] == "global8"
    assert kwargs["text_core_mode"] == "single"


def test_asr_benchmark_helper_uses_encoder_decoder_overrides() -> None:
    """Verify ASR encoder-decoder helper propagates base to unset prefixes."""
    kwargs = asr_bench._apply_asr_core_mode_model_kwargs(
        {},
        "openai/whisper-tiny",
        "single",
        encoder_core_mode="global8",
    )
    assert kwargs["encoder_core_mode"] == "global8"
    assert kwargs["decoder_core_mode"] == "single"


def test_asr_benchmark_helper_ignores_subconfig_for_non_encoder_decoder() -> None:
    """Verify non encoder-decoder ASR models keep top-level core-mode kwargs untouched."""
    kwargs = asr_bench._apply_asr_core_mode_model_kwargs(
        {},
        "some/plain-asr-model",
        "single",
        encoder_core_mode="global8",
    )
    assert kwargs.get("core_mode") == "single"
    assert "encoder_core_mode" not in kwargs


def test_asr_benchmark_parser_accepts_encoder_decoder_core_modes() -> None:
    """Verify the ASR benchmark CLI parses encoder/decoder core-mode options."""
    args = asr_bench._parse_args(["--core-mode", "single", "--encoder-core-mode", "global8"])
    assert args.encoder_core_mode == "global8"
    assert args.decoder_core_mode is None
    assert args.core_mode == "single"


def test_vlm_benchmark_parser_accepts_vision_text_core_modes() -> None:
    """Verify the VLM benchmark CLI parses vision/text core-mode options for measure and sweep."""
    parser = vlm_bench._build_arg_parser()
    for command in ("measure", "sweep"):
        args = parser.parse_args(
            [command, "--core-mode", "single", "--vision-core-mode", "global8"]
        )
        assert args.vision_core_mode == "global8"
        assert args.text_core_mode is None
        assert args.core_mode == "single"


def test_tps_cli_parser_accepts_vlm_subconfig_options() -> None:
    """Verify the TPS CLI parses vision/text subconfig options for measure and sweep."""
    from mblt_model_zoo.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "tps",
            "measure",
            "--task",
            "image-text-to-text",
            "--model",
            "mobilint/vlm",
            "--core-mode",
            "single",
            "--vision-core-mode",
            "global8",
            "--text-mxq-path",
            "/tmp/text.mxq",
        ]
    )
    assert args.vision_core_mode == "global8"
    assert args.text_core_mode is None
    assert args.core_mode == "single"
    assert args.text_mxq_path == "/tmp/text.mxq"


def test_tps_cli_build_pipeline_vlm_propagates_subconfig(monkeypatch) -> None:
    """Verify TPS `_build_pipeline` threads subconfig overrides for a VLM task."""
    captured: dict[str, object] = {}

    def _fake_pipeline(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model_kwargs=kwargs.get("model_kwargs", {}))

    monkeypatch.setattr(tps_cli, "_require_transformers_deps", lambda: None)
    monkeypatch.setattr(importlib.import_module("transformers"), "pipeline", _fake_pipeline)

    subconfig_options = tps_cli.SubconfigPipelineOptions(vision_core_mode="global8")

    tps_cli._build_pipeline(
        task="image-text-to-text",
        model="mobilint/vlm",
        tokenizer=None,
        device="cpu",
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
        eagle3_options=tps_cli.Eagle3PipelineOptions(),
        mxq_path=None,
        core_mode="single",
        target_cores=None,
        target_clusters=None,
        default_single_target_cores=("0:0",),
        subconfig_options=subconfig_options,
    )

    model_kwargs = captured["model_kwargs"]
    assert isinstance(model_kwargs, dict)
    assert model_kwargs["vision_core_mode"] == "global8"
    assert model_kwargs["text_core_mode"] == "single"
    assert model_kwargs["vision_target_clusters"] == [0, 1]
    assert model_kwargs["text_target_cores"] == ["0:0"]


def test_tps_cli_build_pipeline_vlm_base_only_propagates_to_both(monkeypatch) -> None:
    """Verify VLM base-only core mode is applied to both vision and text subconfigs."""
    captured: dict[str, object] = {}

    def _fake_pipeline(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model_kwargs=kwargs.get("model_kwargs", {}))

    monkeypatch.setattr(tps_cli, "_require_transformers_deps", lambda: None)
    monkeypatch.setattr(importlib.import_module("transformers"), "pipeline", _fake_pipeline)

    tps_cli._build_pipeline(
        task="image-text-to-text",
        model="mobilint/vlm",
        tokenizer=None,
        device="cpu",
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
        eagle3_options=tps_cli.Eagle3PipelineOptions(),
        mxq_path=None,
        core_mode="global4",
        target_cores=None,
        target_clusters=None,
        default_single_target_cores=("0:0",),
    )

    model_kwargs = captured["model_kwargs"]
    assert model_kwargs["vision_core_mode"] == "global4"
    assert model_kwargs["text_core_mode"] == "global4"
    assert model_kwargs["vision_target_clusters"] == [0]
    assert model_kwargs["text_target_clusters"] == [0]


def test_tps_cli_build_pipeline_vlm_subconfig_mxq_overrides_base_mxq(monkeypatch) -> None:
    """Verify per-subconfig mxq path overrides do not collide with base mxq_path."""
    captured: dict[str, object] = {}

    def _fake_pipeline(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model_kwargs=kwargs.get("model_kwargs", {}))

    monkeypatch.setattr(tps_cli, "_require_transformers_deps", lambda: None)
    monkeypatch.setattr(importlib.import_module("transformers"), "pipeline", _fake_pipeline)

    subconfig_options = tps_cli.SubconfigPipelineOptions(vision_mxq_path="/tmp/vision.mxq")

    tps_cli._build_pipeline(
        task="image-text-to-text",
        model="mobilint/vlm",
        tokenizer=None,
        device="cpu",
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
        eagle3_options=tps_cli.Eagle3PipelineOptions(),
        mxq_path="/tmp/global.mxq",
        core_mode="single",
        target_cores=None,
        target_clusters=None,
        default_single_target_cores=("0:0",),
        subconfig_options=subconfig_options,
    )

    model_kwargs = captured["model_kwargs"]
    assert model_kwargs["mxq_path"] == "/tmp/global.mxq"
    assert model_kwargs["vision_mxq_path"] == "/tmp/vision.mxq"
    assert "text_mxq_path" not in model_kwargs


def test_extract_subconfig_pipeline_kwargs_from_namespace() -> None:
    """Verify subconfig option extraction reads all recognized fields from CLI args."""
    import argparse

    args = argparse.Namespace(
        vision_core_mode="global8",
        text_core_mode=None,
        vision_target_cores=["0:0"],
        text_target_cores=None,
        vision_target_clusters=None,
        text_target_clusters=[1],
        vision_mxq_path=None,
        text_mxq_path="/tmp/text.mxq",
    )
    options = tps_cli._extract_subconfig_pipeline_kwargs(args)
    assert options.vision_core_mode == "global8"
    assert options.text_target_clusters == [1]
    assert options.text_mxq_path == "/tmp/text.mxq"
    assert options.vision_target_cores == ["0:0"]
