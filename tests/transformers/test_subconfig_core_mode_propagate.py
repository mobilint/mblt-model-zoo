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


def test_append_asr_subconfig_core_mode_suffix_noop_without_overrides() -> None:
    """Verify no suffix is added when neither encoder nor decoder overrides are set."""
    label, base = asr_bench._append_asr_subconfig_core_mode_suffix(
        "openai/whisper-tiny-single",
        "openai__whisper-tiny-single",
        "single",
    )
    assert label == "openai/whisper-tiny-single"
    assert base == "openai__whisper-tiny-single"


def test_append_asr_subconfig_core_mode_suffix_noop_when_matches_base() -> None:
    """Verify a subconfig override that matches the base core mode adds no suffix."""
    label, base = asr_bench._append_asr_subconfig_core_mode_suffix(
        "openai/whisper-tiny-single",
        "openai__whisper-tiny-single",
        "single",
        encoder_core_mode="single",
        decoder_core_mode="single",
    )
    assert label == "openai/whisper-tiny-single"
    assert base == "openai__whisper-tiny-single"


def test_append_asr_subconfig_core_mode_suffix_encoder_only() -> None:
    """Verify encoder-only overrides append an `enc<mode>` suffix."""
    label, base = asr_bench._append_asr_subconfig_core_mode_suffix(
        "openai/whisper-tiny-single",
        "openai__whisper-tiny-single",
        "single",
        encoder_core_mode="global8",
    )
    assert label == "openai/whisper-tiny-single-encglobal8"
    assert base == "openai__whisper-tiny-single-encglobal8"


def test_append_asr_subconfig_core_mode_suffix_decoder_only() -> None:
    """Verify decoder-only overrides append a `dec<mode>` suffix."""
    label, base = asr_bench._append_asr_subconfig_core_mode_suffix(
        "openai/whisper-tiny-single",
        "openai__whisper-tiny-single",
        "single",
        decoder_core_mode="global4",
    )
    assert label == "openai/whisper-tiny-single-decglobal4"
    assert base == "openai__whisper-tiny-single-decglobal4"


def test_append_asr_subconfig_core_mode_suffix_encoder_and_decoder() -> None:
    """Verify both overrides combine into an `enc<mode>-dec<mode>` suffix."""
    label, base = asr_bench._append_asr_subconfig_core_mode_suffix(
        "openai/whisper-tiny-single",
        "openai__whisper-tiny-single",
        "single",
        encoder_core_mode="global8",
        decoder_core_mode="global4",
    )
    assert label == "openai/whisper-tiny-single-encglobal8-decglobal4"
    assert base == "openai__whisper-tiny-single-encglobal8-decglobal4"


def test_asr_build_run_targets_distinguishes_subconfig_variants() -> None:
    """Verify _build_run_targets writes distinct mode_base values per encoder/decoder combo."""
    common_flags = [
        "--model",
        "openai/whisper-tiny",
        "--core-mode",
        "single",
    ]

    default_base = asr_bench._build_run_targets(asr_bench._parse_args(common_flags))[0][3]
    encoder_override_base = asr_bench._build_run_targets(
        asr_bench._parse_args(common_flags + ["--encoder-core-mode", "global8"])
    )[0][3]
    decoder_override_base = asr_bench._build_run_targets(
        asr_bench._parse_args(common_flags + ["--decoder-core-mode", "global4"])
    )[0][3]
    both_override_base = asr_bench._build_run_targets(
        asr_bench._parse_args(
            common_flags
            + ["--encoder-core-mode", "global8", "--decoder-core-mode", "global4"]
        )
    )[0][3]

    assert default_base.endswith("-single")
    assert encoder_override_base.endswith("-single-encglobal8")
    assert decoder_override_base.endswith("-single-decglobal4")
    assert both_override_base.endswith("-single-encglobal8-decglobal4")
    assert len({default_base, encoder_override_base, decoder_override_base, both_override_base}) == 4


def test_asr_build_run_targets_ignores_subconfig_for_non_encoder_decoder_models() -> None:
    """Verify non encoder-decoder ASR models keep the current mode_base layout."""
    args = asr_bench._parse_args(
        [
            "--model",
            "facebook/wav2vec2-base-960h",
            "--core-mode",
            "single",
            "--encoder-core-mode",
            "global8",
        ]
    )
    mode_base = asr_bench._build_run_targets(args)[0][3]

    assert mode_base.endswith("-single")
    assert "enc" not in mode_base.split("-single", 1)[1]


def test_asr_write_target_json_records_subconfig_core_modes(tmp_path) -> None:
    """Verify _write_target_json records encoder/decoder core-mode fields in the payload."""
    import json

    target = asr_bench.ASRBenchmarkTarget(
        model_id="openai/whisper-tiny",
        revision_candidates=[None],
        label="openai/whisper-tiny",
        base="openai__whisper-tiny",
        mxq_path=None,
        is_original=False,
    )
    args = asr_bench._parse_args(
        [
            "--num-beams",
            "1",
            "--encoder-core-mode",
            "global8",
            "--decoder-core-mode",
            "global4",
        ]
    )
    summary = asr_bench.ASRMetricSummary(
        num_samples=1,
        total_audio_s=1.0,
        total_generate_s=0.5,
        wer=0.0,
        cer=0.0,
        mean_latency_s=0.5,
        p50_latency_s=0.5,
        p95_latency_s=0.5,
        throughput_samples_per_s=2.0,
        rtf=0.5,
        inverse_rtf=2.0,
        decode_tokens_per_s=10.0,
        avg_tokens_per_sample=5.0,
    )
    timing = asr_bench.SampleTiming(
        sample_id="s1",
        audio_duration_s=1.0,
        generate_time_s=0.5,
        num_generated_tokens=5,
        num_beams=1,
        reference="hello",
        hypothesis="hello",
        effective_generate_kwargs={"num_beams": 1},
    )
    out_path = tmp_path / "whisper-tiny-single-encglobal8-decglobal4_beams1.json"

    asr_bench._write_target_json(
        out_path,
        target=target,
        label="openai/whisper-tiny-single-encglobal8-decglobal4",
        args=args,
        revision=None,
        core_mode="single",
        summary=summary,
        device_metric={},
        device_trace={},
        sample_timings=[timing],
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["core_mode"] == "single"
    assert payload["encoder_core_mode"] == "global8"
    assert payload["decoder_core_mode"] == "global4"


def test_resolve_asr_subconfig_core_modes_drops_overrides_for_original_native_runs() -> None:
    """Verify --original-models without --mxq-dir yields (None, None) for subconfig core modes."""
    args = asr_bench._parse_args(
        [
            "--model",
            "openai/whisper-tiny",
            "--original-models",
            "--core-mode",
            "single",
            "--encoder-core-mode",
            "global8",
            "--decoder-core-mode",
            "global4",
        ]
    )
    assert asr_bench._resolve_asr_subconfig_core_modes(args) == (None, None)


def test_resolve_asr_subconfig_core_modes_keeps_overrides_with_mxq_dir(tmp_path) -> None:
    """Verify --original-models WITH --mxq-dir still keeps encoder/decoder overrides."""
    args = asr_bench._parse_args(
        [
            "--original-models",
            "--mxq-dir",
            str(tmp_path),
            "--encoder-core-mode",
            "global8",
            "--decoder-core-mode",
            "global4",
        ]
    )
    assert asr_bench._resolve_asr_subconfig_core_modes(args) == ("global8", "global4")


def test_resolve_asr_subconfig_core_modes_keeps_overrides_without_original_models() -> None:
    """Verify overrides pass through when --original-models is not set."""
    args = asr_bench._parse_args(
        [
            "--model",
            "openai/whisper-tiny",
            "--encoder-core-mode",
            "global8",
            "--decoder-core-mode",
            "global4",
        ]
    )
    assert asr_bench._resolve_asr_subconfig_core_modes(args) == ("global8", "global4")


def test_asr_build_run_targets_drops_subconfig_suffix_for_original_native(monkeypatch) -> None:
    """Verify --original-models without --mxq-dir strips enc/dec suffixes from filenames.

    The pipeline is called with encoder/decoder_core_mode=None in this case, so the
    generated filename must not claim overrides that never reached the model.
    """
    monkeypatch.setattr(asr_bench, "_resolve_original_model_ids", lambda model_ids: list(model_ids))
    args = asr_bench._parse_args(
        [
            "--model",
            "openai/whisper-tiny",
            "--original-models",
            "--encoder-core-mode",
            "global8",
            "--decoder-core-mode",
            "global4",
        ]
    )
    mode_base = asr_bench._build_run_targets(args)[0][3]
    assert "-enc" not in mode_base
    assert "-dec" not in mode_base


def test_asr_write_target_json_drops_subconfig_core_modes_for_original_native(tmp_path) -> None:
    """Verify _write_target_json records None for encoder/decoder core-mode on native original runs."""
    import json

    target = asr_bench.ASRBenchmarkTarget(
        model_id="openai/whisper-tiny",
        revision_candidates=[None],
        label="openai/whisper-tiny",
        base="openai__whisper-tiny",
        mxq_path=None,
        is_original=True,
    )
    args = asr_bench._parse_args(
        [
            "--model",
            "openai/whisper-tiny",
            "--original-models",
            "--num-beams",
            "1",
            "--encoder-core-mode",
            "global8",
            "--decoder-core-mode",
            "global4",
        ]
    )
    summary = asr_bench.ASRMetricSummary(
        num_samples=1,
        total_audio_s=1.0,
        total_generate_s=0.5,
        wer=0.0,
        cer=0.0,
        mean_latency_s=0.5,
        p50_latency_s=0.5,
        p95_latency_s=0.5,
        throughput_samples_per_s=2.0,
        rtf=0.5,
        inverse_rtf=2.0,
        decode_tokens_per_s=10.0,
        avg_tokens_per_sample=5.0,
    )
    timing = asr_bench.SampleTiming(
        sample_id="s1",
        audio_duration_s=1.0,
        generate_time_s=0.5,
        num_generated_tokens=5,
        num_beams=1,
        reference="hello",
        hypothesis="hello",
        effective_generate_kwargs={"num_beams": 1},
    )
    out_path = tmp_path / "whisper-tiny_beams1.json"

    asr_bench._write_target_json(
        out_path,
        target=target,
        label="openai/whisper-tiny",
        args=args,
        revision=None,
        core_mode=None,
        summary=summary,
        device_metric={},
        device_trace={},
        sample_timings=[timing],
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["core_mode"] is None
    assert payload["encoder_core_mode"] is None
    assert payload["decoder_core_mode"] is None


def test_build_asr_pipeline_no_subconfig_kwargs_when_disabled(monkeypatch) -> None:
    """Verify _build_asr_pipeline with encoder/decoder=None doesn't leak subconfig kwargs.

    Mirrors what ``main()`` passes when the shared --core-mode is ignored for
    original-model native runs.
    """
    captured: dict[str, object] = {}

    def _fake_pipeline(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model=types.SimpleNamespace(generation_config=None))

    import transformers  # local import to avoid module-import time cost

    monkeypatch.setattr(transformers, "pipeline", _fake_pipeline)

    target = asr_bench.ASRBenchmarkTarget(
        model_id="openai/whisper-tiny",
        revision_candidates=[None],
        label="openai/whisper-tiny",
        base="openai__whisper-tiny",
        mxq_path=None,
        is_original=True,
    )
    asr_bench._build_asr_pipeline(
        target,
        revision=None,
        device=None,
        device_map=None,
        dtype=None,
        trust_remote_code=True,
        core_mode=None,
        native_generate_kwargs=None,
        encoder_core_mode=None,
        decoder_core_mode=None,
    )

    model_kwargs = captured.get("model_kwargs", {})
    assert "encoder_core_mode" not in model_kwargs
    assert "decoder_core_mode" not in model_kwargs
    assert "encoder_target_cores" not in model_kwargs
    assert "decoder_target_cores" not in model_kwargs
    assert "encoder_target_clusters" not in model_kwargs
    assert "decoder_target_clusters" not in model_kwargs
