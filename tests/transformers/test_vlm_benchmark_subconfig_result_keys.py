"""Tests for VLM benchmark result-key identity when vision/text subconfig core modes differ."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

_TRANSFORMERS_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "transformers"
if str(_TRANSFORMERS_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_TRANSFORMERS_BENCHMARK_DIR))

from benchmark.transformers import benchmark_image_text_to_text_models as vlm_bench  # noqa: E402
from benchmark.transformers.benchmark_text_generation_models import (  # noqa: E402
    TextBenchmarkTarget,
)


def test_append_vlm_subconfig_core_mode_suffix_noop_without_overrides() -> None:
    """Verify no suffix is added when neither vision nor text overrides are set."""
    label, base = vlm_bench._append_vlm_subconfig_core_mode_suffix(
        "Qwen/Qwen2-VL-2B-single",
        "Qwen__Qwen2-VL-2B-single",
        "single",
    )
    assert label == "Qwen/Qwen2-VL-2B-single"
    assert base == "Qwen__Qwen2-VL-2B-single"


def test_append_vlm_subconfig_core_mode_suffix_noop_when_matches_base() -> None:
    """Verify a subconfig override that matches the base core mode adds no suffix."""
    label, base = vlm_bench._append_vlm_subconfig_core_mode_suffix(
        "Qwen/Qwen2-VL-2B-single",
        "Qwen__Qwen2-VL-2B-single",
        "single",
        vision_core_mode="single",
        text_core_mode="single",
    )
    assert label == "Qwen/Qwen2-VL-2B-single"
    assert base == "Qwen__Qwen2-VL-2B-single"


def test_append_vlm_subconfig_core_mode_suffix_vision_only() -> None:
    """Verify vision-only overrides append a `vis<mode>` suffix."""
    label, base = vlm_bench._append_vlm_subconfig_core_mode_suffix(
        "Qwen/Qwen2-VL-2B-single",
        "Qwen__Qwen2-VL-2B-single",
        "single",
        vision_core_mode="global8",
    )
    assert label == "Qwen/Qwen2-VL-2B-single-visglobal8"
    assert base == "Qwen__Qwen2-VL-2B-single-visglobal8"


def test_append_vlm_subconfig_core_mode_suffix_text_only() -> None:
    """Verify text-only overrides append a `txt<mode>` suffix."""
    label, base = vlm_bench._append_vlm_subconfig_core_mode_suffix(
        "Qwen/Qwen2-VL-2B-single",
        "Qwen__Qwen2-VL-2B-single",
        "single",
        text_core_mode="global4",
    )
    assert label == "Qwen/Qwen2-VL-2B-single-txtglobal4"
    assert base == "Qwen__Qwen2-VL-2B-single-txtglobal4"


def test_append_vlm_subconfig_core_mode_suffix_vision_and_text() -> None:
    """Verify both overrides combine into a `vis<mode>-txt<mode>` suffix."""
    label, base = vlm_bench._append_vlm_subconfig_core_mode_suffix(
        "Qwen/Qwen2-VL-2B-single",
        "Qwen__Qwen2-VL-2B-single",
        "single",
        vision_core_mode="global8",
        text_core_mode="global4",
    )
    assert label == "Qwen/Qwen2-VL-2B-single-visglobal8-txtglobal4"
    assert base == "Qwen__Qwen2-VL-2B-single-visglobal8-txtglobal4"


def test_append_vlm_subconfig_core_mode_suffix_handles_none_base() -> None:
    """Verify overrides still apply when the base core_mode is None (e.g., --original-models)."""
    label, base = vlm_bench._append_vlm_subconfig_core_mode_suffix(
        "Qwen/Qwen2-VL-2B",
        "Qwen__Qwen2-VL-2B",
        None,
        vision_core_mode="global8",
    )
    assert label == "Qwen/Qwen2-VL-2B-visglobal8"
    assert base == "Qwen__Qwen2-VL-2B-visglobal8"


def test_vlm_subconfig_core_mode_payload_fields_default() -> None:
    """Verify payload fields are None when subconfig overrides are absent."""
    args = argparse.Namespace(vision_core_mode=None, text_core_mode=None)
    fields = vlm_bench._vlm_subconfig_core_mode_payload_fields(args, "single")
    assert fields == {
        "core_mode": "single",
        "vision_core_mode": None,
        "text_core_mode": None,
    }


def test_vlm_subconfig_core_mode_payload_fields_with_overrides() -> None:
    """Verify payload fields capture the effective per-subconfig core modes."""
    args = argparse.Namespace(vision_core_mode="global8", text_core_mode="global4")
    fields = vlm_bench._vlm_subconfig_core_mode_payload_fields(args, "single")
    assert fields == {
        "core_mode": "single",
        "vision_core_mode": "global8",
        "text_core_mode": "global4",
    }


def test_vlm_subconfig_core_mode_payload_fields_missing_attrs() -> None:
    """Verify the helper tolerates argparse namespaces missing vision/text attrs."""
    args = argparse.Namespace()
    fields = vlm_bench._vlm_subconfig_core_mode_payload_fields(args, None)
    assert fields == {
        "core_mode": None,
        "vision_core_mode": None,
        "text_core_mode": None,
    }


def _make_text_target(model_id: str = "Qwen/Qwen2-VL-2B") -> TextBenchmarkTarget:
    return TextBenchmarkTarget(
        model_id=model_id,
        revision_candidates=[None],
        label=model_id,
        base=model_id.replace("/", "__"),
        mxq_path=None,
        max_batch_size=1,
        batch_mode="non_batch",
    )


def _mode_bases_for(
    monkeypatch,
    *,
    core_mode: str,
    vision_core_mode: str | None = None,
    text_core_mode: str | None = None,
) -> list[str]:
    """Drive `_collect_vlm_run_targets` with stubbed deps and return the mode_bases."""

    text_target = _make_text_target()
    monkeypatch.setattr(
        vlm_bench,
        "_iter_targets",
        lambda model_ids, revision, all_revisions: [
            (text_target.model_id, None, text_target.label, text_target.base)
        ],
    )
    monkeypatch.setattr(
        vlm_bench,
        "_filter_text_targets_by_batch_mode",
        lambda raw_targets, *, batch_mode=None, task=None: [text_target],
    )
    monkeypatch.setattr(
        vlm_bench,
        "_iter_core_modes_for_target",
        lambda args, batch_mode, disable_npu_specific_args=False: [core_mode],
    )
    args = argparse.Namespace(
        models=[text_target.model_id],
        mxq_dir=None,
        mxq_path=None,
        revision=None,
        all=False,
        original_models=False,
        batch_mode="non_batch",
        core_mode=core_mode,
        vision_core_mode=vision_core_mode,
        text_core_mode=text_core_mode,
        output_dir=None,
    )
    _, _, run_targets = vlm_bench._collect_vlm_run_targets(args)
    # run_targets tuple layout: (model_id, revision, mode_label, mode_base, mxq_path, core_mode, ...)
    return [rt[3] for rt in run_targets]


def test_collect_vlm_run_targets_distinguishes_subconfig_variants(monkeypatch, tmp_path) -> None:
    """Verify vision/text overrides yield distinct mode_base filenames per combo."""
    monkeypatch.chdir(tmp_path)

    default_base = _mode_bases_for(monkeypatch, core_mode="single")[0]
    vision_base = _mode_bases_for(
        monkeypatch, core_mode="single", vision_core_mode="global8"
    )[0]
    text_base = _mode_bases_for(
        monkeypatch, core_mode="single", text_core_mode="global4"
    )[0]
    both_base = _mode_bases_for(
        monkeypatch,
        core_mode="single",
        vision_core_mode="global8",
        text_core_mode="global4",
    )[0]
    matching_base = _mode_bases_for(
        monkeypatch,
        core_mode="single",
        vision_core_mode="single",
        text_core_mode="single",
    )[0]

    assert default_base.endswith("-single")
    assert vision_base.endswith("-single-visglobal8")
    assert text_base.endswith("-single-txtglobal4")
    assert both_base.endswith("-single-visglobal8-txtglobal4")
    assert matching_base == default_base
    assert len({default_base, vision_base, text_base, both_base}) == 4


def _make_vlm_pipeline_args(
    *,
    original_models: bool,
    mxq_dir: str | None,
    vision_core_mode: str | None,
    text_core_mode: str | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        trust_remote_code=True,
        tokenizer=None,
        device=None,
        device_map=None,
        dtype=None,
        original_models=original_models,
        mxq_dir=mxq_dir,
        vision_core_mode=vision_core_mode,
        text_core_mode=text_core_mode,
    )


def _capture_build_pipeline_kwargs(
    monkeypatch,
    args: argparse.Namespace,
    *,
    core_mode: str | None,
) -> dict[str, object]:
    captured: dict[str, object] = {}

    def _fake_pipeline(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(model_kwargs=kwargs.get("model_kwargs", {}))

    monkeypatch.setattr(vlm_bench, "hf_pipeline", _fake_pipeline)
    vlm_bench._build_pipeline(
        args,
        "mobilint/Qwen2-VL-2B",
        None,
        None,
        core_mode,
    )
    return captured


def test_build_pipeline_original_models_drops_subconfig_core_modes(monkeypatch) -> None:
    """Verify --original-models without --mxq-dir strips vision/text overrides from model_kwargs."""
    args = _make_vlm_pipeline_args(
        original_models=True,
        mxq_dir=None,
        vision_core_mode="global8",
        text_core_mode="global4",
    )
    captured = _capture_build_pipeline_kwargs(monkeypatch, args, core_mode=None)
    model_kwargs = captured.get("model_kwargs", {})
    assert "vision_core_mode" not in model_kwargs
    assert "text_core_mode" not in model_kwargs
    assert "vision_target_cores" not in model_kwargs
    assert "vision_target_clusters" not in model_kwargs
    assert "text_target_cores" not in model_kwargs
    assert "text_target_clusters" not in model_kwargs


def test_build_pipeline_original_models_with_mxq_dir_keeps_subconfig_core_modes(monkeypatch, tmp_path) -> None:
    """Verify --original-models WITH --mxq-dir still applies vision/text overrides."""
    args = _make_vlm_pipeline_args(
        original_models=True,
        mxq_dir=str(tmp_path),
        vision_core_mode="global8",
        text_core_mode="global4",
    )
    captured = _capture_build_pipeline_kwargs(monkeypatch, args, core_mode="single")
    model_kwargs = captured.get("model_kwargs", {})
    assert model_kwargs["vision_core_mode"] == "global8"
    assert model_kwargs["text_core_mode"] == "global4"


def test_build_pipeline_non_original_keeps_subconfig_core_modes(monkeypatch) -> None:
    """Verify overrides still reach model_kwargs when --original-models is not set."""
    args = _make_vlm_pipeline_args(
        original_models=False,
        mxq_dir=None,
        vision_core_mode="global8",
        text_core_mode="global4",
    )
    captured = _capture_build_pipeline_kwargs(monkeypatch, args, core_mode="single")
    model_kwargs = captured.get("model_kwargs", {})
    assert model_kwargs["vision_core_mode"] == "global8"
    assert model_kwargs["text_core_mode"] == "global4"
