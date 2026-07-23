"""Tests for the standardized multi-model vision benchmark tools."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytest

from benchmark.vision import benchmark_vision_models, compare_benchmark_results
from mblt_model_zoo.vision.utils.evaluation import ImageNetResult


def test_benchmark_records_imagenet_metrics_in_primary_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use Top-1 as the score while retaining Top-5 in benchmark metrics."""

    class FakeModel:
        """Minimal classification model double."""

        post_cfg = {"task": "image_classification"}

    import mblt_model_zoo.vision.utils.evaluation as evaluation_module

    monkeypatch.setattr(
        evaluation_module,
        "eval_imagenet",
        lambda *args, **kwargs: ImageNetResult(top1=0.75, top5=0.95),
    )
    args = argparse.Namespace(
        task="image_classification",
        data_path=str(tmp_path),
        batch_size=1,
    )

    score, score_name, metrics = benchmark_vision_models._evaluate(FakeModel(), args, tmp_path)

    assert score == 0.75
    assert score_name == "top1_accuracy"
    assert metrics == {"top1_accuracy": 0.75, "top5_accuracy": 0.95}


def test_benchmark_continues_after_evaluator_type_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Record an unsupported evaluator output without aborting later targets."""

    class FakeEngine:
        """Minimal engine used to exercise per-target error handling."""

        def __init__(self, *, model_cls: str, **kwargs: object) -> None:
            self.model_cls = model_cls

        def dispose(self) -> None:
            """Release the fake benchmark engine."""

    def fake_evaluate(
        model: FakeEngine,
        args: object,
        run_dir: Path,
    ) -> tuple[float, str, dict[str, float]]:
        if model.model_cls == "invalid-output":
            raise TypeError("Unsupported model output")
        return 0.9, "top1_accuracy", {"top1_accuracy": 0.9}

    import mblt_model_zoo.vision as vision

    monkeypatch.setattr(vision, "MBLT_Engine", FakeEngine)
    monkeypatch.setattr(benchmark_vision_models, "_evaluate", fake_evaluate)

    result = benchmark_vision_models.main(
        [
            "--models",
            "invalid-output",
            "valid-output",
            "--task",
            "image_classification",
            "--data-path",
            str(tmp_path / "dataset"),
            "--results-dir",
            str(tmp_path / "results"),
            "--no-plot",
        ]
    )

    with (tmp_path / "results" / "results.csv").open(newline="", encoding="utf-8") as results_file:
        rows = list(csv.DictReader(results_file))

    assert result == 1
    assert [row["status"] for row in rows] == ["error", "ok"]
    assert rows[0]["error"] == "TypeError: Unsupported model output"


def test_comparison_rejects_matching_metrics_from_different_tasks(tmp_path: Path) -> None:
    """Reject task-incompatible inputs even when their score metric matches."""

    for name, task in (("detection", "object_detection"), ("segmentation", "instance_segmentation")):
        results_path = tmp_path / name / "results.csv"
        results_path.parent.mkdir()
        results_path.write_text(
            f"model,core_mode,task,status,score_name,score\nmodel-a,global8,{task},ok,map50_95,0.5\n",
            encoding="utf-8",
        )

    with pytest.raises(SystemExit, match="incompatible benchmark tasks"):
        compare_benchmark_results.main([str(tmp_path / "detection"), str(tmp_path / "segmentation")])


def test_onnx_benchmark_uses_one_neutral_runtime_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Avoid recording repeated NPU core-mode runs for ONNX inference."""

    captured_modes: list[str] = []

    def fake_run_target(model_name: str, core_mode: str, args: object, results_dir: Path) -> dict[str, object]:
        captured_modes.append(core_mode)
        return {
            "model": model_name,
            "core_mode": core_mode,
            "task": "image_classification",
            "batch_size": 1,
            "status": "ok",
            "score": 0.9,
            "score_name": "top1",
            "elapsed_s": 0.0,
        }

    monkeypatch.setattr(benchmark_vision_models, "_run_target", fake_run_target)

    assert (
        benchmark_vision_models.main(
            [
                "--models",
                "model-a",
                "--task",
                "image_classification",
                "--framework",
                "onnx",
                "--core-mode",
                "all",
                "--data-path",
                str(tmp_path / "dataset"),
                "--results-dir",
                str(tmp_path / "results"),
                "--no-plot",
            ]
        )
        == 0
    )
    assert captured_modes == ["onnx"]


def test_comparison_uses_result_directory_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Derive default chart paths and legend labels from result directories."""

    for name in ("baseline", "candidate"):
        results_path = tmp_path / name / "results.csv"
        results_path.parent.mkdir()
        results_path.write_text(
            "model,core_mode,task,status,score_name,score\nmodel-a,global8,object_detection,ok,map50_95,0.5\n",
            encoding="utf-8",
        )
    import benchmark.common.chart_utils as chart_utils

    captured: dict[str, object] = {}

    def fake_default_charts_dir(script_dir: Path, sources: list[Path], **kwargs: object) -> Path:
        captured["sources"] = sources
        return tmp_path / "charts"

    monkeypatch.setattr(chart_utils, "default_charts_dir", fake_default_charts_dir)
    monkeypatch.setattr(chart_utils, "plot_grouped_scalar_barh", lambda **kwargs: captured.update(kwargs))

    assert compare_benchmark_results.main([str(tmp_path / "baseline"), str(tmp_path / "candidate")]) == 0
    sources = captured["sources"]
    assert isinstance(sources, list)
    source_paths: list[Path] = []
    for source in sources:
        assert isinstance(source, Path)
        source_paths.append(source)
    assert [path.name for path in source_paths] == ["baseline", "candidate"]
    assert captured["group_labels"] == ["baseline", "candidate"]
