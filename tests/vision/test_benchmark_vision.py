"""Tests for the standardized multi-model vision benchmark tools."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from benchmark.vision import benchmark_vision_models, compare_benchmark_results


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
