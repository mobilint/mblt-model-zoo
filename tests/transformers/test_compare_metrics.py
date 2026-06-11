from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from benchmark.transformers.compare_metrics import (
    TASK_REGISTRY,
    ASRCompareMetric,
    LLMCompareMetric,
    VLMCompareMetric,
    collect_metrics,
    common_model_ids,
    normalize_model_key,
    render_charts,
)


def test_llm_compare_metric_from_payload() -> None:
    payload = {
        "model": "group__repo/model-a",
        "benchmark": {
            "prefill_sweep": {
                "x_values": [128, 256],
                "tps_values": [10.0, 20.0],
                "time_values": [1.2, 2.3],
            },
            "decode_sweep": {
                "x_values": [128, 256],
                "tps_values": [30.0, 40.0],
                "time_values": [3.4, 4.5],
            },
        },
        "device": {
            "avg_power_w": 11.0,
            "prefill_tok_per_j_last": 1.5,
            "decode_tok_per_j_last": 2.5,
        },
    }
    metric = LLMCompareMetric.from_payload(payload)
    assert metric is not None
    assert metric.prefill_tps == {128: 10.0, 256: 20.0}
    assert metric.decode_tps == {128: 30.0, 256: 40.0}
    assert metric.prefill_latency_ms == {128: 1200.0, 256: 2300.0}
    assert metric.avg_power_w == 11.0
    assert metric.prefill_tokens_per_j == 1.5


def test_vlm_compare_metric_from_payload() -> None:
    payload = {
        "model": "repo/vlm-a",
        "benchmark": {
            "llm_results": {
                "summary": {
                    "llm_prefill_tps": {"mean": 12.0},
                    "llm_decode_tps": {"mean": 34.0},
                    "prefill_tok_per_j": {"mean": 0.9},
                }
            },
            "vision_summary": {
                "vision_encode_ms": {"mean": 45.0},
                "vision_fps": {"mean": 22.0},
            },
        },
        "device": {"avg_power_w": 55.0},
    }
    metric = VLMCompareMetric.from_payload(payload)
    assert metric is not None
    assert metric.llm_prefill_tps == 12.0
    assert metric.llm_decode_tps == 34.0
    assert metric.llm_prefill_tok_per_j == 0.9
    assert metric.vision_encode_ms == 45.0
    assert metric.vision_fps == 22.0
    assert metric.avg_power_w == 55.0


def test_asr_compare_metric_from_payload() -> None:
    payload = {
        "model": "repo/asr-a",
        "asr": {
            "wer": 0.1,
            "cer": 0.02,
            "rtf": 0.5,
            "inverse_rtf": 2.0,
            "mean_latency_s": 1.2,
            "p50_latency_s": 1.0,
            "p95_latency_s": 1.9,
            "throughput_samples_per_s": 3.0,
            "decode_tokens_per_s": 77.0,
            "avg_tokens_per_sample": 10.0,
        },
        "device": {"avg_power_w": 8.0, "total_energy_j": 4.0, "sec_per_j": 3.0, "rtf_per_w": 0.0625},
    }
    metric = ASRCompareMetric.from_payload(payload)
    assert metric is not None
    assert metric.wer == 0.1
    assert metric.cer == 0.02
    assert metric.wer_pct == 10.0
    assert metric.cer_pct == 2.0
    assert metric.rtf == 0.5
    assert metric.decode_tokens_per_s == 77.0
    assert metric.avg_power_w == 8.0
    assert metric.total_energy_j == 4.0
    assert metric.sec_per_j == 3.0
    assert metric.rtf_per_w == 0.0625


def test_task_registry_contains_all_tasks() -> None:
    assert TASK_REGISTRY[LLMCompareMetric.TASK] is LLMCompareMetric
    assert TASK_REGISTRY[VLMCompareMetric.TASK] is VLMCompareMetric
    assert TASK_REGISTRY[ASRCompareMetric.TASK] is ASRCompareMetric


def test_collect_metrics_and_common_models(tmp_path: Path) -> None:
    folder_a = tmp_path / "a"
    folder_b = tmp_path / "b"
    folder_a.mkdir()
    folder_b.mkdir()
    payload = {
        "model": "group__repo/model-a",
        "benchmark": {
            "prefill_sweep": {"x_values": [128], "tps_values": [10.0], "time_values": [1.0]},
            "decode_sweep": {"x_values": [128], "tps_values": [20.0], "time_values": [2.0]},
        },
        "device": {"avg_power_w": 1.0},
    }
    (folder_a / "one.json").write_text(json.dumps(payload), encoding="utf-8")
    (folder_b / "two.json").write_text(json.dumps(payload), encoding="utf-8")
    metrics_a = collect_metrics(folder_a, LLMCompareMetric)
    metrics_b = collect_metrics(folder_b, LLMCompareMetric)
    assert list(metrics_a.keys()) == ["repo/model-a"]
    assert common_model_ids([metrics_a, metrics_b]) == ["repo/model-a"]


def test_render_charts_smoke(tmp_path: Path) -> None:
    metric = LLMCompareMetric(
        prefill_tps={128: 10.0},
        decode_tps={128: 20.0},
        prefill_tokens_per_j=1.1,
        decode_tokens_per_j=2.2,
        avg_power_w=3.3,
    )
    output_dir = tmp_path / "charts"
    output_dir.mkdir()
    render_charts(
        metric_cls=LLMCompareMetric,
        models=["model-a"],
        labels=["folder-a", "folder-b"],
        metrics_by_folder=[{"model-a": metric}, {"model-a": metric}],
        output_dir=output_dir,
    )
    assert (output_dir / "prefill_tps.png").is_file()
    assert (output_dir / "avg_power_w.png").is_file()


def test_normalize_model_key_keeps_asr_beam_suffix() -> None:
    assert normalize_model_key(Path("whisper-small_beams1.json"), "whisper-small_beams1") == "whisper-small_beams1"
    assert normalize_model_key(Path("whisper-small_beams5.json"), "whisper-small_beams5") == "whisper-small_beams5"
    assert (
        normalize_model_key(Path("openai__whisper-small_beamsdefault.json"), "openai/whisper-small")
        == "openai/whisper-small_beamsdefault"
    )
    assert normalize_model_key(Path("openai__whisper-small_beams4.json"), "openai/whisper-small") == (
        "openai/whisper-small_beams4"
    )


def test_normalize_model_key_keeps_owner_name_by_default() -> None:
    assert normalize_model_key(Path("one.json"), "org-a/model-x") == "org-a/model-x"
    assert normalize_model_key(Path("org-a__model-x.json"), "org-a/model-x") == "org-a/model-x"


def test_normalize_model_key_strips_owner_when_requested() -> None:
    assert normalize_model_key(Path("one.json"), "org-a/model-x", strip_owner=True) == "model-x"
    assert normalize_model_key(Path("org-a__model-x.json"), "org-a/model-x", strip_owner=True) == "model-x"


def test_collect_metrics_keeps_distinct_asr_beam_keys(tmp_path: Path, capsys) -> None:
    """Verify compare collection keeps beam-specific ASR results as distinct model keys."""

    payload = {
        "benchmark_type": "measure",
        "task": "automatic-speech-recognition",
        "model": "openai/whisper-small_beamsdefault",
        "asr": {
            "wer": 0.1,
            "cer": 0.02,
            "rtf": 0.5,
            "inverse_rtf": 2.0,
            "mean_latency_s": 1.2,
            "p50_latency_s": 1.0,
            "p95_latency_s": 1.9,
            "throughput_samples_per_s": 3.0,
            "decode_tokens_per_s": 77.0,
            "avg_tokens_per_sample": 10.0,
        },
        "device": {"avg_power_w": 8.0},
    }
    beam_payload = dict(payload)
    beam_payload["model"] = "openai/whisper-small"

    (tmp_path / "openai__whisper-small_beamsdefault.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "openai__whisper-small_beams1.json").write_text(json.dumps(beam_payload), encoding="utf-8")

    metrics = collect_metrics(tmp_path, ASRCompareMetric)
    captured = capsys.readouterr()

    assert list(metrics.keys()) == ["openai/whisper-small_beams1", "openai/whisper-small_beamsdefault"]
    assert metrics["openai/whisper-small_beamsdefault"].wer == 0.1
    assert captured.out == ""


def test_collect_metrics_keeps_same_basename_from_different_owners(tmp_path: Path) -> None:
    """Verify owner/name keys prevent collisions between same-basename models."""

    base_payload = {
        "benchmark": {
            "prefill_sweep": {"x_values": [128], "tps_values": [10.0], "time_values": [1.0]},
            "decode_sweep": {"x_values": [128], "tps_values": [20.0], "time_values": [2.0]},
        },
        "device": {"avg_power_w": 1.0},
    }
    payload_a = {**base_payload, "model": "org-a/model-x"}
    payload_b = {**base_payload, "model": "org-b/model-x"}
    (tmp_path / "a.json").write_text(json.dumps(payload_a), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(payload_b), encoding="utf-8")

    metrics = collect_metrics(tmp_path, LLMCompareMetric)

    assert list(metrics.keys()) == ["org-a/model-x", "org-b/model-x"]


def test_collect_metrics_strips_owner_when_requested(tmp_path: Path) -> None:
    """Verify owner stripping is opt-in for model comparisons."""

    payload = {
        "model": "org-a/model-x",
        "benchmark": {
            "prefill_sweep": {"x_values": [128], "tps_values": [10.0], "time_values": [1.0]},
            "decode_sweep": {"x_values": [128], "tps_values": [20.0], "time_values": [2.0]},
        },
    }
    (tmp_path / "one.json").write_text(json.dumps(payload), encoding="utf-8")

    metrics = collect_metrics(tmp_path, LLMCompareMetric, strip_owner=True)

    assert list(metrics.keys()) == ["model-x"]


def test_plot_compare_benchmark_results_module_help_smoke() -> None:
    """Verify package/module execution path works for compare help output."""

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "benchmark.transformers.plot_compare_benchmark_results", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Compare N benchmark result folders" in result.stdout


def test_plot_compare_benchmark_results_auto_detects_asr_task(tmp_path: Path) -> None:
    """Verify compare CLI detects ASR payloads when --task is omitted."""

    repo_root = Path(__file__).resolve().parents[2]
    folder_a = tmp_path / "linux_asr"
    folder_b = tmp_path / "windows_asr"
    output_dir = tmp_path / "charts"
    folder_a.mkdir()
    folder_b.mkdir()
    payload = {
        "benchmark_type": "measure",
        "task": "automatic-speech-recognition",
        "model": "mobilint/whisper-small",
        "asr": {
            "wer": 0.1,
            "cer": 0.02,
            "rtf": 0.5,
            "inverse_rtf": 2.0,
            "mean_latency_s": 1.2,
            "p50_latency_s": 1.0,
            "p95_latency_s": 1.9,
            "throughput_samples_per_s": 3.0,
            "decode_tokens_per_s": 77.0,
            "avg_tokens_per_sample": 10.0,
        },
    }
    (folder_a / "mobilint__whisper-small_beams1.json").write_text(json.dumps(payload), encoding="utf-8")
    (folder_b / "mobilint__whisper-small_beams1.json").write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark.transformers.plot_compare_benchmark_results",
            str(folder_a),
            str(folder_b),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Auto-detected task: automatic-speech-recognition" in result.stdout
    assert "Common models across all folders: 1" in result.stdout
    assert (output_dir / "wer.png").is_file()
