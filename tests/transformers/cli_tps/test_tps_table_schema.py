"""Schema tests for :mod:`mblt_model_zoo.cli.tps_table`."""

from __future__ import annotations

import pytest

from mblt_model_zoo.cli.tps_table import (
    SECTION_LLM_MEASURE,
    SECTION_LLM_SWEEP,
    SECTION_VLM_MEASURE,
    SECTION_VLM_SWEEP_LLM,
    SECTION_VLM_SWEEP_VISION,
    emit_table,
    iter_section_rows,
    label_for,
)


def _emit_labels(section: str, *, device_metrics: bool, keys: set[str] | None = None) -> list[str]:
    """Return the labels emit_table produces for ``section``.

    ``keys`` restricts which schema keys carry data.  When ``None`` every row
    that belongs to the section is emitted (as with default runtime data).
    """
    if keys is None:
        keys = {row.key for row in iter_section_rows(section, device_metrics=device_metrics)}
    values_by_key = {key: [] for key in keys}
    captured: list[str] = []

    def _capture(label, values, unit):
        captured.append(label)

    emit_table(
        section,
        values_by_key,
        device_metrics=device_metrics,
        print_summary=_capture,
    )
    return captured


def test_llm_only_sections_have_no_llm_prefix():
    """LLM measure / LLM sweep must expose unprefixed labels like ``avg_power``."""
    llm_measure = _emit_labels(SECTION_LLM_MEASURE, device_metrics=True)
    assert "avg_power" in llm_measure
    assert "avg_util" in llm_measure
    assert "avg_temp" in llm_measure
    assert "avg_mem_used" in llm_measure
    assert "avg_mem_used_pct" in llm_measure
    assert not any(label.startswith("llm_") for label in llm_measure), llm_measure

    llm_sweep = _emit_labels(SECTION_LLM_SWEEP, device_metrics=True)
    assert "avg_power" in llm_sweep
    assert not any(label.startswith("llm_") for label in llm_sweep), llm_sweep


def test_vlm_sections_prefix_llm_overall_rows():
    """VLM measure and VLM sweep LLM must expose ``llm_avg_power`` and friends."""
    vlm_measure = _emit_labels(SECTION_VLM_MEASURE, device_metrics=True)
    for expected in (
        "llm_avg_power",
        "llm_p99_power",
        "llm_avg_util",
        "llm_p99_util",
        "llm_avg_temp",
        "llm_p99_temp",
        "llm_avg_mem_used",
        "llm_p99_mem_used",
        "llm_avg_mem_used_pct",
        "llm_p99_mem_used_pct",
    ):
        assert expected in vlm_measure, expected
    # The unprefixed forms must not leak into VLM tables.
    assert "avg_power" not in vlm_measure

    vlm_sweep_llm = _emit_labels(SECTION_VLM_SWEEP_LLM, device_metrics=True)
    assert "llm_avg_power" in vlm_sweep_llm
    assert "avg_power" not in vlm_sweep_llm


def test_sweep_sections_add_last_suffix_to_perpoint_rows():
    """Sweep tables mark sweep-varying metrics with a ``(last)`` suffix."""
    llm_sweep = _emit_labels(SECTION_LLM_SWEEP, device_metrics=True)
    assert "prefill_tps(last)" in llm_sweep
    assert "decode_tps(last)" in llm_sweep
    assert "ttft(last)" in llm_sweep
    assert "prefill_tps_per_w(last)" in llm_sweep
    # Device aggregates (non-sweep-varying) stay unsuffixed even in a sweep.
    assert "avg_power" in llm_sweep
    assert "avg_power(last)" not in llm_sweep


def test_total_npu_lat_labels_across_sections():
    """total_npu_lat mirrors prefill/decode_npu_lat: llm_ prefix in VLM sections,
    (last) suffix in sweep sections."""
    assert "total_npu_lat" in _emit_labels(SECTION_LLM_MEASURE, device_metrics=True)
    assert "total_npu_lat(last)" in _emit_labels(SECTION_LLM_SWEEP, device_metrics=True)
    vlm_measure = _emit_labels(SECTION_VLM_MEASURE, device_metrics=True)
    assert "llm_total_npu_lat" in vlm_measure
    assert "total_npu_lat" not in vlm_measure
    assert "llm_total_npu_lat(last)" in _emit_labels(SECTION_VLM_SWEEP_LLM, device_metrics=True)


def test_vision_rows_only_appear_in_vlm_sections():
    """vision_* rows must never surface in LLM-only tables."""
    for section in (SECTION_LLM_MEASURE, SECTION_LLM_SWEEP):
        labels = _emit_labels(section, device_metrics=True)
        assert not any(label.startswith("vision_") for label in labels), (section, labels)
    for section in (SECTION_VLM_MEASURE, SECTION_VLM_SWEEP_VISION):
        labels = _emit_labels(section, device_metrics=True)
        assert any(label.startswith("vision_") for label in labels), section


def test_device_metric_gate_hides_device_rows():
    """Setting device_metrics=False must drop every device-only row."""
    labels = _emit_labels(SECTION_LLM_MEASURE, device_metrics=False)
    for forbidden in ("avg_power", "prefill_energy", "prefill_tps_per_w", "total_mem"):
        assert forbidden not in labels, forbidden


def test_missing_keys_are_skipped():
    """emit_table skips rows whose key is absent from values_by_key."""
    captured: list[str] = []

    def _capture(label, values, unit):
        captured.append(label)

    emit_table(
        SECTION_LLM_SWEEP,
        {"prefill_tps": [], "decode_tps": []},
        device_metrics=True,
        print_summary=_capture,
    )
    assert captured == ["prefill_tps(last)", "decode_tps(last)"]


@pytest.mark.parametrize(
    "section, expected_head",
    [
        (SECTION_LLM_MEASURE, "prefill_tps"),
        (SECTION_LLM_SWEEP, "prefill_tps(last)"),
        (SECTION_VLM_MEASURE, "vision_encode"),
        (SECTION_VLM_SWEEP_LLM, "llm_prefill_tps(last)"),
        (SECTION_VLM_SWEEP_VISION, "vision_encode"),
    ],
)
def test_first_emitted_label_matches_section(section, expected_head):
    """Each section's leading row identifies which table it is."""
    labels = _emit_labels(section, device_metrics=True)
    assert labels[0] == expected_head


def test_label_for_transforms_stack_llm_prefix_and_sweep_suffix():
    """``prefill_tps`` transforms cleanly in every section it appears in."""
    rows = {row.key: row for row in iter_section_rows(SECTION_LLM_MEASURE, device_metrics=False)}
    prefill_tps = rows["prefill_tps"]
    assert label_for(prefill_tps, SECTION_LLM_MEASURE) == "prefill_tps"
    assert label_for(prefill_tps, SECTION_LLM_SWEEP) == "prefill_tps(last)"
    assert label_for(prefill_tps, SECTION_VLM_MEASURE) == "llm_prefill_tps"
    assert label_for(prefill_tps, SECTION_VLM_SWEEP_LLM) == "llm_prefill_tps(last)"
