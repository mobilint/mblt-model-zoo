"""Schema tests for :mod:`mblt_model_zoo.cli.tps_table`."""

from __future__ import annotations

import pytest

from mblt_model_zoo.cli.tps_table import (
    SECTION_LLM_MEASURE,
    SECTION_LLM_SWEEP,
    SECTION_VLM_MEASURE,
    SECTION_VLM_SWEEP_LLM,
    SECTION_VLM_SWEEP_VISION,
    TPS_TABLE_SPEC,
    emit_table,
    iter_json_rows,
    iter_section_rows,
    json_key_for,
    label_for,
    render_units,
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
    # Per-phase efficiency rows are phase-wide aggregates in sweep tables, not
    # last-sweep-point metrics, so they must not carry the (last) suffix.
    assert "prefill_tps_per_w" in llm_sweep
    assert "prefill_tps_per_w(last)" not in llm_sweep
    assert "decode_tps_per_w(last)" not in llm_sweep
    assert "prefill_j_per_tok(last)" not in llm_sweep
    assert "decode_j_per_tok(last)" not in llm_sweep
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


@pytest.mark.parametrize(
    "section",
    [SECTION_LLM_MEASURE, SECTION_LLM_SWEEP, SECTION_VLM_MEASURE, SECTION_VLM_SWEEP_LLM, SECTION_VLM_SWEEP_VISION],
)
def test_iter_json_rows_matches_cli_table_membership(section):
    """The JSON row set for a section must cover every row the CLI table can print.

    ``iter_json_rows`` differs from ``iter_section_rows`` only in that it
    does not apply the ``device_metrics`` gate — the JSON payload always
    dumps whatever the run produced.
    """
    cli_rows = {row.key for row in iter_section_rows(section, device_metrics=True)}
    json_rows = {row.key for row in iter_json_rows(section)}
    assert cli_rows <= json_rows, section
    # Also: every JSON row must belong to the section (no spillover from spec).
    assert all(section in row.sections for row in iter_json_rows(section))


def test_json_key_for_uses_underscore_last_not_parenthesized_last():
    """JSON keys must use ``_last`` where CLI labels use ``(last)``."""
    rows = {row.key: row for row in iter_json_rows(SECTION_VLM_SWEEP_LLM)}
    prefill_tps = rows["prefill_tps"]
    # sweep_suffix=True + VLM section (llm_prefix=True) → both transformations.
    assert label_for(prefill_tps, SECTION_VLM_SWEEP_LLM) == "llm_prefill_tps(last)"
    assert json_key_for(prefill_tps, SECTION_VLM_SWEEP_LLM) == "llm_prefill_tps_last"
    # Non-sweep, non-VLM: identity.
    avg_power = rows["avg_power"]
    assert json_key_for(avg_power, SECTION_LLM_MEASURE) == "avg_power"
    assert json_key_for(avg_power, SECTION_VLM_SWEEP_LLM) == "llm_avg_power"
    # Sweep-suffix but no llm_prefix: only _last is applied.
    npu_row = next(r for r in TPS_TABLE_SPEC if r.key == "prefill_npu_lat")
    assert json_key_for(npu_row, SECTION_LLM_SWEEP) == "prefill_npu_lat_last"


def test_json_key_never_encodes_singular_unit_suffix():
    """No canonical JSON key should encode a raw unit like ``_w`` / ``_mb`` / ``_j``.

    Compound-unit keys like ``prefill_tps_per_w`` (tok/s/W) and
    ``prefill_j_per_tok`` (J/tok) are semantic: the ``_per_`` / ``j_per``
    fragment describes *what the metric measures*, not the unit split.
    """
    forbidden_suffixes = ("_w", "_mb", "_j", "_c", "_ms")
    for section in (
        SECTION_LLM_MEASURE,
        SECTION_LLM_SWEEP,
        SECTION_VLM_MEASURE,
        SECTION_VLM_SWEEP_LLM,
        SECTION_VLM_SWEEP_VISION,
    ):
        for row in iter_json_rows(section):
            key = json_key_for(row, section)
            base_key = key.removesuffix("_last")
            # Skip compound-unit keys — the fragment is semantic, not a
            # standalone unit tag.
            if "_per_" in base_key or "_j_per_" in base_key:
                continue
            for suffix in forbidden_suffixes:
                assert not base_key.endswith(suffix), (section, key)


def test_render_units_returns_unit_per_emitted_row():
    """``render_units`` must return an entry for every row that appears in the JSON payload."""
    section = SECTION_LLM_MEASURE
    all_keys = {json_key_for(row, section) for row in iter_json_rows(section)}
    units = render_units(section, all_keys)
    for row in iter_json_rows(section):
        assert units[json_key_for(row, section)] == row.unit


def test_llm_total_energy_row_sources_from_llm_only_field():
    """Regression guard for Codex Review gamma: llm_total_energy must be
    sourced from ``llm_total_energy_j`` (LLM-only), never from
    ``total_energy_j`` (vision + LLM in VLM contexts)."""
    from types import SimpleNamespace

    row = next(r for r in TPS_TABLE_SPEC if r.key == "llm_total_energy")
    # A synthetic run with mismatched LLM-only vs total energies.
    run = SimpleNamespace(llm_total_energy_j=7.0, total_energy_j=99.0)
    assert row.from_run is not None
    assert row.from_run(run) == 7.0
    # If llm_total_energy_j is missing, the row must not silently fall back
    # to total_energy_j.
    run_without_llm = SimpleNamespace(total_energy_j=99.0)
    assert row.from_run(run_without_llm) is None


def test_summary_extractors_produce_scalars_from_synthetic_runs():
    """from_runs_for_summary must return the same scalars fed to _summary."""
    from types import SimpleNamespace

    # Two synthetic VLM sweep LLM runs.
    runs = [
        SimpleNamespace(
            prefill_sweep=SimpleNamespace(
                x_values=[64, 128],
                tps_values=[100.0, 200.0],
                time_values=[0.1, 0.2],
                avg_total_token_latency_values=[0.001, 0.001],
                avg_npu_token_latency_values=[0.0005, 0.0005],
            ),
            decode_sweep=SimpleNamespace(
                x_values=[32, 64],
                tps_values=[50.0, 60.0],
                time_values=[0.5, 0.5],
                avg_total_token_latency_values=[0.01, 0.01],
                avg_npu_token_latency_values=[0.005, 0.005],
            ),
            avg_power_w=3.0,
            llm_total_energy_j=10.0,
        ),
        SimpleNamespace(
            prefill_sweep=SimpleNamespace(
                x_values=[64, 128],
                tps_values=[110.0, 220.0],
                time_values=[0.11, 0.22],
                avg_total_token_latency_values=[0.001, 0.001],
                avg_npu_token_latency_values=[0.0005, 0.0005],
            ),
            decode_sweep=SimpleNamespace(
                x_values=[32, 64],
                tps_values=[55.0, 66.0],
                time_values=[0.55, 0.55],
                avg_total_token_latency_values=[0.01, 0.01],
                avg_npu_token_latency_values=[0.005, 0.005],
            ),
            avg_power_w=4.0,
            llm_total_energy_j=12.0,
        ),
    ]
    rows = {r.key: r for r in iter_json_rows(SECTION_VLM_SWEEP_LLM)}
    # Sweep-suffix row: last point per run.
    assert rows["prefill_tps"].from_runs_for_summary(runs) == [200.0, 220.0]
    # Scalar row: filtered attr.
    assert rows["avg_power"].from_runs_for_summary(runs) == [3.0, 4.0]
    # llm_total_energy: LLM-only, not vision+LLM.
    assert rows["llm_total_energy"].from_runs_for_summary(runs) == [10.0, 12.0]
