"""Schema tests for :mod:`mblt_model_zoo.cli.tps_table`."""

from __future__ import annotations

import pytest

from mblt_model_zoo.cli.tps_table import (
    SECTION_LLM_MEASURE,
    SECTION_LLM_SWEEP,
    SECTION_VLM_MEASURE,
    SECTION_VLM_SWEEP_LLM,
    SECTION_VLM_SWEEP_VISION,
    TEMPERATURE_JSON_KEY,
    TPS_TABLE_SPEC,
    emit_table,
    format_temperature_display,
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
    """JSON summary keys must use ``_last`` where CLI labels use ``(last)``."""
    rows = {row.key: row for row in iter_json_rows(SECTION_VLM_SWEEP_LLM)}
    prefill_tps = rows["prefill_tps"]
    # sweep_suffix=True + VLM section (llm_prefix=True) → both transformations
    # kick in for the summary form; runs/aggregate keep the bare name.
    assert label_for(prefill_tps, SECTION_VLM_SWEEP_LLM) == "llm_prefill_tps(last)"
    assert json_key_for(prefill_tps, SECTION_VLM_SWEEP_LLM, is_summary=True) == "llm_prefill_tps_last"
    assert json_key_for(prefill_tps, SECTION_VLM_SWEEP_LLM) == "llm_prefill_tps"
    # Non-sweep, non-VLM: identity.
    avg_power = rows["avg_power"]
    assert json_key_for(avg_power, SECTION_LLM_MEASURE) == "avg_power"
    assert json_key_for(avg_power, SECTION_VLM_SWEEP_LLM) == "llm_avg_power"
    # Sweep-suffix but no llm_prefix: only _last is applied, and only for summary.
    npu_row = next(r for r in TPS_TABLE_SPEC if r.key == "prefill_npu_lat")
    assert json_key_for(npu_row, SECTION_LLM_SWEEP, is_summary=True) == "prefill_npu_lat_last"
    assert json_key_for(npu_row, SECTION_LLM_SWEEP) == "prefill_npu_lat"


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


_SWEEP_SECTIONS_FOR_TEST = (SECTION_LLM_SWEEP, SECTION_VLM_SWEEP_LLM, SECTION_VLM_SWEEP_VISION)


def test_json_key_last_suffix_is_summary_only():
    """``_last`` fires on summary keys only; runs/aggregate use bare names.

    Regression guard for the CHANGELOG 2.3.0 contract: top-level curves in
    runs/aggregate must not carry ``_last`` (they hold the full curve, not
    a scalar), and every sweep-suffix row must expose the ``_last`` scalar
    key in the summary block.
    """
    for row in TPS_TABLE_SPEC:
        if not row.sweep_suffix:
            continue
        for section in row.sections:
            if section not in _SWEEP_SECTIONS_FOR_TEST:
                continue
            bare = json_key_for(row, section)
            summary = json_key_for(row, section, is_summary=True)
            assert not bare.endswith("_last"), (row.key, section, bare)
            assert summary.endswith("_last"), (row.key, section, summary)
            assert summary == f"{bare}_last", (row.key, section, bare, summary)


def test_ms_scale_rows_convert_sweep_curves_to_milliseconds():
    """ms_scale=True rows return millisecond-scaled curves from every extractor.

    Regression guard for the SoT-unification bug where runs[].ttft /
    aggregate.ttft came out 1000x smaller than summary.ttft because the
    curve extractors returned SweepData.time_values (seconds) verbatim
    while the summary already multiplied by 1000.
    """
    from types import SimpleNamespace

    ms_rows = [row for row in TPS_TABLE_SPEC if row.key in ("ttft", "decode_duration")]
    assert {r.key for r in ms_rows} == {"ttft", "decode_duration"}

    sweep_attr_by_key = {"ttft": "prefill_sweep", "decode_duration": "decode_sweep"}
    time_values = [0.001, 0.002, 0.005]
    expected_curve = [1.0, 2.0, 5.0]
    expected_last = 5.0

    for row in ms_rows:
        sweep_attr = sweep_attr_by_key[row.key]
        sweep = SimpleNamespace(
            x_values=[1, 2, 3],
            tps_values=[10.0, 20.0, 30.0],
            time_values=list(time_values),
            avg_total_token_latency_values=[],
            avg_npu_token_latency_values=[],
        )
        obj = SimpleNamespace(**{sweep_attr: sweep})

        run_curve = row.from_run(obj)
        assert run_curve == expected_curve, (row.key, run_curve)

        agg_curve = row.from_aggregate(obj)
        assert agg_curve == expected_curve, (row.key, agg_curve)

        summary_values = row.from_runs_for_summary([obj])
        assert summary_values == [expected_last], (row.key, summary_values)


def test_total_row_from_run_combines_vision_and_llm_for_vlm_measure():
    """``total.from_run`` for a VLM measurement returns vision+LLM wall time in ms.

    Regression guard for the SoT-unification bug where runs[].total came
    back as ``llm.total_time`` alone (LLM-only) while summary.total was
    ``vision_encode_latency * batch_size + llm.total_time`` (combined) —
    the two carried mismatched quantities under the same key.
    """
    from types import SimpleNamespace

    row = next(r for r in TPS_TABLE_SPEC if r.key == "total")
    assert row.from_run is not None

    vlm_run = SimpleNamespace(
        image_resolution=224,
        vision_encode_latency=0.010,
        vision_fps=100.0,
        batch_size=4,
        llm=SimpleNamespace(total_time=1.0),
    )
    # (0.010 * 4 + 1.0) * 1000 = 1040 ms.
    assert row.from_run(vlm_run) == pytest.approx(1040.0)

    # batch_size defaults to 1 when the measurement doesn't carry it.
    vlm_run_no_batch = SimpleNamespace(
        vision_encode_latency=0.010,
        vision_fps=100.0,
        llm=SimpleNamespace(total_time=1.0),
    )
    # (0.010 * 1 + 1.0) * 1000 = 1010 ms.
    assert row.from_run(vlm_run_no_batch) == pytest.approx(1010.0)

    # Plain LLM measurement still uses the flat total_time attribute (in seconds).
    llm_run = SimpleNamespace(total_time=2.0)
    assert row.from_run(llm_run) == pytest.approx(2000.0)

    # VLM measurement missing llm.total_time yields None rather than a wrong total.
    vlm_run_incomplete = SimpleNamespace(
        vision_encode_latency=0.010,
        vision_fps=100.0,
        batch_size=1,
        llm=SimpleNamespace(),
    )
    assert row.from_run(vlm_run_incomplete) is None


def test_temperature_json_key_is_stable():
    """``TEMPERATURE_JSON_KEY`` is the canonical JSON key emitted by ``tps measure``."""
    assert TEMPERATURE_JSON_KEY == "temperature"


def test_format_temperature_display_greedy_labels_zero():
    """Zero temperature should render as ``0.0 (greedy)`` in the CLI header."""
    assert format_temperature_display(0.0) == "0.0 (greedy)"


def test_format_temperature_display_positive_values():
    """Positive temperatures should render as their numeric value."""
    assert format_temperature_display(0.7) == "0.7"
    assert format_temperature_display(1.5) == "1.5"


def _accept_row_keys() -> set[str]:
    """Return the canonical keys of speculative-decoding acceptance rows."""
    return {row.key for row in TPS_TABLE_SPEC if row.spec_decode_only}


def test_accept_rows_are_gated_by_is_speculative():
    """Acceptance rows must appear only when ``is_speculative=True``."""
    expected_keys = {"accept_steps", "tokens_sum", "tokens_per_step", "draft_accept_ratio"}
    assert _accept_row_keys() == expected_keys

    non_spec = _emit_labels(SECTION_LLM_MEASURE, device_metrics=True)
    for label in expected_keys:
        assert label not in non_spec, label

    spec_keys = {
        row.key
        for row in iter_section_rows(
            SECTION_LLM_MEASURE,
            device_metrics=True,
            is_speculative=True,
        )
    }
    assert expected_keys <= spec_keys

    non_spec_keys = {
        row.key
        for row in iter_section_rows(
            SECTION_LLM_MEASURE,
            device_metrics=True,
            is_speculative=False,
        )
    }
    assert expected_keys.isdisjoint(non_spec_keys)


def test_accept_rows_survive_no_device_metrics_when_speculative():
    """``--no-device-metrics`` must not hide acceptance rows on a speculative model.

    Acceptance metrics are not device telemetry; the only gate that hides them
    is ``is_speculative=False``.
    """
    rows_with_device = {
        row.key for row in iter_section_rows(SECTION_LLM_MEASURE, device_metrics=True, is_speculative=True)
    }
    rows_without_device = {
        row.key for row in iter_section_rows(SECTION_LLM_MEASURE, device_metrics=False, is_speculative=True)
    }
    for key in ("accept_steps", "tokens_sum", "tokens_per_step", "draft_accept_ratio"):
        assert key in rows_with_device, key
        assert key in rows_without_device, key


def test_accept_rows_dropped_from_json_when_not_speculative():
    """``iter_json_rows`` (JSON payload) also omits acceptance rows for plain LLMs."""
    non_spec_keys = {row.key for row in iter_json_rows(SECTION_LLM_MEASURE)}
    spec_keys = {row.key for row in iter_json_rows(SECTION_LLM_MEASURE, is_speculative=True)}
    for key in ("accept_steps", "tokens_sum", "tokens_per_step", "draft_accept_ratio"):
        assert key not in non_spec_keys, key
        assert key in spec_keys, key


def test_tokens_per_step_row_adds_root_token():
    """``tokens_per_step = acceptance_tokens_avg + 1`` (base's forced root token per step)."""
    from types import SimpleNamespace

    row = next(r for r in TPS_TABLE_SPEC if r.key == "tokens_per_step")
    assert row.from_run is not None
    assert row.spec_decode_only is True

    run = SimpleNamespace(acceptance_tokens_avg=2.242, acceptance_steps=8)
    assert row.from_run(run) == pytest.approx(3.242)

    missing = SimpleNamespace(acceptance_tokens_avg=None)
    assert row.from_run(missing) is None

    runs = [
        SimpleNamespace(acceptance_tokens_avg=2.0, acceptance_steps=5),
        SimpleNamespace(acceptance_tokens_avg=3.0, acceptance_steps=4),
    ]
    assert row.from_runs_for_summary(runs) == [3.0, 4.0]


def test_tokens_sum_row_adds_root_tokens_per_step():
    """``tokens_sum = acceptance_tokens_sum + acceptance_steps`` (roots + accepted drafts)."""
    from types import SimpleNamespace

    row = next(r for r in TPS_TABLE_SPEC if r.key == "tokens_sum")
    assert row.from_run is not None
    assert row.spec_decode_only is True

    run = SimpleNamespace(acceptance_tokens_sum=18, acceptance_steps=8)
    assert row.from_run(run) == pytest.approx(26.0)

    missing_sum = SimpleNamespace(acceptance_tokens_sum=None, acceptance_steps=8)
    assert row.from_run(missing_sum) is None
    missing_steps = SimpleNamespace(acceptance_tokens_sum=18, acceptance_steps=None)
    assert row.from_run(missing_steps) is None

    runs = [
        SimpleNamespace(acceptance_tokens_sum=10, acceptance_steps=5),
        SimpleNamespace(acceptance_tokens_sum=12, acceptance_steps=4),
    ]
    assert row.from_runs_for_summary(runs) == [15.0, 16.0]


def test_draft_accept_ratio_row_scales_ratio_to_percent():
    """``draft_accept_ratio`` renames the historical ``accept_ratio`` and stays x100."""
    from types import SimpleNamespace

    row = next(r for r in TPS_TABLE_SPEC if r.key == "draft_accept_ratio")
    assert row.spec_decode_only is True
    assert row.unit == "%"
    assert row.from_run is not None

    run = SimpleNamespace(acceptance_ratio=0.5)
    assert row.from_run(run) == pytest.approx(50.0)


def test_emit_table_speculative_flag_prints_new_labels():
    """When speculative, acceptance labels are emitted with the new names."""
    captured: list[str] = []

    def _capture(label, values, unit):
        captured.append(label)

    values = {
        "accept_steps": [8.0],
        "tokens_sum": [26.0],
        "tokens_per_step": [3.242],
        "draft_accept_ratio": [70.0],
    }
    emit_table(
        SECTION_LLM_MEASURE,
        values,
        device_metrics=False,
        print_summary=_capture,
        is_speculative=True,
    )
    assert captured == ["accept_steps", "tokens_sum", "tokens_per_step", "draft_accept_ratio"]
