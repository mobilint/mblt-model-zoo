"""Declarative schema for the ``mblt-model-zoo tps`` summary tables.

Four callers (LLM measure/sweep, VLM measure/sweep) previously hardcoded their
own ``_print_summary`` lists.  Renaming or reordering a row required editing
each site independently and it was easy for the four tables to drift.  This
module centralizes the schema so a single edit propagates to every table.

Each :class:`TpsRow` records the base label, unit, and which sections it
belongs to.  Two flags describe how the label is transformed per section:

* ``llm_prefix``  — prepend ``llm_`` when emitted in a VLM section.  This is
  how a shared row like ``avg_power`` becomes ``llm_avg_power`` in the VLM
  tables while staying ``avg_power`` in the LLM-only tables.
* ``sweep_suffix`` — append ``(last)`` in a sweep section.  Sweep tables show
  the metric at the final sweep point; the suffix mirrors that.

Row order in :data:`TPS_TABLE_SPEC` is the emitted order.  Sections filter
which rows appear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

SECTION_LLM_MEASURE = "llm_measure"
SECTION_LLM_SWEEP = "llm_sweep"
SECTION_VLM_MEASURE = "vlm_measure"
SECTION_VLM_SWEEP_LLM = "vlm_sweep_llm"
SECTION_VLM_SWEEP_VISION = "vlm_sweep_vision"

_VLM_SECTIONS = frozenset({SECTION_VLM_MEASURE, SECTION_VLM_SWEEP_LLM, SECTION_VLM_SWEEP_VISION})
_SWEEP_SECTIONS = frozenset({SECTION_LLM_SWEEP, SECTION_VLM_SWEEP_LLM, SECTION_VLM_SWEEP_VISION})


@dataclass(frozen=True)
class TpsRow:
    """One row of a TPS summary table."""

    key: str
    label: str
    unit: str
    sections: frozenset[str]
    device_metric: bool = False
    llm_prefix: bool = False
    sweep_suffix: bool = False


def _row(
    key: str,
    label: str,
    unit: str,
    sections: Sequence[str],
    *,
    device_metric: bool = False,
    llm_prefix: bool = False,
    sweep_suffix: bool = False,
) -> TpsRow:
    return TpsRow(
        key=key,
        label=label,
        unit=unit,
        sections=frozenset(sections),
        device_metric=device_metric,
        llm_prefix=llm_prefix,
        sweep_suffix=sweep_suffix,
    )


_ALL_LLM = (SECTION_LLM_MEASURE, SECTION_LLM_SWEEP, SECTION_VLM_MEASURE, SECTION_VLM_SWEEP_LLM)
_VLM_LLM = (SECTION_VLM_MEASURE, SECTION_VLM_SWEEP_LLM)
_VLM_VISION = (SECTION_VLM_MEASURE, SECTION_VLM_SWEEP_VISION)
_MEASURE_LLM = (SECTION_LLM_MEASURE, SECTION_VLM_MEASURE)
_ALL_DEVICE_SECTIONS = (
    SECTION_LLM_MEASURE,
    SECTION_LLM_SWEEP,
    SECTION_VLM_MEASURE,
    SECTION_VLM_SWEEP_LLM,
    SECTION_VLM_SWEEP_VISION,
)


TPS_TABLE_SPEC: list[TpsRow] = [
    # --- Vision throughput/latency (VLM measure + VLM sweep vision) ---
    _row("vision_encode", "vision_encode", "ms", _VLM_VISION),
    _row("vision_fps", "vision_fps", "fps", _VLM_VISION),

    # --- LLM throughput/latency (all four LLM sections; llm_ prefix in VLM) ---
    _row("prefill_tps", "prefill_tps", "tok/s", _ALL_LLM, llm_prefix=True, sweep_suffix=True),
    _row("decode_tps", "decode_tps", "tok/s", _ALL_LLM, llm_prefix=True, sweep_suffix=True),
    _row("ttft", "ttft", "ms", _ALL_LLM, llm_prefix=True, sweep_suffix=True),
    _row("decode_duration", "decode_duration", "ms", _ALL_LLM, llm_prefix=True, sweep_suffix=True),

    # --- Total wall time (measure only; represents phase totals) ---
    _row("total", "total", "ms", _MEASURE_LLM),

    # --- NPU latency ---
    _row("prefill_npu_lat", "prefill_npu_lat", "%", _ALL_LLM, llm_prefix=True, sweep_suffix=True),
    _row("decode_npu_lat", "decode_npu_lat", "%", _ALL_LLM, llm_prefix=True, sweep_suffix=True),
    # total_npu_lat: LLM measure / LLM sweep / VLM measure keep the unprefixed
    # label (no vision NPU tracking, so no disambiguation was needed
    # historically).  The new VLM sweep LLM subtable row (added in this task)
    # uses ``llm_total_npu_lat(last)`` — the extra prefix disambiguates from
    # vision-side metrics in the paired vision subtable.
    _row(
        "total_npu_lat",
        "total_npu_lat",
        "%",
        (SECTION_LLM_MEASURE, SECTION_LLM_SWEEP, SECTION_VLM_MEASURE),
        sweep_suffix=True,
    ),
    _row(
        "total_npu_lat",
        "llm_total_npu_lat",
        "%",
        (SECTION_VLM_SWEEP_LLM,),
        sweep_suffix=True,
    ),

    # --- EAGLE-3 acceptance (LLM measure only) ---
    _row("accept_steps", "accept_steps", "count", (SECTION_LLM_MEASURE,)),
    _row("accept_tok_sum", "accept_tok_sum", "tok", (SECTION_LLM_MEASURE,)),
    _row("accept_tok_avg", "accept_tok_avg", "tok", (SECTION_LLM_MEASURE,)),
    _row("accept_ratio", "accept_ratio", "%", (SECTION_LLM_MEASURE,)),

    # --- Device metrics: power ---
    _row("avg_power", "avg_power", "W", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("p99_power", "p99_power", "W", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("prefill_avg_power", "prefill_avg_power", "W", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("prefill_p99_power", "prefill_p99_power", "W", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("decode_avg_power", "decode_avg_power", "W", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("decode_p99_power", "decode_p99_power", "W", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("vision_avg_power", "vision_avg_power", "W", _VLM_VISION, device_metric=True),
    _row("vision_p99_power", "vision_p99_power", "W", _VLM_VISION, device_metric=True),

    # --- Device metrics: utilization ---
    _row("avg_util", "avg_util", "%", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("p99_util", "p99_util", "%", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("prefill_avg_util", "prefill_avg_util", "%", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("prefill_p99_util", "prefill_p99_util", "%", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("decode_avg_util", "decode_avg_util", "%", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("decode_p99_util", "decode_p99_util", "%", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("vision_avg_util", "vision_avg_util", "%", _VLM_VISION, device_metric=True),
    _row("vision_p99_util", "vision_p99_util", "%", _VLM_VISION, device_metric=True),

    # --- Device metrics: temperature ---
    _row("avg_temp", "avg_temp", "C", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("p99_temp", "p99_temp", "C", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("prefill_avg_temp", "prefill_avg_temp", "C", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("prefill_p99_temp", "prefill_p99_temp", "C", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("decode_avg_temp", "decode_avg_temp", "C", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("decode_p99_temp", "decode_p99_temp", "C", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("vision_avg_temp", "vision_avg_temp", "C", _VLM_VISION, device_metric=True),
    _row("vision_p99_temp", "vision_p99_temp", "C", _VLM_VISION, device_metric=True),

    # --- Device metrics: memory (MB) ---
    _row("avg_mem_used", "avg_mem_used", "MB", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("p99_mem_used", "p99_mem_used", "MB", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("prefill_avg_mem_used", "prefill_avg_mem_used", "MB", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("prefill_p99_mem_used", "prefill_p99_mem_used", "MB", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("decode_avg_mem_used", "decode_avg_mem_used", "MB", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("decode_p99_mem_used", "decode_p99_mem_used", "MB", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("vision_avg_mem_used", "vision_avg_mem_used", "MB", _VLM_VISION, device_metric=True),
    _row("vision_p99_mem_used", "vision_p99_mem_used", "MB", _VLM_VISION, device_metric=True),

    # --- Total memory (aggregate; no llm_ prefix; appears in every device section) ---
    _row("total_mem", "total_mem", "MB", _ALL_DEVICE_SECTIONS, device_metric=True),

    # --- Device metrics: memory (%) ---
    _row("avg_mem_used_pct", "avg_mem_used_pct", "%", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("p99_mem_used_pct", "p99_mem_used_pct", "%", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row(
        "prefill_avg_mem_used_pct",
        "prefill_avg_mem_used_pct",
        "%",
        _ALL_LLM,
        device_metric=True,
        llm_prefix=True,
    ),
    _row(
        "prefill_p99_mem_used_pct",
        "prefill_p99_mem_used_pct",
        "%",
        _ALL_LLM,
        device_metric=True,
        llm_prefix=True,
    ),
    _row(
        "decode_avg_mem_used_pct",
        "decode_avg_mem_used_pct",
        "%",
        _ALL_LLM,
        device_metric=True,
        llm_prefix=True,
    ),
    _row(
        "decode_p99_mem_used_pct",
        "decode_p99_mem_used_pct",
        "%",
        _ALL_LLM,
        device_metric=True,
        llm_prefix=True,
    ),
    _row("vision_avg_mem_used_pct", "vision_avg_mem_used_pct", "%", _VLM_VISION, device_metric=True),
    _row("vision_p99_mem_used_pct", "vision_p99_mem_used_pct", "%", _VLM_VISION, device_metric=True),

    # --- Energy ---
    _row("prefill_energy", "prefill_energy", "J", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("decode_energy", "decode_energy", "J", _ALL_LLM, device_metric=True, llm_prefix=True),
    _row("vision_energy", "vision_energy", "J", _VLM_VISION, device_metric=True),
    # llm_total_energy: LLM-only aggregate energy in VLM contexts; label is
    # literal (no dynamic prefix) since it's distinct from total_energy.
    _row("llm_total_energy", "llm_total_energy", "J", _VLM_LLM, device_metric=True),
    # total_energy: LLM+vision total in VLM measure; LLM-only total in LLM
    # measure/sweep. Absent from vlm_sweep_llm since that subtable has no
    # vision component.
    _row(
        "total_energy",
        "total_energy",
        "J",
        (SECTION_LLM_MEASURE, SECTION_LLM_SWEEP, SECTION_VLM_MEASURE),
        device_metric=True,
    ),

    # --- Efficiency (per-phase, sweep-varying) ---
    _row(
        "prefill_tps_per_w",
        "prefill_tps_per_w",
        "tok/s/W",
        _ALL_LLM,
        device_metric=True,
        llm_prefix=True,
        sweep_suffix=True,
    ),
    _row(
        "decode_tps_per_w",
        "decode_tps_per_w",
        "tok/s/W",
        _ALL_LLM,
        device_metric=True,
        llm_prefix=True,
        sweep_suffix=True,
    ),
    _row(
        "prefill_j_per_tok",
        "prefill_j_per_tok",
        "J/tok",
        _ALL_LLM,
        device_metric=True,
        llm_prefix=True,
        sweep_suffix=True,
    ),
    _row(
        "decode_j_per_tok",
        "decode_j_per_tok",
        "J/tok",
        _ALL_LLM,
        device_metric=True,
        llm_prefix=True,
        sweep_suffix=True,
    ),
    _row("vision_img_per_j", "vision_img_per_j", "img/J", _VLM_VISION, device_metric=True),
    _row("vision_j_per_img", "vision_j_per_img", "J/img", _VLM_VISION, device_metric=True),
]


def label_for(row: TpsRow, section: str) -> str:
    """Return the display label for ``row`` when emitted in ``section``."""
    label = row.label
    if row.llm_prefix and section in _VLM_SECTIONS:
        label = f"llm_{label}"
    if row.sweep_suffix and section in _SWEEP_SECTIONS:
        label = f"{label}(last)"
    return label


def iter_section_rows(section: str, *, device_metrics: bool) -> list[TpsRow]:
    """Return the rows that should be emitted in ``section``."""
    rows: list[TpsRow] = []
    for row in TPS_TABLE_SPEC:
        if section not in row.sections:
            continue
        if row.device_metric and not device_metrics:
            continue
        rows.append(row)
    return rows


def emit_table(
    section: str,
    values_by_key: Mapping[str, Sequence[float]],
    *,
    device_metrics: bool,
    print_summary,
) -> None:
    """Emit a summary table for ``section`` using ``values_by_key``.

    A row is skipped entirely when its ``key`` is absent from
    ``values_by_key`` — callers use this to omit rows that have no data
    (e.g. sweeps that produced no prefill points).  A row whose key maps to
    an empty list is still emitted, matching the historical behaviour where
    an empty summary prints as a zeros row.

    ``print_summary`` is injected so this module stays free of the display
    formatting concerns living in :mod:`mblt_model_zoo.cli.tps`.
    """
    for row in iter_section_rows(section, device_metrics=device_metrics):
        if row.key not in values_by_key:
            continue
        print_summary(label_for(row, section), values_by_key[row.key], row.unit)
