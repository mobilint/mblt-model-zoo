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
* ``sweep_suffix`` — append ``(last)`` in a sweep section (CLI label) or
  ``_last`` in JSON summary keys.  Sweep tables show the metric at the final
  sweep point; the suffix mirrors that.

Row order in :data:`TPS_TABLE_SPEC` is the emitted order.  Sections filter
which rows appear.

Extractor callables on each row make the same schema drive the JSON output
of the ``tps`` CLI:

* ``from_run`` projects a per-run value from a benchmark run object.
* ``from_aggregate`` projects a value from the aggregated benchmark result.
* ``from_runs_for_summary`` returns the list of scalars fed to :func:`_summary`.

Rendering helpers (:func:`json_key_for`, :func:`iter_json_rows`,
:func:`emit_units`) use the same section-aware transformations as the CLI
label helpers, so the JSON key set stays in lockstep with the printed table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

SECTION_LLM_MEASURE = "llm_measure"
SECTION_LLM_SWEEP = "llm_sweep"
SECTION_VLM_MEASURE = "vlm_measure"
SECTION_VLM_SWEEP_LLM = "vlm_sweep_llm"
SECTION_VLM_SWEEP_VISION = "vlm_sweep_vision"

_VLM_SECTIONS = frozenset({SECTION_VLM_MEASURE, SECTION_VLM_SWEEP_LLM, SECTION_VLM_SWEEP_VISION})
_SWEEP_SECTIONS = frozenset({SECTION_LLM_SWEEP, SECTION_VLM_SWEEP_LLM, SECTION_VLM_SWEEP_VISION})


Extractor = Callable[[Any], Any]
RunsExtractor = Callable[[Sequence[Any]], "list[float]"]


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
    from_run: Optional[Extractor] = None
    from_aggregate: Optional[Extractor] = None
    from_runs_for_summary: Optional[RunsExtractor] = None


def _get_optional(obj: Any, name: str) -> Any:
    """Return ``obj.name`` or ``obj.llm.name``, whichever exists and is not None.

    LLM measure runs are ``SingleMeasurement`` (flat).  VLM measure runs are
    ``VLMSingleMeasurement`` (with a nested ``.llm``).  Sweep runs are
    ``BenchmarkResult`` (flat).  This helper hides the difference for
    extractors that pull a scalar attribute.
    """
    value = getattr(obj, name, None)
    if value is not None:
        return value
    inner = getattr(obj, "llm", None)
    if inner is None:
        return None
    return getattr(inner, name, None)


def _attr(name: str) -> Extractor:
    """Return an extractor that reads ``obj.name`` (with ``.llm`` fallback)."""
    def _f(obj: Any) -> Any:
        return _get_optional(obj, name)
    return _f


def _direct_attr(name: str) -> Extractor:
    """Return an extractor that reads ``obj.name`` without the ``.llm`` fallback."""
    def _f(obj: Any) -> Any:
        return getattr(obj, name, None)
    return _f


def _sweep_curve(sweep_attr: str, values_attr: str) -> Extractor:
    """Return an extractor that reads a full sweep curve as a list."""
    def _f(obj: Any) -> Any:
        sweep = getattr(obj, sweep_attr, None)
        if sweep is None:
            inner = getattr(obj, "llm", None)
            if inner is not None:
                sweep = getattr(inner, sweep_attr, None)
        if sweep is None:
            return None
        values = getattr(sweep, values_attr, None)
        if values is None:
            return None
        return list(values)
    return _f


def _sweep_last(sweep_attr: str, values_attr: str) -> Extractor:
    """Return an extractor for the last point of a sweep curve."""
    curve = _sweep_curve(sweep_attr, values_attr)

    def _f(obj: Any) -> Any:
        values = curve(obj)
        if not values:
            return None
        last = values[-1]
        if last is None:
            return None
        return last
    return _f


def _sweep_last_ms(sweep_attr: str, values_attr: str) -> Extractor:
    """Return an extractor for the last point of a sweep curve, scaled by 1000."""
    last = _sweep_last(sweep_attr, values_attr)

    def _f(obj: Any) -> Any:
        v = last(obj)
        if v is None:
            return None
        return float(v) * 1000.0
    return _f


def _list_attr(name: str, *, scale: float = 1.0) -> RunsExtractor:
    """Return a from_runs_for_summary that pulls ``obj.name`` from every run."""
    def _f(runs: Sequence[Any]) -> list[float]:
        out: list[float] = []
        for run in runs:
            v = _get_optional(run, name)
            if v is None:
                continue
            out.append(float(v) * scale)
        return out
    return _f


def _list_sweep_last(sweep_attr: str, values_attr: str, *, scale: float = 1.0) -> RunsExtractor:
    """Return a from_runs_for_summary that pulls the last sweep point per run."""
    last = _sweep_last(sweep_attr, values_attr)

    def _f(runs: Sequence[Any]) -> list[float]:
        out: list[float] = []
        for run in runs:
            v = last(run)
            if v is None:
                continue
            out.append(float(v) * scale)
        return out
    return _f


def _row(
    key: str,
    label: str,
    unit: str,
    sections: Sequence[str],
    *,
    device_metric: bool = False,
    llm_prefix: bool = False,
    sweep_suffix: bool = False,
    from_run: Optional[Extractor] = None,
    from_aggregate: Optional[Extractor] = None,
    from_runs_for_summary: Optional[RunsExtractor] = None,
) -> TpsRow:
    return TpsRow(
        key=key,
        label=label,
        unit=unit,
        sections=frozenset(sections),
        device_metric=device_metric,
        llm_prefix=llm_prefix,
        sweep_suffix=sweep_suffix,
        from_run=from_run,
        from_aggregate=from_aggregate,
        from_runs_for_summary=from_runs_for_summary,
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


def _throughput_row(
    key: str,
    label: str,
    unit: str,
    *,
    per_run_attr: str,
    sweep_attr: str,
    sweep_values_attr: str,
    ms_scale: bool = False,
) -> TpsRow:
    """Build a throughput/latency row that maps to both flat and sweep runs.

    Measure sections read ``per_run_attr`` from ``SingleMeasurement`` /
    ``VLMSingleMeasurement.llm``.  Sweep sections read the last point of
    ``run.<sweep_attr>.<sweep_values_attr>``.  For sweep sections the
    aggregate exposes the full curve; runs expose the full curve too.
    """
    scale = 1000.0 if ms_scale else 1.0
    per_run_attr_final = per_run_attr

    def _scale_curve(curve: Optional[list]) -> Optional[list]:
        # SweepData stores latency values in seconds; ms_scale rows advertise
        # milliseconds and must convert to keep runs/aggregate curves aligned
        # with the ms-scaled summary values.
        if curve is None:
            return None
        if not ms_scale:
            return curve
        return [None if v is None else float(v) * 1000.0 for v in curve]

    def _from_run(obj: Any) -> Any:
        # Prefer sweep curve when present (BenchmarkResult / VLM sweep LLM
        # runs).  Fall back to the flat attribute (SingleMeasurement /
        # VLMSingleMeasurement.llm).
        curve = _sweep_curve(sweep_attr, sweep_values_attr)(obj)
        if curve:
            return _scale_curve(curve)
        v = _get_optional(obj, per_run_attr_final)
        if v is None:
            return None
        return float(v) * scale

    def _from_aggregate(obj: Any) -> Any:
        curve = _sweep_curve(sweep_attr, sweep_values_attr)(obj)
        if curve:
            return _scale_curve(curve)
        return None

    def _from_runs_for_summary(runs: Sequence[Any]) -> list[float]:
        out: list[float] = []
        for run in runs:
            # Prefer sweep-last-value; fall back to flat attr.
            last = _sweep_last(sweep_attr, sweep_values_attr)(run)
            if last is not None:
                out.append(float(last) * scale)
                continue
            v = _get_optional(run, per_run_attr_final)
            if v is None:
                continue
            out.append(float(v) * scale)
        return out

    return TpsRow(
        key=key,
        label=label,
        unit=unit,
        sections=frozenset(_ALL_LLM),
        device_metric=False,
        llm_prefix=True,
        sweep_suffix=True,
        from_run=_from_run,
        from_aggregate=_from_aggregate,
        from_runs_for_summary=_from_runs_for_summary,
    )


# --- Sweep NPU-latency helpers (percent = 100 * npu_time / total_time) ---


def _npu_latency_pct(total: Any, npu: Any) -> Optional[float]:
    if total is None or npu is None:
        return None
    try:
        total_f = float(total)
        npu_f = float(npu)
    except (TypeError, ValueError):
        return None
    if total_f <= 0:
        return None
    return (npu_f / total_f) * 100.0


def _sweep_last_npu_pct(sweep_attr: str) -> Extractor:
    def _f(obj: Any) -> Any:
        sweep = getattr(obj, sweep_attr, None)
        if sweep is None:
            return None
        totals = getattr(sweep, "avg_total_token_latency_values", None) or []
        npus = getattr(sweep, "avg_npu_token_latency_values", None) or []
        if not totals or not npus:
            return None
        return _npu_latency_pct(totals[-1], npus[-1])
    return _f


def _npu_latency_row(
    key: str,
    label: str,
    per_run_attr: str,
    sweep_attr: str,
) -> TpsRow:
    """NPU-latency percentage row: measure uses flat attr, sweep uses last point."""

    def _from_run(obj: Any) -> Any:
        v = _get_optional(obj, per_run_attr)
        if v is not None:
            return float(v)
        return _sweep_last_npu_pct(sweep_attr)(obj)

    def _from_aggregate(obj: Any) -> Any:
        return _sweep_last_npu_pct(sweep_attr)(obj)

    def _from_runs_for_summary(runs: Sequence[Any]) -> list[float]:
        out: list[float] = []
        for run in runs:
            v = _get_optional(run, per_run_attr)
            if v is None:
                v = _sweep_last_npu_pct(sweep_attr)(run)
            if v is None:
                continue
            out.append(float(v))
        return out

    return TpsRow(
        key=key,
        label=label,
        unit="%",
        sections=frozenset(_ALL_LLM),
        device_metric=False,
        llm_prefix=True,
        sweep_suffix=True,
        from_run=_from_run,
        from_aggregate=_from_aggregate,
        from_runs_for_summary=_from_runs_for_summary,
    )


def _total_npu_latency_row() -> TpsRow:
    """total_npu_lat combines prefill+decode NPU-latency percentages."""

    def _last(run: Any) -> Optional[float]:
        prefill = _sweep_last_npu_pct("prefill_sweep")(run)
        decode = _sweep_last_npu_pct("decode_sweep")(run)
        if prefill is None or decode is None:
            return None
        prefill_sweep = getattr(run, "prefill_sweep", None)
        decode_sweep = getattr(run, "decode_sweep", None)
        p_t = (
            float(prefill_sweep.time_values[-1])
            if prefill_sweep and prefill_sweep.time_values
            else 0.0
        )
        d_t = (
            float(decode_sweep.time_values[-1])
            if decode_sweep and decode_sweep.time_values
            else 0.0
        )
        weight_sum = p_t + d_t
        if weight_sum <= 0:
            return None
        return (float(prefill) * p_t + float(decode) * d_t) / weight_sum

    def _from_run(obj: Any) -> Any:
        v = _get_optional(obj, "total_npu_latency_pct")
        if v is not None:
            return float(v)
        return _last(obj)

    def _from_aggregate(obj: Any) -> Any:
        return _last(obj)

    def _from_runs_for_summary(runs: Sequence[Any]) -> list[float]:
        out: list[float] = []
        for run in runs:
            v = _get_optional(run, "total_npu_latency_pct")
            if v is None:
                v = _last(run)
            if v is None:
                continue
            out.append(float(v))
        return out

    return TpsRow(
        key="total_npu_lat",
        label="total_npu_lat",
        unit="%",
        sections=frozenset(_ALL_LLM),
        device_metric=False,
        llm_prefix=True,
        sweep_suffix=True,
        from_run=_from_run,
        from_aggregate=_from_aggregate,
        from_runs_for_summary=_from_runs_for_summary,
    )


def _scalar_device_row(
    key: str,
    label: str,
    unit: str,
    sections: Sequence[str],
    attr: str,
    *,
    llm_prefix: bool = False,
    scale: float = 1.0,
) -> TpsRow:
    """Row for a scalar device metric that reads ``obj.attr`` uniformly."""
    def _from_run(obj: Any) -> Any:
        v = _get_optional(obj, attr)
        if v is None:
            return None
        return float(v) * scale

    def _from_aggregate(obj: Any) -> Any:
        v = _get_optional(obj, attr)
        if v is None:
            return None
        return float(v) * scale

    return TpsRow(
        key=key,
        label=label,
        unit=unit,
        sections=frozenset(sections),
        device_metric=True,
        llm_prefix=llm_prefix,
        sweep_suffix=False,
        from_run=_from_run,
        from_aggregate=_from_aggregate,
        from_runs_for_summary=_list_attr(attr, scale=scale),
    )


def _accept_row(
    key: str,
    label: str,
    unit: str,
    attr: str,
    *,
    scale: float = 1.0,
) -> TpsRow:
    """Acceptance metric row (LLM measure only)."""
    return TpsRow(
        key=key,
        label=label,
        unit=unit,
        sections=frozenset((SECTION_LLM_MEASURE,)),
        device_metric=False,
        llm_prefix=False,
        sweep_suffix=False,
        from_run=lambda obj, _attr=attr, _s=scale: (
            None if _get_optional(obj, _attr) is None else float(_get_optional(obj, _attr)) * _s
        ),
        from_runs_for_summary=_list_attr(attr, scale=scale),
    )


def _total_measure_from_run(obj: Any) -> Optional[float]:
    """Return per-run total wall time in ms for a measure section.

    VLMSingleMeasurement has no top-level ``total_time`` — the vision phase
    latency lives in ``vision_encode_latency`` (per-image, seconds) and the
    LLM phase in ``llm.total_time`` (seconds).  The summary emits the
    combined wall time ``vision_encode_latency * batch_size + llm.total_time``;
    the per-run extractor mirrors that so ``runs[i].total`` and
    ``summary.total`` describe the same quantity.

    ``batch_size`` is read from the measurement when present (defaulting to
    ``1``) so callers that later attach ``obj.batch_size`` — or add the
    field to :class:`VLMSingleMeasurement` — get correct batched totals
    without further changes here.
    """
    vision = getattr(obj, "vision_encode_latency", None)
    if vision is not None:
        llm = getattr(obj, "llm", None)
        llm_total = getattr(llm, "total_time", None) if llm is not None else None
        if llm_total is None:
            return None
        batch_size = getattr(obj, "batch_size", 1)
        return (float(vision) * float(batch_size) + float(llm_total)) * 1000.0
    total = _get_optional(obj, "total_time")
    if total is None:
        return None
    return float(total) * 1000.0


TPS_TABLE_SPEC: list[TpsRow] = [
    # --- Vision throughput/latency (VLM measure + VLM sweep vision) ---
    _row(
        "vision_encode",
        "vision_encode",
        "ms",
        _VLM_VISION,
        from_run=lambda obj: (
            None
            if getattr(obj, "vision_encode_latency", None) is None
            else float(obj.vision_encode_latency) * 1000.0
        ),
        from_runs_for_summary=lambda runs: [
            float(v) * 1000.0
            for r in runs
            if (v := getattr(r, "vision_encode_latency", None)) is not None
        ],
    ),
    _row(
        "vision_fps",
        "vision_fps",
        "fps",
        _VLM_VISION,
        from_run=lambda obj: getattr(obj, "vision_fps", None),
        from_runs_for_summary=lambda runs: [
            float(v) for r in runs if (v := getattr(r, "vision_fps", None)) is not None
        ],
    ),

    # --- LLM throughput/latency (all four LLM sections; llm_ prefix in VLM) ---
    _throughput_row(
        "prefill_tps",
        "prefill_tps",
        "tok/s",
        per_run_attr="prefill_tps",
        sweep_attr="prefill_sweep",
        sweep_values_attr="tps_values",
    ),
    _throughput_row(
        "decode_tps",
        "decode_tps",
        "tok/s",
        per_run_attr="decode_tps",
        sweep_attr="decode_sweep",
        sweep_values_attr="tps_values",
    ),
    _throughput_row(
        "ttft",
        "ttft",
        "ms",
        per_run_attr="prefill_latency",
        sweep_attr="prefill_sweep",
        sweep_values_attr="time_values",
        ms_scale=True,
    ),
    _throughput_row(
        "decode_duration",
        "decode_duration",
        "ms",
        per_run_attr="decode_duration",
        sweep_attr="decode_sweep",
        sweep_values_attr="time_values",
        ms_scale=True,
    ),

    # --- Total wall time (measure only; represents phase totals) ---
    _row(
        "total",
        "total",
        "ms",
        _MEASURE_LLM,
        from_run=_total_measure_from_run,
        from_runs_for_summary=_list_attr("total_time", scale=1000.0),
    ),

    # --- NPU latency ---
    _npu_latency_row("prefill_npu_lat", "prefill_npu_lat", "prefill_npu_latency_pct", "prefill_sweep"),
    _npu_latency_row("decode_npu_lat", "decode_npu_lat", "decode_npu_latency_pct", "decode_sweep"),
    _total_npu_latency_row(),

    # --- EAGLE-3 acceptance (LLM measure only) ---
    _accept_row("accept_steps", "accept_steps", "count", "acceptance_steps"),
    _accept_row("accept_tok_sum", "accept_tok_sum", "tok", "acceptance_tokens_sum"),
    _accept_row("accept_tok_avg", "accept_tok_avg", "tok", "acceptance_tokens_avg"),
    _accept_row("accept_ratio", "accept_ratio", "%", "acceptance_ratio", scale=100.0),

    # --- Device metrics: power ---
    _scalar_device_row("avg_power", "avg_power", "W", _ALL_LLM, "avg_power_w", llm_prefix=True),
    _scalar_device_row("p99_power", "p99_power", "W", _ALL_LLM, "p99_power_w", llm_prefix=True),
    _scalar_device_row(
        "prefill_avg_power", "prefill_avg_power", "W", _ALL_LLM, "prefill_avg_power_w", llm_prefix=True
    ),
    _scalar_device_row(
        "prefill_p99_power", "prefill_p99_power", "W", _ALL_LLM, "prefill_p99_power_w", llm_prefix=True
    ),
    _scalar_device_row(
        "decode_avg_power", "decode_avg_power", "W", _ALL_LLM, "decode_avg_power_w", llm_prefix=True
    ),
    _scalar_device_row(
        "decode_p99_power", "decode_p99_power", "W", _ALL_LLM, "decode_p99_power_w", llm_prefix=True
    ),
    _scalar_device_row("vision_avg_power", "vision_avg_power", "W", _VLM_VISION, "vision_avg_power_w"),
    _scalar_device_row("vision_p99_power", "vision_p99_power", "W", _VLM_VISION, "vision_p99_power_w"),

    # --- Device metrics: utilization ---
    _scalar_device_row("avg_util", "avg_util", "%", _ALL_LLM, "avg_utilization_pct", llm_prefix=True),
    _scalar_device_row("p99_util", "p99_util", "%", _ALL_LLM, "p99_utilization_pct", llm_prefix=True),
    _scalar_device_row(
        "prefill_avg_util", "prefill_avg_util", "%", _ALL_LLM, "prefill_avg_utilization_pct", llm_prefix=True
    ),
    _scalar_device_row(
        "prefill_p99_util", "prefill_p99_util", "%", _ALL_LLM, "prefill_p99_utilization_pct", llm_prefix=True
    ),
    _scalar_device_row(
        "decode_avg_util", "decode_avg_util", "%", _ALL_LLM, "decode_avg_utilization_pct", llm_prefix=True
    ),
    _scalar_device_row(
        "decode_p99_util", "decode_p99_util", "%", _ALL_LLM, "decode_p99_utilization_pct", llm_prefix=True
    ),
    _scalar_device_row("vision_avg_util", "vision_avg_util", "%", _VLM_VISION, "vision_avg_utilization_pct"),
    _scalar_device_row("vision_p99_util", "vision_p99_util", "%", _VLM_VISION, "vision_p99_utilization_pct"),

    # --- Device metrics: temperature ---
    _scalar_device_row("avg_temp", "avg_temp", "C", _ALL_LLM, "avg_temperature_c", llm_prefix=True),
    _scalar_device_row("p99_temp", "p99_temp", "C", _ALL_LLM, "p99_temperature_c", llm_prefix=True),
    _scalar_device_row(
        "prefill_avg_temp", "prefill_avg_temp", "C", _ALL_LLM, "prefill_avg_temperature_c", llm_prefix=True
    ),
    _scalar_device_row(
        "prefill_p99_temp", "prefill_p99_temp", "C", _ALL_LLM, "prefill_p99_temperature_c", llm_prefix=True
    ),
    _scalar_device_row(
        "decode_avg_temp", "decode_avg_temp", "C", _ALL_LLM, "decode_avg_temperature_c", llm_prefix=True
    ),
    _scalar_device_row(
        "decode_p99_temp", "decode_p99_temp", "C", _ALL_LLM, "decode_p99_temperature_c", llm_prefix=True
    ),
    _scalar_device_row("vision_avg_temp", "vision_avg_temp", "C", _VLM_VISION, "vision_avg_temperature_c"),
    _scalar_device_row("vision_p99_temp", "vision_p99_temp", "C", _VLM_VISION, "vision_p99_temperature_c"),

    # --- Device metrics: memory (MB) ---
    _scalar_device_row(
        "avg_mem_used", "avg_mem_used", "MB", _ALL_LLM, "avg_memory_used_mb", llm_prefix=True
    ),
    _scalar_device_row(
        "p99_mem_used", "p99_mem_used", "MB", _ALL_LLM, "p99_memory_used_mb", llm_prefix=True
    ),
    _scalar_device_row(
        "prefill_avg_mem_used",
        "prefill_avg_mem_used",
        "MB",
        _ALL_LLM,
        "prefill_avg_memory_used_mb",
        llm_prefix=True,
    ),
    _scalar_device_row(
        "prefill_p99_mem_used",
        "prefill_p99_mem_used",
        "MB",
        _ALL_LLM,
        "prefill_p99_memory_used_mb",
        llm_prefix=True,
    ),
    _scalar_device_row(
        "decode_avg_mem_used",
        "decode_avg_mem_used",
        "MB",
        _ALL_LLM,
        "decode_avg_memory_used_mb",
        llm_prefix=True,
    ),
    _scalar_device_row(
        "decode_p99_mem_used",
        "decode_p99_mem_used",
        "MB",
        _ALL_LLM,
        "decode_p99_memory_used_mb",
        llm_prefix=True,
    ),
    _scalar_device_row(
        "vision_avg_mem_used", "vision_avg_mem_used", "MB", _VLM_VISION, "vision_avg_memory_used_mb"
    ),
    _scalar_device_row(
        "vision_p99_mem_used", "vision_p99_mem_used", "MB", _VLM_VISION, "vision_p99_memory_used_mb"
    ),

    # --- Total memory (aggregate; no llm_ prefix; appears in every device section) ---
    _scalar_device_row(
        "total_mem", "total_mem", "MB", _ALL_DEVICE_SECTIONS, "total_memory_mb"
    ),

    # --- Device metrics: memory (%) ---
    _scalar_device_row(
        "avg_mem_used_pct", "avg_mem_used_pct", "%", _ALL_LLM, "avg_memory_used_pct", llm_prefix=True
    ),
    _scalar_device_row(
        "p99_mem_used_pct", "p99_mem_used_pct", "%", _ALL_LLM, "p99_memory_used_pct", llm_prefix=True
    ),
    _scalar_device_row(
        "prefill_avg_mem_used_pct",
        "prefill_avg_mem_used_pct",
        "%",
        _ALL_LLM,
        "prefill_avg_memory_used_pct",
        llm_prefix=True,
    ),
    _scalar_device_row(
        "prefill_p99_mem_used_pct",
        "prefill_p99_mem_used_pct",
        "%",
        _ALL_LLM,
        "prefill_p99_memory_used_pct",
        llm_prefix=True,
    ),
    _scalar_device_row(
        "decode_avg_mem_used_pct",
        "decode_avg_mem_used_pct",
        "%",
        _ALL_LLM,
        "decode_avg_memory_used_pct",
        llm_prefix=True,
    ),
    _scalar_device_row(
        "decode_p99_mem_used_pct",
        "decode_p99_mem_used_pct",
        "%",
        _ALL_LLM,
        "decode_p99_memory_used_pct",
        llm_prefix=True,
    ),
    _scalar_device_row(
        "vision_avg_mem_used_pct",
        "vision_avg_mem_used_pct",
        "%",
        _VLM_VISION,
        "vision_avg_memory_used_pct",
    ),
    _scalar_device_row(
        "vision_p99_mem_used_pct",
        "vision_p99_mem_used_pct",
        "%",
        _VLM_VISION,
        "vision_p99_memory_used_pct",
    ),

    # --- Energy ---
    _scalar_device_row(
        "prefill_energy", "prefill_energy", "J", _ALL_LLM, "prefill_energy_j", llm_prefix=True
    ),
    _scalar_device_row(
        "decode_energy", "decode_energy", "J", _ALL_LLM, "decode_energy_j", llm_prefix=True
    ),
    _scalar_device_row(
        "vision_energy", "vision_energy", "J", _VLM_VISION, "vision_energy_j"
    ),
    # llm_total_energy: LLM-only aggregate energy in VLM contexts.  Sourced
    # from ``llm_total_energy_j`` so it never accidentally reads the
    # vision+LLM combined ``total_energy_j``.  The label is literal (no
    # dynamic prefix) since it must be distinct from ``total_energy``.
    TpsRow(
        key="llm_total_energy",
        label="llm_total_energy",
        unit="J",
        sections=frozenset(_VLM_LLM),
        device_metric=True,
        llm_prefix=False,
        sweep_suffix=False,
        from_run=_direct_attr("llm_total_energy_j"),
        from_aggregate=_direct_attr("llm_total_energy_j"),
        from_runs_for_summary=_list_attr("llm_total_energy_j"),
    ),
    # total_energy: LLM+vision total in VLM measure; LLM-only total in LLM
    # measure/sweep. Absent from vlm_sweep_llm since that subtable has no
    # vision component.
    _scalar_device_row(
        "total_energy",
        "total_energy",
        "J",
        (SECTION_LLM_MEASURE, SECTION_LLM_SWEEP, SECTION_VLM_MEASURE),
        "total_energy_j",
    ),

    # --- Efficiency (per-phase, phase-wide aggregate) ---
    _scalar_device_row(
        "prefill_tps_per_w",
        "prefill_tps_per_w",
        "tok/s/W",
        _ALL_LLM,
        "prefill_tps_per_w",
        llm_prefix=True,
    ),
    _scalar_device_row(
        "decode_tps_per_w",
        "decode_tps_per_w",
        "tok/s/W",
        _ALL_LLM,
        "decode_tps_per_w",
        llm_prefix=True,
    ),
    _scalar_device_row(
        "prefill_j_per_tok",
        "prefill_j_per_tok",
        "J/tok",
        _ALL_LLM,
        "prefill_j_per_token",
        llm_prefix=True,
    ),
    _scalar_device_row(
        "decode_j_per_tok",
        "decode_j_per_tok",
        "J/tok",
        _ALL_LLM,
        "decode_j_per_token",
        llm_prefix=True,
    ),
    _scalar_device_row(
        "vision_img_per_j", "vision_img_per_j", "img/J", _VLM_VISION, "vision_img_per_j"
    ),
    _scalar_device_row(
        "vision_j_per_img", "vision_j_per_img", "J/img", _VLM_VISION, "vision_j_per_img"
    ),
]


def label_for(row: TpsRow, section: str) -> str:
    """Return the display label for ``row`` when emitted in ``section``."""
    label = row.label
    if row.llm_prefix and section in _VLM_SECTIONS:
        label = f"llm_{label}"
    if row.sweep_suffix and section in _SWEEP_SECTIONS:
        label = f"{label}(last)"
    return label


def json_key_for(row: TpsRow, section: str, *, is_summary: bool = False) -> str:
    """Return the JSON key for ``row`` when emitted in ``section``.

    Same ``llm_`` prefix rule as :func:`label_for`.  The ``_last`` suffix
    is reserved for summary scalars: it fires only when ``is_summary=True``
    and the row has ``sweep_suffix`` set, mirroring the CHANGELOG 2.3.0
    contract where runs/aggregate curves surface at the top level with
    bare names (``prefill_tps``, ``llm_ttft``) and the summary section's
    ``_last`` keys mark scalar statistics blocks derived from the last
    sweep point.
    """
    key = row.label
    if row.llm_prefix and section in _VLM_SECTIONS:
        key = f"llm_{key}"
    if is_summary and row.sweep_suffix and section in _SWEEP_SECTIONS:
        key = f"{key}_last"
    return key


def iter_section_rows(section: str, *, device_metrics: bool) -> list[TpsRow]:
    """Return the rows that should be emitted in ``section`` (CLI table view)."""
    rows: list[TpsRow] = []
    for row in TPS_TABLE_SPEC:
        if section not in row.sections:
            continue
        if row.device_metric and not device_metrics:
            continue
        rows.append(row)
    return rows


def iter_json_rows(section: str) -> list[TpsRow]:
    """Return the rows that should appear in the JSON output for ``section``.

    Unlike :func:`iter_section_rows`, this does *not* apply the
    ``device_metrics`` gate — JSON payloads always dump whatever the run
    produced; the CLI table is what respects ``--device-metrics``.
    """
    return [row for row in TPS_TABLE_SPEC if section in row.sections]


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


def render_units(section: str, keys_present: Optional[set[str]] = None) -> dict[str, str]:
    """Return a ``{canonical_key: unit}`` dict for JSON's ``units`` metadata.

    ``keys_present``, when provided, filters the output to canonical keys
    that actually appear in the JSON payload — this drops units for rows
    whose data was omitted because the run did not produce it.  Keys use
    the bare (non-summary) form since units apply equally to the top-level
    curves and to the derived ``_last`` summary scalars.
    """
    out: dict[str, str] = {}
    for row in iter_json_rows(section):
        key = json_key_for(row, section)
        if keys_present is not None and key not in keys_present:
            continue
        out[key] = row.unit
    return out


def render_summary_json(
    section: str,
    values_by_key: Mapping[str, Sequence[float]],
    summary_fn: Callable[[Sequence[float]], dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Build the ``summary`` JSON block for ``section``.

    ``values_by_key`` is the same canonical-keyed dict used to drive the CLI
    table.  Sweep-suffix rows render with the ``_last`` suffix here — the
    summary block is the only place that suffix appears (see
    :func:`json_key_for`).
    """
    out: dict[str, dict[str, float]] = {}
    for row in iter_json_rows(section):
        values = values_by_key.get(row.key)
        if values is None:
            continue
        out[json_key_for(row, section, is_summary=True)] = summary_fn(values)
    return out


def render_summary_json_from_runs(
    section: str,
    runs: Sequence[Any],
    summary_fn: Callable[[Sequence[float]], dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Build the ``summary`` block by invoking each row's ``from_runs_for_summary``.

    Rows without ``from_runs_for_summary`` are skipped.  Emits summary-form
    keys (``_last`` suffix for sweep-suffix rows).
    """
    out: dict[str, dict[str, float]] = {}
    for row in iter_json_rows(section):
        if row.from_runs_for_summary is None:
            continue
        values = row.from_runs_for_summary(runs)
        out[json_key_for(row, section, is_summary=True)] = summary_fn(values)
    return out


def render_run_json(section: str, run: Any) -> dict[str, Any]:
    """Build a per-run canonical projection for ``run``.

    Missing/None extractions are skipped, matching the CLI table gating.
    Sweep-curve rows surface with bare names (no ``_last`` suffix) — that
    suffix is reserved for summary scalars.
    """
    out: dict[str, Any] = {}
    for row in iter_json_rows(section):
        if row.from_run is None:
            continue
        value = row.from_run(run)
        if value is None:
            continue
        # Skip empty curves too: they carry no information and mirror the
        # CLI table skipping rows with no data.
        if isinstance(value, list) and not value:
            continue
        out[json_key_for(row, section)] = value
    return out


def render_aggregate_json(section: str, aggregate: Any) -> dict[str, Any]:
    """Build a canonical projection of the aggregate object for ``section``."""
    out: dict[str, Any] = {}
    for row in iter_json_rows(section):
        if row.from_aggregate is None:
            continue
        value = row.from_aggregate(aggregate)
        if value is None:
            continue
        if isinstance(value, list) and not value:
            continue
        out[json_key_for(row, section)] = value
    return out
