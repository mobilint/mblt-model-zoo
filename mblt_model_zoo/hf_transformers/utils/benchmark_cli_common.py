from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional, Protocol, TypeAlias

DEVICE_TRACKER_INTERVAL_SEC = 1.0
DEVICE_BACKEND_CHOICES = ("none", "auto", "gpu", "npu")
DEFAULT_DEVICE_BACKEND = "none"
NPU_RAIL_METRIC_CHOICES = ("npu", "ddr", "pmic", "goldfinger")
CORE_MODE_CHOICES = ("single", "global4", "global8")
CORE_MODE_SWEEP_VALUES = ("single", "global4", "global8")
DEFAULT_SINGLE_TARGET_CORES = ("0:0",)
DEVICE_METRIC_KEYS = (
    "avg_power_w",
    "p99_power_w",
    "avg_utilization_pct",
    "p99_utilization_pct",
    "avg_temperature_c",
    "p99_temperature_c",
    "avg_memory_used_mb",
    "p99_memory_used_mb",
    "total_memory_mb",
    "avg_memory_used_pct",
    "p99_memory_used_pct",
)

DeviceMetricValue: TypeAlias = float | None
DeviceMetricMap: TypeAlias = dict[str, DeviceMetricValue]
DeviceTracePoint: TypeAlias = dict[str, float]
DeviceTimeSeriesMap: TypeAlias = dict[str, list[DeviceTracePoint]]
NpuRailMetrics: TypeAlias = str | list[str]
RawDeviceMetricMap: TypeAlias = Mapping[str, object]


class DeviceTracker(Protocol):
    """Runtime protocol for mblt-tracker 1.x-compatible benchmark helpers.

    Optional trace methods are intentionally discovered dynamically because GPU, NPU, fake, and
    future tracker implementations do not expose identical time-series APIs.
    """

    def start(self) -> None:
        """Start device metric collection."""

    def stop(self) -> None:
        """Stop device metric collection."""

    def get_metric(self) -> RawDeviceMetricMap:
        """Return raw metrics from mblt-tracker."""


def parse_positive_int(spec: str) -> int:
    try:
        value = int(spec)
    except ValueError as e:
        raise argparse.ArgumentTypeError("expected integer") from e
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def parse_positive_int_optional(spec: str | None) -> int | None:
    if spec is None:
        return None
    text = str(spec).strip()
    if not text:
        return None
    return parse_positive_int(text)


def parse_non_negative_int_list_optional(spec: str | None, *, name: str = "device id") -> list[int] | None:
    """Parse a comma-separated list of non-negative integer device IDs."""
    if spec is None:
        return None
    text = spec.strip()
    if not text:
        return None
    try:
        values = [int(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{name} values must be integers") from e
    if not values:
        return None
    if any(v < 0 for v in values):
        raise argparse.ArgumentTypeError(f"{name} values must be >= 0")
    return values


def parse_int_list_optional(spec: str | None) -> list[int] | None:
    """Parse comma-separated GPU tracker IDs for backward-compatible callers."""
    return parse_non_negative_int_list_optional(spec, name="device-gpu-id")


def parse_npu_rail_metrics(spec: str | None) -> NpuRailMetrics:
    """Parse NPU rail metrics for mblt-tracker NPU device tracking.

    Returns ``"npu"`` for the default low-latency rail, ``"all"`` for every supported rail, or a
    de-duplicated list of rail names when a comma-separated subset is provided.
    """
    if spec is None:
        return "npu"
    text = str(spec).strip().lower()
    if not text:
        return "npu"
    if text == "all":
        return "all"

    rails: list[str] = []
    valid_rails = set(NPU_RAIL_METRIC_CHOICES)
    for raw_rail in text.split(","):
        rail = raw_rail.strip()
        if not rail:
            raise argparse.ArgumentTypeError("device-npu-rail-metrics values must not be empty")
        if rail not in valid_rails:
            choices = ", ".join((*NPU_RAIL_METRIC_CHOICES, "all"))
            raise argparse.ArgumentTypeError(f"unknown NPU rail metric '{rail}' (expected one of: {choices})")
        if rail not in rails:
            rails.append(rail)

    if not rails:
        return "npu"
    if len(rails) == 1:
        return rails[0]
    return rails


def iter_core_modes(core_mode: str | None) -> list[str | None]:
    if core_mode == "all":
        return list(CORE_MODE_SWEEP_VALUES)
    return [core_mode]


def is_mobilint_target(model_id: str | None, mxq_path: str | None = None, mxq_dir: str | None = None) -> bool:
    """Return whether a target should use Mobilint-oriented runtime defaults."""
    if mxq_path or mxq_dir:
        return True
    return bool(model_id and str(model_id).strip().startswith("mobilint/"))


def resolve_default_device(
    *,
    device: str | None,
    device_explicit: bool,
    model_id: str | None,
    mxq_path: str | None = None,
    mxq_dir: str | None = None,
    original_models: bool = False,
) -> str | None:
    """Resolve a pipeline device while preserving explicit user input."""
    if device_explicit:
        return device
    if original_models:
        return "cuda:0"
    if is_mobilint_target(model_id, mxq_path=mxq_path, mxq_dir=mxq_dir):
        return "cpu"
    return device if device is not None else "cpu"


def resolve_default_device_backend(
    *,
    device_backend: str,
    device_backend_explicit: bool,
    model_id: str | None,
    mxq_path: str | None = None,
    mxq_dir: str | None = None,
    original_models: bool = False,
) -> str:
    """Resolve a device-metric backend while preserving explicit user input."""
    if device_backend_explicit:
        return device_backend
    if original_models:
        return "auto"
    if is_mobilint_target(model_id, mxq_path=mxq_path, mxq_dir=mxq_dir):
        return "npu"
    return DEFAULT_DEVICE_BACKEND


def append_core_mode_suffix(
    label: str,
    base: str,
    core_mode: str | None,
) -> tuple[str, str]:
    if not core_mode:
        return label, base
    suffix = f"-{core_mode}"
    return f"{label}{suffix}", f"{base}{suffix}"


def apply_core_mode_model_kwargs(
    model_kwargs: dict[str, Any],
    core_mode: str | None,
    *,
    prefix: str | None = None,
    target_cores: list[str] | None = None,
    target_clusters: list[int] | None = None,
    default_single_target_cores: Sequence[str] | None = DEFAULT_SINGLE_TARGET_CORES,
) -> dict[str, Any]:
    key_prefix = f"{prefix}_" if prefix else ""
    if not core_mode:
        if target_cores is not None:
            model_kwargs[f"{key_prefix}target_cores"] = target_cores
        if target_clusters is not None:
            model_kwargs[f"{key_prefix}target_clusters"] = target_clusters
        return model_kwargs

    model_kwargs[f"{key_prefix}core_mode"] = core_mode
    if target_cores is not None:
        model_kwargs[f"{key_prefix}target_cores"] = target_cores
    elif core_mode == "single" and default_single_target_cores is not None:
        model_kwargs[f"{key_prefix}target_cores"] = list(default_single_target_cores)

    if target_clusters is not None:
        model_kwargs[f"{key_prefix}target_clusters"] = target_clusters
    elif not target_cores and core_mode == "global4":
        model_kwargs[f"{key_prefix}target_clusters"] = [0]
    elif not target_cores and core_mode == "global8":
        model_kwargs[f"{key_prefix}target_clusters"] = [0, 1]
    return model_kwargs


def infer_gpu_ids(device: str | None, device_gpu_id: Optional[list[int]]) -> Optional[int | list[int]]:
    if device_gpu_id is not None:
        return device_gpu_id[0] if len(device_gpu_id) == 1 else device_gpu_id
    text = (device or "").strip().lower()
    if text.startswith("cuda:"):
        try:
            gpu_id = int(text.split(":", 1)[1])
            return gpu_id
        except ValueError:
            return None
    return None


def infer_npu_ids(device_npu_id: list[int] | None) -> int | list[int] | None:
    """Return NPU IDs in the shape accepted by mblt-tracker."""
    if device_npu_id is None:
        return None
    return device_npu_id[0] if len(device_npu_id) == 1 else device_npu_id


def add_device_tracking_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable device metrics tracking (default: on, disable via --no-device-metrics)",
    )
    parser.add_argument(
        "--device-backend",
        choices=list(DEVICE_BACKEND_CHOICES),
        default=DEFAULT_DEVICE_BACKEND,
        help=f"device backend selection (default: {DEFAULT_DEVICE_BACKEND})",
    )
    parser.add_argument(
        "--device-gpu-id",
        type=parse_int_list_optional,
        default=None,
        help="comma-separated GPU ids for device tracking (e.g., 0,1)",
    )
    parser.add_argument(
        "--device-npu-id",
        type=lambda spec: parse_non_negative_int_list_optional(spec, name="device-npu-id"),
        default=None,
        help="comma-separated NPU logical card ids for device tracking (e.g., 0,1)",
    )
    parser.add_argument(
        "--device-npu-rail-metrics",
        type=parse_npu_rail_metrics,
        default="npu",
        help=(
            "NPU rail metrics to collect with mblt-tracker: npu, ddr, pmic, goldfinger, all, or a "
            "comma-separated combination (default: npu)"
        ),
    )


def add_pipeline_device_args(
    parser: argparse.ArgumentParser,
    *,
    device_default: str | None = None,
    trust_remote_code_default: bool = True,
) -> None:
    parser.add_argument(
        "--device",
        default=device_default,
        help='pipeline device (e.g., "cpu", "cuda:0")',
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help='pipeline device_map (e.g., "auto")',
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help='dtype for pipeline (e.g., "float16", "bfloat16")',
    )
    parser.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        default=trust_remote_code_default,
        help="whether to trust remote code when loading from HF",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
    )


def build_device_tracker(args: argparse.Namespace, pipeline: Any) -> DeviceTracker | None:
    if not args.device_metrics:
        return None

    def _has_npu_backend(obj: Any, depth: int = 0, seen: Optional[set[int]] = None) -> bool:
        if obj is None:
            return False
        if seen is None:
            seen = set()
        oid = id(obj)
        if oid in seen:
            return False
        seen.add(oid)
        if hasattr(obj, "npu_backend"):
            return True
        if depth >= 2:
            return False
        for name in ("model", "language_model", "vision_model", "text_model", "encoder", "decoder"):
            child = getattr(obj, name, None)
            if child is not None and _has_npu_backend(child, depth + 1, seen):
                return True
        return False

    is_mobilint_model = False
    try:
        from mblt_model_zoo.hf_transformers.utils.modeling_utils import MobilintModelMixin

        is_mobilint_model = isinstance(pipeline.model, MobilintModelMixin) or _has_npu_backend(pipeline.model)
    except Exception:
        is_mobilint_model = _has_npu_backend(getattr(pipeline, "model", None))

    backend = args.device_backend
    if backend == "auto":
        if is_mobilint_model:
            backend = "npu"
        else:
            device_text = (args.device or "").strip().lower()
            backend = "gpu" if device_text.startswith("cuda") else "none"

    if backend == "none":
        return None

    if backend == "npu":
        from mblt_tracker import NPUDeviceTracker

        npu_id = infer_npu_ids(getattr(args, "device_npu_id", None))
        rail_metrics = getattr(args, "device_npu_rail_metrics", "npu")
        try:
            return NPUDeviceTracker(
                interval=DEVICE_TRACKER_INTERVAL_SEC,
                npu_id=npu_id,
                rail_metrics=rail_metrics,
            )
        except Exception as e:
            print(f"[device] failed to initialize NPU tracker: {e}")
            return None

    if backend == "gpu":
        from mblt_tracker import GPUDeviceTracker

        gpu_id = infer_gpu_ids(args.device, args.device_gpu_id)
        try:
            return GPUDeviceTracker(interval=DEVICE_TRACKER_INTERVAL_SEC, gpu_id=gpu_id)
        except Exception as e:
            print(f"[device] failed to initialize GPU tracker: {e}")
            return None

    return None


def build_phase_trackers(args: argparse.Namespace, pipeline: Any) -> tuple[DeviceTracker | None, DeviceTracker | None]:
    if not args.device_metrics:
        return None, None
    return build_device_tracker(args, pipeline), build_device_tracker(args, pipeline)


def stop_tracker_safe(tracker: DeviceTracker | None) -> None:
    if tracker is None:
        return
    try:
        tracker.stop()
    except Exception:
        pass


def extract_device_metric(tracker: DeviceTracker) -> DeviceMetricMap:
    """Normalize raw mblt-tracker metrics to the benchmark scalar schema."""
    metric = tracker.get_metric()
    out: DeviceMetricMap = {}
    for key in DEVICE_METRIC_KEYS:
        val = metric.get(key)
        out[key] = float(val) if isinstance(val, (int, float)) else None
    return out


def integrate_power_trace_j(power_trace: Sequence[Mapping[str, object]]) -> float | None:
    """Integrate a power time series into energy in joules.

    Args:
        power_trace: Trace points containing ``timestamp_s`` and ``value`` in watts.

    Returns:
        Energy in joules computed with the trapezoidal rule, or ``None`` when fewer than two valid
        monotonic points are available.
    """
    points: list[tuple[float, float]] = []
    for item in power_trace:
        timestamp_s = item.get("timestamp_s")
        value = item.get("value")
        if isinstance(timestamp_s, (int, float)) and isinstance(value, (int, float)):
            points.append((float(timestamp_s), float(value)))
    if len(points) < 2:
        print(
            "[device] warning: at least two power trace points are required for trace-integrated energy; "
            "short measurements may leave energy-efficiency metrics empty."
        )
        return None

    points.sort(key=lambda point: point[0])
    energy_j = 0.0
    valid_segments = 0
    for (left_t, left_w), (right_t, right_w) in zip(points, points[1:]):
        delta_s = right_t - left_t
        if delta_s <= 0.0:
            continue
        energy_j += ((left_w + right_w) / 2.0) * delta_s
        valid_segments += 1
    return energy_j if valid_segments else None


def energy_from_device_time_series(
    device_time_series: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    power_key: str = "power_w",
) -> float | None:
    """Return integrated energy from a benchmark device time-series payload."""
    power_trace = device_time_series.get(power_key)
    if power_trace is None:
        return None
    return integrate_power_trace_j(power_trace)


def add_trace_energy_to_device_metric(
    device_metric: Mapping[str, DeviceMetricValue],
    device_time_series: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    power_key: str = "power_w",
) -> DeviceMetricMap:
    """Return device metrics augmented with trace-integrated ``total_energy_j``."""
    augmented: DeviceMetricMap = dict(device_metric)
    augmented["total_energy_j"] = energy_from_device_time_series(device_time_series, power_key=power_key)
    return augmented


def _normalize_trace(raw_trace: object) -> list[DeviceTracePoint]:
    """Convert mblt-tracker trace tuples into JSON-safe time-series points."""
    if not isinstance(raw_trace, list):
        return []
    out: list[DeviceTracePoint] = []
    for item in raw_trace:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        timestamp_s, value = item
        if isinstance(timestamp_s, (int, float)) and isinstance(value, (int, float)):
            out.append({"timestamp_s": float(timestamp_s), "value": float(value)})
    return out


def _trace_from_method(tracker: DeviceTracker, method_name: str) -> list[DeviceTracePoint]:
    method = getattr(tracker, method_name, None)
    if not callable(method):
        return []
    try:
        return _normalize_trace(method())
    except Exception as exc:
        print(f"[device] warning: failed to read tracker trace via {method_name}: {exc}")
        return []


def extract_device_time_series(tracker: DeviceTracker) -> DeviceTimeSeriesMap:
    """Extract JSON-safe device metric time-series from mblt-tracker 1.x trackers."""
    trace_getters: dict[str, Callable[[], list[DeviceTracePoint]]] = {
        "power_w": lambda: _trace_from_method(tracker, "get_total_power_trace"),
        "utilization_pct": lambda: _trace_from_method(tracker, "get_total_utilization_trace"),
        "temperature_c": lambda: _trace_from_method(tracker, "get_temperature_trace"),
        "memory_used_mb": lambda: _trace_from_method(tracker, "get_memory_used_trace"),
        "memory_used_pct": lambda: _trace_from_method(tracker, "get_memory_used_pct_trace"),
        "npu_power_w": lambda: _trace_from_method(tracker, "get_npu_rail_power_trace"),
        "ddr_power_w": lambda: _trace_from_method(tracker, "get_ddr_rail_power_trace"),
        "pmic_power_w": lambda: _trace_from_method(tracker, "get_pmic_rail_power_trace"),
        "goldfinger_power_w": lambda: _trace_from_method(tracker, "get_goldfinger_rail_power_trace"),
    }
    return {key: getter() for key, getter in trace_getters.items()}


def weighted_two(
    a: float | None,
    a_weight: float,
    b: float | None,
    b_weight: float,
) -> float | None:
    values = []
    weights = []
    if a is not None and a_weight > 0:
        values.append(float(a))
        weights.append(float(a_weight))
    if b is not None and b_weight > 0:
        values.append(float(b))
        weights.append(float(b_weight))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    total_w = weights[0] + weights[1]
    if total_w <= 0:
        return None
    return (values[0] * weights[0] + values[1] * weights[1]) / total_w


def print_device_status(args: argparse.Namespace, tracker: DeviceTracker | None) -> None:
    if not args.device_metrics:
        print("[device] disabled by --no-device-metrics")
        return
    if tracker is None:
        print("[device] enabled but no compatible tracker initialized (auto detection fallback)")
        return
    print(f"[device] enabled with {tracker.__class__.__name__} (interval={DEVICE_TRACKER_INTERVAL_SEC}s fixed)")
