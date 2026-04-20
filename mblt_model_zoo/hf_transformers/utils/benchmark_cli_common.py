from __future__ import annotations

import argparse
from typing import Any, Optional

DEVICE_TRACKER_INTERVAL_SEC = 1.0
DEVICE_BACKEND_CHOICES = ("none", "auto", "gpu", "npu")
DEFAULT_DEVICE_BACKEND = "none"
CORE_MODE_SWEEP_VALUES = ("single", "global4", "global8")
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


def parse_int_list_optional(spec: str | None) -> list[int] | None:
    if spec is None:
        return None
    text = spec.strip()
    if not text:
        return None
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        return None
    if any(v < 0 for v in values):
        raise argparse.ArgumentTypeError("device-gpu-id values must be >= 0")
    return values


def iter_core_modes(core_mode: str | None) -> list[str | None]:
    if core_mode == "all":
        return list(CORE_MODE_SWEEP_VALUES)
    return [core_mode]


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
) -> dict[str, Any]:
    if not core_mode:
        return model_kwargs

    model_kwargs["core_mode"] = core_mode
    if core_mode == "single":
        model_kwargs["target_cores"] = ["0:0"]
    elif core_mode == "global4":
        model_kwargs["target_clusters"] = [0]
    elif core_mode == "global8":
        model_kwargs["target_clusters"] = [0, 1]
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


def add_pipeline_device_args(
    parser: argparse.ArgumentParser,
    *,
    device_default: str | None = None,
    trust_remote_code_default: bool = True,
) -> None:
    parser.add_argument(
        "--device",
        default=device_default,
        help='pipeline device (default: None; e.g., "cpu", "cuda:0")',
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


def build_device_tracker(args: argparse.Namespace, pipeline: Any):
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

        try:
            return NPUDeviceTracker(interval=DEVICE_TRACKER_INTERVAL_SEC)
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


def build_phase_trackers(args: argparse.Namespace, pipeline: Any) -> tuple[Any, Any]:
    if not args.device_metrics:
        return None, None
    return build_device_tracker(args, pipeline), build_device_tracker(args, pipeline)


def stop_tracker_safe(tracker: Any) -> None:
    if tracker is None:
        return
    try:
        tracker.stop()
    except Exception:
        pass


def extract_device_metric(tracker: Any) -> dict[str, float | None]:
    metric = tracker.get_metric()
    out: dict[str, float | None] = {}
    for key in DEVICE_METRIC_KEYS:
        val = metric.get(key)
        out[key] = float(val) if isinstance(val, (int, float)) else None
    return out


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


def print_device_status(args: argparse.Namespace, tracker: Any) -> None:
    if not args.device_metrics:
        print("[device] disabled by --no-device-metrics")
        return
    if tracker is None:
        print("[device] enabled but no compatible tracker initialized (auto detection fallback)")
        return
    print(f"[device] enabled with {tracker.__class__.__name__} (interval={DEVICE_TRACKER_INTERVAL_SEC}s fixed)")
