"""Probe device HBM saturation from replicated ``qbruntime.Model`` loads.

Purpose
-------
Each ``qbruntime.Model(mxq_path, ...)`` allocation places a fresh copy of the
MXQ weights on device HBM; the runtime does not deduplicate identical MXQ
paths. A candidate ``MobilintNPUBackend`` redesign wants ``N`` ``Model``
instances (up to some max) per device. Users need to know the ``N`` at which
HBM saturates on a given device+MXQ so they can pick a safe
``max_batch_size`` without hitting ``QbRuntimeError`` at runtime.

This probe walks ``N = 1, 2, 4, 8, ...`` up to ``--n-models-max``, allocating
one additional ``Model`` per step and sampling device memory before and after
each ``launch()``. The first step that raises ``QbRuntimeError`` (or any other
runtime error) is recorded as the BadAlloc boundary; every previously launched
handle is then disposed and the probe advances to the next requested device.

Memory is sampled through ``mblt-status -q -s MEMORY -i <dev>``, if it is
available on ``PATH``. When it is not, ``memory_before_mb`` and
``memory_after_mb`` are left ``null`` and the delta is inferred only when both
samples exist.

Result usage
------------
The ``bad_alloc_at_n`` per device is the first ``N`` that failed. The largest
successful ``N`` (last row with ``ok == true``) is the safe upper bound for
that device+MXQ+core_mode combination.

Non-goals
---------
No inference is exercised beyond ``launch``; timing here is orthogonal to the
parallel probe.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from qbruntime import Accelerator, Cluster, Core, CoreId, Model, ModelConfig, QbRuntimeError

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


_STATUS_EXE = "mblt-status"
_USAGE_RE = re.compile(r"Usage\s*:\s*(\d+(?:\.\d+)?)\s*MB", re.IGNORECASE)
_TOTAL_RE = re.compile(r"Total\s*:\s*(\d+(?:\.\d+)?)\s*MB", re.IGNORECASE)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the HBM-saturation probe."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mxq-path", required=True, help="Path to the MXQ artifact to load repeatedly.")
    parser.add_argument(
        "--dev-no",
        default="0",
        help="Comma-separated list of accelerator device numbers to sweep, e.g. '0' or '0,1'.",
    )
    parser.add_argument(
        "--core-mode",
        choices=("single", "global4", "global8"),
        default="single",
        help=(
            "Core mode applied to every Model. 'single' packs many Models onto shared cores; "
            "'global4' allocates a 4-core cluster per Model; 'global8' allocates all 8 cores."
        ),
    )
    parser.add_argument(
        "--n-schedule",
        default="1,2,4,8,16,32",
        help=(
            "Comma-separated schedule of cumulative N values to launch, e.g. '1,2,4,8,16'. "
            "The probe launches (N_i - N_(i-1)) additional Models per step."
        ),
    )
    parser.add_argument(
        "--n-models-max",
        type=int,
        default=32,
        help="Hard upper bound on total launched Models per device, independent of --n-schedule.",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.2,
        help="Sleep between the last launch of a step and the memory reading.",
    )
    parser.add_argument(
        "--output-dir",
        default="debug/multi_model_hbm_probe",
        help="Directory that receives probe_report.json.",
    )
    return parser.parse_args()


def _parse_int_list(raw: str, name: str) -> list[int]:
    """Parse a comma-separated int list; raise on empty."""
    values: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            values.append(int(tok))
        except ValueError as e:
            raise SystemExit(f"Invalid {name} entry {tok!r}: {e}") from None
    if not values:
        raise SystemExit(f"{name} resolved to an empty list.")
    return values


def _resolve_status_exe() -> Optional[str]:
    """Return an absolute path to ``mblt-status`` on ``PATH``, or ``None``."""
    return shutil.which(_STATUS_EXE)


def _read_memory_mb(dev_no: int, exe: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    """Return ``(usage_mb, total_mb)`` for ``dev_no`` from ``mblt-status``.

    Both fields fall back to ``None`` when the CLI is unavailable, the query
    fails, or the output cannot be parsed.
    """
    if exe is None:
        return None, None
    try:
        proc = subprocess.run(
            [exe, "-q", "-s", "MEMORY", "-i", str(dev_no)],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None, None
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    usage_match = _USAGE_RE.search(text)
    total_match = _TOTAL_RE.search(text)
    usage = float(usage_match.group(1)) if usage_match else None
    total = float(total_match.group(1)) if total_match else None
    return usage, total


def _default_cores() -> list["CoreId"]:
    """Return every ``CoreId`` on the accelerator (both clusters, four cores each)."""
    return [
        CoreId(Cluster.Cluster0, Core.Core0),
        CoreId(Cluster.Cluster0, Core.Core1),
        CoreId(Cluster.Cluster0, Core.Core2),
        CoreId(Cluster.Cluster0, Core.Core3),
        CoreId(Cluster.Cluster1, Core.Core0),
        CoreId(Cluster.Cluster1, Core.Core1),
        CoreId(Cluster.Cluster1, Core.Core2),
        CoreId(Cluster.Cluster1, Core.Core3),
    ]


def _make_model_config(core_mode: str) -> "ModelConfig":
    """Build a saturating shared-resource ``ModelConfig`` for the HBM probe.

    Every ``Model`` launched by this probe is intentionally handed the *same*
    resource pool: ``single`` claims all 8 cores, ``global4`` claims both
    clusters, ``global8`` claims all 8 cores. The runtime is free to serialize
    ``.infer`` calls on those shared resources; this probe cares about
    ``launch()`` succeeding, not concurrency.
    """
    mc = ModelConfig()
    if core_mode == "single":
        mc.set_single_core_mode(None, _default_cores())
    elif core_mode == "global4":
        mc.set_global4_core_mode([Cluster.Cluster0, Cluster.Cluster1])
    elif core_mode == "global8":
        mc.set_global8_core_mode()
    else:
        raise SystemExit(f"Unsupported core_mode: {core_mode}")
    return mc


def _sanitize_schedule(schedule: list[int], hard_max: int) -> list[int]:
    """Return a deduped, sorted, ``hard_max``-clamped schedule (>=1 only)."""
    kept = sorted({n for n in schedule if 1 <= n <= hard_max})
    if not kept:
        raise SystemExit(
            f"--n-schedule empty after clamping to [1, {hard_max}]; check --n-schedule / --n-models-max."
        )
    return kept


def _launch_up_to(
    acc: "Accelerator",
    models: list["Model"],
    target_n: int,
    mxq_path: str,
    core_mode: str,
) -> tuple[bool, Optional[str]]:
    """Launch enough new ``Model`` instances to reach ``target_n`` cumulative handles.

    Returns:
        ``(ok, error)``. When ``ok`` is ``False``, ``error`` is a short message
        identifying the failure.
    """
    while len(models) < target_n:
        mc = _make_model_config(core_mode)
        try:
            mm = Model(mxq_path, mc)
            mm.launch(acc)
        except QbRuntimeError as e:
            return False, f"QbRuntimeError: {e}"
        except Exception as e:  # noqa: BLE001 — surface class + message for the report
            return False, f"{type(e).__name__}: {e}"
        models.append(mm)
    return True, None


def _dispose_all(models: list["Model"]) -> None:
    """Best-effort ``dispose()`` of every model; swallow individual failures."""
    for mm in models:
        try:
            mm.dispose()
        except Exception as e:  # noqa: BLE001 — release path must not raise
            print(f"warning: dispose failed: {e}", file=sys.stderr)
    models.clear()


def _sweep_device(dev_no: int, args: argparse.Namespace, status_exe: Optional[str]) -> dict[str, Any]:
    """Walk the ``--n-schedule`` on one device until BadAlloc or schedule end."""
    schedule = _sanitize_schedule(
        _parse_int_list(args.n_schedule, "--n-schedule"),
        int(args.n_models_max),
    )

    if args.core_mode == "global8":
        # global8 claims every core, so more than one Model on the same accelerator is impossible;
        # honor the request but keep only the head of the schedule.
        schedule = [n for n in schedule if n == 1]
        if not schedule:
            schedule = [1]

    baseline_usage, total_mb = _read_memory_mb(dev_no, status_exe)

    acc = Accelerator(int(dev_no))
    models: list["Model"] = []
    rows: list[dict[str, Any]] = []
    bad_alloc_at_n: Optional[int] = None
    bad_alloc_error: Optional[str] = None

    try:
        for target_n in schedule:
            usage_before, total_before = _read_memory_mb(dev_no, status_exe)
            t0 = time.perf_counter()
            ok, err = _launch_up_to(acc, models, target_n, args.mxq_path, args.core_mode)
            launch_wall_s = time.perf_counter() - t0
            if not ok:
                bad_alloc_at_n = target_n
                bad_alloc_error = err
                rows.append(
                    {
                        "n_models": target_n,
                        "n_launched": len(models),
                        "ok": False,
                        "error": err,
                        "memory_before_mb": usage_before,
                        "memory_after_mb": None,
                        "memory_delta_mb": None,
                        "launch_wall_s": launch_wall_s,
                    }
                )
                break
            time.sleep(max(0.0, float(args.settle_s)))
            usage_after, total_after = _read_memory_mb(dev_no, status_exe)
            delta = (
                (usage_after - usage_before)
                if (usage_after is not None and usage_before is not None)
                else None
            )
            rows.append(
                {
                    "n_models": target_n,
                    "n_launched": len(models),
                    "ok": True,
                    "error": None,
                    "memory_before_mb": usage_before,
                    "memory_after_mb": usage_after,
                    "memory_delta_mb": delta,
                    "launch_wall_s": launch_wall_s,
                }
            )
            print(
                f"dev={dev_no} n={target_n}  usage={usage_after}MB "
                f"delta={delta}MB  launch_time={launch_wall_s:.2f}s"
            )
    finally:
        _dispose_all(models)

    return {
        "dev_no": int(dev_no),
        "total_mb": total_mb,
        "baseline_usage_mb": baseline_usage,
        "schedule_requested": schedule,
        "rows": rows,
        "bad_alloc_at_n": bad_alloc_at_n,
        "bad_alloc_error": bad_alloc_error,
        "largest_ok_n": max((r["n_models"] for r in rows if r["ok"]), default=None),
    }


def main() -> int:
    """Run the HBM probe across every requested device and dump a JSON report."""
    args = _parse_args()

    if not Path(args.mxq_path).is_file():
        raise SystemExit(f"--mxq-path not found: {args.mxq_path}")

    dev_list = _parse_int_list(args.dev_no, "--dev-no")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    status_exe = _resolve_status_exe()
    if status_exe is None:
        print(
            f"note: {_STATUS_EXE} not on PATH; memory columns will be null.",
            file=sys.stderr,
        )

    device_reports: list[dict[str, Any]] = []
    for dev in dev_list:
        try:
            device_reports.append(_sweep_device(dev, args, status_exe))
        except QbRuntimeError as e:
            device_reports.append(
                {
                    "dev_no": int(dev),
                    "error": f"QbRuntimeError: {e}",
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception as e:  # noqa: BLE001
            device_reports.append(
                {
                    "dev_no": int(dev),
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
            )

    report = {
        "mxq_path": args.mxq_path,
        "core_mode": args.core_mode,
        "dev_no": dev_list,
        "n_models_max": int(args.n_models_max),
        "n_schedule": args.n_schedule,
        "settle_s": float(args.settle_s),
        "status_exe": status_exe,
        "devices": device_reports,
    }

    json_path = output_dir / "probe_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
