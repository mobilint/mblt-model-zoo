"""Probe in-process parallelism of multiple ``qbruntime.Model`` instances.

Purpose
-------
``qbruntime.Model.infer(...)`` is a blocking call, so the current
``MobilintNPUBackend`` design (one ``Model`` per backend) cannot dispatch more
than one request at a time. A candidate redesign is ``N`` ``Model`` instances
sharing one ``Accelerator``, with ``K`` per-``Model`` batch giving a total
concurrent capacity of ``N x K``. Before committing to that refactor we need
to know whether ``N > 1`` actually yields concurrent NPU utilization or serializes.

This probe launches ``N`` ``qbruntime.Model`` instances against a single
``qbruntime.Accelerator``, submits identical synthetic ``.infer`` calls
concurrently through ``ThreadPoolExecutor``, and reports:

* the median wall time (max end - min start of the parallel batch),
* median per-``Model`` ``.infer`` latency,
* ``speedup_vs_N1`` = ``N * per_model_time(N=1) / wall_time(N=N)``.

``speedup_vs_n1`` is measured only when ``--n-models`` includes ``1``; otherwise
the column is null in JSON, empty in CSV, and ``n/a`` in the console print so
the missing baseline is not confused with a valid ``1.0`` measurement.

An ``--output-parity`` mode also compares each ``Model``'s output argmax to a
reference ``Model``, i.e. in-process determinism across replicated handles.

Result usage
------------
A per-``Model`` ``.infer`` time that stays roughly flat as ``N`` grows means
inference actually overlaps across handles and the ``N`` handles refactor is
worth doing. A per-``Model`` time that grows linearly with ``N`` means the
runtime serializes on the shared cores and the refactor buys nothing over
batching in a single handle.

Non-goals
---------
No refactor is done here. Only measurement. Summary the caller writes into
either the PR body or the invocation docstring.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np
from qbruntime import Accelerator, Cluster, Core, CoreId, Model, ModelConfig, QbRuntimeError

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


_CLUSTER_MAP = {0: Cluster.Cluster0, 1: Cluster.Cluster1}
_CORE_MAP = {0: Core.Core0, 1: Core.Core1, 2: Core.Core2, 3: Core.Core3}
_DTYPE_MAP = {
    "DataType.Float32": np.float32,
    "DataType.Float16": np.float16,
    "DataType.Int8": np.int8,
    "DataType.Uint8": np.uint8,
}


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the parallel-``Model`` probe."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mxq-path", required=True, help="Path to the MXQ artifact to load N times.")
    parser.add_argument(
        "--n-models",
        default="1,2,4",
        help="Comma-separated list of N values to sweep, e.g. '1,2,4,8'.",
    )
    parser.add_argument(
        "--dev-no",
        type=int,
        default=0,
        help="Accelerator device number. Only one device is used by this probe.",
    )
    parser.add_argument(
        "--core-mode",
        choices=("single", "global4", "global8"),
        default="single",
        help=(
            "Core mode applied to each Model. 'single' claims a single core, "
            "'global4' claims a 4-core cluster, 'global8' claims all 8 cores. "
            "Global8 forces N=1 (any larger --n-models values are silently dropped)."
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Number of concurrent-batch repetitions per N value; median is reported.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of untimed concurrent batches to run before measurement per N.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1,
        help=(
            "If the model's declared input has a leading time dimension, override it with this "
            "value. Otherwise the declared shape is used verbatim."
        ),
    )
    parser.add_argument(
        "--partition-cores",
        action="store_true",
        help=(
            "For --core-mode single, allocate one distinct core per Model up to 8 cores. "
            "For --core-mode global4, allocate one distinct cluster per Model up to 2. "
            "If disabled, every Model shares the same (all-core) target."
        ),
    )
    parser.add_argument(
        "--output-parity",
        action="store_true",
        help=(
            "Compare per-Model output argmax against the first Model on identical inputs. "
            "Reports the fraction of Models that match on the reference batch."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the synthetic input tensor.",
    )
    parser.add_argument(
        "--output-dir",
        default="debug/multi_model_parallel_probe",
        help="Directory that receives probe_report.json and probe_report.csv.",
    )
    return parser.parse_args()


def _parse_n_list(raw: str) -> list[int]:
    """Parse a comma-separated ``--n-models`` list into sorted positive ints."""
    values: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        n = int(tok)
        if n < 1:
            raise SystemExit(f"--n-models entries must be >= 1; got {n}.")
        values.append(n)
    if not values:
        raise SystemExit("--n-models resolved to an empty list.")
    return sorted(set(values))


def _dtype_from_model(mxq_model: "Model") -> np.dtype:
    """Return a numpy dtype corresponding to the model's declared input data type."""
    kind = str(mxq_model.get_model_input_data_type())
    dtype = _DTYPE_MAP.get(kind)
    if dtype is None:
        raise SystemExit(f"Unsupported model input DataType: {kind}")
    return np.dtype(dtype)


def _build_input(mxq_model: "Model", seq_len: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Return a list of synthetic input arrays matching the model's declared shapes.

    Args:
        mxq_model: A ``qbruntime.Model`` already ``launch``'ed on the accelerator.
        seq_len: Value used to replace any dynamic dimension (``-1``) in the declared shape.
        rng: Numpy RNG used for input generation (kept per-Model to make outputs identical
            across replicated Models given identical seeds).

    Returns:
        A list of numpy arrays with the correct dtype for ``.infer``.
    """
    shapes = mxq_model.get_model_input_shape()
    dtype = _dtype_from_model(mxq_model)
    inputs: list[np.ndarray] = []
    for raw_shape in shapes:
        shape = [int(d) if int(d) > 0 else int(seq_len) for d in raw_shape]
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            arr = rng.integers(info.min, info.max + 1, size=shape, dtype=dtype)
        else:
            arr = rng.standard_normal(size=shape).astype(dtype, copy=False)
        inputs.append(np.ascontiguousarray(arr))
    return inputs


def _make_model_config(
    core_mode: str,
    single_cores: Optional[list["CoreId"]],
    clusters: Optional[list["Cluster"]],
) -> "ModelConfig":
    """Build a per-``Model`` ``ModelConfig`` for the requested core allocation."""
    mc = ModelConfig()
    if core_mode == "single":
        mc.set_single_core_mode(None, single_cores if single_cores is not None else [])
    elif core_mode == "global4":
        mc.set_global4_core_mode(clusters if clusters is not None else [])
    elif core_mode == "global8":
        mc.set_global8_core_mode()
    else:
        raise SystemExit(f"Unsupported core_mode: {core_mode}")
    return mc


def _allocate_cores(
    core_mode: str, n: int, partition: bool
) -> list[tuple[Optional[list["CoreId"]], Optional[list["Cluster"]]]]:
    """Return per-``Model`` (single_cores, clusters) tuples.

    Args:
        core_mode: One of ``"single"``, ``"global4"``, ``"global8"``.
        n: Number of ``Model`` instances to configure.
        partition: If true, hand each ``Model`` a distinct core/cluster; else share.

    Raises:
        SystemExit: If the requested (core_mode, n, partition) combination is infeasible.
    """
    if core_mode == "global8":
        if n != 1:
            raise SystemExit("--core-mode global8 requires n_models=1 (occupies all 8 cores).")
        return [(None, None)]

    if core_mode == "global4":
        if not partition:
            return [(None, [Cluster.Cluster0, Cluster.Cluster1])] * n
        if n > 2:
            raise SystemExit("--core-mode global4 with --partition-cores supports at most n_models=2.")
        clusters_seq = [Cluster.Cluster0, Cluster.Cluster1]
        return [(None, [clusters_seq[i]]) for i in range(n)]

    # single
    if not partition:
        return [(None, None)] * n
    if n > 8:
        raise SystemExit("--core-mode single with --partition-cores supports at most n_models=8.")
    core_seq = [CoreId(_CLUSTER_MAP[cl], _CORE_MAP[co]) for cl in (0, 1) for co in (0, 1, 2, 3)]
    return [(list([core_seq[i]]), None) for i in range(n)]


def _dispose_all(models: list["Model"]) -> None:
    """Best-effort ``dispose()`` of every model; swallow individual failures."""
    for mm in models:
        try:
            mm.dispose()
        except Exception as e:  # noqa: BLE001 — release path must not raise
            print(f"warning: dispose failed: {e}", file=sys.stderr)


def _run_parallel_once(
    models: list["Model"],
    inputs_per_model: list[list[np.ndarray]],
) -> tuple[list[float], float, list[list[np.ndarray]]]:
    """Submit one ``.infer`` per ``Model`` concurrently through a pool.

    Returns:
        ``(per_model_times, wall_time, outputs)`` where ``wall_time`` is
        ``max(end_i) - min(start_i)``.
    """
    n = len(models)
    starts: list[float] = [0.0] * n
    ends: list[float] = [0.0] * n
    outputs: list[Optional[list[np.ndarray]]] = [None] * n

    def _one(idx: int) -> None:
        mm = models[idx]
        starts[idx] = time.perf_counter()
        out = mm.infer(inputs_per_model[idx])
        ends[idx] = time.perf_counter()
        outputs[idx] = [np.asarray(o) for o in out] if out is not None else []

    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(_one, range(n)))

    per_model = [ends[i] - starts[i] for i in range(n)]
    wall = max(ends) - min(starts)
    return per_model, wall, [outputs[i] or [] for i in range(n)]


def _argmax_of(outputs: list[np.ndarray]) -> list[int]:
    """Global argmax of every output array — used for parity comparison."""
    result: list[int] = []
    for arr in outputs:
        if arr.size == 0:
            result.append(-1)
        else:
            result.append(int(np.asarray(arr).ravel().argmax()))
    return result


def _summarize(values: list[float]) -> dict[str, float]:
    """Return ``median``, ``min``, ``max`` for a list of floats."""
    if not values:
        return {"median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _run_for_n(
    n: int,
    args: argparse.Namespace,
    baseline_per_model_median: Optional[float],
) -> dict[str, Any]:
    """Launch ``N`` Models, warm them, run ``--repeat`` timed batches, and summarize."""
    per_slot = _allocate_cores(args.core_mode, n, args.partition_cores)

    acc = Accelerator(int(args.dev_no))
    models: list["Model"] = []
    try:
        for slot_idx, (single_cores, clusters) in enumerate(per_slot):
            mc = _make_model_config(args.core_mode, single_cores, clusters)
            mm = Model(args.mxq_path, mc)
            try:
                mm.launch(acc)
            except BaseException:
                # ``mm`` was constructed (allocating LPDDR / runtime state) but
                # not yet appended to ``models``, so the outer ``finally``
                # branch's ``_dispose_all(models)`` cannot release it. ``main``
                # keeps probing larger ``N`` on the same device even after a
                # per-``N`` failure, so a leaked handle here causes cascading
                # BadAlloc that corrupts the reported saturation boundary.
                # ``BaseException`` also releases on ``KeyboardInterrupt``.
                try:
                    mm.dispose()
                except Exception as _dispose_exc:  # noqa: BLE001 — release path must not raise
                    print(
                        f"warning: dispose failed on launch-failure path: {_dispose_exc}",
                        file=sys.stderr,
                    )
                raise
            models.append(mm)

        inputs_per_model: list[list[np.ndarray]] = []
        for slot_idx in range(n):
            rng = np.random.default_rng(int(args.seed) + slot_idx)
            inputs_per_model.append(_build_input(models[slot_idx], int(args.seq_len), rng))

        for _ in range(max(0, int(args.warmup))):
            _run_parallel_once(models, inputs_per_model)

        wall_times: list[float] = []
        per_model_times: list[float] = []
        parity_rates: list[float] = []
        for _ in range(int(args.repeat)):
            per_model, wall, outputs = _run_parallel_once(models, inputs_per_model)
            wall_times.append(wall)
            per_model_times.extend(per_model)
            if args.output_parity and n > 1:
                # Recompute deterministically: identical seeds per slot -> compare argmax.
                ref_seed = int(args.seed)
                same_seed_inputs = _build_input(models[0], int(args.seq_len), np.random.default_rng(ref_seed))

                # Run every model on the SAME input to check handle-level parity.
                def _one(idx: int) -> list[np.ndarray]:
                    out = models[idx].infer(same_seed_inputs)
                    return [np.asarray(o) for o in out] if out is not None else []

                with ThreadPoolExecutor(max_workers=n) as pool:
                    parity_outputs = list(pool.map(_one, range(n)))
                ref_argmax = _argmax_of(parity_outputs[0])
                matches = sum(1 for i in range(1, n) if _argmax_of(parity_outputs[i]) == ref_argmax)
                parity_rates.append(matches / max(1, n - 1))

        wall_stats = _summarize(wall_times)
        per_model_stats = _summarize(per_model_times)
        # Speedup relative to a single Model of the same core_mode. When no
        # ``N=1`` baseline is available (either the sweep skipped ``N=1`` or
        # the baseline run failed) we emit ``None`` rather than a fabricated
        # ``1.0``: downstream renderers treat ``None`` as "unmeasured" so a
        # missing baseline cannot be mistaken for flat parallel scaling.
        speedup: Optional[float]
        if baseline_per_model_median is not None and wall_stats["median"] > 0:
            speedup = float(n * baseline_per_model_median / wall_stats["median"])
        else:
            speedup = None
        return {
            "n_models": n,
            "wall_time_s": wall_stats,
            "per_model_time_s": per_model_stats,
            "n_batches": int(args.repeat),
            "throughput_infers_per_s": (float(n) / wall_stats["median"]) if wall_stats["median"] > 0 else 0.0,
            "speedup_vs_n1": speedup,
            "parity_median": (float(statistics.median(parity_rates)) if parity_rates else None),
        }
    finally:
        _dispose_all(models)


def _write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    """Write the per-``N`` measurement table to a CSV file."""
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "n_models",
                "wall_median_s",
                "wall_min_s",
                "wall_max_s",
                "per_model_median_s",
                "per_model_min_s",
                "per_model_max_s",
                "throughput_infers_per_s",
                "speedup_vs_n1",
                "parity_median",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    row["n_models"],
                    f"{row['wall_time_s']['median']:.6f}",
                    f"{row['wall_time_s']['min']:.6f}",
                    f"{row['wall_time_s']['max']:.6f}",
                    f"{row['per_model_time_s']['median']:.6f}",
                    f"{row['per_model_time_s']['min']:.6f}",
                    f"{row['per_model_time_s']['max']:.6f}",
                    f"{row['throughput_infers_per_s']:.6f}",
                    "" if row["speedup_vs_n1"] is None else f"{row['speedup_vs_n1']:.6f}",
                    "" if row["parity_median"] is None else f"{row['parity_median']:.4f}",
                ]
            )


def main() -> int:
    """Run the parallel-``Model`` probe across every requested ``N`` and dump JSON+CSV."""
    args = _parse_args()

    n_values = _parse_n_list(args.n_models)

    if args.core_mode == "global8":
        # global8 claims every core, so more than one Model on the same accelerator is
        # impossible; filter the sweep down to N=1 (falling back to [1] when the caller
        # explicitly asked only for N>1) and warn so partial reports still get written.
        # This matches probe_multi_model_hbm._sweep_device. The per-iteration guard in
        # ``_allocate_cores`` stays as defense in depth against callers that bypass main.
        original_n_values = list(n_values)
        n_values = [n for n in n_values if n == 1] or [1]
        if n_values != original_n_values:
            print(
                f"--core-mode global8 forces N=1 (all 8 cores claimed by a single Model); "
                f"filtered --n-models {original_n_values} -> {n_values}.",
                file=sys.stderr,
            )

    if 1 not in n_values:
        # The baseline is measured only when N=1 runs, so users who deliberately
        # skip it (e.g. tight LPDDR budgets) get null speedups rather than a
        # fabricated 1.0. Warn once so the null column isn't mistaken for a
        # tooling bug.
        print(
            "warning: --n-models does not include 1; speedup_vs_n1 will be reported "
            "as null (no baseline measurement). Add 1 to --n-models to enable "
            "speedup reporting.",
            file=sys.stderr,
        )

    if not Path(args.mxq_path).is_file():
        raise SystemExit(f"--mxq-path not found: {args.mxq_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    errors: dict[int, str] = {}
    baseline_median: Optional[float] = None
    for n in n_values:
        try:
            row = _run_for_n(n, args, baseline_median)
        except QbRuntimeError as e:
            errors[n] = f"QbRuntimeError: {e}"
            print(f"n_models={n}: QbRuntimeError: {e}", file=sys.stderr)
            continue
        except Exception as e:  # noqa: BLE001
            errors[n] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            print(f"n_models={n}: {type(e).__name__}: {e}", file=sys.stderr)
            continue

        rows.append(row)
        if n == 1 and baseline_median is None:
            baseline_median = row["per_model_time_s"]["median"]
            row["speedup_vs_n1"] = 1.0
        speedup_display = "n/a" if row["speedup_vs_n1"] is None else f"{row['speedup_vs_n1']:.2f}"
        print(
            f"n_models={n}  wall_median={row['wall_time_s']['median']:.4f}s  "
            f"per_model_median={row['per_model_time_s']['median']:.4f}s  "
            f"throughput={row['throughput_infers_per_s']:.2f} infers/s  "
            f"speedup_vs_n1={speedup_display}"
        )

    report = {
        "mxq_path": args.mxq_path,
        "dev_no": int(args.dev_no),
        "core_mode": args.core_mode,
        "partition_cores": bool(args.partition_cores),
        "seq_len": int(args.seq_len),
        "seed": int(args.seed),
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "n_values": n_values,
        "baseline_per_model_median_s": baseline_median,
        "rows": rows,
        "errors": errors,
    }

    json_path = output_dir / "probe_report.json"
    csv_path = output_dir / "probe_report.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(rows, csv_path)
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
