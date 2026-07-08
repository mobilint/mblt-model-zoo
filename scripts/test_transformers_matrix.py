"""Run `pytest tests/transformers` across supported transformers major.minor versions.

For each target version this script:
  1. Deletes `.venv/` (and `uv.lock` if present).
  2. Recreates a fresh venv with `uv venv --python <PY>`.
  3. Installs the project with `-e ".[transformers]" --group dev "transformers==<V>"`.
  4. Runs `pytest tests/transformers` in three phases:
     - Phase A (parallel): every test file except batch and eagle3 suites,
       split round-robin across N workers. Each worker pins its NPU backend
       to a distinct core via `--core-mode single --target-cores <cluster>:<core>`
       (and the encoder / decoder / vision / text variants) so workers do
       not contend on the same NPU core.
     - Phase B (serial): batch text-generation tests run alone with all 8 NPU
       cores. Their conftest already validates `--core-mode` == single and pops
       `target_cores` unless the user provided one.
     - Phase C (serial): eagle3 text-generation tests run with
       `--core-mode=global4` because the compiled MXQ only supports that
       layout; forcing global4 explicitly keeps `--full-matrix` (which would
       otherwise sweep single/global4/global8) from breaking the phase.

Version selection defaults to the latest patch of every major.minor release of
`transformers` on PyPI that falls within the range declared in `pyproject.toml`
(`>=4.54.0, <=5.12.1`).

Logs land under `logs/tx-matrix/<timestamp>/`:
  - install-<V>.log
  - pytest-<V>-w<i>.log + junit-<V>-w<i>.xml     (Phase A, one per worker)
  - pytest-<V>-batch.log + junit-<V>-batch.xml   (Phase B)
  - pytest-<V>-eagle3.log + junit-<V>-eagle3.xml (Phase C)
  - summary.txt (tab-separated: version, status, counts, elapsed [, note])

Counts merge every worker + batch junit into one compact string:
  702p/4s/44d/42w  → 702 passed, 4 skipped, 44 deselected, 42 warnings
  10F/50p          → 10 failed, 50 passed
Letters: p=passed, F=failed, E=error, s=skipped, d=deselected, xF=xfailed,
         xP=xpassed, w=warnings.

Caveats:
  - `--full-matrix` and high parallelism together will likely OOM the NPU. If
    `--full-matrix` is detected in the forwarded pytest args, workers are
    forced to 1 with a warning.
  - When `--full-matrix` is forwarded, this runner only exercises the full
    *model* matrix, not the core-mode sweep. Phase A hard-pins each pytest
    process to `--core-mode=single` (per-worker core placement is what makes
    parallelism safe) and Phase C hard-pins to `--core-mode=global4` (the
    only layout the compiled EAGLE-3 MXQ supports). To exercise the
    single/global4/global8 sweep, run pytest directly without this wrapper.
  - If a worker log contains `BadAlloc` / `out of memory`, the version's row
    in summary.txt is annotated with `BAD_ALLOC`; reduce `--workers`.

Usage (Linux/macOS/Windows, uv required):
    python scripts/test_transformers_matrix.py                           # workers=8
    python scripts/test_transformers_matrix.py --workers 4
    python scripts/test_transformers_matrix.py --no-parallel             # workers=1
    python scripts/test_transformers_matrix.py -v 5.11.0 5.12.1
    python scripts/test_transformers_matrix.py --start-from 5.5.4
    python scripts/test_transformers_matrix.py --dry-run
    python scripts/test_transformers_matrix.py -- -k Qwen                # extra pytest args
    python scripts/test_transformers_matrix.py --rebuild-summary logs/tx-matrix/<ts>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = REPO_ROOT / ".venv"
UV_LOCK = REPO_ROOT / "uv.lock"
DEFAULT_PYTHON = "3.12"
VERSION_MIN = (4, 54, 0)
VERSION_MAX = (5, 12, 1)

TESTS_ROOT = REPO_ROOT / "tests" / "transformers"
BATCH_DIR = TESTS_ROOT / "text_generation" / "batch"
EAGLE3_DIR = TESTS_ROOT / "text_generation" / "eagle3"
DEFAULT_CORE_MAP = ("0:0", "0:1", "0:2", "0:3", "1:0", "1:1", "1:2", "1:3")
MAX_WORKERS = len(DEFAULT_CORE_MAP)
# Sentinel for the venv reset path; kept out of the signal range (-1..-255) so
# `status_str` can distinguish a reset failure from a subprocess killed by a
# signal (`subprocess.Popen.returncode` returns `-signal_number` in that case).
_RESET_FAIL_SENTINEL = -1000
_BAD_ALLOC_RE = re.compile(r"BadAlloc|out of memory", re.IGNORECASE)
_COUNT_LETTER_ORDER = ("p", "F", "E", "s", "d", "xF", "xP", "w")
_COUNT_TOKEN_RE = re.compile(r"^(\d+)(p|F|E|s|d|xF|xP|w)$")


def parse_version(raw: str) -> tuple[int, int, int] | None:
    parts = raw.split(".")
    if len(parts) != 3:
        return None
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None


def fetch_latest_patches() -> list[str]:
    url = "https://pypi.org/pypi/transformers/json"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
    per_minor: dict[tuple[int, int], tuple[int, int, int]] = {}
    for raw in data["releases"].keys():
        parsed = parse_version(raw)
        if parsed is None:
            continue
        if not (VERSION_MIN <= parsed <= VERSION_MAX):
            continue
        key = (parsed[0], parsed[1])
        if key not in per_minor or parsed > per_minor[key]:
            per_minor[key] = parsed
    return [".".join(str(x) for x in v) for _, v in sorted(per_minor.items())]


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def running_from_target_venv() -> bool:
    try:
        return VENV_DIR.resolve() in Path(sys.executable).resolve().parents
    except OSError:
        return False


def run(cmd: list[str], log_file: Path | None = None) -> int:
    printable = " ".join(cmd)
    print(f"$ {printable}", flush=True)
    if log_file is None:
        return subprocess.run(cmd, cwd=REPO_ROOT).returncode
    with log_file.open("wb") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
            fh.write(line)
        return proc.wait()


def reset_venv(python_version: str) -> None:
    if VENV_DIR.exists():
        print(f"[reset] removing {VENV_DIR}", flush=True)
        shutil.rmtree(VENV_DIR)
    if UV_LOCK.exists():
        print(f"[reset] removing {UV_LOCK}", flush=True)
        UV_LOCK.unlink()
    rc = run(["uv", "venv", "--python", python_version, str(VENV_DIR)])
    if rc != 0:
        raise RuntimeError(f"`uv venv` failed with code {rc}")


def install_matrix(version: str, log_file: Path) -> int:
    return run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv_python()),
            "-e",
            ".[transformers]",
            "--group",
            "dev",
            f"transformers=={version}",
        ],
        log_file=log_file,
    )


def collect_phase_a_test_files() -> list[Path]:
    """Return relative paths of test_*.py under tests/transformers eligible for phase A.

    Excludes batch dir (phase B: all-8-core serial) and eagle3 dir (phase C: default
    global4 serial). Phase A forces `--core-mode single --target-cores <c>:<c>` per
    worker, which mismatches the compiled MXQ layout for those two suites.
    """
    files: list[Path] = []
    for p in sorted(TESTS_ROOT.rglob("test_*.py")):
        rel = p.relative_to(REPO_ROOT)
        try:
            p.relative_to(BATCH_DIR)
            continue
        except ValueError:
            pass
        try:
            p.relative_to(EAGLE3_DIR)
            continue
        except ValueError:
            pass
        files.append(rel)
    return files


def partition_round_robin(items: list[Path], n: int) -> list[list[Path]]:
    buckets: list[list[Path]] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        buckets[i % n].append(item)
    return buckets


def spawn_pytest_process(cmd: list[str], log_path: Path) -> tuple[subprocess.Popen, "object"]:
    print(f"$ {' '.join(cmd)}  > {log_path.name}", flush=True)
    fh = log_path.open("wb")
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=fh,
        stderr=subprocess.STDOUT,
    )
    return proc, fh


def run_phase_parallel(
    version: str,
    log_dir: Path,
    extra_args: list[str],
    workers: int,
    core_map: tuple[str, ...],
) -> dict[int, int]:
    """Launch `workers` pytest processes on non-batch tests, one per core in core_map."""
    files = collect_phase_a_test_files()
    if not files:
        return {}
    buckets = partition_round_robin(files, workers)

    procs: list[tuple[int, subprocess.Popen, object]] = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        core = core_map[i]
        log_path = log_dir / f"pytest-{version}-w{i}.log"
        junit_path = log_dir / f"junit-{version}-w{i}.xml"
        # Runner-owned flags come after `extra_args` so a forwarded `-- --core-mode ...`
        # or `-- --target-cores ...` cannot silently reroute a worker off its assigned core.
        cmd = [
            str(venv_python()),
            "-m",
            "pytest",
            *(str(f) for f in bucket),
            *extra_args,
            "--core-mode=single",
            f"--target-cores={core}",
            f"--vision-target-cores={core}",
            f"--text-target-cores={core}",
            f"--encoder-target-cores={core}",
            f"--decoder-target-cores={core}",
            f"--junit-xml={junit_path}",
        ]
        proc, fh = spawn_pytest_process(cmd, log_path)
        procs.append((i, proc, fh))

    rcs: dict[int, int] = {}
    for i, proc, fh in procs:
        rc = proc.wait()
        try:
            fh.close()
        except Exception:
            pass
        rcs[i] = rc
        print(f"  [phase A worker {i}] rc={rc}", flush=True)
    return rcs


def run_phase_batch(version: str, log_dir: Path, extra_args: list[str]) -> int:
    """Run batch text-generation tests serially; batch conftest uses all 8 cores by default."""
    log_path = log_dir / f"pytest-{version}-batch.log"
    junit_path = log_dir / f"junit-{version}-batch.xml"
    # Runner-owned flags come after `extra_args` so a forwarded core-mode override
    # cannot bypass the single-only invariant the batch conftest expects.
    cmd = [
        str(venv_python()),
        "-m",
        "pytest",
        str(BATCH_DIR.relative_to(REPO_ROOT)),
        *extra_args,
        "--core-mode=single",
        f"--junit-xml={junit_path}",
    ]
    return run(cmd, log_file=log_path)


def run_phase_eagle3(version: str, log_dir: Path, extra_args: list[str]) -> int:
    """Run EAGLE-3 tests serially and pin them to their compiled `global4` layout.

    The default in quick mode is already `global4`, but `--full-matrix` (or a
    user-provided `--core-mode=all`) would otherwise expand
    `build_eagle3_specs` into `single`/`global4`/`global8`, and the compiled
    MXQ is only compatible with `global4`. Forcing `--core-mode=global4` here
    keeps that invariant regardless of what the caller forwards.
    """
    log_path = log_dir / f"pytest-{version}-eagle3.log"
    junit_path = log_dir / f"junit-{version}-eagle3.xml"
    # Runner-owned flags come after `extra_args` so pytest's argparse (last value wins)
    # collapses any forwarded --core-mode back onto the layout the MXQ was built for.
    cmd = [
        str(venv_python()),
        "-m",
        "pytest",
        str(EAGLE3_DIR.relative_to(REPO_ROOT)),
        *extra_args,
        "--core-mode=global4",
        f"--junit-xml={junit_path}",
    ]
    return run(cmd, log_file=log_path)


def status_str(rc: int) -> str:
    if rc == 0:
        return "PASS"
    if rc == _RESET_FAIL_SENTINEL:
        return "RESET_FAIL"
    if rc < 0:
        # subprocess returncode is -signal_number when the child is killed by
        # a signal. SIGKILL from OOM (this runner's headline diagnostic) shows
        # up here, so surface it explicitly instead of hiding behind FAIL().
        return f"SIGNAL({-rc})"
    if rc == 5:
        return "NO_TESTS"
    return f"FAIL({rc})"


_PYTEST_SUMMARY_LINE = re.compile(
    r"^=+\s*(?P<body>.*?(?:passed|failed|error|errors|no tests ran|xfailed|xpassed|deselected|skipped).*?)\s*=+\s*$"
)
_COUNT_TOKEN = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|deselected|xfailed|xpassed|warnings|warning)")


def extract_pytest_summary(log_path: Path) -> str:
    """Return a compact `passed/failed/skipped/...` summary from a pytest log.

    Falls back to a marker if the log has no recognizable summary line
    (e.g. pytest crashed during collection).
    """
    if not log_path.exists():
        return "NO_LOG"
    last: str | None = None
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _PYTEST_SUMMARY_LINE.match(line.strip())
            if m:
                last = m.group("body")
    if last is None:
        return "NO_SUMMARY"
    counts = _COUNT_TOKEN.findall(last)
    if not counts:
        return last.strip()
    letter_map = {
        "passed": "p",
        "failed": "F",
        "error": "E",
        "errors": "E",
        "skipped": "s",
        "deselected": "d",
        "xfailed": "xF",
        "xpassed": "xP",
        "warnings": "w",
        "warning": "w",
    }
    parts = [f"{n}{letter_map.get(name, name)}" for n, name in counts]
    return "/".join(parts)


def merge_count_strings(counts_list: list[str]) -> str:
    """Sum multiple compact count strings (e.g. `702p/4s/44d`) into a single one."""
    totals: dict[str, int] = {}
    for cs in counts_list:
        if cs in ("NO_LOG", "NO_SUMMARY", "-", ""):
            continue
        for token in cs.split("/"):
            m = _COUNT_TOKEN_RE.match(token.strip())
            if not m:
                continue
            n, letter = m.groups()
            totals[letter] = totals.get(letter, 0) + int(n)
    if not totals:
        return "-"
    parts: list[str] = []
    for letter in _COUNT_LETTER_ORDER:
        if letter in totals:
            parts.append(f"{totals[letter]}{letter}")
    for letter, n in totals.items():
        if letter not in _COUNT_LETTER_ORDER:
            parts.append(f"{n}{letter}")
    return "/".join(parts)


def _log_group_for_version(name: str) -> str | None:
    """Map `pytest-<V>*.log` variants (worker, batch, eagle3) to `<V>`."""
    if not name.startswith("pytest-") or not name.endswith(".log"):
        return None
    stem = name[len("pytest-") : -len(".log")]
    m = re.match(r"^(?P<v>.+?)(?:-(?:w\d+|batch|eagle3))?$", stem)
    if not m:
        return None
    return m.group("v")


def scan_for_bad_alloc(log_dir: Path, version: str) -> bool:
    for p in log_dir.glob(f"pytest-{version}-*.log"):
        try:
            with p.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if _BAD_ALLOC_RE.search(line):
                        return True
        except OSError:
            continue
    return False


def rebuild_summary(log_dir: Path) -> int:
    if not log_dir.is_dir():
        print(f"ERROR: log dir not found: {log_dir}", file=sys.stderr)
        return 2
    groups: dict[str, list[Path]] = {}
    for log in sorted(log_dir.glob("pytest-*.log")):
        version = _log_group_for_version(log.name)
        if version is None:
            continue
        groups.setdefault(version, []).append(log)
    if not groups:
        print(f"ERROR: no pytest-*.log files under {log_dir}", file=sys.stderr)
        return 2
    entries: list[tuple[str, str]] = []
    for version, logs in sorted(groups.items()):
        counts = merge_count_strings([extract_pytest_summary(p) for p in logs])
        entries.append((version, counts))
    if not entries:
        print(f"ERROR: no pytest-*.log files under {log_dir}", file=sys.stderr)
        return 2
    print(f"log_dir: {log_dir}")
    for version, counts in entries:
        print(f"  {version:<10} {counts}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "-v",
        "--versions",
        nargs="+",
        help="Explicit transformers versions to test (bypasses PyPI discovery).",
    )
    ap.add_argument(
        "--python",
        default=DEFAULT_PYTHON,
        help=f"Python version passed to `uv venv --python` (default: {DEFAULT_PYTHON}).",
    )
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Override the log directory (default: logs/tx-matrix/<timestamp>).",
    )
    ap.add_argument(
        "--start-from",
        help="Skip versions until this one is reached (for resuming after a failure).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without touching the venv or running tests.",
    )
    ap.add_argument(
        "--rebuild-summary",
        type=Path,
        default=None,
        metavar="LOG_DIR",
        help="Rescan pytest-*.log in LOG_DIR and print a summary table without running anything.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of parallel pytest workers for phase A (1..{MAX_WORKERS}, default: {MAX_WORKERS}).",
    )
    ap.add_argument(
        "--no-parallel",
        action="store_true",
        help="Shortcut for --workers 1.",
    )
    ap.add_argument(
        "--core-map",
        default=",".join(DEFAULT_CORE_MAP),
        help=(
            "Comma-separated cluster:core assignments for phase A workers "
            f"(default: {','.join(DEFAULT_CORE_MAP)})."
        ),
    )
    ap.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to pytest (place after `--`).",
    )
    args = ap.parse_args()

    if args.rebuild_summary is not None:
        return rebuild_summary(args.rebuild_summary)

    if not args.dry_run and running_from_target_venv():
        print(
            "ERROR: this script would delete the .venv it is currently running under. "
            "Invoke it with an interpreter outside of .venv (e.g. system Python or `uv run --no-project`).",
            file=sys.stderr,
        )
        return 2

    if shutil.which("uv") is None:
        print("ERROR: `uv` was not found on PATH.", file=sys.stderr)
        return 2

    versions = args.versions or fetch_latest_patches()
    if not versions:
        print("ERROR: no versions to run.", file=sys.stderr)
        return 2

    if args.start_from:
        if args.start_from not in versions:
            print(
                f"ERROR: --start-from {args.start_from} not in version list: {versions}",
                file=sys.stderr,
            )
            return 2
        versions = versions[versions.index(args.start_from) :]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = args.log_dir or (REPO_ROOT / "logs" / "tx-matrix" / ts)

    extra = list(args.pytest_args)
    if extra and extra[0] == "--":
        extra = extra[1:]

    core_map = tuple(item.strip() for item in args.core_map.split(",") if item.strip())
    if not core_map:
        print("ERROR: --core-map must list at least one cluster:core.", file=sys.stderr)
        return 2

    workers = 1 if args.no_parallel else args.workers
    if workers < 1:
        print("ERROR: --workers must be >= 1.", file=sys.stderr)
        return 2
    if workers > len(core_map):
        print(
            f"ERROR: --workers ({workers}) exceeds core map size ({len(core_map)}).",
            file=sys.stderr,
        )
        return 2
    if any("--full-matrix" in a for a in extra):
        print(
            "WARNING: --full-matrix only expands the model matrix in this runner. "
            "Phase A pins --core-mode=single and Phase C pins --core-mode=global4, "
            "so the single/global4/global8 core sweep is NOT exercised here; "
            "run pytest directly without this wrapper to cover that sweep.",
            file=sys.stderr,
        )
        if workers > 1:
            print(
                f"WARNING: --full-matrix detected in pytest args; forcing workers=1 (was {workers}) to avoid NPU BadAlloc.",
                file=sys.stderr,
            )
            workers = 1

    print(f"[plan] python:       {args.python}")
    print(f"[plan] versions:     {versions}")
    print(f"[plan] log_dir:      {log_dir}")
    print(f"[plan] workers:      {workers}")
    print(f"[plan] core_map:     {list(core_map[:workers])}")
    print(f"[plan] pytest extra: {extra}")
    if args.dry_run:
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "summary.txt"
    results: list[tuple[str, int, str, float, str]] = []

    def append_summary(version: str, status: str, counts: str, elapsed: float, note: str = "") -> None:
        with summary_path.open("a", encoding="utf-8") as fh:
            row = f"{version}\t{status}\t{counts}\t{elapsed:.1f}s"
            if note:
                row += f"\t{note}"
            fh.write(row + "\n")

    for v in versions:
        print(f"\n===== transformers=={v} =====", flush=True)
        start = time.monotonic()
        try:
            reset_venv(args.python)
        except Exception as exc:
            elapsed = time.monotonic() - start
            print(f"[reset failed] {exc}", file=sys.stderr)
            results.append((v, _RESET_FAIL_SENTINEL, "-", elapsed, ""))
            append_summary(v, "RESET_FAIL", "-", elapsed)
            continue

        rc = install_matrix(v, log_dir / f"install-{v}.log")
        if rc != 0:
            elapsed = time.monotonic() - start
            print(f"[install failed] rc={rc}", file=sys.stderr)
            results.append((v, rc, "-", elapsed, ""))
            append_summary(v, f"INSTALL_FAIL({rc})", "-", elapsed)
            continue

        print(f"[phase A] {workers} worker(s) on non-batch, non-eagle3 tests", flush=True)
        worker_rcs = run_phase_parallel(v, log_dir, extra, workers, core_map)
        print("[phase B] batch tests (serial, all 8 cores)", flush=True)
        batch_rc = run_phase_batch(v, log_dir, extra)
        print("[phase C] eagle3 tests (serial, default global4)", flush=True)
        eagle3_rc = run_phase_eagle3(v, log_dir, extra)

        elapsed = time.monotonic() - start
        all_rcs = list(worker_rcs.values()) + [batch_rc, eagle3_rc]
        # rc 5 = pytest "no tests collected"; benign when `-k`/`-m` filters
        # deselect every test in a shard, so remap to 0 for aggregation.
        # But if *every* phase came back with 5, no test actually ran for this
        # version and PASS would be misleading -- surface it as NO_TESTS instead.
        # Signal terminations (negative rc, e.g. SIGKILL from OOM) take
        # precedence over positive failures because that is the failure mode
        # this runner is trying to diagnose; a plain `max()` would let a
        # concurrent positive rc from another phase mask the signal.
        non_empty_rcs = [r for r in all_rcs if r != 5]
        if not non_empty_rcs:
            overall_rc = 5
        else:
            signal_rcs = [r for r in non_empty_rcs if r < 0]
            positive_rcs = [r for r in non_empty_rcs if r > 0]
            if signal_rcs:
                overall_rc = min(signal_rcs)
            elif positive_rcs:
                overall_rc = max(positive_rcs)
            else:
                overall_rc = 0

        version_logs = sorted(log_dir.glob(f"pytest-{v}-*.log"))
        counts = merge_count_strings([extract_pytest_summary(p) for p in version_logs])
        notes: list[str] = []
        if scan_for_bad_alloc(log_dir, v):
            notes.append("BAD_ALLOC")
        # Partial-empty case only: some shards were empty but at least one ran.
        # When *all* were empty, overall_rc == 5 already labels the version.
        if overall_rc != 5 and any(r == 5 for r in all_rcs):
            notes.append("NO_TESTS")
        note = ",".join(notes)
        results.append((v, overall_rc, counts, elapsed, note))
        append_summary(v, status_str(overall_rc), counts, elapsed, note)
        suffix = f" [{note}]" if note else ""
        print(
            f"[done] transformers=={v}: {status_str(overall_rc)} {counts} in {elapsed:.1f}s{suffix}",
            flush=True,
        )

    print("\n===== Summary =====")
    print(f"log_dir: {log_dir}")
    for v, rc, counts, elapsed, note in results:
        note_col = f" {note}" if note else ""
        print(f"  {v:<10} {status_str(rc):<16} {counts:<40} {elapsed:>7.1f}s{note_col}")
    return 0 if all(rc == 0 for _, rc, _, _, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
