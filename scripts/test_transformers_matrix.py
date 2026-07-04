"""Run `pytest tests/transformers` across supported transformers major.minor versions.

For each target version this script:
  1. Deletes `.venv/` (and `uv.lock` if present).
  2. Recreates a fresh venv with `uv venv --python <PY>`.
  3. Installs the project with `-e ".[transformers]" --group dev "transformers==<V>"`.
  4. Runs `pytest tests/transformers` with a per-version JUnit XML.

Version selection defaults to the latest patch of every major.minor release of
`transformers` on PyPI that falls within the range declared in `pyproject.toml`
(`>=4.54.0, <=5.12.1`).

Logs land under `logs/tx-matrix/<timestamp>/`:
  - install-<V>.log, pytest-<V>.log, junit-<V>.xml
  - summary.txt (tab-separated: version, status, elapsed)

Usage (Linux/macOS/Windows, uv required):
    python scripts/test_transformers_matrix.py
    python scripts/test_transformers_matrix.py -v 5.11.0 5.12.1
    python scripts/test_transformers_matrix.py --start-from 5.5.4
    python scripts/test_transformers_matrix.py --dry-run
    python scripts/test_transformers_matrix.py -- -k Qwen        # extra pytest args
"""

from __future__ import annotations

import argparse
import json
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


def run_pytest(version: str, log_dir: Path, extra_args: list[str]) -> int:
    junit = log_dir / f"junit-{version}.xml"
    log = log_dir / f"pytest-{version}.log"
    cmd = [
        str(venv_python()),
        "-m",
        "pytest",
        "tests/transformers",
        f"--junit-xml={junit}",
        *extra_args,
    ]
    return run(cmd, log_file=log)


def status_str(rc: int) -> str:
    if rc == 0:
        return "PASS"
    if rc < 0:
        return "RESET_FAIL"
    return f"FAIL({rc})"


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
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to pytest (place after `--`).",
    )
    args = ap.parse_args()

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
    log_dir.mkdir(parents=True, exist_ok=True)

    extra = list(args.pytest_args)
    if extra and extra[0] == "--":
        extra = extra[1:]

    print(f"[plan] python:       {args.python}")
    print(f"[plan] versions:     {versions}")
    print(f"[plan] log_dir:      {log_dir}")
    print(f"[plan] pytest extra: {extra}")
    if args.dry_run:
        return 0

    summary_path = log_dir / "summary.txt"
    results: list[tuple[str, int, float]] = []

    for v in versions:
        print(f"\n===== transformers=={v} =====", flush=True)
        start = time.monotonic()
        try:
            reset_venv(args.python)
        except Exception as exc:
            elapsed = time.monotonic() - start
            print(f"[reset failed] {exc}", file=sys.stderr)
            results.append((v, -1, elapsed))
            with summary_path.open("a", encoding="utf-8") as fh:
                fh.write(f"{v}\tRESET_FAIL\t{elapsed:.1f}s\n")
            continue

        rc = install_matrix(v, log_dir / f"install-{v}.log")
        if rc != 0:
            elapsed = time.monotonic() - start
            print(f"[install failed] rc={rc}", file=sys.stderr)
            results.append((v, rc, elapsed))
            with summary_path.open("a", encoding="utf-8") as fh:
                fh.write(f"{v}\tINSTALL_FAIL({rc})\t{elapsed:.1f}s\n")
            continue

        rc = run_pytest(v, log_dir, extra)
        elapsed = time.monotonic() - start
        results.append((v, rc, elapsed))
        with summary_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{v}\t{status_str(rc)}\t{elapsed:.1f}s\n")
        print(f"[done] transformers=={v}: {status_str(rc)} in {elapsed:.1f}s", flush=True)

    print("\n===== Summary =====")
    print(f"log_dir: {log_dir}")
    for v, rc, elapsed in results:
        print(f"  {v:<10} {status_str(rc):<16} {elapsed:>7.1f}s")
    return 0 if all(rc == 0 for _, rc, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
