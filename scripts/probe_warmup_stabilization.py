"""Probe whether a single warmup ``generate`` stabilizes cross-process EAGLE-3 output.

Prior probes established that MXQ backends are bit-exact deterministic per call
(:mod:`scripts.probe_mxq_determinism`) and that within a single Python process
``model.generate()`` is fully deterministic across repeated iterations
(:mod:`scripts.probe_generate_same_process`). Yet three consecutive runs of
``debug_eagle3_tree_trace.py --seed 0`` in fresh subprocesses produced two
identical outputs and one different output. That leaves *cross-process* SDK,
DMA, or scratch-buffer state as the remaining suspect.

This probe launches multiple fresh Python subprocesses of itself in an inner
runner mode. Each inner run does a warmup ``generate`` followed by a measured
``generate`` (identical args, both re-seed ``torch.manual_seed(seed)``) and
prints a single JSON line describing the measured output. A baseline mode
(``--skip-warmup``) skips the warmup so the measured generate is the first one
the process ever runs — this is the state the earlier flip experiment saw.

Verdicts:

- Baseline all-identical AND warmup all-identical: cross-process is already
  deterministic on this machine. The earlier 3-run flip may have been rare;
  rerun with ``--n-baseline-runs 10`` to increase the trigger probability.
- Baseline NOT all-identical AND warmup all-identical: warmup FIXES the
  cross-process variance. Adopt a warmup generate as standard for benchmarks.
- Baseline NOT all-identical AND warmup NOT all-identical: warmup does not
  help. State survives across the warmup. Escalate to the MXQ runtime team.
- Baseline all-identical AND warmup NOT all-identical: warmup HURTS. Unlikely,
  but log it and inspect the warmup path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add args shared by the inner runner and the outer orchestrator."""
    parser.add_argument("--model", default="mobilint/EAGLE3-Qwen3-4B")
    parser.add_argument("--prompt", default="What is the transformer model?")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Maps to tokenizer.apply_chat_template(enable_thinking=...). "
            "Only affects templates that recognize the kwarg."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for both the inner runner and the outer orchestrator."""
    parser = argparse.ArgumentParser(description=__doc__)
    _add_common_args(parser)
    parser.add_argument(
        "--inner",
        action="store_true",
        help="Run as a subprocess: load model, do (warmup + )measured generate, emit one JSON line.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Inner mode only: skip the warmup generate; measured generate is the process's first.",
    )
    parser.add_argument("--n-warmup-runs", type=int, default=3, help="Outer mode: subprocess count WITH warmup.")
    parser.add_argument("--n-baseline-runs", type=int, default=3, help="Outer mode: subprocess count WITHOUT warmup.")
    parser.add_argument("--output-dir", default="debug/warmup_probe", help="Outer mode: directory for probe_report.json.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Outer mode: Python interpreter path used to launch inner subprocesses.",
    )
    return parser.parse_args()


def _apply_chat_template(tokenizer: Any, prompt: str, enable_thinking: bool) -> str:
    """Apply the tokenizer chat template with optional ``enable_thinking`` support."""
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            if not enable_thinking:
                print(
                    "warning: tokenizer.apply_chat_template does not accept enable_thinking; "
                    "--no-enable-thinking has no effect for this template",
                    file=sys.stderr,
                )
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
    print("warning: tokenizer has no chat_template; using raw prompt", file=sys.stderr)
    return prompt


def _resolve_sampling(args: argparse.Namespace, model: Any) -> tuple[bool, Optional[float], Optional[int]]:
    """Resolve effective ``(do_sample, temperature, top_k)`` mirroring the other probes.

    A temperature at or below ``1e-5`` collapses to greedy regardless of ``--do-sample``.
    """
    do_sample = bool(args.do_sample)
    resolved_temperature = (
        args.temperature if args.temperature is not None else getattr(model.generation_config, "temperature", None)
    )
    resolved_top_k = args.top_k if args.top_k is not None else getattr(model.generation_config, "top_k", None)
    if resolved_temperature is not None and float(resolved_temperature) <= 1e-5:
        do_sample = False
    return do_sample, resolved_temperature, resolved_top_k


def _run_inner(args: argparse.Namespace) -> int:
    """Load the model, optionally warm up, run a measured generate, and print one JSON line."""
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from mblt_model_zoo.cli.chat import register_mobilint_models

    trust_remote_code = bool(args.trust_remote_code)

    registration_args = Namespace(
        model_name_or_path_or_address=args.model,
        model_revision=None,
        trust_remote_code=trust_remote_code,
    )
    register_mobilint_models(registration_args, transformers)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=trust_remote_code)

    chat_prompt = _apply_chat_template(tokenizer, args.prompt, bool(args.enable_thinking))
    encoded = tokenizer(chat_prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]

    do_sample, temperature, top_k = _resolve_sampling(args, model)
    seed = int(args.seed)
    max_new_tokens = int(args.max_new_tokens)

    if not bool(args.skip_warmup):
        torch.manual_seed(seed)
        with torch.inference_mode():
            _ = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
            )

    torch.manual_seed(seed)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
        )

    row = output_ids[0] if output_ids.ndim > 1 else output_ids
    token_ids = [int(t) for t in row.detach().cpu().tolist()]
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    decoded_sha256 = hashlib.sha256(decoded.encode("utf-8")).hexdigest()
    output_ids_sha256 = hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    payload = {
        "pid": int(os.getpid()),
        "skip_warmup": bool(args.skip_warmup),
        "seed": seed,
        "output_ids": token_ids,
        "decoded": decoded,
        "decoded_sha256": decoded_sha256,
        "output_ids_sha256": output_ids_sha256,
    }
    print("__PROBE_JSON__ " + json.dumps(payload, ensure_ascii=False))
    return 0


def _inner_cli_args(args: argparse.Namespace, *, skip_warmup: bool) -> list[str]:
    """Build the argv slice passed to an inner subprocess, mirroring the current common args."""
    cli: list[str] = [
        "--inner",
        "--model", str(args.model),
        "--prompt", str(args.prompt),
        "--seed", str(int(args.seed)),
        "--max-new-tokens", str(int(args.max_new_tokens)),
    ]
    cli.append("--do-sample" if bool(args.do_sample) else "--no-do-sample")
    if args.temperature is not None:
        cli.extend(["--temperature", str(float(args.temperature))])
    if args.top_k is not None:
        cli.extend(["--top-k", str(int(args.top_k))])
    cli.append("--enable-thinking" if bool(args.enable_thinking) else "--no-enable-thinking")
    cli.append("--trust-remote-code" if bool(args.trust_remote_code) else "--no-trust-remote-code")
    if skip_warmup:
        cli.append("--skip-warmup")
    return cli


def _parse_inner_output(stdout: str) -> dict[str, Any]:
    """Return the payload from the last ``__PROBE_JSON__`` line emitted by an inner subprocess.

    NPU driver banners and framework logs often print earlier lines, so we scan bottom-up for
    the sentinel prefix and parse the JSON that follows it.
    """
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("__PROBE_JSON__ "):
            return json.loads(stripped[len("__PROBE_JSON__ "):])
    raise RuntimeError("inner subprocess did not emit a __PROBE_JSON__ result line")


def _launch_inner(
    args: argparse.Namespace, *, skip_warmup: bool, index: int, label: str
) -> dict[str, Any]:
    """Run one inner subprocess, print a live status line, and return the parsed payload."""
    cmd = [str(args.python), str(Path(__file__).resolve()), *_inner_cli_args(args, skip_warmup=skip_warmup)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(
            f"{label} run {index + 1}: FAILED with exit code {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
            file=sys.stderr,
        )
        result.check_returncode()
    payload = _parse_inner_output(result.stdout)
    print(
        f"{label} run {index + 1} (pid {payload['pid']}): decoded_sha256 = {payload['decoded_sha256'][:12]}..."
    )
    return payload


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return uniqueness stats plus the ordered list of decoded SHA-256 digests."""
    digests = [r["decoded_sha256"] for r in results]
    unique = sorted(set(digests))
    return {
        "count": len(results),
        "unique_count": len(unique),
        "all_identical": len(unique) <= 1,
        "digests": digests,
        "unique_digests": unique,
    }


def _verdict(baseline: dict[str, Any], warmup: dict[str, Any]) -> str:
    """Map baseline/warmup uniqueness into the four verdict strings from the docstring."""
    baseline_ok = bool(baseline["all_identical"])
    warmup_ok = bool(warmup["all_identical"])
    if baseline_ok and warmup_ok:
        return (
            "INCONCLUSIVE: baseline is already deterministic on this machine. "
            "The earlier flip may have been rare; rerun with --n-baseline-runs 10."
        )
    if not baseline_ok and warmup_ok:
        return "warmup FIXES the cross-process variance."
    if not baseline_ok and not warmup_ok:
        return "warmup does NOT help; state survives across warmup. Escalate to MXQ runtime team."
    return "warmup HURTS (baseline deterministic, warmup diverges). Investigate the warmup path."


def _print_block(title: str, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Print the per-process digest table and the uniqueness summary for one condition."""
    print(f"=== {title} ===")
    for idx, payload in enumerate(results):
        note = ""
        if idx > 0 and payload["decoded_sha256"] != results[0]["decoded_sha256"]:
            note = "   <-- differs from process 1"
        print(
            f"process {idx + 1} (pid {payload['pid']}): decoded_sha256 = {payload['decoded_sha256'][:16]}{note}"
        )
    print()
    print(f"{title} unique output count: {summary['unique_count']} / {summary['count']}")
    print(f"{title} all-identical: {summary['all_identical']}")
    print()


def _run_outer(args: argparse.Namespace) -> int:
    """Launch baseline then warmup subprocesses sequentially, then summarize and write JSON."""
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_baseline = int(args.n_baseline_runs)
    n_warmup = int(args.n_warmup_runs)
    if n_baseline < 0 or n_warmup < 0:
        raise SystemExit("--n-baseline-runs and --n-warmup-runs must be non-negative.")
    if n_baseline == 0 and n_warmup == 0:
        raise SystemExit("at least one of --n-baseline-runs or --n-warmup-runs must be > 0.")

    print(
        f"Launching {n_baseline} baseline (no-warmup) subprocesses then {n_warmup} warmup subprocesses "
        f"with seed={int(args.seed)} on {args.model}.\n"
    )

    baseline_results: list[dict[str, Any]] = []
    for i in range(n_baseline):
        baseline_results.append(_launch_inner(args, skip_warmup=True, index=i, label="baseline"))
    print()

    warmup_results: list[dict[str, Any]] = []
    for i in range(n_warmup):
        warmup_results.append(_launch_inner(args, skip_warmup=False, index=i, label="warmup"))
    print()

    baseline_summary = _summarize(baseline_results) if baseline_results else {
        "count": 0,
        "unique_count": 0,
        "all_identical": True,
        "digests": [],
        "unique_digests": [],
    }
    warmup_summary = _summarize(warmup_results) if warmup_results else {
        "count": 0,
        "unique_count": 0,
        "all_identical": True,
        "digests": [],
        "unique_digests": [],
    }

    if baseline_results:
        _print_block("Baseline (no warmup, --skip-warmup)", baseline_results, baseline_summary)
    if warmup_results:
        _print_block("With warmup", warmup_results, warmup_summary)

    if baseline_results and warmup_results:
        verdict = _verdict(baseline_summary, warmup_summary)
    else:
        verdict = "SKIPPED: need both baseline and warmup runs to render a verdict."
    print(f"VERDICT: {verdict}")

    report = {
        "model": args.model,
        "prompt": args.prompt,
        "seed": int(args.seed),
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": bool(args.do_sample),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "enable_thinking": bool(args.enable_thinking),
        "n_baseline_runs": n_baseline,
        "n_warmup_runs": n_warmup,
        "python": str(args.python),
        "baseline": {"summary": baseline_summary, "runs": baseline_results},
        "warmup": {"summary": warmup_summary, "runs": warmup_results},
        "verdict": verdict,
    }
    report_path = output_dir / "probe_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {report_path}")
    return 0


def main() -> int:
    """Dispatch to the inner runner or the outer orchestrator based on ``--inner``."""
    args = _parse_args()
    if bool(args.inner):
        return _run_inner(args)
    return _run_outer(args)


if __name__ == "__main__":
    raise SystemExit(main())
