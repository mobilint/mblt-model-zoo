"""Probe within-process determinism of ``model.generate()`` for EAGLE-3.

Runs ``model.generate()`` back-to-back N times in the same Python process with
identical seed and prompt, then compares outputs bit-exactly. Also SHA-256
hashes the first draft-MXQ ``.infer(...)`` input and output on each generate
call to distinguish upstream (CPU-side) drift from downstream (MXQ kernel)
drift.

Interpretation of the four boolean lines printed at the end:

- All True: within-process is fully deterministic. Any cross-process variance
  seen previously is SDK-init / DMA-timing / warmup state, not a within-process
  race.
- Outputs differ, first-call input hashes EQUAL but first-call output hashes
  DIFFER: MXQ draft has call-history-dependent internal state across generates
  (unlikely given prior probe results, but rules it in/out directly).
- Outputs differ, first-call input hashes DIFFER: CPU-side preprocessing
  produced a different numpy input to the draft between generates. Chase
  upstream: torch threading FP order, rotary lookup, or attention mask.
- Outputs differ, first-call hashes all equal: divergence appears in later
  draft calls or in the sampling RNG cascade after the first tree step. Extend
  the wrapper to capture all draft calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")

import transformers  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from mblt_model_zoo.cli.chat import register_mobilint_models  # noqa: E402


def _sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _describe_array(arr: np.ndarray) -> dict[str, Any]:
    """Return a JSON-friendly descriptor of a numpy array with its byte digest."""
    contig = np.ascontiguousarray(arr)
    return {
        "sha256": _sha256_bytes(contig.tobytes()),
        "shape": list(contig.shape),
        "dtype": str(contig.dtype),
        "nbytes": int(contig.nbytes),
    }


def _describe_input_arg(arg: Any) -> Any:
    """Describe one positional argument to ``.infer(...)``.

    The Mobilint MXQ ``.infer`` convention passes ``[np.ndarray, ...]`` as the
    first positional argument (the list of input tensors), an optional extra
    ``None``/scratch buffer as the second, and an integer cache position as the
    third. This helper handles all three shapes uniformly.
    """
    if isinstance(arg, np.ndarray):
        return _describe_array(arg)
    if isinstance(arg, list):
        return [_describe_input_arg(item) for item in arg]
    if isinstance(arg, tuple):
        return [_describe_input_arg(item) for item in arg]
    if isinstance(arg, (int, float, str, bool)) or arg is None:
        return arg
    return f"<{type(arg).__name__}>"


class _FirstCallCapture:
    """Shared state that arms the draft-infer wrapper once per iteration."""

    def __init__(self) -> None:
        self.armed: bool = False
        self.iteration: int = -1
        self.captured: Optional[dict[str, Any]] = None

    def arm(self, iteration: int) -> None:
        """Arm the wrapper to capture the next ``.infer(...)`` call."""
        self.armed = True
        self.iteration = iteration
        self.captured = None

    def record(self, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
        """Record the input and output of the current ``.infer(...)`` call, then disarm.

        ``result`` is assumed to be an iterable of numpy arrays (the standard
        draft MXQ backend returns two output tensors); we digest each one.
        """
        if not self.armed:
            return
        input_descriptors = [_describe_input_arg(a) for a in args]
        input_kwargs = {k: _describe_input_arg(v) for k, v in kwargs.items()}
        try:
            out_list = list(result)
            output_descriptors = [_describe_array(np.asarray(o)) for o in out_list]
        except (TypeError, ValueError):
            output_descriptors = [{"error": "output not iterable of arrays", "repr": repr(result)}]
        self.captured = {
            "iteration": self.iteration,
            "input_args": input_descriptors,
            "input_kwargs": input_kwargs,
            "outputs": output_descriptors,
        }
        self.armed = False


class _DraftInferProxy:
    """Transparent proxy over the draft MXQ handle with a one-shot capture hook."""

    def __init__(self, real: Any, capture: _FirstCallCapture) -> None:
        self._real = real
        self._capture = capture

    def infer(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the real handle; on the first armed call, record hashes."""
        result = self._real.infer(*args, **kwargs)
        if self._capture.armed:
            self._capture.record(args, kwargs, result)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _install_draft_proxy(model: Any, capture: _FirstCallCapture) -> None:
    """Replace ``model.eagle3_draft_model.get_mxq_model`` with a proxy factory.

    The proxy is created once against the real handle and is stable across
    generate calls, so the MXQ handle itself is never reset — only the ``armed``
    flag on ``capture`` toggles per iteration.
    """
    submodule = model.eagle3_draft_model
    real_handle = submodule.get_mxq_model()
    proxy = _DraftInferProxy(real_handle, capture)
    submodule.get_mxq_model = lambda proxy=proxy: proxy  # type: ignore[method-assign]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the same-process generate determinism probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mobilint/EAGLE3-Qwen3-4B")
    parser.add_argument("--prompt", default="What is the transformer model?")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--n-generates", type=int, default=2)
    parser.add_argument("--output-dir", default="debug/generate_same_process_probe")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Maps to tokenizer.apply_chat_template(enable_thinking=...). "
            "Only affects templates that recognize the kwarg."
        ),
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


def _first_diff_index(a: list[int], b: list[int]) -> Optional[int]:
    """Return the first index where two token id lists differ, or ``None`` if equal-length identical."""
    lim = min(len(a), len(b))
    for i in range(lim):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return lim
    return None


def _input_output_hash_tuple(capture: Optional[dict[str, Any]]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract input and output digest tuples from a capture dict.

    Returns a pair ``(input_hashes, output_hashes)`` where each element is a
    tuple of SHA-256 hex strings ordered by the arg / output slot. Missing
    captures yield empty tuples.
    """
    if capture is None:
        return (), ()
    input_hashes: list[str] = []

    def walk(desc: Any) -> None:
        if isinstance(desc, dict) and "sha256" in desc:
            input_hashes.append(str(desc["sha256"]))
            return
        if isinstance(desc, list):
            for item in desc:
                walk(item)

    for arg_desc in capture.get("input_args", []):
        walk(arg_desc)

    output_hashes = tuple(
        str(o.get("sha256", "")) for o in capture.get("outputs", []) if isinstance(o, dict)
    )
    return tuple(input_hashes), output_hashes


def main() -> int:
    """Run ``generate`` ``--n-generates`` times, capture first-draft-call digests, and report."""
    args = _parse_args()
    n_generates = int(args.n_generates)
    if n_generates < 2:
        raise SystemExit(f"--n-generates must be >= 2 to compare iterations; got {n_generates}.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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

    enable_thinking = bool(args.enable_thinking)
    chat_prompt = _apply_chat_template(tokenizer, args.prompt, enable_thinking)

    encoded = tokenizer(chat_prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]

    do_sample = bool(args.do_sample)
    resolved_temperature = (
        args.temperature if args.temperature is not None else getattr(model.generation_config, "temperature", None)
    )
    resolved_top_k = args.top_k if args.top_k is not None else getattr(model.generation_config, "top_k", None)
    if resolved_temperature is not None and float(resolved_temperature) <= 1e-5:
        do_sample = False

    capture = _FirstCallCapture()
    _install_draft_proxy(model, capture)

    iterations: list[dict[str, Any]] = []
    for i in range(n_generates):
        torch.manual_seed(int(args.seed))
        capture.arm(i)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=do_sample,
                temperature=resolved_temperature,
                top_k=resolved_top_k,
            )
        row = output_ids[0] if output_ids.ndim > 1 else output_ids
        token_ids = [int(t) for t in row.detach().cpu().tolist()]
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        iterations.append(
            {
                "iteration": i,
                "output_ids": token_ids,
                "decoded": decoded,
                "draft_first_call": capture.captured,
            }
        )

    output_ids_lists = [it["output_ids"] for it in iterations]
    decoded_list = [it["decoded"] for it in iterations]
    first_ids = output_ids_lists[0]
    first_dec = decoded_list[0]
    output_ids_all_equal = all(ids == first_ids for ids in output_ids_lists[1:])
    decoded_all_equal = all(dec == first_dec for dec in decoded_list[1:])

    hash_tuples = [_input_output_hash_tuple(it["draft_first_call"]) for it in iterations]
    first_input_hashes, first_output_hashes = hash_tuples[0]
    input_hashes_all_equal = all(ih == first_input_hashes for ih, _ in hash_tuples[1:])
    output_hashes_all_equal = all(oh == first_output_hashes for _, oh in hash_tuples[1:])

    pairwise: dict[str, Optional[int]] = {}
    for a in range(n_generates):
        for b in range(a + 1, n_generates):
            pairwise[f"({a},{b})"] = _first_diff_index(output_ids_lists[a], output_ids_lists[b])

    report = {
        "model": args.model,
        "prompt": args.prompt,
        "chat_prompt": chat_prompt,
        "seed": int(args.seed),
        "generation_config_effective": {
            "do_sample": do_sample,
            "temperature": resolved_temperature,
            "top_k": resolved_top_k,
            "max_new_tokens": int(args.max_new_tokens),
        },
        "enable_thinking": enable_thinking,
        "n_generates": n_generates,
        "iterations": iterations,
        "comparison": {
            "output_ids_all_equal": bool(output_ids_all_equal),
            "decoded_all_equal": bool(decoded_all_equal),
            "draft_first_call_input_hashes_all_equal": bool(input_hashes_all_equal),
            "draft_first_call_output_hashes_all_equal": bool(output_hashes_all_equal),
            "pairwise_output_ids_diff_positions": pairwise,
        },
    }

    report_path = output_dir / "probe_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"n_generates={n_generates}")
    print(f"output_ids equal across all iterations: {bool(output_ids_all_equal)}")
    print(f"decoded equal: {bool(decoded_all_equal)}")
    print(f"first draft.infer input  hashes equal: {bool(input_hashes_all_equal)}")
    print(f"first draft.infer output hashes equal: {bool(output_hashes_all_equal)}")
    print(f"\nWrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
