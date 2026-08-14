"""Probe run-to-run determinism of Mobilint EAGLE-3 MXQ backends.

Captures the exact numpy input tuple that the base, draft, and FC-projector MXQ
backends see during a single ``initialize_tree`` prefill, then replays that
identical input ``--n-runs`` times against each backend's ``.infer(...)`` and
reports per-output-slot bit-exact identity plus max/mean absolute diffs against
the first run.

The intent is to pin down which MXQ module (if any) produces run-to-run drift
when fed byte-identical inputs, which is what causes intermittent acceptance
divergence in :mod:`scripts.debug_eagle3_tree_trace`.
"""

from __future__ import annotations

import argparse
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
from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintEagle3Cache  # noqa: E402
from mblt_model_zoo.hf_transformers.utils.eagle3.tree_decoding import initialize_tree  # noqa: E402


_MODULE_KEYS = ("base", "draft", "fc")

_KNOWN_OUTPUT_LABELS: dict[str, dict[int, list[str]]] = {
    "base": {4: ["hidden1", "hidden2", "hidden3", "logits"]},
    "draft": {2: ["layer_outputs", "last_hidden_logits"]},
    "fc": {1: ["projected"]},
}


class _InferSpy:
    """Transparent proxy over an MXQ model that snapshots the first ``.infer`` call.

    The proxy delegates every attribute lookup to the wrapped runtime handle. On
    its first ``.infer(...)`` call, it deep-copies the positional and keyword
    arguments into ``storage``; subsequent calls delegate without re-recording,
    which keeps prefill chunking or draft-tree depth iterations transparent.
    """

    def __init__(self, real: Any, storage: dict[str, Any]) -> None:
        self._real = real
        self._storage = storage

    def infer(self, *args: Any, **kwargs: Any) -> Any:
        """Record args on the first invocation, then delegate to the real handle."""
        if "args" not in self._storage:
            self._storage["args"] = tuple(_clone_arg(a) for a in args)
            self._storage["kwargs"] = {k: _clone_arg(v) for k, v in kwargs.items()}
        return self._real.infer(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _clone_arg(arg: Any) -> Any:
    """Deep-copy numpy arrays and lists so later mutations do not corrupt the capture."""
    if isinstance(arg, np.ndarray):
        return arg.copy()
    if isinstance(arg, list):
        return [_clone_arg(item) for item in arg]
    if isinstance(arg, tuple):
        return tuple(_clone_arg(item) for item in arg)
    return arg


def _install_spies(model: Any, modules: list[str]) -> dict[str, dict[str, Any]]:
    """Replace ``get_mxq_model`` on each requested submodule with an ``_InferSpy``-returning stub.

    Args:
        model: An EAGLE-3 causal LM that exposes ``eagle3_base_model``, ``eagle3_draft_model``,
            and ``eagle3_fc_projector`` children.
        modules: Subset of ``{"base", "draft", "fc"}`` naming which backends to spy on.

    Returns:
        Mapping ``module_name -> storage`` where each storage dict is populated with ``args``,
        ``kwargs``, and ``handle`` (the real MXQ model) once the first ``.infer`` call fires.
    """
    submodules = {
        "base": model.eagle3_base_model,
        "draft": model.eagle3_draft_model,
        "fc": model.eagle3_fc_projector,
    }
    storages: dict[str, dict[str, Any]] = {}
    for name in modules:
        submodule = submodules[name]
        real_handle = submodule.get_mxq_model()
        storage: dict[str, Any] = {"handle": real_handle}
        spy = _InferSpy(real_handle, storage)
        submodule.get_mxq_model = lambda spy=spy: spy  # type: ignore[method-assign]
        storages[name] = storage
    return storages


def _replay(handle: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[np.ndarray]:
    """Call the real ``handle.infer`` once with copies of the captured args and return outputs.

    Each replay uses freshly cloned inputs so the runtime cannot mutate the shared capture, and
    the returned array list is copied so subsequent replays do not overwrite it.
    """
    call_args = tuple(_clone_arg(a) for a in args)
    call_kwargs = {k: _clone_arg(v) for k, v in kwargs.items()}
    result = handle.infer(*call_args, **call_kwargs)
    return [np.asarray(item).copy() for item in result]


def _diff_stats(runs: list[list[np.ndarray]], labels: list[str]) -> list[dict[str, Any]]:
    """Compute per-output-slot bit-exact identity count and max/mean |diff| against run 0.

    Args:
        runs: One list per replayed run, each of length ``num_output_slots``.
        labels: Human-readable names for each slot (e.g. ``"hidden1"``, ``"logits"``).

    Returns:
        A list of per-slot descriptors, one per output slot, ordered like ``labels``.
    """
    if not runs:
        return []
    ref_run = runs[0]
    n_runs = len(runs)
    descriptors: list[dict[str, Any]] = []
    for slot_idx, ref_arr in enumerate(ref_run):
        n_identical = 1
        max_abs = 0.0
        mean_abs_of_worst = 0.0
        ref_bytes = ref_arr.tobytes()
        ref_double: Optional[np.ndarray] = None
        for other in runs[1:]:
            other_arr = other[slot_idx]
            if other_arr.shape != ref_arr.shape:
                raise RuntimeError(
                    f"MXQ output slot {slot_idx} shape changed across runs: {ref_arr.shape} vs {other_arr.shape}"
                )
            if other_arr.tobytes() == ref_bytes:
                n_identical += 1
                continue
            if ref_double is None:
                ref_double = ref_arr.astype(np.float64, copy=False)
            diff = np.abs(other_arr.astype(np.float64, copy=False) - ref_double)
            slot_max = float(diff.max())
            slot_mean = float(diff.mean())
            if slot_max > max_abs:
                max_abs = slot_max
            if slot_mean > mean_abs_of_worst:
                mean_abs_of_worst = slot_mean
        descriptors.append(
            {
                "name": labels[slot_idx] if slot_idx < len(labels) else f"out{slot_idx}",
                "shape": list(ref_arr.shape),
                "dtype": str(ref_arr.dtype),
                "n_bitwise_identical": n_identical,
                "n_runs": n_runs,
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs_of_worst,
            }
        )
    return descriptors


def _labels_for(module: str, count: int) -> list[str]:
    labels = _KNOWN_OUTPUT_LABELS.get(module, {}).get(count)
    return labels if labels is not None else [f"out{i}" for i in range(count)]


def _describe_inputs(args: tuple[Any, ...]) -> tuple[list[list[int]], Optional[int]]:
    """Return the shapes of the captured input array list and the cache position, if present.

    MXQ ``.infer`` conventions in this repo:
    - Base/draft: ``.infer([np.ndarray, ...], None, cache_position:int)``
    - FC projector: ``.infer([np.ndarray])`` (no cache position)
    """
    if not args:
        return [], None
    first = args[0]
    if isinstance(first, list):
        shapes = [list(np.asarray(arr).shape) for arr in first]
    else:
        shapes = [list(np.asarray(first).shape)]
    cache_position: Optional[int] = None
    if len(args) >= 3 and isinstance(args[2], int):
        cache_position = int(args[2])
    return shapes, cache_position


def _format_shape(shape: list[int]) -> str:
    return "[" + ",".join(str(s) for s in shape) + "]"


def _print_summary(module_reports: dict[str, dict[str, Any]]) -> None:
    """Print a compact per-module stdout summary, tagging any non-deterministic slot."""
    for module_name, report in module_reports.items():
        print(f"== {module_name} MXQ ==")
        outputs = report.get("outputs", [])
        if not outputs:
            print("  (no capture — module was not exercised)")
            continue
        max_name_len = max(len(o["name"]) for o in outputs)
        max_shape_len = max(len(_format_shape(o["shape"])) for o in outputs)
        for slot in outputs:
            tag = ""
            if int(slot["n_bitwise_identical"]) < int(slot["n_runs"]):
                tag = "   <-- NON-DETERMINISTIC"
            name = slot["name"].ljust(max_name_len)
            shape_str = _format_shape(slot["shape"]).ljust(max_shape_len)
            identical = f"{slot['n_bitwise_identical']}/{slot['n_runs']}"
            max_abs = float(slot["max_abs_diff"])
            print(
                f"  {name} {shape_str}   identical: {identical}    max_abs_diff: {max_abs:.6g}{tag}"
            )


def _run_consecutive(
    module_name: str, storages: dict[str, dict[str, Any]], n_runs: int
) -> list[list[np.ndarray]]:
    storage = storages[module_name]
    if "args" not in storage:
        return []
    handle = storage["handle"]
    args = storage["args"]
    kwargs = storage.get("kwargs", {})
    return [_replay(handle, args, kwargs) for _ in range(n_runs)]


def _run_interleaved(
    module_order: list[str], storages: dict[str, dict[str, Any]], n_runs: int
) -> dict[str, list[list[np.ndarray]]]:
    ready = [name for name in module_order if "args" in storages.get(name, {})]
    collected: dict[str, list[list[np.ndarray]]] = {name: [] for name in ready}
    for _ in range(n_runs):
        for name in ready:
            storage = storages[name]
            collected[name].append(_replay(storage["handle"], storage["args"], storage.get("kwargs", {})))
    return collected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mobilint/EAGLE3-Qwen3-4B")
    parser.add_argument("--prompt", default="What is the transformer model?")
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument(
        "--modules",
        default="base,draft,fc",
        help="Comma-separated MXQ modules to probe; subset of base,draft,fc.",
    )
    parser.add_argument("--output-dir", default="debug/mxq_determinism_probe")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Passed to tokenizer.apply_chat_template when the template supports it.",
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help=(
            "Instead of N consecutive calls per module, cycle base -> draft -> fc "
            "N times. Reveals cross-handle interference on determinism."
        ),
    )
    return parser.parse_args()


def _resolve_modules(raw: str) -> list[str]:
    requested = [tok.strip().lower() for tok in raw.split(",") if tok.strip()]
    unknown = [tok for tok in requested if tok not in _MODULE_KEYS]
    if unknown:
        raise SystemExit(f"Unknown module(s) in --modules: {unknown}. Choose from {list(_MODULE_KEYS)}.")
    ordered = [name for name in _MODULE_KEYS if name in requested]
    if not ordered:
        raise SystemExit(f"--modules resolved to empty set; choose from {list(_MODULE_KEYS)}.")
    return ordered


def main() -> int:
    """Load an EAGLE-3 model, capture per-backend MXQ inputs, and report replay diffs."""
    args = _parse_args()
    modules = _resolve_modules(args.modules)
    n_runs = int(args.n_runs)
    if n_runs < 2:
        raise SystemExit(f"--n-runs must be >= 2 to compare runs; got {n_runs}.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trust_remote_code = bool(args.trust_remote_code)
    torch.manual_seed(int(args.seed))

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
    if getattr(tokenizer, "chat_template", None):
        try:
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
    else:
        print("warning: tokenizer has no chat_template; using raw prompt", file=sys.stderr)
        chat_prompt = args.prompt

    encoded = tokenizer(chat_prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    prompt_len = int(input_ids.shape[1])

    base_mxq, draft_mxq = model.get_cache_mxq_models()
    cache = MobilintEagle3Cache(base_mxq, draft_mxq)

    storages = _install_spies(model, modules)

    with torch.inference_mode():
        initialize_tree(input_ids, model, cache, logits_processor=None)

    module_reports: dict[str, dict[str, Any]] = {}

    if args.interleave:
        collected = _run_interleaved(modules, storages, n_runs)
        for name in modules:
            runs = collected.get(name, [])
            storage = storages.get(name, {})
            shapes, cache_position = _describe_inputs(storage.get("args", ()))
            outputs = _diff_stats(runs, _labels_for(name, len(runs[0]) if runs else 0))
            module_reports[name] = {
                "captured": "args" in storage,
                "input_shapes": shapes,
                "cache_position": cache_position,
                "n_runs_actual": len(runs),
                "outputs": outputs,
            }
    else:
        for name in modules:
            runs = _run_consecutive(name, storages, n_runs)
            storage = storages.get(name, {})
            shapes, cache_position = _describe_inputs(storage.get("args", ()))
            outputs = _diff_stats(runs, _labels_for(name, len(runs[0]) if runs else 0))
            module_reports[name] = {
                "captured": "args" in storage,
                "input_shapes": shapes,
                "cache_position": cache_position,
                "n_runs_actual": len(runs),
                "outputs": outputs,
            }

    report = {
        "model": args.model,
        "prompt": args.prompt,
        "chat_prompt": chat_prompt,
        "prompt_len": prompt_len,
        "n_runs": n_runs,
        "interleave": bool(args.interleave),
        "modules": module_reports,
    }

    _print_summary(module_reports)

    report_path = output_dir / "probe_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
