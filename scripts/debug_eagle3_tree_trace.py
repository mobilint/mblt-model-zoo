"""Trace EAGLE-3 draft trees and posterior acceptance across a ``generate`` call.

Runs one or more ``model.generate()`` calls on a Mobilint EAGLE-3 causal LM and,
for every speculative step, dumps the draft tree plus verification result as JSON
and prints a human-readable ASCII tree.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

import transformers  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from mblt_model_zoo.cli.chat import register_mobilint_models  # noqa: E402
from mblt_model_zoo.hf_transformers.utils.eagle3 import decoding as _decoding_mod  # noqa: E402
from mblt_model_zoo.hf_transformers.utils.eagle3 import tree_decoding as _tree_mod  # noqa: E402


_TRACE: list[dict[str, Any]] = []
_LAST_TREE: dict[str, torch.Tensor] = {}
_STEP_DATA: dict[str, Any] = {}
_STEP_INDEX = 0
_TOKENIZER: Any = None


def _escape_token_text(text: str) -> str:
    """Escape backslash, quote, and whitespace control characters for tree display."""
    return (
        text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace('"', '\\"')
    )


def _decode_token(token_id: int) -> str:
    """Decode a single token id via the global tokenizer, without special-token stripping."""
    if _TOKENIZER is None:
        return f"<{token_id}>"
    try:
        return _TOKENIZER.decode([int(token_id)], skip_special_tokens=False)
    except (RuntimeError, ValueError, TypeError):
        return f"<{token_id}>"


def _build_draft_tree_dict(
    draft_tokens: torch.Tensor,
    tree_mask: torch.Tensor,
    tree_position_ids: torch.Tensor,
) -> dict[str, Any]:
    """Reconstruct the draft tree nodes from mask + position ids.

    Args:
        draft_tokens: ``[1, tree_nodes]`` tensor of token ids (root + expansions).
        tree_mask: ``[1, 1, tree_nodes, tree_nodes]`` mask; ``mask[i, j] == 1`` iff ``j``
            is on ``i``'s ancestor path (inclusive).
        tree_position_ids: ``[tree_nodes]`` per-node depth.

    Returns:
        Dict with ``nodes`` list and ``total_tokens`` count.
    """
    tokens = draft_tokens[0].detach().cpu().tolist()
    positions = tree_position_ids.detach().cpu().tolist()
    mask = tree_mask.detach().cpu()[0, 0]
    total = len(tokens)
    nodes: list[dict[str, Any]] = []
    for i in range(total):
        parent_id: Optional[int] = None
        if i > 0:
            for j in range(i):
                if int(mask[i, j]) == 1:
                    parent_id = j
        tok = int(tokens[i])
        nodes.append(
            {
                "node_id": i,
                "parent_id": parent_id,
                "depth": int(positions[i]),
                "token_id": tok,
                "token_text": _decode_token(tok),
            }
        )
    return {"nodes": nodes, "total_tokens": total}


def _build_candidates(candidates_tensor: torch.Tensor, retrieve_indices: torch.Tensor) -> list[dict[str, Any]]:
    """Convert ``candidates`` and ``retrieve_indices`` into per-leaf path descriptors."""
    if candidates_tensor is None or retrieve_indices.numel() == 0:
        return []
    ri = retrieve_indices.detach().cpu()
    cand = candidates_tensor.detach().cpu()
    result: list[dict[str, Any]] = []
    for leaf_idx in range(ri.shape[0]):
        row_nodes = ri[leaf_idx].tolist()
        row_tokens = cand[leaf_idx].tolist()
        clean_nodes: list[int] = []
        clean_tokens: list[int] = []
        for node_id, token_id in zip(row_nodes, row_tokens):
            if int(node_id) < 0:
                break
            clean_nodes.append(int(node_id))
            clean_tokens.append(int(token_id))
        result.append(
            {
                "leaf_index": leaf_idx,
                "node_ids": clean_nodes,
                "token_ids": clean_tokens,
                "token_texts": [_decode_token(t) for t in clean_tokens],
            }
        )
    return result


def _sample_p_topk(
    sample_p: torch.Tensor,
    sampled_indices: Optional[torch.Tensor],
    k: int = 10,
) -> list[dict[str, Any]]:
    """Return the top-``k`` (token_id, prob) entries from ``sample_p``.

    When ``sampled_indices`` is ``None`` (greedy branch), ``sample_p`` is a raw
    vocab-wide logits row; softmax before topk. Otherwise ``sample_p`` and
    ``sampled_indices`` are aligned 1-D vectors of length up to ``k``.
    """
    sp = sample_p.detach().float().cpu()
    if sampled_indices is None:
        probs = torch.softmax(sp, dim=-1)
        limit = min(k, int(probs.numel()))
        top_vals, top_idx = torch.topk(probs, k=limit, dim=-1)
        return [
            {
                "token_id": int(top_idx[i].item()),
                "token_text": _decode_token(int(top_idx[i].item())),
                "prob": float(top_vals[i].item()),
            }
            for i in range(limit)
        ]
    si = sampled_indices.detach().cpu().view(-1)
    sp = sp.view(-1)
    limit = min(int(sp.numel()), int(si.numel()), k)
    return [
        {
            "token_id": int(si[i].item()),
            "token_text": _decode_token(int(si[i].item())),
            "prob": float(sp[i].item()),
        }
        for i in range(limit)
    ]


def _render_tree_ascii(
    *,
    step_index: int,
    prompt_len: int,
    tree_nodes: list[dict[str, Any]],
    accepted_count: int,
    tree_size: int,
    best_leaf: Optional[int],
    per_node_accepted: list[bool],
    best_path_set: set[int],
) -> str:
    """Render a single step as an ASCII tree block with a header line."""
    if accepted_count >= 0:
        header_acc = f"{accepted_count + 1}/{tree_size}"
    else:
        header_acc = f"-/{tree_size}"
    header_leaf = "-" if best_leaf is None else str(best_leaf)
    lines: list[str] = [
        f"=== step {step_index} | prompt_len={prompt_len} | accepted={header_acc} | best_leaf={header_leaf} ==="
    ]

    parent_to_children: dict[Optional[int], list[int]] = {}
    node_by_id: dict[int, dict[str, Any]] = {}
    for node in tree_nodes:
        parent_to_children.setdefault(node["parent_id"], []).append(node["node_id"])
        node_by_id[node["node_id"]] = node
    for children in parent_to_children.values():
        children.sort()

    def fmt(node: dict[str, Any], prefix: str, connector: str) -> str:
        accepted = per_node_accepted[node["node_id"]] if node["node_id"] < len(per_node_accepted) else False
        acc_marker = "[OK]" if accepted else "[X ]"
        best_marker = "*" if node["node_id"] in best_path_set else " "
        text = _escape_token_text(node["token_text"])
        return (
            f"{prefix}{connector}{acc_marker}{best_marker} "
            f"id={node['node_id']:3d} d={node['depth']} tok={node['token_id']:>6d} \"{text}\""
        )

    def walk(node_id: int, prefix: str, is_last: bool) -> None:
        node = node_by_id[node_id]
        if node["parent_id"] is None:
            lines.append(fmt(node, "", ""))
            child_prefix = ""
        else:
            connector = "\\-- " if is_last else "+-- "
            lines.append(fmt(node, prefix, connector))
            child_prefix = prefix + ("    " if is_last else "|   ")
        children = parent_to_children.get(node_id, [])
        for idx, cid in enumerate(children):
            walk(cid, child_prefix, idx == len(children) - 1)

    if tree_nodes:
        walk(0, "", True)
    return "\n".join(lines)


def _apply_wrappers() -> Any:
    """Monkey-patch the four EAGLE-3 tree primitives; return a restore callable."""
    orig_initialize_tree = _tree_mod.initialize_tree
    orig_tree_decoding = _tree_mod.tree_decoding
    orig_evaluate_posterior = _tree_mod.evaluate_posterior
    orig_update_inference_inputs = _tree_mod.update_inference_inputs

    def wrap_initialize_tree(
        input_ids: torch.LongTensor,
        model: Any,
        cache: Any,
        logits_processor: Any,
        *,
        remaining_tokens: Optional[int] = None,
        count_npu_time: bool = False,
    ) -> Any:
        result = orig_initialize_tree(
            input_ids,
            model,
            cache,
            logits_processor,
            remaining_tokens=remaining_tokens,
            count_npu_time=count_npu_time,
        )
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _ = result
        _LAST_TREE.clear()
        _LAST_TREE.update(
            {
                "draft_tokens": draft_tokens.detach().cpu(),
                "retrieve_indices": retrieve_indices.detach().cpu(),
                "tree_mask": tree_mask.detach().cpu(),
                "tree_position_ids": tree_position_ids.detach().cpu(),
            }
        )
        draft_tree = _build_draft_tree_dict(draft_tokens, tree_mask, tree_position_ids)
        _TRACE.append(
            {
                "step_index": -1,
                "prompt_len_so_far": int(input_ids.shape[1]),
                "draft_tree": draft_tree,
                "candidates": [],
                "posterior": None,
                "committed_tokens": [],
                "eos_stopped": False,
                "timings_ms": {},
            }
        )
        return result

    def wrap_tree_decoding(
        model: Any,
        cache: Any,
        tree_candidates: torch.LongTensor,
        input_ids: torch.LongTensor,
        retrieve_indices: torch.LongTensor,
        tree_position_ids: torch.LongTensor,
        *,
        count_npu_time: bool = False,
    ) -> Any:
        _STEP_DATA["step_index"] = _STEP_INDEX
        _STEP_DATA["prompt_len_so_far"] = int(input_ids.shape[1])
        _STEP_DATA["draft_tokens"] = _LAST_TREE["draft_tokens"]
        _STEP_DATA["retrieve_indices"] = _LAST_TREE["retrieve_indices"]
        _STEP_DATA["tree_mask"] = _LAST_TREE["tree_mask"]
        _STEP_DATA["tree_position_ids"] = _LAST_TREE["tree_position_ids"]
        start = time.perf_counter()
        result = orig_tree_decoding(
            model,
            cache,
            tree_candidates,
            input_ids,
            retrieve_indices,
            tree_position_ids,
            count_npu_time=count_npu_time,
        )
        _STEP_DATA["t_tree_ms"] = (time.perf_counter() - start) * 1000.0
        return result

    def wrap_evaluate_posterior(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor: Any,
        retrieve_indices: torch.Tensor,
    ) -> Any:
        _STEP_DATA["candidates_tensor"] = candidates.detach().cpu()
        _STEP_DATA["logits_processor_none"] = logits_processor is None
        start = time.perf_counter()
        result = orig_evaluate_posterior(logits, candidates, logits_processor, retrieve_indices)
        _STEP_DATA["t_post_ms"] = (time.perf_counter() - start) * 1000.0
        best_candidate, accepted_draft_count, sample_p, sampled_indices = result
        _STEP_DATA["best_candidate"] = int(best_candidate.item())
        _STEP_DATA["accepted_draft_count"] = int(accepted_draft_count.item())
        _STEP_DATA["sample_p"] = sample_p.detach().cpu() if isinstance(sample_p, torch.Tensor) else sample_p
        _STEP_DATA["sampled_indices"] = None if sampled_indices is None else sampled_indices.detach().cpu()
        return result

    def wrap_update_inference_inputs(
        input_ids: torch.LongTensor,
        candidates: torch.Tensor,
        best_candidate: torch.Tensor,
        accepted_draft_count: torch.Tensor,
        retrieve_indices: torch.Tensor,
        logits_processor: Any,
        new_token_count: int,
        model: Any,
        cache: Any,
        hidden_state_new: torch.Tensor,
        sample_p: torch.Tensor,
        sampled_indices: Optional[torch.Tensor],
        *,
        remaining_tokens: Optional[int] = None,
        eos_token_id: Any = None,
        count_npu_time: bool = False,
    ) -> Any:
        global _STEP_INDEX
        input_ids_before = input_ids.detach().cpu().clone()
        start = time.perf_counter()
        result = orig_update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accepted_draft_count,
            retrieve_indices,
            logits_processor,
            new_token_count,
            model,
            cache,
            hidden_state_new,
            sample_p,
            sampled_indices,
            remaining_tokens=remaining_tokens,
            eos_token_id=eos_token_id,
            count_npu_time=count_npu_time,
        )
        t_update_ms = (time.perf_counter() - start) * 1000.0
        (
            new_input_ids,
            new_draft_tokens,
            new_retrieve_indices,
            new_tree_mask,
            new_tree_position_ids,
            _new_token_count,
            should_stop,
        ) = result
        committed = new_input_ids[0, input_ids_before.shape[1] :].detach().cpu().tolist()
        _STEP_DATA["committed_tokens_ids"] = [int(t) for t in committed]
        _STEP_DATA["eos_stopped"] = bool(should_stop)
        _STEP_DATA["t_update_ms"] = t_update_ms
        _finalize_step()
        if not bool(should_stop) and new_draft_tokens.numel() > 0:
            _LAST_TREE.clear()
            _LAST_TREE.update(
                {
                    "draft_tokens": new_draft_tokens.detach().cpu(),
                    "retrieve_indices": new_retrieve_indices.detach().cpu(),
                    "tree_mask": new_tree_mask.detach().cpu(),
                    "tree_position_ids": new_tree_position_ids.detach().cpu(),
                }
            )
        _STEP_INDEX += 1
        _STEP_DATA.clear()
        return result

    for mod in (_tree_mod, _decoding_mod):
        mod.initialize_tree = wrap_initialize_tree
        mod.tree_decoding = wrap_tree_decoding
        mod.evaluate_posterior = wrap_evaluate_posterior
        mod.update_inference_inputs = wrap_update_inference_inputs

    def restore() -> None:
        for mod in (_tree_mod, _decoding_mod):
            mod.initialize_tree = orig_initialize_tree
            mod.tree_decoding = orig_tree_decoding
            mod.evaluate_posterior = orig_evaluate_posterior
            mod.update_inference_inputs = orig_update_inference_inputs

    return restore


def _finalize_step() -> None:
    """Compose the current step's JSON dict from ``_STEP_DATA`` and append to ``_TRACE``."""
    step_index = _STEP_DATA["step_index"]
    draft_tokens = _STEP_DATA["draft_tokens"]
    retrieve_indices = _STEP_DATA["retrieve_indices"]
    tree_mask = _STEP_DATA["tree_mask"]
    tree_position_ids = _STEP_DATA["tree_position_ids"]
    candidates_tensor = _STEP_DATA.get("candidates_tensor")
    best_candidate = int(_STEP_DATA["best_candidate"])
    accepted_draft_count = int(_STEP_DATA["accepted_draft_count"])
    sample_p = _STEP_DATA["sample_p"]
    sampled_indices = _STEP_DATA["sampled_indices"]

    draft_tree = _build_draft_tree_dict(draft_tokens, tree_mask, tree_position_ids)
    total_nodes = draft_tree["total_tokens"]
    per_node_accepted = [False] * total_nodes
    if total_nodes > 0:
        per_node_accepted[0] = True

    ri_row = retrieve_indices[best_candidate].tolist() if retrieve_indices.numel() else []
    accepted_node_ids: list[int] = []
    for idx, node_id in enumerate(ri_row):
        if idx > accepted_draft_count:
            break
        if int(node_id) < 0:
            break
        node_int = int(node_id)
        accepted_node_ids.append(node_int)
        if 0 <= node_int < total_nodes:
            per_node_accepted[node_int] = True

    posterior = {
        "best_candidate": best_candidate,
        "accepted_draft_count": accepted_draft_count,
        "accepted_node_ids": accepted_node_ids,
        "per_node_accepted": per_node_accepted,
        "sample_p_topk": _sample_p_topk(sample_p, sampled_indices),
    }

    committed_ids = _STEP_DATA["committed_tokens_ids"]
    committed_tokens = [{"token_id": int(t), "token_text": _decode_token(int(t))} for t in committed_ids]

    _TRACE.append(
        {
            "step_index": step_index,
            "prompt_len_so_far": _STEP_DATA["prompt_len_so_far"],
            "draft_tree": draft_tree,
            "candidates": _build_candidates(candidates_tensor, retrieve_indices),
            "posterior": posterior,
            "committed_tokens": committed_tokens,
            "eos_stopped": _STEP_DATA["eos_stopped"],
            "timings_ms": {
                "tree_decoding_ms": _STEP_DATA.get("t_tree_ms", 0.0),
                "posterior_ms": _STEP_DATA.get("t_post_ms", 0.0),
                "update_inputs_ms": _STEP_DATA.get("t_update_ms", 0.0),
            },
        }
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mobilint/EAGLE3-Qwen3-4B")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--eagle3-tree-depth", type=int, default=None)
    parser.add_argument("--eagle3-tree-top-k", type=int, default=None)
    parser.add_argument("--num-assistant-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", default="debug/eagle3_tree_trace")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    """Load an EAGLE-3 model, run one ``generate`` call, and dump the tree trace."""
    args = _parse_args()
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
    if args.eagle3_tree_depth is not None:
        config.eagle3_tree_depth = int(args.eagle3_tree_depth)
    if args.eagle3_tree_top_k is not None:
        config.eagle3_tree_top_k = int(args.eagle3_tree_top_k)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=trust_remote_code)

    global _TOKENIZER, _TRACE, _STEP_INDEX
    _TOKENIZER = tokenizer

    if args.num_assistant_tokens is not None:
        model.generation_config.num_assistant_tokens = int(args.num_assistant_tokens)

    if getattr(tokenizer, "chat_template", None):
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

    do_sample = bool(args.do_sample)
    resolved_temperature = (
        args.temperature if args.temperature is not None else getattr(model.generation_config, "temperature", None)
    )
    resolved_top_k = args.top_k if args.top_k is not None else getattr(model.generation_config, "top_k", None)
    if resolved_temperature is not None and float(resolved_temperature) <= 1e-5:
        do_sample = False

    torch.manual_seed(int(args.seed))

    restore = _apply_wrappers()
    _TRACE = []
    _STEP_INDEX = 0
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=do_sample,
                temperature=resolved_temperature,
                top_k=resolved_top_k,
            )
    finally:
        restore()

    if isinstance(output, torch.Tensor):
        generated_ids = output[0].tolist()
    else:
        generated_ids = list(output)
    new_tokens = generated_ids[input_ids.shape[1] :]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)

    render_blocks: list[str] = []
    for step in _TRACE:
        tree_nodes = step["draft_tree"]["nodes"]
        total = step["draft_tree"]["total_tokens"]
        posterior = step["posterior"]
        if posterior is not None:
            accepted_count = int(posterior["accepted_draft_count"])
            best_leaf = int(posterior["best_candidate"])
            per_node_accepted = list(posterior["per_node_accepted"])
            candidates_list = step["candidates"]
            if candidates_list and 0 <= best_leaf < len(candidates_list):
                best_path_set = set(int(nid) for nid in candidates_list[best_leaf]["node_ids"])
            else:
                best_path_set = set(int(nid) for nid in posterior["accepted_node_ids"])
        else:
            accepted_count = -1
            best_leaf = None
            per_node_accepted = [False] * total
            if total > 0:
                per_node_accepted[0] = True
            best_path_set = {0} if total > 0 else set()

        block = _render_tree_ascii(
            step_index=step["step_index"],
            prompt_len=step["prompt_len_so_far"],
            tree_nodes=tree_nodes,
            accepted_count=accepted_count,
            tree_size=total,
            best_leaf=best_leaf,
            per_node_accepted=per_node_accepted,
            best_path_set=best_path_set,
        )
        print(block)
        print()
        render_blocks.append(block)
    render_text = "\n\n----\n\n".join(render_blocks) + "\n"

    real_step_records = [s for s in _TRACE if s["step_index"] >= 0]
    total_committed = sum(len(s["committed_tokens"]) for s in _TRACE)
    total_steps = len(real_step_records)
    mean_accepted_per_step = (
        sum(int(s["posterior"]["accepted_draft_count"]) + 1 for s in real_step_records) / float(total_steps)
        if total_steps > 0
        else 0.0
    )
    eos_reason = "eos" if any(s.get("eos_stopped") for s in _TRACE) else "max_new_tokens_or_stop"
    summary = {
        "generated_tokens": total_committed,
        "speculative_steps": total_steps,
        "mean_accepted_per_step": mean_accepted_per_step,
        "stop_reason": eos_reason,
        "decoded": decoded,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))

    trace_json = {
        "model": args.model,
        "prompt": args.prompt,
        "chat_prompt": chat_prompt,
        "config_overrides": {
            "eagle3_tree_depth": args.eagle3_tree_depth,
            "eagle3_tree_top_k": args.eagle3_tree_top_k,
            "num_assistant_tokens": args.num_assistant_tokens,
        },
        "generation_config_effective": {
            "do_sample": do_sample,
            "temperature": resolved_temperature,
            "top_k": resolved_top_k,
            "max_new_tokens": int(args.max_new_tokens),
            "seed": int(args.seed),
        },
        "summary": summary,
        "steps": _TRACE,
    }
    (output_dir / "trace.json").write_text(
        json.dumps(trace_json, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output_dir / "tree_render.txt").write_text(render_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
