"""Collect comparable Whisper beam-search traces for GPU and Mobilint NPU models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _load_sample(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.dataset_split,
        streaming=True,
        trust_remote_code=True,
    )
    if args.seed is not None:
        dataset = dataset.shuffle(seed=int(args.seed), buffer_size=1000)
    for index, sample in enumerate(dataset):
        if index < int(args.sample_index):
            continue
        audio = sample["audio"]
        return {
            "id": str(sample.get("id", index)),
            "audio": audio,
            "reference": str(sample.get("text", sample.get("reference", ""))),
        }
    raise RuntimeError(f"No sample found at index {args.sample_index}")


def _topk_scores(scores: tuple[torch.Tensor, ...], *, limit_steps: int, top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step, score in enumerate(scores[:limit_steps]):
        top_values, top_indices = torch.topk(score.detach().cpu(), k=min(top_k, int(score.shape[-1])), dim=-1)
        rows.append(
            {
                "step": step,
                "shape": list(score.shape),
                "top_token_ids": top_indices.tolist(),
                "top_scores": top_values.tolist(),
            }
        )
    return rows


def _run_model(
    *,
    label: str,
    model_id: str,
    sample: dict[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if label == "gpu":
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["decoder_core_mode"] = args.core_mode
        model_kwargs["encoder_core_mode"] = args.core_mode
        if args.core_mode == "single":
            model_kwargs["decoder_target_cores"] = ["0:0"]
            model_kwargs["encoder_target_cores"] = ["0:0"]
        elif args.core_mode == "global4":
            model_kwargs["decoder_target_clusters"] = [0]
            model_kwargs["encoder_target_clusters"] = [0]
        elif args.core_mode == "global8":
            model_kwargs["decoder_target_clusters"] = [0, 1]
            model_kwargs["encoder_target_clusters"] = [0, 1]
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, **model_kwargs)
    if label == "gpu":
        model.to(args.gpu_device)
    model.eval()

    audio = sample["audio"]
    inputs = processor(
        audio["array"],
        sampling_rate=int(audio["sampling_rate"]),
        return_tensors="pt",
    )
    if label == "gpu":
        model_dtype = next(model.parameters()).dtype
        inputs = {
            key: (
                value.to(args.gpu_device, dtype=model_dtype)
                if torch.is_floating_point(value)
                else value.to(args.gpu_device)
            )
            for key, value in inputs.items()
        }

    previous_trace = os.environ.get("MBLT_WHISPER_BEAM_DEBUG_TRACE")
    npu_trace_path = out_dir / f"{label}_cache_trace.jsonl"
    if label == "npu":
        if npu_trace_path.exists():
            npu_trace_path.unlink()
        os.environ["MBLT_WHISPER_BEAM_DEBUG_TRACE"] = str(npu_trace_path)

    try:
        with torch.inference_mode():
            generate_kwargs: dict[str, Any] = {
                "language": args.language,
                "task": args.task,
                "num_beams": int(args.num_beams),
                "early_stopping": True,
                "max_new_tokens": int(args.max_new_tokens),
            }
            if args.no_cache:
                generate_kwargs["use_cache"] = False
            if label == "gpu":
                generate_kwargs["return_dict_in_generate"] = True
                generate_kwargs["output_scores"] = True
            output = model.generate(**inputs, **generate_kwargs)
    finally:
        if label == "npu":
            if previous_trace is None:
                os.environ.pop("MBLT_WHISPER_BEAM_DEBUG_TRACE", None)
            else:
                os.environ["MBLT_WHISPER_BEAM_DEBUG_TRACE"] = previous_trace

    sequences = output.sequences.detach().cpu() if hasattr(output, "sequences") else output.detach().cpu()
    decoded = processor.batch_decode(sequences, skip_special_tokens=True)
    scores = tuple(output.scores) if hasattr(output, "scores") and output.scores is not None else ()
    payload = {
        "label": label,
        "model_id": model_id,
        "sample_id": sample["id"],
        "reference": sample["reference"],
        "decoded": decoded,
        "sequences": sequences.tolist(),
        "scores_topk": _topk_scores(scores, limit_steps=int(args.trace_steps), top_k=int(args.top_k)),
        "num_scores": len(scores),
    }
    (out_dir / f"{label}_generate_trace.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-model-id", default="openai/whisper-small")
    parser.add_argument("--npu-model-id", default="mobilint/whisper-small")
    parser.add_argument("--dataset", default="openslr/librispeech_asr")
    parser.add_argument("--dataset-config", default="clean")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--language", default="en")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--trace-steps", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--core-mode", default="single")
    parser.add_argument("--gpu-device", default="cuda:0")
    parser.add_argument("--output-dir", default="debug/whisper_beam_trace")
    parser.add_argument("--backend", choices=["gpu", "npu", "both"], default="both")
    parser.add_argument("--no-cache", action="store_true", help="pass use_cache=False to generate()")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = _load_sample(args)
    (out_dir / "sample.json").write_text(
        json.dumps(sample, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    if args.backend in {"gpu", "both"}:
        _run_model(label="gpu", model_id=args.gpu_model_id, sample=sample, args=args, out_dir=out_dir)
    if args.backend in {"npu", "both"}:
        _run_model(label="npu", model_id=args.npu_model_id, sample=sample, args=args, out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())