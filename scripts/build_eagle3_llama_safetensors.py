"""Build the top-level ``model.safetensors`` for the Mobilint EAGLE3 Llama-3.1-8B release.

Mirrors the layout of the reference cache at::

    ~/.cache/huggingface/hub/models--mobilint--EAGLE3-Qwen3-4B/snapshots/
        27c8eabf826765e29dc03b5171a7be4115394c4f/model.safetensors

which stores exactly four tensors:

    eagle3_base_model.embed_tokens.weight   F32
    eagle3_draft_model.embed_tokens.weight  F16
    eagle3_draft_model.d2t                  I64
    eagle3_draft_model.t2d                  BOOL

The rest of the draft transformer weights (``fc``, ``lm_head``, ``midlayer.*``, ``norm``,
``act_scale_*``, ``qat_step``) are baked into the shipped ``*-Draft.mxq`` at compile time and
must not appear in the release safetensors.
"""

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_SOURCE_DIR = Path(r"C:/Users/Beomsu/mblt-model-zoo/EAGLE3-Llama-3.1-8B-Instruct")


def main() -> int:
    """Assemble the release safetensors from the two ``.pth`` embeddings and draft safetensors."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Release folder containing target_emb.pth, llama_emb.pth, and the draft subdir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination model.safetensors (defaults to <source-dir>/model.safetensors).",
    )
    parser.add_argument(
        "--draft-subdir",
        type=str,
        default="epoch_9_step_735000",
        help="Subdirectory of --source-dir holding the draft model.safetensors.",
    )
    args = parser.parse_args()

    source: Path = args.source_dir
    target_emb_path = source / "target_emb.pth"
    llama_emb_path = source / "llama_emb.pth"
    draft_st_path = source / args.draft_subdir / "model.safetensors"
    output: Path = args.output or (source / "model.safetensors")

    print(f"[load] target_emb.pth from {target_emb_path}")
    target_emb = torch.load(target_emb_path, map_location="cpu", weights_only=True)
    assert target_emb.dtype == torch.float32, f"expected float32, got {target_emb.dtype}"
    assert target_emb.shape == (128256, 4096), f"unexpected shape {tuple(target_emb.shape)}"

    print(f"[load] llama_emb.pth from {llama_emb_path}")
    draft_emb = torch.load(llama_emb_path, map_location="cpu", weights_only=True)
    assert draft_emb.dtype == torch.float32, f"expected float32, got {draft_emb.dtype}"
    assert draft_emb.shape == (128256, 4096), f"unexpected shape {tuple(draft_emb.shape)}"

    print(f"[load] draft d2t / t2d from {draft_st_path}")
    with safe_open(str(draft_st_path), framework="pt") as f:
        d2t = f.get_tensor("d2t")
        t2d = f.get_tensor("t2d")
    assert d2t.dtype == torch.int64 and tuple(d2t.shape) == (32000,), (
        f"unexpected d2t: dtype={d2t.dtype} shape={tuple(d2t.shape)}"
    )
    assert t2d.dtype == torch.bool and tuple(t2d.shape) == (128256,), (
        f"unexpected t2d: dtype={t2d.dtype} shape={tuple(t2d.shape)}"
    )

    print("[cast] draft embed float32 -> float16")
    draft_emb_f16 = draft_emb.to(torch.float16).contiguous()
    target_emb_f32 = target_emb.contiguous()

    tensors = {
        "eagle3_base_model.embed_tokens.weight": target_emb_f32,
        "eagle3_draft_model.embed_tokens.weight": draft_emb_f16,
        "eagle3_draft_model.d2t": d2t.contiguous(),
        "eagle3_draft_model.t2d": t2d.contiguous(),
    }

    print(f"[save] {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output))

    print("[verify] reading back")
    with safe_open(str(output), framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            s = f.get_slice(k)
            print(f"  {k} shape={list(s.get_shape())} dtype={s.get_dtype()}")
    expected = {
        "eagle3_base_model.embed_tokens.weight",
        "eagle3_draft_model.embed_tokens.weight",
        "eagle3_draft_model.d2t",
        "eagle3_draft_model.t2d",
    }
    assert set(keys) == expected, f"unexpected key set: {set(keys)!r}"
    print("[done] 4-key EAGLE3 safetensors written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
