"""Build the top-level ``model.safetensors`` for any Mobilint EAGLE3 release.

Mirrors the four-key layout used by every Mobilint EAGLE3 release (the
``mobilint/EAGLE3-Qwen3-4B`` snapshot is the reference), which stores exactly:

    eagle3_base_model.embed_tokens.weight   F32  (V, H)
    eagle3_draft_model.embed_tokens.weight  F16  (V, H)
    eagle3_draft_model.d2t                  I64  (Vd,)
    eagle3_draft_model.t2d                  BOOL (V,)

where ``V`` is the target-model vocab size, ``H`` is the shared hidden size, and ``Vd`` is the
draft-model vocab size (always ``Vd <= V``).

The rest of the draft transformer weights (``fc``, ``lm_head``, ``midlayer.*``, ``norm``,
``act_scale_*``, ``qat_step``) are baked into the shipped ``*-Draft.mxq`` at compile time and
must not appear in the release safetensors.

Inputs (per release folder):
    - ``target_emb.pth``    : target-model input embedding, F32 ``(V, H)`` (fixed filename).
    - ``<prefix>_emb.pth``  : draft-model input embedding, F32 ``(V, H)`` (auto-detected).
    - ``<draft-subdir>/model.safetensors`` : draft checkpoint holding ``d2t`` and ``t2d``.
"""

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TARGET_EMB_FILENAME = "target_emb.pth"
DEFAULT_DRAFT_SUBDIR = "epoch_9_step_735000"


def _resolve_draft_emb(source: Path, override: Path | None) -> Path:
    """Return the release's draft-embed ``.pth`` path.

    Args:
        source: Release folder to search.
        override: Optional explicit path provided via ``--draft-emb``.

    Returns:
        The resolved path to the draft-embed ``.pth`` file.

    Raises:
        SystemExit: If zero or multiple ``*_emb.pth`` files remain after excluding
            ``target_emb.pth`` and no override is supplied.
    """

    if override is not None:
        if not override.is_file():
            raise SystemExit(f"[error] --draft-emb {override} is not a file")
        return override

    candidates = sorted(p for p in source.glob("*_emb.pth") if p.name != TARGET_EMB_FILENAME)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(
            f"[error] no draft embed found in {source}: expected exactly one <prefix>_emb.pth "
            f"besides {TARGET_EMB_FILENAME}. Pass --draft-emb <path> to disambiguate."
        )
    listed = ", ".join(p.name for p in candidates)
    raise SystemExit(
        f"[error] multiple draft-embed candidates in {source}: {listed}. Pass --draft-emb <path> to disambiguate."
    )


def _resolve_draft_subdir(source: Path, requested: str | None) -> Path:
    """Return the release's draft-checkpoint subdirectory.

    Auto-detection only fires when the caller did not supply ``--draft-subdir``. An explicit
    request that resolves to a missing ``model.safetensors`` fails fast so a typo cannot silently
    pair the wrong draft checkpoint's ``d2t``/``t2d`` with the embeddings.

    Args:
        source: Release folder to search.
        requested: The ``--draft-subdir`` value when the caller passed one, or ``None`` when the
            argparse default was used. Explicit non-``None`` values are required to exist.

    Returns:
        The resolved subdirectory containing ``model.safetensors``.

    Raises:
        SystemExit: If ``requested`` was supplied explicitly but its ``model.safetensors`` is
            missing, or (when defaulted) if neither ``DEFAULT_DRAFT_SUBDIR`` nor a single
            auto-detected candidate is available.
    """

    if requested is not None:
        explicit = source / requested
        if (explicit / "model.safetensors").is_file():
            return explicit
        raise SystemExit(
            f"[error] --draft-subdir '{requested}': {explicit / 'model.safetensors'} not found. "
            f"Auto-detection is skipped because you explicitly requested this subdir; "
            f"pass a different --draft-subdir or omit the flag to auto-detect."
        )

    default_dir = source / DEFAULT_DRAFT_SUBDIR
    if (default_dir / "model.safetensors").is_file():
        return default_dir

    candidates = sorted(p for p in source.iterdir() if p.is_dir() and (p / "model.safetensors").is_file())
    if len(candidates) == 1:
        chosen = candidates[0]
        print(f"[detect] default draft subdir '{DEFAULT_DRAFT_SUBDIR}' missing; using '{chosen.name}'")
        return chosen
    if not candidates:
        raise SystemExit(
            f"[error] no draft subdirectory with model.safetensors found under {source} "
            f"(default '{DEFAULT_DRAFT_SUBDIR}' also absent)."
        )
    listed = ", ".join(p.name for p in candidates)
    raise SystemExit(
        f"[error] multiple draft-subdir candidates under {source}: {listed}. "
        f"Pass --draft-subdir <name> to disambiguate."
    )


def main() -> int:
    """Assemble the release safetensors from the two ``.pth`` embeddings and draft safetensors."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Release folder containing target_emb.pth, <prefix>_emb.pth, and the draft subdir.",
    )
    parser.add_argument(
        "--source-dir",
        dest="source_dir_flag",
        type=Path,
        default=None,
        help="Alternate flag spelling for the release folder (mutually exclusive with positional).",
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
        default=None,
        help=(
            f"Subdirectory of --source-dir holding the draft model.safetensors. "
            f"When omitted, '{DEFAULT_DRAFT_SUBDIR}' is used if present and otherwise "
            f"auto-detected from the single available candidate. When supplied explicitly, "
            f"the path must exist (no auto-detect fallback)."
        ),
    )
    parser.add_argument(
        "--draft-emb",
        type=Path,
        default=None,
        help="Explicit draft-embed .pth path (overrides auto-detection of <prefix>_emb.pth).",
    )
    args = parser.parse_args()

    if args.source_dir is not None and args.source_dir_flag is not None:
        parser.error("pass the release folder either positionally or via --source-dir, not both")
    source: Path | None = args.source_dir or args.source_dir_flag
    if source is None:
        parser.error("release folder is required (positional argument or --source-dir)")
    if not source.is_dir():
        raise SystemExit(f"[error] --source-dir {source} is not a directory")

    target_emb_path = source / TARGET_EMB_FILENAME
    draft_emb_path = _resolve_draft_emb(source, args.draft_emb)
    draft_subdir = _resolve_draft_subdir(source, args.draft_subdir)
    draft_st_path = draft_subdir / "model.safetensors"
    output: Path = args.output or (source / "model.safetensors")

    print(f"[load] target_emb.pth from {target_emb_path}")
    target_emb = torch.load(target_emb_path, map_location="cpu", weights_only=True)
    assert isinstance(target_emb, torch.Tensor), f"target_emb.pth is not a tensor (got {type(target_emb)!r})"
    assert target_emb.dim() == 2, f"target_emb must be rank-2 (V, H); got shape {tuple(target_emb.shape)}"
    assert target_emb.dtype == torch.float32, f"target_emb expected float32, got {target_emb.dtype}"

    print(f"[load] draft embed from {draft_emb_path}")
    draft_emb = torch.load(draft_emb_path, map_location="cpu", weights_only=True)
    assert isinstance(draft_emb, torch.Tensor), f"{draft_emb_path.name} is not a tensor (got {type(draft_emb)!r})"
    assert draft_emb.dim() == 2, f"draft embed must be rank-2 (V, H); got shape {tuple(draft_emb.shape)}"
    assert draft_emb.dtype == torch.float32, f"draft embed expected float32, got {draft_emb.dtype}"
    # Contract: Mobilint EAGLE-3 releases train base and draft at a matched hidden size,
    # so packaged embeddings must be shape-identical. The runtime FCProjector branch in
    # MobilintEagle3DraftModelMixin is legacy/future scaffolding and is NOT grounds to relax
    # this equality — see the EAGLE-3 contract bullet in AGENTS.md.
    assert draft_emb.shape == target_emb.shape, (
        f"draft embed shape {tuple(draft_emb.shape)} != target embed shape {tuple(target_emb.shape)}; "
        f"both must share the target-model tokenizer vocab and hidden size."
    )

    vocab_size, hidden_size = int(target_emb.shape[0]), int(target_emb.shape[1])

    print(f"[load] draft d2t / t2d from {draft_st_path}")
    with safe_open(str(draft_st_path), framework="pt") as f:
        d2t = f.get_tensor("d2t")
        t2d = f.get_tensor("t2d")
    assert d2t.dim() == 1, f"d2t must be rank-1; got shape {tuple(d2t.shape)}"
    assert d2t.dtype == torch.int64, f"d2t expected int64, got {d2t.dtype}"
    assert t2d.dim() == 1, f"t2d must be rank-1; got shape {tuple(t2d.shape)}"
    assert t2d.dtype == torch.bool, f"t2d expected bool, got {t2d.dtype}"
    assert t2d.shape[0] == vocab_size, (
        f"t2d length {t2d.shape[0]} != target vocab {vocab_size}; t2d must be a per-target-token mask."
    )
    draft_vocab = int(d2t.shape[0])
    assert draft_vocab <= vocab_size, (
        f"draft vocab (d2t length) {draft_vocab} exceeds target vocab {vocab_size}; "
        f"the draft vocab must be a subset of the target vocab."
    )

    # Every draft-vocab index ``i`` is translated to target-vocab index ``i + d2t[i]`` at
    # runtime (see ``MobilintEagle3DraftModelMixin.topk_generate``); an out-of-range offset
    # only surfaces as an IndexError when that candidate is actually sampled. Fail here so
    # a malformed draft checkpoint never reaches disk as a "built" release.
    translated = torch.arange(draft_vocab, dtype=torch.int64) + d2t
    invalid_mask = (translated < 0) | (translated >= vocab_size)
    n_invalid = int(invalid_mask.sum().item())
    if n_invalid:
        first_i = int(invalid_mask.nonzero(as_tuple=False)[0, 0].item())
        raise SystemExit(
            f"[error] d2t maps {n_invalid} draft-vocab indices outside target vocab "
            f"[0, {vocab_size}). first violation: i={first_i}, "
            f"d2t[{first_i}]={int(d2t[first_i].item())}, "
            f"i + d2t[i]={int(translated[first_i].item())}, vocab_size={vocab_size}. "
            f"the draft checkpoint's d2t is inconsistent with the target embeddings; "
            f"regenerate or replace the draft artifact."
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

    release_name = source.resolve().name
    print(
        f"[done] EAGLE3 release {release_name}: V={vocab_size}, H={hidden_size}, Vd={draft_vocab}. "
        f"4-key safetensors written to {output}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
