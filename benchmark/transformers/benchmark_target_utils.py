"""Shared benchmark target resolution helpers for Transformers benchmark scripts.

This module centralizes benchmark-target discovery logic shared by the text-generation,
image-text-to-text, and automatic-speech-recognition benchmark entry points.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Callable

from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)


def normalize_repo_id(value: str) -> str:
    """Normalize a Hugging Face repository identifier or URL.

    Args:
        value: Raw repository identifier or URL.

    Returns:
        A normalized ``owner/name`` style repository identifier.
    """
    text = value.strip()
    if text.startswith("https://huggingface.co/"):
        text = text[len("https://huggingface.co/") :]
    return text.strip("/")


def extract_parent_model_id(info: Any) -> str | None:
    """Extract a best-effort parent/base model id from Hub model metadata.

    Args:
        info: Hugging Face Hub model info object.

    Returns:
        The normalized parent model id when present, otherwise ``None``.
    """
    card_data = getattr(info, "cardData", None)
    if card_data is None:
        card_data = getattr(info, "card_data", None)

    payload: dict[str, Any] | None = None
    if isinstance(card_data, dict):
        payload = card_data
    elif card_data is not None and hasattr(card_data, "to_dict"):
        try:
            payload = card_data.to_dict()
        except (AttributeError, TypeError, ValueError):
            payload = None
    elif card_data is not None and hasattr(card_data, "__dict__"):
        payload = dict(card_data.__dict__)

    if not payload:
        return None

    def _pick_candidate(raw: Any) -> str | None:
        if isinstance(raw, str):
            candidate = normalize_repo_id(raw)
            return candidate if "/" in candidate else None
        if isinstance(raw, dict):
            for key in ("model_id", "repo_id", "id", "name"):
                value = raw.get(key)
                if isinstance(value, str):
                    candidate = normalize_repo_id(value)
                    if "/" in candidate:
                        return candidate
            return None
        if isinstance(raw, list):
            for item in raw:
                picked = _pick_candidate(item)
                if picked:
                    return picked
            return None
        return None

    for key in ("base_model", "base_models", "baseModel", "parent_model"):
        candidate = _pick_candidate(payload.get(key))
        if candidate:
            return candidate
    return None


def resolve_original_model_ids(model_ids: Iterable[str]) -> list[str]:
    """Resolve Mobilint model ids to parent/original Hugging Face model ids.

    Args:
        model_ids: Candidate model ids from benchmark discovery.

    Returns:
        A de-duplicated list of parent/original model ids.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
    except (ImportError, OSError) as exc:
        print(
            "Failed to initialize Hugging Face Hub API for --original-models. "
            f"Using original list_models output. Error: {exc}"
        )
        return list(model_ids)

    resolved: list[str] = []
    seen: set[str] = set()
    for model_id in model_ids:
        target_id = model_id
        try:
            info = api.model_info(model_id)
            parent_id = extract_parent_model_id(info)
            if parent_id:
                target_id = parent_id
        except (
            EntryNotFoundError,
            HfHubHTTPError,
            LocalEntryNotFoundError,
            OSError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
            ValueError,
        ) as exc:
            print(f"Warning: failed to resolve parent model for {model_id}: {exc}")

        if target_id not in seen:
            resolved.append(target_id)
            seen.add(target_id)
    return resolved


def revision_exists(model_id: str, revision: str) -> bool | None:
    """Check whether a Hugging Face model revision exists.

    Args:
        model_id: Hugging Face model id.
        revision: Revision or branch name to check.

    Returns:
        ``True`` when the revision exists, ``False`` when it does not, and ``None`` when the
        check cannot be completed.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        refs = api.list_repo_refs(model_id, repo_type="model")
        return any(branch.name == revision for branch in getattr(refs, "branches", []))
    except (
        EntryNotFoundError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        OSError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
        ValueError,
    ):
        return None


def select_revision(model_id: str, candidates: list[str | None]) -> str | None:
    """Select the first usable model revision from a candidate list.

    Args:
        model_id: Hugging Face model id.
        candidates: Ordered revision candidates.

    Returns:
        The selected revision, or ``None`` when the default revision should be used.
    """
    for candidate in candidates:
        if not candidate:
            return candidate
        exists = revision_exists(model_id, candidate)
        if exists is None:
            print(f"Warning: failed to verify revision '{candidate}' for {model_id}; trying it anyway.")
            return candidate
        if exists is True:
            return candidate
    return None


def resolve_model_id_from_mxq_name(model_part: str, available_model_ids: Sequence[str]) -> str | None:
    """Resolve a model id from an MXQ filename stem.

    Args:
        model_part: Filename model portion before the revision suffix.
        available_model_ids: Known benchmark model ids.

    Returns:
        The resolved model id, or ``None`` when no unique match is found.
    """
    if model_part in available_model_ids:
        return model_part
    model_part_slash = model_part.replace("__", "/")
    if model_part_slash in available_model_ids:
        return model_part_slash

    basename_matches = [model_id for model_id in available_model_ids if model_id.split("/", 1)[-1] == model_part]
    if len(basename_matches) == 1:
        return basename_matches[0]
    basename_matches_slash = [
        model_id for model_id in available_model_ids if model_id.split("/", 1)[-1] == model_part_slash
    ]
    if len(basename_matches_slash) == 1:
        return basename_matches_slash[0]
    return None


def iter_revision_targets(
    model_ids: Iterable[str],
    *,
    revision: str | None,
    all_revisions: bool,
    safe_filename: Callable[[str], str],
) -> Iterable[tuple[str, list[str | None], str, str, str | None]]:
    """Iterate benchmark targets expanded by requested revision policy.

    Args:
        model_ids: Model ids to expand.
        revision: Explicit revision to use when ``all_revisions`` is disabled.
        all_revisions: Whether to expand into the benchmark's supported quantized revisions.
        safe_filename: Filename normalizer used for output base names.

    Yields:
        Tuples of ``(model_id, revision_candidates, label, base, mxq_path)``.
    """
    if not all_revisions:
        for model_id in model_ids:
            yield model_id, [revision], model_id, safe_filename(model_id), None
        return

    revision_map: list[tuple[list[str | None], str]] = [(["W8"], "-W8"), (["W4V8"], "-W4V8")]
    for model_id in model_ids:
        for revision_candidates, suffix in revision_map:
            yield (
                model_id,
                revision_candidates,
                f"{model_id}{suffix}",
                f"{safe_filename(model_id)}{suffix}",
                None,
            )


def iter_targets_from_mxq_dir(
    *,
    mxq_dir: Path,
    available_model_ids: Sequence[str],
    safe_filename: Callable[[str], str],
) -> list[tuple[str, list[str | None], str, str, str | None]]:
    """Resolve benchmark targets from local MXQ filenames.

    Args:
        mxq_dir: Directory containing MXQ files.
        available_model_ids: Known benchmark model ids.
        safe_filename: Filename normalizer used for output base names.

    Returns:
        A list of ``(model_id, revision_candidates, label, base, mxq_path)`` tuples.
    """
    out: list[tuple[str, list[str | None], str, str, str | None]] = []
    seen_bases: set[str] = set()
    for path in sorted(mxq_dir.glob("*.mxq")):
        stem = path.stem
        if "-" not in stem:
            print(f"Skipping mxq (name format mismatch): {path.name}")
            continue
        model_part, rev_part = stem.rsplit("-", 1)
        revision = rev_part.upper()
        if revision not in ("W8", "W4V8"):
            print(f"Skipping mxq (unsupported revision suffix): {path.name}")
            continue
        resolved_model_id = resolve_model_id_from_mxq_name(model_part, available_model_ids)
        if not resolved_model_id:
            print(
                f"Skipping mxq (cannot resolve model_id from filename): {path.name} "
                "(expected <model_id>-<W8|W4V8>.mxq)"
            )
            continue
        label = f"{resolved_model_id}-{revision}"
        base = f"{safe_filename(resolved_model_id)}-{revision}"
        if base in seen_bases:
            print(f"Skipping mxq (duplicate target key): {path.name}")
            continue
        seen_bases.add(base)
        out.append((resolved_model_id, [revision], label, base, str(path)))
    return out


def args_for_target_device_backend(
    args: argparse.Namespace,
    *,
    model_id: str,
    mxq_path: str | None,
    resolve_default_device: Callable[..., str | None] | None = None,
    resolve_default_device_backend: Callable[..., str | None],
) -> argparse.Namespace:
    """Return an args copy with runtime policy resolved for one target.

    Args:
        args: Parsed benchmark arguments.
        model_id: Target model id.
        mxq_path: Optional MXQ path for the specific target.
        resolve_default_device: Optional shared device-policy resolver.
        resolve_default_device_backend: Shared backend-policy resolver.

    Returns:
        A shallow copy of ``args`` with per-target ``device`` and ``device_backend`` values.
    """
    resolved = copy.copy(args)
    if resolve_default_device is not None:
        requested_device = getattr(args, "_device_requested", args.device)
        resolved.device = resolve_default_device(
            device=requested_device,
            device_explicit=bool(getattr(args, "_device_explicit", False)),
            model_id=model_id,
            mxq_path=mxq_path,
            mxq_dir=args.mxq_dir,
            original_models=args.original_models,
        )
    requested_backend = getattr(args, "_device_backend_requested", args.device_backend)
    resolved.device_backend = resolve_default_device_backend(
        device_backend=requested_backend,
        device_backend_explicit=bool(getattr(args, "_device_backend_explicit", False)),
        model_id=model_id,
        mxq_path=mxq_path,
        mxq_dir=args.mxq_dir,
        original_models=args.original_models,
    )
    return resolved