"""Update Hugging Face config.json files with npu_prefill_chunk_size values.

This script reads `prefill_chunk_size.csv`, groups rows by `(model_id, revision)`,
and writes an `npu_prefill_chunk_size` dict into each target branch's `config.json`.
It also updates the `main` branch by inferring whether `main` currently points to
the `W8` or `W4V8` mxq variant from `config.json.mxq_path`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Update HF config.json npu_prefill_chunk_size entries.")
    parser.add_argument(
        "--csv",
        default=Path(__file__).with_name("prefill_chunk_size.csv"),
        help="CSV path with columns core_mode,revision,model_id,best_chunk_size",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="optional model_id filter; repeatable",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="push changes to the Hugging Face Hub; default is dry-run",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or get_token(),
        help="HF token; defaults to HF_TOKEN env var or local hf auth login cache",
    )
    return parser.parse_args()


def _coerce_positive_int(value: Any) -> int | None:
    """Convert a CSV value into a positive int when possible.

    Args:
        value: Raw CSV value.

    Returns:
        Parsed positive integer, or `None` when invalid.
    """
    try:
        parsed = int(float(str(value)))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _load_prefill_chunk_sizes(csv_path: Path) -> dict[tuple[str, str], dict[str, int]]:
    """Load per-model per-revision chunk sizes from CSV.

    Args:
        csv_path: Source CSV path.

    Returns:
        Mapping from `(model_id, revision)` to `{core_mode: best_chunk_size}`.
    """
    out: dict[tuple[str, str], dict[str, int]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model_id = str(row.get("model_id", "")).strip()
            revision = str(row.get("revision", "")).strip()
            core_mode = str(row.get("core_mode", "")).strip()
            best_chunk_size = _coerce_positive_int(row.get("best_chunk_size"))
            if not model_id.startswith("mobilint/"):
                continue
            if not model_id or not revision or not core_mode or best_chunk_size is None:
                continue
            out.setdefault((model_id, revision), {})[core_mode] = best_chunk_size
    return out


def _download_config(api: HfApi, model_id: str, revision: str | None, token: str | None) -> tuple[dict[str, Any], str]:
    """Download a config.json payload from the Hub.

    Args:
        api: Hugging Face Hub API client.
        model_id: Target model repository.
        revision: Target branch or tag.

    Returns:
        Tuple of parsed config payload and source file path.
    """
    path = hf_hub_download(repo_id=model_id, filename="config.json", revision=revision, token=token)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle), path


def _infer_main_variant(config: dict[str, Any]) -> str | None:
    """Infer which quantized revision `main` currently points to.

    Args:
        config: Parsed config payload from the `main` branch.

    Returns:
        `W8`, `W4V8`, or `None` when inference is not possible.
    """
    mxq_path = str(config.get("mxq_path", "")).upper()
    if "W4V8" in mxq_path:
        return "W4V8"
    if "W8" in mxq_path:
        return "W8"
    return None


def _write_config_copy(config: dict[str, Any]) -> str:
    """Write a temporary config.json copy.

    Args:
        config: Config payload to serialize.

    Returns:
        Path to the temporary file.
    """
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False)
    with handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return handle.name


def _update_branch(
    api: HfApi,
    *,
    model_id: str,
    revision: str | None,
    prefill_chunk_size: dict[str, int],
    apply_changes: bool,
    token: str | None,
) -> bool:
    """Update a single branch's config.json payload.

    Args:
        api: Hugging Face Hub API client.
        model_id: Target model repository.
        revision: Target branch or tag.
        prefill_chunk_size: Chunk-size mapping to write.
        apply_changes: Whether to push the update.

    Returns:
        `True` when a change is needed, otherwise `False`.
    """
    config, _ = _download_config(api, model_id, revision, token)
    if config.get("npu_prefill_chunk_size") == prefill_chunk_size:
        print(f"[skip] {model_id}@{revision or 'main'} already up to date")
        return False

    config["npu_prefill_chunk_size"] = prefill_chunk_size
    print(f"[plan] {model_id}@{revision or 'main'} -> {prefill_chunk_size}")
    if not apply_changes:
        return True

    temp_path = _write_config_copy(config)
    try:
        api.create_commit(
            repo_id=model_id,
            repo_type="model",
            revision=revision,
            operations=[CommitOperationAdd(path_in_repo="config.json", path_or_fileobj=temp_path)],
            commit_message="chore: update prefill chunk size",
        )
    finally:
        os.unlink(temp_path)
    print(f"[done] {model_id}@{revision or 'main'}")
    return True


def main() -> int:
    """Run the config update workflow.

    Returns:
        Process exit code.
    """
    args = _parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    api = HfApi(token=args.token)
    grouped = _load_prefill_chunk_sizes(csv_path)
    if args.model:
        allowed = set(args.model)
        grouped = {key: value for key, value in grouped.items() if key[0] in allowed}

    per_model: dict[str, dict[str, dict[str, int]]] = {}
    for (model_id, revision), mapping in grouped.items():
        per_model.setdefault(model_id, {})[revision] = mapping

    changed = 0
    for model_id, revision_map in sorted(per_model.items()):
        for revision, mapping in sorted(revision_map.items()):
            if _update_branch(
                api,
                model_id=model_id,
                revision=revision,
                prefill_chunk_size=mapping,
                apply_changes=args.apply,
                token=args.token,
            ):
                changed += 1

        try:
            main_config, _ = _download_config(api, model_id, None, args.token)
        except Exception as exc:
            print(f"[warn] failed to read {model_id}@main: {exc}")
            continue

        main_variant = _infer_main_variant(main_config)
        if main_variant is None:
            print(f"[warn] could not infer main variant for {model_id}; skipping main")
            continue
        main_mapping = revision_map.get(main_variant)
        if main_mapping is None:
            print(f"[warn] no CSV mapping for {model_id}@{main_variant}; skipping main")
            continue
        if _update_branch(
            api,
            model_id=model_id,
            revision=None,
            prefill_chunk_size=main_mapping,
            apply_changes=args.apply,
            token=args.token,
        ):
            changed += 1

    print(f"planned_updates={changed} apply={args.apply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
