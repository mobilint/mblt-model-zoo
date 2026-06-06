import argparse
import csv
import io
import itertools
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# ruff: noqa: E402
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chart_utils import plot_scalar_chart
from tqdm import tqdm

from benchmark.common.io_utils import safe_filename as _safe_filename_common
from benchmark.common.runtime_utils import clear_cuda_memory as _clear_cuda_memory
from benchmark.common.runtime_utils import is_cuda_device as _is_cuda_device
from benchmark.common.runtime_utils import is_cuda_oom_error as _is_cuda_oom_error
from benchmark.common.runtime_utils import release_pipeline as _release_pipeline
from benchmark.common.summary_utils import HOST_PC_INFO_FILENAME as _HOST_PC_INFO_FILENAME
from benchmark.common.summary_utils import collect_host_pc_info as _collect_host_pc_info
from benchmark.common.summary_utils import existing_png_paths as _existing_png_paths
from benchmark.common.summary_utils import markdown_table as _markdown_table_common
from benchmark.common.summary_utils import write_summary_markdown as _write_summary_markdown
from benchmark.transformers.asr_metrics import (
    ASRMetricSummary,
    SampleTiming,
    format_metrics_row,
    normalize_transcript,
    summarize_timings,
    summary_to_dict,
)
from mblt_model_zoo.hf_transformers.utils import list_models
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import CORE_MODE_CHOICES as _CORE_MODE_CHOICES_COMMON
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    add_device_tracking_args as _add_device_tracking_args,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    add_pipeline_device_args as _add_pipeline_device_args,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    append_core_mode_suffix as _append_core_mode_suffix_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    apply_core_mode_model_kwargs as _apply_core_mode_model_kwargs_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    build_device_tracker as _build_device_tracker_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    extract_device_metric as _extract_device_metric_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    extract_device_time_series as _extract_device_time_series_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import iter_core_modes as _iter_core_modes_common
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import print_device_status as _print_device_status_common
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    resolve_default_device as _resolve_default_device_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    resolve_default_device_backend as _resolve_default_device_backend_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import stop_tracker_safe as _stop_tracker_safe_common


@dataclass(frozen=True)
class ASRBenchmarkTarget:
    """Resolved ASR benchmark target with revision and optional MXQ metadata."""

    model_id: str
    revision_candidates: list[str | None]
    label: str
    base: str
    mxq_path: str | None
    is_original: bool


def _safe_filename(model_id: str) -> str:
    return _safe_filename_common(model_id, replace_slash_only=True)


def _flag_present(raw_argv: Sequence[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in raw_argv)


def _format_exception(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}: {message}"
    return f"{type(exc).__name__}: {exc!r}"


def _print_exception(message: str, exc: BaseException, *, debug_errors: bool) -> None:
    print(f"{message}: {_format_exception(exc)}")
    if debug_errors:
        traceback.print_exception(type(exc), exc, exc.__traceback__)


def _normalize_repo_id(value: str) -> str:
    text = value.strip()
    if text.startswith("https://huggingface.co/"):
        text = text[len("https://huggingface.co/") :]
    return text.strip("/")


def _extract_parent_model_id(info: Any) -> str | None:
    card_data = getattr(info, "cardData", None)
    if card_data is None:
        card_data = getattr(info, "card_data", None)

    payload: dict[str, Any] | None = None
    if isinstance(card_data, dict):
        payload = card_data
    elif card_data is not None and hasattr(card_data, "to_dict"):
        try:
            payload = card_data.to_dict()
        except Exception:
            payload = None
    elif card_data is not None and hasattr(card_data, "__dict__"):
        payload = dict(card_data.__dict__)

    if not payload:
        return None

    def _pick_candidate(raw: Any) -> str | None:
        if isinstance(raw, str):
            candidate = _normalize_repo_id(raw)
            return candidate if "/" in candidate else None
        if isinstance(raw, dict):
            for key in ("model_id", "repo_id", "id", "name"):
                value = raw.get(key)
                if isinstance(value, str):
                    candidate = _normalize_repo_id(value)
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


def _resolve_original_model_ids(model_ids: Iterable[str]) -> list[str]:
    try:
        from huggingface_hub import HfApi

        api = HfApi()
    except Exception as exc:
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
            parent_id = _extract_parent_model_id(info)
            if parent_id:
                target_id = parent_id
        except Exception as exc:
            print(f"Warning: failed to resolve parent model for {model_id}: {exc}")

        if target_id not in seen:
            resolved.append(target_id)
            seen.add(target_id)
    return resolved


def _revision_exists(model_id: str, revision: str) -> bool | None:
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        refs = api.list_repo_refs(model_id, repo_type="model")
        return any(branch.name == revision for branch in getattr(refs, "branches", []))
    except Exception:
        return None


def _select_revision(model_id: str, candidates: list[str | None]) -> str | None:
    for candidate in candidates:
        if not candidate:
            return candidate
        exists = _revision_exists(model_id, candidate)
        if exists is True or exists is None:
            return candidate
    return None


def _list_default_asr_models() -> list[str]:
    available = list_models(tasks="automatic-speech-recognition")
    return [
        str(model_id)
        for model_id in available.get("automatic-speech-recognition", [])
        if not _is_excluded_asr_model_id(str(model_id))
    ]


def _is_excluded_asr_model_id(model_id: str) -> bool:
    """Return whether an ASR model id should be skipped for Transformers pipeline benchmarks."""
    normalized = model_id.lower()
    return normalized.endswith("/whisper.cpp") or normalized.endswith("\\whisper.cpp") or "whisper.cpp" in normalized


def _is_whisper_like_model(model_id: str) -> bool:
    return "whisper" in model_id.lower()


def _is_qwen3_asr_model(model_id: str) -> bool:
    normalized = model_id.lower()
    return "qwen3-asr" in normalized or "qwen3_asr" in normalized


def _ensure_qwen3_asr_backend_registered() -> None:
    """Import upstream Qwen3-ASR Transformers backend so Auto classes recognize it.

    The original upstream checkpoint uses ``model_type='qwen3_asr'``. In some
    environments, simply calling ``transformers.pipeline(...)`` is not enough to
    make that architecture discoverable unless the optional ``qwen_asr`` package
    has already imported and registered its Transformers backend.
    """
    try:
        import qwen_asr.core.transformers_backend  # noqa: F401
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == "qwen_asr" or missing.startswith("qwen_asr."):
            raise ModuleNotFoundError(
                "Qwen3-ASR original-model benchmarks require the optional 'qwen-asr' package. "
                "Install it with: pip install -U qwen-asr"
            ) from exc
        raise

    try:
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
        from transformers.models.auto.modeling_auto import AutoModelForSpeechSeq2Seq

        Qwen3ASRForConditionalGeneration.main_input_name = "input_features"
        AutoModelForSpeechSeq2Seq.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, exist_ok=True)

        try:
            from transformers.pipelines.automatic_speech_recognition import MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES

            MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES.setdefault("qwen3_asr", "Qwen3ASRForConditionalGeneration")
        except Exception:
            pass
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == "qwen_asr" or missing.startswith("qwen_asr."):
            raise ModuleNotFoundError(
                "Qwen3-ASR original-model benchmarks require the optional 'qwen-asr' package. "
                "Install it with: pip install -U qwen-asr"
            ) from exc
        raise


def _apply_asr_core_mode_model_kwargs(
    model_kwargs: dict[str, Any],
    model_id: str,
    core_mode: str | None,
) -> dict[str, Any]:
    """Apply core-mode kwargs for ASR models, expanding composite encoder/decoder configs when needed."""
    if not _is_qwen3_asr_model(model_id):
        return _apply_core_mode_model_kwargs_common(model_kwargs, core_mode)

    expanded: dict[str, Any] = {}
    _apply_core_mode_model_kwargs_common(expanded, core_mode)
    for prefix in ("encoder", "decoder"):
        for key, value in expanded.items():
            model_kwargs[f"{prefix}_{key}"] = value
    return model_kwargs


def _optional_generate_kwargs_for_model(args: argparse.Namespace, model_id: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if args.task and _is_whisper_like_model(model_id):
        kwargs["task"] = args.task
    if args.language and _is_whisper_like_model(model_id):
        kwargs["language"] = args.language
    return kwargs


def _asr_pipeline_call_kwargs(generate_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {"generate_kwargs": dict(generate_kwargs)} if generate_kwargs else {}


def _asr_pipeline_inputs(sample: Mapping[str, Any]) -> list[tuple[Any, dict[str, Any]]]:
    audio = sample["audio"]
    audio_array = audio["array"]
    sampling_rate = int(audio["sampling_rate"])
    return [({"raw": audio_array, "sampling_rate": sampling_rate}, {})]


def _retryable_generate_kwargs(generate_kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
    current = dict(generate_kwargs)
    attempts = [dict(current)]
    for key in ("task", "language", "return_timestamps", "early_stopping"):
        if key in current:
            current = dict(current)
            current.pop(key, None)
            attempts.append(dict(current))
    return attempts


def _extract_hypothesis_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        text = output.get("text")
        if isinstance(text, str):
            return text
        chunks = output.get("chunks")
        if isinstance(chunks, list):
            parts: list[str] = []
            for chunk in chunks:
                if isinstance(chunk, dict):
                    chunk_text = chunk.get("text")
                    if isinstance(chunk_text, str) and chunk_text.strip():
                        parts.append(chunk_text.strip())
            if parts:
                return " ".join(parts)
    return str(output)


def _resolve_model_id_from_mxq_name(model_part: str, available_model_ids: Sequence[str]) -> str | None:
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


def _iter_asr_targets(
    model_ids: Iterable[str],
    *,
    revision: str | None,
    all_revisions: bool,
    is_original: bool,
) -> Iterable[ASRBenchmarkTarget]:
    if not all_revisions:
        for model_id in model_ids:
            yield ASRBenchmarkTarget(
                model_id=model_id,
                revision_candidates=[revision],
                label=model_id,
                base=_safe_filename(model_id),
                mxq_path=None,
                is_original=is_original,
            )
        return

    revision_map: list[tuple[list[str | None], str]] = [(["W8"], "-W8"), (["W4V8"], "-W4V8")]
    for model_id in model_ids:
        for revisions, suffix in revision_map:
            yield ASRBenchmarkTarget(
                model_id=model_id,
                revision_candidates=revisions,
                label=f"{model_id}{suffix}",
                base=f"{_safe_filename(model_id)}{suffix}",
                mxq_path=None,
                is_original=is_original,
            )


def _iter_asr_targets_from_mxq_dir(mxq_dir: Path, available_model_ids: Sequence[str]) -> list[ASRBenchmarkTarget]:
    targets: list[ASRBenchmarkTarget] = []
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
        resolved_model_id = _resolve_model_id_from_mxq_name(model_part, available_model_ids)
        if not resolved_model_id:
            print(f"Skipping mxq (cannot resolve model_id from filename): {path.name}")
            continue
        base = f"{_safe_filename(resolved_model_id)}-{revision}"
        if base in seen_bases:
            print(f"Skipping mxq (duplicate target key): {path.name}")
            continue
        seen_bases.add(base)
        targets.append(
            ASRBenchmarkTarget(
                model_id=resolved_model_id,
                revision_candidates=[revision],
                label=f"{resolved_model_id}-{revision}",
                base=base,
                mxq_path=str(path.resolve()),
                is_original=False,
            )
        )
    return targets


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Hugging Face Transformers automatic-speech-recognition pipeline-compatible models."
        )
    )
    _add_pipeline_device_args(parser, device_default=None, trust_remote_code_default=True)
    parser.add_argument("--model-id", dest="model_ids", nargs="*", default=None, help="model id list to benchmark")
    parser.add_argument("--revision", default=None, help="model revision (e.g. W8)")
    parser.add_argument("--all-revisions", action="store_true", help="benchmark W8 and W4V8 revisions only")
    parser.add_argument("--mxq-dir", default=None, help="directory containing local mxq files")
    parser.add_argument("--mxq-path", default=None, help="override mxq_path for pipeline loading")
    parser.add_argument(
        "--original-models",
        action="store_true",
        help="resolve Mobilint ids to parent/original model ids",
    )
    parser.add_argument("--dataset", default="openslr/librispeech_asr", help="HF dataset name")
    parser.add_argument("--dataset-config", default="clean", help="HF dataset config name")
    parser.add_argument("--dataset-split", default="test", help="HF dataset split")
    parser.add_argument(
        "--language",
        default="en",
        help="language hint for Whisper-like ASR models; ignored for unsupported pipelines",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="decoding task hint for Whisper-like ASR models; ignored for unsupported pipelines",
    )
    parser.add_argument("--num-samples", type=int, default=50, help="number of evaluation samples")
    parser.add_argument("--num-beams", type=int, default=1, help="single beam value to benchmark")
    parser.add_argument("--max-new-tokens", type=int, default=444, help="maximum generated token count")
    parser.add_argument("--warmup", type=int, default=2, help="number of warmup samples")
    parser.add_argument("--seed", type=int, default=0, help="dataset shuffle seed")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "results" / "automatic_speech_recognition"),
        help="results directory",
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="include per-sample rows in per-target JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve targets and load dataset without model inference",
    )
    parser.add_argument("--debug-errors", action="store_true", help="print full tracebacks for failures")
    parser.add_argument(
        "--core-mode",
        choices=[*list(_CORE_MODE_CHOICES_COMMON), "all"],
        default=None,
        help="core mode passed to model_kwargs; all expands to single/global4/global8",
    )
    _add_device_tracking_args(parser)
    return parser.parse_args(argv)


def _resolve_runtime_defaults(args: argparse.Namespace, raw_argv: Sequence[str]) -> None:
    device_explicit = _flag_present(raw_argv, "--device")
    device_backend_explicit = _flag_present(raw_argv, "--device-backend")
    args._device_backend_explicit = device_backend_explicit
    args._device_backend_requested = args.device_backend
    first_model_id = None if args.mxq_dir else ((args.model_ids or [None])[0])
    args.device = _resolve_default_device_common(
        device=args.device,
        device_explicit=device_explicit,
        model_id=first_model_id,
        mxq_path=args.mxq_path,
        mxq_dir=args.mxq_dir,
        original_models=args.original_models,
    )
    args.device_backend = _resolve_default_device_backend_common(
        device_backend=args.device_backend,
        device_backend_explicit=device_backend_explicit,
        model_id=first_model_id,
        mxq_path=args.mxq_path,
        mxq_dir=args.mxq_dir,
        original_models=args.original_models,
    )
    if not device_explicit:
        print(f"Auto-set --device={args.device}")
    if not device_backend_explicit:
        if first_model_id or args.mxq_path or args.mxq_dir:
            print(f"Auto-set --device-backend={args.device_backend} (based on target/device policy)")
        else:
            print("Auto-set --device-backend per target (based on target/device policy)")


def _args_for_target_device_backend(
    args: argparse.Namespace,
    *,
    model_id: str,
    mxq_path: str | None = None,
) -> argparse.Namespace:
    resolved = argparse.Namespace(**vars(args))
    requested_backend = getattr(args, "_device_backend_requested", args.device_backend)
    resolved.device_backend = _resolve_default_device_backend_common(
        device_backend=requested_backend,
        device_backend_explicit=bool(getattr(args, "_device_backend_explicit", False)),
        model_id=model_id,
        mxq_path=mxq_path,
        mxq_dir=args.mxq_dir,
        original_models=args.original_models,
    )
    return resolved


def _build_asr_pipeline(
    target: ASRBenchmarkTarget,
    *,
    revision: str | None,
    device: str | None,
    device_map: str | None,
    dtype: str | None,
    trust_remote_code: bool,
    core_mode: str | None,
):
    if target.is_original and _is_qwen3_asr_model(target.model_id):
        try:
            import qwen_asr
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if missing == "qwen_asr" or missing.startswith("qwen_asr."):
                raise ModuleNotFoundError(
                    "Qwen3-ASR original-model benchmarks require the optional 'qwen-asr' package. "
                    "Install it with: pip install -U qwen-asr"
                ) from exc
            raise

        native_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "max_inference_batch_size": 1,
            "max_new_tokens": 512,
        }
        if revision:
            native_kwargs["revision"] = revision
        return qwen_asr.Qwen3ASRModel.from_pretrained(target.model_id, **native_kwargs)

    from transformers import pipeline as hf_pipeline

    if _is_qwen3_asr_model(target.model_id):
        _ensure_qwen3_asr_backend_registered()

    kwargs: dict[str, Any] = {
        "task": "automatic-speech-recognition",
        "model": target.model_id,
        "trust_remote_code": trust_remote_code,
    }
    if revision:
        kwargs["revision"] = revision
    if device is not None:
        kwargs["device"] = device
    if device_map:
        kwargs["device_map"] = device_map
    model_kwargs: dict[str, Any] = {}
    model_kwargs = _apply_asr_core_mode_model_kwargs(model_kwargs, target.model_id, core_mode)
    if target.mxq_path:
        model_kwargs["mxq_path"] = target.mxq_path
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    if dtype:
        kwargs["dtype"] = dtype
        try:
            return hf_pipeline(**kwargs)
        except TypeError:
            kwargs.pop("dtype", None)
            kwargs["torch_dtype"] = dtype
            return hf_pipeline(**kwargs)
    return hf_pipeline(**kwargs)


def _load_librispeech(args: argparse.Namespace) -> list[dict[str, Any]]:
    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    def _decode_audio(raw_audio: Mapping[str, Any]) -> tuple[Any, int]:
        path = raw_audio.get("path")
        audio_bytes = raw_audio.get("bytes")
        if isinstance(audio_bytes, (bytes, bytearray)):
            audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        elif isinstance(path, str) and path:
            audio_array, sampling_rate = sf.read(path, dtype="float32")
        else:
            raise ValueError("Dataset audio row does not contain readable path or bytes payload.")

        if getattr(audio_array, "ndim", 1) > 1:
            audio_array = np.mean(audio_array, axis=1)

        sampling_rate = int(sampling_rate)
        if sampling_rate != 16000:
            duration_s = float(len(audio_array)) / float(sampling_rate)
            target_length = max(int(round(duration_s * 16000.0)), 1)
            source_positions = np.linspace(0.0, duration_s, num=len(audio_array), endpoint=False)
            target_positions = np.linspace(0.0, duration_s, num=target_length, endpoint=False)
            audio_array = np.interp(target_positions, source_positions, audio_array).astype("float32")
            sampling_rate = 16000
        return audio_array, sampling_rate

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split, streaming=True)
    if hasattr(dataset, "cast_column"):
        dataset = dataset.cast_column("audio", Audio(decode=False))
    if hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=args.seed)
    sample_count = max(int(args.num_samples), 0)
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(itertools.islice(dataset, sample_count)):
        audio_array, sampling_rate = _decode_audio(row["audio"])
        samples.append(
            {
                "id": str(row.get("id", index)),
                "audio": {"array": audio_array, "sampling_rate": sampling_rate},
                "reference": str(row.get("text", "")),
            }
        )
    return samples


def _resolve_generate_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_beams": int(args.num_beams),
        "max_new_tokens": int(args.max_new_tokens),
        "return_timestamps": False,
    }
    if int(args.num_beams) > 1:
        kwargs["early_stopping"] = True
    return kwargs


def _extract_generated_token_count(pipe: Any, output: Any, text: str) -> int:
    if isinstance(output, dict):
        for key in ("token_ids", "tokens", "generated_token_ids"):
            value = output.get(key)
            if isinstance(value, list):
                return len(value)
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is None:
        return len(text.split())
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        input_ids = encoded.get("input_ids", [])
        return len(input_ids) if isinstance(input_ids, list) else len(text.split())
    except Exception:
        return len(text.split())


def _run_one_sample(pipe: Any, sample: Mapping[str, Any], generate_kwargs: Mapping[str, Any]) -> SampleTiming:
    audio = sample["audio"]
    audio_array = audio["array"]
    sampling_rate = int(audio["sampling_rate"])
    start = time.perf_counter()

    if hasattr(pipe, "transcribe"):
        results = pipe.transcribe(audio=(audio_array, sampling_rate), language=None)
        if not results:
            raise RuntimeError("Qwen3-ASR native transcribe returned no results.")
        elapsed = time.perf_counter() - start
        hypothesis_raw = str(results[0].text)
        reference = normalize_transcript(str(sample["reference"]))
        hypothesis = normalize_transcript(hypothesis_raw)
        return SampleTiming(
            sample_id=str(sample["id"]),
            audio_duration_s=float(len(audio_array)) / float(sampling_rate),
            generate_time_s=float(elapsed),
            num_generated_tokens=len(hypothesis_raw.split()),
            num_beams=int(generate_kwargs.get("num_beams", 1)),
            reference=reference,
            hypothesis=hypothesis,
        )

    output = None
    last_error: BaseException | None = None
    for pipeline_input, extra_kwargs in _asr_pipeline_inputs(sample):
        for attempt_kwargs in _retryable_generate_kwargs(generate_kwargs):
            try:
                output = pipe(
                    pipeline_input,
                    **extra_kwargs,
                    **_asr_pipeline_call_kwargs(attempt_kwargs),
                )
                break
            except TypeError as exc:
                last_error = exc
                continue
            except ValueError as exc:
                message = str(exc).lower()
                if "unexpected" in message or "unsupported" in message or "unused" in message:
                    last_error = exc
                    continue
                raise
        if output is not None:
            break
    if output is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("ASR pipeline produced no output.")
    elapsed = time.perf_counter() - start
    hypothesis_raw = _extract_hypothesis_text(output)
    reference = normalize_transcript(str(sample["reference"]))
    hypothesis = normalize_transcript(hypothesis_raw)
    token_count = _extract_generated_token_count(pipe, output, hypothesis_raw)
    return SampleTiming(
        sample_id=str(sample["id"]),
        audio_duration_s=float(len(audio_array)) / float(sampling_rate),
        generate_time_s=float(elapsed),
        num_generated_tokens=int(token_count),
        num_beams=int(generate_kwargs.get("num_beams", 1)),
        reference=reference,
        hypothesis=hypothesis,
    )


def _warmup(pipe: Any, samples: Sequence[Mapping[str, Any]], generate_kwargs: Mapping[str, Any], n_warmup: int) -> None:
    for sample in samples[: min(int(n_warmup), len(samples))]:
        _run_one_sample(pipe, sample, generate_kwargs)


def _measure_target(
    target_args: argparse.Namespace,
    pipe: Any,
    samples: Sequence[Mapping[str, Any]],
    generate_kwargs: Mapping[str, Any],
) -> tuple[list[SampleTiming], dict[str, float | None], dict[str, list[dict[str, float]]]]:
    tracker = _build_device_tracker_common(target_args, pipe)
    _print_device_status_common(target_args, tracker)
    timings: list[SampleTiming] = []
    try:
        if tracker is not None:
            tracker.start()
        for sample in tqdm(samples, desc="ASR samples", leave=False, unit="sample"):
            timings.append(_run_one_sample(pipe, sample, generate_kwargs))
    finally:
        _stop_tracker_safe_common(tracker)
    device_metric = _extract_device_metric_common(tracker) if tracker is not None else {}
    device_trace = _extract_device_time_series_common(tracker) if tracker is not None else {}
    return timings, device_metric, device_trace


def _write_target_json(
    out_path: Path,
    *,
    target: ASRBenchmarkTarget,
    args: argparse.Namespace,
    revision: str | None,
    core_mode: str | None,
    summary: ASRMetricSummary,
    device_metric: Mapping[str, float | None],
    device_trace: Mapping[str, list[dict[str, float]]],
    sample_timings: Sequence[SampleTiming],
) -> None:
    payload: dict[str, Any] = {
        "benchmark_type": "automatic-speech-recognition",
        "model": target.label,
        "model_id": target.model_id,
        "label": target.label,
        "revision": revision,
        "num_beams": args.num_beams,
        "dataset": {
            "name": args.dataset,
            "config": args.dataset_config,
            "split": args.dataset_split,
            "language": args.language,
            "task": args.task,
        },
        "mxq_path": target.mxq_path,
        "core_mode": core_mode,
        "asr": summary_to_dict(summary),
        "device": dict(device_metric),
        "device_trace": dict(device_trace),
    }
    if args.save_samples:
        payload["samples"] = [asdict(item) for item in sample_timings]
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _write_combined_outputs(out_dir: Path, num_beams: int) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob(f"*_beams{num_beams}.json")):
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
        asr = payload.get("asr")
        if not isinstance(asr, dict):
            continue
        summary = ASRMetricSummary(**asr)
        device_metric = payload.get("device") if isinstance(payload.get("device"), dict) else {}
        rows.append(format_metrics_row(str(payload.get("model", path.stem)), num_beams, summary, device_metric))

    combined_csv = out_dir / f"combined_beams{num_beams}.csv"
    combined_md = out_dir / f"combined_beams{num_beams}.md"
    if rows:
        headers = list(rows[0].keys())
        with combined_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: ("" if value is None else value) for key, value in row.items()})

        markdown_rows = []
        for row in rows:
            markdown_rows.append([
                row.get("model", ""),
                row.get("num_beams", ""),
                f"{100.0 * float(row.get('wer', 0.0)):.2f}",
                f"{100.0 * float(row.get('cer', 0.0)):.2f}",
                f"{float(row.get('mean_latency_s', 0.0)):.4f}",
                f"{float(row.get('p95_latency_s', 0.0)):.4f}",
                f"{float(row.get('throughput_samples_per_s', 0.0)):.4f}",
                f"{float(row.get('rtf', 0.0)):.4f}",
                f"{float(row.get('inverse_rtf', 0.0)):.4f}",
                f"{float(row.get('decode_tokens_per_s', 0.0)):.4f}",
            ])
        combined_md.write_text(
            _markdown_table_common(
                [
                    "model",
                    "num_beams",
                    "WER(%)",
                    "CER(%)",
                    "mean_latency_s",
                    "p95_latency_s",
                    "samples_per_s",
                    "RTF",
                    "inverse_RTF",
                    "decode_tokens_per_s",
                ],
                markdown_rows,
            ),
            encoding="utf-8",
        )
    else:
        combined_md.write_text("No ASR results found.\n", encoding="utf-8")

    _make_rtf_chart(out_dir, num_beams, rows)
    _write_summary_markdown(
        out_dir / f"summary_beams{num_beams}.md",
        title=f"Automatic Speech Recognition Benchmark Summary (beams={num_beams})",
        host_info_path=out_dir / _HOST_PC_INFO_FILENAME,
        table_markdown_path=combined_md,
        plot_paths=_existing_png_paths(
            out_dir,
            prefixes=(f"rtf_beams{num_beams}", f"wer_beams{num_beams}", f"cer_beams{num_beams}"),
        ),
        plot_tables={},
    )


def _make_rtf_chart(out_dir: Path, num_beams: int, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    metrics_by_folder: list[dict[str, Any]] = []
    folder_metrics: dict[str, Any] = {}
    for row in rows:
        model_name = str(row.get("model", ""))
        folder_metrics[model_name] = row
    metrics_by_folder.append(folder_metrics)
    models = sorted(folder_metrics.keys())
    labels = [f"beams={num_beams}"]

    def _selector(key: str):
        return lambda item: None if item.get(key) is None else float(item[key])

    for filename, key, title, x_label in (
        (f"rtf_beams{num_beams}.png", "rtf", "Real-Time Factor", "RTF"),
        (f"wer_beams{num_beams}.png", "wer", "Word Error Rate", "WER"),
        (f"cer_beams{num_beams}.png", "cer", "Character Error Rate", "CER"),
    ):
        try:
            plot_scalar_chart(
                models=models,
                folder_labels=labels,
                metrics_by_folder=metrics_by_folder,
                scalar_selector=_selector(key),
                title=title,
                x_label=x_label,
                output_path=out_dir / filename,
            )
        except Exception as exc:
            print(f"Warning: failed to build {filename}: {exc}")


def _resolve_results_dir(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _build_run_targets(args: argparse.Namespace) -> list[tuple[ASRBenchmarkTarget, str | None, str, str]]:
    available_model_ids = _list_default_asr_models()
    targets: list[ASRBenchmarkTarget]
    if args.mxq_dir:
        mxq_dir = Path(args.mxq_dir).expanduser().resolve()
        if not mxq_dir.is_dir():
            raise SystemExit(f"--mxq-dir is not a directory: {mxq_dir}")
        if args.model_ids or args.original_models or args.all_revisions or args.revision or args.mxq_path:
            print(
                "Note: --mxq-dir is set, so --model-id/--original-models/"
                "--all-revisions/--revision/--mxq-path are ignored."
            )
        targets = _iter_asr_targets_from_mxq_dir(mxq_dir, available_model_ids)
        if not targets:
            raise SystemExit("No valid mxq targets found. Expected files named <model_id>-<W8|W4V8>.mxq in --mxq-dir.")
    else:
        model_ids = [str(item) for item in args.model_ids] if args.model_ids else available_model_ids
        if args.original_models:
            original_count = len(model_ids)
            model_ids = _resolve_original_model_ids(model_ids)
            print(
                f"Using parent/original model ids: {len(model_ids)} unique models "
                f"(from {original_count} listed models)."
            )
        targets = list(
            _iter_asr_targets(
                model_ids,
                revision=args.revision,
                all_revisions=args.all_revisions,
                is_original=args.original_models,
            )
        )
        if args.mxq_path:
            targets = [
                ASRBenchmarkTarget(
                    model_id=target.model_id,
                    revision_candidates=target.revision_candidates,
                    label=target.label,
                    base=target.base,
                    mxq_path=args.mxq_path,
                    is_original=target.is_original,
                )
                for target in targets
            ]

    run_targets: list[tuple[ASRBenchmarkTarget, str | None, str, str]] = []
    core_modes = [None] if (args.original_models and not args.mxq_dir) else _iter_core_modes_common(args.core_mode)
    for target in targets:
        for core_mode in core_modes:
            mode_label, mode_base = _append_core_mode_suffix_common(target.label, target.base, core_mode)
            run_targets.append((target, core_mode, mode_label, mode_base))
    return run_targets


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = _parse_args(argv)
    _resolve_runtime_defaults(args, raw_argv)
    os.environ.setdefault("MPLBACKEND", "Agg")
    out_dir = _resolve_results_dir(args)
    _collect_host_pc_info(out_dir)
    run_targets = _build_run_targets(args)
    samples = _load_librispeech(args)
    base_generate_kwargs = _resolve_generate_kwargs(args)

    if args.dry_run:
        print(f"Resolved {len(run_targets)} target(s).")
        if samples:
            first = samples[0]
            print(
                "First sample: "
                f"id={first['id']} sr={first['audio']['sampling_rate']} len={len(first['audio']['array'])} "
                f"reference={first['reference'][:80]}"
            )
        return 0

    for target, core_mode, mode_label, mode_base in tqdm(
        run_targets,
        desc="Benchmarking ASR models",
        unit="model-mode",
    ):
        generate_kwargs = {
            **base_generate_kwargs,
            **_optional_generate_kwargs_for_model(args, target.model_id),
        }
        target_args = _args_for_target_device_backend(args, model_id=target.model_id, mxq_path=target.mxq_path)
        if _is_cuda_device(args.device):
            _clear_cuda_memory(args.device)
        revision = (
            target.revision_candidates[0]
            if target.mxq_path
            else _select_revision(target.model_id, target.revision_candidates)
        )
        if args.all_revisions and not args.mxq_dir and revision is None:
            print(f"Skipping {mode_label} (missing revisions).")
            continue
        json_path = out_dir / f"{mode_base}_beams{args.num_beams}.json"
        print(f"=== {mode_label} ===")
        print(
            f"Run config: revision={revision or 'main'} num_beams={args.num_beams} core_mode={core_mode or 'default'} "
            f"device={args.device} device_backend={target_args.device_backend} samples={len(samples)}"
        )
        pipe = None
        try:
            try:
                pipe = _build_asr_pipeline(
                    target,
                    revision=revision,
                    device=args.device,
                    device_map=args.device_map,
                    dtype=args.dtype,
                    trust_remote_code=args.trust_remote_code,
                    core_mode=core_mode,
                )
            except Exception as exc:
                if _is_cuda_oom_error(exc):
                    print(f"Skipping (CUDA OOM while loading model): {exc}")
                    _clear_cuda_memory(args.device)
                    continue
                _print_exception("Skipping (failed to load model)", exc, debug_errors=args.debug_errors)
                continue

            _warmup(pipe, samples, generate_kwargs, args.warmup)
            timings, device_metric, device_trace = _measure_target(target_args, pipe, samples, generate_kwargs)
            summary = summarize_timings(timings)
            _write_target_json(
                json_path,
                target=target,
                args=args,
                revision=revision,
                core_mode=core_mode,
                summary=summary,
                device_metric=device_metric,
                device_trace=device_trace,
                sample_timings=timings,
            )
            print(
                f"WER={100.0 * summary.wer:.2f}% CER={100.0 * summary.cer:.2f}% "
                f"RTF={summary.rtf:.4f} throughput={summary.throughput_samples_per_s:.4f} samples/s"
            )
        except Exception as exc:
            if _is_cuda_oom_error(exc):
                print(f"Skipping (CUDA OOM during benchmark): {exc}")
                _release_pipeline(pipe, args.device)
                continue
            _print_exception("Skipping (benchmark failed)", exc, debug_errors=args.debug_errors)
            _release_pipeline(pipe, args.device)
            continue
        _release_pipeline(pipe, args.device)

    _write_combined_outputs(out_dir, int(args.num_beams))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())