import argparse
import csv
import functools
import inspect
import itertools
import json
import logging
import os
import sys
import time
import traceback
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

# ruff: noqa: E402
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from benchmark.transformers.chart_utils import plot_scalar_chart
except ModuleNotFoundError:
    from chart_utils import plot_scalar_chart
from tqdm import tqdm

from benchmark.common.dataset_utils import load_streaming_audio_text_samples
from benchmark.common.dataset_utils import resample_audio as _resample_audio_common
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
from benchmark.transformers.benchmark_target_utils import args_for_target_device_backend as _args_for_target_device_backend_shared
from benchmark.transformers.benchmark_target_utils import iter_targets_from_mxq_dir as _iter_targets_from_mxq_dir_shared
from benchmark.transformers.benchmark_target_utils import resolve_model_id_from_mxq_name as _resolve_model_id_from_mxq_name_shared
from benchmark.transformers.benchmark_target_utils import resolve_original_model_ids as _resolve_original_model_ids_shared
from benchmark.transformers.benchmark_target_utils import revision_exists as _revision_exists_shared
from benchmark.transformers.benchmark_target_utils import select_revision as _select_revision_shared
from benchmark.transformers.asr_metrics import (
    ASRMetricSummary,
    SampleTiming,
    format_metrics_row,
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


def _resolve_original_model_ids(model_ids: Iterable[str]) -> list[str]:
    return _resolve_original_model_ids_shared(model_ids)


def _revision_exists(model_id: str, revision: str) -> bool | None:
    return _revision_exists_shared(model_id, revision)


def _select_revision(model_id: str, candidates: list[str | None]) -> str | None:
    return _select_revision_shared(model_id, candidates)


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


def _sample_audio_duration_s(sample: Mapping[str, Any]) -> float:
    """Return one sample's audio duration in seconds."""

    audio = sample["audio"]
    audio_array = audio["array"]
    sampling_rate = int(audio["sampling_rate"])
    return float(len(audio_array)) / float(sampling_rate)


def _should_skip_whisper_long_form_sample(model_id: str, sample: Mapping[str, Any]) -> bool:
    """Return whether a Whisper sample should be skipped for exceeding the 30s short-form limit."""

    return _is_whisper_like_model(model_id) and _sample_audio_duration_s(sample) > 30.0


def _retryable_generate_kwargs(generate_kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
    current = dict(generate_kwargs)
    attempts = [dict(current)]
    for key in ("task", "language", "return_timestamps", "early_stopping"):
        if key in current:
            current = dict(current)
            current.pop(key, None)
            attempts.append(dict(current))
    return attempts


def _is_retryable_generate_kwargs_error(exc: TypeError | ValueError) -> bool:
    """Return whether an exception indicates unsupported pipeline generate kwargs."""

    message = str(exc).lower()
    return "unexpected" in message or "unsupported" in message or "unused" in message


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
    return _resolve_model_id_from_mxq_name_shared(model_part, available_model_ids)


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
    return [
        ASRBenchmarkTarget(
            model_id=model_id,
            revision_candidates=revision_candidates,
            label=label,
            base=base,
            mxq_path=mxq_path,
            is_original=False,
        )
        for model_id, revision_candidates, label, base, mxq_path in _iter_targets_from_mxq_dir_shared(
            mxq_dir=mxq_dir,
            available_model_ids=available_model_ids,
            safe_filename=_safe_filename,
        )
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=("Benchmark Hugging Face Transformers automatic-speech-recognition pipeline-compatible models.")
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
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help=(
            "number of evaluation samples; defaults to 50. Use --full-split to evaluate the full dataset split "
            "(streaming avoids eager full download, but runtime still scales with the full split)"
        ),
    )
    parser.add_argument(
        "--full-split",
        action="store_true",
        help="evaluate the full dataset split instead of the default --num-samples subset",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=None,
        help="beam value to benchmark; omit to use model default",
    )
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
    args = parser.parse_args(argv)
    num_samples_explicit = _flag_present(raw_argv, "--num-samples")
    if isinstance(args.dataset_config, str) and args.dataset_config.casefold() == "none":
        args.dataset_config = None
    if args.full_split and num_samples_explicit:
        print("Note: --full-split and --num-samples were both provided; using --num-samples and ignoring --full-split.")
        args.full_split = False
    elif args.full_split:
        args.num_samples = None
    return args


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
    return _args_for_target_device_backend_shared(
        args,
        model_id=model_id,
        mxq_path=mxq_path,
        resolve_default_device_backend=_resolve_default_device_backend_common,
    )


def _build_asr_pipeline(
    target: ASRBenchmarkTarget,
    *,
    revision: str | None,
    device: str | None,
    device_map: str | None,
    dtype: str | None,
    trust_remote_code: bool,
    core_mode: str | None,
    native_generate_kwargs: Mapping[str, Any] | None = None,
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

        _quiet_apscheduler_info_logs()

        resolved_native_generate_kwargs = dict(native_generate_kwargs or {})
        native_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "max_inference_batch_size": 1,
            "max_new_tokens": int(resolved_native_generate_kwargs.get("max_new_tokens", 512)),
        }
        torch_dtype = _resolve_torch_dtype(dtype)
        if device_map:
            native_kwargs["device_map"] = device_map
        elif _is_cuda_device(device):
            native_kwargs["device_map"] = device
        if torch_dtype is not None:
            native_kwargs["torch_dtype"] = torch_dtype
        if revision:
            native_kwargs["revision"] = revision
        pipe = qwen_asr.Qwen3ASRModel.from_pretrained(target.model_id, **native_kwargs)
        _move_native_qwen3_asr_to_device(pipe, device=device, device_map=device_map)
        return _configure_native_qwen3_asr_generate(pipe, resolved_native_generate_kwargs)

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


def _configure_native_qwen3_asr_generate(pipe: Any, generate_kwargs: Mapping[str, Any] | None) -> Any:
    """Attach benchmark generation defaults to upstream native Qwen3-ASR objects.

    The upstream ``Qwen3ASRModel.transcribe()`` helper does not expose ``num_beams`` in its
    public signature, but internally delegates to ``self.model.generate(...)``. For benchmark
    runs we wrap that inner ``generate`` method so CLI decoding controls such as ``--num-beams``
    are applied while preserving explicit kwargs provided by upstream code.
    """
    if not generate_kwargs:
        _ensure_native_qwen3_asr_generation_config(pipe)
        return pipe

    _ensure_native_qwen3_asr_generation_config(pipe)
    inner_model = getattr(pipe, "model", None)
    original_generate = getattr(inner_model, "generate", None)
    if inner_model is None or not callable(original_generate):
        return pipe

    resolved_pad_token_id = _resolve_native_qwen3_asr_pad_token_id(pipe)
    overrides = {
        key: value
        for key, value in dict(generate_kwargs).items()
        if key not in {"return_timestamps"} and value is not None
    }
    if resolved_pad_token_id is not None:
        overrides.setdefault("pad_token_id", resolved_pad_token_id)
    if not overrides:
        return pipe

    @functools.wraps(original_generate)
    def _generate_with_overrides(*args: Any, **kwargs: Any) -> Any:
        merged_kwargs = dict(overrides)
        merged_kwargs.update(kwargs)
        return original_generate(*args, **merged_kwargs)

    inner_model.generate = _generate_with_overrides
    return pipe


def _ensure_native_qwen3_asr_generation_config(pipe: Any) -> Any:
    """Populate native upstream Qwen3-ASR generation config defaults needed by benchmark runs."""
    inner_model = getattr(pipe, "model", None)
    if inner_model is None:
        return pipe

    model_config = getattr(inner_model, "config", None)
    generation_config = getattr(inner_model, "generation_config", None)
    if generation_config is None:
        return pipe

    eos_token_id = getattr(generation_config, "eos_token_id", None)
    if eos_token_id is None and model_config is not None:
        eos_token_id = getattr(model_config, "eos_token_id", None)

    pad_token_id = getattr(generation_config, "pad_token_id", None)
    if pad_token_id is None and model_config is not None:
        pad_token_id = getattr(model_config, "pad_token_id", None)

    if pad_token_id is None and eos_token_id is not None:
        generation_config.pad_token_id = eos_token_id
        if model_config is not None and getattr(model_config, "pad_token_id", None) is None:
            model_config.pad_token_id = eos_token_id

    resolved_pad_token_id = _resolve_native_qwen3_asr_pad_token_id(pipe)
    if resolved_pad_token_id is not None:
        generation_config.pad_token_id = resolved_pad_token_id
        if model_config is not None:
            model_config.pad_token_id = resolved_pad_token_id

    return pipe


def _resolve_native_qwen3_asr_pad_token_id(pipe: Any) -> int | list[int] | None:
    """Resolve a stable pad token id for native upstream Qwen3-ASR generation.

    Transformers emits a runtime log when ``generate()`` sees ``pad_token_id`` unset and
    falls back to ``eos_token_id`` automatically. Benchmark runs do not need that fallback log,
    so we proactively resolve the pad token id from generation/model/tokenizer metadata.
    """
    inner_model = getattr(pipe, "model", None)
    generation_config = getattr(inner_model, "generation_config", None) if inner_model is not None else None
    model_config = getattr(inner_model, "config", None) if inner_model is not None else None
    tokenizer = getattr(pipe, "tokenizer", None)

    candidates = [
        getattr(generation_config, "pad_token_id", None),
        getattr(model_config, "pad_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(generation_config, "eos_token_id", None),
        getattr(model_config, "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def _resolve_torch_dtype(dtype: str | None) -> torch.dtype | None:
    """Resolve CLI dtype text into a torch dtype object when possible."""
    if dtype is None:
        return None
    text = str(dtype).strip()
    if not text:
        return None
    normalized = text.removeprefix("torch.")
    resolved = getattr(torch, normalized, None)
    return resolved if isinstance(resolved, torch.dtype) else None


def _move_native_qwen3_asr_to_device(pipe: Any, *, device: str | None, device_map: str | None) -> Any:
    """Ensure native upstream Qwen3-ASR model is placed on the requested device."""
    if device_map:
        return pipe
    if not device:
        return pipe
    inner_model = getattr(pipe, "model", None)
    move_to = getattr(inner_model, "to", None)
    if not callable(move_to):
        return pipe
    move_to(device)
    return pipe


def _quiet_apscheduler_info_logs() -> None:
    """Prevent APScheduler INFO job logs from leaking after qwen_asr configures root logging."""
    aps_logger = logging.getLogger("apscheduler")
    if aps_logger.level == logging.NOTSET or aps_logger.level < logging.WARNING:
        aps_logger.setLevel(logging.WARNING)


def _resample_audio(audio_array: Any, source_rate: int, target_rate: int = 16000) -> Any:
    """Backward-compatible wrapper around shared benchmark audio resampling."""

    return _resample_audio_common(audio_array, source_rate, target_rate)


def _load_librispeech(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    """Load LibriSpeech-style ASR samples via the shared streaming dataset utility."""

    return load_streaming_audio_text_samples(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        audio_column="audio",
        text_column="text",
        id_column="id",
        num_samples=args.num_samples,
        seed=args.seed,
        target_sampling_rate=16000,
    )


def _resolve_generate_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(args.max_new_tokens),
        "return_timestamps": False,
    }
    if args.num_beams is not None:
        kwargs["num_beams"] = int(args.num_beams)
    if args.num_beams is not None and int(args.num_beams) > 1:
        kwargs["early_stopping"] = True
    return kwargs


def _supports_native_transcribe_language(pipe: Any) -> bool:
    """Return whether a native ASR transcribe callable accepts ``language``."""

    transcribe = getattr(pipe, "transcribe", None)
    if not callable(transcribe):
        return False
    try:
        signature = inspect.signature(transcribe)
    except (TypeError, ValueError):
        return False
    return "language" in signature.parameters


def _sample_preview(samples: Iterable[Mapping[str, Any]]) -> tuple[Mapping[str, Any] | None, Iterable[Mapping[str, Any]]]:
    """Return the first sample plus a replayable iterable including that first sample."""

    iterator = iter(samples)
    first = next(iterator, None)
    if first is None:
        return None, []

    def _replayed() -> Iterator[Mapping[str, Any]]:
        yield first
        yield from iterator

    return first, _replayed()


def _beam_tag(num_beams: int | None) -> str:
    return "default" if num_beams is None else str(int(num_beams))


def _extract_generated_token_count(pipe: Any, output: Any, text: str) -> int:
    if isinstance(output, dict):
        for key in ("token_ids", "tokens", "generated_token_ids"):
            value = output.get(key)
            if isinstance(value, list):
                return len(value)

    def _extract_input_ids_length(encoded: Any) -> int | None:
        if isinstance(encoded, Mapping):
            input_ids = encoded.get("input_ids")
        else:
            input_ids = getattr(encoded, "input_ids", None)
        if input_ids is None:
            return None
        try:
            return len(input_ids)
        except TypeError:
            return None

    processor = getattr(pipe, "processor", None)
    candidates = [
        getattr(pipe, "tokenizer", None),
        getattr(processor, "tokenizer", None) if processor is not None else None,
        processor,
    ]
    for tokenizer in candidates:
        if not callable(tokenizer):
            continue
        try:
            encoded = tokenizer(text, add_special_tokens=False)
        except TypeError:
            try:
                encoded = tokenizer(text)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                continue
        except (AttributeError, ValueError, RuntimeError):
            continue
        input_ids_length = _extract_input_ids_length(encoded)
        if input_ids_length is not None:
            return input_ids_length
    return len(text.split())


def _run_one_sample(
    pipe: Any,
    sample: Mapping[str, Any],
    generate_kwargs: Mapping[str, Any],
    *,
    native_language: str | None = None,
) -> SampleTiming:
    audio = sample["audio"]
    audio_array = audio["array"]
    sampling_rate = int(audio["sampling_rate"])
    start = time.perf_counter()

    if hasattr(pipe, "transcribe"):
        transcribe_kwargs: dict[str, Any] = {"audio": (audio_array, sampling_rate)}
        if native_language is not None and _supports_native_transcribe_language(pipe):
            transcribe_kwargs["language"] = native_language
        results = pipe.transcribe(**transcribe_kwargs)
        if not results:
            raise RuntimeError("Qwen3-ASR native transcribe returned no results.")
        elapsed = time.perf_counter() - start
        hypothesis_raw = str(results[0].text)
        token_count = _extract_generated_token_count(pipe, results[0], hypothesis_raw)
        return SampleTiming(
            sample_id=str(sample["id"]),
            audio_duration_s=float(len(audio_array)) / float(sampling_rate),
            generate_time_s=float(elapsed),
            num_generated_tokens=int(token_count),
            num_beams=(int(generate_kwargs["num_beams"]) if generate_kwargs.get("num_beams") is not None else None),
            reference=str(sample["reference"]),
            hypothesis=hypothesis_raw,
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
                if _is_retryable_generate_kwargs_error(exc):
                    last_error = exc
                    continue
                raise
            except ValueError as exc:
                if _is_retryable_generate_kwargs_error(exc):
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
    token_count = _extract_generated_token_count(pipe, output, hypothesis_raw)
    return SampleTiming(
        sample_id=str(sample["id"]),
        audio_duration_s=float(len(audio_array)) / float(sampling_rate),
        generate_time_s=float(elapsed),
        num_generated_tokens=int(token_count),
        num_beams=(int(generate_kwargs["num_beams"]) if generate_kwargs.get("num_beams") is not None else None),
        reference=str(sample["reference"]),
        hypothesis=hypothesis_raw,
    )


def _warmup(
    model_id: str,
    pipe: Any,
    samples: Iterable[Mapping[str, Any]],
    generate_kwargs: Mapping[str, Any],
    n_warmup: int,
    *,
    native_language: str | None = None,
) -> None:
    completed = 0
    for sample in samples:
        if completed >= int(n_warmup):
            break
        if _should_skip_whisper_long_form_sample(model_id, sample):
            print(
                "Skipping warmup sample (>30s Whisper limit): "
                f"model={model_id} sample_id={sample['id']} duration_s={_sample_audio_duration_s(sample):.2f}"
            )
            continue
        _run_one_sample(pipe, sample, generate_kwargs, native_language=native_language)
        completed += 1


def _measure_target(
    model_id: str,
    target_args: argparse.Namespace,
    pipe: Any,
    samples: Iterable[Mapping[str, Any]],
    generate_kwargs: Mapping[str, Any],
    *,
    native_language: str | None = None,
) -> tuple[list[SampleTiming], dict[str, float | None], dict[str, list[dict[str, float]]]]:
    tracker = _build_device_tracker_common(target_args, pipe)
    _print_device_status_common(target_args, tracker)
    timings: list[SampleTiming] = []
    try:
        if tracker is not None:
            tracker.start()
        for sample in tqdm(samples, desc="ASR samples", leave=False, unit="sample"):
            if _should_skip_whisper_long_form_sample(model_id, sample):
                print(
                    "Skipping sample (>30s Whisper limit): "
                    f"model={model_id} sample_id={sample['id']} duration_s={_sample_audio_duration_s(sample):.2f}"
                )
                continue
            timings.append(_run_one_sample(pipe, sample, generate_kwargs, native_language=native_language))
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


def _write_combined_outputs(out_dir: Path, num_beams: int | None) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("*.json")):
        if path.name == _HOST_PC_INFO_FILENAME:
            continue
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("benchmark_type") != "automatic-speech-recognition":
            continue
        asr = payload.get("asr")
        if not isinstance(asr, dict):
            continue
        summary = ASRMetricSummary(**asr)
        device_metric = payload.get("device") if isinstance(payload.get("device"), dict) else {}
        rows.append(format_metrics_row(str(payload.get("model", path.stem)), num_beams, summary, device_metric))

    combined_csv = out_dir / "combined.csv"
    combined_md = out_dir / "combined.md"
    if rows:
        headers = list(rows[0].keys())
        with combined_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: ("" if value is None else value) for key, value in row.items()})

        markdown_rows = []
        for row in rows:
            markdown_rows.append(
                [
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
                ]
            )
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
        out_dir / "summary.md",
        title="Automatic Speech Recognition Benchmark Summary",
        host_info_path=out_dir / _HOST_PC_INFO_FILENAME,
        table_markdown_path=combined_md,
        plot_paths=_existing_png_paths(
            out_dir,
            prefixes=("rtf", "wer", "cer"),
        ),
        plot_tables={},
    )


def _make_rtf_chart(out_dir: Path, num_beams: int | None, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    metrics_by_folder: list[dict[str, Any]] = []
    folder_metrics: dict[str, Any] = {}
    for row in rows:
        model_name = str(row.get("model", ""))
        folder_metrics[model_name] = row
    metrics_by_folder.append(folder_metrics)
    models = sorted(folder_metrics.keys())
    labels = [f"beams={_beam_tag(num_beams)}"]

    def _selector(key: str, scale: float = 1.0):
        return lambda item: None if item.get(key) is None else scale * float(item[key])

    for filename, key, title, x_label, scale in (
        ("rtf.png", "rtf", "Real-Time Factor", "RTF", 1.0),
        ("wer.png", "wer", "Word Error Rate", "WER (%)", 100.0),
        ("cer.png", "cer", "Character Error Rate", "CER (%)", 100.0),
    ):
        try:
            plot_scalar_chart(
                models=models,
                folder_labels=labels,
                metrics_by_folder=metrics_by_folder,
                scalar_selector=_selector(key, scale),
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
    targets: list[ASRBenchmarkTarget]
    if args.mxq_dir:
        available_model_ids = _list_default_asr_models()
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
        model_ids = [str(item) for item in args.model_ids] if args.model_ids else _list_default_asr_models()
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
    sampled_samples = None if args.num_samples is None else list(_load_librispeech(args))
    base_generate_kwargs = _resolve_generate_kwargs(args)

    if args.dry_run:
        print(f"Resolved {len(run_targets)} target(s).")
        preview_samples = _load_librispeech(args) if args.num_samples is None else (sampled_samples or [])
        first, _ = _sample_preview(preview_samples)
        if first is not None:
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
        beam_tag = _beam_tag(args.num_beams)
        json_path = out_dir / f"{mode_base}.json"
        current_samples = _load_librispeech(args) if args.num_samples is None else (sampled_samples or [])
        print(f"=== {mode_label} ===")
        print(
            f"Run config: revision={revision or 'main'} num_beams={beam_tag} core_mode={core_mode or 'default'} "
            f"device={args.device} device_backend={target_args.device_backend} "
            f"samples={('full-split' if args.num_samples is None else len(current_samples))}"
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
                    native_generate_kwargs=generate_kwargs,
                )
            except Exception as exc:
                if _is_cuda_oom_error(exc):
                    print(f"Skipping (CUDA OOM while loading model): {exc}")
                    _clear_cuda_memory(args.device)
                    continue
                _print_exception("Skipping (failed to load model)", exc, debug_errors=args.debug_errors)
                continue
            if args.num_samples is None:
                warmup_samples, measure_samples = itertools.tee(current_samples)
                _warmup(
                    target.model_id,
                    pipe,
                    warmup_samples,
                    generate_kwargs,
                    args.warmup,
                    native_language=args.language,
                )
            else:
                _warmup(
                    target.model_id,
                    pipe,
                    current_samples,
                    generate_kwargs,
                    args.warmup,
                    native_language=args.language,
                )
                measure_samples = current_samples
            timings, device_metric, device_trace = _measure_target(
                target.model_id,
                target_args,
                pipe,
                measure_samples,
                generate_kwargs,
                native_language=args.language,
            )
            summary = summarize_timings(timings, language=args.language)
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

    _write_combined_outputs(out_dir, args.num_beams)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
