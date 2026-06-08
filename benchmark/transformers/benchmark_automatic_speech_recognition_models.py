import argparse
import json
import logging  # noqa: F401
import os
import sys
import time
import traceback
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch  # noqa: F401

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
from benchmark.transformers.asr_metrics import (
    ASRMetricSummary,
    SampleTiming,
    format_metrics_row,
    summarize_timings,
    summary_to_dict,
)
from benchmark.transformers.asr_output_utils import make_rtf_chart as _make_rtf_chart_shared
from benchmark.transformers.asr_output_utils import write_combined_outputs as _write_combined_outputs_shared
from benchmark.transformers.asr_pipeline_utils import asr_pipeline_inputs as _asr_pipeline_inputs_shared
from benchmark.transformers.asr_pipeline_utils import build_asr_pipeline as _build_asr_pipeline_shared
from benchmark.transformers.asr_pipeline_utils import (
    extract_generated_token_count as _extract_generated_token_count_shared,
)
from benchmark.transformers.asr_pipeline_utils import extract_hypothesis_text as _extract_hypothesis_text_shared
from benchmark.transformers.asr_pipeline_utils import (
    is_retryable_generate_kwargs_error as _is_retryable_generate_kwargs_error_shared,
)
from benchmark.transformers.asr_pipeline_utils import retryable_generate_kwargs as _retryable_generate_kwargs_shared
from benchmark.transformers.asr_pipeline_utils import run_one_sample as _run_one_sample_shared
from benchmark.transformers.asr_qwen_utils import (  # noqa: F401
    configure_native_qwen3_asr_generate as _configure_native_qwen3_asr_generate,
)
from benchmark.transformers.asr_qwen_utils import (  # noqa: F401
    ensure_native_qwen3_asr_generation_config as _ensure_native_qwen3_asr_generation_config,
)
from benchmark.transformers.asr_qwen_utils import (  # noqa: F401
    ensure_qwen3_asr_backend_registered as _ensure_qwen3_asr_backend_registered,
)
from benchmark.transformers.asr_qwen_utils import is_qwen3_asr_model as _is_qwen3_asr_model
from benchmark.transformers.asr_qwen_utils import move_native_qwen3_asr_to_device as _move_native_qwen3_asr_to_device
from benchmark.transformers.asr_qwen_utils import quiet_apscheduler_info_logs as _quiet_apscheduler_info_logs
from benchmark.transformers.asr_qwen_utils import (  # noqa: F401
    resolve_native_qwen3_asr_pad_token_id as _resolve_native_qwen3_asr_pad_token_id,
)
from benchmark.transformers.asr_qwen_utils import resolve_torch_dtype as _resolve_torch_dtype
from benchmark.transformers.asr_qwen_utils import (  # noqa: F401
    supports_native_transcribe_language as _supports_native_transcribe_language,
)
from benchmark.transformers.benchmark_target_utils import (
    args_for_target_device_backend as _args_for_target_device_backend_shared,
)
from benchmark.transformers.benchmark_target_utils import (
    iter_targets_from_mxq_dir as _iter_targets_from_mxq_dir_shared,
)
from benchmark.transformers.benchmark_target_utils import (
    resolve_model_id_from_mxq_name as _resolve_model_id_from_mxq_name_shared,
)
from benchmark.transformers.benchmark_target_utils import (
    resolve_original_model_ids as _resolve_original_model_ids_shared,
)
from benchmark.transformers.benchmark_target_utils import revision_exists as _revision_exists_shared
from benchmark.transformers.benchmark_target_utils import select_revision as _select_revision_shared
from mblt_model_zoo.hf_transformers.utils import list_models
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    CORE_MODE_CHOICES as _CORE_MODE_CHOICES_COMMON,
)
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
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    print_device_status as _print_device_status_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    resolve_default_device as _resolve_default_device_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    resolve_default_device_backend as _resolve_default_device_backend_common,
)
from mblt_model_zoo.hf_transformers.utils.benchmark_cli_common import (
    stop_tracker_safe as _stop_tracker_safe_common,
)


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
    return _asr_pipeline_inputs_shared(sample)


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
    return _retryable_generate_kwargs_shared(generate_kwargs)


def _is_retryable_generate_kwargs_error(exc: TypeError | ValueError) -> bool:
    return _is_retryable_generate_kwargs_error_shared(exc)


def _extract_hypothesis_text(output: Any) -> str:
    return _extract_hypothesis_text_shared(output)


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
        "--skip-existing",
        action="store_true",
        help="skip target/beam outputs that already exist instead of failing",
    )
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
    from transformers import pipeline as hf_pipeline

    return _build_asr_pipeline_shared(
        target,
        revision=revision,
        device=device,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        core_mode=core_mode,
        native_generate_kwargs=native_generate_kwargs,
        is_cuda_device=_is_cuda_device,
        resolve_torch_dtype=_resolve_torch_dtype,
        is_qwen3_asr_model=_is_qwen3_asr_model,
        quiet_apscheduler_info_logs=_quiet_apscheduler_info_logs,
        move_native_qwen3_asr_to_device=_move_native_qwen3_asr_to_device,
        configure_native_qwen3_asr_generate=_configure_native_qwen3_asr_generate,
        ensure_qwen3_asr_backend_registered=_ensure_qwen3_asr_backend_registered,
        apply_asr_core_mode_model_kwargs=_apply_asr_core_mode_model_kwargs,
        hf_pipeline=hf_pipeline,
    )


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


def _load_measurement_candidate_samples(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    """Load enough bounded ASR samples to cover warmup plus measured samples.

    For bounded runs, ``--num-samples`` is treated as the number of samples that should
    contribute to measured metrics, while ``--warmup`` remains an additional prefix.
    Full-split runs keep streaming semantics and are handled separately by callers.
    """

    if args.num_samples is None:
        return _load_librispeech(args)

    requested = max(int(args.num_samples), 0)
    warmup = max(int(args.warmup), 0)
    bounded_args = argparse.Namespace(**vars(args))
    bounded_args.num_samples = requested + warmup
    return _load_librispeech(bounded_args)


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


def _sample_preview(
    samples: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, Iterable[Mapping[str, Any]]]:
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


def _result_json_path(out_dir: Path, mode_base: str, num_beams: int | None) -> Path:
    """Return the per-target JSON path for one beam configuration."""

    return out_dir / f"{mode_base}_beams{_beam_tag(num_beams)}.json"


def _handle_existing_result(path: Path, *, skip_existing: bool) -> bool:
    """Return whether the caller should skip because the result already exists."""

    if not path.exists():
        return False
    if skip_existing:
        print(f"Skipping existing result: {path.name}")
        return True
    raise SystemExit(
        f"Result already exists: {path}. Reuse --skip-existing to keep the current file or choose a different "
        "--output-dir."
    )


def _consume_warmup_samples(
    model_id: str,
    pipe: Any,
    samples: Iterable[Mapping[str, Any]],
    generate_kwargs: Mapping[str, Any],
    n_warmup: int,
    *,
    native_language: str | None = None,
) -> Iterator[Mapping[str, Any]]:
    """Consume warmup samples from one iterator and yield the remaining measurement samples."""

    iterator = iter(samples)
    _warmup(
        model_id,
        pipe,
        iterator,
        generate_kwargs,
        n_warmup,
        native_language=native_language,
    )
    return iterator


def _limit_measurement_samples(
    samples: Iterable[Mapping[str, Any]],
    *,
    num_samples: int | None,
) -> Iterable[Mapping[str, Any]]:
    """Limit post-warmup measurement samples when a bounded target count is requested."""

    if num_samples is None:
        return samples

    def _limited() -> Iterator[Mapping[str, Any]]:
        yielded = 0
        for sample in samples:
            if yielded >= int(num_samples):
                break
            yield sample
            yielded += 1

    return _limited()


def _build_no_samples_payload(
    *,
    target: ASRBenchmarkTarget,
    args: argparse.Namespace,
    revision: str | None,
    core_mode: str | None,
    reason: str,
) -> dict[str, Any]:
    """Build a status-only payload for targets that produced no measured ASR samples."""

    return {
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
        "status": "no_samples",
        "reason": reason,
    }


def _write_status_json(out_path: Path, payload: Mapping[str, Any]) -> None:
    """Write a status-only benchmark payload."""

    with out_path.open("w", encoding="utf-8") as file:
        json.dump(dict(payload), file, indent=2, ensure_ascii=False)


def _extract_generated_token_count(pipe: Any, output: Any, text: str) -> int:
    return _extract_generated_token_count_shared(pipe, output, text)


def _run_one_sample(
    pipe: Any,
    sample: Mapping[str, Any],
    generate_kwargs: Mapping[str, Any],
    *,
    native_language: str | None = None,
) -> SampleTiming:
    return _run_one_sample_shared(
        pipe=pipe,
        sample=sample,
        generate_kwargs=generate_kwargs,
        native_language=native_language,
        time_module=time,
        supports_native_transcribe_language=_supports_native_transcribe_language,
        pipeline_inputs_builder=_asr_pipeline_inputs,
        pipeline_call_kwargs_builder=_asr_pipeline_call_kwargs,
        retryable_generate_kwargs_builder=_retryable_generate_kwargs,
        retryable_error_checker=_is_retryable_generate_kwargs_error,
        hypothesis_text_extractor=_extract_hypothesis_text,
        generated_token_count_extractor=_extract_generated_token_count,
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


def _write_combined_outputs(out_dir: Path) -> None:
    _write_combined_outputs_shared(
        out_dir=out_dir,
        host_pc_info_filename=_HOST_PC_INFO_FILENAME,
        asr_metric_summary_cls=ASRMetricSummary,
        format_metrics_row_func=format_metrics_row,
        markdown_table_func=_markdown_table_common,
        existing_png_paths_func=_existing_png_paths,
        write_summary_markdown_func=_write_summary_markdown,
        make_rtf_chart_func=_make_rtf_chart,
    )


def _make_rtf_chart(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _make_rtf_chart_shared(out_dir=out_dir, rows=rows, plot_scalar_chart_func=plot_scalar_chart)


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
    if args.original_models and not args.mxq_dir and getattr(args, "_core_mode_explicit", False):
        print("Note: --core-mode is not supported for --original-models ASR native runs and will be ignored.")
    for target in targets:
        for core_mode in core_modes:
            mode_label, mode_base = _append_core_mode_suffix_common(target.label, target.base, core_mode)
            run_targets.append((target, core_mode, mode_label, mode_base))
    return run_targets


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = _parse_args(argv)
    args._core_mode_explicit = _flag_present(raw_argv, "--core-mode")
    _resolve_runtime_defaults(args, raw_argv)
    os.environ.setdefault("MPLBACKEND", "Agg")
    out_dir = _resolve_results_dir(args)
    _collect_host_pc_info(out_dir)
    run_targets = _build_run_targets(args)
    sampled_samples = None if args.num_samples is None else list(_load_measurement_candidate_samples(args))
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
        json_path = _result_json_path(out_dir, mode_base, args.num_beams)
        current_samples = _load_librispeech(args) if args.num_samples is None else (sampled_samples or [])
        print(f"=== {mode_label} ===")
        print(
            f"Run config: revision={revision or 'main'} num_beams={beam_tag} core_mode={core_mode or 'default'} "
            f"device={args.device} device_backend={target_args.device_backend} "
            f"samples={('full-split' if args.num_samples is None else int(args.num_samples))}"
        )
        if _handle_existing_result(json_path, skip_existing=args.skip_existing):
            continue
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
            measure_samples = _consume_warmup_samples(
                target.model_id,
                pipe,
                iter(current_samples),
                generate_kwargs,
                args.warmup,
                native_language=args.language,
            )
            measure_samples = _limit_measurement_samples(measure_samples, num_samples=args.num_samples)
            timings, device_metric, device_trace = _measure_target(
                target.model_id,
                target_args,
                pipe,
                measure_samples,
                generate_kwargs,
                native_language=args.language,
            )
            if not timings:
                reason = "No measured samples remained after warmup/skip filtering."
                print(f"Warning: {reason} model={mode_label}")
                _write_status_json(
                    json_path,
                    _build_no_samples_payload(
                        target=target,
                        args=args,
                        revision=revision,
                        core_mode=core_mode,
                        reason=reason,
                    ),
                )
                _release_pipeline(pipe, args.device)
                continue
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

    _write_combined_outputs(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
