"""Pipeline helpers for the ASR benchmark CLI.

This module isolates model-loading and per-sample execution behavior so the main
ASR benchmark entry script can stay focused on CLI orchestration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.transformers.asr_metrics import SampleTiming


def asr_pipeline_inputs(sample: Mapping[str, Any]) -> list[tuple[Any, dict[str, Any]]]:
    """Build pipeline-compatible ASR inputs for one sample."""

    audio = sample["audio"]
    audio_array = audio["array"]
    sampling_rate = int(audio["sampling_rate"])
    return [({"raw": audio_array, "sampling_rate": sampling_rate}, {})]


def extract_hypothesis_text(output: Any) -> str:
    """Extract transcript text from heterogeneous ASR pipeline outputs."""

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


def retryable_generate_kwargs(generate_kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return retry variants that progressively drop optional kwargs."""

    current = dict(generate_kwargs)
    attempts = [dict(current)]
    for key in ("task", "language", "return_timestamps", "early_stopping", "max_new_tokens", "num_beams"):
        if key in current:
            current = dict(current)
            current.pop(key, None)
            attempts.append(dict(current))
    return attempts


def is_retryable_generate_kwargs_error(exc: TypeError | ValueError) -> bool:
    """Return whether an exception indicates unsupported pipeline kwargs."""

    message = str(exc).lower()
    known_kwargs = (
        "task",
        "language",
        "return_timestamps",
        "early_stopping",
        "max_new_tokens",
        "num_beams",
        "generate_kwargs",
    )
    has_known_kwarg = any(keyword in message for keyword in known_kwargs)
    has_unsupported_shape = any(keyword in message for keyword in ("unexpected keyword", "unsupported", "unused"))
    return has_known_kwarg and has_unsupported_shape


def extract_generated_token_count(pipe: Any, output: Any, text: str) -> int:
    """Best-effort token count extraction for benchmark throughput reporting."""

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


def build_asr_pipeline(
    target: Any,
    *,
    revision: str | None,
    device: str | None,
    device_map: str | None,
    dtype: str | None,
    trust_remote_code: bool,
    core_mode: str | None,
    native_generate_kwargs: Mapping[str, Any] | None,
    is_cuda_device: Any,
    resolve_torch_dtype: Any,
    is_qwen3_asr_model: Any,
    quiet_apscheduler_info_logs: Any,
    move_native_qwen3_asr_to_device: Any,
    configure_native_qwen3_asr_generate: Any,
    ensure_qwen3_asr_backend_registered: Any,
    apply_asr_core_mode_model_kwargs: Any,
    hf_pipeline: Any,
) -> Any:
    """Build one ASR pipeline/native model for benchmarking."""

    if target.is_original and is_qwen3_asr_model(target.model_id):
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

        quiet_apscheduler_info_logs()

        resolved_native_generate_kwargs = dict(native_generate_kwargs or {})
        native_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "max_inference_batch_size": 1,
            "max_new_tokens": int(resolved_native_generate_kwargs.get("max_new_tokens", 512)),
        }
        torch_dtype = resolve_torch_dtype(dtype)
        if device_map:
            native_kwargs["device_map"] = device_map
        elif is_cuda_device(device):
            native_kwargs["device_map"] = device
        if torch_dtype is not None:
            native_kwargs["torch_dtype"] = torch_dtype
        if revision:
            native_kwargs["revision"] = revision
        pipe = qwen_asr.Qwen3ASRModel.from_pretrained(target.model_id, **native_kwargs)
        move_native_qwen3_asr_to_device(pipe, device=device, device_map=device_map)
        return configure_native_qwen3_asr_generate(pipe, resolved_native_generate_kwargs)

    if is_qwen3_asr_model(target.model_id):
        ensure_qwen3_asr_backend_registered()

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
    model_kwargs = apply_asr_core_mode_model_kwargs(model_kwargs, target.model_id, core_mode)
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


def run_one_sample(
    *,
    pipe: Any,
    sample: Mapping[str, Any],
    generate_kwargs: Mapping[str, Any],
    native_language: str | None,
    time_module: Any,
    supports_native_transcribe_language: Any,
    pipeline_inputs_builder: Any,
    pipeline_call_kwargs_builder: Any,
    retryable_generate_kwargs_builder: Any,
    retryable_error_checker: Any,
    hypothesis_text_extractor: Any,
    generated_token_count_extractor: Any,
) -> SampleTiming:
    """Execute one ASR sample through a pipeline/native backend."""

    audio = sample["audio"]
    audio_array = audio["array"]
    sampling_rate = int(audio["sampling_rate"])
    start = time_module.perf_counter()

    if hasattr(pipe, "transcribe"):
        transcribe_kwargs: dict[str, Any] = {"audio": (audio_array, sampling_rate)}
        if native_language is not None and supports_native_transcribe_language(pipe):
            transcribe_kwargs["language"] = native_language
        results = pipe.transcribe(**transcribe_kwargs)
        if not results:
            raise RuntimeError("Qwen3-ASR native transcribe returned no results.")
        elapsed = time_module.perf_counter() - start
        hypothesis_raw = str(results[0].text)
        token_count = generated_token_count_extractor(pipe, results[0], hypothesis_raw)
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
    effective_generate_kwargs: dict[str, Any] | None = None
    for pipeline_input, extra_kwargs in pipeline_inputs_builder(sample):
        for attempt_kwargs in retryable_generate_kwargs_builder(generate_kwargs):
            try:
                output = pipe(
                    pipeline_input,
                    **extra_kwargs,
                    **pipeline_call_kwargs_builder(attempt_kwargs),
                )
                effective_generate_kwargs = dict(attempt_kwargs)
                break
            except TypeError as exc:
                if retryable_error_checker(exc):
                    last_error = exc
                    continue
                raise
            except ValueError as exc:
                if retryable_error_checker(exc):
                    last_error = exc
                    continue
                raise
        if output is not None:
            break
    if output is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("ASR pipeline produced no output.")
    elapsed = time_module.perf_counter() - start
    hypothesis_raw = hypothesis_text_extractor(output)
    token_count = generated_token_count_extractor(pipe, output, hypothesis_raw)
    return SampleTiming(
        sample_id=str(sample["id"]),
        audio_duration_s=float(len(audio_array)) / float(sampling_rate),
        generate_time_s=float(elapsed),
        num_generated_tokens=int(token_count),
        num_beams=(int(generate_kwargs["num_beams"]) if generate_kwargs.get("num_beams") is not None else None),
        reference=str(sample["reference"]),
        hypothesis=hypothesis_raw,
        effective_generate_kwargs=effective_generate_kwargs,
    )


def write_combined_outputs(
    *,
    out_dir: Path,
    host_pc_info_filename: str,
    asr_metric_summary_cls: type,
    format_metrics_row_func: Any,
    markdown_table_func: Any,
    existing_png_paths_func: Any,
    write_summary_markdown_func: Any,
    make_rtf_chart_func: Any,
) -> None:
    """Build combined CSV/Markdown/chart outputs from per-target ASR JSON files."""

    import csv
    import dataclasses
    import json

    rows: list[dict[str, Any]] = []
    status_rows: list[list[Any]] = []
    summary_field_names = {field.name for field in dataclasses.fields(asr_metric_summary_cls)}
    for path in sorted(out_dir.glob("*.json")):
        if path.name == host_pc_info_filename:
            continue
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("benchmark_type") != "automatic-speech-recognition":
            continue
        if payload.get("status"):
            status_rows.append(
                [
                    str(payload.get("model", path.stem)),
                    payload.get("num_beams", ""),
                    payload.get("status", ""),
                    payload.get("reason", ""),
                ]
            )
            continue
        asr = payload.get("asr")
        if not isinstance(asr, dict):
            continue
        try:
            summary_payload = {name: asr[name] for name in summary_field_names}
            summary = asr_metric_summary_cls(**summary_payload)
        except KeyError as exc:
            print(f"Warning: skipping {path.name} because ASR summary is missing field: {exc.args[0]}")
            continue
        except TypeError as exc:
            print(f"Warning: skipping {path.name} because ASR summary is invalid: {exc}")
            continue
        device_metric = payload.get("device") if isinstance(payload.get("device"), dict) else {}
        payload_num_beams = payload.get("num_beams")
        row_num_beams = int(payload_num_beams) if isinstance(payload_num_beams, int) else None
        rows.append(
            format_metrics_row_func(
                str(payload.get("model", path.stem)),
                row_num_beams,
                summary,
                device_metric,
            )
        )

    combined_csv = out_dir / "combined.csv"
    combined_md = out_dir / "combined.md"
    status_md = out_dir / "combined_status.md"
    if rows:
        headers: list[str] = []
        seen_headers: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen_headers:
                    headers.append(key)
                    seen_headers.add(key)
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
            markdown_table_func(
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
    status_markdown = ""
    if status_rows:
        status_markdown = markdown_table_func(["model", "num_beams", "status", "reason"], status_rows)
        status_md.write_text(status_markdown, encoding="utf-8")
        with combined_md.open("a", encoding="utf-8") as file:
            file.write("\n## ASR status-only targets\n\n")
            file.write(status_markdown)
    elif status_md.exists():
        status_md.unlink()

    make_rtf_chart_func(out_dir, rows)
    write_summary_markdown_func(
        out_dir / "summary.md",
        title="Automatic Speech Recognition Benchmark Summary",
        host_info_path=out_dir / host_pc_info_filename,
        table_markdown_path=combined_md,
        plot_paths=existing_png_paths_func(
            out_dir,
            prefixes=("rtf", "wer", "cer"),
        ),
        plot_tables={},
    )


def make_rtf_chart(
    *,
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    plot_scalar_chart_func: Any,
) -> None:
    """Render summary RTF/WER/CER charts from combined ASR rows."""

    if not rows:
        return

    def _chart_model_label(row: Mapping[str, Any]) -> str:
        model_name = str(row.get("model", ""))
        num_beams = row.get("num_beams")
        beam_tag = "default" if num_beams is None or num_beams == "" else str(int(num_beams))
        return f"{model_name}_beams{beam_tag}"

    metrics_by_folder: list[dict[str, Any]] = []
    folder_metrics: dict[str, Any] = {}
    for row in rows:
        folder_metrics[_chart_model_label(row)] = row
    metrics_by_folder.append(folder_metrics)
    models = sorted(folder_metrics.keys())
    labels = [out_dir.name]

    def _selector(key: str, scale: float = 1.0):
        return lambda item: None if item.get(key) is None else scale * float(item[key])

    for filename, key, title, x_label, scale in (
        ("rtf.png", "rtf", "Real-Time Factor", "RTF", 1.0),
        ("wer.png", "wer", "Word Error Rate", "WER (%)", 100.0),
        ("cer.png", "cer", "Character Error Rate", "CER (%)", 100.0),
    ):
        try:
            plot_scalar_chart_func(
                models=models,
                folder_labels=labels,
                metrics_by_folder=metrics_by_folder,
                scalar_selector=_selector(key, scale),
                title=title,
                x_label=x_label,
                output_path=out_dir / filename,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"Warning: failed to build {filename}: {exc}")