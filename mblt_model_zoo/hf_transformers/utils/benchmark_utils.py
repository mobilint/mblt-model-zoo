import time
from dataclasses import dataclass, field
from threading import Thread
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
from transformers import TextIteratorStreamer

from .cache_utils import MobilintCache


def npu_latency_pct(total_latency: Optional[float], npu_latency: Optional[float]) -> Optional[float]:
    """Return the NPU latency percentage of total latency.

    Args:
        total_latency: Total latency value in any time unit.
        npu_latency: NPU latency value in the same time unit as ``total_latency``.

    Returns:
        NPU latency percentage, or ``None`` when the value cannot be computed.
    """
    if total_latency is None or npu_latency is None:
        return None
    if total_latency <= 0:
        return None
    return (npu_latency / total_latency) * 100.0


def _call_maybe_getter(obj: object, name: str) -> object | None:
    """Return an attribute value, calling it when it is a zero-argument getter."""
    candidate = getattr(obj, name, None)
    if candidate is None:
        return None
    if callable(candidate):
        try:
            return candidate()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
    return candidate


def _get_language_model_candidate(model: object) -> object | None:
    """Resolve a nested language model commonly used by VLM wrappers."""
    nested_model = getattr(model, "model", None)
    if nested_model is not None:
        language_model = getattr(nested_model, "language_model", None)
        if language_model is not None:
            return language_model
    return getattr(model, "language_model", None)


def _get_cache_mxq_model(model: object) -> object | None:
    """Resolve the qbruntime model used by ``MobilintCache`` for decode."""
    for candidate in (model, _get_language_model_candidate(model)):
        if candidate is None:
            continue
        for getter_name in ("get_cache_mxq_model", "get_mxq_model"):
            mxq_model = _call_maybe_getter(candidate, getter_name)
            if mxq_model is not None:
                return mxq_model
    return None


def _is_mobilint_npu_model(model: object) -> bool:
    """Return whether a model or its language model is Mobilint NPU-backed."""
    if hasattr(model, "npu_backend"):
        return True
    language_model = _get_language_model_candidate(model)
    return language_model is not None and hasattr(language_model, "npu_backend")


def _supports_fake_decode_prefill(model: object) -> bool:
    """Return whether decode TPS can use fake prefilled Mobilint cache state."""
    return _is_mobilint_npu_model(model) and _get_cache_mxq_model(model) is not None


def _resolve_config_vocab_size(config) -> int:
    """Resolve vocabulary size from text-only or vision-language model configs.

    Args:
        config: Model configuration object. VLM configs may keep language settings under
            ``text_config`` instead of exposing them at the top level.

    Returns:
        Vocabulary size used for synthetic text token generation.

    Raises:
        AttributeError: If neither ``config.vocab_size`` nor ``config.text_config.vocab_size`` exists.
        ValueError: If the resolved vocabulary size is not positive.
    """
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is None:
        text_config = getattr(config, "text_config", None)
        vocab_size = getattr(text_config, "vocab_size", None)
    if vocab_size is None:
        raise AttributeError("Model config must define vocab_size or text_config.vocab_size.")

    vocab_size = int(vocab_size)
    if vocab_size <= 0:
        raise ValueError(f"Model vocab_size must be positive, got {vocab_size}.")
    return vocab_size


def _resolve_image_features_tensor(image_features) -> torch.Tensor:
    """Resolve image embeddings from Tensor, tuple, or structured HF vision outputs.

    Args:
        image_features: Vision encoder output returned by a VLM family.

    Returns:
        Image feature tensor used to replace image placeholder token embeddings.

    Raises:
        TypeError: If a tensor image feature cannot be resolved.
    """
    if isinstance(image_features, torch.Tensor):
        return image_features

    def _resolve_tensor_sequence(features: list | tuple) -> torch.Tensor | None:
        tensors = [feature for feature in features if isinstance(feature, torch.Tensor)]
        if not tensors:
            return None
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=0)

    pooler_output = getattr(image_features, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output
    if isinstance(pooler_output, (list, tuple)):
        tensor = _resolve_tensor_sequence(pooler_output)
        if tensor is not None:
            return tensor

    if isinstance(image_features, (list, tuple)):
        tensor = _resolve_tensor_sequence(image_features)
        if tensor is not None:
            return tensor
        for feature in image_features:
            pooler_output = getattr(feature, "pooler_output", None)
            if isinstance(pooler_output, torch.Tensor):
                return pooler_output
            if isinstance(pooler_output, (list, tuple)):
                tensor = _resolve_tensor_sequence(pooler_output)
                if tensor is not None:
                    return tensor

    raise TypeError(f"Could not resolve image feature tensor from {type(image_features).__name__}.")


class TokenIteratorStreamer(TextIteratorStreamer):
    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TokenIteratorStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        for _ in value.tolist():
            self.text_queue.put("", timeout=self.timeout)

    def end(self):
        self.next_tokens_are_prompt = True
        self.text_queue.put(self.stop_signal, timeout=self.timeout)


@dataclass
class SingleMeasurement:
    num_prefill: int
    num_decode: int
    prefill_latency: float  # seconds
    prefill_tps: float      # tokens/sec
    decode_duration: float  # seconds
    decode_tps: float       # tokens/sec
    total_time: float       # seconds
    avg_total_prefill_token_latency: float  # seconds
    avg_npu_prefill_token_latency: Optional[float]  # seconds
    avg_total_decode_token_latency: float  # seconds
    avg_npu_decode_token_latency: Optional[float]  # seconds
    prefill_npu_latency_pct: Optional[float] = None
    decode_npu_latency_pct: Optional[float] = None
    total_npu_latency_pct: Optional[float] = None
    npu_prefill_time: Optional[float] = None
    npu_decode_time: Optional[float] = None
    avg_power_w: Optional[float] = None
    p99_power_w: Optional[float] = None
    prefill_avg_power_w: Optional[float] = None
    prefill_p99_power_w: Optional[float] = None
    decode_avg_power_w: Optional[float] = None
    decode_p99_power_w: Optional[float] = None
    avg_utilization_pct: Optional[float] = None
    p99_utilization_pct: Optional[float] = None
    prefill_avg_utilization_pct: Optional[float] = None
    prefill_p99_utilization_pct: Optional[float] = None
    decode_avg_utilization_pct: Optional[float] = None
    decode_p99_utilization_pct: Optional[float] = None
    avg_temperature_c: Optional[float] = None
    p99_temperature_c: Optional[float] = None
    prefill_avg_temperature_c: Optional[float] = None
    prefill_p99_temperature_c: Optional[float] = None
    decode_avg_temperature_c: Optional[float] = None
    decode_p99_temperature_c: Optional[float] = None
    avg_memory_used_mb: Optional[float] = None
    p99_memory_used_mb: Optional[float] = None
    prefill_avg_memory_used_mb: Optional[float] = None
    prefill_p99_memory_used_mb: Optional[float] = None
    decode_avg_memory_used_mb: Optional[float] = None
    decode_p99_memory_used_mb: Optional[float] = None
    total_memory_mb: Optional[float] = None
    avg_memory_used_pct: Optional[float] = None
    p99_memory_used_pct: Optional[float] = None
    prefill_avg_memory_used_pct: Optional[float] = None
    prefill_p99_memory_used_pct: Optional[float] = None
    decode_avg_memory_used_pct: Optional[float] = None
    decode_p99_memory_used_pct: Optional[float] = None
    total_energy_j: Optional[float] = None
    prefill_tokens_per_j: Optional[float] = None
    prefill_j_per_token: Optional[float] = None
    decode_tokens_per_j: Optional[float] = None
    decode_j_per_token: Optional[float] = None
    total_tokens_per_j: Optional[float] = None
    total_j_per_token: Optional[float] = None
    decode_prefill_mode: str = "real"

@dataclass
class SweepData:
    x_values: List[int] = field(default_factory=list)      # Token Counts
    tps_values: List[float] = field(default_factory=list)  # TPS
    time_values: List[float] = field(default_factory=list) # Latency/Duration
    avg_total_token_latency_values: List[Optional[float]] = field(default_factory=list)
    avg_npu_token_latency_values: List[Optional[float]] = field(default_factory=list)

@dataclass
class BenchmarkResult:
    prefill_sweep: SweepData = field(default_factory=SweepData)
    decode_sweep: SweepData = field(default_factory=SweepData)
    decode_prefill_modes: List[str] = field(default_factory=list)
    prefill_phase_duration_s: Optional[float] = None
    decode_phase_duration_s: Optional[float] = None

    @staticmethod
    def iter_rows(model_id: str, result: "BenchmarkResult") -> Iterable[dict[str, Union[float, int, str, None]]]:
        for x, tps, t, avg_total, avg_npu in zip(
            result.prefill_sweep.x_values,
            result.prefill_sweep.tps_values,
            result.prefill_sweep.time_values,
            result.prefill_sweep.avg_total_token_latency_values,
            result.prefill_sweep.avg_npu_token_latency_values,
        ):
            yield {
                "model": model_id,
                "phase": "prefill",
                "tokens": x,
                "tps": tps,
                "time_ms": t * 1000.0,
                "avg_total_token_latency_ms": avg_total * 1000.0 if avg_total is not None else None,
                "avg_npu_token_latency_ms": avg_npu * 1000.0 if avg_npu is not None else None,
                "avg_npu_token_latency_pct": npu_latency_pct(avg_total, avg_npu),
            }
        decode_prefill_modes = result.decode_prefill_modes or [None] * len(result.decode_sweep.x_values)
        for x, tps, t, avg_total, avg_npu, prefill_mode in zip(
            result.decode_sweep.x_values,
            result.decode_sweep.tps_values,
            result.decode_sweep.time_values,
            result.decode_sweep.avg_total_token_latency_values,
            result.decode_sweep.avg_npu_token_latency_values,
            decode_prefill_modes,
        ):
            yield {
                "model": model_id,
                "phase": "decode",
                "tokens": x,
                "tps": tps,
                "time_ms": t * 1000.0,
                "avg_total_token_latency_ms": avg_total * 1000.0 if avg_total is not None else None,
                "avg_npu_token_latency_ms": avg_npu * 1000.0 if avg_npu is not None else None,
                "avg_npu_token_latency_pct": npu_latency_pct(avg_total, avg_npu),
                "decode_prefill_mode": prefill_mode,
            }

    @staticmethod
    def write_combined_csv(
        path: str, rows: Iterable[dict[str, Union[float, int, str, None]]]
    ) -> None:
        import csv

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "phase",
                    "tokens",
                    "tps",
                    "time_ms",
                    "avg_total_token_latency_ms",
                    "avg_npu_token_latency_ms",
                    "avg_npu_token_latency_pct",
                    "decode_prefill_mode",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({k: ("" if v is None else v) for k, v in row.items()})

    @staticmethod
    def write_combined_markdown(
        path: str, rows: Iterable[dict[str, Union[float, int, str, None]]]
    ) -> None:
        row_list = list(rows)
        model_ids = sorted({str(row["model"]) for row in row_list})
        prefill_tokens = sorted(
            {
                int(row["tokens"])
                for row in row_list
                if str(row["phase"]) == "prefill"
            }
        )
        decode_tokens = sorted(
            {
                int(row["tokens"])
                for row in row_list
                if str(row["phase"]) == "decode"
            }
        )

        tps_map: dict[tuple[str, str, int], float] = {}
        avg_total_map: dict[tuple[str, str, int], Optional[float]] = {}
        avg_npu_map: dict[tuple[str, str, int], Optional[float]] = {}
        avg_npu_pct_map: dict[tuple[str, str, int], Optional[float]] = {}
        for row in row_list:
            model = str(row["model"])
            phase = str(row["phase"])
            tokens = int(row["tokens"])
            key = (model, phase, tokens)
            tps_value = row.get("tps")
            if tps_value is not None:
                tps_map[key] = float(tps_value)
            avg_total_map[key] = row.get("avg_total_token_latency_ms")
            avg_npu_map[key] = row.get("avg_npu_token_latency_ms")
            avg_npu_pct = row.get("avg_npu_token_latency_pct")
            avg_npu_pct_map[key] = float(avg_npu_pct) if avg_npu_pct is not None else None

        lines = [
            "<table>\n",
            "  <thead>\n",
            "    <tr>\n",
            '      <th rowspan="2">model</th>\n',
            f'      <th colspan="{len(prefill_tokens)}">prefill TPS</th>\n',
            f'      <th colspan="{len(decode_tokens)}">decode TPS</th>\n',
            "    </tr>\n",
            "    <tr>\n",
        ]
        for token in prefill_tokens:
            lines.append(f"      <th>{token}</th>\n")
        for token in decode_tokens:
            lines.append(f"      <th>{token}</th>\n")
        lines.extend(
            [
                "    </tr>\n",
                "  </thead>\n",
                "  <tbody>\n",
            ]
        )

        sort_token = decode_tokens[-1] if decode_tokens else None
        if sort_token is not None:
            model_ids = sorted(
                model_ids,
                key=lambda m: tps_map.get((m, "decode", sort_token), float("-inf")),
                reverse=True,
            )

        for model_id in model_ids:
            lines.append("    <tr>\n")
            lines.append(f"      <td>{model_id}</td>\n")
            for token in prefill_tokens:
                tps = tps_map.get((model_id, "prefill", token))
                avg_total = avg_total_map.get((model_id, "prefill", token))
                avg_npu = avg_npu_map.get((model_id, "prefill", token))
                if (
                    tps is not None
                    and avg_total is not None
                    and avg_total > 0
                    and avg_npu is not None
                ):
                    cell = f"{tps:.4f} ({npu_latency_pct(avg_total, avg_npu):.1f}%)"
                elif tps is not None:
                    cell = f"{tps:.4f}"
                else:
                    cell = ""
                lines.append(f"      <td>{cell}</td>\n")
            for token in decode_tokens:
                tps = tps_map.get((model_id, "decode", token))
                avg_total = avg_total_map.get((model_id, "decode", token))
                avg_npu = avg_npu_map.get((model_id, "decode", token))
                if (
                    tps is not None
                    and avg_total is not None
                    and avg_total > 0
                    and avg_npu is not None
                ):
                    cell = f"{tps:.4f} ({npu_latency_pct(avg_total, avg_npu):.1f}%)"
                elif tps is not None:
                    cell = f"{tps:.4f}"
                else:
                    cell = ""
                lines.append(f"      <td>{cell}</td>\n")
            lines.append("    </tr>\n")

        lines.extend(["  </tbody>\n", "</table>\n"])

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        def _write_avg_table(
            title: str,
            value_map: dict[tuple[str, str, int], Optional[float]],
            phase: str,
            tokens: list[int],
            fmt: str = "{:.6f}",
        ) -> None:
            if not tokens:
                return
            table_lines = [
                "\n<table>\n",
                "  <thead>\n",
                "    <tr>\n",
                '      <th rowspan="2">model</th>\n',
                f'      <th colspan="{len(tokens)}">{title}</th>\n',
                "    </tr>\n",
                "    <tr>\n",
            ]
            for token in tokens:
                table_lines.append(f"      <th>{token}</th>\n")
            table_lines.extend(
                [
                    "    </tr>\n",
                    "  </thead>\n",
                    "  <tbody>\n",
                ]
            )
            for model_id in model_ids:
                table_lines.append("    <tr>\n")
                table_lines.append(f"      <td>{model_id}</td>\n")
                for token in tokens:
                    value = value_map.get((model_id, phase, token))
                    cell = fmt.format(value) if value is not None else ""
                    table_lines.append(f"      <td>{cell}</td>\n")
                table_lines.append("    </tr>\n")
            table_lines.extend(["  </tbody>\n", "</table>\n"])
            with open(path, "a", encoding="utf-8") as f:
                f.writelines(table_lines)

        _write_avg_table("prefill avg total token latency (ms)", avg_total_map, "prefill", prefill_tokens)
        _write_avg_table("prefill avg npu token latency (ms)", avg_npu_map, "prefill", prefill_tokens)
        _write_avg_table("prefill avg npu token latency (%)", avg_npu_pct_map, "prefill", prefill_tokens)
        _write_avg_table("decode avg total token latency (ms)", avg_total_map, "decode", decode_tokens)
        _write_avg_table("decode avg npu token latency (ms)", avg_npu_map, "decode", decode_tokens)
        _write_avg_table("decode avg npu token latency (%)", avg_npu_pct_map, "decode", decode_tokens)

class TPSMeasurer:
    def __init__(self, pipeline):
        self.model = pipeline.model
        self.tokenizer = pipeline.tokenizer
        self.device = self.model.device
        self.model.eval()
        
        plt.switch_backend('Agg') 

    def _supports_npu_timing(self) -> bool:
        return hasattr(self.model, "npu_backend")

    @staticmethod
    def _start_trace(trace_path: Union[str, None]):
        if not trace_path:
            return None
        try:
            import qbruntime  # type: ignore
        except Exception as e:
            raise RuntimeError("Tracing requires qbruntime to be available.") from e
        qbruntime.start_tracing_events(trace_path)
        return qbruntime

    @staticmethod
    def _stop_trace(handle):
        if handle is None:
            return
        handle.stop_tracing_events()

    def measure(
        self,
        num_prefill=512,
        num_decode=128,
        prefill_chunk_size: Optional[int] = None,
        trace_path: Union[str, None] = None,
        show_progress: bool = False,
        progress_desc: Union[str, None] = None,
        on_prefill_start: Optional[Callable[[], None]] = None,
        on_prefill_end: Optional[Callable[[], None]] = None,
        on_decode_start: Optional[Callable[[], None]] = None,
        on_decode_end: Optional[Callable[[], None]] = None,
    ) -> SingleMeasurement:
        trace_handle = self._start_trace(trace_path)
        try:
            assert num_prefill > 0, "num_prefill should be positive! num_prefill: %d" % num_prefill
            assert num_decode > 0, "num_decode should be positive! num_decode: %d" % num_decode

            # 1. Synthetic Input
            vocab_size = _resolve_config_vocab_size(self.model.config)
            low = 100 if vocab_size > 100 else 0
            input_ids = torch.randint(low, vocab_size, (1, num_prefill))
            input_ids = input_ids.to(self.device)

            # 2. Setup
            streamer = TokenIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(
                input_ids=input_ids,
                streamer=streamer,
                min_new_tokens=num_decode + 1,
                max_new_tokens=num_decode + 1,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            if prefill_chunk_size is not None:
                gen_kwargs["prefill_chunk_size"] = int(prefill_chunk_size)
            if self._supports_npu_timing():
                gen_kwargs["count_npu_time"] = True

            # 3. Execution
            thread_error: list[Exception] = []

            def _run_generate():
                try:
                    self.model.generate(**gen_kwargs)
                except Exception as e:
                    thread_error.append(e)
                    try:
                        streamer.end()
                    except Exception:
                        pass

            thread = Thread(target=_run_generate)
            
            t_start = time.perf_counter()
            if on_prefill_start is not None:
                on_prefill_start()
            thread.start()
            
            first_token_time = None
            decoded_tokens = 0
            npu_prefill_time = 0.0
            npu_decode_time = 0.0
            has_npu_time = False

            token_pbar = None
            if show_progress:
                token_pbar = tqdm(
                    total=num_decode + 1,
                    desc=progress_desc or f"generate (prefill={num_prefill}, decode={num_decode})",
                    leave=False,
                )

            stream_error: Exception | None = None
            try:
                for _ in streamer:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                        if on_prefill_end is not None:
                            on_prefill_end()
                        if on_decode_start is not None:
                            on_decode_start()
                    decoded_tokens += 1
                    if token_pbar is not None:
                        token_pbar.update(1)
                    npu_time = getattr(self.model, "npu_time", None)
                    if npu_time is not None:
                        has_npu_time = True
                        if decoded_tokens == 1:
                            npu_prefill_time += npu_time
                        else:
                            npu_decode_time += npu_time
            except Exception as e:
                stream_error = e
            t_end = time.perf_counter()
            thread.join()
            if thread_error:
                raise RuntimeError(f"generate failed: {thread_error[0]}") from thread_error[0]
            if stream_error is not None:
                raise stream_error
            if on_decode_end is not None:
                on_decode_end()
            if token_pbar is not None:
                token_pbar.close()
            
            if first_token_time is None:
                raise RuntimeError("Generation ended before the first token.")

            # 4. Calculation
            prefill_latency = first_token_time - t_start
            prefill_tps = num_prefill / prefill_latency if prefill_latency > 0 else 0
            
            decode_duration = t_end - first_token_time
            decode_tps = (decoded_tokens - 1) / decode_duration if decode_duration > 0 else 0
            
            total_time = t_end - t_start

            decode_count = max(decoded_tokens - 1, 0)
            avg_total_prefill_token_latency = prefill_latency / num_prefill if num_prefill > 0 else 0
            avg_total_decode_token_latency = decode_duration / decode_count if decode_count > 0 else 0
            avg_npu_prefill_token_latency = (
                npu_prefill_time / num_prefill if has_npu_time and num_prefill > 0 else None
            )
            avg_npu_decode_token_latency = (
                npu_decode_time / decode_count if has_npu_time and decode_count > 0 else None
            )
            total_npu_time = (npu_prefill_time + npu_decode_time) if has_npu_time else None

            return SingleMeasurement(
                num_prefill=num_prefill,
                num_decode=num_decode,
                prefill_latency=prefill_latency,
                prefill_tps=prefill_tps,
                decode_duration=decode_duration,
                decode_tps=decode_tps,
                total_time=total_time,
                avg_total_prefill_token_latency=avg_total_prefill_token_latency,
                avg_npu_prefill_token_latency=avg_npu_prefill_token_latency,
                avg_total_decode_token_latency=avg_total_decode_token_latency,
                avg_npu_decode_token_latency=avg_npu_decode_token_latency,
                prefill_npu_latency_pct=npu_latency_pct(
                    avg_total_prefill_token_latency,
                    avg_npu_prefill_token_latency,
                ),
                decode_npu_latency_pct=npu_latency_pct(
                    avg_total_decode_token_latency,
                    avg_npu_decode_token_latency,
                ),
                total_npu_latency_pct=npu_latency_pct(total_time, total_npu_time),
                npu_prefill_time=npu_prefill_time if has_npu_time else None,
                npu_decode_time=npu_decode_time if has_npu_time else None,
            )
        finally:
            self._stop_trace(trace_handle)

    def measure_decode_with_fake_prefill(
        self,
        cache_len: int,
        num_decode: int,
        trace_path: Union[str, None] = None,
        show_progress: bool = False,
        progress_desc: Union[str, None] = None,
    ) -> SingleMeasurement:
        """Measure decode TPS after faking a prefilled Mobilint cache length."""
        trace_handle = self._start_trace(trace_path)
        try:
            assert cache_len > 0, "cache_len should be positive! cache_len: %d" % cache_len
            assert num_decode > 0, "num_decode should be positive! num_decode: %d" % num_decode

            mxq_model = _get_cache_mxq_model(self.model)
            if mxq_model is None:
                raise RuntimeError("Fake decode prefill requires a Mobilint cache MXQ model.")

            vocab_size = _resolve_config_vocab_size(self.model.config)
            low = 100 if vocab_size > 100 else 0
            input_ids = torch.randint(low, vocab_size, (1, 1), device=self.device)
            past_key_values = MobilintCache(cast(Any, mxq_model), batch_size=1)
            past_key_values.fake_prefill(cache_len)

            streamer = TokenIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(
                input_ids=input_ids,
                past_key_values=past_key_values,
                streamer=streamer,
                min_new_tokens=num_decode,
                max_new_tokens=num_decode,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            if self._supports_npu_timing():
                gen_kwargs["count_npu_time"] = True

            thread_error: list[Exception] = []

            def _run_generate():
                try:
                    self.model.generate(**gen_kwargs)
                except Exception as e:
                    thread_error.append(e)
                    try:
                        streamer.end()
                    except Exception:
                        pass

            thread = Thread(target=_run_generate)
            t_start = time.perf_counter()
            thread.start()

            decoded_tokens = 0
            npu_decode_time = 0.0
            has_npu_time = False
            token_pbar = None
            if show_progress:
                token_pbar = tqdm(
                    total=num_decode,
                    desc=progress_desc or f"fake decode (cache={cache_len}, window={num_decode})",
                    leave=False,
                )

            stream_error: Exception | None = None
            try:
                for _ in streamer:
                    decoded_tokens += 1
                    if token_pbar is not None:
                        token_pbar.update(1)
                    npu_time = getattr(self.model, "npu_time", None)
                    if npu_time is not None:
                        has_npu_time = True
                        npu_decode_time += npu_time
            except Exception as e:
                stream_error = e
            t_end = time.perf_counter()
            thread.join()
            if thread_error:
                raise RuntimeError(f"fake decode generate failed: {thread_error[0]}") from thread_error[0]
            if stream_error is not None:
                raise stream_error
            if token_pbar is not None:
                token_pbar.close()

            decode_duration = t_end - t_start
            decode_count = decoded_tokens
            decode_tps = decode_count / decode_duration if decode_duration > 0 else 0.0
            avg_total_decode_token_latency = decode_duration / decode_count if decode_count > 0 else 0.0
            avg_npu_decode_token_latency = (
                npu_decode_time / decode_count if has_npu_time and decode_count > 0 else None
            )
            total_npu_time = npu_decode_time if has_npu_time else None
            return SingleMeasurement(
                num_prefill=cache_len,
                num_decode=decode_count,
                prefill_latency=0.0,
                prefill_tps=0.0,
                decode_duration=decode_duration,
                decode_tps=decode_tps,
                total_time=decode_duration,
                avg_total_prefill_token_latency=0.0,
                avg_npu_prefill_token_latency=0.0 if has_npu_time else None,
                avg_total_decode_token_latency=avg_total_decode_token_latency,
                avg_npu_decode_token_latency=avg_npu_decode_token_latency,
                prefill_npu_latency_pct=None,
                decode_npu_latency_pct=npu_latency_pct(
                    avg_total_decode_token_latency,
                    avg_npu_decode_token_latency,
                ),
                total_npu_latency_pct=npu_latency_pct(decode_duration, total_npu_time),
                npu_prefill_time=0.0 if has_npu_time else None,
                npu_decode_time=npu_decode_time if has_npu_time else None,
                decode_prefill_mode="fake",
            )
        finally:
            self._stop_trace(trace_handle)

    def measure_full(
        self,
        prefill_range: Tuple[int, int, int] = (128, 2048, 128),
        cache_lengths: Optional[Iterable[int]] = None,
        decode_window: int = 128,
        prefill_chunk_size: Optional[int] = None,
        trace_path: Union[str, None] = None,
        show_progress: bool = False,
        progress_prefix: str = "",
        on_prefill_start: Optional[Callable[[], None]] = None,
        on_prefill_end: Optional[Callable[[], None]] = None,
        on_decode_start: Optional[Callable[[], None]] = None,
        on_decode_end: Optional[Callable[[], None]] = None,
    ) -> BenchmarkResult:
        trace_handle = self._start_trace(trace_path)
        try:
            full_result = BenchmarkResult()
            prefix = f"{progress_prefix} " if progress_prefix else ""
            resolved_cache_lengths = list(cache_lengths or [1024, 2048, 4096, 8192])
            t_prefill_start = time.perf_counter()
            if on_prefill_start is not None:
                on_prefill_start()

            # 1. Prefill Sweep
            p_start, p_end, p_step = prefill_range
            prefill_iter = range(p_start, p_end + 1, p_step)
            if show_progress:
                prefill_iter = tqdm(prefill_iter, desc=f"{prefix}prefill sweep", leave=False)

            for p_len in prefill_iter:
                res = self.measure(
                    num_prefill=p_len,
                    num_decode=1,
                    prefill_chunk_size=prefill_chunk_size,
                    show_progress=show_progress,
                    progress_desc=f"{prefix}prefill generate ({p_len})",
                )

                full_result.prefill_sweep.x_values.append(p_len)
                full_result.prefill_sweep.tps_values.append(res.prefill_tps)
                full_result.prefill_sweep.time_values.append(res.prefill_latency)
                full_result.prefill_sweep.avg_total_token_latency_values.append(res.avg_total_prefill_token_latency)
                full_result.prefill_sweep.avg_npu_token_latency_values.append(res.avg_npu_prefill_token_latency)
            t_prefill_end = time.perf_counter()
            if on_prefill_end is not None:
                on_prefill_end()
            full_result.prefill_phase_duration_s = max(0.0, t_prefill_end - t_prefill_start)
                
            # 2. Decode Sweep
            t_decode_start = time.perf_counter()
            if on_decode_start is not None:
                on_decode_start()
            decode_iter = resolved_cache_lengths
            if show_progress:
                decode_iter = tqdm(decode_iter, desc=f"{prefix}decode sweep", leave=False)
            use_fake_decode_prefill = _supports_fake_decode_prefill(self.model)
            for cache_len in decode_iter:
                progress_desc = f"{prefix}decode generate (cache={cache_len}, window={decode_window})"
                if use_fake_decode_prefill:
                    res = self.measure_decode_with_fake_prefill(
                        cache_len=cache_len,
                        num_decode=decode_window,
                        show_progress=show_progress,
                        progress_desc=progress_desc,
                    )
                else:
                    res = self.measure(
                        num_prefill=cache_len,
                        num_decode=decode_window,
                        prefill_chunk_size=prefill_chunk_size,
                        show_progress=show_progress,
                        progress_desc=progress_desc,
                    )
                full_result.decode_sweep.x_values.append(cache_len)
                full_result.decode_sweep.tps_values.append(res.decode_tps)
                full_result.decode_sweep.time_values.append(res.decode_duration)
                full_result.decode_sweep.avg_total_token_latency_values.append(res.avg_total_decode_token_latency)
                full_result.decode_sweep.avg_npu_token_latency_values.append(res.avg_npu_decode_token_latency)
                full_result.decode_prefill_modes.append(res.decode_prefill_mode)
            t_decode_end = time.perf_counter()
            if on_decode_end is not None:
                on_decode_end()
            full_result.decode_phase_duration_s = max(0.0, t_decode_end - t_decode_start)

            return full_result
        finally:
            self._stop_trace(trace_handle)

    def plot_and_save(self, result: BenchmarkResult, save_path: str = "tps_benchmark.png"):
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('LLM Performance Benchmark (NPU)', fontsize=16)

        # 1. Prefill: Token vs TPS
        axs[0, 0].plot(result.prefill_sweep.x_values, result.prefill_sweep.tps_values, 'b-o')
        axs[0, 0].set_title('Prefill: Tokens vs TPS (Higher is Better)')
        axs[0, 0].set_xlabel('Input Tokens')
        axs[0, 0].set_ylabel('TPS (tokens/sec)')
        axs[0, 0].grid(True)

        # 2. Prefill: Token vs Latency
        axs[0, 1].plot(
            result.prefill_sweep.x_values,
            [t * 1000.0 for t in result.prefill_sweep.time_values],
            'r-o',
        )
        axs[0, 1].set_title('Prefill: Tokens vs Latency (TTFT)')
        axs[0, 1].set_xlabel('Input Tokens')
        axs[0, 1].set_ylabel('Latency (ms)')
        axs[0, 1].grid(True)

        # 3. Decode: Cache length vs TPS
        axs[1, 0].plot(result.decode_sweep.x_values, result.decode_sweep.tps_values, 'g-o')
        axs[1, 0].set_title('Decode: Cache Length vs TPS')
        axs[1, 0].set_xlabel('Cache Length (tokens)')
        axs[1, 0].set_ylabel('TPS (tokens/sec)')
        axs[1, 0].grid(True)

        # 4. Decode: Cache length vs Time
        axs[1, 1].plot(
            result.decode_sweep.x_values,
            [t * 1000.0 for t in result.decode_sweep.time_values],
            'm-o',
        )
        axs[1, 1].set_title('Decode: Cache Length vs Time (Duration)')
        axs[1, 1].set_xlabel('Cache Length (tokens)')
        axs[1, 1].set_ylabel('Decode Window Duration (ms)')
        axs[1, 1].grid(True)

        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        
        plt.savefig(save_path, dpi=300)
        print(f"\n✅ Graph saved to: {save_path}")
        
        plt.close(fig)

    @staticmethod
    def plot_and_save_results(
        results: List["BenchmarkResult"],
        labels: List[str],
        save_path: str = "tps_benchmark_all.png",
    ):
        fig, axs = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('LLM Performance Benchmark (NPU)', fontsize=16)

        for result, label in zip(results, labels):
            axs[0, 0].plot(result.prefill_sweep.x_values, result.prefill_sweep.tps_values, '-o', label=label)
            axs[0, 1].plot(
                result.prefill_sweep.x_values,
                [t * 1000.0 for t in result.prefill_sweep.time_values],
                '-o',
                label=label,
            )
            axs[1, 0].plot(result.decode_sweep.x_values, result.decode_sweep.tps_values, '-o', label=label)
            axs[1, 1].plot(
                result.decode_sweep.x_values,
                [t * 1000.0 for t in result.decode_sweep.time_values],
                '-o',
                label=label,
            )

        axs[0, 0].set_title('Prefill: Tokens vs TPS (Higher is Better)')
        axs[0, 0].set_xlabel('Input Tokens')
        axs[0, 0].set_ylabel('TPS (tokens/sec)')
        axs[0, 0].grid(True)

        axs[0, 1].set_title('Prefill: Tokens vs Latency (TTFT)')
        axs[0, 1].set_xlabel('Input Tokens')
        axs[0, 1].set_ylabel('Latency (ms)')
        axs[0, 1].grid(True)

        axs[1, 0].set_title('Decode: Cache Length vs TPS')
        axs[1, 0].set_xlabel('Cache Length (tokens)')
        axs[1, 0].set_ylabel('TPS (tokens/sec)')
        axs[1, 0].grid(True)

        axs[1, 1].set_title('Decode: Cache Length vs Time (Duration)')
        axs[1, 1].set_xlabel('Cache Length (tokens)')
        axs[1, 1].set_ylabel('Decode Window Duration (ms)')
        axs[1, 1].grid(True)

        for ax in axs.flat:
            ax.legend()

        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.savefig(save_path, dpi=300)
        print(f"\n✅ Graph saved to: {save_path}")
        plt.close(fig)

@dataclass
class VLMSingleMeasurement:
    image_resolution: int
    vision_encode_latency: float  # seconds/image
    vision_fps: float  # images/sec
    llm: SingleMeasurement


class VLMTPSMeasurer:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.model = pipeline.model
        self.tokenizer = pipeline.tokenizer
        self.processor = getattr(pipeline, "processor", None)
        if self.processor is None:
            raise ValueError("VLM benchmark requires a pipeline with processor.")
        self.model.eval()

    @staticmethod
    def _supports_npu_timing(model) -> bool:
        return hasattr(model, "npu_backend")

    def _build_inputs(self, image_resolution: int, prompt: str):
        image = torch.randint(
            low=0,
            high=256,
            size=(3, image_resolution, image_resolution),
            dtype=torch.uint8,
        )

        text = prompt
        image_token = getattr(self.processor, "image_token", None)
        if isinstance(image_token, str) and image_token:
            if image_token not in text:
                text = f"{image_token}\n{text}"

        processor_kwargs = dict(
            images=image,
            text=text,
            return_tensors="pt",
        )
        try:
            return self.processor(**processor_kwargs)
        except ValueError as e:
            msg = str(e).lower()
            if "placeholder" not in msg or "image" not in msg:
                raise
            image_token = getattr(self.processor, "image_token", "<image>")
            if image_token not in processor_kwargs["text"]:
                processor_kwargs["text"] = f"{image_token}\n{processor_kwargs['text']}"
            return self.processor(**processor_kwargs)

    def _get_language_model(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            return self.model.model.language_model
        if hasattr(self.model, "language_model"):
            return self.model.language_model
        raise ValueError("Could not find language_model from VLM model.")

    def _measure_vision_encode(self, inputs: dict):
        pixel_values = inputs["pixel_values"].to(self.model.device)

        with torch.no_grad():
            t0 = time.perf_counter()
            if (
                "image_grid_thw" in inputs
                and hasattr(self.model, "model")
                and hasattr(self.model.model, "get_image_features")
            ):
                image_grid_thw = inputs["image_grid_thw"].to(self.model.device)
                image_features = self.model.model.get_image_features(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
            else:
                image_features = self.model.get_image_features(pixel_values=pixel_values)
            t1 = time.perf_counter()

        return t1 - t0, _resolve_image_features_tensor(image_features)

    def _build_inputs_embeds(self, inputs: dict, image_features: torch.Tensor) -> torch.Tensor:
        image_features = _resolve_image_features_tensor(image_features)
        input_ids = inputs["input_ids"].to(self.model.device)
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            inputs_embeds = self.model.model.language_model.get_input_embeddings()(input_ids)

            image_token_id = self.model.config.image_token_id
            image_mask = input_ids == image_token_id
            n_image_tokens = int(image_mask.sum().item())

            hidden_size = int(inputs_embeds.shape[-1])
            # Different VLM families return image features with different layouts.
            # Normalize to [num_image_tokens_total, hidden_size] before masked_scatter.
            if image_features.ndim == 4:
                # e.g. [batch, h, w, hidden] or [batch, hidden, h, w]
                if int(image_features.shape[-1]) == hidden_size:
                    image_features = image_features.reshape(-1, hidden_size)
                elif int(image_features.shape[1]) == hidden_size:
                    image_features = image_features.permute(0, 2, 3, 1).contiguous()
                    image_features = image_features.reshape(-1, hidden_size)
                else:
                    raise AssertionError(
                        "Unexpected 4D image feature layout: "
                        f"{tuple(int(x) for x in image_features.shape)}"
                    )
            elif image_features.ndim == 3:
                if int(image_features.shape[-1]) == hidden_size:
                    image_features = image_features.reshape(-1, hidden_size)
                else:
                    raise AssertionError(
                        "Unexpected 3D image feature layout: "
                        f"{tuple(int(x) for x in image_features.shape)}"
                    )
            elif image_features.ndim == 2:
                if int(image_features.shape[-1]) == hidden_size:
                    pass
                elif int(image_features.shape[0]) == hidden_size:
                    image_features = image_features.transpose(0, 1).contiguous()
                else:
                    raise AssertionError(
                        "Unexpected 2D image feature layout: "
                        f"{tuple(int(x) for x in image_features.shape)}"
                    )
            else:
                raise AssertionError(
                    "Unsupported image feature rank: "
                    f"{image_features.ndim} for shape "
                    f"{tuple(int(x) for x in image_features.shape)}"
                )

            assert n_image_tokens == int(image_features.shape[0]), (
                "Image token count does not match image features after normalization: "
                f"{n_image_tokens} vs {int(image_features.shape[0])}, "
                f"feature_shape={tuple(int(x) for x in image_features.shape)}"
            )

            image_features = image_features.to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            return inputs_embeds.masked_scatter(image_mask, image_features)

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        image_features = image_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        special_image_mask = self.model.get_placeholder_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
        )
        return inputs_embeds.masked_scatter(special_image_mask, image_features)

    def _measure_llm_once(
        self,
        inputs_embeds: torch.Tensor,
        num_decode: int,
        prefill_chunk_size: Optional[int] = None,
    ) -> SingleMeasurement:
        seq_len = int(inputs_embeds.shape[1])
        lm_for_npu = self._get_language_model()
        gen_model = self.model

        streamer = TokenIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        attention_mask = torch.ones(
            (inputs_embeds.shape[0], inputs_embeds.shape[1]),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        seq_len_tensor = int(inputs_embeds.shape[1])
        batch_size = int(inputs_embeds.shape[0])
        cache_position = torch.arange(seq_len_tensor, device=inputs_embeds.device)
        position_ids = cache_position.view(1, -1).expand(batch_size, -1)
        gen_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            streamer=streamer,
            min_new_tokens=num_decode + 1,
            max_new_tokens=num_decode + 1,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if prefill_chunk_size is not None:
            gen_kwargs["prefill_chunk_size"] = int(prefill_chunk_size)
        if self._supports_npu_timing(lm_for_npu) or self._supports_npu_timing(gen_model):
            gen_kwargs["count_npu_time"] = True

        thread_error: list[Exception] = []

        def _run_generate():
            try:
                gen_model.generate(**gen_kwargs)
            except Exception as e:
                thread_error.append(e)
                streamer.end()

        thread = Thread(target=_run_generate)
        t_start = time.perf_counter()
        thread.start()

        first_token_time = None
        decoded_tokens = 0
        npu_prefill_time = 0.0
        npu_decode_time = 0.0
        has_npu_time = False

        for _ in streamer:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            decoded_tokens += 1
            npu_time = getattr(lm_for_npu, "npu_time", None)
            if npu_time is not None:
                has_npu_time = True
                if decoded_tokens == 1:
                    npu_prefill_time += npu_time
                else:
                    npu_decode_time += npu_time

        t_end = time.perf_counter()
        thread.join()
        if thread_error:
            raise RuntimeError(f"VLM LLM-phase generate failed: {thread_error[0]}") from thread_error[0]
        assert first_token_time is not None

        prefill_latency = first_token_time - t_start
        prefill_tps = seq_len / prefill_latency if prefill_latency > 0 else 0.0

        decode_duration = t_end - first_token_time
        decode_count = max(decoded_tokens - 1, 0)
        decode_tps = decode_count / decode_duration if decode_duration > 0 else 0.0
        total_time = t_end - t_start

        avg_total_prefill_token_latency = prefill_latency / seq_len if seq_len > 0 else 0.0
        avg_total_decode_token_latency = decode_duration / decode_count if decode_count > 0 else 0.0
        avg_npu_prefill_token_latency = (
            npu_prefill_time / seq_len if has_npu_time and seq_len > 0 else None
        )
        avg_npu_decode_token_latency = (
            npu_decode_time / decode_count if has_npu_time and decode_count > 0 else None
        )
        total_npu_time = (npu_prefill_time + npu_decode_time) if has_npu_time else None

        return SingleMeasurement(
            num_prefill=seq_len,
            num_decode=decode_count,
            prefill_latency=prefill_latency,
            prefill_tps=prefill_tps,
            decode_duration=decode_duration,
            decode_tps=decode_tps,
            total_time=total_time,
            avg_total_prefill_token_latency=avg_total_prefill_token_latency,
            avg_npu_prefill_token_latency=avg_npu_prefill_token_latency,
            avg_total_decode_token_latency=avg_total_decode_token_latency,
            avg_npu_decode_token_latency=avg_npu_decode_token_latency,
            prefill_npu_latency_pct=npu_latency_pct(avg_total_prefill_token_latency, avg_npu_prefill_token_latency),
            decode_npu_latency_pct=npu_latency_pct(avg_total_decode_token_latency, avg_npu_decode_token_latency),
            total_npu_latency_pct=npu_latency_pct(total_time, total_npu_time),
            npu_prefill_time=npu_prefill_time if has_npu_time else None,
            npu_decode_time=npu_decode_time if has_npu_time else None,
        )

    def _measure_llm_decode_with_fake_prefill(self, cache_len: int, num_decode: int) -> SingleMeasurement:
        """Measure VLM language-model decode TPS with fake prefilled cache length."""
        assert cache_len > 0, "cache_len should be positive! cache_len: %d" % cache_len
        assert num_decode > 0, "num_decode should be positive! num_decode: %d" % num_decode

        lm_for_npu = self._get_language_model()
        gen_model = self.model
        mxq_model = _get_cache_mxq_model(lm_for_npu)
        if mxq_model is None:
            raise RuntimeError("Fake VLM LLM decode prefill requires a Mobilint cache MXQ model.")

        device = getattr(lm_for_npu, "device", self.model.device)
        vocab_size = _resolve_config_vocab_size(self.model.config)
        low = 100 if vocab_size > 100 else 0
        input_ids = torch.randint(low, vocab_size, (1, 1), device=device)
        inputs_embeds = lm_for_npu.get_input_embeddings()(input_ids)
        cache_factory = getattr(lm_for_npu, "_get_cache", None) or getattr(gen_model, "_get_cache", None)
        if callable(cache_factory):
            past_key_values = cache_factory("mobilint", 1, cache_len)
        else:
            past_key_values = MobilintCache(cast(Any, mxq_model), batch_size=1)
        past_key_values.fake_prefill(cache_len)

        count_npu_time = self._supports_npu_timing(lm_for_npu)
        t_start = time.perf_counter()
        decoded_tokens = 0
        npu_decode_time = 0.0
        has_npu_time = False

        with torch.no_grad():
            for decode_idx in range(num_decode):
                current_cache_len = cache_len + decode_idx
                attention_mask = torch.ones((1, current_cache_len + 1), dtype=torch.long, device=device)
                position_ids = torch.full((1, 1), current_cache_len, dtype=torch.long, device=device)
                cache_position = torch.tensor([current_cache_len], dtype=torch.long, device=device)

                outputs = lm_for_npu(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                    count_npu_time=count_npu_time,
                )
                logits = outputs.last_hidden_state
                next_token_id = torch.argmax(logits.reshape(-1, logits.shape[-1])[-1], dim=-1).view(1, 1)
                inputs_embeds = lm_for_npu.get_input_embeddings()(next_token_id.to(device))

                decoded_tokens += 1
                npu_time = getattr(lm_for_npu, "npu_time", None)
                if npu_time is not None:
                    has_npu_time = True
                    npu_decode_time += npu_time

        t_end = time.perf_counter()

        decode_duration = t_end - t_start
        decode_count = decoded_tokens
        decode_tps = decode_count / decode_duration if decode_duration > 0 else 0.0
        avg_total_decode_token_latency = decode_duration / decode_count if decode_count > 0 else 0.0
        avg_npu_decode_token_latency = (
            npu_decode_time / decode_count if has_npu_time and decode_count > 0 else None
        )
        total_npu_time = npu_decode_time if has_npu_time else None

        return SingleMeasurement(
            num_prefill=cache_len,
            num_decode=decode_count,
            prefill_latency=0.0,
            prefill_tps=0.0,
            decode_duration=decode_duration,
            decode_tps=decode_tps,
            total_time=decode_duration,
            avg_total_prefill_token_latency=0.0,
            avg_npu_prefill_token_latency=0.0 if has_npu_time else None,
            avg_total_decode_token_latency=avg_total_decode_token_latency,
            avg_npu_decode_token_latency=avg_npu_decode_token_latency,
            prefill_npu_latency_pct=None,
            decode_npu_latency_pct=npu_latency_pct(avg_total_decode_token_latency, avg_npu_decode_token_latency),
            total_npu_latency_pct=npu_latency_pct(decode_duration, total_npu_time),
            npu_prefill_time=0.0 if has_npu_time else None,
            npu_decode_time=npu_decode_time if has_npu_time else None,
            decode_prefill_mode="fake",
        )

    def _build_reference_inputs_embeds(
        self,
        image_resolution: int,
        prompt: str,
    ) -> torch.Tensor:
        inputs = self._build_inputs(image_resolution=image_resolution, prompt=prompt)
        _, image_features = self._measure_vision_encode(inputs)
        return self._build_inputs_embeds(inputs, image_features=image_features)

    def _build_text_suffix_ids(self, length: int, device: torch.device) -> torch.Tensor:
        if length <= 0:
            return torch.empty((1, 0), dtype=torch.long, device=device)

        vocab_size = _resolve_config_vocab_size(self.model.config)
        low = 100 if vocab_size > 100 else 0
        suffix_ids = torch.randint(low, vocab_size, (1, length), device=device)

        blocked_ids = {
            int(x)
            for x in (
                getattr(self.tokenizer, "bos_token_id", None),
                getattr(self.tokenizer, "eos_token_id", None),
                getattr(self.tokenizer, "pad_token_id", None),
                getattr(self.model.config, "image_token_id", None),
            )
            if x is not None
        }
        if blocked_ids:
            replacement_id = next((idx for idx in range(vocab_size) if idx not in blocked_ids), 0)
            for blocked_id in blocked_ids:
                suffix_ids[suffix_ids == blocked_id] = replacement_id

        return suffix_ids

    def _build_inputs_embeds_from_base(
        self,
        inputs: dict,
        image_features: torch.Tensor,
        total_prefill_len: int,
    ) -> tuple[torch.Tensor | None, int]:
        input_ids = inputs["input_ids"].to(self.model.device)
        base_total_len = int(input_ids.shape[1])
        if total_prefill_len < base_total_len:
            return None, base_total_len

        extra_len = total_prefill_len - base_total_len
        if extra_len > 0:
            suffix_ids = self._build_text_suffix_ids(extra_len, device=input_ids.device)
            input_ids = torch.cat((input_ids, suffix_ids), dim=1)

        adjusted_inputs = dict(inputs)
        adjusted_inputs["input_ids"] = input_ids
        if "attention_mask" in adjusted_inputs:
            adjusted_inputs["attention_mask"] = torch.ones_like(input_ids, device=input_ids.device)

        inputs_embeds = self._build_inputs_embeds(adjusted_inputs, image_features=image_features)
        if int(inputs_embeds.shape[1]) != total_prefill_len:
            raise AssertionError(
                "Adjusted VLM inputs_embeds length does not match target total prefill length: "
                f"{int(inputs_embeds.shape[1])} vs {total_prefill_len}"
            )
        return inputs_embeds, base_total_len

    def measure_llm_full(
        self,
        image_resolution: int,
        prompt: str,
        prefill_range: Tuple[int, int, int] = (128, 2048, 128),
        cache_lengths: Optional[Iterable[int]] = None,
        decode_window: int = 128,
        prefill_chunk_size: Optional[int] = None,
        show_progress: bool = False,
        progress_prefix: str = "",
    ) -> BenchmarkResult:
        full_result = BenchmarkResult()
        prefix = f"{progress_prefix} " if progress_prefix else ""
        resolved_cache_lengths = list(cache_lengths or [1024, 2048, 4096, 8192])
        base_inputs = self._build_inputs(image_resolution=image_resolution, prompt=prompt)
        _, image_features = self._measure_vision_encode(base_inputs)

        p_start, p_end, p_step = prefill_range
        prefill_iter = range(p_start, p_end + 1, p_step)
        if show_progress:
            prefill_iter = tqdm(prefill_iter, desc=f"{prefix}vlm llm prefill sweep", leave=False)

        t_prefill_start = time.perf_counter()
        min_total_len_seen: int | None = None
        for p_len in prefill_iter:
            inputs_embeds, min_total_len = self._build_inputs_embeds_from_base(
                inputs=base_inputs,
                image_features=image_features,
                total_prefill_len=p_len,
            )
            min_total_len_seen = min_total_len if min_total_len_seen is None else min(min_total_len_seen, min_total_len)
            if inputs_embeds is None:
                if show_progress:
                    tqdm.write(
                        f"{prefix}skip prefill target={p_len}: "
                        f"minimum multimodal prefix length is {min_total_len}"
                    )
                continue
            res = self._measure_llm_once(
                inputs_embeds=inputs_embeds,
                num_decode=1,
                prefill_chunk_size=prefill_chunk_size,
            )
            full_result.prefill_sweep.x_values.append(p_len)
            full_result.prefill_sweep.tps_values.append(res.prefill_tps)
            full_result.prefill_sweep.time_values.append(res.prefill_latency)
            full_result.prefill_sweep.avg_total_token_latency_values.append(res.avg_total_prefill_token_latency)
            full_result.prefill_sweep.avg_npu_token_latency_values.append(res.avg_npu_prefill_token_latency)
        t_prefill_end = time.perf_counter()
        full_result.prefill_phase_duration_s = max(0.0, t_prefill_end - t_prefill_start)

        decode_iter = resolved_cache_lengths
        if show_progress:
            decode_iter = tqdm(decode_iter, desc=f"{prefix}vlm llm decode sweep", leave=False)

        t_decode_start = time.perf_counter()
        use_fake_decode_prefill = _supports_fake_decode_prefill(self._get_language_model())
        for cache_len in decode_iter:
            if use_fake_decode_prefill:
                res = self._measure_llm_decode_with_fake_prefill(cache_len=cache_len, num_decode=decode_window)
            else:
                inputs_embeds, min_total_len = self._build_inputs_embeds_from_base(
                    inputs=base_inputs,
                    image_features=image_features,
                    total_prefill_len=cache_len,
                )
                min_total_len_seen = (
                    min_total_len if min_total_len_seen is None else min(min_total_len_seen, min_total_len)
                )
                if inputs_embeds is None:
                    if show_progress:
                        tqdm.write(
                            f"{prefix}skip cache length={cache_len}: "
                            f"minimum multimodal prefix length is {min_total_len}"
                        )
                    continue
                res = self._measure_llm_once(
                    inputs_embeds=inputs_embeds,
                    num_decode=decode_window,
                    prefill_chunk_size=prefill_chunk_size,
                )
            full_result.decode_sweep.x_values.append(cache_len)
            full_result.decode_sweep.tps_values.append(res.decode_tps)
            full_result.decode_sweep.time_values.append(res.decode_duration)
            full_result.decode_sweep.avg_total_token_latency_values.append(res.avg_total_decode_token_latency)
            full_result.decode_sweep.avg_npu_token_latency_values.append(res.avg_npu_decode_token_latency)
            full_result.decode_prefill_modes.append(res.decode_prefill_mode)
        t_decode_end = time.perf_counter()
        full_result.decode_phase_duration_s = max(0.0, t_decode_end - t_decode_start)

        if (
            show_progress
            and min_total_len_seen is not None
            and not full_result.prefill_sweep.x_values
            and not full_result.decode_sweep.x_values
        ):
            tqdm.write(
                f"{prefix}all VLM LLM sweep points were skipped: "
                f"minimum multimodal prefix length is {min_total_len_seen}"
            )

        return full_result

    def measure_vision(
        self,
        image_resolution: int,
        repeat: int,
        prompt: str,
        show_progress: bool = False,
    ) -> list[tuple[float, float]]:
        assert repeat > 0, "repeat must be > 0"
        results: list[tuple[float, float]] = []
        for idx in range(repeat):
            if show_progress:
                print(
                    f"[vlm][vision] resolution={image_resolution} run={idx + 1}/{repeat}: measuring..."
                )
            inputs = self._build_inputs(image_resolution=image_resolution, prompt=prompt)
            vision_encode_latency, _ = self._measure_vision_encode(inputs)
            vision_fps = (1.0 / vision_encode_latency) if vision_encode_latency > 0 else 0.0
            results.append((vision_encode_latency, vision_fps))
            if show_progress:
                print(
                    f"[vlm][vision] resolution={image_resolution} run={idx + 1}/{repeat}: done "
                    f"(vision={vision_encode_latency * 1000.0:.2f}ms, fps={vision_fps:.2f})"
                )
        return results

    def measure_llm(
        self,
        image_resolution: int,
        num_decode: int,
        repeat: int,
        prompt: str,
        show_progress: bool = False,
    ) -> list[SingleMeasurement]:
        assert repeat > 0, "repeat must be > 0"
        inputs_embeds = self._build_reference_inputs_embeds(
            image_resolution=image_resolution,
            prompt=prompt,
        )
        results: list[SingleMeasurement] = []
        for idx in range(repeat):
            if show_progress:
                print(
                    f"[vlm][llm] ref_resolution={image_resolution} run={idx + 1}/{repeat}: measuring..."
                )
            llm = self._measure_llm_once(inputs_embeds=inputs_embeds, num_decode=num_decode)
            results.append(llm)
            if show_progress:
                print(
                    f"[vlm][llm] ref_resolution={image_resolution} run={idx + 1}/{repeat}: done "
                    f"(prefill_tps={llm.prefill_tps:.2f}, decode_tps={llm.decode_tps:.2f})"
                )
        return results

    def measure(
        self,
        image_resolution: int,
        num_decode: int,
        repeat: int,
        prompt: str,
        show_progress: bool = False,
    ) -> list[VLMSingleMeasurement]:
        assert repeat > 0, "repeat must be > 0"
        results: list[VLMSingleMeasurement] = []
        for idx in range(repeat):
            if show_progress:
                print(
                    f"[vlm] resolution={image_resolution} run={idx + 1}/{repeat}: measuring..."
                )
            inputs = self._build_inputs(image_resolution=image_resolution, prompt=prompt)
            vision_encode_latency, image_features = self._measure_vision_encode(inputs)
            inputs_embeds = self._build_inputs_embeds(inputs, image_features=image_features)
            llm = self._measure_llm_once(inputs_embeds=inputs_embeds, num_decode=num_decode)
            results.append(
                VLMSingleMeasurement(
                    image_resolution=image_resolution,
                    vision_encode_latency=vision_encode_latency,
                    vision_fps=(1.0 / vision_encode_latency) if vision_encode_latency > 0 else 0.0,
                    llm=llm,
                )
            )
            if show_progress:
                print(
                    f"[vlm] resolution={image_resolution} run={idx + 1}/{repeat}: done "
                    f"(vision={vision_encode_latency * 1000.0:.2f}ms, "
                    f"prefill_tps={llm.prefill_tps:.2f}, decode_tps={llm.decode_tps:.2f})"
                )
        return results
