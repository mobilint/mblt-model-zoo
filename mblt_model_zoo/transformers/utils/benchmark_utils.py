import time
from dataclasses import dataclass, field
from threading import Thread
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import torch
from transformers import TextIteratorStreamer


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

@dataclass
class SweepData:
    x_values: List[int] = field(default_factory=list)      # Token Counts
    tps_values: List[float] = field(default_factory=list)  # TPS
    time_values: List[float] = field(default_factory=list) # Latency/Duration

@dataclass
class BenchmarkResult:
    prefill_sweep: SweepData = field(default_factory=SweepData)
    decode_sweep: SweepData = field(default_factory=SweepData)

    @staticmethod
    def iter_rows(model_id: str, result: "BenchmarkResult") -> Iterable[dict[str, float | int | str]]:
        for x, tps, t in zip(
            result.prefill_sweep.x_values,
            result.prefill_sweep.tps_values,
            result.prefill_sweep.time_values,
        ):
            yield {
                "model": model_id,
                "phase": "prefill",
                "tokens": x,
                "tps": tps,
                "time_s": t,
            }
        for x, tps, t in zip(
            result.decode_sweep.x_values,
            result.decode_sweep.tps_values,
            result.decode_sweep.time_values,
        ):
            yield {
                "model": model_id,
                "phase": "decode",
                "tokens": x,
                "tps": tps,
                "time_s": t,
            }

    @staticmethod
    def write_combined_csv(
        path: str, rows: Iterable[dict[str, float | int | str]]
    ) -> None:
        import csv

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["model", "phase", "tokens", "tps", "time_s"]
            )
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def write_combined_markdown(
        path: str, rows: Iterable[dict[str, float | int | str]]
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
        for row in row_list:
            key = (str(row["model"]), str(row["phase"]), int(row["tokens"]))
            tps_map[key] = float(row["tps"])

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
                cell = f"{tps:.4f}" if tps is not None else ""
                lines.append(f"      <td>{cell}</td>\n")
            for token in decode_tokens:
                tps = tps_map.get((model_id, "decode", token))
                cell = f"{tps:.4f}" if tps is not None else ""
                lines.append(f"      <td>{cell}</td>\n")
            lines.append("    </tr>\n")

        lines.extend(["  </tbody>\n", "</table>\n"])

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

class TPSMeasurer:
    def __init__(self, pipeline):
        self.model = pipeline.model
        self.tokenizer = pipeline.tokenizer
        self.device = self.model.device
        self.model.eval()
        
        plt.switch_backend('Agg') 

    @staticmethod
    def _start_trace(trace_path: str | None):
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
        trace_path: str | None = None,
    ) -> SingleMeasurement:
        trace_handle = self._start_trace(trace_path)
        try:
            assert num_prefill > 0, "num_prefill should be positive! num_prefill: %d" % num_prefill
            assert num_decode > 0, "num_decode should be positive! num_decode: %d" % num_decode

            # 1. Synthetic Input
            input_ids = torch.randint(100, self.model.config.vocab_size, (1, num_prefill))
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
                pad_token_id=self.tokenizer.eos_token_id
            )

            # 3. Execution
            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            
            t_start = time.perf_counter()
            thread.start()
            
            first_token_time = None
            decoded_tokens = 0
            
            for _ in streamer:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                decoded_tokens += 1
                
            t_end = time.perf_counter()
            thread.join()
            
            assert first_token_time is not None

            # 4. Calculation
            prefill_latency = first_token_time - t_start
            prefill_tps = num_prefill / prefill_latency if prefill_latency > 0 else 0
            
            decode_duration = t_end - first_token_time
            decode_tps = (decoded_tokens - 1) / decode_duration if decode_duration > 0 else 0
            
            total_time = t_end - t_start

            return SingleMeasurement(
                num_prefill=num_prefill,
                num_decode=num_decode,
                prefill_latency=prefill_latency,
                prefill_tps=prefill_tps,
                decode_duration=decode_duration,
                decode_tps=decode_tps,
                total_time=total_time
            )
        finally:
            self._stop_trace(trace_handle)

    def measure_full(self, 
                     prefill_range: Tuple[int, int, int] = (128, 2048, 128), 
                     decode_range: Tuple[int, int, int] = (128, 1024, 128),
                     fixed_decode_len=10, 
                     fixed_prefill_len=128,
                     trace_path: str | None = None) -> BenchmarkResult:
        trace_handle = self._start_trace(trace_path)
        try:
            full_result = BenchmarkResult()

            print(f"🚀 Starting Full Measurement...")

            # 1. Prefill Sweep
            p_start, p_end, p_step = prefill_range
            print(f"--- [1/2] Prefill Sweep ({p_start} ~ {p_end}) ---")
            
            for p_len in range(p_start, p_end + 1, p_step):
                res = self.measure(num_prefill=p_len, num_decode=fixed_decode_len)
                
                full_result.prefill_sweep.x_values.append(p_len)
                full_result.prefill_sweep.tps_values.append(res.prefill_tps)
                full_result.prefill_sweep.time_values.append(res.prefill_latency)
                
                print(f"In: {p_len} | TPS: {res.prefill_tps:.1f} | Latency: {res.prefill_latency:.4f}s")

            # 2. Decode Sweep
            d_start, d_end, d_step = decode_range
            print(f"--- [2/2] Decode Sweep ({d_start} ~ {d_end}) ---")
            input_ids = torch.randint(100, self.model.config.vocab_size, (1, fixed_prefill_len))
            input_ids = input_ids.to(self.device)

            streamer = TokenIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(
                input_ids=input_ids,
                streamer=streamer,
                min_new_tokens=d_end,
                max_new_tokens=d_end,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=self.tokenizer.eos_token_id
            )

            targets = set(range(d_start, d_end + 1, d_step))

            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()

            first_token_time = None
            decoded_tokens = 0
            for _ in streamer:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                decoded_tokens += 1
                if decoded_tokens in targets:
                    t_hit = time.perf_counter()
                    decode_duration = t_hit - first_token_time
                    decode_count = decoded_tokens - 1
                    decode_tps = decode_count / decode_duration if decode_duration > 0 else 0

                    full_result.decode_sweep.x_values.append(decoded_tokens)
                    full_result.decode_sweep.tps_values.append(decode_tps)
                    full_result.decode_sweep.time_values.append(decode_duration)

                    print(f"Out: {decoded_tokens} | TPS: {decode_tps:.1f} | Time: {decode_duration:.4f}s")

            thread.join()

            assert first_token_time is not None

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
        axs[0, 1].plot(result.prefill_sweep.x_values, result.prefill_sweep.time_values, 'r-o')
        axs[0, 1].set_title('Prefill: Tokens vs Latency (TTFT)')
        axs[0, 1].set_xlabel('Input Tokens')
        axs[0, 1].set_ylabel('Latency (seconds)')
        axs[0, 1].grid(True)

        # 3. Decode: Token vs TPS
        axs[1, 0].plot(result.decode_sweep.x_values, result.decode_sweep.tps_values, 'g-o')
        axs[1, 0].set_title('Decode: Tokens vs TPS')
        axs[1, 0].set_xlabel('Output Tokens')
        axs[1, 0].set_ylabel('TPS (tokens/sec)')
        axs[1, 0].grid(True)

        # 4. Decode: Token vs Time
        axs[1, 1].plot(result.decode_sweep.x_values, result.decode_sweep.time_values, 'm-o')
        axs[1, 1].set_title('Decode: Tokens vs Time (Duration)')
        axs[1, 1].set_xlabel('Output Tokens')
        axs[1, 1].set_ylabel('Total Generation Time (seconds)')
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
            axs[0, 1].plot(result.prefill_sweep.x_values, result.prefill_sweep.time_values, '-o', label=label)
            axs[1, 0].plot(result.decode_sweep.x_values, result.decode_sweep.tps_values, '-o', label=label)
            axs[1, 1].plot(result.decode_sweep.x_values, result.decode_sweep.time_values, '-o', label=label)

        axs[0, 0].set_title('Prefill: Tokens vs TPS (Higher is Better)')
        axs[0, 0].set_xlabel('Input Tokens')
        axs[0, 0].set_ylabel('TPS (tokens/sec)')
        axs[0, 0].grid(True)

        axs[0, 1].set_title('Prefill: Tokens vs Latency (TTFT)')
        axs[0, 1].set_xlabel('Input Tokens')
        axs[0, 1].set_ylabel('Latency (seconds)')
        axs[0, 1].grid(True)

        axs[1, 0].set_title('Decode: Tokens vs TPS')
        axs[1, 0].set_xlabel('Output Tokens')
        axs[1, 0].set_ylabel('TPS (tokens/sec)')
        axs[1, 0].grid(True)

        axs[1, 1].set_title('Decode: Tokens vs Time (Duration)')
        axs[1, 1].set_xlabel('Output Tokens')
        axs[1, 1].set_ylabel('Total Generation Time (seconds)')
        axs[1, 1].grid(True)

        for ax in axs.flat:
            ax.legend()

        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.savefig(save_path, dpi=300)
        print(f"\n✅ Graph saved to: {save_path}")
        plt.close(fig)