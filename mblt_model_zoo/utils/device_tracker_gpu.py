import time
from typing import Dict, List, Optional, Union

import numpy as np
import pynvml
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import STATE_RUNNING

from .device_tracker import BaseDeviceTracker


class GPUDeviceTracker(BaseDeviceTracker):
    """Track GPU power and utilization through NVML."""

    def __init__(self, interval: float = 0.1, gpu_id: Union[int, List[int], None] = None):
        super().__init__(interval=interval)
        pynvml.nvmlInit()
        self.num_gpus = self.gpu_num()
        if self.num_gpus == 0:
            raise ValueError("No GPUs available")

        if gpu_id is None:
            gpu_id = list(range(self.num_gpus))
        elif isinstance(gpu_id, int):
            if gpu_id >= self.num_gpus:
                raise AssertionError(f"Invalid GPU ID: {gpu_id}")
            gpu_id = [gpu_id]
        else:
            for i in gpu_id:
                if i >= self.num_gpus:
                    raise AssertionError(f"Invalid GPU ID: {i}")

        self._gpu_id = gpu_id
        self._scheduler: Optional[BackgroundScheduler] = None
        self._job_id = "gpu_device_track"
        self._power_glance = {gpu: [] for gpu in self._gpu_id}
        self._gpu_util_glance = {gpu: [] for gpu in self._gpu_id}
        self._mem_util_glance = {gpu: [] for gpu in self._gpu_id}
        self._mem_used_glance = {gpu: [] for gpu in self._gpu_id}
        self._mem_used_pct_glance = {gpu: [] for gpu in self._gpu_id}
        self._mem_total_mb = {}
        self._power_trace: list[tuple[float, float]] = []
        self._gpu_util_trace: list[tuple[float, float]] = []
        self._mem_util_trace: list[tuple[float, float]] = []
        self._mem_used_trace: list[tuple[float, float]] = []
        self._mem_used_pct_trace: list[tuple[float, float]] = []
        self.driver_version = pynvml.nvmlSystemGetDriverVersion()
        self.cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
        self.device_name = {
            gpu: pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(gpu))
            for gpu in self._gpu_id
        }

    def gpu_num(self) -> int:
        return pynvml.nvmlDeviceGetCount()

    def gpu_power(self) -> list[int]:
        gpu_power = []
        for i in self._gpu_id:
            power = pynvml.nvmlDeviceGetPowerUsage(pynvml.nvmlDeviceGetHandleByIndex(i))
            gpu_power.append(power)
        return gpu_power

    def gpu_utilization(self) -> list[tuple[float, float]]:
        utilization = []
        for i in self._gpu_id:
            util = pynvml.nvmlDeviceGetUtilizationRates(
                pynvml.nvmlDeviceGetHandleByIndex(i)
            )
            utilization.append((float(util.gpu), float(util.memory)))
        return utilization

    def gpu_memory_info(self) -> list[tuple[float, float]]:
        mem_info = []
        for i in self._gpu_id:
            info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i))
            used_mb = float(info.used) / (1024.0 * 1024.0)
            total_mb = float(info.total) / (1024.0 * 1024.0)
            mem_info.append((used_mb, total_mb))
        return mem_info

    def _func_for_sched(self) -> None:
        power_usage = self.gpu_power()
        utilization = self.gpu_utilization()
        memory_info = self.gpu_memory_info()
        ts = time.time()
        total_power_w = 0.0
        total_gpu_util_pct = 0.0
        total_mem_util_pct = 0.0
        total_mem_used_mb = 0.0
        total_mem_capacity_mb = 0.0

        for idx, gpu in enumerate(self._gpu_id):
            power_w = float(power_usage[idx]) / 1000.0
            gpu_util_pct, mem_util_pct = utilization[idx]
            mem_used_mb, mem_total_mb = memory_info[idx]
            mem_used_pct = (mem_used_mb / mem_total_mb * 100.0) if mem_total_mb > 0 else 0.0
            self._power_glance[gpu].append(power_w)
            self._gpu_util_glance[gpu].append(gpu_util_pct)
            self._mem_util_glance[gpu].append(mem_util_pct)
            self._mem_used_glance[gpu].append(mem_used_mb)
            self._mem_used_pct_glance[gpu].append(mem_used_pct)
            self._mem_total_mb[gpu] = mem_total_mb
            total_power_w += power_w
            total_gpu_util_pct += gpu_util_pct
            total_mem_util_pct += mem_util_pct
            total_mem_used_mb += mem_used_mb
            total_mem_capacity_mb += mem_total_mb

        divisor = float(len(self._gpu_id))
        self._power_trace.append((ts, total_power_w))
        self._gpu_util_trace.append((ts, total_gpu_util_pct / divisor))
        self._mem_util_trace.append((ts, total_mem_util_pct / divisor))
        self._mem_used_trace.append((ts, total_mem_used_mb))
        total_mem_used_pct = (
            total_mem_used_mb / total_mem_capacity_mb * 100.0
            if total_mem_capacity_mb > 0
            else 0.0
        )
        self._mem_used_pct_trace.append((ts, total_mem_used_pct))

    def start(self) -> None:
        self.reset()
        try:
            self._func_for_sched()
        except Exception:
            pass
        if self._scheduler is None or self._scheduler.state != STATE_RUNNING:
            self._scheduler = BackgroundScheduler()
            self._scheduler.start()
        if self._scheduler.get_job(self._job_id) is not None:
            self._scheduler.remove_job(self._job_id)
        self._scheduler.add_job(
            self._func_for_sched,
            "interval",
            seconds=self._interval,
            id=self._job_id,
        )

    def stop(self) -> None:
        try:
            self._func_for_sched()
        except Exception:
            pass
        if self._scheduler is not None:
            try:
                self._scheduler.shutdown(wait=True)
            finally:
                self._scheduler = None

    def get_metric(self) -> Dict[str, object]:
        gpu_stats: Dict[int, Dict[str, Optional[float]]] = {}
        for gpu in self._gpu_id:
            power_samples = self._power_glance[gpu]
            gpu_util_samples = self._gpu_util_glance[gpu]
            mem_util_samples = self._mem_util_glance[gpu]
            mem_used_samples = self._mem_used_glance[gpu]
            mem_used_pct_samples = self._mem_used_pct_glance[gpu]
            gpu_stats[gpu] = {
                "avg_power_w": float(np.mean(power_samples)) if power_samples else None,
                "p99_power_w": float(np.percentile(power_samples, 99))
                if power_samples
                else None,
                "avg_gpu_util_pct": float(np.mean(gpu_util_samples))
                if gpu_util_samples
                else None,
                "p99_gpu_util_pct": float(np.percentile(gpu_util_samples, 99))
                if gpu_util_samples
                else None,
                "avg_mem_util_pct": float(np.mean(mem_util_samples))
                if mem_util_samples
                else None,
                "p99_mem_util_pct": float(np.percentile(mem_util_samples, 99))
                if mem_util_samples
                else None,
                "avg_memory_used_mb": float(np.mean(mem_used_samples))
                if mem_used_samples
                else None,
                "p99_memory_used_mb": float(np.percentile(mem_used_samples, 99))
                if mem_used_samples
                else None,
                "total_memory_mb": self._mem_total_mb.get(gpu),
                "avg_memory_used_pct": float(np.mean(mem_used_pct_samples))
                if mem_used_pct_samples
                else None,
                "p99_memory_used_pct": float(np.percentile(mem_used_pct_samples, 99))
                if mem_used_pct_samples
                else None,
            }

        total_power_samples = [p for _, p in self._power_trace]
        total_gpu_util_samples = [u for _, u in self._gpu_util_trace]
        total_mem_util_samples = [u for _, u in self._mem_util_trace]
        total_mem_used_samples = [m for _, m in self._mem_used_trace]
        total_mem_used_pct_samples = [m for _, m in self._mem_used_pct_trace]
        total_mem_capacity_mb = float(sum(self._mem_total_mb.values())) if self._mem_total_mb else None
        avg_gpu_util = (
            float(np.mean(total_gpu_util_samples)) if total_gpu_util_samples else None
        )
        p99_gpu_util = (
            float(np.percentile(total_gpu_util_samples, 99))
            if total_gpu_util_samples
            else None
        )
        return {
            "avg_power_w": float(np.mean(total_power_samples))
            if total_power_samples
            else None,
            "p99_power_w": float(np.percentile(total_power_samples, 99))
            if total_power_samples
            else None,
            "avg_gpu_util_pct": avg_gpu_util,
            "p99_gpu_util_pct": p99_gpu_util,
            "avg_mem_util_pct": float(np.mean(total_mem_util_samples))
            if total_mem_util_samples
            else None,
            "p99_mem_util_pct": float(np.percentile(total_mem_util_samples, 99))
            if total_mem_util_samples
            else None,
            # Generic names for cross-device consumers.
            "avg_utilization_pct": avg_gpu_util,
            "p99_utilization_pct": p99_gpu_util,
            "avg_memory_used_mb": float(np.mean(total_mem_used_samples))
            if total_mem_used_samples
            else None,
            "p99_memory_used_mb": float(np.percentile(total_mem_used_samples, 99))
            if total_mem_used_samples
            else None,
            "total_memory_mb": total_mem_capacity_mb,
            "avg_memory_used_pct": float(np.mean(total_mem_used_pct_samples))
            if total_mem_used_pct_samples
            else None,
            "p99_memory_used_pct": float(np.percentile(total_mem_used_pct_samples, 99))
            if total_mem_used_pct_samples
            else None,
            "samples": len(total_power_samples),
            "util_samples": len(total_gpu_util_samples),
            "gpu": gpu_stats,
        }

    def get_trace(self) -> list[tuple[float, float]]:
        return list(self._power_trace)

    def get_util_trace(self) -> list[tuple[float, float]]:
        return list(self._gpu_util_trace)

    def reset(self) -> None:
        self._power_glance = {gpu: [] for gpu in self._gpu_id}
        self._gpu_util_glance = {gpu: [] for gpu in self._gpu_id}
        self._mem_util_glance = {gpu: [] for gpu in self._gpu_id}
        self._mem_used_glance = {gpu: [] for gpu in self._gpu_id}
        self._mem_used_pct_glance = {gpu: [] for gpu in self._gpu_id}
        self._mem_total_mb = {}
        self._power_trace = []
        self._gpu_util_trace = []
        self._mem_util_trace = []
        self._mem_used_trace = []
        self._mem_used_pct_trace = []
