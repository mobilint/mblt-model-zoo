import json
import os
import platform
import shlex
import subprocess
import time
from typing import Dict, Optional

import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import STATE_RUNNING

from .device_tracker import BaseDeviceTracker


class NPUDeviceTracker(BaseDeviceTracker):
    """Track NPU power and utilization by polling `mobilint-cli status`."""

    def __init__(self, interval: float = 0.5, status_cmd: Optional[str] = None):
        super().__init__(interval=interval)
        if platform.system() != "Linux":
            raise RuntimeError("NPUDeviceTracker currently supports Linux only")
        script_path = os.path.join(os.path.dirname(__file__), "device_tracker_npu.sh")
        self._status_cmd = (
            status_cmd
            if status_cmd is not None
            else f"bash {script_path} --sample-once --json"
        )
        self._scheduler: Optional[BackgroundScheduler] = None
        self._job_id = "npu_device_track"
        self._npu_power_glance: list[float] = []
        self._total_power_glance: list[float] = []
        self._npu_util_glance: list[float] = []
        self._power_trace: list[tuple[float, float]] = []
        self._util_trace: list[tuple[float, float]] = []

    def _fetch_metrics(self) -> Optional[tuple[float, float, Optional[float]]]:
        try:
            result = subprocess.run(
                shlex.split(self._status_cmd),
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None

        if result.returncode != 0 or not result.stdout:
            return None

        try:
            payload = json.loads(result.stdout.strip())
        except Exception:
            return None

        if not payload.get("ok", False):
            return None
        if "npu_power_w" not in payload or "total_power_w" not in payload:
            return None

        npu_power_w = float(payload["npu_power_w"])
        total_power_w = float(payload["total_power_w"])
        npu_util_pct = payload.get("npu_util_pct")
        if npu_util_pct is not None:
            npu_util_pct = float(npu_util_pct)
        return npu_power_w, total_power_w, npu_util_pct

    def _func_for_sched(self) -> None:
        metrics = self._fetch_metrics()
        if metrics is None:
            return
        npu_power_w, total_power_w, npu_util_pct = metrics
        ts = time.time()
        self._npu_power_glance.append(npu_power_w)
        self._total_power_glance.append(total_power_w)
        self._power_trace.append((ts, total_power_w))
        if npu_util_pct is not None:
            self._npu_util_glance.append(npu_util_pct)
            self._util_trace.append((ts, npu_util_pct))

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

    def get_metric(self) -> Dict[str, Optional[float]]:
        npu_avg = (
            float(np.mean(self._npu_power_glance)) if self._npu_power_glance else None
        )
        total_avg = (
            float(np.mean(self._total_power_glance))
            if self._total_power_glance
            else None
        )
        npu_p99 = (
            float(np.percentile(self._npu_power_glance, 99))
            if self._npu_power_glance
            else None
        )
        total_p99 = (
            float(np.percentile(self._total_power_glance, 99))
            if self._total_power_glance
            else None
        )
        npu_util_avg = (
            float(np.mean(self._npu_util_glance)) if self._npu_util_glance else None
        )
        npu_util_p99 = (
            float(np.percentile(self._npu_util_glance, 99))
            if self._npu_util_glance
            else None
        )
        return {
            "avg_power_w": total_avg,
            "p99_power_w": total_p99,
            "avg_npu_power_w": npu_avg,
            "p99_npu_power_w": npu_p99,
            "avg_total_power_w": total_avg,
            "p99_total_power_w": total_p99,
            "avg_npu_util_pct": npu_util_avg,
            "p99_npu_util_pct": npu_util_p99,
            # Generic names for cross-device consumers.
            "avg_utilization_pct": npu_util_avg,
            "p99_utilization_pct": npu_util_p99,
            "samples": len(self._power_trace),
            "util_samples": len(self._util_trace),
        }

    def get_trace(self) -> list[tuple[float, float]]:
        return list(self._power_trace)

    def get_util_trace(self) -> list[tuple[float, float]]:
        return list(self._util_trace)

    def reset(self) -> None:
        self._npu_power_glance = []
        self._total_power_glance = []
        self._npu_util_glance = []
        self._power_trace = []
        self._util_trace = []
