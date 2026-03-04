import json
import os
import platform
import shlex
import subprocess
import time
from typing import Dict, Optional

import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler

from .power_tracker import BasePowerTracker


class NPUPowerTracker(BasePowerTracker):
    """Track NPU power by polling `mobilint-cli status`."""

    def __init__(self, interval: float = 0.5, status_cmd: Optional[str] = None):
        super().__init__(interval=interval)
        if platform.system() != "Linux":
            raise RuntimeError("NPUPowerTracker currently supports Linux only")
        script_path = os.path.join(os.path.dirname(__file__), "power_tracker_npu.sh")
        self._status_cmd = (
            status_cmd
            if status_cmd is not None
            else f"bash {script_path} --sample-once --json"
        )
        self._scheduler = BackgroundScheduler()
        self._npu_power_glance = []
        self._total_power_glance = []
        self._power_trace = []

    def _fetch_power(self) -> Optional[tuple[float, float]]:
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
        return npu_power_w, total_power_w

    def _func_for_sched(self):
        power = self._fetch_power()
        if power is None:
            return
        npu_power_w, total_power_w = power
        ts = time.time()
        self._npu_power_glance.append(npu_power_w)
        self._total_power_glance.append(total_power_w)
        self._power_trace.append((ts, npu_power_w))

    def start(self):
        self.reset()
        self._scheduler.add_job(
            self._func_for_sched, "interval", seconds=self._interval
        )
        self._scheduler.start()

    def stop(self):
        if self._scheduler.running:
            self._scheduler.shutdown()

    def get_power_metric(self) -> Dict[str, Optional[float]]:
        npu_avg = (
            float(np.mean(self._npu_power_glance))
            if self._npu_power_glance
            else None
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
        return {
            "avg_power_w": npu_avg,
            "p99_power_w": npu_p99,
            "avg_npu_power_w": npu_avg,
            "p99_npu_power_w": npu_p99,
            "avg_total_power_w": total_avg,
            "p99_total_power_w": total_p99,
            "samples": len(self._power_trace),
        }

    def get_power_trace(self):
        return list(self._power_trace)

    def reset(self):
        self._npu_power_glance = []
        self._total_power_glance = []
        self._power_trace = []


# Backward compatibility
PowerTracker = NPUPowerTracker
