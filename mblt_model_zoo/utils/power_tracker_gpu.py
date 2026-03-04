import time
from typing import List, Union

import numpy as np
import pynvml
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import STATE_RUNNING

from .power_tracker import BasePowerTracker


class GPUPowerTracker(BasePowerTracker):
    """This class is used to track the power consumption of GPUs.
    In order to calculate the power consumption, you should create the PowerTracker before the inference starts and stop it after the inference ends.
    It is recommended to create a new PowerTracker for each inference task.

    Example:
        tracker = PowerTracker()
        ---Do some other stuff---
        tracker.start()
        # your inference code here
        tracker.stop()
        gpu_power = tracker.get_consumption()
    """

    def __init__(self, interval=0.1, gpu_id: Union[int, List[int]] = None):
        """Initialize the PowerTracker

        Args:
            interval (float, optional): The interval between each power consumption measurement (in seconds). Defaults to 0.1.
            gpu_id (Union[int, List[int]], optional): The GPU IDs to track. If None, all available GPUs will be tracked.
        """
        super().__init__(interval=interval)
        pynvml.nvmlInit()
        self.num_gpus = self.gpu_num()
        if self.num_gpus == 0:
            raise ValueError("No GPUs available")

        if gpu_id is None:  # If gpu_id is not provided, use all available GPUs
            gpu_id = list(range(self.num_gpus))
        elif isinstance(gpu_id, int):
            assert gpu_id < self.num_gpus, f"Invalid GPU ID: {gpu_id}"
            gpu_id = [gpu_id]
        else:
            for i in gpu_id:
                assert i < self.num_gpus, f"Invalid GPU ID: {i}"

        self._gpu_id = gpu_id
        self._scheduler = None
        self._job_id = "gpu_power_track"
        self._power_glance = {
            gpu: [] for gpu in self._gpu_id
        }  # Store power consumption per GPU
        self._power_trace = []
        self.driver_version = pynvml.nvmlSystemGetDriverVersion()
        self.cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
        self.device_name = {
            gpu: pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(gpu))
            for gpu in self._gpu_id
        }

    def gpu_num(self):
        """Get the number of available GPUs"""
        return pynvml.nvmlDeviceGetCount()

    def gpu_power(self):
        """Get the current power usage of the GPUs in milliwatts"""
        gpu_power = []
        for i in self._gpu_id:
            power = pynvml.nvmlDeviceGetPowerUsage(
                pynvml.nvmlDeviceGetHandleByIndex(i)
            )  # Returns power in milliwatts
            gpu_power.append(power)
        return gpu_power

    def _func_for_sched(self):
        """Function to periodically track GPU power usage"""
        power_usage = self.gpu_power()
        ts = time.time()
        total_power = 0.0
        for idx, gpu in enumerate(self._gpu_id):
            power_w = power_usage[idx] / 1000  # Convert milliwatts to watts
            self._power_glance[gpu].append(power_w)
            total_power += power_w
        self._power_trace.append((ts, total_power))

    def start(self):
        """Start tracking GPU power consumption"""
        self.reset()
        # Capture one immediate sample so very short phases (e.g., vision encode)
        # still have power data even before the first interval tick.
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
            self._func_for_sched, "interval", seconds=self._interval, id=self._job_id
        )

    def stop(self):
        """Stop tracking GPU power consumption"""
        # Capture one final sample near phase end.
        try:
            self._func_for_sched()
        except Exception:
            pass
        if self._scheduler is not None:
            try:
                self._scheduler.shutdown(wait=True)
            finally:
                self._scheduler = None

    def get_power_metric(self):
        """Get the average power consumption in watts over the tracked period"""
        gpu_stats = {}
        for gpu, consumption in self._power_glance.items():
            if consumption:
                gpu_stats[gpu] = {
                    "avg_power_w": float(np.mean(consumption)),
                    "p99_power_w": float(np.percentile(consumption, 99)),
                }
            else:
                gpu_stats[gpu] = {"avg_power_w": None, "p99_power_w": None}

        total_samples = [p for _, p in self._power_trace]
        return {
            "avg_power_w": float(np.mean(total_samples)) if total_samples else None,
            "p99_power_w": float(np.percentile(total_samples, 99))
            if total_samples
            else None,
            "samples": len(total_samples),
            "gpu": gpu_stats,
        }

    def get_power_trace(self):
        return list(self._power_trace)

    def reset(self):
        """Reset the power tracker consumption values"""
        self._power_glance = {gpu: [] for gpu in self._gpu_id}
        self._power_trace = []


# Backward compatibility
PowerTracker = GPUPowerTracker
