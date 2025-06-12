import numpy as np
import torch
import sys
import os
from typing import Union
from urllib.parse import urlparse
import maccel
from .utils.downloads import download_url_to_file
from .utils.types import TensorLike
from .utils.preprocess import build_preprocess
from .utils.postprocess import build_postprocess
from .utils.results import Results


class MBLT_Engine:
    def __init__(self, model_cfg: dict, pre_cfg: dict, post_cfg: dict):
        self.model_cfg = model_cfg
        self.pre_cfg = pre_cfg
        self.post_cfg = post_cfg

        self.model = MXQ_Model(**self.model_cfg)
        self._preprocess = build_preprocess(self.pre_cfg)
        self._postprocess = build_postprocess(self.pre_cfg, self.post_cfg)

        self.device = torch.device("cpu")

    def __call__(self, x: TensorLike):
        return self.model(x)

    def preprocess(self, x, **kwargs):
        return self._preprocess(x, **kwargs)

    def postprocess(self, x, **kwargs):
        pre_result = self._postprocess(x, **kwargs)
        return Results(self.pre_cfg, self.post_cfg, pre_result, **kwargs)

    def to(self, device: Union[str, torch.device]):
        self._preprocess.to(device)
        self._postprocess.to(device)

        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise TypeError(f"Got unexpected type for device={type(device)}.")

    def cpu(self):
        self.to(device="cpu")

    def gpu(self):
        self.to(device="cuda")

    def cuda(self, device: Union[str, int] = 0):
        if isinstance(device, int):
            device = f"cuda:{device}"
        elif isinstance(device, str):
            if not device.startswith("cuda:"):
                raise ValueError("Invalid device string. It should start with 'cuda:'.")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your environment.")
        self.to(device=device)

    def dispose(self):
        self.model.dispose()


class MXQ_Model:
    def __init__(
        self,
        url,
        local_path: str = None,
        trace: bool = False,
    ):
        self.trace = trace
        self.acc = maccel.Accelerator()
        mc = maccel.ModelConfig()
        mc.set_global_core_mode(
            [maccel.Cluster.Cluster0, maccel.Cluster.Cluster1]
        )  # Cluster0, Cluster1 모두 사용

        parts = urlparse(url)
        filename = os.path.basename(parts.path)

        if local_path is None:  # default option
            model_dir = os.path.expanduser("~/.mblt_model_zoo")
            os.makedirs(model_dir, exist_ok=True)
            cached_file = os.path.join(model_dir, filename)

        else:
            if local_path.endswith(".mxq"):
                cached_file = local_path
            else:
                os.makedirs(local_path, exist_ok=True)
                cached_file = os.path.join(local_path, filename)

        if not os.path.exists(cached_file):
            sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
            hash_prefix = None
            download_url_to_file(url, cached_file, hash_prefix, progress=True)

        self.model = maccel.Model(cached_file, mc)
        self.model.launch(self.acc)

        if self.trace:
            maccel.start_tracing_events(self.trace)

    def __call__(self, x: TensorLike):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        assert isinstance(x, np.ndarray), "Input should be a numpy array"

        npu_outs = self.model.infer(x)
        return npu_outs

    def dispose(self):
        """Dispose the model and stop tracing if enabled."""
        if self.trace:
            maccel.stop_tracing_events()
        self.model.dispose()
