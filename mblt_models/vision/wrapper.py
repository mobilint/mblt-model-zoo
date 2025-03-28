import maccel
import numpy as np
import torch
import sys
import os
from urllib.parse import urlparse
from .utils.downloads import download_url_to_file
from .utils.types import TensorLike
from .utils.preprocess import build_preprocess
from .utils.postprocess import build_postprocess


class MBLT_Engine:
    def __init__(self, model_cfg: dict, pre_cfg: dict, post_cfg: dict):
        self.model_cfg = model_cfg
        self.pre_cfg = pre_cfg
        self.post_cfg = post_cfg

        self.model = MXQ_Model(**self.model_cfg)

    def __call__(self, x):
        return self.model(x)

    def get_preprocess(self, **kwargs):
        return build_preprocess(self.pre_cfg, **kwargs)

    def get_postprocess(self, **kwargs):
        return build_postprocess(self.pre_cfg, self.post_cfg, **kwargs)


class MXQ_Model:
    def __init__(self, url, core_info: dict = None, trace: bool = False):
        self.trace = trace
        self.acc = maccel.Accelerator()
        mc = maccel.ModelConfig()
        mc.exclude_all_cores()
        mc.set_global_core_mode(
            [maccel.Cluster.Cluster0, maccel.Cluster.Cluster1]
        )  # Cluster0, Cluster1 모두 사용

        if os.path.exists(url) and os.path.isfile(url) and url.endswith(".mxq"):
            cached_file = url

        else:
            model_dir = os.path.expanduser("~/.mblt_models")

            os.makedirs(model_dir, exist_ok=True)

            parts = urlparse(url)
            filename = os.path.basename(parts.path)
            cached_file = os.path.join(model_dir, filename)
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
