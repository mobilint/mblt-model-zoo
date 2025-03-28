import maccel
import numpy as np
import torch
from .utils.types import TensorLike
from .utils.preprocess import build_preprocess
from .utils.postprocess import build_postprocess


class MBLT_Engine:
    def __init__(self, model_cfg: dict, pre_cfg: dict, post_cfg: dict):
        self.model_cfg = model_cfg
        self.pre_cfg = pre_cfg
        self.post_cfg = post_cfg

        self.model = MXQ_Model(**self.model_cfg)
        self.preprocess = build_preprocess(self.pre_cfg)
        self.postprocess = build_postprocess(self.pre_cfg, self.post_cfg)

    def __call__(self, x):
        return self.model(x)


class MXQ_Model:
    def __init__(self, mxq_model_path, core_info: dict = None, trace: bool = False):
        self.trace = trace
        self.acc = maccel.Accelerator()
        mc = maccel.ModelConfig()
        mc.exclude_all_cores()

        if core_info is not None:
            mc.include(*core_info)
            print(f"Using cluster {list(core_info)[0]} core {list(core_info)[1]}")
        else:
            mc.include_all_cores()
            print("Using all cores")

        self.model = maccel.Model(mxq_model_path, mc)
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
