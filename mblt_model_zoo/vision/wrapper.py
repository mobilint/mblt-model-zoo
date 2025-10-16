import numpy as np
import torch
import sys
import os
from typing import Union
from urllib.parse import urlparse
import maccel
from ..utils.downloads import download_url_to_file
from .utils.types import TensorLike, ModelInfoSet
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

    @classmethod
    def from_model_info_set(
        cls,
        model_info_set: ModelInfoSet,
        local_path: str = None,
        model_type: str = "DEFAULT",
        infer_mode: str = "global",
        product: str = "aries",
    ):
        """
        Create an instance of the model from a ModelInfoSet.
        """

        assert (
            model_type in model_info_set.__dict__.keys()
        ), f"model_type {model_type} not found. Available types: {model_info_set.__dict__.keys()}"
        assert (
            product
            in model_info_set.__dict__[model_type].value.model_cfg["url_dict"].keys()
        ), f"product {product} not found in model_type {model_type}. Available products: {model_info_set.__dict__[model_type].value.model_cfg['url_dict'].keys()}"
        assert (
            infer_mode
            in model_info_set.__dict__[model_type]
            .value.model_cfg["url_dict"][product]
            .keys()
        ), f"infer_mode {infer_mode} not found in model_type {model_type} for product {product}. Available infer modes: {model_info_set.__dict__[model_type].value.model_cfg['url_dict'][product].keys()}"

        model_cfg = model_info_set.__dict__[model_type].value.model_cfg
        model_cfg["local_path"] = local_path
        model_cfg["infer_mode"] = infer_mode
        model_cfg["product"] = product

        pre_cfg = model_info_set.__dict__[model_type].value.pre_cfg
        post_cfg = model_info_set.__dict__[model_type].value.post_cfg

        return cls(model_cfg, pre_cfg, post_cfg)

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
        url_dict,
        local_path: str = None,
        infer_mode: str = "global",
        product: str = "aries",
    ):
        self.infer_mode = infer_mode
        self.product = product
        if self.product not in url_dict.keys():  # execption handling
            url_dict[self.product] = {self.infer_mode: None}

        self.acc = maccel.Accelerator()
        # ----------------Core Allocation-------------------------
        mc = maccel.ModelConfig()
        if self.product == "aries":
            if self.infer_mode == "single":
                pass  # default is single with all cores
            elif self.infer_mode == "multi":
                mc.set_multi_core_mode(
                    [maccel.Cluster.Cluster0, maccel.Cluster.Cluster1]
                )
            elif self.infer_mode == "global4":
                mc.set_global4_core_mode(
                    [maccel.Cluster.Cluster0, maccel.Cluster.Cluster1]
                )
            elif self.infer_mode == "global" or self.infer_mode == "global8":
                mc.set_global8_core_mode()
            else:
                raise ValueError("Inappropriate inferece mode")
        elif self.product == "regulus":
            assert (
                self.infer_mode == "single"
            ), "Only single core mode is available on Regulus"
        else:
            raise ValueError("Inappropriate product")

        # -----------------Model Preparation-----------------------
        url = url_dict[self.product].get(self.infer_mode)
        if url is not None:
            parts = urlparse(url)
            filename = os.path.basename(parts.path)

            if local_path is None:  # default option
                model_dir = os.path.expanduser(
                    f"~/.mblt_model_zoo/vision/{product}/{infer_mode}"
                )
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
                download_url_to_file(url, cached_file, progress=True)

        else:
            if local_path is not None:
                if os.path.isfile(local_path):
                    cached_file = local_path
                else:
                    raise ValueError(
                        "The model should be prepared on server or local path"
                    )
            else:
                raise ValueError("The model should be prepared on server or local path")

        # ----------------Initialize Model----------------------
        self.model = maccel.Model(cached_file, mc)
        self.model.launch(self.acc)

    def __call__(self, x: TensorLike):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        assert isinstance(x, np.ndarray), "Input should be a numpy array"

        npu_outs = self.model.infer(x)
        return npu_outs

    def dispose(self):
        """Dispose the model."""
        self.model.dispose()
