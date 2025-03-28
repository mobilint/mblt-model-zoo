import maccel
import numpy as np
import torch
import sys
import os
import tempfile
import uuid
import errno
import hashlib
import shutil
from typing import Optional
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from .utils.types import TensorLike
from .utils.preprocess import build_preprocess
from .utils.postprocess import build_postprocess

# The code below is copied from torch.hub.download_url_to_file

class _Faketqdm:  # type: ignore[no-redef]
    def __init__(self, total=None, disable=False, unit=None, *args, **kwargs):
        self.total = total
        self.disable = disable
        self.n = 0
        # Ignore all extra *args and **kwargs lest you want to reinvent tqdm

    def update(self, n):
        if self.disable:
            return

        self.n += n
        if self.total is None:
            sys.stderr.write(f"\r{self.n:.1f} bytes")
        else:
            sys.stderr.write(f"\r{100 * self.n / float(self.total):.1f}%")
        sys.stderr.flush()

    # Don't bother implementing; use real tqdm if you want
    def set_description(self, *args, **kwargs):
        pass

    def write(self, s):
        sys.stderr.write(f"{s}\n")

    def close(self):
        self.disable = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disable:
            return

        sys.stderr.write("\n")


try:
    from tqdm import tqdm  # If tqdm is installed use it, otherwise use the fake wrapper
except ImportError:
    tqdm = _Faketqdm

READ_DATA_CHUNK = 128 * 1024

def download_url_to_file(
    url: str,
    dst: str,
    hash_prefix: Optional[str] = None,
    progress: bool = True,
) -> None:
    r"""Download object at the given URL to a local path.

    Args:
        url (str): URL of the object to download
        dst (str): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_prefix (str, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> # xdoctest: +REQUIRES(POSIX)
        >>> torch.hub.download_url_to_file(
        ...     "https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth",
        ...     "/tmp/temporary_file",
        ... )

    """
    file_size = None
    req = Request(url, headers={"User-Agent": "mblt_models"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    # We deliberately do not use NamedTemporaryFile to avoid restrictive
    # file permissions being applied to the downloaded file.
    dst = os.path.expanduser(dst)
    for seq in range(tempfile.TMP_MAX):
        tmp_dst = dst + "." + uuid.uuid4().hex + ".partial"
        try:
            f = open(tmp_dst, "w+b")
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError(errno.EEXIST, "No usable temporary file name found")

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(
            total=file_size,
            disable=not progress,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            while True:
                buffer = u.read(READ_DATA_CHUNK)
                if len(buffer) == 0:
                    break
                f.write(buffer)  # type: ignore[possibly-undefined]
                if hash_prefix is not None:
                    sha256.update(buffer)  # type: ignore[possibly-undefined]
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()  # type: ignore[possibly-undefined]
            if digest[: len(hash_prefix)] != hash_prefix:
                raise RuntimeError(
                    f'invalid hash value (expected "{hash_prefix}", got "{digest}")'
                )
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


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
    def __init__(self, url, core_info: dict = None, trace: bool = False):
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

        model_dir = "~/.mblt_models"

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
