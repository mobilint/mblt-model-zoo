import warnings
from dataclasses import dataclass
from typing import Any, List, Optional

import pytest

_WARNED_UNUSED_PREFIXES: set[str] = set()


def _parse_target_cores(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return [item.strip() for item in text.split(";") if item.strip()]


def _parse_target_clusters(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    clusters: list[int] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        clusters.append(int(item))
    return clusters


def _collect_npu_kwargs(
    config: pytest.Config, prefix: str
) -> tuple[dict[str, Any], bool]:
    opt_prefix = f"--{prefix}-" if prefix else "--"
    mxq_path = config.getoption(f"{opt_prefix}mxq-path")
    dev_no = config.getoption(f"{opt_prefix}dev-no")
    core_mode = config.getoption(f"{opt_prefix}core-mode")
    target_cores_raw = config.getoption(f"{opt_prefix}target-cores")
    target_cores = _parse_target_cores(target_cores_raw)
    target_clusters_raw = config.getoption(f"{opt_prefix}target-clusters")
    target_clusters = _parse_target_clusters(target_clusters_raw)

    kwargs: dict[str, Any] = {}
    provided = False

    if mxq_path:
        kwargs[f"{prefix + '_' if prefix else ''}mxq_path"] = mxq_path
        provided = True
    if dev_no is not None:
        kwargs[f"{prefix + '_' if prefix else ''}dev_no"] = dev_no
        provided = True
    if core_mode:
        kwargs[f"{prefix + '_' if prefix else ''}core_mode"] = core_mode
        provided = True
    if target_cores is not None:
        kwargs[f"{prefix + '_' if prefix else ''}target_cores"] = target_cores
        provided = True
    if target_clusters is not None:
        kwargs[f"{prefix + '_' if prefix else ''}target_clusters"] = target_clusters
        provided = True

    return kwargs, provided


@dataclass(frozen=True)
class NpuParams:
    base: dict[str, Any]
    encoder: dict[str, Any]
    decoder: dict[str, Any]
    text: dict[str, Any]
    vision: dict[str, Any]
    _provided: dict[str, bool]

    def warn_unused(self, used_prefixes: set[str]) -> None:
        for prefix, provided in self._provided.items():
            if (
                provided
                and prefix not in used_prefixes
                and prefix not in _WARNED_UNUSED_PREFIXES
            ):
                _WARNED_UNUSED_PREFIXES.add(prefix)
                warnings.warn(
                    f"Provided {prefix} NPU backend options will be ignored for this model.",
                    UserWarning,
                )


def pytest_addoption(parser):
    parser.addoption(
        "--mxq-path",
        action="store",
        default=None,
        help="Override default mxq_path for pipeline loading.",
    )
    parser.addoption(
        "--dev-no",
        action="store",
        default=None,
        type=int,
        help="NPU device number.",
    )
    parser.addoption(
        "--core-mode",
        action="store",
        default=None,
        help="NPU core mode (single, multi, global4, global8).",
    )
    parser.addoption(
        "--target-cores",
        action="store",
        default=None,
        help='Target cores (e.g., "0:0;0:1;0:2;0:3").',
    )
    parser.addoption(
        "--target-clusters",
        action="store",
        default=None,
        help='Target clusters (e.g., "0;1").',
    )
    for prefix in ("encoder", "decoder", "vision", "text"):
        parser.addoption(
            f"--{prefix}-mxq-path",
            action="store",
            default=None,
            help=f"Override {prefix} mxq_path.",
        )
        parser.addoption(
            f"--{prefix}-dev-no",
            action="store",
            default=None,
            type=int,
            help=f"{prefix} NPU device number.",
        )
        parser.addoption(
            f"--{prefix}-core-mode",
            action="store",
            default=None,
            help=f"{prefix} NPU core mode (single, multi, global4, global8).",
        )
        parser.addoption(
            f"--{prefix}-target-cores",
            action="store",
            default=None,
            help=f'{prefix} target cores (e.g., "0:0;0:1;0:2;0:3").',
        )
        parser.addoption(
            f"--{prefix}-target-clusters",
            action="store",
            default=None,
            help=f'{prefix} target clusters (e.g., "0;1").',
        )
    parser.addoption(
        "--revision",
        action="store",
        default=None,
        help="Override model revision (e.g., W8).",
    )
    parser.addoption(
        "--embedding-weight",
        action="store",
        default=None,
        help="Path to custom embedding weights.",
    )


@pytest.fixture(scope="module")
def mxq_path(request):
    return request.config.getoption("--mxq-path")


@pytest.fixture(scope="module")
def revision(request):
    return request.config.getoption("--revision")


@pytest.fixture(scope="module")
def embedding_weight(request):
    return request.config.getoption("--embedding-weight")


@pytest.fixture(scope="module")
def npu_params(request, embedding_weight):
    config = request.config
    base_kwargs, base_provided = _collect_npu_kwargs(config, "")
    if embedding_weight:
        base_kwargs["embedding_weight"] = embedding_weight
        base_provided = True

    encoder_kwargs, encoder_provided = _collect_npu_kwargs(config, "encoder")
    decoder_kwargs, decoder_provided = _collect_npu_kwargs(config, "decoder")
    vision_kwargs, vision_provided = _collect_npu_kwargs(config, "vision")
    text_kwargs, text_provided = _collect_npu_kwargs(config, "text")

    return NpuParams(
        base=base_kwargs,
        encoder=encoder_kwargs,
        decoder=decoder_kwargs,
        text=text_kwargs,
        vision=vision_kwargs,
        _provided={
            "base": base_provided,
            "encoder": encoder_provided,
            "decoder": decoder_provided,
            "vision": vision_provided,
            "text": text_provided,
        },
    )
