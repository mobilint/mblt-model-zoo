import inspect

import pytest
import torch

from mblt_model_zoo.cli.main import build_parser
from mblt_model_zoo.hf_transformers.models.qwen2_vl.modeling_qwen2_vl import MobilintQwen2VLForConditionalGeneration
from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import MobilintQwen3VLForConditionalGeneration
from mblt_model_zoo.hf_transformers.utils.benchmark_utils import (
    _resolve_config_vocab_size,
    _resolve_image_features_tensor,
)


class _DummyConfig:
    def __init__(self, vocab_size: int | None = None, text_config=None):
        if vocab_size is not None:
            self.vocab_size = vocab_size
        if text_config is not None:
            self.text_config = text_config


class _DummyVisionOutput:
    def __init__(self, pooler_output=None):
        self.pooler_output = pooler_output


def test_cli_tps_sweep_range_parsing():
    parser = build_parser()
    args = parser.parse_args(
        [
            "tps",
            "sweep",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct",
            "--prefill-range",
            "1:3:1",
            "--cache-lengths",
            "1024,2048,4096",
            "--no-plot",
        ]
    )
    assert args.prefill_range == (1, 3, 1)
    assert args.cache_lengths == [1024, 2048, 4096]
    assert args.plot is None
    assert args.device_backend == "none"


def test_resolve_config_vocab_size_uses_top_level_config():
    config = _DummyConfig(vocab_size=128)

    assert _resolve_config_vocab_size(config) == 128


def test_resolve_config_vocab_size_uses_text_config():
    config = _DummyConfig(text_config=_DummyConfig(vocab_size=256))

    assert _resolve_config_vocab_size(config) == 256


def test_resolve_config_vocab_size_requires_vocab_size():
    with pytest.raises(AttributeError, match="vocab_size"):
        _resolve_config_vocab_size(_DummyConfig())


def test_resolve_image_features_tensor_uses_tensor():
    image_features = torch.zeros(2, 4)

    assert _resolve_image_features_tensor(image_features) is image_features


def test_resolve_image_features_tensor_uses_pooler_output():
    pooler_output = torch.zeros(2, 4)
    image_features = _DummyVisionOutput(pooler_output=pooler_output)

    assert _resolve_image_features_tensor(image_features) is pooler_output


def test_resolve_image_features_tensor_concatenates_pooler_output_tuple():
    first = torch.zeros(2, 4)
    second = torch.ones(3, 4)
    image_features = _DummyVisionOutput(pooler_output=(first, second))

    resolved = _resolve_image_features_tensor(image_features)

    assert torch.equal(resolved, torch.cat((first, second), dim=0))


def test_resolve_image_features_tensor_uses_tuple_tensor():
    image_features = torch.zeros(2, 4)

    assert _resolve_image_features_tensor((image_features, None)) is image_features


def test_resolve_image_features_tensor_requires_tensor():
    with pytest.raises(TypeError, match="image feature tensor"):
        _resolve_image_features_tensor(_DummyVisionOutput())


@pytest.mark.parametrize(
    ("model_cls", "method_name"),
    [
        (MobilintQwen2VLForConditionalGeneration, "forward"),
        (MobilintQwen3VLForConditionalGeneration, "forward"),
        (MobilintQwen3VLForConditionalGeneration, "prepare_inputs_for_generation"),
    ],
)
def test_mobilint_generation_hooks_accept_count_npu_time(model_cls, method_name: str):
    signature = inspect.signature(getattr(model_cls, method_name))

    assert "count_npu_time" in signature.parameters


@pytest.mark.parametrize("spec", ["", "1", "1:2", "1:2:0", "2:1:1", "a:b:c"])
def test_cli_tps_invalid_range_exits(spec: str):
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "tps",
                "sweep",
                "--model",
                "mobilint/Llama-3.2-1B-Instruct",
                "--prefill-range",
                spec,
                "--no-plot",
            ]
        )
    assert excinfo.value.code == 2


def test_cli_tps_device_backend_none_parsing():
    parser = build_parser()
    args = parser.parse_args(
        [
            "tps",
            "measure",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct",
            "--device-backend",
            "none",
        ]
    )
    assert args.device_backend == "none"


@pytest.mark.parametrize(
    "argv",
    [
        [
            "tps",
            "measure",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct",
            "--prefill",
            "0",
        ],
        [
            "tps",
            "measure",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct",
            "--decode",
            "-1",
        ],
        [
            "tps",
            "sweep",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct",
            "--decode-window",
            "0",
            "--no-plot",
        ],
    ],
)
def test_cli_tps_positive_int_enforced(argv: list[str]):
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(argv)
    assert excinfo.value.code == 2
