"""TPS CLI parser and EAGLE-3 option handling tests."""

from __future__ import annotations

import argparse
import importlib
import types
import warnings
from types import SimpleNamespace

import pytest

from mblt_model_zoo.cli import tps as tps_cli
from mblt_model_zoo.cli.main import build_parser


class _DummyConfig:
    def __init__(self, max_batch_size=None, text_config=None, vision_config=None):
        if max_batch_size is not None:
            self.max_batch_size = max_batch_size
        if text_config is not None:
            self.text_config = text_config
        if vision_config is not None:
            self.vision_config = vision_config


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
    assert args.device_backend is None


def test_cli_tps_measure_defaults():
    parser = build_parser()
    args = parser.parse_args(["tps", "measure", "--model", "mobilint/Llama-3.2-1B-Instruct"])

    assert args.prefill == 128
    assert args.decode == 32
    assert args.batch_size is None


def test_cli_tps_measure_batch_size_override():
    parser = build_parser()
    args = parser.parse_args(["tps", "measure", "--model", "mobilint/Llama-3.2-1B-Instruct", "--batch-size", "4"])

    assert args.batch_size == 4


def test_cli_tps_sweep_defaults():
    parser = build_parser()
    args = parser.parse_args(["tps", "sweep", "--model", "mobilint/Llama-3.2-1B-Instruct"])

    assert args.prefill_range == (512, 2048, 512)
    assert args.cache_lengths == [128, 512, 1024, 2048]
    assert args.decode_window == 32
    assert args.batch_size is None


def test_cli_tps_batch_size_rejects_non_positive_values():
    parser = build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["tps", "measure", "--model", "mobilint/Llama-3.2-1B-Instruct", "--batch-size", "0"])

    assert excinfo.value.code == 2


def test_cli_resolve_model_max_batch_size_uses_top_level_config():
    pipeline = SimpleNamespace(model=SimpleNamespace(config=_DummyConfig(max_batch_size=4)))
    assert tps_cli._resolve_model_max_batch_size(pipeline, task="text-generation") == 4


def test_cli_resolve_model_max_batch_size_uses_text_config():
    config = _DummyConfig(text_config=_DummyConfig(max_batch_size=8))
    pipeline = SimpleNamespace(model=SimpleNamespace(config=config))
    assert tps_cli._resolve_model_max_batch_size(pipeline, task="text-generation") == 8


def test_cli_resolve_model_max_batch_size_uses_vlm_vision_config():
    config = _DummyConfig(vision_config=_DummyConfig(max_batch_size=2))
    pipeline = SimpleNamespace(model=SimpleNamespace(config=config))
    assert tps_cli._resolve_model_max_batch_size(pipeline, task="image-text-to-text") == 2


@pytest.mark.parametrize(
    ("value", "expected"),
    [("bad", None), (None, None), (0, 1), (-3, 1), ("5", 5)],
)
def test_cli_normalize_max_batch_size(value, expected):
    assert tps_cli._normalize_max_batch_size(value) == expected


def test_cli_resolve_cli_batch_size_prefers_explicit_override():
    args = argparse.Namespace(task="text-generation", batch_size=6)
    pipeline = SimpleNamespace(model=SimpleNamespace(config=_DummyConfig(max_batch_size=3)))
    assert tps_cli._resolve_cli_batch_size(args, pipeline) == 6


def test_extract_eagle3_pipeline_kwargs_returns_dataclass() -> None:
    args = argparse.Namespace(
        base_embedding_path="base.bin",
        draft_embedding_path="draft.bin",
        base_mxq_path="base.mxq",
        draft_mxq_path="draft.mxq",
        fc_mxq_path="fc.mxq",
        base_core_mode="single",
        draft_core_mode="global4",
        fc_core_mode="global8",
        base_target_cores=["npu0"],
        draft_target_cores=["npu1"],
        fc_target_cores=["npu2"],
        base_target_clusters=[0],
        draft_target_clusters=[1],
        fc_target_clusters=[2],
    )
    options = tps_cli._extract_eagle3_pipeline_kwargs(args)
    assert isinstance(options, tps_cli.Eagle3PipelineOptions)
    assert options.base_embedding_path == "base.bin"
    assert options.draft_embedding_path == "draft.bin"
    assert options.base_mxq_path == "base.mxq"
    assert options.draft_mxq_path == "draft.mxq"
    assert options.fc_mxq_path == "fc.mxq"


def test_build_pipeline_eagle3_prefixed_options_override_global_with_warning(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_pipeline(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(tps_cli, "_require_transformers_deps", lambda: None)
    monkeypatch.setattr(importlib.import_module("transformers"), "pipeline", _fake_pipeline)

    eagle3_options = tps_cli.Eagle3PipelineOptions(
        base_mxq_path="base.mxq",
        draft_mxq_path="draft.mxq",
        fc_mxq_path="fc.mxq",
        base_core_mode="single",
        draft_core_mode="global4",
        fc_core_mode="global8",
        base_target_cores=["npu0"],
        draft_target_cores=["npu1"],
        fc_target_cores=["npu2"],
        base_target_clusters=[0],
        draft_target_clusters=[1],
        fc_target_clusters=[2],
    )

    with pytest.warns(UserWarning, match="Conflicting options detected"):
        tps_cli._build_pipeline(
            task="text-generation",
            model="dummy/model",
            tokenizer=None,
            device="cpu",
            trust_remote_code=True,
            dtype=None,
            device_map=None,
            revision=None,
            embedding_weight=None,
            eagle3_options=eagle3_options,
            mxq_path="global.mxq",
            core_mode="single",
            target_cores=["npu9"],
            target_clusters=[9],
        )

    model_kwargs = captured.get("model_kwargs")
    assert isinstance(model_kwargs, dict)
    assert model_kwargs["base_mxq_path"] == "base.mxq"
    assert model_kwargs["draft_mxq_path"] == "draft.mxq"
    assert model_kwargs["fc_mxq_path"] == "fc.mxq"


def test_build_pipeline_eagle3_prefixed_options_no_warning_when_same_values(monkeypatch) -> None:
    monkeypatch.setattr(tps_cli, "_require_transformers_deps", lambda: None)
    monkeypatch.setattr(importlib.import_module("transformers"), "pipeline", lambda **kwargs: types.SimpleNamespace(**kwargs))

    eagle3_options = tps_cli.Eagle3PipelineOptions(
        base_core_mode="single",
        draft_core_mode="single",
        fc_core_mode="single",
        base_target_cores=["npu0"],
        draft_target_cores=["npu0"],
        fc_target_cores=["npu0"],
        base_target_clusters=[0],
        draft_target_clusters=[0],
        fc_target_clusters=[0],
    )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        tps_cli._build_pipeline(
            task="text-generation",
            model="dummy/model",
            tokenizer=None,
            device="cpu",
            trust_remote_code=True,
            dtype=None,
            device_map=None,
            revision=None,
            embedding_weight=None,
            eagle3_options=eagle3_options,
            mxq_path=None,
            core_mode="single",
            target_cores=["npu0"],
            target_clusters=[0],
        )

    conflict_warnings = [w for w in record if "Conflicting options detected" in str(w.message)]
    assert len(conflict_warnings) == 0
