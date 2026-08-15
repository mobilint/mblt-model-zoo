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


def test_cli_tps_measure_print_output_defaults_false():
    parser = build_parser()
    args = parser.parse_args(["tps", "measure", "--model", "mobilint/Llama-3.2-1B-Instruct"])

    assert args.print_output is False


def test_cli_tps_measure_print_output_flag():
    parser = build_parser()
    args = parser.parse_args(["tps", "measure", "--model", "mobilint/Llama-3.2-1B-Instruct", "--print-output"])

    assert args.print_output is True


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-inf", "Infinity"])
def test_cli_tps_measure_temperature_rejects_non_finite(value):
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "tps",
                "measure",
                "--model",
                "mobilint/Llama-3.2-1B-Instruct",
                "--temperature",
                value,
            ]
        )

    assert excinfo.value.code == 2


@pytest.mark.parametrize(("value", "expected"), [("0.0", 0.0), ("0.5", 0.5), ("1.0", 1.0)])
def test_cli_tps_measure_temperature_accepts_finite_non_negative(value, expected):
    parser = build_parser()
    args = parser.parse_args(
        [
            "tps",
            "measure",
            "--model",
            "mobilint/Llama-3.2-1B-Instruct",
            "--temperature",
            value,
        ]
    )

    assert args.temperature == pytest.approx(expected)


def test_cli_tps_measure_temperature_rejects_negative():
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "tps",
                "measure",
                "--model",
                "mobilint/Llama-3.2-1B-Instruct",
                "--temperature",
                "-0.1",
            ]
        )

    assert excinfo.value.code == 2


def test_cli_tps_measure_thinking_defaults_to_none():
    parser = build_parser()
    args = parser.parse_args(["tps", "measure", "--model", "mobilint/Llama-3.2-1B-Instruct"])

    assert args.enable_thinking is None


def test_cli_tps_measure_enable_thinking_flag():
    parser = build_parser()
    args = parser.parse_args(["tps", "measure", "--model", "mobilint/Llama-3.2-1B-Instruct", "--enable-thinking"])

    assert args.enable_thinking is True


def test_cli_tps_measure_disable_thinking_flag():
    parser = build_parser()
    args = parser.parse_args(["tps", "measure", "--model", "mobilint/Llama-3.2-1B-Instruct", "--disable-thinking"])

    assert args.enable_thinking is False


def test_cli_tps_measure_thinking_flags_are_mutually_exclusive():
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "tps",
                "measure",
                "--model",
                "mobilint/Llama-3.2-1B-Instruct",
                "--enable-thinking",
                "--disable-thinking",
            ]
        )

    assert excinfo.value.code == 2


class _RecordingTokenizer:
    """Minimal tokenizer stub that records apply_chat_template calls."""

    def __init__(self, *, chat_template: str | None = "dummy", accepts_enable_thinking: bool = True):
        self.chat_template = chat_template
        self.accepts_enable_thinking = accepts_enable_thinking
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        if not self.accepts_enable_thinking and "enable_thinking" in kwargs:
            raise TypeError("apply_chat_template() got an unexpected keyword argument 'enable_thinking'")
        self.calls.append({"messages": messages, "kwargs": kwargs})
        import torch

        return {"input_ids": torch.zeros((1, 3), dtype=torch.long)}

    def __call__(self, text, **kwargs):  # pragma: no cover - fallback branch
        import torch

        return {"input_ids": torch.zeros((1, 2), dtype=torch.long)}


def _measure_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        input_mode="synthetic-text",
        prompt_text="hello",
        prompt_file=None,
        prompt_file_strategy="first",
        prompt_file_seed=0,
        apply_chat_template=True,
        enable_thinking=None,
        prefill=8,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_tokenize_prompt_text_omits_enable_thinking_when_unset():
    tokenizer = _RecordingTokenizer()
    pipeline = SimpleNamespace(tokenizer=tokenizer)

    tps_cli._resolve_text_measure_inputs(_measure_args(), pipeline)

    assert len(tokenizer.calls) == 1
    assert "enable_thinking" not in tokenizer.calls[0]["kwargs"]


def test_tokenize_prompt_text_forwards_enable_thinking_true():
    tokenizer = _RecordingTokenizer()
    pipeline = SimpleNamespace(tokenizer=tokenizer)

    tps_cli._resolve_text_measure_inputs(_measure_args(enable_thinking=True), pipeline)

    assert tokenizer.calls[0]["kwargs"].get("enable_thinking") is True


def test_tokenize_prompt_text_forwards_enable_thinking_false():
    tokenizer = _RecordingTokenizer()
    pipeline = SimpleNamespace(tokenizer=tokenizer)

    tps_cli._resolve_text_measure_inputs(_measure_args(enable_thinking=False), pipeline)

    assert tokenizer.calls[0]["kwargs"].get("enable_thinking") is False


def test_tokenize_prompt_text_falls_back_when_tokenizer_rejects_kwarg(capsys):
    tokenizer = _RecordingTokenizer(accepts_enable_thinking=False)
    pipeline = SimpleNamespace(tokenizer=tokenizer)

    tps_cli._resolve_text_measure_inputs(_measure_args(enable_thinking=False), pipeline)

    assert len(tokenizer.calls) == 1
    assert "enable_thinking" not in tokenizer.calls[0]["kwargs"]
    stderr = capsys.readouterr().err
    assert "--disable-thinking" in stderr


def test_tokenize_prompt_text_warns_when_chat_template_disabled(capsys):
    tokenizer = _RecordingTokenizer()
    pipeline = SimpleNamespace(tokenizer=tokenizer)

    tps_cli._resolve_text_measure_inputs(_measure_args(enable_thinking=True, apply_chat_template=False), pipeline)

    assert tokenizer.calls == []
    stderr = capsys.readouterr().err
    assert "--enable-thinking/--disable-thinking is ignored" in stderr


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
    monkeypatch.setattr(
        importlib.import_module("transformers"), "pipeline", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )

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


def test_build_pipeline_single_mode_can_omit_default_target_cores(monkeypatch) -> None:
    """Verify batch TPS paths can keep single-mode target cores unset."""
    monkeypatch.setattr(tps_cli, "_require_transformers_deps", lambda: None)
    monkeypatch.setattr(
        importlib.import_module("transformers"), "pipeline", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )

    pipe = tps_cli._build_pipeline(
        task="text-generation",
        model="dummy/model",
        tokenizer=None,
        device="cpu",
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
        eagle3_options=tps_cli.Eagle3PipelineOptions(),
        mxq_path=None,
        core_mode="single",
        target_cores=None,
        target_clusters=None,
        default_single_target_cores=None,
    )

    assert pipe.model_kwargs == {"core_mode": "single"}


def test_build_pipeline_single_mode_preserves_explicit_target_cores(monkeypatch) -> None:
    """Verify explicit target cores still override batch TPS default suppression."""
    monkeypatch.setattr(tps_cli, "_require_transformers_deps", lambda: None)
    monkeypatch.setattr(
        importlib.import_module("transformers"), "pipeline", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )

    pipe = tps_cli._build_pipeline(
        task="text-generation",
        model="dummy/model",
        tokenizer=None,
        device="cpu",
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
        eagle3_options=tps_cli.Eagle3PipelineOptions(),
        mxq_path=None,
        core_mode="single",
        target_cores=["0:1", "0:2"],
        target_clusters=None,
        default_single_target_cores=None,
    )

    assert pipe.model_kwargs == {
        "core_mode": "single",
        "target_cores": ["0:1", "0:2"],
    }


def test_default_single_target_cores_for_args_disables_explicit_batch_size() -> None:
    """Verify explicit batched TPS runs disable implicit single target cores."""
    assert tps_cli._default_single_target_cores_for_args(argparse.Namespace(batch_size=2)) is None
    assert tps_cli._default_single_target_cores_for_args(argparse.Namespace(batch_size=1)) == ("0:0",)
    assert tps_cli._default_single_target_cores_for_args(argparse.Namespace(batch_size=None)) == ("0:0",)


def test_default_single_target_cores_for_args_disables_list_dev_no() -> None:
    """List-shaped ``--dev-no`` skips the ``"0:0"`` sentinel so dev_no sugar drives expansion.

    The sentinel is a legacy 2-part core string that would migrate to a single-device
    canonical target under the backend setter's ``_fallback_dev()``. Combined with a
    multi-device ``--dev-no 0,1``, the resulting mismatch used to trigger a silent
    single-device pin; leaving ``target_cores`` unset lets sugar expansion cover both.
    """
    assert tps_cli._default_single_target_cores_for_args(argparse.Namespace(batch_size=1, dev_no=[0, 1])) is None
    # Scalar dev_no keeps the sentinel; the initial CLI default remains stable.
    assert tps_cli._default_single_target_cores_for_args(argparse.Namespace(batch_size=1, dev_no=0)) == ("0:0",)


def test_cli_tps_measure_dev_no_defaults_none() -> None:
    parser = build_parser()
    args = parser.parse_args(["tps", "measure", "--model", "mobilint/Llama-3.2-1B-Instruct"])
    assert args.dev_no is None
    assert args.base_dev_no is None
    assert args.draft_dev_no is None
    assert args.fc_dev_no is None
    assert args.vision_dev_no is None
    assert args.text_dev_no is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0", 0), ("1", 1), ("0,1", [0, 1]), ("2,3,4", [2, 3, 4])],
)
def test_cli_tps_measure_dev_no_accepts_scalar_and_list(value, expected) -> None:
    parser = build_parser()
    args = parser.parse_args(["tps", "measure", "--model", "mobilint/Llama-3.2-1B-Instruct", "--dev-no", value])
    assert args.dev_no == expected


@pytest.mark.parametrize("value", ["-1", "abc", "0,-1", "1,foo"])
def test_cli_tps_measure_dev_no_rejects_invalid(value) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["tps", "measure", "--model", "mobilint/Llama-3.2-1B-Instruct", "--dev-no", value])
    assert excinfo.value.code == 2


def test_cli_tps_sweep_dev_no_accepts_scalar() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["tps", "sweep", "--model", "mobilint/Llama-3.2-1B-Instruct", "--dev-no", "1", "--no-plot"]
    )
    assert args.dev_no == 1


def test_extract_eagle3_pipeline_kwargs_includes_dev_no() -> None:
    args = argparse.Namespace(
        base_embedding_path=None,
        draft_embedding_path=None,
        base_mxq_path=None,
        draft_mxq_path=None,
        fc_mxq_path=None,
        base_core_mode=None,
        draft_core_mode=None,
        fc_core_mode=None,
        base_target_cores=None,
        draft_target_cores=None,
        fc_target_cores=None,
        base_target_clusters=None,
        draft_target_clusters=None,
        fc_target_clusters=None,
        base_dev_no=1,
        draft_dev_no=[0, 1],
        fc_dev_no=0,
    )
    options = tps_cli._extract_eagle3_pipeline_kwargs(args)
    assert options.base_dev_no == 1
    assert options.draft_dev_no == [0, 1]
    assert options.fc_dev_no == 0


def test_extract_subconfig_pipeline_kwargs_includes_dev_no() -> None:
    args = argparse.Namespace(
        vision_core_mode=None,
        text_core_mode=None,
        vision_target_cores=None,
        text_target_cores=None,
        vision_target_clusters=None,
        text_target_clusters=None,
        vision_mxq_path=None,
        text_mxq_path=None,
        vision_dev_no=2,
        text_dev_no=[3, 4],
    )
    options = tps_cli._extract_subconfig_pipeline_kwargs(args)
    assert options.vision_dev_no == 2
    assert options.text_dev_no == [3, 4]


def test_build_pipeline_plain_sets_dev_no_when_scalar(monkeypatch) -> None:
    monkeypatch.setattr(tps_cli, "_require_transformers_deps", lambda: None)
    monkeypatch.setattr(
        importlib.import_module("transformers"), "pipeline", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )

    pipe = tps_cli._build_pipeline(
        task="text-generation",
        model="dummy/model",
        tokenizer=None,
        device="cpu",
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
        eagle3_options=tps_cli.Eagle3PipelineOptions(),
        mxq_path=None,
        core_mode=None,
        target_cores=None,
        target_clusters=None,
        default_single_target_cores=None,
        dev_no=1,
    )
    assert pipe.model_kwargs == {"dev_no": 1}


def test_build_pipeline_plain_sets_dev_no_when_list(monkeypatch) -> None:
    monkeypatch.setattr(tps_cli, "_require_transformers_deps", lambda: None)
    monkeypatch.setattr(
        importlib.import_module("transformers"), "pipeline", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )

    pipe = tps_cli._build_pipeline(
        task="text-generation",
        model="dummy/model",
        tokenizer=None,
        device="cpu",
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
        eagle3_options=tps_cli.Eagle3PipelineOptions(),
        mxq_path=None,
        core_mode=None,
        target_cores=None,
        target_clusters=None,
        default_single_target_cores=None,
        dev_no=[0, 1],
    )
    assert pipe.model_kwargs == {"dev_no": [0, 1]}


def test_build_pipeline_vlm_expands_dev_no_to_both_subconfigs(monkeypatch) -> None:
    monkeypatch.setattr(tps_cli, "_require_transformers_deps", lambda: None)
    monkeypatch.setattr(
        importlib.import_module("transformers"), "pipeline", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )

    pipe = tps_cli._build_pipeline(
        task="image-text-to-text",
        model="dummy/model",
        tokenizer=None,
        device="cpu",
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
        eagle3_options=tps_cli.Eagle3PipelineOptions(),
        mxq_path=None,
        core_mode=None,
        target_cores=None,
        target_clusters=None,
        default_single_target_cores=None,
        dev_no=2,
        subconfig_options=tps_cli.SubconfigPipelineOptions(),
    )
    assert pipe.model_kwargs == {"vision_dev_no": 2, "text_dev_no": 2}


def test_build_pipeline_vlm_text_dev_no_override_takes_precedence(monkeypatch) -> None:
    monkeypatch.setattr(tps_cli, "_require_transformers_deps", lambda: None)
    monkeypatch.setattr(
        importlib.import_module("transformers"), "pipeline", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )

    pipe = tps_cli._build_pipeline(
        task="image-text-to-text",
        model="dummy/model",
        tokenizer=None,
        device="cpu",
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
        eagle3_options=tps_cli.Eagle3PipelineOptions(),
        mxq_path=None,
        core_mode=None,
        target_cores=None,
        target_clusters=None,
        default_single_target_cores=None,
        dev_no=0,
        subconfig_options=tps_cli.SubconfigPipelineOptions(text_dev_no=3),
    )
    assert pipe.model_kwargs == {"vision_dev_no": 0, "text_dev_no": 3}


def test_build_pipeline_eagle3_dev_no_prefix_warns_and_coalesces(monkeypatch) -> None:
    monkeypatch.setattr(tps_cli, "_require_transformers_deps", lambda: None)
    monkeypatch.setattr(
        importlib.import_module("transformers"), "pipeline", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )

    with pytest.warns(UserWarning, match="Conflicting options detected"):
        pipe = tps_cli._build_pipeline(
            task="text-generation",
            model="dummy/model",
            tokenizer=None,
            device="cpu",
            trust_remote_code=True,
            dtype=None,
            device_map=None,
            revision=None,
            embedding_weight=None,
            eagle3_options=tps_cli.Eagle3PipelineOptions(base_dev_no=5),
            mxq_path=None,
            core_mode=None,
            target_cores=None,
            target_clusters=None,
            default_single_target_cores=None,
            dev_no=1,
        )

    assert pipe.model_kwargs["base_dev_no"] == 5
    assert pipe.model_kwargs["draft_dev_no"] == 1
    assert pipe.model_kwargs["fc_dev_no"] == 1


def test_is_mobilint_model_target_fast_path_repo_prefix(monkeypatch) -> None:
    """The ``mobilint/`` HuggingFace namespace short-circuits config resolution."""

    def _fail_autoconfig(*args, **kwargs):
        raise AssertionError("AutoConfig should not be consulted on the mobilint/* fast path")

    monkeypatch.setattr(importlib.import_module("transformers").AutoConfig, "from_pretrained", _fail_autoconfig)

    assert tps_cli._is_mobilint_model_target(
        "mobilint/Qwen3-4B-W4V8-Anything",
        trust_remote_code=True,
        revision=None,
    )


def test_is_mobilint_model_target_returns_false_when_config_load_fails(monkeypatch) -> None:
    """Any AutoConfig error (offline, wrong path, missing extras) is a safe non-Mobilint signal."""

    def _raise_from_pretrained(*args, **kwargs):
        raise OSError("simulated network failure")

    monkeypatch.setattr(importlib.import_module("transformers").AutoConfig, "from_pretrained", _raise_from_pretrained)

    assert not tps_cli._is_mobilint_model_target(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True,
        revision=None,
    )


def test_is_mobilint_model_target_isinstance_check(monkeypatch) -> None:
    """A resolved config that is a Mobilint mixin subclass triggers injection."""
    mixins = tps_cli._resolve_mobilint_config_mixins()
    if mixins is None:
        pytest.skip("mblt_model_zoo.hf_transformers is not importable in this environment")
    mobilint_mixin = mixins[0]

    class _StubMobilintConfig(mobilint_mixin):
        pass

    def _fake_from_pretrained(*args, **kwargs):
        # Skip full config __init__ so the stub does not require an NPU backend to construct.
        return _StubMobilintConfig.__new__(_StubMobilintConfig)

    monkeypatch.setattr(importlib.import_module("transformers").AutoConfig, "from_pretrained", _fake_from_pretrained)

    assert tps_cli._is_mobilint_model_target(
        "some-non-namespaced-checkpoint",
        trust_remote_code=True,
        revision=None,
    )


def _capture_pipeline_kwargs(monkeypatch) -> dict[str, object]:
    """Replace ``transformers.pipeline`` with a stub that records its kwargs."""
    monkeypatch.setattr(tps_cli, "_require_transformers_deps", lambda: None)
    captured: dict[str, object] = {}

    def _fake(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return types.SimpleNamespace(**kwargs)

    monkeypatch.setattr(importlib.import_module("transformers"), "pipeline", _fake)
    return captured


def test_build_pipeline_skips_max_batch_size_for_non_mobilint(monkeypatch) -> None:
    """A non-Mobilint model target must not receive backend-only ``max_batch_size``."""
    captured = _capture_pipeline_kwargs(monkeypatch)
    monkeypatch.setattr(
        tps_cli,
        "_is_mobilint_model_target",
        lambda model, *, trust_remote_code, revision: False,
    )

    tps_cli._build_pipeline(
        task="text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        tokenizer=None,
        device="cpu",
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
        eagle3_options=tps_cli.Eagle3PipelineOptions(),
        mxq_path=None,
        core_mode=None,
        target_cores=None,
        target_clusters=None,
        default_single_target_cores=None,
        max_batch_size=4,
    )

    # No Mobilint-only kwargs were requested and the target is non-Mobilint, so
    # backend-only fields must not reach the model constructor. --batch-size
    # still becomes the measurement batch size via the CLI's synthetic-input
    # path (verified by dedicated tests elsewhere).
    model_kwargs = captured.get("model_kwargs", {})
    assert "max_batch_size" not in model_kwargs
    assert "text_max_batch_size" not in model_kwargs
    assert "base_max_batch_size" not in model_kwargs


def test_build_pipeline_injects_max_batch_size_for_mobilint(monkeypatch) -> None:
    """A Mobilint model target continues to receive ``max_batch_size``."""
    captured = _capture_pipeline_kwargs(monkeypatch)
    monkeypatch.setattr(
        tps_cli,
        "_is_mobilint_model_target",
        lambda model, *, trust_remote_code, revision: True,
    )

    tps_cli._build_pipeline(
        task="text-generation",
        model="mobilint/Qwen3-4B-W4V8",
        tokenizer=None,
        device="cpu",
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
        eagle3_options=tps_cli.Eagle3PipelineOptions(),
        mxq_path=None,
        core_mode=None,
        target_cores=None,
        target_clusters=None,
        default_single_target_cores=None,
        max_batch_size=4,
    )

    model_kwargs = captured.get("model_kwargs", {})
    assert model_kwargs.get("max_batch_size") == 4


def test_build_pipeline_vlm_gates_text_max_batch_size(monkeypatch) -> None:
    """VLM path uses ``text_max_batch_size`` and must gate on the Mobilint check."""
    captured = _capture_pipeline_kwargs(monkeypatch)
    monkeypatch.setattr(
        tps_cli,
        "_is_mobilint_model_target",
        lambda model, *, trust_remote_code, revision: False,
    )

    tps_cli._build_pipeline(
        task="image-text-to-text",
        model="google/gemma-3-vlm",  # placeholder non-Mobilint VLM string
        tokenizer=None,
        device="cpu",
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
        eagle3_options=tps_cli.Eagle3PipelineOptions(),
        mxq_path=None,
        core_mode=None,
        target_cores=None,
        target_clusters=None,
        default_single_target_cores=None,
        subconfig_options=tps_cli.SubconfigPipelineOptions(),
        max_batch_size=2,
    )

    assert "text_max_batch_size" not in captured.get("model_kwargs", {})

    monkeypatch.setattr(
        tps_cli,
        "_is_mobilint_model_target",
        lambda model, *, trust_remote_code, revision: True,
    )

    tps_cli._build_pipeline(
        task="image-text-to-text",
        model="mobilint/Qwen3-VL-8B",
        tokenizer=None,
        device="cpu",
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
        eagle3_options=tps_cli.Eagle3PipelineOptions(),
        mxq_path=None,
        core_mode=None,
        target_cores=None,
        target_clusters=None,
        default_single_target_cores=None,
        subconfig_options=tps_cli.SubconfigPipelineOptions(),
        max_batch_size=2,
    )

    assert captured.get("model_kwargs", {}).get("text_max_batch_size") == 2


def test_build_pipeline_eagle3_gates_base_max_batch_size(monkeypatch) -> None:
    """EAGLE-3 path uses ``base_max_batch_size`` and must gate on the Mobilint check."""
    captured = _capture_pipeline_kwargs(monkeypatch)
    monkeypatch.setattr(
        tps_cli,
        "_is_mobilint_model_target",
        lambda model, *, trust_remote_code, revision: True,
    )

    tps_cli._build_pipeline(
        task="text-generation",
        model="mobilint/EAGLE3-Qwen3-4B",
        tokenizer=None,
        device="cpu",
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
        eagle3_options=tps_cli.Eagle3PipelineOptions(base_mxq_path="base.mxq"),
        mxq_path=None,
        core_mode=None,
        target_cores=None,
        target_clusters=None,
        default_single_target_cores=None,
        max_batch_size=8,
    )

    model_kwargs = captured.get("model_kwargs", {})
    assert model_kwargs.get("base_max_batch_size") == 8
    assert "max_batch_size" not in model_kwargs


def test_build_pipeline_eagle3_broadcasts_bare_dev_no(monkeypatch) -> None:
    """A global --dev-no on an EAGLE-3 release must broadcast to base_/draft_/fc_ prefixes.

    MobilintEagle3ConfigMixin exposes only prefixed dev_no setters, so an unprefixed
    ``dev_no`` model kwarg would otherwise be silently dropped by HF ``from_pretrained``.
    """
    monkeypatch.setattr(tps_cli, "_require_transformers_deps", lambda: None)
    monkeypatch.setattr(
        importlib.import_module("transformers"), "pipeline", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(
        tps_cli,
        "_detect_eagle3_model",
        lambda model, *, trust_remote_code, revision: True,
    )

    pipe = tps_cli._build_pipeline(
        task="text-generation",
        model="dummy/eagle3",
        tokenizer=None,
        device="cpu",
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
        eagle3_options=tps_cli.Eagle3PipelineOptions(),
        mxq_path=None,
        core_mode=None,
        target_cores=None,
        target_clusters=None,
        default_single_target_cores=None,
        dev_no=1,
    )

    assert "dev_no" not in pipe.model_kwargs
    assert pipe.model_kwargs["base_dev_no"] == 1
    assert pipe.model_kwargs["draft_dev_no"] == 1
    assert pipe.model_kwargs["fc_dev_no"] == 1


def test_build_pipeline_eagle3_bare_dev_no_respects_explicit_prefix(monkeypatch) -> None:
    """Explicit --draft-dev-no still wins when the global --dev-no is broadcast for EAGLE-3."""
    monkeypatch.setattr(tps_cli, "_require_transformers_deps", lambda: None)
    monkeypatch.setattr(
        importlib.import_module("transformers"), "pipeline", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    # Explicit prefixed option already triggers the EAGLE-3 branch; no need to detect the config.
    monkeypatch.setattr(
        tps_cli,
        "_detect_eagle3_model",
        lambda *args, **kwargs: pytest.fail("_detect_eagle3_model should not run when prefixed options are set"),
    )

    with pytest.warns(UserWarning, match="Conflicting options detected"):
        pipe = tps_cli._build_pipeline(
            task="text-generation",
            model="dummy/eagle3",
            tokenizer=None,
            device="cpu",
            trust_remote_code=True,
            dtype=None,
            device_map=None,
            revision=None,
            embedding_weight=None,
            eagle3_options=tps_cli.Eagle3PipelineOptions(draft_dev_no=7),
            mxq_path=None,
            core_mode=None,
            target_cores=None,
            target_clusters=None,
            default_single_target_cores=None,
            dev_no=1,
        )

    assert "dev_no" not in pipe.model_kwargs
    assert pipe.model_kwargs["base_dev_no"] == 1
    assert pipe.model_kwargs["draft_dev_no"] == 7
    assert pipe.model_kwargs["fc_dev_no"] == 1


def test_build_pipeline_non_eagle3_keeps_unprefixed_dev_no(monkeypatch) -> None:
    """A non-EAGLE-3 model must continue to receive an unprefixed ``dev_no`` model kwarg."""
    monkeypatch.setattr(tps_cli, "_require_transformers_deps", lambda: None)
    monkeypatch.setattr(
        importlib.import_module("transformers"), "pipeline", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(
        tps_cli,
        "_detect_eagle3_model",
        lambda model, *, trust_remote_code, revision: False,
    )

    pipe = tps_cli._build_pipeline(
        task="text-generation",
        model="dummy/plain",
        tokenizer=None,
        device="cpu",
        trust_remote_code=True,
        dtype=None,
        device_map=None,
        revision=None,
        embedding_weight=None,
        eagle3_options=tps_cli.Eagle3PipelineOptions(),
        mxq_path=None,
        core_mode=None,
        target_cores=None,
        target_clusters=None,
        default_single_target_cores=None,
        dev_no=1,
    )

    assert pipe.model_kwargs == {"dev_no": 1}


def test_is_eagle3_config_detects_model_type_marker() -> None:
    """`_is_eagle3_config` recognizes the ``eagle3`` marker in ``model_type`` and ``architectures``."""

    assert tps_cli._is_eagle3_config(SimpleNamespace(model_type="qwen3_eagle3"))
    assert tps_cli._is_eagle3_config(SimpleNamespace(model_type="qwen3", architectures=["Qwen3ForCausalLMEagle3"]))
    assert not tps_cli._is_eagle3_config(SimpleNamespace(model_type="qwen3", architectures=["Qwen3ForCausalLM"]))
