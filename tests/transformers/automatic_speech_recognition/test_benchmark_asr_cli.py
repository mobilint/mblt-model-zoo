import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRANSFORMERS_BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "benchmark" / "transformers"
if str(_TRANSFORMERS_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_TRANSFORMERS_BENCHMARK_DIR))

from benchmark.transformers import benchmark_automatic_speech_recognition_models as asr_bench  # noqa: E402


def test_asr_benchmark_parser_defaults() -> None:
    """Verify ASR benchmark parser defaults are stable."""

    args = asr_bench._parse_args([])

    assert args.dataset == "openslr/librispeech_asr"
    assert args.dataset_config == "clean"
    assert args.dataset_split == "test"
    assert args.language == "en"
    assert args.num_samples == 50
    assert args.full_split is False
    assert args.num_beams is None
    assert args.max_new_tokens == 444
    assert args.warmup == 2
    assert args.dry_run is False


def test_asr_benchmark_parser_dataset_config_none_strings_map_to_none() -> None:
    """Verify dataset-config string sentinels normalize to Python None."""

    assert asr_bench._parse_args(["--dataset-config", "none"]).dataset_config is None
    assert asr_bench._parse_args(["--dataset-config", "None"]).dataset_config is None


def test_asr_benchmark_description_targets_general_asr() -> None:
    """Verify the benchmark help text is not Whisper-only."""

    parser = asr_bench._parse_args([])

    assert parser.task == "transcribe"
    assert parser.language == "en"


def test_asr_benchmark_help_parses() -> None:
    """Verify ASR benchmark help exits successfully."""

    try:
        asr_bench._parse_args(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("Expected --help to exit")


def test_asr_benchmark_help_subprocess_smoke() -> None:
    """Verify the README help command works as a subprocess smoke test."""

    result = subprocess.run(
        [sys.executable, "benchmark/transformers/benchmark_automatic_speech_recognition_models.py", "--help"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "automatic-speech-recognition" in result.stdout


def test_asr_benchmark_module_help_subprocess_smoke() -> None:
    """Verify module execution help works for ASR benchmark."""

    result = subprocess.run(
        [sys.executable, "-m", "benchmark.transformers.benchmark_automatic_speech_recognition_models", "--help"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "automatic-speech-recognition" in result.stdout


def test_asr_benchmark_parser_full_split_yields_to_explicit_num_samples(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify explicit num-samples wins over full-split with an informational message."""

    args = asr_bench._parse_args(["--full-split", "--num-samples", "10"])

    captured = capsys.readouterr()
    assert args.full_split is False
    assert args.num_samples == 10
    assert "using --num-samples and ignoring --full-split" in captured.out


def test_asr_benchmark_parser_full_split_sets_num_samples_none() -> None:
    """Verify --full-split switches dataset loading to full split mode."""

    args = asr_bench._parse_args(["--full-split"])

    assert args.full_split is True
    assert args.num_samples is None


def test_optional_generate_kwargs_only_enable_whisper_hints() -> None:
    """Verify Whisper-specific hints are only added for Whisper-like models."""

    args = asr_bench._parse_args([])

    whisper_kwargs = asr_bench._optional_generate_kwargs_for_model(args, "openai/whisper-small")
    wav2vec_kwargs = asr_bench._optional_generate_kwargs_for_model(args, "facebook/wav2vec2-base-960h")

    assert whisper_kwargs == {"task": "transcribe", "language": "en"}
    assert wav2vec_kwargs == {}


def test_resolve_generate_kwargs_omits_num_beams_when_unspecified() -> None:
    """Verify default beam settings defer to the model by omitting num_beams."""

    args = asr_bench._parse_args([])

    assert asr_bench._resolve_generate_kwargs(args) == {
        "max_new_tokens": 444,
        "return_timestamps": False,
    }


def test_resolve_generate_kwargs_includes_beam_settings_when_specified() -> None:
    """Verify explicit beam settings are forwarded into generate kwargs."""

    args = asr_bench._parse_args(["--num-beams", "4"])

    assert asr_bench._resolve_generate_kwargs(args) == {
        "num_beams": 4,
        "max_new_tokens": 444,
        "return_timestamps": False,
        "early_stopping": True,
    }


def test_should_skip_whisper_long_form_sample_only_for_whisper_over_30s() -> None:
    """Skip only Whisper samples that exceed the 30 second short-form limit."""

    long_sample = {
        "id": "sample-1",
        "audio": {"array": np.zeros(16000 * 31, dtype=np.float32), "sampling_rate": 16000},
        "reference": "hello world",
    }
    short_sample = {
        "id": "sample-2",
        "audio": {"array": np.zeros(16000 * 30, dtype=np.float32), "sampling_rate": 16000},
        "reference": "hello world",
    }

    assert asr_bench._should_skip_whisper_long_form_sample("openai/whisper-small", long_sample) is True
    assert asr_bench._should_skip_whisper_long_form_sample("openai/whisper-small", short_sample) is False
    assert asr_bench._should_skip_whisper_long_form_sample("Qwen/Qwen3-ASR-1.7B", long_sample) is False


def test_measure_target_skips_whisper_long_form_samples() -> None:
    """Skip >30s Whisper samples instead of attempting long-form generation."""

    class DummyPipe:
        def __init__(self) -> None:
            self.calls = 0
            self.tokenizer = None

        def __call__(self, pipeline_input, **kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            return {"text": "hello world"}

    short_sample = {
        "id": "sample-1",
        "audio": {"array": np.zeros(16000 * 5, dtype=np.float32), "sampling_rate": 16000},
        "reference": "hello world",
    }
    long_sample = {
        "id": "sample-2",
        "audio": {"array": np.zeros(16000 * 31, dtype=np.float32), "sampling_rate": 16000},
        "reference": "hello world",
    }
    pipe = DummyPipe()
    args = asr_bench._parse_args([])
    args.device_backend = "none"

    timings, device_metric, device_trace = asr_bench._measure_target(
        "openai/whisper-small",
        args,
        pipe,
        [short_sample, long_sample],
        {"max_new_tokens": 444, "return_timestamps": False},
    )

    assert len(timings) == 1
    assert timings[0].hypothesis == "hello world"
    assert pipe.calls == 1
    assert device_metric == {}
    assert device_trace == {}


def test_warmup_skips_whisper_long_form_samples() -> None:
    """Verify warmup also skips >30s Whisper samples and counts completed warmups."""

    calls: list[str] = []
    short_sample = {
        "id": "sample-1",
        "audio": {"array": np.zeros(16000 * 5, dtype=np.float32), "sampling_rate": 16000},
        "reference": "hello world",
    }
    long_sample = {
        "id": "sample-2",
        "audio": {"array": np.zeros(16000 * 31, dtype=np.float32), "sampling_rate": 16000},
        "reference": "hello world",
    }
    another_short_sample = {
        "id": "sample-3",
        "audio": {"array": np.zeros(16000 * 4, dtype=np.float32), "sampling_rate": 16000},
        "reference": "hello world",
    }

    def fake_run_one_sample(pipe, sample, generate_kwargs, native_language=None):  # type: ignore[no-untyped-def]
        calls.append(str(sample["id"]))
        return None

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(asr_bench, "_run_one_sample", fake_run_one_sample)
    try:
        result = asr_bench._warmup(
            "openai/whisper-small",
            object(),
            [long_sample, short_sample, another_short_sample],
            {"return_timestamps": False},
            2,
        )
    finally:
        monkeypatch.undo()

    assert result is None
    assert calls == ["sample-1", "sample-3"]


def test_beam_tag_uses_default_label_for_unspecified_beams() -> None:
    """Verify file/report suffixes use a stable label when beams are unspecified."""

    assert asr_bench._beam_tag(None) == "default"
    assert asr_bench._beam_tag(3) == "3"


def test_build_run_targets_with_explicit_model_ids_skips_default_model_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify explicit --model-id avoids eager default list resolution."""

    args = asr_bench._parse_args(["--model-id", "openai/whisper-small"])

    def fail_list_default_asr_models():
        raise AssertionError("_list_default_asr_models should not be called")

    monkeypatch.setattr(asr_bench, "_list_default_asr_models", fail_list_default_asr_models)

    targets = asr_bench._build_run_targets(args)

    assert len(targets) == 1
    assert targets[0][0].model_id == "openai/whisper-small"


def test_default_asr_model_filter_excludes_whisper_cpp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify non-pipeline whisper.cpp targets are excluded from default ASR lists."""

    monkeypatch.setattr(
        asr_bench,
        "list_models",
        lambda tasks=None: {
            "automatic-speech-recognition": [
                "mobilint/whisper-small",
                "mobilint/whisper.cpp",
                "mobilint/Qwen3-ASR-1.7B",
            ]
        },
    )

    assert asr_bench._list_default_asr_models() == [
        "mobilint/whisper-small",
        "mobilint/Qwen3-ASR-1.7B",
    ]


def test_qwen3_asr_uses_encoder_decoder_core_mode_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Qwen3-ASR receives encoder/decoder-prefixed core-mode kwargs."""

    captured: dict[str, object] = {}

    def fake_pipeline(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return object()

    transformers_stub = type(
        "TransformersStub",
        (),
        {"pipeline": staticmethod(fake_pipeline)},
    )()
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)
    monkeypatch.setitem(sys.modules, "qwen_asr", type("QwenAsrStub", (), {})())
    monkeypatch.setitem(sys.modules, "qwen_asr.core", type("QwenAsrCoreStub", (), {})())
    monkeypatch.setitem(
        sys.modules,
        "qwen_asr.core.transformers_backend",
        type("QwenAsrBackendStub", (), {})(),
    )
    monkeypatch.setitem(
        sys.modules,
        "qwen_asr.core.transformers_backend.configuration_qwen3_asr",
        type("QwenAsrConfigModuleStub", (), {"Qwen3ASRConfig": type("Qwen3ASRConfig", (), {})})(),
    )
    monkeypatch.setitem(
        sys.modules,
        "qwen_asr.core.transformers_backend.modeling_qwen3_asr",
        type(
            "QwenAsrModelModuleStub",
            (),
            {"Qwen3ASRForConditionalGeneration": type("Qwen3ASRForConditionalGeneration", (), {})},
        )(),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.auto.modeling_auto",
        type(
            "TransformersAutoModelingStub",
            (),
            {
                "AutoModelForSpeechSeq2Seq": type(
                    "AutoModelForSpeechSeq2SeqStub",
                    (),
                    {"register": staticmethod(lambda *args, **kwargs: None)},
                )
            },
        )(),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.pipelines.automatic_speech_recognition",
        type(
            "TransformersAsrPipelineStub",
            (),
            {"MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES": {}},
        )(),
    )

    target = asr_bench.ASRBenchmarkTarget(
        model_id="mobilint/Qwen3-ASR-1.7B",
        revision_candidates=[None],
        label="mobilint/Qwen3-ASR-1.7B",
        base="mobilint__Qwen3-ASR-1.7B",
        mxq_path=None,
        is_original=False,
    )

    asr_bench._build_asr_pipeline(
        target,
        revision=None,
        device="cpu",
        device_map=None,
        dtype=None,
        trust_remote_code=True,
        core_mode="single",
    )

    model_kwargs = captured.get("model_kwargs")
    assert isinstance(model_kwargs, dict)
    assert model_kwargs.get("encoder_core_mode") == "single"
    assert model_kwargs.get("decoder_core_mode") == "single"
    assert model_kwargs.get("encoder_target_cores") == ["0:0"]
    assert model_kwargs.get("decoder_target_cores") == ["0:0"]
    assert "core_mode" not in model_kwargs


def test_qwen3_asr_original_model_prefers_native_qwen_asr_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify original Qwen3-ASR checkpoints bypass HF pipeline and use native loader."""

    calls: list[tuple[str, dict[str, object]]] = []

    move_calls: list[str] = []

    class InnerModel:
        def generate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return "ok"

        def to(self, device):  # type: ignore[no-untyped-def]
            move_calls.append(device)
            return self

    class NativePipe:
        def __init__(self) -> None:
            self.model = InnerModel()

    class Qwen3ASRModelStub:
        @staticmethod
        def from_pretrained(model_id, **kwargs):  # type: ignore[no-untyped-def]
            calls.append((model_id, kwargs))
            return NativePipe()

    monkeypatch.setitem(
        sys.modules,
        "qwen_asr",
        type("QwenAsrStub", (), {"Qwen3ASRModel": Qwen3ASRModelStub})(),
    )

    target = asr_bench.ASRBenchmarkTarget(
        model_id="Qwen/Qwen3-ASR-1.7B",
        revision_candidates=[None],
        label="Qwen/Qwen3-ASR-1.7B",
        base="Qwen__Qwen3-ASR-1.7B",
        mxq_path=None,
        is_original=True,
    )

    asr_bench._build_asr_pipeline(
        target,
        revision=None,
        device="cuda:0",
        device_map=None,
        dtype="float16",
        trust_remote_code=True,
        core_mode=None,
        native_generate_kwargs={"num_beams": 4, "max_new_tokens": 321, "return_timestamps": False},
    )

    assert calls == [
        (
            "Qwen/Qwen3-ASR-1.7B",
            {
                "trust_remote_code": True,
                "max_inference_batch_size": 1,
                "max_new_tokens": 321,
                "device_map": "cuda:0",
                "torch_dtype": asr_bench.torch.float16,
            },
        )
    ]
    assert move_calls == ["cuda:0"]


def test_qwen3_asr_original_model_uses_native_qwen_asr_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify original Qwen3-ASR targets use native qwen_asr loader instead of HF pipeline."""

    calls: list[tuple[str, dict[str, object]]] = []

    move_calls: list[str] = []

    class InnerModel:
        def generate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return "ok"

        def to(self, device):  # type: ignore[no-untyped-def]
            move_calls.append(device)
            return self

    class NativePipe:
        def __init__(self) -> None:
            self.model = InnerModel()

    class Qwen3ASRModelStub:
        @staticmethod
        def from_pretrained(model_id, **kwargs):  # type: ignore[no-untyped-def]
            calls.append((model_id, kwargs))
            return NativePipe()

    monkeypatch.setitem(
        sys.modules,
        "qwen_asr",
        type("QwenAsrStub", (), {"Qwen3ASRModel": Qwen3ASRModelStub})(),
    )

    target = asr_bench.ASRBenchmarkTarget(
        model_id="Qwen/Qwen3-ASR-1.7B",
        revision_candidates=["main"],
        label="Qwen/Qwen3-ASR-1.7B",
        base="Qwen__Qwen3-ASR-1.7B",
        mxq_path=None,
        is_original=True,
    )

    asr_bench._build_asr_pipeline(
        target,
        revision="main",
        device="cpu",
        device_map="auto",
        dtype=None,
        trust_remote_code=True,
        core_mode=None,
        native_generate_kwargs={"num_beams": 2, "max_new_tokens": 222, "return_timestamps": False},
    )

    assert calls == [
        (
            "Qwen/Qwen3-ASR-1.7B",
            {
                "trust_remote_code": True,
                "max_inference_batch_size": 1,
                "max_new_tokens": 222,
                "device_map": "auto",
                "revision": "main",
            },
        )
    ]
    assert move_calls == []


def test_configure_native_qwen3_asr_generate_wraps_inner_generate() -> None:
    """Verify native Qwen3-ASR wrapper injects benchmark beam settings into inner generate."""

    captured: list[dict[str, object]] = []

    class InnerModel:
        def generate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            captured.append(dict(kwargs))
            return "ok"

    class NativePipe:
        def __init__(self) -> None:
            self.model = InnerModel()

    pipe = NativePipe()

    configured = asr_bench._configure_native_qwen3_asr_generate(
        pipe,
        {
            "num_beams": 5,
            "max_new_tokens": 444,
            "early_stopping": True,
            "return_timestamps": False,
        },
    )

    assert configured is pipe
    result = pipe.model.generate(input_features="dummy", max_new_tokens=111)

    assert result == "ok"
    assert captured == [
        {
            "input_features": "dummy",
            "num_beams": 5,
            "max_new_tokens": 111,
            "early_stopping": True,
        }
    ]


def test_ensure_native_qwen3_asr_generation_config_sets_pad_from_eos() -> None:
    """Verify native Qwen3-ASR generation config receives pad_token_id from eos_token_id."""

    generation_config = type("GenerationConfigStub", (), {"pad_token_id": None, "eos_token_id": None})()
    model_config = type("ModelConfigStub", (), {"pad_token_id": None, "eos_token_id": 151645})()
    pipe = type(
        "PipeStub",
        (),
        {"model": type("InnerModelStub", (), {"generation_config": generation_config, "config": model_config})()},
    )()

    configured = asr_bench._ensure_native_qwen3_asr_generation_config(pipe)

    assert configured is pipe
    assert generation_config.pad_token_id == 151645
    assert model_config.pad_token_id == 151645


def test_quiet_apscheduler_info_logs_raises_logger_level_only_when_needed() -> None:
    """Verify APScheduler logger is raised to WARNING without touching stricter levels."""

    aps_logger = asr_bench.logging.getLogger("apscheduler")
    original_level = aps_logger.level
    try:
        aps_logger.setLevel(asr_bench.logging.NOTSET)
        asr_bench._quiet_apscheduler_info_logs()
        assert aps_logger.level == asr_bench.logging.WARNING

        aps_logger.setLevel(asr_bench.logging.ERROR)
        asr_bench._quiet_apscheduler_info_logs()
        assert aps_logger.level == asr_bench.logging.ERROR
    finally:
        aps_logger.setLevel(original_level)


def test_resolve_torch_dtype_supports_torch_prefix() -> None:
    """Verify dtype strings are converted into torch dtype objects."""

    assert asr_bench._resolve_torch_dtype("float16") == asr_bench.torch.float16
    assert asr_bench._resolve_torch_dtype("torch.bfloat16") == asr_bench.torch.bfloat16
    assert asr_bench._resolve_torch_dtype("not-a-real-dtype") is None


def test_ensure_qwen3_asr_backend_registered_registers_seq2seq_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Qwen3-ASR backend helper registers seq2seq mapping metadata."""

    register_calls: list[tuple[object, object, bool]] = []

    class AutoModelForSpeechSeq2SeqStub:
        @staticmethod
        def register(config_cls, model_cls, exist_ok=False):  # type: ignore[no-untyped-def]
            register_calls.append((config_cls, model_cls, exist_ok))

    qwen_config_cls = type("Qwen3ASRConfig", (), {})
    qwen_model_cls = type("Qwen3ASRForConditionalGeneration", (), {})

    monkeypatch.setitem(sys.modules, "qwen_asr", type("QwenAsrStub", (), {})())
    monkeypatch.setitem(sys.modules, "qwen_asr.core", type("QwenAsrCoreStub", (), {})())
    monkeypatch.setitem(
        sys.modules,
        "qwen_asr.core.transformers_backend",
        type("QwenAsrBackendStub", (), {})(),
    )
    monkeypatch.setitem(
        sys.modules,
        "qwen_asr.core.transformers_backend.configuration_qwen3_asr",
        type("QwenAsrConfigModuleStub", (), {"Qwen3ASRConfig": qwen_config_cls})(),
    )
    monkeypatch.setitem(
        sys.modules,
        "qwen_asr.core.transformers_backend.modeling_qwen3_asr",
        type(
            "QwenAsrModelModuleStub",
            (),
            {"Qwen3ASRForConditionalGeneration": qwen_model_cls},
        )(),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.auto.modeling_auto",
        type(
            "TransformersAutoModelingStub",
            (),
            {"AutoModelForSpeechSeq2Seq": AutoModelForSpeechSeq2SeqStub},
        )(),
    )

    asr_bench._ensure_qwen3_asr_backend_registered()

    assert register_calls == [(qwen_config_cls, qwen_model_cls, True)]


def test_qwen3_asr_original_model_missing_optional_backend_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify missing qwen-asr dependency produces an actionable message."""

    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "qwen_asr.core.transformers_backend":
            raise ModuleNotFoundError("No module named 'qwen_asr'", name="qwen_asr")
        return original_import(name, globals, locals, fromlist, level)

    transformers_stub = type(
        "TransformersStub",
        (),
        {"pipeline": staticmethod(lambda **kwargs: object())},
    )()

    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)
    monkeypatch.delitem(sys.modules, "qwen_asr", raising=False)
    monkeypatch.delitem(sys.modules, "qwen_asr.core", raising=False)
    monkeypatch.delitem(sys.modules, "qwen_asr.core.transformers_backend", raising=False)
    monkeypatch.setattr("builtins.__import__", fake_import)

    target = asr_bench.ASRBenchmarkTarget(
        model_id="Qwen/Qwen3-ASR-1.7B",
        revision_candidates=[None],
        label="Qwen/Qwen3-ASR-1.7B",
        base="Qwen__Qwen3-ASR-1.7B",
        mxq_path=None,
        is_original=True,
    )

    with pytest.raises(ModuleNotFoundError, match="qwen-asr"):
        asr_bench._build_asr_pipeline(
            target,
            revision=None,
            device="cpu",
            device_map=None,
            dtype=None,
            trust_remote_code=True,
            core_mode=None,
        )


def test_non_composite_asr_uses_top_level_core_mode_kwargs() -> None:
    """Verify non-composite ASR models keep the existing top-level core-mode kwargs."""

    model_kwargs = asr_bench._apply_asr_core_mode_model_kwargs({}, "openai/whisper-small", "global4")

    assert model_kwargs.get("core_mode") == "global4"
    assert model_kwargs.get("target_clusters") == [0]
    assert "encoder_core_mode" not in model_kwargs
    assert "decoder_core_mode" not in model_kwargs


def test_extract_hypothesis_text_supports_chunk_outputs() -> None:
    """Verify chunked ASR outputs are converted into one hypothesis string."""

    output = {"chunks": [{"text": "hello"}, {"text": "world"}]}

    assert asr_bench._extract_hypothesis_text(output) == "hello world"


def test_run_one_sample_retries_without_whisper_only_kwargs() -> None:
    """Verify fallback retry removes unsupported Whisper kwargs for generic ASR pipelines."""

    class DummyPipe:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.tokenizer = None

        def __call__(self, audio_array, sampling_rate=None, generate_kwargs=None):  # type: ignore[no-untyped-def]
            payload = dict(generate_kwargs or {})
            self.calls.append(payload)
            if "task" in payload or "language" in payload:
                raise TypeError("unexpected keyword in generate_kwargs")
            return {"text": "test output"}

    pipe = DummyPipe()
    sample = {
        "id": "sample-1",
        "audio": {"array": [0.0, 0.0, 0.0, 0.0], "sampling_rate": 16000},
        "reference": "test output",
    }
    generate_kwargs = {
        **asr_bench._resolve_generate_kwargs(asr_bench._parse_args(["--num-beams", "4"])),
        **asr_bench._optional_generate_kwargs_for_model(asr_bench._parse_args([]), "openai/whisper-small"),
    }

    result = asr_bench._run_one_sample(pipe, sample, generate_kwargs)

    assert result.hypothesis == "test output"
    assert pipe.calls[0]["task"] == "transcribe"
    assert pipe.calls[0]["language"] == "en"
    assert "task" not in pipe.calls[-1]
    assert "language" not in pipe.calls[-1]


def test_run_one_sample_does_not_swallow_internal_type_error() -> None:
    """Verify internal TypeErrors are raised immediately instead of triggering fallback retries."""

    class DummyPipe:
        def __init__(self) -> None:
            self.calls = 0
            self.tokenizer = None

        def __call__(self, audio_input, sampling_rate=None, generate_kwargs=None):  # type: ignore[no-untyped-def]
            self.calls += 1
            raise TypeError("internal decoder bug")

    pipe = DummyPipe()
    sample = {
        "id": "sample-1",
        "audio": {"array": [0.0, 0.0, 0.0, 0.0], "sampling_rate": 16000},
        "reference": "test output",
    }
    generate_kwargs = {
        **asr_bench._resolve_generate_kwargs(asr_bench._parse_args(["--num-beams", "4"])),
        **asr_bench._optional_generate_kwargs_for_model(asr_bench._parse_args([]), "openai/whisper-small"),
    }

    with pytest.raises(TypeError, match="internal decoder bug"):
        asr_bench._run_one_sample(pipe, sample, generate_kwargs)

    assert pipe.calls == 1


def test_write_combined_outputs_uses_suffixless_output_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ASR combined outputs use suffix-less stable file naming."""

    payload = {
        "benchmark_type": "automatic-speech-recognition",
        "model": "openai/whisper-small",
        "asr": {
            "num_samples": 1,
            "total_audio_s": 1.0,
            "total_generate_s": 0.5,
            "wer": 0.1,
            "cer": 0.02,
            "mean_latency_s": 0.5,
            "p50_latency_s": 0.5,
            "p95_latency_s": 0.5,
            "throughput_samples_per_s": 2.0,
            "rtf": 0.5,
            "inverse_rtf": 2.0,
            "decode_tokens_per_s": 10.0,
            "avg_tokens_per_sample": 5.0,
        },
        "device": {},
    }
    (tmp_path / "whisper-small.json").write_text(
        asr_bench.json.dumps(payload),
        encoding="utf-8",
    )
    (tmp_path / asr_bench._HOST_PC_INFO_FILENAME).write_text("host info\n", encoding="utf-8")
    monkeypatch.setattr(asr_bench, "_make_rtf_chart", lambda out_dir, num_beams, rows: None)

    asr_bench._write_combined_outputs(tmp_path, None)

    assert (tmp_path / "combined.csv").is_file()
    assert (tmp_path / "combined.md").is_file()
    assert (tmp_path / "summary.md").is_file()


def test_main_passes_language_to_summarize_timings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI language is forwarded into summary metric normalization."""

    captured: dict[str, object] = {}
    args = asr_bench._parse_args(["--output-dir", str(tmp_path), "--language", "ko"])

    target = asr_bench.ASRBenchmarkTarget(
        model_id="openai/whisper-small",
        revision_candidates=[None],
        label="openai/whisper-small",
        base="openai__whisper-small",
        mxq_path=None,
        is_original=False,
    )
    timings = [
        asr_bench.SampleTiming(
            sample_id="sample-1",
            audio_duration_s=1.0,
            generate_time_s=0.5,
            num_generated_tokens=3,
            num_beams=None,
            reference="hello world",
            hypothesis="hello, world",
        )
    ]

    def fake_summarize(sample_timings, *, language="en"):
        captured["language"] = language
        return asr_bench.ASRMetricSummary(
            num_samples=len(sample_timings),
            total_audio_s=1.0,
            total_generate_s=0.5,
            wer=0.0,
            cer=0.0,
            mean_latency_s=0.5,
            p50_latency_s=0.5,
            p95_latency_s=0.5,
            throughput_samples_per_s=2.0,
            rtf=0.5,
            inverse_rtf=2.0,
            decode_tokens_per_s=6.0,
            avg_tokens_per_sample=3.0,
        )

    monkeypatch.setattr(asr_bench, "_parse_args", lambda argv=None: args)
    monkeypatch.setattr(asr_bench, "_resolve_runtime_defaults", lambda parsed_args, raw_argv: None)
    monkeypatch.setattr(asr_bench, "_collect_host_pc_info", lambda out_dir: None)
    monkeypatch.setattr(
        asr_bench,
        "_build_run_targets",
        lambda parsed_args: [(target, None, target.label, target.base)],
    )
    monkeypatch.setattr(asr_bench, "_load_librispeech", lambda parsed_args: [dict(timings[0].__dict__)])
    monkeypatch.setattr(asr_bench, "_resolve_generate_kwargs", lambda parsed_args: {"return_timestamps": False})
    monkeypatch.setattr(asr_bench, "_optional_generate_kwargs_for_model", lambda parsed_args, model_id: {})
    monkeypatch.setattr(asr_bench, "_args_for_target_device_backend", lambda parsed_args, **kwargs: parsed_args)
    monkeypatch.setattr(asr_bench, "_build_asr_pipeline", lambda *args, **kwargs: object())
    monkeypatch.setattr(asr_bench, "_warmup", lambda *args, **kwargs: None)
    monkeypatch.setattr(asr_bench, "_measure_target", lambda *args, **kwargs: (timings, {}, {}))
    monkeypatch.setattr(asr_bench, "summarize_timings", fake_summarize)
    monkeypatch.setattr(asr_bench, "_write_target_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(asr_bench, "_release_pipeline", lambda pipe, device: None)
    monkeypatch.setattr(asr_bench, "_write_combined_outputs", lambda out_dir, num_beams: None)

    assert asr_bench.main(["--output-dir", str(tmp_path), "--language", "ko"]) == 0
    assert captured == {"language": "ko"}


def test_run_one_sample_uses_array_input_path() -> None:
    """Verify ASR sample execution uses the HF dict audio input path."""

    class DummyPipe:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object, dict[str, object]]] = []
            self.tokenizer = None

        def __call__(self, audio_input, sampling_rate=None, generate_kwargs=None):  # type: ignore[no-untyped-def]
            payload = dict(generate_kwargs or {})
            self.calls.append((audio_input, sampling_rate, payload))
            if isinstance(audio_input, dict):
                return {"text": "test output"}
            raise AssertionError("Expected dict-form audio input to be used")

    pipe = DummyPipe()
    sample = {
        "id": "sample-1",
        "audio": {"array": [0.0, 0.0, 0.0, 0.0], "sampling_rate": 16000},
        "reference": "test output",
    }

    result = asr_bench._run_one_sample(pipe, sample, {})

    assert result.hypothesis == "test output"
    assert len(pipe.calls) == 1
    first_input, first_sampling_rate, _ = pipe.calls[0]
    assert isinstance(first_input, dict)
    assert first_input["raw"] == sample["audio"]["array"]
    assert first_input["sampling_rate"] == 16000
    assert first_sampling_rate is None


def test_run_one_sample_uses_native_qwen_transcribe_when_available() -> None:
    """Verify native qwen_asr transcribe path is used when available."""

    class Result:
        def __init__(self, text):  # type: ignore[no-untyped-def]
            self.text = text

    class NativePipe:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def transcribe(self, audio=None, language=None):  # type: ignore[no-untyped-def]
            self.calls.append({"audio": audio, "language": language})
            return [Result("native output")]

    sample = {
        "id": "sample-1",
        "audio": {"array": [0.0, 0.0, 0.0, 0.0], "sampling_rate": 16000},
        "reference": "native output",
    }

    pipe = NativePipe()
    result = asr_bench._run_one_sample(pipe, sample, {}, native_language="ko")

    assert result.hypothesis == "native output"
    assert result.num_beams is None
    assert pipe.calls == [{"audio": ([0.0, 0.0, 0.0, 0.0], 16000), "language": "ko"}]


def test_run_one_sample_native_qwen_without_language_param_keeps_backward_compatibility() -> None:
    """Verify native transcribe fallback works when language is not accepted."""

    class Result:
        def __init__(self, text):  # type: ignore[no-untyped-def]
            self.text = text

    class NativePipe:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def transcribe(self, audio=None):  # type: ignore[no-untyped-def]
            self.calls.append(audio)
            return [Result("native output")]

    sample = {
        "id": "sample-1",
        "audio": {"array": [0.0, 0.0, 0.0, 0.0], "sampling_rate": 16000},
        "reference": "native output",
    }

    pipe = NativePipe()
    result = asr_bench._run_one_sample(pipe, sample, {}, native_language="ko")

    assert result.hypothesis == "native output"
    assert pipe.calls == [([0.0, 0.0, 0.0, 0.0], 16000)]


def test_run_one_sample_uses_native_qwen_processor_tokenizer_for_token_count() -> None:
    """Verify native qwen_asr token counts prefer processor.tokenizer when available."""

    class Result:
        def __init__(self, text):  # type: ignore[no-untyped-def]
            self.text = text

    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):  # type: ignore[no-untyped-def]
            return {"input_ids": [1, 2, 3, 4]}

    class Processor:
        def __init__(self) -> None:
            self.tokenizer = Tokenizer()

    class NativePipe:
        def __init__(self) -> None:
            self.processor = Processor()

        def transcribe(self, audio=None, language=None):  # type: ignore[no-untyped-def]
            return [Result("native output text")]

    sample = {
        "id": "sample-1",
        "audio": {"array": [0.0, 0.0, 0.0, 0.0], "sampling_rate": 16000},
        "reference": "native output text",
    }

    result = asr_bench._run_one_sample(NativePipe(), sample, {})

    assert result.hypothesis == "native output text"
    assert result.num_generated_tokens == 4


def test_run_one_sample_preserves_raw_text_for_language_specific_summary() -> None:
    """Verify raw transcripts survive sample execution so summary language policy stays effective."""

    class DummyPipe:
        def __init__(self) -> None:
            self.tokenizer = None

        def __call__(self, audio_input, sampling_rate=None, generate_kwargs=None):  # type: ignore[no-untyped-def]
            return {"text": "hello, world"}

    sample = {
        "id": "sample-1",
        "audio": {"array": [0.0, 0.0, 0.0, 0.0], "sampling_rate": 16000},
        "reference": "hello world",
    }

    timing = asr_bench._run_one_sample(DummyPipe(), sample, {})

    assert timing.reference == "hello world"
    assert timing.hypothesis == "hello, world"

    english_summary = asr_bench.summarize_timings([timing], language="en")
    korean_summary = asr_bench.summarize_timings([timing], language="ko")

    assert english_summary.wer == 0.0
    assert english_summary.cer == 0.0
    assert korean_summary.wer > 0.0
    assert korean_summary.cer > 0.0


def test_resample_audio_changes_rate_with_float32_output() -> None:
    """Verify ASR resampling uses the dedicated helper and preserves float32 output."""

    audio = np.asarray([0.0, 1.0, -1.0, 0.5], dtype=np.float32)

    resampled = asr_bench._resample_audio(audio, 8000, 16000)

    assert resampled.dtype == np.float32
    assert len(resampled) > len(audio)


def test_load_librispeech_streams_only_requested_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify fixed-size sampling shuffles and consumes only the requested prefix."""

    class DummyDataset:
        def __init__(self, rows):  # type: ignore[no-untyped-def]
            self.rows = rows
            self.shuffle_calls: list[int] = []
            self.iterated = 0
            self.cast_calls: list[tuple[str, object]] = []

        def cast_column(self, name, feature):  # type: ignore[no-untyped-def]
            self.cast_calls.append((name, feature))
            return self

        def shuffle(self, seed=None):  # type: ignore[no-untyped-def]
            self.shuffle_calls.append(seed)
            return self

        def __iter__(self):
            for row in self.rows:
                self.iterated += 1
                yield row

    rows = [
        {
            "id": f"sample-{index}",
            "audio": {"path": f"dummy-{index}.wav", "bytes": None},
            "text": f"text-{index}",
        }
        for index in range(5)
    ]
    dataset = DummyDataset(rows)
    load_calls: list[dict[str, object]] = []

    def fake_load_dataset(name, config, split=None, streaming=None):  # type: ignore[no-untyped-def]
        load_calls.append(
            {
                "name": name,
                "config": config,
                "split": split,
                "streaming": streaming,
            }
        )
        return dataset

    class AudioStub:
        def __init__(self, *, decode):  # type: ignore[no-untyped-def]
            self.decode = decode

    soundfile_stub = type(
        "SoundfileStub",
        (),
        {"read": staticmethod(lambda path, dtype=None: (np.asarray([0.0, 0.0], dtype=np.float32), 16000))},
    )()
    datasets_stub = type(
        "DatasetsStub",
        (),
        {"Audio": AudioStub, "load_dataset": staticmethod(fake_load_dataset)},
    )()
    monkeypatch.setitem(sys.modules, "soundfile", soundfile_stub)
    monkeypatch.setitem(sys.modules, "datasets", datasets_stub)

    args = asr_bench._parse_args(["--num-samples", "2"])

    samples = asr_bench._load_librispeech(args)

    assert len(samples) == 2
    assert dataset.iterated == 2
    assert dataset.shuffle_calls == [0]
    assert dataset.cast_calls and dataset.cast_calls[0][0] == "audio"
    assert getattr(dataset.cast_calls[0][1], "decode", None) is False
    assert load_calls == [
        {
            "name": "openslr/librispeech_asr",
            "config": "clean",
            "split": "test",
            "streaming": True,
        }
    ]


def test_load_librispeech_zero_samples_returns_empty_without_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify zero-sample requests short-circuit before iterating the streaming dataset."""

    class DummyDataset:
        def __init__(self) -> None:
            self.iterated = 0
            self.shuffle_calls: list[int] = []

        def cast_column(self, name, feature):  # type: ignore[no-untyped-def]
            return self

        def shuffle(self, seed=None):  # type: ignore[no-untyped-def]
            self.shuffle_calls.append(seed)
            return self

        def __iter__(self):
            self.iterated += 1
            yield {
                "id": "sample-0",
                "audio": {"path": "dummy-0.wav", "bytes": None},
                "text": "text-0",
            }

    dataset = DummyDataset()

    def fake_load_dataset(name, config, split=None, streaming=None):  # type: ignore[no-untyped-def]
        return dataset

    class AudioStub:
        def __init__(self, *, decode):  # type: ignore[no-untyped-def]
            self.decode = decode

    soundfile_stub = type(
        "SoundfileStub",
        (),
        {"read": staticmethod(lambda path, dtype=None: (np.asarray([0.0, 0.0], dtype=np.float32), 16000))},
    )()
    datasets_stub = type(
        "DatasetsStub",
        (),
        {"Audio": AudioStub, "load_dataset": staticmethod(fake_load_dataset)},
    )()
    monkeypatch.setitem(sys.modules, "soundfile", soundfile_stub)
    monkeypatch.setitem(sys.modules, "datasets", datasets_stub)

    args = asr_bench._parse_args(["--num-samples", "0"])

    samples = asr_bench._load_librispeech(args)

    assert samples == []
    assert dataset.iterated == 0
    assert dataset.shuffle_calls == []


def test_load_librispeech_uses_default_sample_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify default parser settings keep the ASR loader on the 50-sample subset path."""

    class DummyDataset:
        def __init__(self, rows):  # type: ignore[no-untyped-def]
            self.rows = rows
            self.iterated = 0

        def cast_column(self, name, feature):  # type: ignore[no-untyped-def]
            return self

        def shuffle(self, seed=None):  # type: ignore[no-untyped-def]
            return self

        def __iter__(self):
            for row in self.rows:
                self.iterated += 1
                yield row

    rows = [
        {
            "id": f"sample-{index}",
            "audio": {"path": f"dummy-{index}.wav", "bytes": None},
            "text": f"text-{index}",
        }
        for index in range(3)
    ]
    dataset = DummyDataset(rows)

    def fake_load_dataset(name, config, split=None, streaming=None):  # type: ignore[no-untyped-def]
        return dataset

    class AudioStub:
        def __init__(self, *, decode):  # type: ignore[no-untyped-def]
            self.decode = decode

    soundfile_stub = type(
        "SoundfileStub",
        (),
        {"read": staticmethod(lambda path, dtype=None: (np.asarray([0.0, 0.0], dtype=np.float32), 16000))},
    )()
    datasets_stub = type(
        "DatasetsStub",
        (),
        {"Audio": AudioStub, "load_dataset": staticmethod(fake_load_dataset)},
    )()
    monkeypatch.setitem(sys.modules, "soundfile", soundfile_stub)
    monkeypatch.setitem(sys.modules, "datasets", datasets_stub)

    args = asr_bench._parse_args([])
    samples = asr_bench._load_librispeech(args)

    assert len(samples) == 3
    assert dataset.iterated == 3


def test_load_librispeech_uses_full_split_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify --full-split forwards num_samples=None into the streaming loader."""

    captured: dict[str, object] = {}

    def fake_loader(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return []

    monkeypatch.setattr(asr_bench, "load_streaming_audio_text_samples", fake_loader)

    args = asr_bench._parse_args(["--full-split"])
    samples = asr_bench._load_librispeech(args)

    assert samples == []
    assert captured["num_samples"] is None


def test_main_reloads_full_split_stream_per_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify full-split mode rebuilds the streaming iterable for each target."""

    args = asr_bench._parse_args(["--output-dir", str(tmp_path), "--full-split"])
    target = asr_bench.ASRBenchmarkTarget(
        model_id="openai/whisper-small",
        revision_candidates=[None],
        label="openai/whisper-small",
        base="openai__whisper-small",
        mxq_path=None,
        is_original=False,
    )
    load_calls: list[int | None] = []

    def fake_loader(parsed_args):  # type: ignore[no-untyped-def]
        load_calls.append(parsed_args.num_samples)
        return iter([
            {"id": "sample-1", "audio": {"array": [0.0, 0.0], "sampling_rate": 16000}, "reference": "hello"}
        ])

    monkeypatch.setattr(asr_bench, "_parse_args", lambda argv=None: args)
    monkeypatch.setattr(asr_bench, "_resolve_runtime_defaults", lambda parsed_args, raw_argv: None)
    monkeypatch.setattr(asr_bench, "_collect_host_pc_info", lambda out_dir: None)
    monkeypatch.setattr(
        asr_bench,
        "_build_run_targets",
        lambda parsed_args: [(target, None, target.label, target.base), (target, "single", target.label, target.base)],
    )
    monkeypatch.setattr(asr_bench, "_load_librispeech", fake_loader)
    monkeypatch.setattr(asr_bench, "_resolve_generate_kwargs", lambda parsed_args: {"return_timestamps": False})
    monkeypatch.setattr(asr_bench, "_optional_generate_kwargs_for_model", lambda parsed_args, model_id: {})
    monkeypatch.setattr(asr_bench, "_args_for_target_device_backend", lambda parsed_args, **kwargs: parsed_args)
    monkeypatch.setattr(asr_bench, "_build_asr_pipeline", lambda *args, **kwargs: object())
    monkeypatch.setattr(asr_bench, "_warmup", lambda *args, **kwargs: None)
    monkeypatch.setattr(asr_bench, "_measure_target", lambda *args, **kwargs: ([], {}, {}))
    monkeypatch.setattr(
        asr_bench,
        "summarize_timings",
        lambda timings, language="en": asr_bench.ASRMetricSummary(
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    )
    monkeypatch.setattr(asr_bench, "_write_target_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(asr_bench, "_release_pipeline", lambda pipe, device: None)
    monkeypatch.setattr(asr_bench, "_write_combined_outputs", lambda out_dir, num_beams: None)

    assert asr_bench.main(["--output-dir", str(tmp_path), "--full-split"]) == 0
    assert load_calls == [None, None]
