import pytest
from datasets import load_dataset
from transformers import pipeline

MODEL_PATHS = (
    "mobilint/Qwen3-ASR-0.6B",
    "mobilint/Qwen3-ASR-1.7B",
)


@pytest.fixture(params=MODEL_PATHS, scope="module")
def pipe(request, revision, encoder_decoder_npu_params):
    """HuggingFace AutomaticSpeechRecognition pipeline fixture.

    HuggingFace ASR pipeline fixture. 모델별로 한 번 생성되어 같은
    모듈 안에서 재사용된다.
    """
    model_path = request.param
    model_kwargs = {**encoder_decoder_npu_params.encoder, **encoder_decoder_npu_params.decoder}

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        trust_remote_code=True,
        revision=revision,
        model_kwargs=model_kwargs or None,
    )
    yield pipe

    del pipe


def test_via_huggingface_pipeline(pipe):
    """Run inference through ``transformers.pipeline`` (HF standard ASR API).

    Comment out this test (and its ``pipe`` fixture above) to skip the HF
    pipeline path.

    transformers.pipeline 경로로 추론을 검증한다 (HF 표준 ASR API).
    이 path 를 건너뛰려면 본 함수와 위쪽 ``pipe`` fixture 를 함께 주석 처리.
    """
    pipe.generation_config.max_new_tokens = 256

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    for i in range(5):
        sample = ds[i]["audio"]  # type: ignore

        output = pipe(
            sample,
            generate_kwargs={
                "num_beams": 1,
            },
        )

        print("Result: %s" % output["text"])
        print("Answer: %s" % ds[i]["text"])  # type: ignore


@pytest.fixture(params=MODEL_PATHS, scope="module")
def transcriber(request, revision):
    """Upstream ``qwen_asr.Qwen3ASRModel`` fixture.

    Upstream qwen-asr 패키지의 Qwen3ASRModel fixture. qwen-asr 패키지가
    없으면 ``pytest.importorskip`` 으로 해당 테스트가 자동 skip.
    """
    qwen_asr = pytest.importorskip("qwen_asr")

    model_path = request.param
    model = qwen_asr.Qwen3ASRModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        revision=revision,
        max_inference_batch_size=1,
        max_new_tokens=256,
    )
    yield model

    del model


def test_via_qwen_asr_transcribe(transcriber):
    """Run inference through ``Qwen3ASRModel.transcribe`` (upstream native API).

    Comment out this test (and its ``transcriber`` fixture above) to skip the
    qwen-asr native path.

    Qwen 팀의 native API ``Qwen3ASRModel.transcribe`` 경로로 추론을 검증한다.
    이 path 를 건너뛰려면 본 함수와 위쪽 ``transcriber`` fixture 를 함께
    주석 처리.
    """
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    for i in range(5):
        sample = ds[i]["audio"]  # type: ignore

        results = transcriber.transcribe(
            audio=(sample["array"], sample["sampling_rate"]),
            language=None,
        )

        print("Result: %s (lang=%s)" % (results[0].text, results[0].language))
        print("Answer: %s" % ds[i]["text"])  # type: ignore
