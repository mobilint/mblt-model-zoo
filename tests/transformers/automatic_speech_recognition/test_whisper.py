import pytest
from datasets import load_dataset
from transformers import pipeline

MODEL_PATHS = (
    "mobilint/whisper-tiny",
    "mobilint/whisper-small",
    "mobilint/whisper-base",
    "mobilint/whisper-medium",
    "mobilint/whisper-large-v3",
    "mobilint/whisper-large-v3-turbo",
)


@pytest.fixture(params=MODEL_PATHS, scope="module")
def pipe(request, revision, encoder_decoder_npu_params):
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


def test_whisper(pipe):
    pipe.generation_config.max_new_tokens = None

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    for i in range(5):
        sample = ds[i]["audio"]  # type: ignore

        output = pipe(
            sample,
            batch_size=8,
            return_timestamps=True,
            generate_kwargs={
                "num_beams": 1,  # Supports for beam search with reorder_cache is not implemented yet
            },
        )

        print("Result: %s" % output["text"])
        print("Answer: %s" % ds[i]["text"])  # type: ignore
