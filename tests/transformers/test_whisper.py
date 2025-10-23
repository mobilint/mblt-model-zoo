import pytest
from datasets import load_dataset
from mblt_model_zoo.transformers import pipeline


@pytest.fixture
def pipe():
    model_path = "mobilint/whisper-small"

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
    )
    yield pipe
    pipe.model.dispose()


def test_whisper(pipe):
    pipe.generation_config.max_new_tokens = None

    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    sample = ds[0]["audio"]

    output = pipe(
        sample.copy(),
        batch_size=8,
        return_timestamps=True,
        generate_kwargs={
            "max_length": 4096,
            "num_beams": 1,  # Supports for beam search with reorder_cache is not implemented yet
        },
    )

    print(output)
