import pytest

from mblt_model_zoo.transformers.text_to_speech.MeloTTS import TTS


@pytest.fixture
def pipe():
    pipe = TTS(language="EN_NEWEST", device="auto")
    yield pipe
    pipe.dispose()


def test_melo(pipe):
    # Speed is adjustable
    speed = 1.0

    # English
    text = "Did you ever hear a folk tale about a giant turtle?"

    speaker_ids = pipe.hps.data.spk2id

    # American accent
    output_path = "en-us.wav"
    pipe.tts_to_file(text, speaker_ids["EN-Newest"], output_path, speed=speed)
