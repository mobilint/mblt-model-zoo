import pytest

from mblt_model_zoo.MeloTTS.api import TTS

LANGUAGES = (
    "EN_NEWEST",
    "KR",
)


@pytest.fixture(params=LANGUAGES, scope="module")
def pipe(request):
    language = request.param

    pipe = TTS(language=language, device="auto")
    yield pipe
    del pipe


def test_melo(pipe: TTS):
    # Speed is adjustable
    speed = 1.0

    texts = {
        "EN": "Did you ever hear a folk tale about a giant turtle?",
        "KR": "안녕하세요! 오늘은 날씨가 정말 좋네요.",
    }
    text = texts[pipe.language]

    speaker_ids = pipe.hps.data.spk2id

    speakers = {
        "EN": "EN-Newest",
        "KR": "KR",
    }
    speaker = speakers[pipe.language]

    output_path = f"tests/tmp/{pipe.language}.wav"
    pipe.tts_to_file(
        text,
        speaker_ids[speaker],
        output_path,
        speed=speed,
        dispose_bert_after_use=True,
    )
