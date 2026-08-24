import os
from types import SimpleNamespace

import pytest

from mblt_model_zoo.MeloTTS import api as melo_api
from mblt_model_zoo.MeloTTS.api import TTS

LANGUAGES = (
    "EN_NEWEST",
    "KR",
)


@pytest.fixture(params=LANGUAGES, scope="module")
def pipe(request):
    language = request.param

    pipe = TTS(
        language=language,
        device="auto",
        trust_remote_code=True,
    )
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

    output_dir = os.path.join(".", "tests", "tmp")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{pipe.language}.wav")

    pipe.tts_to_file(
        text,
        speaker_ids[speaker],
        output_path,
        speed=speed,
    )


@pytest.mark.parametrize(
    ("configured_target_device", "target_device", "expected_target_device"),
    [("aries-rb", "regulus-rb", "regulus-rb"), ("regulus-rb", None, "regulus-rb")],
)
def test_tts_forwards_target_device_to_synthesizer_and_bert(
    monkeypatch: pytest.MonkeyPatch,
    configured_target_device: str,
    target_device: str | None,
    expected_target_device: str,
) -> None:
    """Use an override first, otherwise retain the config's MeloTTS board."""

    class AttrDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    model_config = AttrDict(
        dev_no=0,
        target_core="0:0",
        encoder_mxq_path="encoder.mxq",
        decoder_mxq_path="decoder.mxq",
        bert_model_id="bert",
        target_device=configured_target_device,
    )
    hps = SimpleNamespace(
        model=model_config,
        data=SimpleNamespace(filter_length=2, hop_length=1, n_speakers=1),
        train=SimpleNamespace(segment_size=1),
        num_languages=1,
        num_tones=1,
        symbols=["a"],
    )
    synth_kwargs: dict = {}
    bert_kwargs: dict = {}

    class FakeSynthesizer:
        def __init__(self, *args, **kwargs) -> None:
            synth_kwargs.update(kwargs)

        def to(self, device):
            return self

        def eval(self) -> None:
            return None

        def load_state_dict(self, state_dict, strict: bool) -> None:
            return None

    class FakeBert:
        def to(self, device):
            return self

    monkeypatch.setattr(melo_api, "load_or_download_config", lambda *args, **kwargs: hps)
    monkeypatch.setattr(melo_api, "load_or_download_model", lambda *args, **kwargs: {"model": {}})
    monkeypatch.setattr(melo_api, "MobilintSynthesizerTrn", FakeSynthesizer)
    monkeypatch.setattr(melo_api.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        melo_api.AutoModelForMaskedLM,
        "from_pretrained",
        lambda *args, **kwargs: bert_kwargs.update(kwargs) or FakeBert(),
    )

    tts_kwargs = {"target_device": target_device} if target_device is not None else {}
    TTS(language="EN_NEWEST", device="cpu", **tts_kwargs)

    assert synth_kwargs["target_device"] == expected_target_device
    assert bert_kwargs["target_device"] == expected_target_device
