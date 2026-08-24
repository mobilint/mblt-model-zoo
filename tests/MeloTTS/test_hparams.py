"""Unit tests for MeloTTS configuration preservation."""

from mblt_model_zoo.MeloTTS.utils import ModelHParams


def test_model_hparams_preserves_target_device() -> None:
    """Retain the configured board for TTS API forwarding."""

    config = ModelHParams(
        bert_model_id="bert",
        target_core="0:0",
        target_device="regulus-rb",
        encoder_mxq_path="encoder.mxq",
        decoder_mxq_path="decoder.mxq",
    )

    assert config.target_device == "regulus-rb"
