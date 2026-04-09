from __future__ import annotations

from dataclasses import dataclass

import conftest


@dataclass
class _FakeInvocationParams:
    args: tuple[str, ...]


class _FakeConfig:
    def __init__(self, options: dict[str, str | None], args: tuple[str, ...]):
        self._options = options
        self.invocation_params = _FakeInvocationParams(args=args)

    def getoption(self, name: str):
        return self._options[name]


def _make_config(
    *,
    shared_core_mode: str = "all",
    encoder_core_mode: str | None = None,
    decoder_core_mode: str | None = None,
    vision_core_mode: str | None = None,
    text_core_mode: str | None = None,
    explicit_args: tuple[str, ...] = (),
) -> _FakeConfig:
    return _FakeConfig(
        options={
            "--core-mode": shared_core_mode,
            "--encoder-core-mode": encoder_core_mode,
            "--decoder-core-mode": decoder_core_mode,
            "--vision-core-mode": vision_core_mode,
            "--text-core-mode": text_core_mode,
        },
        args=explicit_args,
    )


def test_encoder_override_does_not_force_decoder_core_mode():
    config = _make_config(
        encoder_core_mode="global8",
        explicit_args=("--encoder-core-mode=global8",),
    )

    specs = conftest._build_encoder_decoder_specs(config)

    assert specs == [conftest.EncoderDecoderNpuSweepSpec(encoder_core_mode="global8", decoder_core_mode=None)]


def test_decoder_uses_shared_core_mode_when_explicitly_provided():
    config = _make_config(
        shared_core_mode="global4",
        encoder_core_mode="global8",
        explicit_args=("--core-mode=global4", "--encoder-core-mode=global8"),
    )

    specs = conftest._build_encoder_decoder_specs(config)

    assert specs == [conftest.EncoderDecoderNpuSweepSpec(encoder_core_mode="global8", decoder_core_mode="global4")]


def test_vision_override_does_not_force_text_core_mode():
    config = _make_config(
        vision_core_mode="global8",
        explicit_args=("--vision-core-mode=global8",),
    )

    specs = conftest._build_vision_text_specs(config)

    assert specs == [conftest.VisionTextNpuSweepSpec(vision_core_mode="global8", text_core_mode=None)]


def test_text_uses_shared_core_mode_when_explicitly_provided():
    config = _make_config(
        shared_core_mode="single",
        vision_core_mode="global8",
        explicit_args=("--core-mode=single", "--vision-core-mode=global8"),
    )

    specs = conftest._build_vision_text_specs(config)

    assert specs == [conftest.VisionTextNpuSweepSpec(vision_core_mode="global8", text_core_mode="single")]
