from __future__ import annotations

from dataclasses import dataclass

import pytest

from tests import conftest, npu_backend_options


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
            "--mxq-path": None,
            "--dev-no": None,
            "--core-mode": shared_core_mode,
            "--target-cores": None,
            "--target-clusters": None,
            "--encoder-mxq-path": None,
            "--encoder-dev-no": None,
            "--encoder-core-mode": encoder_core_mode,
            "--encoder-target-cores": None,
            "--encoder-target-clusters": None,
            "--decoder-mxq-path": None,
            "--decoder-dev-no": None,
            "--decoder-core-mode": decoder_core_mode,
            "--decoder-target-cores": None,
            "--decoder-target-clusters": None,
            "--vision-mxq-path": None,
            "--vision-dev-no": None,
            "--vision-core-mode": vision_core_mode,
            "--vision-target-cores": None,
            "--vision-target-clusters": None,
            "--text-mxq-path": None,
            "--text-dev-no": None,
            "--text-core-mode": text_core_mode,
            "--text-target-cores": None,
            "--text-target-clusters": None,
        },
        args=explicit_args,
    )


def test_encoder_override_does_not_force_decoder_core_mode():
    config = _make_config(
        encoder_core_mode="global8",
        explicit_args=("--encoder-core-mode=global8",),
    )

    specs = conftest.build_encoder_decoder_specs(config)

    assert specs == [conftest.EncoderDecoderNpuSweepSpec(encoder_core_mode="global8", decoder_core_mode=None)]


def test_decoder_uses_shared_core_mode_when_explicitly_provided():
    config = _make_config(
        shared_core_mode="global4",
        encoder_core_mode="global8",
        explicit_args=("--core-mode=global4", "--encoder-core-mode=global8"),
    )

    specs = conftest.build_encoder_decoder_specs(config)

    assert specs == [conftest.EncoderDecoderNpuSweepSpec(encoder_core_mode="global8", decoder_core_mode="global4")]


def test_vision_override_does_not_force_text_core_mode():
    config = _make_config(
        vision_core_mode="global8",
        explicit_args=("--vision-core-mode=global8",),
    )

    specs = conftest.build_vision_text_specs(config)

    assert specs == [conftest.VisionTextNpuSweepSpec(vision_core_mode="global8", text_core_mode=None)]


def test_text_uses_shared_core_mode_when_explicitly_provided():
    config = _make_config(
        shared_core_mode="single",
        vision_core_mode="global8",
        explicit_args=("--core-mode=single", "--vision-core-mode=global8"),
    )

    specs = conftest.build_vision_text_specs(config)

    assert specs == [conftest.VisionTextNpuSweepSpec(vision_core_mode="global8", text_core_mode="single")]


def test_shared_encoder_decoder_all_stays_pairwise_aligned():
    config = _make_config(
        shared_core_mode="all",
        explicit_args=("--core-mode=all",),
    )

    specs = conftest.build_encoder_decoder_specs(config)

    assert specs == [
        conftest.EncoderDecoderNpuSweepSpec(encoder_core_mode="single", decoder_core_mode="single"),
        conftest.EncoderDecoderNpuSweepSpec(encoder_core_mode="global4", decoder_core_mode="global4"),
        conftest.EncoderDecoderNpuSweepSpec(encoder_core_mode="global8", decoder_core_mode="global8"),
    ]


def test_shared_vision_text_all_stays_pairwise_aligned():
    config = _make_config(
        shared_core_mode="all",
        explicit_args=("--core-mode=all",),
    )

    specs = conftest.build_vision_text_specs(config)

    assert specs == [
        conftest.VisionTextNpuSweepSpec(vision_core_mode="single", text_core_mode="single"),
        conftest.VisionTextNpuSweepSpec(vision_core_mode="global4", text_core_mode="global4"),
        conftest.VisionTextNpuSweepSpec(vision_core_mode="global8", text_core_mode="global8"),
    ]


def test_build_base_npu_params_can_force_single_mode():
    config = _make_config(
        shared_core_mode="all",
        explicit_args=("--core-mode=all", "--mxq-path=model.mxq", "--dev-no=2"),
    )
    config._options["--mxq-path"] = "model.mxq"
    config._options["--dev-no"] = 2

    params = npu_backend_options.build_base_npu_params(
        config,
        embedding_weight=None,
        core_mode_override="single",
    )

    assert params.base == {
        "mxq_path": "model.mxq",
        "dev_no": 2,
        "core_mode": "single",
        "target_cores": ["0:0"],
    }


def test_single_only_core_mode_validation_allows_default_all():
    config = _make_config(shared_core_mode="all", explicit_args=("--core-mode=all",))

    npu_backend_options.validate_single_only_core_mode(config, suite_name="Batch text-generation tests")


def test_single_only_core_mode_validation_rejects_global4():
    config = _make_config(shared_core_mode="global4", explicit_args=("--core-mode=global4",))

    with pytest.raises(pytest.UsageError, match="only supports --core-mode single"):
        npu_backend_options.validate_single_only_core_mode(config, suite_name="Batch text-generation tests")
