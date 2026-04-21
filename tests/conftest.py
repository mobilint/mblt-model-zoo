"""Shared pytest fixtures and NPU backend option handling for test suites."""

import pytest

from tests.npu_backend_options import (
    BaseNpuParams,
    BaseNpuSweepSpec,
    EncoderDecoderNpuParams,
    EncoderDecoderNpuSweepSpec,
    VisionTextNpuParams,
    VisionTextNpuSweepSpec,
    build_base_npu_params,
    build_base_specs,
    build_encoder_decoder_specs,
    build_vision_text_specs,
    collect_npu_kwargs,
)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize role-specific NPU sweep fixtures only for tests that use them."""
    if "_base_npu_sweep_spec" in metafunc.fixturenames:
        specs = build_base_specs(metafunc.config)
        metafunc.parametrize(
            "_base_npu_sweep_spec",
            specs,
            ids=[spec.id() for spec in specs],
            scope="module",
        )

    if "_encoder_decoder_npu_sweep_spec" in metafunc.fixturenames:
        specs = build_encoder_decoder_specs(metafunc.config)
        metafunc.parametrize(
            "_encoder_decoder_npu_sweep_spec",
            specs,
            ids=[spec.id() for spec in specs],
            scope="module",
        )

    if "_vision_text_npu_sweep_spec" in metafunc.fixturenames:
        specs = build_vision_text_specs(metafunc.config)
        metafunc.parametrize(
            "_vision_text_npu_sweep_spec",
            specs,
            ids=[spec.id() for spec in specs],
            scope="module",
        )


def pytest_addoption(parser):
    """Register shared CLI options used by tests."""
    parser.addoption(
        "--mxq-path",
        action="store",
        default=None,
        help="Override default mxq_path for pipeline loading.",
    )
    parser.addoption(
        "--dev-no",
        action="store",
        default=None,
        type=int,
        help="NPU device number.",
    )
    parser.addoption(
        "--core-mode",
        action="store",
        default="all",
        help="NPU core mode (default: all=single/global4/global8).",
    )
    parser.addoption(
        "--target-cores",
        action="store",
        default=None,
        help='Target cores (e.g., "0:0;0:1;0:2;0:3").',
    )
    parser.addoption(
        "--target-clusters",
        action="store",
        default=None,
        help='Target clusters (e.g., "0;1").',
    )
    for prefix in ("encoder", "decoder", "vision", "text"):
        parser.addoption(
            f"--{prefix}-mxq-path",
            action="store",
            default=None,
            help=f"Override {prefix} mxq_path.",
        )
        parser.addoption(
            f"--{prefix}-dev-no",
            action="store",
            default=None,
            type=int,
            help=f"{prefix} NPU device number.",
        )
        parser.addoption(
            f"--{prefix}-core-mode",
            action="store",
            default=None,
            help=f"{prefix} NPU core mode (single, multi, global4, global8, all=single/global4/global8).",
        )
        parser.addoption(
            f"--{prefix}-target-cores",
            action="store",
            default=None,
            help=f'{prefix} target cores (e.g., "0:0;0:1;0:2;0:3").',
        )
        parser.addoption(
            f"--{prefix}-target-clusters",
            action="store",
            default=None,
            help=f'{prefix} target clusters (e.g., "0;1").',
        )
    parser.addoption(
        "--revision",
        action="store",
        default=None,
        help="Override model revision (e.g., W8).",
    )
    parser.addoption(
        "--embedding-weight",
        action="store",
        default=None,
        help="Path to custom embedding weights.",
    )


@pytest.fixture(scope="module")
def mxq_path(request):
    """Return the base mxq path option."""
    return request.config.getoption("--mxq-path")


@pytest.fixture(scope="module")
def revision(request):
    """Return the optional model revision."""
    return request.config.getoption("--revision")


@pytest.fixture(scope="module")
def embedding_weight(request):
    """Return the optional embedding weight path."""
    return request.config.getoption("--embedding-weight")


@pytest.fixture(scope="module")
def _base_npu_sweep_spec(request) -> BaseNpuSweepSpec:
    """Return the parametrized base-only NPU sweep spec."""
    return request.param


@pytest.fixture(scope="module")
def _encoder_decoder_npu_sweep_spec(request) -> EncoderDecoderNpuSweepSpec:
    """Return the parametrized encoder-decoder NPU sweep spec."""
    return request.param


@pytest.fixture(scope="module")
def _vision_text_npu_sweep_spec(request) -> VisionTextNpuSweepSpec:
    """Return the parametrized vision-text NPU sweep spec."""
    return request.param


@pytest.fixture(scope="module")
def base_npu_params(
    request,
    embedding_weight,
    _base_npu_sweep_spec: BaseNpuSweepSpec,
) -> BaseNpuParams:
    """Return NPU kwargs for tests that use a single backend config."""
    return build_base_npu_params(
        request.config,
        embedding_weight,
        core_mode_override=_base_npu_sweep_spec.base_core_mode,
    )


@pytest.fixture(scope="module")
def encoder_decoder_npu_params(
    request,
    embedding_weight,
    _encoder_decoder_npu_sweep_spec: EncoderDecoderNpuSweepSpec,
) -> EncoderDecoderNpuParams:
    """Return NPU kwargs for tests that use encoder and decoder backend configs."""
    from tests.npu_backend_options import collect_provided_prefixes, warn_unused_prefixes

    config = request.config
    provided_prefixes = collect_provided_prefixes(config, embedding_weight)
    warn_unused_prefixes(provided_prefixes, {"encoder", "decoder"})

    encoder_kwargs, _ = collect_npu_kwargs(
        config,
        "encoder",
        core_mode_override=_encoder_decoder_npu_sweep_spec.encoder_core_mode,
    )
    decoder_kwargs, _ = collect_npu_kwargs(
        config,
        "decoder",
        core_mode_override=_encoder_decoder_npu_sweep_spec.decoder_core_mode,
    )

    return EncoderDecoderNpuParams(
        encoder=encoder_kwargs,
        decoder=decoder_kwargs,
    )


@pytest.fixture(scope="module")
def vision_text_npu_params(
    request,
    embedding_weight,
    _vision_text_npu_sweep_spec: VisionTextNpuSweepSpec,
) -> VisionTextNpuParams:
    """Return NPU kwargs for tests that use vision and text backend configs."""
    from tests.npu_backend_options import collect_provided_prefixes, warn_unused_prefixes

    config = request.config
    provided_prefixes = collect_provided_prefixes(config, embedding_weight)
    warn_unused_prefixes(provided_prefixes, {"vision", "text"})

    vision_kwargs, _ = collect_npu_kwargs(
        config,
        "vision",
        core_mode_override=_vision_text_npu_sweep_spec.vision_core_mode,
    )
    text_kwargs, _ = collect_npu_kwargs(
        config,
        "text",
        core_mode_override=_vision_text_npu_sweep_spec.text_core_mode,
    )

    return VisionTextNpuParams(
        vision=vision_kwargs,
        text=text_kwargs,
    )
