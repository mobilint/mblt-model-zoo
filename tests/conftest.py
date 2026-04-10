"""Shared pytest fixtures and NPU backend option handling for test suites."""

import warnings
from dataclasses import dataclass
from typing import Any, Optional

import pytest

_WARNED_UNUSED_PREFIXES: set[str] = set()
_CORE_MODE_SWEEP_VALUES = ("single", "global4", "global8")
_ALL_PREFIXES = ("base", "encoder", "decoder", "vision", "text")


def _parse_target_cores(value: Optional[str]) -> Optional[list[str]]:
    """Parse a semicolon-delimited target core option."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return [item.strip() for item in text.split(";") if item.strip()]


def _parse_target_clusters(value: Optional[str]) -> Optional[list[int]]:
    """Parse a semicolon-delimited target cluster option."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    clusters: list[int] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        clusters.append(int(item))
    return clusters


def _expand_core_modes(value: Optional[str]) -> list[Optional[str]]:
    """Expand a core mode option into the values used for parametrization."""
    if value is None:
        return [None]
    text = value.strip()
    if not text:
        return [None]
    if text == "all":
        return list(_CORE_MODE_SWEEP_VALUES)
    return [text]


def _default_target_kwargs(core_mode: Optional[str], *, prefix: str) -> dict[str, Any]:
    """Return default target core or cluster options implied by a core mode."""
    if core_mode == "single":
        return {f"{prefix + '_' if prefix else ''}target_cores": ["0:0"]}
    if core_mode == "global4":
        return {f"{prefix + '_' if prefix else ''}target_clusters": [0]}
    if core_mode == "global8":
        return {f"{prefix + '_' if prefix else ''}target_clusters": [0, 1]}
    return {}


def _collect_npu_kwargs(
    config: pytest.Config,
    prefix: str,
    *,
    core_mode_override: Optional[str] = None,
) -> tuple[dict[str, Any], bool]:
    """Collect backend kwargs for the requested prefix from pytest options."""
    opt_prefix = f"--{prefix}-" if prefix else "--"
    mxq_path = config.getoption(f"{opt_prefix}mxq-path")
    dev_no = config.getoption(f"{opt_prefix}dev-no")
    raw_core_mode = config.getoption(f"{opt_prefix}core-mode")
    core_mode = core_mode_override if core_mode_override is not None else raw_core_mode
    target_cores_raw = config.getoption(f"{opt_prefix}target-cores")
    target_cores = _parse_target_cores(target_cores_raw)
    target_clusters_raw = config.getoption(f"{opt_prefix}target-clusters")
    target_clusters = _parse_target_clusters(target_clusters_raw)

    kwargs: dict[str, Any] = {}
    provided = False

    if mxq_path:
        kwargs[f"{prefix + '_' if prefix else ''}mxq_path"] = mxq_path
        provided = True
    if dev_no is not None:
        kwargs[f"{prefix + '_' if prefix else ''}dev_no"] = dev_no
        provided = True
    if core_mode:
        kwargs[f"{prefix + '_' if prefix else ''}core_mode"] = core_mode
        provided = True
    if target_cores is not None:
        kwargs[f"{prefix + '_' if prefix else ''}target_cores"] = target_cores
        provided = True
    if target_clusters is not None:
        kwargs[f"{prefix + '_' if prefix else ''}target_clusters"] = target_clusters
        provided = True

    if core_mode == "single" and target_cores is None:
        kwargs.update(_default_target_kwargs(core_mode, prefix=prefix))
    elif core_mode in {"global4", "global8"} and target_clusters is None:
        kwargs.update(_default_target_kwargs(core_mode, prefix=prefix))

    return kwargs, provided


@dataclass(frozen=True)
class BaseNpuParams:
    """NPU backend kwargs for single-backend models."""

    base: dict[str, Any]


@dataclass(frozen=True)
class EncoderDecoderNpuParams:
    """NPU backend kwargs for encoder-decoder models."""

    encoder: dict[str, Any]
    decoder: dict[str, Any]


@dataclass(frozen=True)
class VisionTextNpuParams:
    """NPU backend kwargs for vision-text models."""

    vision: dict[str, Any]
    text: dict[str, Any]


@dataclass(frozen=True)
class BaseNpuSweepSpec:
    """Parametrized core mode for base-only models."""

    base_core_mode: Optional[str]

    def id(self) -> str:
        """Return the pytest id fragment for this sweep spec."""
        return f"base={self.base_core_mode}" if self.base_core_mode is not None else "default"


@dataclass(frozen=True)
class EncoderDecoderNpuSweepSpec:
    """Parametrized core modes for encoder-decoder models."""

    encoder_core_mode: Optional[str]
    decoder_core_mode: Optional[str]

    def id(self) -> str:
        """Return the pytest id fragment for this sweep spec."""
        parts = []
        if self.encoder_core_mode is not None:
            parts.append(f"encoder={self.encoder_core_mode}")
        if self.decoder_core_mode is not None:
            parts.append(f"decoder={self.decoder_core_mode}")
        return ",".join(parts) if parts else "default"


@dataclass(frozen=True)
class VisionTextNpuSweepSpec:
    """Parametrized core modes for vision-text models."""

    vision_core_mode: Optional[str]
    text_core_mode: Optional[str]

    def id(self) -> str:
        """Return the pytest id fragment for this sweep spec."""
        parts = []
        if self.vision_core_mode is not None:
            parts.append(f"vision={self.vision_core_mode}")
        if self.text_core_mode is not None:
            parts.append(f"text={self.text_core_mode}")
        return ",".join(parts) if parts else "default"


def _option_flag(prefix: str, option_name: str) -> str:
    """Return the CLI flag name for a backend option."""
    if prefix:
        return f"--{prefix}-{option_name.replace('_', '-')}"
    return f"--{option_name.replace('_', '-')}"


def _option_value_was_provided(config: pytest.Config, prefix: str, option_name: str) -> bool:
    """Return whether the user explicitly set a CLI option."""
    args = config.invocation_params.args
    flag = _option_flag(prefix, option_name)
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def _collect_provided_prefixes(config: pytest.Config, embedding_weight: Optional[str]) -> set[str]:
    """Collect prefixes that have explicit backend options in the current test run."""
    provided: set[str] = set()

    if embedding_weight:
        provided.add("base")

    for prefix in _ALL_PREFIXES:
        option_prefix = "" if prefix == "base" else prefix
        if any(
            (
                _option_value_was_provided(config, option_prefix, "mxq_path"),
                _option_value_was_provided(config, option_prefix, "dev_no"),
                _option_value_was_provided(config, option_prefix, "core_mode"),
                _option_value_was_provided(config, option_prefix, "target_cores"),
                _option_value_was_provided(config, option_prefix, "target_clusters"),
            )
        ):
            provided.add(prefix)

    return provided


def _warn_unused_prefixes(provided_prefixes: set[str], used_prefixes: set[str]) -> None:
    """Warn once for explicit backend prefixes that the active test fixture does not consume."""
    for prefix in sorted(provided_prefixes - used_prefixes):
        if prefix not in _WARNED_UNUSED_PREFIXES:
            _WARNED_UNUSED_PREFIXES.add(prefix)
            warnings.warn(
                f"Provided {prefix} NPU backend options will be ignored for this model.",
                UserWarning,
            )


def _build_base_specs(config: pytest.Config) -> list[BaseNpuSweepSpec]:
    """Build sweep specs for base-only models."""
    return [BaseNpuSweepSpec(base_core_mode=mode) for mode in _expand_core_modes(config.getoption("--core-mode"))]


def _build_encoder_decoder_specs(config: pytest.Config) -> list[EncoderDecoderNpuSweepSpec]:
    """Build synchronized sweep specs for encoder-decoder models."""
    encoder_raw = config.getoption("--encoder-core-mode")
    decoder_raw = config.getoption("--decoder-core-mode")
    shared_raw = config.getoption("--core-mode")
    encoder_explicit = _option_value_was_provided(config, "encoder", "core_mode")
    decoder_explicit = _option_value_was_provided(config, "decoder", "core_mode")
    shared_explicit = _option_value_was_provided(config, "", "core_mode")

    if shared_explicit and not encoder_explicit and not decoder_explicit:
        return [
            EncoderDecoderNpuSweepSpec(
                encoder_core_mode=mode,
                decoder_core_mode=mode,
            )
            for mode in _expand_core_modes(shared_raw)
        ]

    encoder_modes = _expand_core_modes(encoder_raw if encoder_explicit else shared_raw if shared_explicit else None)
    decoder_modes = _expand_core_modes(decoder_raw if decoder_explicit else shared_raw if shared_explicit else None)
    return [
        EncoderDecoderNpuSweepSpec(
            encoder_core_mode=encoder_mode,
            decoder_core_mode=decoder_mode,
        )
        for encoder_mode in encoder_modes
        for decoder_mode in decoder_modes
    ]


def _build_vision_text_specs(config: pytest.Config) -> list[VisionTextNpuSweepSpec]:
    """Build synchronized sweep specs for vision-text models."""
    vision_raw = config.getoption("--vision-core-mode")
    text_raw = config.getoption("--text-core-mode")
    shared_raw = config.getoption("--core-mode")
    vision_explicit = _option_value_was_provided(config, "vision", "core_mode")
    text_explicit = _option_value_was_provided(config, "text", "core_mode")
    shared_explicit = _option_value_was_provided(config, "", "core_mode")

    if shared_explicit and not vision_explicit and not text_explicit:
        return [
            VisionTextNpuSweepSpec(
                vision_core_mode=mode,
                text_core_mode=mode,
            )
            for mode in _expand_core_modes(shared_raw)
        ]

    vision_modes = _expand_core_modes(vision_raw if vision_explicit else shared_raw if shared_explicit else None)
    text_modes = _expand_core_modes(text_raw if text_explicit else shared_raw if shared_explicit else None)
    return [
        VisionTextNpuSweepSpec(
            vision_core_mode=vision_mode,
            text_core_mode=text_mode,
        )
        for vision_mode in vision_modes
        for text_mode in text_modes
    ]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize role-specific NPU sweep fixtures only for tests that use them."""
    if "_base_npu_sweep_spec" in metafunc.fixturenames:
        specs = _build_base_specs(metafunc.config)
        metafunc.parametrize(
            "_base_npu_sweep_spec",
            specs,
            ids=[spec.id() for spec in specs],
            scope="module",
        )

    if "_encoder_decoder_npu_sweep_spec" in metafunc.fixturenames:
        specs = _build_encoder_decoder_specs(metafunc.config)
        metafunc.parametrize(
            "_encoder_decoder_npu_sweep_spec",
            specs,
            ids=[spec.id() for spec in specs],
            scope="module",
        )

    if "_vision_text_npu_sweep_spec" in metafunc.fixturenames:
        specs = _build_vision_text_specs(metafunc.config)
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
    config = request.config
    provided_prefixes = _collect_provided_prefixes(config, embedding_weight)
    _warn_unused_prefixes(provided_prefixes, {"base"})

    base_kwargs, _ = _collect_npu_kwargs(config, "", core_mode_override=_base_npu_sweep_spec.base_core_mode)
    if embedding_weight:
        base_kwargs["embedding_weight"] = embedding_weight

    return BaseNpuParams(base=base_kwargs)


@pytest.fixture(scope="module")
def encoder_decoder_npu_params(
    request,
    embedding_weight,
    _encoder_decoder_npu_sweep_spec: EncoderDecoderNpuSweepSpec,
) -> EncoderDecoderNpuParams:
    """Return NPU kwargs for tests that use encoder and decoder backend configs."""
    config = request.config
    provided_prefixes = _collect_provided_prefixes(config, embedding_weight)
    _warn_unused_prefixes(provided_prefixes, {"encoder", "decoder"})

    encoder_kwargs, _ = _collect_npu_kwargs(
        config,
        "encoder",
        core_mode_override=_encoder_decoder_npu_sweep_spec.encoder_core_mode,
    )
    decoder_kwargs, _ = _collect_npu_kwargs(
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
    config = request.config
    provided_prefixes = _collect_provided_prefixes(config, embedding_weight)
    _warn_unused_prefixes(provided_prefixes, {"vision", "text"})

    vision_kwargs, _ = _collect_npu_kwargs(
        config,
        "vision",
        core_mode_override=_vision_text_npu_sweep_spec.vision_core_mode,
    )
    text_kwargs, _ = _collect_npu_kwargs(
        config,
        "text",
        core_mode_override=_vision_text_npu_sweep_spec.text_core_mode,
    )

    return VisionTextNpuParams(
        vision=vision_kwargs,
        text=text_kwargs,
    )
