"""Shared NPU backend option helpers for pytest suites."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import pytest

WARNED_UNUSED_PREFIXES: set[str] = set()
CORE_MODE_SWEEP_VALUES = ("single", "global4", "global8")
ALL_PREFIXES = ("base", "encoder", "decoder", "vision", "text")


def parse_target_cores(value: str | None) -> list[str] | None:
    """Parse a semicolon-delimited target core option."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return [item.strip() for item in text.split(";") if item.strip()]


def parse_target_clusters(value: str | None) -> list[int] | None:
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


def expand_core_modes(value: str | None) -> list[str | None]:
    """Expand a core mode option into the values used for parametrization."""
    if value is None:
        return [None]
    text = value.strip()
    if not text:
        return [None]
    if text == "all":
        return list(CORE_MODE_SWEEP_VALUES)
    return [text]


def default_target_kwargs(core_mode: str | None, *, prefix: str) -> dict[str, Any]:
    """Return default target core or cluster options implied by a core mode."""
    if core_mode == "single":
        return {f"{prefix + '_' if prefix else ''}target_cores": ["0:0"]}
    if core_mode == "global4":
        return {f"{prefix + '_' if prefix else ''}target_clusters": [0]}
    if core_mode == "global8":
        return {f"{prefix + '_' if prefix else ''}target_clusters": [0, 1]}
    return {}


def collect_npu_kwargs(
    config: pytest.Config,
    prefix: str,
    *,
    core_mode_override: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Collect backend kwargs for the requested prefix from pytest options."""
    opt_prefix = f"--{prefix}-" if prefix else "--"
    mxq_path = config.getoption(f"{opt_prefix}mxq-path")
    dev_no = config.getoption(f"{opt_prefix}dev-no")
    raw_core_mode = config.getoption(f"{opt_prefix}core-mode")
    core_mode = core_mode_override if core_mode_override is not None else raw_core_mode
    target_cores = parse_target_cores(config.getoption(f"{opt_prefix}target-cores"))
    target_clusters = parse_target_clusters(config.getoption(f"{opt_prefix}target-clusters"))

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
        kwargs.update(default_target_kwargs(core_mode, prefix=prefix))
    elif core_mode in {"global4", "global8"} and target_clusters is None:
        kwargs.update(default_target_kwargs(core_mode, prefix=prefix))

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

    base_core_mode: str | None

    def id(self) -> str:
        """Return the pytest id fragment for this sweep spec."""
        return f"base={self.base_core_mode}" if self.base_core_mode is not None else "default"


@dataclass(frozen=True)
class EncoderDecoderNpuSweepSpec:
    """Parametrized core modes for encoder-decoder models."""

    encoder_core_mode: str | None
    decoder_core_mode: str | None

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

    vision_core_mode: str | None
    text_core_mode: str | None

    def id(self) -> str:
        """Return the pytest id fragment for this sweep spec."""
        parts = []
        if self.vision_core_mode is not None:
            parts.append(f"vision={self.vision_core_mode}")
        if self.text_core_mode is not None:
            parts.append(f"text={self.text_core_mode}")
        return ",".join(parts) if parts else "default"


def option_flag(prefix: str, option_name: str) -> str:
    """Return the CLI flag name for a backend option."""
    if prefix:
        return f"--{prefix}-{option_name.replace('_', '-')}"
    return f"--{option_name.replace('_', '-')}"


def option_value_was_provided(config: pytest.Config, prefix: str, option_name: str) -> bool:
    """Return whether the user explicitly set a CLI option."""
    args = config.invocation_params.args
    flag = option_flag(prefix, option_name)
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def full_matrix_enabled(config: pytest.Config) -> bool:
    """Return whether the caller requested the full test matrix."""
    return bool(config.getoption("--full-matrix"))


def should_expand_core_matrix(config: pytest.Config, *, prefixes: tuple[str, ...] = ()) -> bool:
    """Return whether core sweeps should expand beyond the quick single-core default."""
    if full_matrix_enabled(config):
        return True
    if option_value_was_provided(config, "", "core_mode"):
        return True
    return any(option_value_was_provided(config, prefix, "core_mode") for prefix in prefixes)


def collect_provided_prefixes(config: pytest.Config, embedding_weight: str | None) -> set[str]:
    """Collect prefixes that have explicit backend options in the current test run."""
    provided: set[str] = set()

    if embedding_weight:
        provided.add("base")

    for prefix in ALL_PREFIXES:
        option_prefix = "" if prefix == "base" else prefix
        if any(
            (
                option_value_was_provided(config, option_prefix, "mxq_path"),
                option_value_was_provided(config, option_prefix, "dev_no"),
                option_value_was_provided(config, option_prefix, "core_mode"),
                option_value_was_provided(config, option_prefix, "target_cores"),
                option_value_was_provided(config, option_prefix, "target_clusters"),
            )
        ):
            provided.add(prefix)

    return provided


def warn_unused_prefixes(provided_prefixes: set[str], used_prefixes: set[str]) -> None:
    """Warn once for explicit backend prefixes that the active test fixture does not consume."""
    for prefix in sorted(provided_prefixes - used_prefixes):
        if prefix not in WARNED_UNUSED_PREFIXES:
            WARNED_UNUSED_PREFIXES.add(prefix)
            warnings.warn(
                f"Provided {prefix} NPU backend options will be ignored for this model.",
                UserWarning,
            )


def build_base_specs(config: pytest.Config) -> list[BaseNpuSweepSpec]:
    """Build sweep specs for base-only models."""
    if not should_expand_core_matrix(config):
        return [BaseNpuSweepSpec(base_core_mode="single")]
    return [BaseNpuSweepSpec(base_core_mode=mode) for mode in expand_core_modes(config.getoption("--core-mode"))]


def build_encoder_decoder_specs(config: pytest.Config) -> list[EncoderDecoderNpuSweepSpec]:
    """Build synchronized sweep specs for encoder-decoder models."""
    if not should_expand_core_matrix(config, prefixes=("encoder", "decoder")):
        return [
            EncoderDecoderNpuSweepSpec(
                encoder_core_mode="single",
                decoder_core_mode="single",
            )
        ]

    encoder_raw = config.getoption("--encoder-core-mode")
    decoder_raw = config.getoption("--decoder-core-mode")
    shared_raw = config.getoption("--core-mode")
    encoder_explicit = option_value_was_provided(config, "encoder", "core_mode")
    decoder_explicit = option_value_was_provided(config, "decoder", "core_mode")
    shared_explicit = option_value_was_provided(config, "", "core_mode")
    use_shared_defaults = shared_explicit or full_matrix_enabled(config)

    if use_shared_defaults and not encoder_explicit and not decoder_explicit:
        return [
            EncoderDecoderNpuSweepSpec(
                encoder_core_mode=mode,
                decoder_core_mode=mode,
            )
            for mode in expand_core_modes(shared_raw)
        ]

    encoder_modes = expand_core_modes(encoder_raw if encoder_explicit else shared_raw if use_shared_defaults else None)
    decoder_modes = expand_core_modes(decoder_raw if decoder_explicit else shared_raw if use_shared_defaults else None)
    return [
        EncoderDecoderNpuSweepSpec(
            encoder_core_mode=encoder_mode,
            decoder_core_mode=decoder_mode,
        )
        for encoder_mode in encoder_modes
        for decoder_mode in decoder_modes
    ]


def build_vision_text_specs(config: pytest.Config) -> list[VisionTextNpuSweepSpec]:
    """Build synchronized sweep specs for vision-text models."""
    if not should_expand_core_matrix(config, prefixes=("vision", "text")):
        return [
            VisionTextNpuSweepSpec(
                vision_core_mode="single",
                text_core_mode="single",
            )
        ]

    vision_raw = config.getoption("--vision-core-mode")
    text_raw = config.getoption("--text-core-mode")
    shared_raw = config.getoption("--core-mode")
    vision_explicit = option_value_was_provided(config, "vision", "core_mode")
    text_explicit = option_value_was_provided(config, "text", "core_mode")
    shared_explicit = option_value_was_provided(config, "", "core_mode")
    use_shared_defaults = shared_explicit or full_matrix_enabled(config)

    if use_shared_defaults and not vision_explicit and not text_explicit:
        return [
            VisionTextNpuSweepSpec(
                vision_core_mode=mode,
                text_core_mode=mode,
            )
            for mode in expand_core_modes(shared_raw)
        ]

    vision_modes = expand_core_modes(vision_raw if vision_explicit else shared_raw if use_shared_defaults else None)
    text_modes = expand_core_modes(text_raw if text_explicit else shared_raw if use_shared_defaults else None)
    return [
        VisionTextNpuSweepSpec(
            vision_core_mode=vision_mode,
            text_core_mode=text_mode,
        )
        for vision_mode in vision_modes
        for text_mode in text_modes
    ]


def build_base_npu_params(
    config: pytest.Config,
    embedding_weight: str | None,
    *,
    core_mode_override: str | None = None,
) -> BaseNpuParams:
    """Build base backend kwargs for tests that use a single backend config."""
    provided_prefixes = collect_provided_prefixes(config, embedding_weight)
    warn_unused_prefixes(provided_prefixes, {"base"})

    base_kwargs, _ = collect_npu_kwargs(config, "", core_mode_override=core_mode_override)
    if embedding_weight:
        base_kwargs["embedding_weight"] = embedding_weight

    return BaseNpuParams(base=base_kwargs)


def validate_single_only_core_mode(config: pytest.Config, *, suite_name: str) -> None:
    """Reject unsupported core-mode overrides for suites that only support single-core mode."""
    raw_core_mode = config.getoption("--core-mode")
    if raw_core_mode in {None, "", "all", "single"}:
        return
    raise pytest.UsageError(f"{suite_name} only supports --core-mode single. Received --core-mode={raw_core_mode!r}.")
