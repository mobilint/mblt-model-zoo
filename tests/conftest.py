"""Shared pytest fixtures and NPU backend option handling for test suites."""

from types import ModuleType

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
    full_matrix_enabled,
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
        "--full-matrix",
        action="store_true",
        default=False,
        help="Run the full transformer model/core matrix instead of the quick default subset.",
    )
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


def _keyword_filter_was_provided(config: pytest.Config) -> bool:
    """Return whether the current pytest invocation provided a keyword expression."""
    args = config.invocation_params.args
    return any(arg in {"-k", "--keyword"} or arg.startswith("--keyword=") for arg in args)


def _normalize_node_id(value: str) -> str:
    """Return a node id or selection argument with normalized path separators."""
    return value.replace("\\", "/")


def _item_was_explicitly_selected(config: pytest.Config, item: pytest.Item) -> bool:
    """Return whether pytest was invoked with this exact collected item node id."""
    node_id = getattr(item, "nodeid", None)
    if not isinstance(node_id, str) or "::" not in node_id:
        return False

    normalized_node_id = _normalize_node_id(node_id)
    selected_args = getattr(config, "args", ())

    for arg in selected_args:
        normalized_arg = _normalize_node_id(str(arg))
        if "::" not in normalized_arg:
            continue
        if normalized_arg == normalized_node_id:
            return True
        if normalized_arg.endswith(normalized_node_id):
            prefix = normalized_arg[: -len(normalized_node_id)]
            if prefix in {"", "./"} or prefix.endswith("/"):
                return True

    return False


def _is_transformers_test(item: pytest.Item) -> bool:
    """Return whether a collected item belongs to the transformers test tree."""
    return "/tests/transformers/" in item.path.as_posix()


def _module_model_paths(module: ModuleType) -> tuple[str, ...]:
    """Return the module's declared model paths in order."""
    model_paths = getattr(module, "MODEL_PATHS", None)
    if model_paths:
        return tuple(model_paths)

    model_paths_and_prompts = getattr(module, "MODEL_PATHS_AND_PROMPTS", None)
    if model_paths_and_prompts:
        return tuple(model_path for model_path, _ in model_paths_and_prompts)

    return ()


def _extract_model_path(value: object) -> str | None:
    """Extract a mobilint model path from a parametrized value."""
    if isinstance(value, str) and value.startswith("mobilint/"):
        return value

    if isinstance(value, tuple) and value:
        first = value[0]
        if isinstance(first, str) and first.startswith("mobilint/"):
            return first

    return None


def _item_uses_nondefault_model(item: pytest.Item) -> bool:
    """Return whether a collected item targets a non-default model path."""
    if not hasattr(item, "callspec"):
        return False

    model_paths = _module_model_paths(item.module)
    if len(model_paths) <= 1:
        return False

    first_model_path = model_paths[0]
    return any(
        model_path is not None and model_path != first_model_path
        for model_path in (_extract_model_path(value) for value in item.callspec.params.values())
    )


def _item_uses_non_single_core(item: pytest.Item) -> bool:
    """Return whether a collected item uses a non-single-core sweep spec."""
    if not hasattr(item, "callspec"):
        return False

    for value in item.callspec.params.values():
        if isinstance(value, BaseNpuSweepSpec):
            return value.base_core_mode not in {None, "single"}
        if isinstance(value, EncoderDecoderNpuSweepSpec):
            return any(mode not in {None, "single"} for mode in (value.encoder_core_mode, value.decoder_core_mode))
        if isinstance(value, VisionTextNpuSweepSpec):
            return any(mode not in {None, "single"} for mode in (value.vision_core_mode, value.text_core_mode))

    return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Trim the default transformers collection to a quick single-model matrix."""
    keep_all_models = full_matrix_enabled(config) or _keyword_filter_was_provided(config)
    kept_items: list[pytest.Item] = []
    deselected_items: list[pytest.Item] = []

    for item in items:
        if not _is_transformers_test(item):
            kept_items.append(item)
            continue

        uses_nondefault_model = _item_uses_nondefault_model(item)
        if uses_nondefault_model or _item_uses_non_single_core(item):
            item.add_marker("full_matrix")

        if uses_nondefault_model and not keep_all_models and not _item_was_explicitly_selected(config, item):
            deselected_items.append(item)
            continue

        kept_items.append(item)

    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items)
        items[:] = kept_items


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
