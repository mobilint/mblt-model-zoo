from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests import conftest, npu_backend_options


@dataclass
class _FakeInvocationParams:
    args: tuple[str, ...]


class _FakeConfig:
    def __init__(
        self,
        options: dict[str, object],
        args: tuple[str, ...],
        selection_args: tuple[str, ...] = (),
    ):
        self._options = options
        self.invocation_params = _FakeInvocationParams(args=args)
        self.args = list(selection_args)
        self.hook = SimpleNamespace(pytest_deselected=lambda items: None)

    def getoption(self, name: str):
        return self._options[name]


def _make_config(
    *,
    shared_core_mode: str = "all",
    encoder_core_mode: str | None = None,
    decoder_core_mode: str | None = None,
    vision_core_mode: str | None = None,
    text_core_mode: str | None = None,
    full_matrix: bool = False,
    explicit_args: tuple[str, ...] = (),
    selection_args: tuple[str, ...] = (),
) -> _FakeConfig:
    return _FakeConfig(
        options={
            "--full-matrix": full_matrix,
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
        selection_args=selection_args,
    )


@dataclass
class _FakeCallSpec:
    params: dict[str, object]


class _FakeItem:
    def __init__(
        self,
        *,
        path: str,
        module: ModuleType,
        params: dict[str, object],
        nodeid: str | None = None,
    ):
        self.path = Path(path)
        self.module = module
        self.callspec = _FakeCallSpec(params=params)
        self.nodeid = nodeid or self.path.as_posix()
        self.markers: list[str] = []

    def add_marker(self, marker: str) -> None:
        self.markers.append(marker)


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


def test_default_base_specs_use_single_core_without_full_matrix():
    config = _make_config()

    specs = conftest.build_base_specs(config)

    assert specs == [conftest.BaseNpuSweepSpec(base_core_mode="single")]


def test_default_encoder_decoder_specs_use_single_core_without_full_matrix():
    config = _make_config()

    specs = conftest.build_encoder_decoder_specs(config)

    assert specs == [
        conftest.EncoderDecoderNpuSweepSpec(
            encoder_core_mode="single",
            decoder_core_mode="single",
        )
    ]


def test_default_vision_text_specs_use_single_core_without_full_matrix():
    config = _make_config()

    specs = conftest.build_vision_text_specs(config)

    assert specs == [
        conftest.VisionTextNpuSweepSpec(
            vision_core_mode="single",
            text_core_mode="single",
        )
    ]


def test_full_matrix_restores_default_base_core_sweep():
    config = _make_config(full_matrix=True)

    specs = conftest.build_base_specs(config)

    assert specs == [
        conftest.BaseNpuSweepSpec(base_core_mode="single"),
        conftest.BaseNpuSweepSpec(base_core_mode="global4"),
        conftest.BaseNpuSweepSpec(base_core_mode="global8"),
    ]


def test_full_matrix_restores_default_encoder_decoder_core_sweep():
    config = _make_config(full_matrix=True)

    specs = conftest.build_encoder_decoder_specs(config)

    assert specs == [
        conftest.EncoderDecoderNpuSweepSpec(encoder_core_mode="single", decoder_core_mode="single"),
        conftest.EncoderDecoderNpuSweepSpec(encoder_core_mode="global4", decoder_core_mode="global4"),
        conftest.EncoderDecoderNpuSweepSpec(encoder_core_mode="global8", decoder_core_mode="global8"),
    ]


def test_full_matrix_restores_default_vision_text_core_sweep():
    config = _make_config(full_matrix=True)

    specs = conftest.build_vision_text_specs(config)

    assert specs == [
        conftest.VisionTextNpuSweepSpec(vision_core_mode="single", text_core_mode="single"),
        conftest.VisionTextNpuSweepSpec(vision_core_mode="global4", text_core_mode="global4"),
        conftest.VisionTextNpuSweepSpec(vision_core_mode="global8", text_core_mode="global8"),
    ]


def test_full_matrix_with_explicit_vision_override_keeps_text_default_sweep():
    config = _make_config(
        full_matrix=True,
        vision_core_mode="global8",
        explicit_args=("--vision-core-mode=global8",),
    )

    specs = conftest.build_vision_text_specs(config)

    assert specs == [
        conftest.VisionTextNpuSweepSpec(vision_core_mode="global8", text_core_mode="single"),
        conftest.VisionTextNpuSweepSpec(vision_core_mode="global8", text_core_mode="global4"),
        conftest.VisionTextNpuSweepSpec(vision_core_mode="global8", text_core_mode="global8"),
    ]


def test_full_matrix_with_explicit_encoder_override_keeps_decoder_default_sweep():
    config = _make_config(
        full_matrix=True,
        encoder_core_mode="global8",
        explicit_args=("--encoder-core-mode=global8",),
    )

    specs = conftest.build_encoder_decoder_specs(config)

    assert specs == [
        conftest.EncoderDecoderNpuSweepSpec(encoder_core_mode="global8", decoder_core_mode="single"),
        conftest.EncoderDecoderNpuSweepSpec(encoder_core_mode="global8", decoder_core_mode="global4"),
        conftest.EncoderDecoderNpuSweepSpec(encoder_core_mode="global8", decoder_core_mode="global8"),
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


def test_transformers_collection_deselects_nondefault_models_by_default():
    config = _make_config()
    module = SimpleNamespace(
        MODEL_PATHS=(
            "mobilint/Qwen2.5-0.5B-Instruct",
            "mobilint/Qwen2.5-7B-Instruct",
        )
    )
    first_item = _FakeItem(
        path="C:/repo/tests/transformers/text_generation/non_batch/test_qwen2.py",
        module=module,
        params={"pipe": "mobilint/Qwen2.5-0.5B-Instruct"},
    )
    second_item = _FakeItem(
        path="C:/repo/tests/transformers/text_generation/non_batch/test_qwen2.py",
        module=module,
        params={"pipe": "mobilint/Qwen2.5-7B-Instruct"},
    )
    items = [first_item, second_item]

    conftest.pytest_collection_modifyitems(config, items)

    assert items == [first_item]
    assert second_item.markers == ["full_matrix"]


def test_transformers_collection_keeps_nondefault_models_with_keyword_filter():
    config = _make_config(explicit_args=("-k", "7B"))
    module = SimpleNamespace(
        MODEL_PATHS=(
            "mobilint/Qwen2.5-0.5B-Instruct",
            "mobilint/Qwen2.5-7B-Instruct",
        )
    )
    first_item = _FakeItem(
        path="C:/repo/tests/transformers/text_generation/non_batch/test_qwen2.py",
        module=module,
        params={"pipe": "mobilint/Qwen2.5-0.5B-Instruct"},
    )
    second_item = _FakeItem(
        path="C:/repo/tests/transformers/text_generation/non_batch/test_qwen2.py",
        module=module,
        params={"pipe": "mobilint/Qwen2.5-7B-Instruct"},
    )
    items = [first_item, second_item]

    conftest.pytest_collection_modifyitems(config, items)

    assert items == [first_item, second_item]
    assert second_item.markers == ["full_matrix"]


def test_transformers_collection_keeps_explicit_nondefault_nodeid_selection():
    selected_nodeid = (
        "tests/transformers/image_text_to_text/test_qwen3_vl.py::"
        "test_qwen3_vl[mobilint/Qwen3-VL-8B-Instruct]"
    )
    config = _make_config(
        explicit_args=(
            r"C:\repo\tests\transformers\image_text_to_text\test_qwen3_vl.py::"
            "test_qwen3_vl[mobilint/Qwen3-VL-8B-Instruct]",
        ),
        selection_args=(
            r"C:\repo\tests\transformers\image_text_to_text\test_qwen3_vl.py::"
            "test_qwen3_vl[mobilint/Qwen3-VL-8B-Instruct]",
        ),
    )
    module = SimpleNamespace(
        MODEL_PATHS=(
            "mobilint/Qwen3-VL-4B-Instruct",
            "mobilint/Qwen3-VL-8B-Instruct",
        )
    )
    selected_item = _FakeItem(
        path="C:/repo/tests/transformers/image_text_to_text/test_qwen3_vl.py",
        module=module,
        params={"pipe": "mobilint/Qwen3-VL-8B-Instruct"},
        nodeid=selected_nodeid,
    )
    items = [selected_item]

    conftest.pytest_collection_modifyitems(config, items)

    assert items == [selected_item]
    assert selected_item.markers == ["full_matrix"]
