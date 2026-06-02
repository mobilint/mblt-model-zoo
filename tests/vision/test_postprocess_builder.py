"""Tests for vision postprocess builder defaults and output contracts."""

from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np
import pytest
import torch
import yaml

from mblt_model_zoo.vision.utils.postprocess.base import YOLOPostBase
from mblt_model_zoo.vision.utils.postprocess.build_post import build_postprocess
from mblt_model_zoo.vision.utils.postprocess.yolo_dflfree_post import YOLODFLFreePost
from mblt_model_zoo.vision.utils.types import ListTensorLike
from mblt_model_zoo.vision.wrapper import MODEL_CONFIG_DIR, MBLT_Engine


class YOLODecodeBatchPostprocessor(Protocol):
    """Protocol for YOLO postprocessors that expose batch decode helpers."""

    def __call__(self, x: ListTensorLike) -> Any:
        """Run the postprocessor on raw outputs."""
        ...

    def check_input(self, x: ListTensorLike) -> list[torch.Tensor]:
        """Validate and normalize raw model outputs."""
        ...

    def rearrange(self, x: list[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Rearrange raw outputs into decode-ready tensors."""
        ...

    def decode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Decode batched export outputs."""
        ...


def test_yolo_postprocess_uses_yaml_threshold_defaults_and_allows_reset() -> None:
    """Initialize YOLO thresholds from YAML and allow explicit overrides later."""

    config_path = MODEL_CONFIG_DIR / "YOLO11m.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    postprocessor = cast(
        YOLOPostBase,
        build_postprocess(config["DEFAULT"]["pre_cfg"], config["DEFAULT"]["post_cfg"]),
    )

    assert postprocessor.conf_thres == pytest.approx(0.001)
    assert postprocessor.iou_thres == pytest.approx(0.7)

    postprocessor.set_threshold(conf_thres=0.25)
    assert postprocessor.conf_thres == pytest.approx(0.25)
    assert postprocessor.iou_thres == pytest.approx(0.7)

    postprocessor.set_threshold(iou_thres=0.45)
    assert postprocessor.conf_thres == pytest.approx(0.25)
    assert postprocessor.iou_thres == pytest.approx(0.45)


def test_yolo_postprocess_defaults_to_e2e_true() -> None:
    """Keep YOLO postprocessing in end-to-end mode unless disabled explicitly."""

    config_path = MODEL_CONFIG_DIR / "YOLO11m.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    postprocessor = cast(
        YOLOPostBase,
        build_postprocess(config["DEFAULT"]["pre_cfg"], config["DEFAULT"]["post_cfg"]),
    )

    assert postprocessor.e2e is True
    assert postprocessor.nc == 80


def test_yolo_postprocess_detection_defaults_nc_to_80_without_yaml_value() -> None:
    """Default detection class count to 80 when config omits it."""

    config_path = MODEL_CONFIG_DIR / "YOLO11m.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    post_cfg = dict(config["DEFAULT"]["post_cfg"])
    post_cfg.pop("nc", None)
    postprocessor = cast(
        YOLOPostBase,
        build_postprocess(config["DEFAULT"]["pre_cfg"], post_cfg),
    )

    assert postprocessor.nc == 80


def test_yolo_postprocess_pose_defaults_nc_to_1_without_yaml_value() -> None:
    """Default pose class count to 1 when config omits it."""

    config_path = MODEL_CONFIG_DIR / "YOLO11m-pose.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    post_cfg = dict(config["DEFAULT"]["post_cfg"])
    post_cfg.pop("nc", None)
    postprocessor = cast(
        YOLOPostBase,
        build_postprocess(config["DEFAULT"]["pre_cfg"], post_cfg),
    )

    assert postprocessor.nc == 1


def test_yolo_postprocess_obb_defaults_nc_to_15_without_yaml_value() -> None:
    """Default OBB class count to 15 when config omits it."""

    config_path = MODEL_CONFIG_DIR / "YOLO26x-obb.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    post_cfg = dict(config["DEFAULT"]["post_cfg"])
    post_cfg.pop("nc", None)
    postprocessor = YOLODFLFreePost(config["DEFAULT"]["pre_cfg"], post_cfg)

    assert postprocessor.nc == 15


def test_yolo_postprocess_can_return_pre_nms_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the dedicated non-e2e export path when end-to-end NMS is disabled."""

    config_path = MODEL_CONFIG_DIR / "YOLO11m.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    post_cfg = dict(config["DEFAULT"]["post_cfg"])
    post_cfg["e2e"] = False
    postprocessor = build_postprocess(config["DEFAULT"]["pre_cfg"], post_cfg)

    predictions = [object()]

    monkeypatch.setattr(postprocessor, "check_input", lambda x: x)
    monkeypatch.setattr(postprocessor, "non_e2e", lambda x: predictions)

    def fail_nms(_x: object) -> list[torch.Tensor]:
        raise AssertionError("nms should not run when e2e is False")

    monkeypatch.setattr(postprocessor, "nms", fail_nms)

    assert postprocessor([]) is predictions


def test_yolo_postprocess_runtime_kwargs_override_yaml_default() -> None:
    """Allow callers to override the default when building a postprocessor directly."""

    config_path = MODEL_CONFIG_DIR / "YOLO11m.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    postprocessor = cast(
        YOLOPostBase,
        build_postprocess(config["DEFAULT"]["pre_cfg"], config["DEFAULT"]["post_cfg"], e2e=False),
    )

    assert postprocessor.e2e is False


def test_yolo_postprocess_runtime_kwargs_override_nc_default() -> None:
    """Allow callers to override the default class count directly."""

    config_path = MODEL_CONFIG_DIR / "YOLO11m.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    post_cfg = dict(config["DEFAULT"]["post_cfg"])
    post_cfg.pop("nc", None)
    postprocessor = cast(
        YOLOPostBase,
        build_postprocess(config["DEFAULT"]["pre_cfg"], post_cfg, nc=3),
    )

    assert postprocessor.nc == 3


@pytest.mark.parametrize(
    "singleton_output",
    [torch.zeros((84, 8400), dtype=torch.float32), np.zeros((84, 8400), dtype=np.float32)],
)
def test_yolo_postprocess_moves_singleton_outputs_to_device_before_dim_check(
    singleton_output: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Move single raw outputs onto the configured device like list inputs already do."""

    config_path = MODEL_CONFIG_DIR / "YOLO11m.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    postprocessor = cast(
        YOLOPostBase,
        build_postprocess(config["DEFAULT"]["pre_cfg"], config["DEFAULT"]["post_cfg"]),
    )
    postprocessor.to("meta")

    def capture_check_dim(x: list[torch.Tensor]) -> list[torch.Tensor]:
        return x

    monkeypatch.setattr(postprocessor, "check_dim", capture_check_dim)

    checked_inputs = postprocessor.check_input(singleton_output)

    assert len(checked_inputs) == 1
    assert checked_inputs[0].device == postprocessor.device
    assert tuple(checked_inputs[0].shape) == (84, 8400)


def test_engine_passes_postprocess_kwargs_to_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Thread runtime postprocess overrides through the engine constructor."""

    class DummyBackend:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def create(self) -> None:
            pass

        def launch(self) -> None:
            pass

        def get_dtype(self) -> str:
            return "DataType.Float32"

    captured: dict[str, Any] = {}
    sentinel = object()
    config_path = MODEL_CONFIG_DIR / "YOLO11m.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model_cfg = {
        "file_cfg": {"mxq_path": "/tmp/fake.mxq"},
        "pre_cfg": config["DEFAULT"]["pre_cfg"],
        "post_cfg": config["DEFAULT"]["post_cfg"],
    }

    monkeypatch.setattr("mblt_model_zoo.vision.wrapper.MobilintNPUBackend", DummyBackend)
    monkeypatch.setattr("mblt_model_zoo.vision.wrapper.build_preprocess", lambda pre_cfg: sentinel)

    def fake_build_postprocess(pre_cfg: dict[str, Any], post_cfg: dict[str, Any], **kwargs: Any) -> object:
        captured["pre_cfg"] = pre_cfg
        captured["post_cfg"] = post_cfg
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr("mblt_model_zoo.vision.wrapper.build_postprocess", fake_build_postprocess)

    engine = MBLT_Engine(model_cls=model_cfg, postprocess_kwargs={"e2e": False})

    assert engine.postprocessor is sentinel
    assert captured["kwargs"] == {"e2e": False}


def _load_model_config(model_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a YOLO model config from the repo."""

    config_path = MODEL_CONFIG_DIR / f"{model_name.replace('yolo', 'YOLO', 1)}.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config["DEFAULT"]["pre_cfg"], config["DEFAULT"]["post_cfg"]


def _make_non_e2e_inputs(model_name: str, post_cfg: dict[str, Any], batch_size: int = 2) -> ListTensorLike:
    """Build synthetic raw outputs that match each representative YOLO family."""

    nc = post_cfg.get("nc", YOLOPostBase.DEFAULT_NC_BY_TASK[post_cfg["task"]])

    if model_name == "yolov5m":
        no = nc + 5
        return cast(
            ListTensorLike,
            [torch.zeros((batch_size, s, s, no * 3), dtype=torch.float32) for s in (20, 40, 80)],
        )
    if model_name == "yolov5m-seg":
        no = nc + 5 + post_cfg["n_extra"]
        outputs: ListTensorLike = [torch.zeros((batch_size, s, s, no * 3), dtype=torch.float32) for s in (20, 40, 80)]
        outputs.append(torch.zeros((batch_size, 160, 160, post_cfg["n_extra"]), dtype=torch.float32))
        return outputs
    if model_name in {"yolo11m", "yolov10m"}:
        reg_max = post_cfg["reg_max"]
        outputs: ListTensorLike = []
        for s in (20, 40, 80):
            outputs.extend(
                [
                    torch.zeros((batch_size, s, s, reg_max * 4), dtype=torch.float32),
                    torch.zeros((batch_size, s, s, nc), dtype=torch.float32),
                ]
            )
        return outputs
    if model_name == "yolo11m-seg":
        reg_max = post_cfg["reg_max"]
        n_extra = post_cfg["n_extra"]
        outputs: ListTensorLike = [torch.zeros((batch_size, 160, 160, n_extra), dtype=torch.float32)]
        for s in (20, 40, 80):
            outputs.extend(
                [
                    torch.zeros((batch_size, s, s, reg_max * 4), dtype=torch.float32),
                    torch.zeros((batch_size, s, s, nc), dtype=torch.float32),
                    torch.zeros((batch_size, s, s, n_extra), dtype=torch.float32),
                ]
            )
        return outputs
    if model_name == "yolo11m-pose":
        reg_max = post_cfg["reg_max"]
        n_extra = post_cfg["n_extra"]
        outputs: ListTensorLike = []
        for s in (20, 40, 80):
            outputs.extend(
                [
                    torch.zeros((batch_size, s, s, reg_max * 4), dtype=torch.float32),
                    torch.zeros((batch_size, s, s, nc), dtype=torch.float32),
                    torch.zeros((batch_size, s, s, n_extra), dtype=torch.float32),
                ]
            )
        return outputs
    if model_name == "yolo26m":
        outputs: ListTensorLike = []
        for s in (20, 40, 80):
            outputs.extend(
                [
                    torch.zeros((batch_size, s, s, 4), dtype=torch.float32),
                    torch.zeros((batch_size, s, s, nc), dtype=torch.float32),
                ]
            )
        return outputs
    if model_name == "yolo26m-seg":
        n_extra = post_cfg["n_extra"]
        outputs: ListTensorLike = [torch.zeros((batch_size, 160, 160, n_extra), dtype=torch.float32)]
        for s in (20, 40, 80):
            outputs.extend(
                [
                    torch.zeros((batch_size, s, s, 4), dtype=torch.float32),
                    torch.zeros((batch_size, s, s, nc), dtype=torch.float32),
                    torch.zeros((batch_size, s, s, n_extra), dtype=torch.float32),
                ]
            )
        return outputs
    if model_name == "yolo26m-pose":
        n_extra = post_cfg["n_extra"]
        outputs: ListTensorLike = []
        for s in (20, 40, 80):
            outputs.extend(
                [
                    torch.zeros((batch_size, s, s, 4), dtype=torch.float32),
                    torch.zeros((batch_size, s, s, nc), dtype=torch.float32),
                    torch.zeros((batch_size, s, s, n_extra), dtype=torch.float32),
                ]
            )
        return outputs
    raise ValueError(f"Unsupported model for non-e2e shape test: {model_name}")


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("yolov5m", (2, 25200, 85)),
        ("yolov5m-seg", [(2, 25200, 117), (2, 32, 160, 160)]),
        ("yolo11m", (2, 84, 8400)),
        ("yolo11m-seg", [(2, 116, 8400), (2, 32, 160, 160)]),
        ("yolo11m-pose", (2, 56, 8400)),
        ("yolov10m", (2, 300, 6)),
        ("yolo26m", (2, 300, 6)),
        ("yolo26m-seg", [(2, 300, 38), (2, 32, 160, 160)]),
        ("yolo26m-pose", (2, 300, 57)),
    ],
)
def test_yolo_non_e2e_output_shapes_match_onnx_contract(model_name: str, expected: Any) -> None:
    """Match the non-e2e output contract used by the exported ONNX models."""

    pre_cfg, post_cfg = _load_model_config(model_name)
    postprocessor = build_postprocess(pre_cfg, post_cfg, e2e=False)
    output = postprocessor(_make_non_e2e_inputs(model_name, post_cfg))

    if isinstance(expected, tuple):
        assert tuple(output.shape) == expected
    else:
        assert isinstance(output, list)
        assert tuple(output[0].shape) == expected[0]
        assert tuple(output[1].shape) == expected[1]


@pytest.mark.parametrize("model_name", ["yolov5m", "yolov5m-seg"])
def test_anchor_yolo_non_e2e_returns_decoded_batch_outputs(model_name: str) -> None:
    """Anchor-based non-e2e outputs should match batched decode results, not just rearrange."""

    pre_cfg, post_cfg = _load_model_config(model_name)
    postprocessor = cast(YOLODecodeBatchPostprocessor, build_postprocess(pre_cfg, post_cfg, e2e=False))
    inputs = _make_non_e2e_inputs(model_name, post_cfg)

    output = postprocessor(inputs)
    rearranged = postprocessor.rearrange(postprocessor.check_input(inputs))

    if isinstance(rearranged, tuple):
        det_out, proto_out = rearranged
        expected = cast(Any, output)
        assert torch.allclose(expected[0], postprocessor.decode_batch(det_out))
        assert torch.equal(expected[1], proto_out.permute(0, 3, 1, 2))
    else:
        assert isinstance(output, torch.Tensor)
        assert torch.allclose(output, postprocessor.decode_batch(rearranged))


@pytest.mark.parametrize("model_name", ["yolo11m", "yolo11m-seg", "yolo11m-pose"])
def test_anchorless_yolo_decode_matches_non_e2e_axis_order_for_nms(model_name: str) -> None:
    """Anchorless decode outputs should share the non-e2e channel-first layout."""

    pre_cfg, post_cfg = _load_model_config(model_name)
    e2e_postprocessor = cast(YOLOPostBase, build_postprocess(pre_cfg, post_cfg, e2e=True))
    non_e2e_postprocessor = cast(YOLOPostBase, build_postprocess(pre_cfg, post_cfg, e2e=False))
    inputs = _make_non_e2e_inputs(model_name, post_cfg)
    checked_inputs = e2e_postprocessor.check_input(inputs)
    rearranged = e2e_postprocessor.rearrange(checked_inputs)

    decoded = e2e_postprocessor.decode(rearranged[0] if isinstance(rearranged, tuple) else rearranged)
    non_e2e_output = non_e2e_postprocessor(inputs)
    decoded_batch = non_e2e_output[0] if isinstance(non_e2e_output, list) else non_e2e_output

    assert isinstance(decoded, list)
    for batch_index, per_image in enumerate(decoded):
        assert torch.allclose(per_image, decoded_batch[batch_index, :, : per_image.shape[1]])


@pytest.mark.parametrize("model_name", ["yolov5m", "yolov5m-seg"])
def test_anchor_yolo_decode_uses_process_box_cls(model_name: str) -> None:
    """Anchor decode should keep the filtered per-image process_box_cls path."""

    pre_cfg, post_cfg = _load_model_config(model_name)
    postprocessor = cast(YOLOPostBase, build_postprocess(pre_cfg, post_cfg, e2e=True))
    inputs = _make_non_e2e_inputs(model_name, post_cfg)
    rearranged = postprocessor.rearrange(postprocessor.check_input(inputs))
    decoded = postprocessor.decode(rearranged[0] if isinstance(rearranged, tuple) else rearranged)

    assert isinstance(decoded, list)
    assert len(decoded) == 2


@pytest.mark.parametrize("model_name", ["yolo26m", "yolo26m-seg", "yolo26m-pose"])
def test_dflfree_yolo_decode_uses_process_box_cls(model_name: str) -> None:
    """DFL-free decode should keep the filtered per-image process_box_cls path."""

    pre_cfg, post_cfg = _load_model_config(model_name)
    postprocessor = cast(YOLOPostBase, build_postprocess(pre_cfg, post_cfg, e2e=True))
    inputs = _make_non_e2e_inputs(model_name, post_cfg)
    rearranged = postprocessor.rearrange(postprocessor.check_input(inputs))
    decoded = postprocessor.decode(rearranged[0] if isinstance(rearranged, tuple) else rearranged)

    assert isinstance(decoded, list)
    assert len(decoded) == 2
