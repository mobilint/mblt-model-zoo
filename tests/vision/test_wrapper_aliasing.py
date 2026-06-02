"""Tests for vision model name aliasing."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import yaml

from mblt_model_zoo.vision.utils.postprocess.build_post import build_postprocess
from mblt_model_zoo.vision.utils.postprocess.yolo_dflfree_post import YOLODFLFreePost
from mblt_model_zoo.vision.wrapper import MODEL_CONFIG_DIR, MBLT_Engine


@pytest.mark.parametrize(
    ("model_name", "expected_yaml"),
    [
        ("regnet_x_16gf", "RegNet_X_16GF.yaml"),
        ("regnet-x-16gf", "RegNet_X_16GF.yaml"),
        ("RegNet_X_16GF.yaml", "RegNet_X_16GF.yaml"),
        ("regnet_x_1_6gf", "RegNet_X_1_6GF.yaml"),
        ("regnet-x-1-6gf", "RegNet_X_1_6GF.yaml"),
        ("resnet50", "ResNet50.yaml"),
        ("resnet-50", "ResNet50.yaml"),
        ("resnet_50", "ResNet50.yaml"),
    ],
)
def test_model_name_aliasing_resolves_precise_separator_matches(
    model_name: str,
    expected_yaml: str,
) -> None:
    """Resolve aliases without collapsing distinct separator boundaries too early."""

    engine = MBLT_Engine.__new__(MBLT_Engine)

    assert engine.model_name_aliasing(model_name) == expected_yaml


def test_model_name_aliasing_reports_compact_ambiguity() -> None:
    """Keep compact ambiguous names explicit."""

    engine = MBLT_Engine.__new__(MBLT_Engine)

    with pytest.raises(ValueError, match="ambiguous"):
        engine.model_name_aliasing("regnetx16gf")


def test_legacy_model_config_aliases_do_not_keep_duplicate_yaml_files() -> None:
    """Keep removed legacy config names out of the YAML registry."""

    duplicate_config_names = [
        "DeiT3BasePatch16384.yaml",
        "DeiT3LargePatch16224.yaml",
        "DeiT3LargePatch16384.yaml",
        "DeiT3MediumPatch16224.yaml",
        "DeiT3SmallPatch16224.yaml",
        "DeiT3SmallPatch16384.yaml",
        "DeiTBasePatch16384.yaml",
        "DeiTSmallPatch16224.yaml",
        "DeiTTinyPatch16224.yaml",
        "EfficientNetB1.yaml",
        "InceptionV3.yaml",
        "ViTB16.yaml",
        "ViTL16.yaml",
        "WideResNet1012.yaml",
        "WideResNet502.yaml",
    ]

    for config_name in duplicate_config_names:
        assert not (MODEL_CONFIG_DIR / config_name).exists()


def test_yolo_postprocess_uses_yaml_threshold_defaults_and_allows_reset() -> None:
    """Initialize YOLO thresholds from YAML and allow explicit overrides later."""

    config_path = MODEL_CONFIG_DIR / "YOLO11m.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    postprocessor = build_postprocess(config["DEFAULT"]["pre_cfg"], config["DEFAULT"]["post_cfg"])

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

    postprocessor = build_postprocess(config["DEFAULT"]["pre_cfg"], config["DEFAULT"]["post_cfg"])

    assert postprocessor.e2e is True
    assert postprocessor.nc == 80


def test_yolo_postprocess_detection_defaults_nc_to_80_without_yaml_value() -> None:
    """Default detection class count to 80 when config omits it."""

    config_path = MODEL_CONFIG_DIR / "YOLO11m.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    post_cfg = dict(config["DEFAULT"]["post_cfg"])
    post_cfg.pop("nc", None)
    postprocessor = build_postprocess(config["DEFAULT"]["pre_cfg"], post_cfg)

    assert postprocessor.nc == 80


def test_yolo_postprocess_pose_defaults_nc_to_1_without_yaml_value() -> None:
    """Default pose class count to 1 when config omits it."""

    config_path = MODEL_CONFIG_DIR / "YOLO11m-pose.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    post_cfg = dict(config["DEFAULT"]["post_cfg"])
    post_cfg.pop("nc", None)
    postprocessor = build_postprocess(config["DEFAULT"]["pre_cfg"], post_cfg)

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
    """Return decoded predictions directly when end-to-end NMS is disabled."""

    config_path = MODEL_CONFIG_DIR / "YOLO11m.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    post_cfg = dict(config["DEFAULT"]["post_cfg"])
    post_cfg["e2e"] = False
    postprocessor = build_postprocess(config["DEFAULT"]["pre_cfg"], post_cfg)

    predictions = [object()]

    monkeypatch.setattr(postprocessor, "check_input", lambda x: x)
    monkeypatch.setattr(postprocessor, "_pre_process", lambda x: (predictions, None))

    def fail_nms(_x: object) -> list[torch.Tensor]:
        raise AssertionError("nms should not run when e2e is False")

    monkeypatch.setattr(postprocessor, "nms", fail_nms)

    assert postprocessor([]) is predictions


def test_yolo_postprocess_runtime_kwargs_override_yaml_default() -> None:
    """Allow callers to override the default when building a postprocessor directly."""

    config_path = MODEL_CONFIG_DIR / "YOLO11m.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    postprocessor = build_postprocess(config["DEFAULT"]["pre_cfg"], config["DEFAULT"]["post_cfg"], e2e=False)

    assert postprocessor.e2e is False


def test_yolo_postprocess_runtime_kwargs_override_nc_default() -> None:
    """Allow callers to override the default class count directly."""

    config_path = MODEL_CONFIG_DIR / "YOLO11m.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    post_cfg = dict(config["DEFAULT"]["post_cfg"])
    post_cfg.pop("nc", None)
    postprocessor = build_postprocess(config["DEFAULT"]["pre_cfg"], post_cfg, nc=3)

    assert postprocessor.nc == 3


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
