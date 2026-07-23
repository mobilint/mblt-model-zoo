"""Tests for model dataset metadata and dataset-aware postprocessing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mblt_model_zoo.vision.utils.postprocess import build_postprocess
from mblt_model_zoo.vision.utils.postprocess.base import YOLOPostBase
from mblt_model_zoo.vision.wrapper import resolve_model_config

MODEL_CONFIG_DIR = Path(__file__).parents[2] / "mblt_model_zoo" / "vision" / "models"

DATASETS_BY_TASK = {
    "depth_estimation": {"nyu-depth"},
    "face_detection": {"widerface"},
    "image_classification": {"imagenet"},
    "instance_segmentation": {"coco"},
    "object_detection": {"coco", "open-images-v7"},
    "obb": {"dotav1"},
    "pose_estimation": {"coco"},
    "semantic_segmentation": {"ade20k", "cityscapes"},
}


def test_all_model_variants_declare_a_supported_dataset() -> None:
    """Require every resolved model variant to identify its output taxonomy."""

    checked = 0
    for config_path in sorted(MODEL_CONFIG_DIR.glob("*.yaml")):
        full_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert isinstance(full_config, dict)
        for variant in full_config:
            config = resolve_model_config(str(config_path), variant)
            post_cfg = config["post_cfg"]
            task = post_cfg["task"]
            assert post_cfg["dataset"] in DATASETS_BY_TASK[task], f"{config_path.name}:{variant}"
            checked += 1

    assert checked > 0


@pytest.mark.parametrize(
    ("model_name", "dataset"),
    [
        ("yolo26n-sem", "cityscapes"),
        ("yolo26n-sem-ade20k", "ade20k"),
        ("yolov8n", "coco"),
        ("yolov8n-oiv7", "open-images-v7"),
    ],
)
def test_model_families_resolve_distinct_datasets(model_name: str, dataset: str) -> None:
    """Keep same-task model families tied to their actual output taxonomies."""

    assert resolve_model_config(model_name)["post_cfg"]["dataset"] == dataset


def test_oiv7_postprocessor_uses_601_classes() -> None:
    """Derive Open Images V7 output shape from its dataset metadata."""

    config = resolve_model_config("yolov8n-oiv7")
    postprocessor = build_postprocess(config["pre_cfg"], config["post_cfg"])

    assert isinstance(postprocessor, YOLOPostBase)
    assert postprocessor.dataset == "open-images-v7"
    assert postprocessor.nc == 601


def test_postprocessor_rejects_dataset_class_count_conflicts() -> None:
    """Fail early when explicit class metadata conflicts with the dataset taxonomy."""

    config = resolve_model_config("yolov8n-oiv7")
    config["post_cfg"]["nc"] = 80

    with pytest.raises(ValueError, match=r"open-images-v7.*nc=601"):
        build_postprocess(config["pre_cfg"], config["post_cfg"])
