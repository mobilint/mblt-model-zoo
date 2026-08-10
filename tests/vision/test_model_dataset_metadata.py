"""Tests for model dataset metadata and dataset-aware postprocessing."""

from __future__ import annotations

from pathlib import Path

import mblt_vision
import yaml

from mblt_model_zoo.vision.wrapper import resolve_model_config

MODEL_CONFIG_DIR = Path(mblt_vision.__file__).parent / "models"

DATASETS_BY_TASK = {
    "depth_estimation": {"nyu-depth"},
    "face_detection": {"widerface"},
    "image_classification": {"imagenet"},
    "instance_segmentation": {"coco"},
    "object_detection": {"coco"},
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
