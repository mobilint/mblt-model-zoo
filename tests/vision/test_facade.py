"""Generic compatibility checks for the Model Zoo Vision facade."""

from __future__ import annotations

import importlib

import mblt_vision
from mblt_vision.compile.vision import DEFAULT_PERCENTILE as StandaloneDefaultPercentile
from mblt_vision.datasets import get_dataset_config as standalone_dataset_config
from mblt_vision.utils.letterbox import LetterBoxGeometry
from mblt_vision.utils.preprocess.base import PreOps
from mblt_vision.utils.results import Results

import mblt_model_zoo.vision as compatibility_vision
from mblt_model_zoo.compile.vision import DEFAULT_PERCENTILE as CompatibilityDefaultPercentile
from mblt_model_zoo.utils.npu_target import cluster_to_int, core_to_int
from mblt_model_zoo.vision.datasets import get_dataset_config
from mblt_model_zoo.vision.utils import Results as CompatibilityResults


def test_model_zoo_vision_exports_standalone_objects() -> None:
    """Keep the public facade tied to the standalone Vision package."""

    assert compatibility_vision.MBLT_Engine is mblt_vision.MBLT_Engine
    assert compatibility_vision.list_models is mblt_vision.list_models
    assert compatibility_vision.ResNet50 is mblt_vision.ResNet50
    assert CompatibilityResults is Results
    assert get_dataset_config is standalone_dataset_config


def test_every_standalone_task_is_deep_importable() -> None:
    """A task added to mblt_vision must be reachable without a Model Zoo edit."""

    for task in mblt_vision.list_tasks():
        compatibility_task_module = importlib.import_module(f"mblt_model_zoo.vision.{task}")
        assert compatibility_task_module is getattr(mblt_vision, task)


def test_legacy_utility_modules_are_standalone_aliases() -> None:
    """Compatibility paths must not load copied Vision implementations."""

    assert importlib.import_module("mblt_model_zoo.vision.utils.letterbox") is importlib.import_module(
        "mblt_vision.utils.letterbox"
    )
    assert importlib.import_module("mblt_model_zoo.vision.utils.results") is importlib.import_module(
        "mblt_vision.utils.results"
    )
    assert importlib.import_module("mblt_model_zoo.vision.datasets.registry") is importlib.import_module(
        "mblt_vision.datasets.registry"
    )
    for utility in ("preprocess", "postprocess", "datasets", "evaluation"):
        assert importlib.import_module(f"mblt_model_zoo.vision.utils.{utility}") is importlib.import_module(
            f"mblt_vision.utils.{utility}"
        )
    assert importlib.import_module("mblt_model_zoo.vision.utils.preprocess.base").PreOps is PreOps
    assert LetterBoxGeometry.__module__.startswith("mblt_vision.")


def test_legacy_compilation_and_npu_target_exports_are_forwarded() -> None:
    """Keep imported constants and conversion helpers available via Model Zoo."""

    assert CompatibilityDefaultPercentile == StandaloneDefaultPercentile
    assert callable(cluster_to_int)
    assert callable(core_to_int)
