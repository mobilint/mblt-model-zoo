"""Generic Model Zoo Vision compatibility tests.

The hardware smoke test intentionally accepts its model and runtime settings through
pytest options so one test can exercise any supported Vision model without maintaining
a per-model test matrix in Model Zoo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mblt_model_zoo import vision
from tests.npu_backend_options import parse_target_clusters, parse_target_cores


def test_model_zoo_vision_registry_is_available() -> None:
    """Expose every standalone Vision task through the Model Zoo facade."""

    tasks = vision.list_tasks()
    models_by_task = vision.list_models()

    assert set(models_by_task) == set(tasks)
    assert all(models_by_task[task] for task in tasks)


@pytest.mark.requires_npu
def test_selected_vision_model_runs_through_model_zoo(
    request: pytest.FixtureRequest,
) -> None:
    """Load, infer, postprocess, and dispose one user-selected Vision model."""

    config = request.config
    if not config.getoption("run_vision_smoke"):
        pytest.skip("pass --run-vision-smoke to run a model download and hardware inference")

    model_path = config.getoption("vision_model_path") or ""
    mxq_path = config.getoption("vision_mxq_path") or ""
    kwargs: dict[str, Any] = {
        "model_cls": config.getoption("vision_model"),
        "model_type": config.getoption("vision_model_type"),
        "model_path": model_path,
        "mxq_path": mxq_path,
        "framework": config.getoption("vision_framework"),
        "target_device": config.getoption("vision_target_device"),
    }
    if config.getoption("vision_dev_no") is not None:
        kwargs["dev_no"] = config.getoption("vision_dev_no")
    if config.getoption("vision_core_mode") is not None:
        kwargs["core_mode"] = config.getoption("vision_core_mode")
    target_cores = parse_target_cores(config.getoption("vision_target_cores"))
    target_clusters = parse_target_clusters(config.getoption("vision_target_clusters"))
    if target_cores is not None:
        kwargs["target_cores"] = target_cores
    if target_clusters is not None:
        kwargs["target_clusters"] = target_clusters

    model = vision.MBLT_Engine(**kwargs)
    try:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        output = model(model.preprocess(image))
        assert model.postprocess(output) is not None
    finally:
        model.dispose()
