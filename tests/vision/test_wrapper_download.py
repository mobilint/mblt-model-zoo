"""Tests for vision wrapper MXQ path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from huggingface_hub.errors import EntryNotFoundError

import mblt_model_zoo.vision.wrapper as wrapper
from mblt_model_zoo.vision.utils.datasets import CustomCocodata
from mblt_model_zoo.vision.utils.postprocess import build_postprocess
from mblt_model_zoo.vision.utils.postprocess.base import YOLOPostBase
from mblt_model_zoo.vision.utils.postprocess.common import crop_mask, nmsout2eval, scale_coords, scale_masks
from mblt_model_zoo.vision.utils.types import ListTensorLike
from mblt_model_zoo.vision.wrapper import MBLT_Engine


def test_file_config_cleansing_prefers_existing_mxq_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use an existing local MXQ path without attempting a Hub download."""

    mxq_path = tmp_path / "model.mxq"
    mxq_path.write_bytes(b"mxq")

    def _unexpected_download(**kwargs: Any) -> str:
        raise AssertionError("hf_hub_download should not be called")

    monkeypatch.setattr(wrapper, "hf_hub_download", _unexpected_download)

    engine = MBLT_Engine.__new__(MBLT_Engine)
    engine.file_cfg = {
        "mxq_path": str(mxq_path),
        "repo_id": "mobilint/example",
        "filename": "model.mxq",
        "revision": "main",
        "core_mode": "global8",
    }

    engine.file_config_cleansing()

    assert engine.file_cfg["mxq_path"] == str(mxq_path)
    assert "repo_id" not in engine.file_cfg
    assert "filename" not in engine.file_cfg
    assert "revision" not in engine.file_cfg


def test_model_path_defaults_to_local_mxq_for_mxq_framework(tmp_path: Path) -> None:
    """Map ``model_path`` to ``mxq_path`` when MXQ inference is requested."""

    mxq_path = tmp_path / "model.mxq"
    mxq_path.write_bytes(b"mxq")

    engine = MBLT_Engine.__new__(MBLT_Engine)
    engine.framework = "mxq"
    engine.file_cfg = {"model_path": str(mxq_path)}

    engine.file_config_cleansing()

    assert engine.file_cfg["mxq_path"] == str(mxq_path)
    assert "onnx_path" not in engine.file_cfg or not engine.file_cfg["onnx_path"]


def test_model_path_defaults_to_local_onnx_for_onnx_framework(tmp_path: Path) -> None:
    """Map ``model_path`` to ``onnx_path`` when ONNX inference is requested."""

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")

    engine = MBLT_Engine.__new__(MBLT_Engine)
    engine.framework = "onnx"
    engine.file_cfg = {"model_path": str(onnx_path)}

    engine.file_config_cleansing()

    assert engine.file_cfg["onnx_path"] == str(onnx_path)
    assert "mxq_path" not in engine.file_cfg or not engine.file_cfg["mxq_path"]


def test_engine_init_accepts_local_mxq_model_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Route API ``model_path`` to the MXQ backend for local MXQ inference."""

    mxq_path = tmp_path / "model.mxq"
    mxq_path.write_bytes(b"mxq")
    backend_kwargs: dict[str, Any] = {}

    class _FakeBackend:
        def __init__(self, **kwargs: Any) -> None:
            backend_kwargs.update(kwargs)

        def create(self) -> None:
            return None

        def launch(self) -> None:
            return None

        def get_dtype(self) -> str:
            return "DataType.Float32"

        def dispose(self) -> None:
            return None

    monkeypatch.setattr(wrapper, "MobilintNPUBackend", _FakeBackend)
    monkeypatch.setattr(wrapper, "build_preprocess", lambda config: config)
    monkeypatch.setattr(wrapper, "build_postprocess", lambda pre_cfg, post_cfg, **kwargs: (pre_cfg, post_cfg, kwargs))

    engine = MBLT_Engine(
        model_cls={
            "file_cfg": {},
            "pre_cfg": {},
            "post_cfg": {},
        },
        model_path=str(mxq_path),
    )

    try:
        assert engine.file_cfg["mxq_path"] == str(mxq_path)
        assert backend_kwargs["mxq_path"] == str(mxq_path)
    finally:
        engine.dispose()


def test_engine_init_auto_detects_mxq_framework_from_model_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Infer the MXQ framework from a local MXQ path when framework is omitted."""

    mxq_path = tmp_path / "model.mxq"
    mxq_path.write_bytes(b"mxq")

    class _FakeBackend:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def create(self) -> None:
            return None

        def launch(self) -> None:
            return None

        def get_dtype(self) -> str:
            return "DataType.Float32"

        def dispose(self) -> None:
            return None

    monkeypatch.setattr(wrapper, "MobilintNPUBackend", _FakeBackend)
    monkeypatch.setattr(wrapper, "build_preprocess", lambda config: config)
    monkeypatch.setattr(wrapper, "build_postprocess", lambda pre_cfg, post_cfg, **kwargs: (pre_cfg, post_cfg, kwargs))

    engine = MBLT_Engine(
        model_cls={
            "file_cfg": {},
            "pre_cfg": {},
            "post_cfg": {},
        },
        model_path=str(mxq_path),
    )

    try:
        assert engine.framework == "mxq"
        assert engine.file_cfg["mxq_path"] == str(mxq_path)
    finally:
        engine.dispose()


def test_engine_init_accepts_local_onnx_model_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Route API ``model_path`` to the ONNX runtime session for local ONNX inference."""

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")

    class _FakeInput:
        name = "input"
        shape = [1, 3, 224, 224]

    class _FakeOutput:
        name = "output"

    class _FakeSession:
        def __init__(self, path: str, providers: list[str]) -> None:
            self.path = path
            self.providers = providers

        def get_inputs(self) -> list[_FakeInput]:
            return [_FakeInput()]

        def get_outputs(self) -> list[_FakeOutput]:
            return [_FakeOutput()]

    class _FakeOrt:
        def __init__(self) -> None:
            self.session: _FakeSession | None = None

        @staticmethod
        def get_available_providers() -> list[str]:
            return ["CPUExecutionProvider"]

        def InferenceSession(self, path: str, providers: list[str]) -> _FakeSession:
            self.session = _FakeSession(path, providers)
            return self.session

    fake_ort = _FakeOrt()
    monkeypatch.setattr(wrapper, "_load_onnxruntime", lambda: fake_ort)
    monkeypatch.setattr(wrapper, "build_preprocess", lambda config: config)
    monkeypatch.setattr(wrapper, "build_postprocess", lambda pre_cfg, post_cfg, **kwargs: (pre_cfg, post_cfg, kwargs))

    engine = MBLT_Engine(
        model_cls={
            "file_cfg": {},
            "pre_cfg": {},
            "post_cfg": {},
        },
        framework="onnx",
        model_path=str(onnx_path),
    )

    assert engine.file_cfg["onnx_path"] == str(onnx_path)
    assert fake_ort.session is not None
    assert fake_ort.session.path == str(onnx_path)
    assert engine.framework == "onnx"


def test_engine_init_rejects_conflicting_framework_and_model_path(tmp_path: Path) -> None:
    """Fail fast when the explicit framework conflicts with the local model suffix."""

    mxq_path = tmp_path / "model.mxq"
    mxq_path.write_bytes(b"mxq")

    with pytest.raises(ValueError, match="conflicts with model path"):
        MBLT_Engine(
            model_cls={
                "file_cfg": {},
                "pre_cfg": {},
                "post_cfg": {},
            },
            framework="onnx",
            model_path=str(mxq_path),
        )


def test_engine_init_defaults_to_mxq_without_model_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use MXQ as the fallback framework when no model path is provided."""

    backend_kwargs: dict[str, Any] = {}

    class _FakeBackend:
        def __init__(self, **kwargs: Any) -> None:
            backend_kwargs.update(kwargs)

        def create(self) -> None:
            return None

        def launch(self) -> None:
            return None

        def get_dtype(self) -> str:
            return "DataType.Float32"

        def dispose(self) -> None:
            return None

    monkeypatch.setattr(wrapper, "MobilintNPUBackend", _FakeBackend)
    monkeypatch.setattr(wrapper, "build_preprocess", lambda config: config)
    monkeypatch.setattr(wrapper, "build_postprocess", lambda pre_cfg, post_cfg, **kwargs: (pre_cfg, post_cfg, kwargs))
    monkeypatch.setattr(wrapper.MBLT_Engine, "_download_hub_artifact", lambda self, **kwargs: "/tmp/model.mxq")

    engine = MBLT_Engine(
        model_cls={
            "file_cfg": {
                "repo_id": "mobilint/example",
                "filename": "model.mxq",
                "revision": "main",
            },
            "pre_cfg": {},
            "post_cfg": {},
        },
    )

    try:
        assert engine.framework == "mxq"
        assert "mxq_path" in backend_kwargs
    finally:
        engine.dispose()


def test_custom_coco_data_filters_visible_keypoints(tmp_path: Path) -> None:
    """Keep only images that Ultralytics includes for COCO pose validation."""

    ann_file = tmp_path / "person_keypoints_val2017.json"
    ann_file.write_text(
        """
        {
          "images": [
            {"id": 1, "file_name": "one.jpg", "height": 10, "width": 10},
            {"id": 2, "file_name": "two.jpg", "height": 10, "width": 10},
            {"id": 3, "file_name": "three.jpg", "height": 10, "width": 10}
          ],
          "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "num_keypoints": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "num_keypoints": 5}
          ],
          "categories": [{"id": 1, "name": "person"}]
        }
        """,
        encoding="utf-8",
    )

    dataset = CustomCocodata(str(tmp_path), str(ann_file), min_keypoints=0)

    assert dataset.ids == [2]


def test_crop_mask_matches_ultralytics_rounding() -> None:
    """Crop mask boxes with Ultralytics-compatible rounded boundaries."""

    masks = torch.ones((1, 5, 5), dtype=torch.float32)
    boxes = torch.tensor([[1.2, 1.8, 3.6, 4.2]], dtype=torch.float32)

    cropped = crop_mask(masks, boxes)

    expected = torch.zeros((1, 5, 5), dtype=torch.float32)
    expected[0, 2:4, 1:4] = 1
    assert torch.equal(cropped, expected)


def test_file_config_cleansing_downloads_aries_before_core_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download only the MXQ artifact and try the legacy aries layout first."""

    calls: list[str] = []

    def _fake_download(**kwargs: Any) -> str:
        subfolder = kwargs["subfolder"]
        calls.append(subfolder)
        if subfolder == "aries":
            raise EntryNotFoundError("missing")
        return "/tmp/global8.mxq"

    monkeypatch.setattr(wrapper, "hf_hub_download", _fake_download)

    engine = MBLT_Engine.__new__(MBLT_Engine)
    engine.framework = "mxq"
    engine.file_cfg = {
        "mxq_path": "",
        "repo_id": "mobilint/example",
        "filename": "model.mxq",
        "revision": "main",
        "core_mode": "global8",
    }

    engine.file_config_cleansing()

    assert calls == ["aries", "aries/global8"]
    assert engine.file_cfg["mxq_path"] == "/tmp/global8.mxq"
    assert engine.file_cfg["onnx_filename"] == "model.onnx"
    assert "onnx_path" not in engine.file_cfg


def test_file_config_cleansing_downloads_only_onnx_for_onnx_framework(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download only the ONNX artifact when ONNX inference is requested."""

    calls: list[dict[str, Any]] = []

    def _fake_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        assert kwargs["filename"] == "model.onnx"
        assert "subfolder" not in kwargs
        return "/tmp/model.onnx"

    monkeypatch.setattr(wrapper, "hf_hub_download", _fake_download)

    engine = MBLT_Engine.__new__(MBLT_Engine)
    engine.framework = "onnx"
    engine.file_cfg = {
        "onnx_path": "",
        "repo_id": "mobilint/example",
        "filename": "model.mxq",
        "revision": "main",
        "core_mode": "global8",
    }

    engine.file_config_cleansing()

    assert len(calls) == 1
    assert engine.file_cfg["onnx_filename"] == "model.onnx"
    assert engine.file_cfg["onnx_path"] == "/tmp/model.onnx"
    assert "mxq_path" not in engine.file_cfg or not engine.file_cfg["mxq_path"]


def test_file_config_cleansing_resolves_local_onnx(
    tmp_path: Path,
) -> None:
    """Resolve ONNX file path next to local MXQ file when they exist locally."""

    mxq_path = tmp_path / "model.mxq"
    mxq_path.write_bytes(b"mxq")
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")

    engine = MBLT_Engine.__new__(MBLT_Engine)
    engine.file_cfg = {
        "mxq_path": str(mxq_path),
        "repo_id": "mobilint/example",
        "filename": "model.mxq",
        "revision": "main",
        "core_mode": "global8",
    }

    engine.file_config_cleansing()

    assert engine.file_cfg["mxq_path"] == str(mxq_path)
    assert engine.file_cfg["onnx_filename"] == "model.onnx"
    assert engine.file_cfg["onnx_path"] == str(onnx_path)


def test_prepare_onnx_inputs_keeps_batched_nchw_layout() -> None:
    """Preserve existing NCHW batches when feeding ONNX sessions."""

    class _FakeInput:
        name = "input"
        shape = [1, 3, 224, 224]

    class _FakeSession:
        def get_inputs(self) -> list[_FakeInput]:
            return [_FakeInput()]

    engine = MBLT_Engine.__new__(MBLT_Engine)
    engine.framework = "onnx"
    fake_session = _FakeSession()
    engine._onnx_session = fake_session
    engine.model = fake_session
    engine.input_name = "input"

    batch = torch.zeros((2, 3, 224, 224), dtype=torch.float32)

    inputs = engine._prepare_onnx_inputs(batch)

    assert set(inputs) == {"input"}
    assert inputs["input"].shape == (2, 3, 224, 224)
    assert inputs["input"].dtype == np.float32


def test_prepare_onnx_inputs_transposes_hwc_images() -> None:
    """Convert single HWC images to batched NCHW arrays for ONNX runtime."""

    class _FakeInput:
        name = "input"
        shape = [1, 3, 224, 224]

    class _FakeSession:
        def get_inputs(self) -> list[_FakeInput]:
            return [_FakeInput()]

    engine = MBLT_Engine.__new__(MBLT_Engine)
    engine.framework = "onnx"
    fake_session = _FakeSession()
    engine._onnx_session = fake_session
    engine.model = fake_session
    engine.input_name = "input"

    image = np.zeros((224, 224, 3), dtype=np.float32)

    inputs = engine._prepare_onnx_inputs(image)

    assert inputs["input"].shape == (1, 3, 224, 224)


def test_final_onnx_detections_apply_confidence_threshold() -> None:
    """Filter confidence on already-decoded ONNX detection outputs."""

    pre_cfg = {
        "LetterBox": {
            "img_size": [640, 640],
        }
    }
    post_cfg = {
        "task": "object_detection",
        "nl": 3,
        "nmsfree": True,
        "reg_max": 16,
        "conf_thres": 0.5,
        "iou_thres": 0.7,
    }
    postprocessor = cast(YOLOPostBase, build_postprocess(pre_cfg, post_cfg))
    final_output = np.array(
        [
            [
                [10.0, 20.0, 30.0, 40.0, 0.49, 0.0],
                [11.0, 21.0, 31.0, 41.0, 0.50, 1.0],
                [12.0, 22.0, 32.0, 42.0, 0.90, 2.0],
            ]
        ],
        dtype=np.float32,
    )

    result = postprocessor(final_output)

    assert len(result) == 1
    assert result[0].shape == (1, 6)
    assert torch.all(result[0][:, 4] > 0.5)


def test_final_onnx_detections_normalize_singleton_and_channel_first() -> None:
    """Accept common ONNX final-detection layouts without decoding them again."""

    pre_cfg = {
        "LetterBox": {
            "img_size": [640, 640],
        }
    }
    post_cfg = {
        "task": "object_detection",
        "nl": 3,
        "nmsfree": True,
        "reg_max": 16,
        "conf_thres": 0.5,
        "iou_thres": 0.7,
    }
    postprocessor = cast(YOLOPostBase, build_postprocess(pre_cfg, post_cfg))
    final_output = torch.tensor(
        [
            [
                [10.0, 20.0, 30.0, 40.0, 0.90, 2.0],
                [11.0, 21.0, 31.0, 41.0, 0.40, 1.0],
            ]
        ],
        dtype=torch.float32,
    )

    singleton_result = postprocessor(final_output[:, None])
    channel_first_result = postprocessor(final_output.transpose(1, 2))

    assert torch.equal(singleton_result[0], final_output[0, :1])
    assert torch.equal(channel_first_result[0], final_output[0, :1])


def test_anchorless_pose_nms_prefers_row_major_ambiguous_shape() -> None:
    """Keep ONNX row-major pose tensors row-major when candidate count equals channel count."""

    pre_cfg = {
        "LetterBox": {
            "img_size": [640, 640],
        }
    }
    post_cfg = {
        "task": "pose_estimation",
        "nl": 3,
        "n_extra": 51,
        "reg_max": 16,
        "conf_thres": 0.001,
        "iou_thres": 0.7,
    }
    postprocessor = cast(YOLOPostBase, build_postprocess(pre_cfg, post_cfg))
    row_major = torch.zeros((56, 56), dtype=torch.float32)
    row_major[:, :4] = torch.tensor([10.0, 10.0, 20.0, 20.0])
    row_major[:, 4] = torch.linspace(0.9, 0.1, 56)

    result = postprocessor.nms([row_major])

    assert len(result) == 1
    assert result[0].shape == (1, 57)
    assert torch.allclose(result[0][0, 4], torch.tensor(0.9))


def test_final_onnx_segmentation_normalizes_detections_and_proto() -> None:
    """Use final segmentation detections directly while preserving prototype layout."""

    pre_cfg = {
        "LetterBox": {
            "img_size": [640, 640],
        }
    }
    post_cfg = {
        "task": "instance_segmentation",
        "nl": 3,
        "dflfree": True,
        "nc": 80,
        "n_extra": 32,
        "conf_thres": 0.5,
        "iou_thres": 0.7,
    }
    postprocessor = cast(YOLOPostBase, build_postprocess(pre_cfg, post_cfg))
    final_output = torch.zeros((1, 1, 2, 38), dtype=torch.float32)
    final_output[0, 0, 0, :6] = torch.tensor([10.0, 20.0, 30.0, 40.0, 0.90, 2.0])
    final_output[0, 0, 1, :6] = torch.tensor([11.0, 21.0, 31.0, 41.0, 0.40, 1.0])
    proto = torch.zeros((1, 32, 160, 160), dtype=torch.float32)

    result = postprocessor([final_output, proto])

    assert len(result) == 1
    assert result[0][0].shape == (1, 38)
    assert result[0][1].shape == (1, 640, 640)


def test_raw_mxq_like_outputs_are_not_final_detections() -> None:
    """Do not treat split MXQ-style head tensors as already-decoded detections."""

    pre_cfg = {
        "LetterBox": {
            "img_size": [640, 640],
        }
    }
    post_cfg = {
        "task": "object_detection",
        "nl": 3,
        "dflfree": True,
        "nc": 80,
        "conf_thres": 0.5,
        "iou_thres": 0.7,
    }
    postprocessor = cast(YOLOPostBase, build_postprocess(pre_cfg, post_cfg))
    raw_outputs: ListTensorLike = [
        torch.zeros((1, 80, 80, 4), dtype=torch.float32),
        torch.zeros((1, 80, 80, 80), dtype=torch.float32),
    ]

    detections, proto = postprocessor.extract_final_outputs(raw_outputs)

    assert detections is None
    assert proto is None


def test_anchor_segmentation_ignores_auxiliary_onnx_heads() -> None:
    """Use converted YOLOv5-seg ONNX outputs and ignore auxiliary raw heads."""

    pre_cfg = {
        "LetterBox": {
            "img_size": [640, 640],
        }
    }
    post_cfg = {
        "task": "instance_segmentation",
        "anchors": [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326],
        ],
        "n_extra": 32,
        "nc": 80,
        "conf_thres": 0.5,
        "iou_thres": 0.7,
    }
    postprocessor = cast(YOLOPostBase, build_postprocess(pre_cfg, post_cfg))
    det = torch.zeros((1, 2, 117), dtype=torch.float32)
    proto = torch.zeros((1, 32, 160, 160), dtype=torch.float32)
    aux_heads = [
        torch.zeros((1, 3, 80, 80, 117), dtype=torch.float32),
        torch.zeros((1, 3, 40, 40, 117), dtype=torch.float32),
        torch.zeros((1, 3, 20, 20, 117), dtype=torch.float32),
    ]

    result = postprocessor([det, proto, *aux_heads])

    assert len(result) == 1
    assert result[0][0].shape == (0, 38)
    assert result[0][1].shape == (0, 640, 640)


def test_anchor_segmentation_accepts_mxq_raw_heads() -> None:
    """Use raw YOLOv5-seg MXQ heads when no converted detection tensor is present."""

    pre_cfg = {
        "LetterBox": {
            "img_size": [640, 640],
        }
    }
    post_cfg = {
        "task": "instance_segmentation",
        "anchors": [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326],
        ],
        "n_extra": 32,
        "nc": 80,
        "conf_thres": 0.5,
        "iou_thres": 0.7,
    }
    postprocessor = build_postprocess(pre_cfg, post_cfg)
    raw_outputs = [
        torch.zeros((2, 20, 20, 351), dtype=torch.float32),
        torch.zeros((2, 40, 40, 351), dtype=torch.float32),
        torch.zeros((2, 80, 80, 351), dtype=torch.float32),
        torch.zeros((2, 160, 160, 32), dtype=torch.float32),
    ]

    result = postprocessor(raw_outputs)

    assert len(result) == 2
    assert result[0][0].shape == (0, 38)
    assert result[0][1].shape == (0, 640, 640)
    assert result[1][0].shape == (0, 38)
    assert result[1][1].shape == (0, 640, 640)


def test_anchorless_nms_keeps_best_class_per_box() -> None:
    """Use one detection per box to match Ultralytics best-class NMS behavior."""

    pre_cfg = {
        "LetterBox": {
            "img_size": [640, 640],
        }
    }
    post_cfg = {
        "task": "object_detection",
        "nl": 3,
        "reg_max": 16,
        "nc": 3,
        "conf_thres": 0.25,
        "iou_thres": 0.7,
    }
    postprocessor = cast(YOLOPostBase, build_postprocess(pre_cfg, post_cfg))
    decoded = torch.tensor(
        [
            [
                [10.0, 50.0],
                [10.0, 50.0],
                [20.0, 60.0],
                [20.0, 60.0],
                [0.90, 0.80],
                [0.10, 0.85],
                [0.10, 0.70],
            ]
        ],
        dtype=torch.float32,
    )

    result = postprocessor.nms([decoded[0]])

    assert len(result) == 1
    assert result[0].shape == (2, 6)
    assert torch.equal(result[0][:, 5], torch.tensor([0.0, 1.0]))


def test_scale_coords_matches_ultralytics_rounding() -> None:
    """Match upstream letterbox padding rounding for keypoint scaling."""

    coords = torch.tensor([[[160.0, 0.0, 1.0], [480.0, 640.0, 1.0]]], dtype=torch.float32)

    scaled = scale_coords((640, 640), coords.clone(), (481, 640))

    expected = torch.tensor([[[160.0, 0.0, 1.0], [480.0, 481.0, 1.0]]], dtype=torch.float32)
    assert torch.allclose(scaled, expected)


def test_scale_masks_matches_ultralytics_rounding() -> None:
    """Crop mask padding with the same rounding as upstream Ultralytics."""

    masks = torch.zeros((1, 640, 640), dtype=torch.float32)
    masks[:, 80:560, :] = 1.0

    scaled = scale_masks(masks, (481, 640))

    assert scaled.shape == (1, 481, 640)
    assert float(scaled[:, 0, :].max()) == pytest.approx(0.0)
    assert float(scaled[:, 1, :].max()) > 0.0
    assert float(scaled[:, -1, :].max()) == pytest.approx(1.0)


def test_preprocess_with_metadata_returns_letterbox_ratio_pad() -> None:
    """Expose exact LetterBox ratio and integer padding for validation scaling."""

    engine = MBLT_Engine.__new__(MBLT_Engine)
    engine.pre_cfg = {
        "Reader": {"style": "numpy"},
        "LetterBox": {"img_size": [640, 640]},
        "SetOrder": {"shape": "HWC"},
        "Normalize": {"style": "cv"},
    }
    engine.preprocessor = wrapper.build_preprocess(engine.pre_cfg)
    image = np.zeros((481, 640, 3), dtype=np.uint8)

    processed, metadata = engine.preprocess_with_metadata(image)

    assert processed.shape == (640, 640, 3)
    assert metadata["ratio_pad"] == ((1.0, 1.0), (0, 79))


def test_nmsout2eval_matches_coco_json_format_without_mutation() -> None:
    """Serialize detections like Ultralytics validation without changing NMS output."""

    nms_out = torch.tensor(
        [
            [10.12345, 20.23456, 110.34567, 220.45678, 0.876543, 0.0],
        ],
        dtype=torch.float32,
    )
    original = nms_out.clone()

    labels, boxes, scores = nmsout2eval([nms_out], (640, 640), [(640, 640)])

    assert labels == [[1]]
    assert boxes == [[[10.123, 20.235, 100.222, 200.222]]]
    assert scores == [[0.87654]]
    assert torch.equal(nms_out, original)


def test_nmsout2eval_uses_explicit_ratio_pad() -> None:
    """Use dataloader-provided LetterBox padding instead of recomputing from shape."""

    nms_out = torch.tensor([[0.0, 79.0, 10.0, 89.0, 0.9, 0.0]], dtype=torch.float32)

    _labels, boxes, _scores = nmsout2eval(
        [nms_out],
        (640, 640),
        [(481, 640)],
        ratio_pads=[((1.0, 1.0), (0, 79))],
    )

    assert boxes == [[[0.0, 0.0, 10.0, 10.0]]]
