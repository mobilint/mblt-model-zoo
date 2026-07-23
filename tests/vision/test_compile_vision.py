"""Tests for packaged vision compilation and calibration preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mblt_model_zoo.cli.compile import _run_compile
from mblt_model_zoo.cli.main import build_parser
from mblt_model_zoo.compile import vision as compile_module
from mblt_model_zoo.compile.vision import (
    copy_calibration_subset,
    prepare_calibration_arrays,
    resolve_quantization_values,
    select_calibration_images,
)
from mblt_model_zoo.vision.wrapper import resolve_model_config


def _write_images(directory: Path, names: list[str]) -> list[Path]:
    """Create placeholder image files for selection tests.

    Args:
        directory: Destination directory.
        names: Relative filenames to create.

    Returns:
        Created image paths.
    """

    created: list[Path] = []
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        created.append(path)
    return created


def test_resolve_model_config_handles_alias_and_updated_variant() -> None:
    """Resolve YAML aliases and variant updates through the shared helper."""

    default_config = resolve_model_config("mobilenet_v3_large")
    variant_config = resolve_model_config("mobilenet_v3_large", "IMAGENET1K_V1")

    assert default_config["file_cfg"]["filename"].endswith("IMAGENET1K_V2.mxq")
    assert variant_config["file_cfg"]["filename"].endswith("IMAGENET1K_V1.mxq")
    assert default_config["post_cfg"]["task"] == variant_config["post_cfg"]["task"]


def test_resolve_model_config_derives_onnx_filename() -> None:
    """Derive ONNX artifact names consistently from MXQ artifact names."""

    hub_config = resolve_model_config("alexnet")
    direct_config = resolve_model_config(
        {
            "file_cfg": {
                "filename": "example.mxq",
            },
            "pre_cfg": {},
            "post_cfg": {},
        }
    )
    custom_config = resolve_model_config(
        {
            "file_cfg": {
                "filename": "example.mxq",
                "onnx_filename": "exported-model.onnx",
            },
            "pre_cfg": {},
            "post_cfg": {},
        }
    )

    assert hub_config["file_cfg"]["onnx_filename"] == "alexnet_IMAGENET1K_V1.onnx"
    assert direct_config["file_cfg"]["onnx_filename"] == "example.onnx"
    assert custom_config["file_cfg"]["onnx_filename"] == "exported-model.onnx"


@pytest.mark.parametrize(
    ("task", "relative_dir"),
    [
        ("object_detection", "val2017"),
        ("obb", "images"),
    ],
)
def test_non_imagenet_selection_is_deterministic(task: str, relative_dir: str, tmp_path: Path) -> None:
    """Select deterministic total-count COCO and DOTAv1 samples."""

    _write_images(tmp_path / relative_dir, [f"image-{index}.jpg" for index in range(5)])

    first = select_calibration_images(task, tmp_path, subset_size=3, seed=7)
    second = select_calibration_images(task, tmp_path, subset_size=3, seed=7)

    assert first == second
    assert len(first) == 3


@pytest.mark.parametrize(
    ("task", "relative_dir"),
    [
        ("object_detection", "val2017"),
        ("instance_segmentation", "val2017"),
        ("pose_estimation", "val2017"),
        ("obb", "images"),
    ],
)
def test_selection_defaults_to_seed_zero(task: str, relative_dir: str, tmp_path: Path) -> None:
    """Use seed zero whenever callers omit the vision sampling seed."""

    _write_images(tmp_path / relative_dir, [f"image-{index}.jpg" for index in range(5)])

    assert select_calibration_images(task, tmp_path, subset_size=3) == select_calibration_images(
        task,
        tmp_path,
        subset_size=3,
        seed=0,
    )


def test_widerface_selection_uses_per_category_size(tmp_path: Path) -> None:
    """Select the requested number of images from every WiderFace category."""

    _write_images(tmp_path / "images" / "0--Parade", [f"parade-{index}.jpg" for index in range(3)])
    _write_images(tmp_path / "images" / "1--Handshaking", [f"handshake-{index}.jpg" for index in range(3)])

    first = select_calibration_images("face_detection", tmp_path, subset_size=2, seed=7)
    second = select_calibration_images("face_detection", tmp_path, subset_size=2, seed=7)
    default = select_calibration_images("face_detection", tmp_path)
    seed_zero = select_calibration_images("face_detection", tmp_path, seed=0)

    assert first == second
    assert len(first) == 4
    assert len(default) == 2
    assert default == seed_zero
    assert {path.parent.name for path in first} == {"0--Parade", "1--Handshaking"}


def test_imagenet_selection_uses_per_class_size(tmp_path: Path) -> None:
    """Select the requested number of images independently from each class."""

    _write_images(tmp_path / "class-a", ["same.jpg", "a.jpg"])
    _write_images(tmp_path / "class-b", ["same.jpg", "b.jpg"])

    selected = select_calibration_images("image_classification", tmp_path, subset_size=1)
    seed_zero = select_calibration_images("image_classification", tmp_path, subset_size=1, seed=0)

    assert len(selected) == 2
    assert selected == seed_zero
    assert {path.parent.name for path in selected} == {"class-a", "class-b"}


@pytest.mark.parametrize("subset_size", [0, -1, 4])
def test_selection_rejects_invalid_sizes(subset_size: int, tmp_path: Path) -> None:
    """Reject non-positive and unavailable calibration subset sizes."""

    _write_images(tmp_path / "val2017", ["one.jpg", "two.jpg", "three.jpg"])

    with pytest.raises(ValueError, match="subset_size"):
        select_calibration_images("object_detection", tmp_path, subset_size=subset_size)


def test_flat_subset_names_are_collision_safe(tmp_path: Path) -> None:
    """Preserve images with duplicate basenames from nested source directories."""

    images = _write_images(tmp_path / "dataset", ["images/a/same.jpg", "images/b/same.jpg"])

    copied = copy_calibration_subset(images, tmp_path / "dataset", tmp_path / "subset")

    assert len(copied) == 2
    assert copied[0].name != copied[1].name
    assert all(path.is_file() for path in copied)


@pytest.mark.parametrize("output_name", [".", "parent-output", "val2017/subset"])
def test_subset_rejects_overlapping_output_paths(tmp_path: Path, output_name: str) -> None:
    """Protect the organized source dataset from destructive replacement."""

    data_path = tmp_path / "dataset"
    image_path = _write_images(data_path / "val2017", ["one.jpg"])[0]
    output_path = tmp_path if output_name == "." else data_path / output_name

    with pytest.raises(ValueError, match="must not overlap"):
        compile_module.make_calibration_subset("object_detection", data_path, output_path, subset_size=1)

    assert image_path.is_file()


def test_subset_keeps_existing_output_when_selection_fails(tmp_path: Path) -> None:
    """Leave an existing subset untouched when a replacement cannot be selected."""

    data_path = tmp_path / "dataset"
    output_path = tmp_path / "subset"
    _write_images(data_path / "val2017", ["one.jpg"])
    existing = _write_images(output_path, ["existing.jpg"])[0]

    with pytest.raises(ValueError, match="subset_size"):
        compile_module.make_calibration_subset("object_detection", data_path, output_path, subset_size=2)

    assert existing.read_bytes() == b"existing.jpg"


def test_imagenet_readiness_rejects_partial_class_tree(tmp_path: Path) -> None:
    """Organize ImageNet again when the existing class layout is incomplete."""

    _write_images(tmp_path / "class-a", ["image.jpg"])

    assert not compile_module._dataset_ready("image_classification", tmp_path)


def test_prepare_calibration_arrays_preserves_preprocess_output(tmp_path: Path) -> None:
    """Save contiguous HWC float32 arrays matching engine preprocessing."""

    source = _write_images(tmp_path, ["image.jpg"])[0]
    expected = np.arange(18, dtype=np.float64).reshape(2, 3, 3)

    class _Engine:
        def preprocess(self, image_path: str) -> np.ndarray:
            assert image_path == str(source)
            return expected

    saved = prepare_calibration_arrays(_Engine(), [source], tmp_path / "arrays")  # type: ignore[arg-type]
    actual = np.load(saved[0])

    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected.astype(np.float32))


def test_prepare_calibration_arrays_rejects_chw(tmp_path: Path) -> None:
    """Reject preprocessing output that is not three-channel HWC data."""

    source = _write_images(tmp_path, ["image.jpg"])[0]

    class _Engine:
        def preprocess(self, image_path: str) -> np.ndarray:
            return np.zeros((3, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="HWC three-channel"):
        prepare_calibration_arrays(_Engine(), [source], tmp_path / "arrays")  # type: ignore[arg-type]


def test_quantization_explicit_values_do_not_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give explicit quantization values precedence over hosted metadata."""

    monkeypatch.setattr(
        compile_module,
        "_fetch_quantization_config",
        lambda *args: pytest.fail("hosted metadata should not be fetched"),
    )

    assert resolve_quantization_values({"repo_id": "owner/model"}, 0.98, 0.03) == (0.98, 0.03)


@pytest.mark.parametrize("topk_key", ["topk", "topk_ratio"])
def test_quantization_reads_both_hosted_topk_spellings(topk_key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Read hosted percentile conversion and both supported top-k field names."""

    monkeypatch.setattr(
        compile_module,
        "_fetch_quantization_config",
        lambda repo_id, revision: {"percentile": 0.002, topk_key: 0.04},
    )

    percentile, topk = resolve_quantization_values({"repo_id": "owner/model", "revision": "v1"}, None, None)

    assert percentile == pytest.approx(0.998)
    assert topk == 0.04


def test_quantization_resolves_missing_values_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain an explicit percentile while obtaining only missing top-k metadata."""

    monkeypatch.setattr(
        compile_module,
        "_fetch_quantization_config",
        lambda repo_id, revision: {"percentile": 0.25, "topk_ratio": 0.08},
    )

    assert resolve_quantization_values({"repo_id": "owner/model"}, 0.97, None) == (0.97, 0.08)


def test_quantization_warns_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use documented defaults when optional hosted values are unavailable."""

    monkeypatch.setattr(compile_module, "_fetch_quantization_config", lambda repo_id, revision: None)

    with pytest.warns(UserWarning) as warning_records:
        values = resolve_quantization_values({"repo_id": "owner/model"}, None, None)

    assert values == (compile_module.DEFAULT_PERCENTILE, compile_module.DEFAULT_TOPK_RATIO)
    assert len(warning_records) == 2


def test_quantization_rejects_malformed_hosted_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Raise a contextual error for malformed hosted JSON."""

    metadata_path = tmp_path / "best_result.json"
    metadata_path.write_text("{bad json", encoding="utf-8")
    monkeypatch.setattr(compile_module, "hf_hub_download", lambda **kwargs: str(metadata_path))

    with pytest.raises(ValueError, match="Malformed quantization metadata JSON"):
        compile_module._fetch_quantization_config("owner/model", "main")


def test_quantization_rejects_malformed_config_value(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reject non-numeric hosted quantization fields."""

    metadata_path = tmp_path / "best_result.json"
    metadata_path.write_text(json.dumps({"config": {"percentile": "invalid"}}), encoding="utf-8")
    monkeypatch.setattr(compile_module, "hf_hub_download", lambda **kwargs: str(metadata_path))

    with pytest.raises(ValueError, match="must be numeric"):
        resolve_quantization_values({"repo_id": "owner/model"}, None, 0.01)


def _run_fake_compile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    task: str,
    model_path: Path | None,
    entry_level: str = "data",
    fail: bool = False,
    calls: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Run compilation with fake engine and qbcompiler dependencies.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary test root.
        task: Fake model task.
        model_path: Optional user-supplied model path.
        entry_level: Calibration pipeline level to supply.
        fail: Whether the fake compiler should fail.
        calls: Optional mapping populated even when compilation fails.

    Returns:
        Captured calls and resolved hosted ONNX path.
    """

    calls = {} if calls is None else calls
    hosted_onnx = tmp_path / "hosted-model.onnx"
    hosted_onnx.write_bytes(b"onnx")
    dataset_path = tmp_path / "dataset"
    if task == "image_classification":
        _write_images(dataset_path / "class-a", ["one.jpg"])
    else:
        _write_images(dataset_path / "val2017", ["one.jpg"])

    class _Engine:
        def __init__(self, **kwargs: Any) -> None:
            calls["engine_kwargs"] = kwargs
            resolved_path = Path(str(kwargs.get("model_path", hosted_onnx)))
            self.file_cfg = {"onnx_path": str(resolved_path)}

        def preprocess(self, image_path: str) -> np.ndarray:
            if entry_level == "calibration":
                pytest.fail("ready calibration tensors must skip preprocessing")
            calls.setdefault("preprocessed", []).append(image_path)
            return np.arange(12, dtype=np.float64).reshape(2, 2, 3)

        def dispose(self) -> None:
            calls["disposed"] = True

    class _CalibrationConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    def _compile(**kwargs: Any) -> None:
        calls["compile_kwargs"] = kwargs
        array_dir = Path(kwargs["calib_data_path"])
        calls["temporary_root"] = array_dir.parent
        calls["array"] = np.load(next(array_dir.glob("*.npy")))
        calls["calibration_kwargs"] = kwargs["calibration_config"].kwargs
        if fail:
            raise RuntimeError("compile failed")

    monkeypatch.setattr(compile_module, "MBLT_Engine", _Engine)
    monkeypatch.setattr(compile_module, "_load_qbcompiler", lambda: (_compile, _CalibrationConfig))
    monkeypatch.setattr(
        compile_module,
        "resolve_model_config",
        lambda model_cls, model_type: {
            "file_cfg": {
                "repo_id": "owner/model",
                "revision": "main",
                "filename": "hosted-model.mxq",
                "onnx_path": str(hosted_onnx),
            },
            "post_cfg": {"task": task},
        },
    )
    monkeypatch.setattr(compile_module, "resolve_quantization_values", lambda *args: (0.99, 0.02))
    subset_path: Path | None = None
    calib_data_path: Path | None = None
    original_data_path: Path | None = dataset_path
    if entry_level == "subset":
        original_data_path = None
        subset_path = tmp_path / "provided-subset"
        _write_images(subset_path, ["nested/selected.jpg"])
    elif entry_level == "calibration":
        original_data_path = None
        calib_data_path = tmp_path / "provided-calibration"
        calib_data_path.mkdir()
        np.save(calib_data_path / "selected.npy", np.ones((2, 2, 3), dtype=np.float32))
    elif entry_level != "data":
        raise ValueError(f"Unsupported test entry level: {entry_level}")

    def _ensure_dataset(task: str, data_path: str | Path | None) -> Path:
        calls["ensure_dataset"] = True
        if entry_level != "data":
            pytest.fail(f"{entry_level} input must skip original dataset preparation")
        return dataset_path

    monkeypatch.setattr(compile_module, "ensure_calibration_dataset", _ensure_dataset)

    compile_module.compile_vision_model(
        "fake-model",
        model_type="VARIANT",
        model_path=model_path,
        data_path=original_data_path,
        subset_path=subset_path,
        calib_data_path=calib_data_path,
        subset_size=1,
    )
    return calls, hosted_onnx


def test_compile_uses_local_onnx_and_exact_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Prefer a local ONNX and pass exact Aries/GPU compiler options."""

    local_onnx = tmp_path / "local.onnx"
    local_onnx.write_bytes(b"onnx")
    model_dir = tmp_path / ".mblt_model_zoo"
    monkeypatch.setattr(compile_module, "DEFAULT_MODEL_DIR", model_dir)

    calls, _ = _run_fake_compile(monkeypatch, tmp_path, task="image_classification", model_path=local_onnx)

    assert calls["engine_kwargs"]["model_path"] == str(local_onnx)
    assert calls["engine_kwargs"]["model_type"] == "VARIANT"
    assert calls["compile_kwargs"] | {"calibration_config": None} == {
        "model": str(local_onnx),
        "calib_data_path": calls["compile_kwargs"]["calib_data_path"],
        "save_path": str(model_dir / "local.mxq"),
        "image_channels": 3,
        "backend": "onnx",
        "device": "gpu",
        "target_device": "aries-rb",
        "inference_scheme": "all",
        "calibration_config": None,
    }
    assert calls["calibration_kwargs"]["output"] == 0
    assert calls["calibration_kwargs"]["method"] == 1
    assert calls["calibration_kwargs"]["mode"] == 1
    assert calls["array"].dtype == np.float32
    assert calls["disposed"] is True
    assert not calls["temporary_root"].exists()


def test_compile_ignores_missing_local_path_and_uses_hosted_onnx(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Construct the engine without a nonexistent local path so Hub fallback applies."""

    model_dir = tmp_path / ".mblt_model_zoo"
    monkeypatch.setattr(compile_module, "DEFAULT_MODEL_DIR", model_dir)
    calls, hosted_onnx = _run_fake_compile(
        monkeypatch,
        tmp_path,
        task="object_detection",
        model_path=tmp_path / "missing.onnx",
    )

    assert "model_path" not in calls["engine_kwargs"]
    assert calls["compile_kwargs"]["model"] == str(hosted_onnx)
    assert calls["compile_kwargs"]["save_path"] == str(model_dir / "hosted-model.mxq")
    assert calls["calibration_kwargs"]["output"] == 1


def test_compile_starts_from_provided_image_subset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Skip original dataset preparation and sampling for a supplied image subset."""

    calls, _ = _run_fake_compile(
        monkeypatch,
        tmp_path,
        task="object_detection",
        model_path=None,
        entry_level="subset",
    )

    assert "ensure_dataset" not in calls
    assert calls["preprocessed"] == [str(tmp_path / "provided-subset" / "nested" / "selected.jpg")]
    assert not calls["temporary_root"].exists()


def test_compile_uses_provided_calibration_dataset_directly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pass ready NumPy tensors directly without image dataset processing."""

    calls, _ = _run_fake_compile(
        monkeypatch,
        tmp_path,
        task="object_detection",
        model_path=None,
        entry_level="calibration",
    )

    calibration_path = tmp_path / "provided-calibration"
    assert "ensure_dataset" not in calls
    assert "engine_kwargs" not in calls
    assert "preprocessed" not in calls
    assert calls["compile_kwargs"]["calib_data_path"] == str(calibration_path)
    assert (calibration_path / "selected.npy").is_file()


def test_compile_rejects_multiple_data_levels() -> None:
    """Reject ambiguous original, subset, and calibration path combinations."""

    with pytest.raises(ValueError, match="Provide only one calibration pipeline input"):
        compile_module.compile_vision_model("alexnet", data_path="original", subset_path="subset")


def test_compile_cleans_up_and_disposes_after_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Dispose the engine and delete temporary images and arrays after compiler failure."""

    calls: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="compile failed"):
        _run_fake_compile(
            monkeypatch,
            tmp_path,
            task="image_classification",
            model_path=None,
            fail=True,
            calls=calls,
        )

    assert calls["disposed"] is True
    assert not calls["temporary_root"].exists()


def test_cli_compile_parser_and_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Parse compile options and dispatch them to the packaged API."""

    parser = build_parser()
    args = parser.parse_args(
        [
            "compile",
            "--model-cls",
            "alexnet",
            "--onnx-path",
            "./alexnet.onnx",
            "--subset-path",
            "./sampled-images",
            "--subset-size",
            "2",
            "--percentile",
            "0.99",
        ]
    )
    output = tmp_path / "alexnet.mxq"
    calls: dict[str, Any] = {}

    def _compile(**kwargs: Any) -> Path:
        calls.update(kwargs)
        return output

    monkeypatch.setattr("mblt_model_zoo.compile.vision.compile_vision_model", _compile)

    assert args.model_path == "./alexnet.onnx"
    assert args._handler(args) == 0
    assert calls["model_cls"] == "alexnet"
    assert calls["subset_path"] == "./sampled-images"
    assert calls["calib_data_path"] is None
    assert calls["subset_size"] == 2
    assert calls["percentile"] == 0.99


def test_cli_compile_rejects_multiple_data_levels() -> None:
    """Make the three CLI data entry levels mutually exclusive."""

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "compile",
                "--model-cls",
                "alexnet",
                "--data-path",
                "./imagenet",
                "--calib-data-path",
                "./arrays",
            ]
        )


def test_cli_compile_reports_missing_qbcompiler(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit with a concise installation error when qbcompiler is unavailable."""

    parser = build_parser()
    args = parser.parse_args(["compile", "--model-cls", "alexnet"])
    monkeypatch.setattr(
        "mblt_model_zoo.compile.vision.compile_vision_model",
        lambda **kwargs: (_ for _ in ()).throw(ImportError("Vision compilation requires qbcompiler>=1.2.0.")),
    )

    with pytest.raises(SystemExit) as exc_info:
        _run_compile(args)

    assert exc_info.value.code == 2
    assert "qbcompiler>=1.2.0" in capsys.readouterr().err
