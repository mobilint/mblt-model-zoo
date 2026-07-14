"""Vision model compilation and calibration-data preparation."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from mblt_model_zoo.vision.datasets import get_dataset_config_for_task
from mblt_model_zoo.vision.wrapper import MOBILINT_CACHE_DIR, MBLT_Engine, resolve_model_config

DEFAULT_PERCENTILE = 0.9999
DEFAULT_TOPK_RATIO = 0.01
DEFAULT_SEED = 0
DEFAULT_MODEL_DIR = Path(MOBILINT_CACHE_DIR)
DEFAULT_SUBSET_SIZES = {
    "image_classification": 1,
    "object_detection": 100,
    "instance_segmentation": 100,
    "pose_estimation": 100,
    "face_detection": 1,
    "obb": 100,
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _normalize_task(task: str) -> str:
    """Normalize supported task aliases.

    Args:
        task: Task value from a model postprocess configuration.

    Returns:
        Canonical task name used by the dataset registry.

    Raises:
        ValueError: If the task is not supported by vision compilation.
    """

    normalized = task.lower()
    if normalized == "oriented_bounding_boxes":
        normalized = "obb"
    if normalized not in DEFAULT_SUBSET_SIZES:
        raise ValueError(f"Vision compilation does not support task `{task}`.")
    return normalized


def _dataset_ready(task: str, data_path: Path) -> bool:
    """Return whether an organized dataset contains calibration images.

    Args:
        task: Canonical vision task name.
        data_path: Organized dataset root.

    Returns:
        Whether the expected image layout exists and is non-empty.
    """

    if task == "image_classification":
        class_dirs = [child for child in data_path.iterdir() if child.is_dir()] if data_path.is_dir() else []
        return len(class_dirs) == 1000 and all(
            sum(path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES for path in class_dir.iterdir()) == 50
            for class_dir in class_dirs
        )
    image_dir = {
        "object_detection": data_path / "val2017",
        "instance_segmentation": data_path / "val2017",
        "pose_estimation": data_path / "val2017",
        "face_detection": data_path / "images",
        "obb": data_path / "images" / "val",
    }[task]
    return image_dir.is_dir() and any(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES for path in image_dir.rglob("*")
    )


def _organize_dataset(task: str, data_path: Path) -> None:
    """Organize the registry-backed dataset required by a task.

    Args:
        task: Canonical vision task name.
        data_path: Destination dataset root.
    """

    from mblt_model_zoo.vision.utils.datasets import (
        organize_coco,
        organize_dotav1,
        organize_imagenet,
        organize_widerface,
    )

    config = get_dataset_config_for_task(task)
    download = config.get("download")
    if not isinstance(download, dict):
        raise ValueError(f"Dataset `{config.get('name', task)}` does not define download metadata.")

    data_path.parent.mkdir(parents=True, exist_ok=True)
    if task == "image_classification":
        organize_imagenet(
            image_dir=str(download["images"]),
            xml_dir=str(download["annotations"]),
            output_dir=str(data_path),
        )
    elif task in {"object_detection", "instance_segmentation", "pose_estimation"}:
        organize_coco(
            image_dir=str(download["images"]),
            annotation_dir=str(download["annotations"]),
            output_dir=str(data_path),
        )
    elif task == "face_detection":
        organize_widerface(
            image_dir=str(download["images"]),
            annotation_dir=str(download["annotations"]),
            output_dir=str(data_path),
        )
    else:
        organize_dotav1(dataset_path=str(download["url"]), output_dir=str(data_path))


def ensure_calibration_dataset(task: str, data_path: str | Path | None = None) -> Path:
    """Resolve and, when needed, organize a calibration dataset.

    Args:
        task: Vision task name.
        data_path: Optional organized dataset root.

    Returns:
        Expanded organized dataset root.
    """

    normalized_task = _normalize_task(task)
    config = get_dataset_config_for_task(normalized_task)
    resolved_path = Path(data_path if data_path is not None else config["path"]).expanduser()
    if not _dataset_ready(normalized_task, resolved_path):
        _organize_dataset(normalized_task, resolved_path)
    if not _dataset_ready(normalized_task, resolved_path):
        raise ValueError(f"Organized calibration dataset is missing images at {resolved_path}.")
    return resolved_path


def _validate_subset_size(subset_size: int) -> None:
    """Validate a requested calibration subset size.

    Args:
        subset_size: Requested per-class or total sample count.

    Raises:
        ValueError: If the size is not positive.
    """

    if subset_size <= 0:
        raise ValueError("subset_size must be greater than zero.")


def select_calibration_images(
    task: str,
    data_path: str | Path,
    subset_size: int | None = None,
    seed: int = DEFAULT_SEED,
) -> list[Path]:
    """Select deterministic calibration images from an organized dataset.

    For ImageNet and WiderFace, ``subset_size`` is the number selected from
    every category. For all other datasets it is the total number of selected
    validation images.

    Args:
        task: Vision task name.
        data_path: Organized dataset root.
        subset_size: Optional sample count, using task-specific defaults when omitted.
        seed: Random selection seed.

    Returns:
        Selected image paths in deterministic order.

    Raises:
        ValueError: If the dataset layout or requested size is invalid.
    """

    import random

    normalized_task = _normalize_task(task)
    root = Path(data_path).expanduser()
    requested_size = DEFAULT_SUBSET_SIZES[normalized_task] if subset_size is None else subset_size
    _validate_subset_size(requested_size)
    random_generator = random.Random(seed)

    if normalized_task in {"image_classification", "face_detection"}:
        category_root = root if normalized_task == "image_classification" else root / "images"
        dataset_name = "ImageNet" if normalized_task == "image_classification" else "WiderFace"
        category_dirs = (
            sorted(path for path in category_root.iterdir() if path.is_dir()) if category_root.is_dir() else []
        )
        if not category_dirs:
            raise ValueError(f"No {dataset_name} category directories found in {category_root}.")
        selected: list[Path] = []
        for category_dir in category_dirs:
            images = sorted(
                path for path in category_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if requested_size > len(images):
                raise ValueError(
                    f"subset_size ({requested_size}) exceeds the {len(images)} available images in {category_dir.name}."
                )
            selected.extend(random_generator.sample(images, requested_size))
        return selected

    image_dir = {
        "object_detection": root / "val2017",
        "instance_segmentation": root / "val2017",
        "pose_estimation": root / "val2017",
        "obb": root / "images" / "val",
    }[normalized_task]
    images = (
        sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if image_dir.is_dir()
        else []
    )
    if not images:
        raise ValueError(f"No calibration images found in {image_dir}.")
    if requested_size > len(images):
        raise ValueError(f"subset_size ({requested_size}) exceeds the {len(images)} available images in {image_dir}.")
    return random_generator.sample(images, requested_size)


def copy_calibration_subset(images: Sequence[Path], data_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Copy selected images into a flat directory using collision-safe names.

    Args:
        images: Selected source image paths.
        data_path: Dataset root used to derive stable relative names.
        output_dir: Destination directory.

    Returns:
        Copied image paths in input order.
    """

    root = Path(data_path).expanduser().resolve()
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for image_path in images:
        source = image_path.resolve()
        try:
            relative_name = source.relative_to(root).as_posix()
        except ValueError:
            relative_name = source.as_posix()
        digest = hashlib.sha256(relative_name.encode()).hexdigest()[:12]
        target = destination / f"{digest}_{source.name}"
        if target.exists():
            raise ValueError(f"Calibration subset filename collision for {source}.")
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def make_calibration_subset(
    task: str,
    data_path: str | Path,
    output_dir: str | Path,
    subset_size: int | None = None,
    seed: int = DEFAULT_SEED,
) -> list[Path]:
    """Select and copy a deterministic flat calibration subset.

    Args:
        task: Vision task name.
        data_path: Organized dataset root.
        output_dir: Destination for copied images.
        subset_size: Optional task-specific selection count.
        seed: Random selection seed.

    Returns:
        Copied image paths.
    """

    source_root = Path(data_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if source_root.is_relative_to(destination) or destination.is_relative_to(source_root):
        raise ValueError("output_dir must not overlap data_path.")
    images = select_calibration_images(task, data_path, subset_size, seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_destination = Path(mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        copied = copy_calibration_subset(images, data_path, temporary_destination)
        backup_destination: Path | None = None
        if destination.exists():
            backup_destination = Path(mkdtemp(prefix=f".{destination.name}-backup-", dir=destination.parent))
            backup_destination.rmdir()
            destination.replace(backup_destination)
        try:
            temporary_destination.replace(destination)
        except BaseException:
            if backup_destination is not None and not destination.exists():
                backup_destination.replace(destination)
            raise
        if backup_destination is not None:
            shutil.rmtree(backup_destination)
        return [destination / path.name for path in copied]
    finally:
        if temporary_destination.exists():
            shutil.rmtree(temporary_destination)


def _as_hwc_float32(value: Any, image_path: Path) -> np.ndarray:
    """Validate and convert engine preprocessing output for qbcompiler.

    Args:
        value: Engine preprocessing result.
        image_path: Source image used for error context.

    Returns:
        Contiguous HWC float32 array.

    Raises:
        TypeError: If preprocessing did not return a tensor or array.
        ValueError: If preprocessing did not return HWC three-channel data.
    """

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        array = value
    else:
        raise TypeError(f"Preprocessing {image_path} returned unsupported type {type(value).__name__}.")
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            f"Preprocessing {image_path} must produce an HWC three-channel array; got shape {array.shape}."
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def prepare_calibration_arrays(engine: MBLT_Engine, images: Sequence[Path], output_dir: str | Path) -> list[Path]:
    """Preprocess calibration images and save one NumPy array per sample.

    Args:
        engine: ONNX vision engine providing authoritative preprocessing.
        images: Flat calibration image paths.
        output_dir: Destination directory for ``.npy`` arrays.

    Returns:
        Saved NumPy paths.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for index, image_path in enumerate(images):
        array = _as_hwc_float32(engine.preprocess(str(image_path)), image_path)
        array_path = destination / f"{index:06d}.npy"
        np.save(array_path, array)
        saved.append(array_path)
    return saved


def get_subset_images(subset_path: str | Path) -> list[Path]:
    """Load all images from an already-sampled subset.

    Args:
        subset_path: Root containing sampled images, either flat or nested.

    Returns:
        Image paths in deterministic order.

    Raises:
        ValueError: If the subset contains no supported image files.
    """

    root = Path(subset_path).expanduser()
    images = (
        sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if root.is_dir()
        else []
    )
    if not images:
        raise ValueError(f"No calibration subset images found in {root}.")
    return images


def validate_calibration_dataset(calib_data_path: str | Path) -> Path:
    """Validate a ready directory of preprocessed calibration arrays.

    Args:
        calib_data_path: Directory containing HWC float32 ``.npy`` tensors.

    Returns:
        Expanded calibration directory path.

    Raises:
        ValueError: If the directory is empty or contains an invalid calibration tensor.
    """

    root = Path(calib_data_path).expanduser()
    array_paths = sorted(root.glob("*.npy")) if root.is_dir() else []
    if not array_paths:
        raise ValueError(f"No calibration .npy files found in {root}.")
    for array_path in array_paths:
        try:
            array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Unable to load calibration tensor {array_path}: {exc}.") from exc
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Calibration tensor {array_path} must be HWC with three channels; got {array.shape}.")
        if array.dtype != np.float32:
            raise ValueError(f"Calibration tensor {array_path} must use float32; got {array.dtype}.")
        if not array.flags.c_contiguous:
            raise ValueError(f"Calibration tensor {array_path} must be C-contiguous.")
    return root


def _validate_data_level_paths(
    data_path: str | Path | None,
    subset_path: str | Path | None,
    calib_data_path: str | Path | None,
) -> None:
    """Ensure at most one calibration pipeline entry level is supplied.

    Args:
        data_path: Original organized dataset root.
        subset_path: Already-sampled image subset root.
        calib_data_path: Ready preprocessed NumPy dataset root.

    Raises:
        ValueError: If more than one data level is supplied.
    """

    supplied = [
        name
        for name, value in (
            ("data_path", data_path),
            ("subset_path", subset_path),
            ("calib_data_path", calib_data_path),
        )
        if value is not None
    ]
    if len(supplied) > 1:
        raise ValueError(
            "Provide only one calibration pipeline input: `data_path`, `subset_path`, or `calib_data_path`; "
            f"got {', '.join(supplied)}."
        )


def _load_qbcompiler() -> tuple[Callable[..., Any], type[Any]]:
    """Load qbcompiler only when compilation is requested.

    Returns:
        The ``mxq_compile`` function and ``CalibrationConfig`` class.

    Raises:
        ImportError: If qbcompiler is not installed.
    """

    try:
        module = importlib.import_module("qbcompiler")
    except ImportError as exc:
        raise ImportError(
            "Vision compilation requires qbcompiler>=1.2.0. Install the compiler package supplied by Mobilint."
        ) from exc
    return module.mxq_compile, module.CalibrationConfig


def _validate_ratio(name: str, value: Any) -> float:
    """Validate a quantization ratio.

    Args:
        name: Field name used in an error message.
        value: Candidate numeric ratio.

    Returns:
        Validated float value.

    Raises:
        ValueError: If the value is non-numeric or outside zero to one.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Quantization metadata `{name}` must be numeric, got {value!r}.")
    resolved = float(value)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError(f"Quantization metadata `{name}` must be between 0 and 1, got {resolved}.")
    return resolved


def _fetch_quantization_config(repo_id: str, revision: str) -> dict[str, Any] | None:
    """Fetch optional hosted Aries quantization metadata.

    Args:
        repo_id: Hugging Face model repository ID.
        revision: Repository revision.

    Returns:
        Hosted ``config`` mapping, or ``None`` when the optional file is unavailable.

    Raises:
        ValueError: If the hosted JSON or its ``config`` field is malformed.
    """

    try:
        metadata_path = hf_hub_download(
            repo_id=repo_id,
            filename="best_result.json",
            subfolder="aries",
            revision=revision,
        )
    except (HfHubHTTPError, OSError) as exc:
        warnings.warn(
            f"Unable to load optional quantization metadata for {repo_id}: {exc}. Using fallback values.",
            stacklevel=2,
        )
        return None

    try:
        with Path(metadata_path).open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed quantization metadata JSON at {metadata_path}: {exc.msg}.") from exc
    if not isinstance(metadata, dict) or not isinstance(metadata.get("config"), dict):
        raise ValueError(f"Quantization metadata at {metadata_path} must contain a `config` object.")
    return metadata["config"]


def resolve_quantization_values(
    file_cfg: dict[str, Any],
    percentile: float | None,
    topk_ratio: float | None,
) -> tuple[float, float]:
    """Resolve explicit, hosted, and fallback quantization values independently.

    Args:
        file_cfg: Resolved model file configuration.
        percentile: Explicit compiler percentile override.
        topk_ratio: Explicit compiler top-k ratio override.

    Returns:
        Resolved percentile and top-k ratio.

    Raises:
        ValueError: If explicit or hosted values are malformed.
    """

    resolved_percentile = _validate_ratio("percentile", percentile) if percentile is not None else None
    resolved_topk = _validate_ratio("topk_ratio", topk_ratio) if topk_ratio is not None else None
    hosted_config: dict[str, Any] | None = None
    if resolved_percentile is None or resolved_topk is None:
        repo_id = file_cfg.get("repo_id")
        revision = file_cfg.get("revision", "main")
        if isinstance(repo_id, str) and repo_id:
            hosted_config = _fetch_quantization_config(repo_id, str(revision))
        else:
            warnings.warn("Model configuration has no repository ID; using fallback quantization values.", stacklevel=2)

    if resolved_percentile is None and hosted_config is not None and "percentile" in hosted_config:
        hosted_percentile = _validate_ratio("config.percentile", hosted_config["percentile"])
        resolved_percentile = 1.0 - hosted_percentile
    if resolved_topk is None and hosted_config is not None:
        hosted_topk = hosted_config.get("topk_ratio", hosted_config.get("topk"))
        if hosted_topk is not None:
            resolved_topk = _validate_ratio("config.topk_ratio", hosted_topk)

    if resolved_percentile is None:
        warnings.warn(f"Quantization percentile is unavailable; using {DEFAULT_PERCENTILE}.", stacklevel=2)
        resolved_percentile = DEFAULT_PERCENTILE
    if resolved_topk is None:
        warnings.warn(f"Quantization top-k ratio is unavailable; using {DEFAULT_TOPK_RATIO}.", stacklevel=2)
        resolved_topk = DEFAULT_TOPK_RATIO
    return resolved_percentile, resolved_topk


def _configured_onnx_filename(file_cfg: dict[str, Any]) -> str | None:
    """Derive the configured ONNX filename.

    Args:
        file_cfg: Resolved model file configuration.

    Returns:
        ONNX filename when it can be determined.
    """

    onnx_filename = file_cfg.get("onnx_filename")
    if isinstance(onnx_filename, str) and onnx_filename:
        return onnx_filename
    filename = file_cfg.get("filename")
    if isinstance(filename, str) and filename:
        return f"{Path(filename).stem}.onnx"
    return None


def compile_vision_model(
    model_cls: str,
    *,
    model_type: str = "DEFAULT",
    model_path: str | Path | None = None,
    onnx_path: str | Path | None = None,
    data_path: str | Path | None = None,
    subset_path: str | Path | None = None,
    calib_data_path: str | Path | None = None,
    save_path: str | Path | None = None,
    subset_size: int | None = None,
    seed: int = DEFAULT_SEED,
    percentile: float | None = None,
    topk_ratio: float | None = None,
) -> Path:
    """Compile a configured vision ONNX model into an Aries MXQ artifact.

    Args:
        model_cls: Vision model name or YAML path.
        model_type: Model variant from the YAML configuration.
        model_path: Preferred local ONNX path compatibility option.
        onnx_path: Local ONNX path alias.
        data_path: Original organized dataset root. The pipeline organizes it when needed, samples
            an image subset, and preprocesses that subset.
        subset_path: Already-sampled image subset. The pipeline preprocesses every image directly
            without organizing or sampling an original dataset.
        calib_data_path: Ready directory of HWC float32 ``.npy`` tensors. The pipeline passes it
            directly to qbcompiler without dataset preparation, sampling, or preprocessing.
        save_path: Output MXQ path. Defaults to the ONNX stem under ``~/.mblt_model_zoo``.
        subset_size: ImageNet/WiderFace per-category count or total count for other datasets.
        seed: Deterministic calibration selection seed.
        percentile: Explicit quantization percentile.
        topk_ratio: Explicit quantization top-k ratio.

    Returns:
        Output MXQ path.

    Raises:
        ImportError: If qbcompiler is unavailable.
        ValueError: If model metadata, calibration data, or quantization values are invalid.
    """

    _validate_data_level_paths(data_path, subset_path, calib_data_path)
    mxq_compile, calibration_config_class = _load_qbcompiler()
    model_config = resolve_model_config(model_cls, model_type)
    file_cfg = model_config.get("file_cfg")
    post_cfg = model_config.get("post_cfg")
    if not isinstance(file_cfg, dict) or not isinstance(post_cfg, dict):
        raise ValueError("Resolved vision model configuration requires `file_cfg` and `post_cfg` objects.")

    task = _normalize_task(str(post_cfg.get("task", "")))
    selected_local_path = model_path if model_path is not None else onnx_path
    engine_kwargs: dict[str, Any] = {
        "model_cls": model_cls,
        "model_type": model_type,
        "framework": "onnx",
    }
    if selected_local_path is not None:
        expanded_local_path = Path(selected_local_path).expanduser()
        if expanded_local_path.is_file():
            engine_kwargs["model_path"] = str(expanded_local_path)

    engine: MBLT_Engine | None = None
    try:
        engine = MBLT_Engine(**engine_kwargs)
        resolved_onnx = Path(str(engine.file_cfg.get("onnx_path", ""))).expanduser()
        if not resolved_onnx.is_file():
            configured_name = _configured_onnx_filename(file_cfg) or "the configured ONNX artifact"
            raise FileNotFoundError(f"Unable to resolve {configured_name} for model `{model_cls}`.")

        output_path = (
            Path(save_path).expanduser() if save_path is not None else DEFAULT_MODEL_DIR / f"{resolved_onnx.stem}.mxq"
        )
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_percentile, resolved_topk = resolve_quantization_values(file_cfg, percentile, topk_ratio)
        calibration_config = calibration_config_class(
            method=1,
            output=0 if task == "image_classification" else 1,
            mode=1,
            max_percentile={"percentile": resolved_percentile, "topk_ratio": resolved_topk},
        )

        def _compile(calibration_path: Path) -> None:
            """Run qbcompiler with a resolved calibration directory."""

            mxq_compile(
                model=str(resolved_onnx),
                calib_data_path=str(calibration_path),
                save_path=str(output_path),
                image_channels=3,
                backend="onnx",
                device="gpu",
                target_device="aries-rb",
                inference_scheme="all",
                calibration_config=calibration_config,
            )

        if calib_data_path is not None:
            _compile(validate_calibration_dataset(calib_data_path))
            return output_path

        with TemporaryDirectory(prefix="mblt-vision-calibration-") as temporary_root:
            temporary_path = Path(temporary_root)
            if subset_path is not None:
                subset_images = get_subset_images(subset_path)
            else:
                dataset_path = ensure_calibration_dataset(task, data_path)
                subset_images = make_calibration_subset(
                    task,
                    dataset_path,
                    temporary_path / "images",
                    subset_size=subset_size,
                    seed=seed,
                )
            array_dir = temporary_path / "arrays"
            prepare_calibration_arrays(engine, subset_images, array_dir)
            _compile(array_dir)
        return output_path
    finally:
        if engine is not None:
            engine.dispose()


__all__ = [
    "compile_vision_model",
    "copy_calibration_subset",
    "ensure_calibration_dataset",
    "get_subset_images",
    "make_calibration_subset",
    "prepare_calibration_arrays",
    "resolve_quantization_values",
    "select_calibration_images",
    "validate_calibration_dataset",
]
