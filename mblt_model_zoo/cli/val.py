"""Vision validation CLI command."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ._vision import add_threshold_args, parse_target_clusters, parse_target_cores

DEFAULT_IMAGENET_IMAGE_SOURCE = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
DEFAULT_IMAGENET_XML_SOURCE = "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz"
DEFAULT_COCO_IMAGE_SOURCE = "http://images.cocodataset.org/zips/val2017.zip"
DEFAULT_COCO_ANNOTATION_SOURCE = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
DEFAULT_WIDERFACE_IMAGE_SOURCE = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_val.zip"
DEFAULT_WIDERFACE_ANNOTATION_SOURCE = (
    "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/wider_face_split.zip"
)


def _candidate_search_roots(data_path: str) -> list[Path]:
    """Returns directories to inspect for existing raw dataset sources."""

    root = Path(data_path).expanduser()
    candidates = [root, root.parent, Path.cwd()]
    ordered: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def _find_existing_source(data_path: str, candidate_names: list[str]) -> str | None:
    """Finds a nearby raw archive or extracted dataset directory."""

    for root in _candidate_search_roots(data_path):
        for name in candidate_names:
            candidate = root / name
            if candidate.exists():
                return str(candidate)
    return None


def _normalize_coco_annotation_source(annotation_dir: str | None) -> str | None:
    """Normalizes a COCO annotation source for the organizer contract.

    The COCO organizer expects either the annotation archive or the extracted
    parent directory that contains an ``annotations`` subdirectory. When source
    discovery finds the extracted leaf ``annotations`` directory directly,
    return its parent so downstream code does not resolve ``annotations``
    twice.
    """

    if annotation_dir is None:
        return None

    candidate = Path(annotation_dir).expanduser()
    if candidate.is_dir() and candidate.name == "annotations":
        return str(candidate.parent)
    return annotation_dir


def _resolve_imagenet_sources(args: argparse.Namespace, data_path: str) -> tuple[str, str]:
    """Resolves local or remote sources for ImageNet organization."""

    image_dir = args.image_dir
    xml_dir = args.xml_dir
    if not args.force_organize:
        image_dir = image_dir or _find_existing_source(data_path, ["ILSVRC2012_img_val.tar", "ILSVRC2012_img_val"])
        xml_dir = xml_dir or _find_existing_source(data_path, ["ILSVRC2012_bbox_val_v3.tgz", "ILSVRC2012_bbox_val_v3"])
    return image_dir or DEFAULT_IMAGENET_IMAGE_SOURCE, xml_dir or DEFAULT_IMAGENET_XML_SOURCE


def _resolve_coco_sources(args: argparse.Namespace, data_path: str) -> tuple[str, str]:
    """Resolves local or remote sources for COCO organization."""

    image_dir = args.image_dir
    annotation_dir = _normalize_coco_annotation_source(args.annotation_dir)
    if not args.force_organize:
        image_dir = image_dir or _find_existing_source(data_path, ["val2017.zip", "val2017"])
        annotation_dir = annotation_dir or _normalize_coco_annotation_source(
            _find_existing_source(
                data_path,
                ["annotations_trainval2017.zip", "annotations_trainval2017", "annotations"],
            )
        )
    return image_dir or DEFAULT_COCO_IMAGE_SOURCE, annotation_dir or DEFAULT_COCO_ANNOTATION_SOURCE


def _resolve_widerface_sources(args: argparse.Namespace, data_path: str) -> tuple[str, str]:
    """Resolves local or remote sources for WiderFace organization."""

    image_dir = args.image_dir
    annotation_dir = args.annotation_dir
    if not args.force_organize:
        image_dir = image_dir or _find_existing_source(data_path, ["WIDER_val.zip", "WIDER_val"])
        annotation_dir = annotation_dir or _find_existing_source(
            data_path,
            ["wider_face_split.zip", "wider_face_split"],
        )
    return image_dir or DEFAULT_WIDERFACE_IMAGE_SOURCE, annotation_dir or DEFAULT_WIDERFACE_ANNOTATION_SOURCE


def _default_data_path_for_task(task: str) -> str:
    """Returns the default organized dataset path for a vision task."""

    if task == "image_classification":
        return os.path.expanduser("~/.mblt_model_zoo/datasets/imagenet")
    if task in {"object_detection", "instance_segmentation", "pose_estimation"}:
        return os.path.expanduser("~/.mblt_model_zoo/datasets/coco")
    if task == "face_detection":
        return os.path.expanduser("~/.mblt_model_zoo/datasets/widerface")
    raise SystemExit(f"Unsupported vision task for validation: {task}")


def _dataset_ready(task: str, data_path: str) -> bool:
    """Checks whether the organized dataset appears ready for validation."""

    root = Path(data_path).expanduser()
    if task == "image_classification":
        return root.is_dir() and any(child.is_dir() for child in root.iterdir()) if root.exists() else False
    if task in {"object_detection", "instance_segmentation"}:
        return (root / "val2017").is_dir() and (root / "instances_val2017.json").is_file()
    if task == "pose_estimation":
        return (root / "val2017").is_dir() and (root / "person_keypoints_val2017.json").is_file()
    if task == "face_detection":
        return (root / "images").is_dir() and (root / "wider_face_val_bbx_gt.txt").is_file()
    return False


def _ensure_dataset(args: argparse.Namespace, task: str) -> str:
    """Organizes the dataset automatically when the expected layout is missing."""

    data_path = os.path.expanduser(args.data_path or _default_data_path_for_task(task))
    if _dataset_ready(task, data_path) and not args.force_organize:
        print(f"Using organized dataset at {data_path}")
        return data_path

    try:
        from mblt_model_zoo.vision.utils.datasets import organize_coco, organize_imagenet, organize_widerface
    except ImportError as exc:
        print(f"Missing dependencies for vision dataset organization: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(f"Preparing validation dataset for task `{task}` at {data_path}...")
    if task == "image_classification":
        image_dir, xml_dir = _resolve_imagenet_sources(args, data_path)
        organize_imagenet(
            image_dir=image_dir,
            xml_dir=xml_dir,
            output_dir=data_path,
        )
    elif task in {"object_detection", "instance_segmentation", "pose_estimation"}:
        image_dir, annotation_dir = _resolve_coco_sources(args, data_path)
        organize_coco(
            image_dir=image_dir,
            annotation_dir=annotation_dir,
            output_dir=data_path,
        )
    elif task == "face_detection":
        image_dir, annotation_dir = _resolve_widerface_sources(args, data_path)
        organize_widerface(
            image_dir=image_dir,
            annotation_dir=annotation_dir,
            output_dir=data_path,
        )
    else:
        raise SystemExit(f"Unsupported vision task for validation: {task}")

    return data_path


def _run_validation(args: argparse.Namespace) -> float:
    """Runs model validation on the dataset associated with the model task."""

    try:
        from mblt_model_zoo.vision import MBLT_Engine
        from mblt_model_zoo.vision.utils.evaluation import eval_coco, eval_imagenet
        from mblt_model_zoo.vision.wrapper import normalize_core_mode
    except ImportError as exc:
        print(f"Missing dependencies for vision CLI: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    model = MBLT_Engine(
        model_cls=args.model,
        model_type=args.model_type,
        framework=args.framework,
        model_path=args.model_path,
        dev_no=args.dev_no,
        core_mode=normalize_core_mode(args.core_mode),
        target_cores=args.target_cores,
        target_clusters=args.target_clusters,
    )
    try:
        task = str(model.post_cfg.get("task", "")).lower()
        data_path = _ensure_dataset(args, task)

        if task == "image_classification":
            score = eval_imagenet(model=model, data_path=data_path, batch_size=args.batch_size)
            print(f"Validation score (Top-1 accuracy): {score:.5f}")
            return score

        if task in {"object_detection", "instance_segmentation", "pose_estimation"}:
            score = eval_coco(
                model=model,
                data_path=data_path,
                batch_size=args.batch_size,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
            )
            print(f"Validation score (mAP 50-95): {score:.5f}")
            return score

        if task == "face_detection":
            raise SystemExit(
                "Validation for face detection models is not available yet. "
                "WiderFace evaluation is still pending in the current codebase."
            )

        raise SystemExit(f"Unsupported vision task for validation: {task}")
    finally:
        model.dispose()


def _cmd_val(args: argparse.Namespace) -> int:
    """Runs vision validation on the task-appropriate benchmark dataset."""

    _run_validation(args)
    return 0


def add_val_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Registers the unified vision validation CLI command."""

    parser = subparsers.add_parser("val", help="Validate a vision model on its benchmark dataset.")
    parser.set_defaults(_handler=_cmd_val)
    parser.add_argument("--model", required=True, help="Vision model name, for example `resnet50` or `yolo11m`.")
    parser.add_argument(
        "--framework",
        default=None,
        choices=["mxq", "onnx"],
        help="Inference framework to use. When omitted, `--model-path` suffix is used first, then `mxq`.",
    )
    parser.add_argument(
        "--model-path",
        "--mxq-path",
        "--onnx-path",
        dest="model_path",
        default="",
        help="Optional local model path for MXQ or ONNX inference.",
    )
    parser.add_argument("--model-type", default="DEFAULT", help="Model variant from the YAML configuration.")
    parser.add_argument(
        "--core-mode",
        default="global8",
        choices=["single", "multi", "global4", "global8"],
        help="NPU core execution mode.",
    )
    parser.add_argument("--dev-no", type=int, default=0, help="NPU device number.")
    parser.add_argument(
        "--target-cores",
        type=parse_target_cores,
        help="Optional semicolon-separated core list for single-core mode, for example `0:0;0:1`.",
    )
    parser.add_argument(
        "--target-clusters",
        type=parse_target_clusters,
        help="Optional semicolon-separated cluster list for multi/global modes, for example `0;1`.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for validation.")
    parser.add_argument(
        "--data-path",
        help="Path to an already organized validation dataset. If omitted, the default cache path is used.",
    )
    parser.add_argument(
        "--force-organize",
        "--force",
        "--reload",
        action="store_true",
        dest="force_organize",
        help="Rebuild the organized dataset even when the target directory already looks ready.",
    )
    parser.add_argument(
        "--image-dir",
        help="Local archive path or download URL for the dataset images used by automatic organization.",
    )
    parser.add_argument(
        "--xml-dir",
        help="Local archive path or download URL for ImageNet annotations used by automatic organization.",
    )
    parser.add_argument(
        "--annotation-dir",
        help="Local archive path or download URL for dataset annotations used by automatic organization.",
    )
    add_threshold_args(parser, conf_default=None, iou_default=None)
