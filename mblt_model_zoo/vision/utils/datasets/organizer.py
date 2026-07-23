"""
Utilities for organizing datasets.
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from tempfile import TemporaryDirectory
from time import sleep
from typing import Protocol, TypeGuard
from urllib.parse import urlparse

import requests
from gdown.download import download
from gdown.download_folder import download_folder
from PIL import Image
from tqdm import tqdm

from ...datasets import get_dataset_config

DOWNLOAD_CHUNK_SIZE = 1 * 1024 * 1024
DOWNLOAD_RETRY_LIMIT = 4
DOWNLOAD_RETRY_BACKOFF_SECONDS = 2.0
DOWNLOAD_TIMEOUT = (10, 30)
DOTAV1_DOWNLOAD_CONFIG = get_dataset_config("dotav1")["download"]
DOTAV1_GOOGLE_DRIVE_ARCHIVES = {
    DOTAV1_DOWNLOAD_CONFIG["images_archive"],
    DOTAV1_DOWNLOAD_CONFIG["labels_archive"],
}
DOTAV1_CLASS_TO_IDX = {name: int(index) for index, name in get_dataset_config("dotav1")["names"].items()}
DOTAV1_VALIDATION_SAMPLE_COUNT = 458
NYU_DEPTH_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/nyu-depth.zip"
NYU_DEPTH_VALIDATION_SAMPLE_COUNT = 654


class _GoogleDriveDownloadEntry(Protocol):
    """The public attributes needed from a gdown folder-listing entry."""

    id: str
    path: str


def _is_google_drive_download_entry(value: object) -> TypeGuard[_GoogleDriveDownloadEntry]:
    """Returns whether a folder-listing value has the Google Drive file attributes needed here."""

    return isinstance(getattr(value, "id", None), str) and isinstance(getattr(value, "path", None), str)


def _is_url(path_or_url: str) -> bool:
    """Returns whether the given string looks like an HTTP(S) URL."""
    parsed = urlparse(path_or_url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _download_url(url: str, local_path: str) -> str:
    """Downloads a URL to a local file with progress and resume support.

    Args:
        url: HTTP(S) URL to download.
        local_path: Destination file path.

    Returns:
        The local destination path.

    Raises:
        RuntimeError: If all download attempts fail.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    for attempt in range(1, DOWNLOAD_RETRY_LIMIT + 1):
        existing_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
        headers: dict[str, str] = {}
        mode = "wb"
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"

        try:
            with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT, headers=headers) as response:
                response.raise_for_status()

                if response.status_code == 200 and existing_size > 0:
                    existing_size = 0
                    mode = "wb"

                total_size = response.headers.get("Content-Length")
                total_bytes = existing_size + int(total_size) if total_size is not None else None

                desc = f"Downloading {os.path.basename(local_path)}"
                with tqdm(
                    total=total_bytes,
                    initial=existing_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=desc,
                ) as pbar:
                    with open(local_path, mode) as file_obj:
                        for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                            if not chunk:
                                continue
                            file_obj.write(chunk)
                            pbar.update(len(chunk))
            return local_path
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
            if attempt == DOWNLOAD_RETRY_LIMIT:
                raise RuntimeError(f"Failed to download {url} after {DOWNLOAD_RETRY_LIMIT} attempts.") from exc
            resumed_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
            print(
                f"Download interrupted for {os.path.basename(local_path)}; "
                f"retrying from {resumed_size} bytes (attempt {attempt + 1}/{DOWNLOAD_RETRY_LIMIT})..."
            )
            sleep(DOWNLOAD_RETRY_BACKOFF_SECONDS * attempt)

    raise RuntimeError(f"Failed to download {url} after {DOWNLOAD_RETRY_LIMIT} attempts.")


def _should_download_serially(path_or_urls: list[str]) -> bool:
    """Returns whether URL inputs should be downloaded one by one.

    Dataset hosts such as ImageNet often throttle concurrent archive downloads
    from the same origin. Serializing same-host downloads is slower in the best
    case, but much more stable for the large validation archives used here.
    """

    hosts = [urlparse(path_or_url).netloc for path_or_url in path_or_urls if _is_url(path_or_url)]
    return len(hosts) > 1 and len(set(hosts)) == 1


def _download_if_url(path_or_url: str, download_dir: str) -> str:
    """Downloads a remote dataset archive when needed.

    Args:
        path_or_url: Local path or HTTP(S) URL pointing to a dataset archive.
        download_dir: Directory to store downloaded archives.

    Returns:
        A local filesystem path to the archive or directory.

    Raises:
        ValueError: If the URL path does not contain a filename.
    """
    if not _is_url(path_or_url):
        return path_or_url

    parsed = urlparse(path_or_url)
    filename = os.path.basename(parsed.path)
    if not filename:
        raise ValueError(f"Unable to determine a filename from URL: {path_or_url}")

    local_path = os.path.join(download_dir, filename)
    print(f"Downloading dataset archive from {path_or_url} to {local_path}...")
    _download_url(path_or_url, local_path)
    print("Download completed")
    return local_path


def _resolve_source(path_or_url: str, download_dir: str) -> str:
    """Resolves a local path for a dataset source."""

    return _download_if_url(path_or_url, download_dir)


def _resolve_sources(path_or_urls: list[str], download_dir: str) -> list[str]:
    """Resolves multiple dataset sources, downloading URL inputs in parallel."""

    if _should_download_serially(path_or_urls):
        return [_resolve_source(path_or_url, download_dir) for path_or_url in path_or_urls]

    local_paths: list[str | None] = [None] * len(path_or_urls)
    futures: dict[concurrent.futures.Future[str], int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(path_or_urls))) as executor:
        for idx, path_or_url in enumerate(path_or_urls):
            if _is_url(path_or_url):
                futures[executor.submit(_resolve_source, path_or_url, download_dir)] = idx
            else:
                local_paths[idx] = path_or_url

        for future in concurrent.futures.as_completed(futures):
            local_paths[futures[future]] = future.result()

    return [path for path in local_paths if path is not None]


def _get_object_name(obj: ET.Element, xml_file: str) -> str:
    """Extracts a non-empty object name from an ImageNet annotation node.

    Args:
        obj: XML ``object`` element from an annotation file.
        xml_file: Source XML filename used for error context.

    Returns:
        The validated object name.

    Raises:
        ValueError: If the object name node is missing or empty.
    """
    name_element = obj.find("name")
    if name_element is None or name_element.text is None:
        raise ValueError(f"XML file {xml_file} has an object without a valid name")

    object_name = name_element.text.strip()
    if not object_name:
        raise ValueError(f"XML file {xml_file} has an object with an empty name")

    return object_name


def construct_imagenet(image_dir: str, xml_dir: str, output_dir: str) -> None:
    """Constructs the ImageNet dataset by organizing images into category folders.

    Args:
        image_dir (str): Directory containing the ImageNet validation images.
        xml_dir (str): Directory containing the ImageNet bounding box XML files.
        output_dir (str): Directory where the organized dataset will be stored.

    Raises:
        ValueError: If an XML file has no objects or contains multiple object names.
        AssertionError: If the number of XML files and images do not match.
    """

    assert len(os.listdir(xml_dir + "/val")) == len(os.listdir(image_dir)), (
        f"Number of XML and image files do not match: "
        f"{len(os.listdir(xml_dir + '/val'))} != {len(os.listdir(image_dir))}"
    )

    # validate the XML files
    pbar = tqdm(os.listdir(xml_dir + "/val"), desc="Validating XML files")
    for xml_file in pbar:
        xml_path = os.path.join(xml_dir + "/val", xml_file)
        xml_tree = ET.parse(xml_path)
        root = xml_tree.getroot()

        if len(root.findall("object")) < 1:
            raise ValueError(f"XML file {xml_file} has no object, but expected at least 1")

        # check whether the object names in the XML files are the same
        object_names = [_get_object_name(obj, xml_file) for obj in root.findall("object")]
        if len(set(object_names)) != 1:
            raise ValueError(
                f"Object names in XML file {xml_file} are not the same. "
                f"It has {len(set(object_names))} different object names."
            )

    pbar.close()

    # construct the ImageNet dataset
    pbar = tqdm(os.listdir(xml_dir + "/val"), desc="Constructing ImageNet dataset")
    for xml_file in pbar:
        xml_path = os.path.join(xml_dir + "/val", xml_file)
        xml_tree = ET.parse(xml_path)
        root = xml_tree.getroot()
        object_name = _get_object_name(root.findall("object")[0], xml_file)
        image_path = os.path.join(image_dir, xml_file.replace(".xml", ".JPEG"))
        assert os.path.exists(image_path), f"Image file not found: {image_path}"

        os.makedirs(os.path.join(output_dir, object_name), exist_ok=True)  # create the directory for the object
        shutil.copy(
            image_path,
            os.path.join(output_dir, object_name, os.path.basename(image_path)),
        )  # copy the image to the directory with the same name
    pbar.close()

    # validate the ImageNet dataset
    pbar = tqdm(os.listdir(output_dir), desc="Validating ImageNet dataset")
    print(f"Number of categories: {len(os.listdir(output_dir))}")
    for object_name in pbar:
        num_images = len(os.listdir(os.path.join(output_dir, object_name)))
        if num_images != 50:
            raise ValueError(f"Object {object_name} has {num_images} images, but expected 50")
    pbar.close()
    print("Each category has 50 images")
    print("ImageNet dataset constructed successfully")


def organize_imagenet(
    image_dir: str,
    xml_dir: str,
    output_dir: str = os.path.expanduser("~/.mblt_model_zoo/datasets/imagenet"),
) -> None:
    """Organizes the ImageNet dataset, unpacking archives if necessary.

    Args:
        image_dir (str): Path or URL to the image directory or archive (.tar).
        xml_dir (str): Path or URL to the XML directory or archive (.tgz).
        output_dir (str, optional): Directory to store the organized dataset.
            Defaults to ~/.mblt_model_zoo/datasets/imagenet.
    """
    with TemporaryDirectory() as temp_dir:
        local_image_dir, local_xml_dir = _resolve_sources([image_dir, xml_dir], temp_dir)

        if local_image_dir.endswith(".tar") and local_xml_dir.endswith(".tgz"):
            print("Unpacking image and XML files to temporary directory...")
            shutil.unpack_archive(local_image_dir, os.path.join(temp_dir, "ILSVRC2012_img_val"))
            shutil.unpack_archive(local_xml_dir, os.path.join(temp_dir, "ILSVRC2012_bbox_val_v3"))
            print("Unpacking completed")
            construct_imagenet(
                os.path.join(temp_dir, "ILSVRC2012_img_val"),
                os.path.join(temp_dir, "ILSVRC2012_bbox_val_v3"),
                output_dir,
            )
            return

        construct_imagenet(local_image_dir, local_xml_dir, output_dir)


def construct_coco(image_dir: str, annotation_dir: str, output_dir: str) -> None:
    """Constructs the COCO dataset by copying images and annotations to a target directory.

    Args:
        image_dir (str): Directory containing COCO images.
        annotation_dir (str): Directory containing COCO annotations.
        output_dir (str): Directory where the organized dataset will be stored.
    """
    print(f"Constructing COCO dataset from {image_dir} and {annotation_dir} to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(
        image_dir, os.path.join(output_dir, "val2017"), dirs_exist_ok=True
    )  # copy the image directory to the output directory
    # copy *_val2017.json to the output directory
    for file in os.listdir(os.path.join(annotation_dir, "annotations")):
        if file.endswith("_val2017.json"):
            shutil.copy(
                os.path.join(annotation_dir, "annotations", file),
                os.path.join(output_dir, file),
            )
    print("Constructing COCO dataset completed")


def organize_coco(
    image_dir: str,
    annotation_dir: str,
    output_dir: str = os.path.expanduser("~/.mblt_model_zoo/datasets/coco"),
) -> None:
    """Organizes the COCO dataset, unpacking archives if necessary.

    Args:
        image_dir (str): Path or URL to the image zip file or directory.
        annotation_dir (str): Path or URL to the annotation zip file or directory.
        output_dir (str, optional): Directory to store the organized dataset.
            Defaults to ~/.mblt_model_zoo/datasets/coco.
    """
    with TemporaryDirectory() as temp_dir:
        local_image_dir, local_annotation_dir = _resolve_sources([image_dir, annotation_dir], temp_dir)

        if local_image_dir.endswith(".zip") and local_annotation_dir.endswith(".zip"):
            print("Unpacking image and annotation files to temporary directory...")
            shutil.unpack_archive(local_image_dir, temp_dir)
            shutil.unpack_archive(local_annotation_dir, os.path.join(temp_dir, "annotations_trainval2017"))
            print("Unpacking completed")
            construct_coco(
                os.path.join(temp_dir, "val2017"),
                os.path.join(temp_dir, "annotations_trainval2017"),
                output_dir,
            )
            return

        construct_coco(local_image_dir, local_annotation_dir, output_dir)


def construct_widerface(image_dir: str, annotation_dir: str, output_dir: str) -> None:
    """Constructs the WiderFace dataset by copying images and annotations to a target directory.

    Args:
        image_dir (str): Directory containing WiderFace images.
        annotation_dir (str): Directory containing WiderFace annotations.
        output_dir (str): Directory where the organized dataset will be stored.
    """
    print(f"Constructing WiderFace dataset from {image_dir} and {annotation_dir} to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(
        os.path.join(image_dir, "images"),
        os.path.join(output_dir, "images"),
        dirs_exist_ok=True,
    )
    for file in os.listdir(annotation_dir):
        if "_val" in file:
            shutil.copy(os.path.join(annotation_dir, file), output_dir)
    print("Constructing WiderFace dataset completed")


def organize_widerface(
    image_dir: str,
    annotation_dir: str,
    output_dir: str = os.path.expanduser("~/.mblt_model_zoo/datasets/widerface"),
) -> None:
    """Organizes the WiderFace dataset, unpacking archives if necessary.

    Args:
        image_dir (str): Path or URL to the image zip file or directory.
        annotation_dir (str): Path or URL to the annotation zip file or directory.
        output_dir (str, optional): Directory to store the organized dataset.
            Defaults to ~/.mblt_model_zoo/datasets/widerface.
    """
    with TemporaryDirectory() as temp_dir:
        local_image_dir, local_annotation_dir = _resolve_sources([image_dir, annotation_dir], temp_dir)

        if local_image_dir.endswith(".zip") and local_annotation_dir.endswith(".zip"):
            print("Unpacking image and annotation files to temporary directory...")
            shutil.unpack_archive(local_image_dir, temp_dir)
            shutil.unpack_archive(local_annotation_dir, temp_dir)
            print("Unpacking completed")
            construct_widerface(
                os.path.join(temp_dir, "WIDER_val"),
                os.path.join(temp_dir, "wider_face_split"),
                output_dir,
            )
            return

        construct_widerface(local_image_dir, local_annotation_dir, output_dir)


def _resolve_nyu_depth_validation_dirs(dataset_dir: str) -> tuple[str, str]:
    """Resolves NYU Depth validation image and depth directories.

    Args:
        dataset_dir: Directory containing the NYU Depth root or its parent.

    Returns:
        Paths to the validation image and depth directories.

    Raises:
        ValueError: If the expected NYU Depth layout is not present.
    """

    roots = (os.path.join(dataset_dir, "nyu-depth"), dataset_dir)
    for root in roots:
        candidates = (
            (os.path.join(root, "images", "val"), os.path.join(root, "depth", "val")),
            (os.path.join(root, "val", "images"), os.path.join(root, "val", "depth")),
            (os.path.join(root, "images"), os.path.join(root, "depth")),
        )
        for image_dir, depth_dir in candidates:
            if os.path.isdir(image_dir) and os.path.isdir(depth_dir):
                return image_dir, depth_dir
    raise ValueError(f"NYU Depth dataset must contain matching images/ and depth/ directories: {dataset_dir}")


def _collect_nyu_depth_validation_files(image_dir: str, depth_dir: str) -> tuple[dict[str, str], dict[str, str]]:
    """Validates and returns matching NYU Depth validation image/depth pairs."""

    images = {
        os.path.splitext(os.path.basename(path))[0]: path for path in _iter_files(image_dir, [".jpg", ".jpeg", ".png"])
    }
    depths = {os.path.splitext(os.path.basename(path))[0]: path for path in _iter_files(depth_dir, [".npy"])}
    missing_depths = sorted(set(images) - set(depths))
    missing_images = sorted(set(depths) - set(images))
    if missing_depths or missing_images:
        details = []
        if missing_depths:
            details.append(f"images without depth maps: {', '.join(missing_depths[:5])}")
        if missing_images:
            details.append(f"depth maps without images: {', '.join(missing_images[:5])}")
        raise ValueError(f"NYU Depth validation image/depth mismatch ({'; '.join(details)}).")
    if len(images) != NYU_DEPTH_VALIDATION_SAMPLE_COUNT:
        raise ValueError(
            "NYU Depth validation dataset must contain "
            f"{NYU_DEPTH_VALIDATION_SAMPLE_COUNT} matching image/depth pairs, found {len(images)}."
        )
    return images, depths


def construct_nyu_depth(dataset_dir: str, output_dir: str) -> None:
    """Constructs the NYU Depth layout from an extracted dataset directory.

    Args:
        dataset_dir: Directory containing the NYU Depth root or its parent.
        output_dir: Directory where the organized dataset will be stored.
    """

    image_dir, depth_dir = _resolve_nyu_depth_validation_dirs(dataset_dir)
    images, depths = _collect_nyu_depth_validation_files(image_dir, depth_dir)
    print(f"Constructing NYU Depth validation dataset from {dataset_dir} to {output_dir}")

    output_dir = os.path.abspath(output_dir)
    output_parent_dir = os.path.dirname(output_dir)
    os.makedirs(output_parent_dir, exist_ok=True)
    with TemporaryDirectory(dir=output_parent_dir, prefix=".nyu-depth-staging-") as staging_dir:
        staged_image_dir = os.path.join(staging_dir, "images")
        staged_depth_dir = os.path.join(staging_dir, "depth")
        os.makedirs(staged_image_dir)
        os.makedirs(staged_depth_dir)
        for sample_id in sorted(images):
            shutil.copy2(images[sample_id], os.path.join(staged_image_dir, os.path.basename(images[sample_id])))
            shutil.copy2(depths[sample_id], os.path.join(staged_depth_dir, os.path.basename(depths[sample_id])))

        replacements = (
            (staged_image_dir, os.path.join(output_dir, "images")),
            (staged_depth_dir, os.path.join(output_dir, "depth")),
        )
        with TemporaryDirectory(dir=output_parent_dir, prefix=".nyu-depth-backup-") as backup_dir:
            backups: dict[str, str] = {}
            installed_dirs: list[str] = []
            try:
                for _, destination_dir in replacements:
                    if os.path.lexists(destination_dir):
                        backup_path = os.path.join(backup_dir, os.path.basename(destination_dir))
                        os.replace(destination_dir, backup_path)
                        backups[destination_dir] = backup_path

                for staged_dir, destination_dir in replacements:
                    os.makedirs(os.path.dirname(destination_dir), exist_ok=True)
                    os.replace(staged_dir, destination_dir)
                    installed_dirs.append(destination_dir)
            except OSError:
                for directory in installed_dirs:
                    if os.path.isdir(directory) and not os.path.islink(directory):
                        shutil.rmtree(directory)
                    elif os.path.lexists(directory):
                        os.remove(directory)
                for directory, backup_path in backups.items():
                    os.makedirs(os.path.dirname(directory), exist_ok=True)
                    os.replace(backup_path, directory)
                raise
    print(f"Constructed NYU Depth validation dataset with {len(images)} image/depth pairs")


def organize_nyu_depth(
    dataset_path: str = NYU_DEPTH_URL,
    output_dir: str = os.path.expanduser("~/.mblt_model_zoo/datasets/nyu-depth"),
) -> None:
    """Organizes NYU Depth, downloading and unpacking an archive when necessary.

    Args:
        dataset_path: Path or URL to the NYU Depth zip file or extracted dataset directory.
        output_dir: Directory to store the organized dataset.
    """

    with TemporaryDirectory() as temp_dir:
        local_dataset_path = _resolve_source(dataset_path, temp_dir)
        if local_dataset_path.endswith(".zip"):
            print("Unpacking NYU Depth files to temporary directory...")
            shutil.unpack_archive(local_dataset_path, temp_dir)
            print("Unpacking completed")
            construct_nyu_depth(temp_dir, output_dir)
            return

        construct_nyu_depth(local_dataset_path, output_dir)


def _resolve_dotav1_root(dataset_dir: str) -> str:
    """Resolves a DOTAv1 dataset root from a directory path.

    Args:
        dataset_dir: Directory containing the DOTAv1 dataset or its parent.

    Returns:
        Path to the DOTAv1 dataset root.
    """
    dotav1_dir = os.path.join(dataset_dir, "DOTAv1")
    if os.path.isdir(dotav1_dir):
        return dotav1_dir
    return dataset_dir


def _is_google_drive_folder_url(path_or_url: str) -> bool:
    """Returns whether a URL points to a Google Drive folder."""

    parsed = urlparse(path_or_url)
    return parsed.hostname == "drive.google.com" and bool(
        re.fullmatch(r"/drive(?:/u/[^/]+)?/folders/[^/]+/?", parsed.path)
    )


def _download_dotav1_google_drive_archives(folder_url: str, download_dir: str) -> tuple[str, str]:
    """Downloads the DOTAv1 image and v1.0-label archives from a Google Drive folder.

    Args:
        folder_url: Public Google Drive folder URL containing the DOTAv1 archives.
        download_dir: Directory where the selected archives will be stored.

    Returns:
        Paths to the image archive and original v1.0-label archive.

    Raises:
        ValueError: If the required archives are absent from the Drive folder.
        RuntimeError: If gdown fails to download a required archive.
    """

    print(f"Retrieving DOTAv1 archive list from {folder_url}...")
    folder_entries = download_folder(url=folder_url, output=download_dir, quiet=True, skip_download=True)
    if folder_entries is None:
        raise RuntimeError(f"Failed to retrieve the DOTAv1 Google Drive folder listing: {folder_url}")
    files = [entry for entry in folder_entries if _is_google_drive_download_entry(entry)]
    archives: dict[str, _GoogleDriveDownloadEntry] = {}
    for archive_path in DOTAV1_GOOGLE_DRIVE_ARCHIVES:
        matches = [
            drive_file
            for drive_file in files
            if drive_file.path == archive_path or drive_file.path.endswith(f"/{archive_path}")
        ]
        if len(matches) == 1:
            archives[archive_path] = matches[0]
            continue

        available = ", ".join(sorted(drive_file.path for drive_file in files)) or "none"
        if not matches:
            raise ValueError(f"DOTAv1 Drive folder is missing {archive_path}. Available files: {available}.")
        ambiguous = ", ".join(sorted(drive_file.path for drive_file in matches))
        raise ValueError(f"DOTAv1 Drive folder has ambiguous matches for {archive_path}: {ambiguous}.")

    local_archives: dict[str, str] = {}
    for archive_path in sorted(DOTAV1_GOOGLE_DRIVE_ARCHIVES):
        drive_file = archives[archive_path]
        local_path = os.path.join(download_dir, os.path.basename(archive_path))
        print(f"Downloading DOTAv1 {archive_path}...")
        downloaded_path = download(id=drive_file.id, output=local_path, quiet=False, resume=True)
        if not isinstance(downloaded_path, str):
            raise RuntimeError(f"Failed to download DOTAv1 archive {archive_path} from {folder_url}.")
        local_archives[archive_path] = downloaded_path

    return (
        local_archives[DOTAV1_DOWNLOAD_CONFIG["images_archive"]],
        local_archives[DOTAV1_DOWNLOAD_CONFIG["labels_archive"]],
    )


def _iter_files(root: str, extensions: Iterable[str]) -> Iterable[str]:
    """Yields files below a directory with one of the requested suffixes."""

    suffixes = tuple(extension.lower() for extension in extensions)
    for current_root, _, file_names in os.walk(root):
        for file_name in file_names:
            if file_name.lower().endswith(suffixes):
                yield os.path.join(current_root, file_name)


def _write_dotav1_yolo_labels(image_path: str, original_label_path: str, output_path: str) -> None:
    """Converts one official DOTAv1 label file into normalized OBB label format."""

    with Image.open(image_path) as image:
        width, height = image.size
    converted_lines: list[str] = []
    with open(original_label_path, encoding="utf-8") as label_file:
        for line in label_file:
            fields = line.split()
            if len(fields) < 10 or fields[0].startswith("imagesource:") or fields[0].startswith("gsd:"):
                continue
            class_name = fields[8]
            if class_name not in DOTAV1_CLASS_TO_IDX:
                raise ValueError(f"Unsupported DOTAv1 class in {original_label_path}: {class_name}")
            coordinates = [float(value) for value in fields[:8]]
            normalized = [
                coordinate / (width if index % 2 == 0 else height) for index, coordinate in enumerate(coordinates)
            ]
            converted_lines.append(
                f"{DOTAV1_CLASS_TO_IDX[class_name]} " + " ".join(f"{coordinate:.8g}" for coordinate in normalized)
            )
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(converted_lines))
        if converted_lines:
            output_file.write("\n")


def construct_dotav1_from_archives(image_archive: str, label_archive: str, output_dir: str) -> None:
    """Constructs the legacy DOTAv1 validation layout from the Google Drive archives.

    Args:
        image_archive: Path to the DOTAv1 validation-image archive.
        label_archive: Path to the original DOTAv1 v1.0 label archive.
        output_dir: Directory where the organized validation dataset will be stored.

    Raises:
        ValueError: If the archives have no validation files or their image and label stems differ.
        OSError: If staging or replacing the organized dataset files fails.
    """

    with TemporaryDirectory() as extract_dir:
        image_dir = os.path.join(extract_dir, "images")
        label_dir = os.path.join(extract_dir, "labels")
        shutil.unpack_archive(image_archive, image_dir)
        shutil.unpack_archive(label_archive, label_dir)

        labels = {os.path.splitext(os.path.basename(path))[0]: path for path in _iter_files(label_dir, [".txt"])}
        if not labels:
            raise ValueError(f"No DOTAv1 label files found in {label_archive}.")

        images = {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in _iter_files(image_dir, [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"])
        }
        image_ids = set(images)
        label_ids = set(labels)
        missing_labels = sorted(image_ids - label_ids)
        missing_images = sorted(label_ids - image_ids)
        if missing_labels or missing_images:
            details = []
            if missing_labels:
                details.append(f"images without labels: {', '.join(missing_labels[:5])}")
            if missing_images:
                details.append(f"labels without images: {', '.join(missing_images[:5])}")
            raise ValueError(f"DOTAv1 archive stem mismatch ({'; '.join(details)}).")
        matching_ids = sorted(image_ids)
        if len(matching_ids) != DOTAV1_VALIDATION_SAMPLE_COUNT:
            raise ValueError(
                "DOTAv1 validation dataset must contain "
                f"{DOTAV1_VALIDATION_SAMPLE_COUNT} matching image/label pairs, found {len(matching_ids)}."
            )

        output_dir = os.path.abspath(output_dir)
        output_parent_dir = os.path.dirname(output_dir)
        os.makedirs(output_parent_dir, exist_ok=True)
        with TemporaryDirectory(dir=output_parent_dir, prefix=".dotav1-staging-") as staging_dir:
            staged_image_dir = os.path.join(staging_dir, "images", "val")
            staged_label_dir = os.path.join(staging_dir, "labels", "val")
            staged_original_label_dir = os.path.join(staging_dir, "labels", "val_original")
            os.makedirs(staged_image_dir)
            os.makedirs(staged_label_dir)
            os.makedirs(staged_original_label_dir)

            for image_id in matching_ids:
                image_path = images[image_id]
                shutil.copy2(image_path, os.path.join(staged_image_dir, os.path.basename(image_path)))

            for image_id in matching_ids:
                shutil.copy2(labels[image_id], os.path.join(staged_original_label_dir, f"{image_id}.txt"))
                _write_dotav1_yolo_labels(
                    images[image_id], labels[image_id], os.path.join(staged_label_dir, f"{image_id}.txt")
                )

            image_output_dir = os.path.join(output_dir, "images")
            label_output_dir = os.path.join(output_dir, "labels", "val")
            original_label_output_dir = os.path.join(output_dir, "labels", "val_original")
            replacements = (
                (staged_image_dir, image_output_dir),
                (staged_label_dir, label_output_dir),
                (staged_original_label_dir, original_label_output_dir),
            )
            existing_dirs = (image_output_dir, label_output_dir, original_label_output_dir)
            with TemporaryDirectory(dir=output_parent_dir, prefix=".dotav1-backup-") as backup_dir:
                backups: dict[str, str] = {}
                installed_dirs: list[str] = []
                try:
                    for directory in existing_dirs:
                        if os.path.lexists(directory):
                            backup_path = os.path.join(backup_dir, os.path.relpath(directory, output_dir))
                            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                            os.replace(directory, backup_path)
                            backups[directory] = backup_path

                    for staged_dir, output_subdir in replacements:
                        os.makedirs(os.path.dirname(output_subdir), exist_ok=True)
                        os.replace(staged_dir, output_subdir)
                        installed_dirs.append(output_subdir)
                except OSError:
                    for directory in installed_dirs:
                        if os.path.isdir(directory) and not os.path.islink(directory):
                            shutil.rmtree(directory)
                        elif os.path.lexists(directory):
                            os.remove(directory)
                    for directory, backup_path in backups.items():
                        os.makedirs(os.path.dirname(directory), exist_ok=True)
                        os.replace(backup_path, directory)
                    raise

    print(f"Constructed DOTAv1 validation dataset with {len(matching_ids)} images")


def construct_dotav1(dataset_dir: str, output_dir: str) -> None:
    """Constructs a validation-only DOTAv1 dataset.

    Args:
        dataset_dir: Directory containing a DOTAv1 dataset or its parent.
        output_dir: Directory where the organized validation dataset will be stored.

    Raises:
        ValueError: If no validation files or directories are found.
    """
    dataset_root = _resolve_dotav1_root(dataset_dir)
    print(f"Constructing DOTAv1 validation dataset from {dataset_root} to {output_dir}")
    flat_image_dir = os.path.join(dataset_root, "images")
    flat_label_dir = os.path.join(dataset_root, "labels")
    has_flat_images = os.path.isdir(flat_image_dir) and any(
        file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        for file_name in os.listdir(flat_image_dir)
    )
    if has_flat_images and os.path.isdir(os.path.join(flat_label_dir, "val_original")):
        os.makedirs(output_dir, exist_ok=True)
        if os.path.abspath(flat_image_dir) != os.path.abspath(os.path.join(output_dir, "images")):
            shutil.copytree(flat_image_dir, os.path.join(output_dir, "images"), dirs_exist_ok=True)
        for label_directory in ("val", "val_original"):
            source_dir = os.path.join(flat_label_dir, label_directory)
            if os.path.isdir(source_dir):
                shutil.copytree(source_dir, os.path.join(output_dir, "labels", label_directory), dirs_exist_ok=True)
        return
    os.makedirs(output_dir, exist_ok=True)

    copied_count = 0
    for root, dirs, files in os.walk(dataset_root):
        for dir_name in list(dirs):
            if "val" not in dir_name:
                continue

            src_dir = os.path.join(root, dir_name)
            relative_dir = os.path.relpath(src_dir, dataset_root)
            if relative_dir == os.path.join("images", "val"):
                dst_dir = os.path.join(output_dir, "images")
            else:
                dst_dir = os.path.join(output_dir, relative_dir)
            if os.path.abspath(src_dir) != os.path.abspath(dst_dir):
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            copied_count += 1
            dirs.remove(dir_name)

        for file_name in files:
            if "val" not in file_name:
                continue

            src_file = os.path.join(root, file_name)
            relative_file = os.path.relpath(src_file, dataset_root)
            dst_file = os.path.join(output_dir, relative_file)
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            if os.path.abspath(src_file) != os.path.abspath(dst_file):
                shutil.copy2(src_file, dst_file)
            copied_count += 1

    if copied_count == 0:
        raise ValueError(f"No DOTAv1 validation files or directories found in {dataset_root}")

    print("Constructing DOTAv1 validation dataset completed")


def organize_dotav1(
    dataset_path: str,
    output_dir: str = os.path.expanduser("~/.mblt_model_zoo/datasets/dotav1"),
) -> None:
    """Organizes a validation-only DOTAv1 dataset.

    Args:
        dataset_path: Path or URL to the DOTAv1 zip file or extracted dataset directory.
        output_dir: Directory to store the organized dataset.
    """
    with TemporaryDirectory() as temp_dir:
        if _is_google_drive_folder_url(dataset_path):
            image_archive, label_archive = _download_dotav1_google_drive_archives(dataset_path, temp_dir)
            construct_dotav1_from_archives(image_archive, label_archive, output_dir)
            return

        local_dataset_path = _resolve_source(dataset_path, temp_dir)

        if local_dataset_path.endswith(".zip"):
            print("Unpacking DOTAv1 files to temporary directory...")
            shutil.unpack_archive(local_dataset_path, temp_dir)
            print("Unpacking completed")
            construct_dotav1(temp_dir, output_dir)
            return

        construct_dotav1(local_dataset_path, output_dir)
