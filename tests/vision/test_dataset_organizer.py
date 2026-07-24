"""Tests for dataset download organization helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import ZipFile

import numpy as np
import pytest
import requests
from PIL import Image

import mblt_model_zoo.vision.utils.datasets.organizer as organizer


class _DummyTqdm:
    """Minimal tqdm stub for download tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.updated = 0

    def __enter__(self) -> _DummyTqdm:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def update(self, value: int) -> None:
        self.updated += value


class _FakeResponse:
    """Simple streaming response test double."""

    def __init__(self, status_code: int, headers: dict[str, str], chunks: list[bytes | Exception]) -> None:
        self.status_code = status_code
        self.headers = headers
        self._chunks = chunks

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Any:
        del chunk_size
        for chunk in self._chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


def test_download_url_retries_and_resumes_partial_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resume a partial archive download after a transient connection failure."""

    first_chunk = b"abc"
    second_chunk = b"def"
    calls: list[dict[str, str]] = []
    responses = [
        _FakeResponse(
            status_code=200,
            headers={"Content-Length": str(len(first_chunk) + len(second_chunk))},
            chunks=[first_chunk, requests.ConnectionError("interrupted")],
        ),
        _FakeResponse(
            status_code=206,
            headers={"Content-Length": str(len(second_chunk))},
            chunks=[second_chunk],
        ),
    ]

    def _fake_get(url: str, stream: bool, timeout: tuple[int, int], headers: dict[str, str]) -> _FakeResponse:
        del url, stream, timeout
        calls.append(dict(headers))
        return responses.pop(0)

    monkeypatch.setattr(organizer.requests, "get", _fake_get)
    monkeypatch.setattr(organizer, "tqdm", _DummyTqdm)
    monkeypatch.setattr(organizer, "sleep", lambda _: None)

    local_path = tmp_path / "archive.tar"
    result = organizer._download_url("https://example.com/archive.tar", str(local_path))

    assert result == str(local_path)
    assert local_path.read_bytes() == first_chunk + second_chunk
    assert calls == [{}, {"Range": "bytes=3-"}]


def test_should_download_serially_for_same_host_urls() -> None:
    """Serialize same-host dataset archive downloads to avoid throttling."""

    assert organizer._should_download_serially(
        [
            "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
            "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz",
        ]
    )

    assert not organizer._should_download_serially(
        [
            "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
            "https://example.com/data/annotations.tgz",
        ]
    )


def test_organize_nyu_depth_extracts_only_validation_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Install only NYU Depth validation image/depth pairs from an archive."""

    monkeypatch.setattr(organizer, "NYU_DEPTH_VALIDATION_SAMPLE_COUNT", 1)
    archive_path = tmp_path / "nyu-depth.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("nyu-depth/images/train/nyu_train.jpg", b"training image")
        archive.writestr("nyu-depth/depth/train/nyu_train.npy", b"training depth")
        archive.writestr("nyu-depth/images/val/nyu_0000.jpg", b"validation image")
        archive.writestr("nyu-depth/depth/val/nyu_0000.npy", b"validation depth")

    output_dir = tmp_path / "organized"
    organizer.organize_nyu_depth(str(archive_path), str(output_dir))

    assert archive_path.is_file()
    assert (output_dir / "images" / "nyu_0000.jpg").read_bytes() == b"validation image"
    assert (output_dir / "depth" / "nyu_0000.npy").read_bytes() == b"validation depth"
    assert not (output_dir / "images" / "train").exists()
    assert not (output_dir / "depth" / "train").exists()


def test_organize_ade20k_extracts_flat_validation_layout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Install only ADE20K validation image/mask pairs in the reference layout."""

    monkeypatch.setattr(organizer, "ADE20K_VALIDATION_SAMPLE_COUNT", 1)
    archive_path = tmp_path / "ADEChallengeData2016.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("ADEChallengeData2016/images/training/ADE_train_00000001.jpg", b"training")
        archive.writestr("ADEChallengeData2016/annotations/training/ADE_train_00000001.png", b"training")
        archive.writestr("ADEChallengeData2016/images/validation/ADE_val_00000001.jpg", b"validation")
        archive.writestr("ADEChallengeData2016/annotations/validation/ADE_val_00000001.png", b"validation")
        archive.writestr("ADEChallengeData2016/objectInfo150.txt", b"labels")

    output_dir = tmp_path / "organized"
    organizer.organize_ade20k(str(archive_path), str(output_dir))

    assert (output_dir / "images" / "ADE_val_00000001.jpg").read_bytes() == b"validation"
    assert (output_dir / "annotations" / "ADE_val_00000001.png").read_bytes() == b"validation"
    assert (output_dir / "objectInfo150.txt").read_bytes() == b"labels"
    assert not (output_dir / "images" / "training").exists()


def test_cityscapes_loader_requests_only_validation_parquet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select explicit validation shards without requesting train or test data."""

    calls: dict[str, object] = {}

    def _fake_load_dataset(repo_id: str, **kwargs: object) -> list[object]:
        calls["repo_id"] = repo_id
        calls.update(kwargs)
        return []

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        SimpleNamespace(load_dataset=_fake_load_dataset),
    )
    organizer._load_cityscapes_validation()

    assert calls == {
        "repo_id": "Chris1/cityscapes",
        "data_files": {"validation": "data/validation-*.parquet"},
        "split": "validation",
    }


def test_organize_cityscapes_materializes_lossless_validation_pairs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Replace an existing layout with exactly the validation image/mask pairs."""

    monkeypatch.setattr(organizer, "CITYSCAPES_VALIDATION_SAMPLE_COUNT", 2)
    rows = [
        {
            "image": Image.fromarray(np.full((2, 3, 3), index * 50, dtype=np.uint8)),
            "semantic_segmentation": Image.fromarray(
                np.array([[7, 8, 0], [11, 12, 33]], dtype=np.uint8),
            ),
        }
        for index in range(2)
    ]
    monkeypatch.setattr(organizer, "_load_cityscapes_validation", lambda: rows)
    output_dir = tmp_path / "cityscapes"
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "annotations").mkdir()
    (output_dir / "images" / "stale.png").write_bytes(b"stale")
    (output_dir / "annotations" / "stale.png").write_bytes(b"stale")

    organizer.organize_cityscapes(str(output_dir))

    image_paths = sorted((output_dir / "images").glob("*.png"))
    annotation_paths = sorted((output_dir / "annotations").glob("*.png"))
    assert [path.name for path in image_paths] == [
        "cityscapes_val_000000.png",
        "cityscapes_val_000001.png",
    ]
    assert [path.name for path in annotation_paths] == [
        "cityscapes_val_000000.png",
        "cityscapes_val_000001.png",
    ]
    assert np.array_equal(np.asarray(Image.open(annotation_paths[0])), np.asarray(rows[0]["semantic_segmentation"]))


def test_organize_cityscapes_enforces_validation_pair_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject incomplete validation sources before replacing existing data."""

    monkeypatch.setattr(organizer, "CITYSCAPES_VALIDATION_SAMPLE_COUNT", 2)
    monkeypatch.setattr(organizer, "_load_cityscapes_validation", lambda: [])
    output_dir = tmp_path / "cityscapes"
    (output_dir / "images").mkdir(parents=True)
    marker = output_dir / "images" / "keep.png"
    marker.write_bytes(b"keep")

    with pytest.raises(ValueError, match="must contain 2 pairs"):
        organizer.organize_cityscapes(str(output_dir))

    assert marker.read_bytes() == b"keep"


def test_organize_cityscapes_rolls_back_failed_atomic_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Restore both previous directories if the staged installation fails."""

    monkeypatch.setattr(organizer, "CITYSCAPES_VALIDATION_SAMPLE_COUNT", 1)
    rows = [
        {
            "image": Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)),
            "semantic_segmentation": Image.fromarray(np.array([[7]], dtype=np.uint8)),
        }
    ]
    monkeypatch.setattr(organizer, "_load_cityscapes_validation", lambda: rows)
    output_dir = tmp_path / "cityscapes"
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "annotations").mkdir()
    (output_dir / "images" / "keep.png").write_bytes(b"old image")
    (output_dir / "annotations" / "keep.png").write_bytes(b"old annotation")
    real_replace = organizer.os.replace
    failed = False

    def _fail_annotation_install(source: str, destination: str) -> None:
        nonlocal failed
        if not failed and Path(source).name == "annotations" and Path(destination) == output_dir / "annotations":
            failed = True
            raise OSError("simulated install failure")
        real_replace(source, destination)

    monkeypatch.setattr(organizer.os, "replace", _fail_annotation_install)

    with pytest.raises(OSError, match="simulated"):
        organizer.organize_cityscapes(str(output_dir))

    assert (output_dir / "images" / "keep.png").read_bytes() == b"old image"
    assert (output_dir / "annotations" / "keep.png").read_bytes() == b"old annotation"
