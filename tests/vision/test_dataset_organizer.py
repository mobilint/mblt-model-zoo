"""Tests for dataset download organization helpers."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pytest
import requests

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


def test_safe_unpack_archive_preserves_regular_tar_layout(tmp_path: Path) -> None:
    """Extract regular files and directories from a supported tar archive."""

    archive_path = tmp_path / "dataset.tar"
    with tarfile.open(archive_path, "w") as archive:
        directory = tarfile.TarInfo("dataset/")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        payload = b"dataset contents"
        member = tarfile.TarInfo("dataset/sample.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    extract_dir = tmp_path / "extracted"
    organizer._safe_unpack_archive(str(archive_path), str(extract_dir))

    assert (extract_dir / "dataset" / "sample.txt").read_bytes() == b"dataset contents"


def test_safe_unpack_archive_rejects_tar_traversal_before_writing(tmp_path: Path) -> None:
    """Reject a traversal member without extracting earlier valid members."""

    archive_path = tmp_path / "dataset.tar"
    outside_path = tmp_path / "outside.txt"
    with tarfile.open(archive_path, "w") as archive:
        safe_payload = b"safe"
        safe_member = tarfile.TarInfo("dataset/safe.txt")
        safe_member.size = len(safe_payload)
        archive.addfile(safe_member, io.BytesIO(safe_payload))
        outside_payload = b"outside"
        outside_member = tarfile.TarInfo("../outside.txt")
        outside_member.size = len(outside_payload)
        archive.addfile(outside_member, io.BytesIO(outside_payload))

    extract_dir = tmp_path / "extracted"
    with pytest.raises(ValueError, match="Unsafe archive member path"):
        organizer._safe_unpack_archive(str(archive_path), str(extract_dir))

    assert not outside_path.exists()
    assert not (extract_dir / "dataset" / "safe.txt").exists()


def test_organize_imagenet_rejects_tar_symlink_escape(tmp_path: Path) -> None:
    """Reject a tar symlink before ImageNet extraction can write through it."""

    image_archive = tmp_path / "images.tar"
    xml_archive = tmp_path / "annotations.tgz"
    outside_path = tmp_path / "outside.txt"
    with tarfile.open(image_archive, "w") as archive:
        link = tarfile.TarInfo("redirect")
        link.type = tarfile.SYMTYPE
        link.linkname = str(tmp_path)
        archive.addfile(link)
        payload = b"outside"
        member = tarfile.TarInfo("redirect/outside.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    with tarfile.open(xml_archive, "w:gz"):
        pass

    with pytest.raises(ValueError, match="Unsafe archive member type"):
        organizer.organize_imagenet(str(image_archive), str(xml_archive), str(tmp_path / "imagenet"))

    assert not outside_path.exists()


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


def _write_cityscapes_archives(tmp_path: Path, sample_ids: list[str]) -> tuple[Path, Path]:
    """Create compact official-layout Cityscapes ZIP fixtures."""

    image_archive = tmp_path / "leftImg8bit_trainvaltest.zip"
    annotation_archive = tmp_path / "gtFine_trainvaltest.zip"
    with ZipFile(image_archive, "w") as archive:
        archive.writestr("leftImg8bit/train/aachen/aachen_000000_000001_leftImg8bit.png", b"train image")
        archive.writestr("leftImg8bit/test/berlin/berlin_000000_000001_leftImg8bit.png", b"test image")
        for sample_id in sample_ids:
            city = sample_id.split("_", maxsplit=1)[0]
            archive.writestr(f"leftImg8bit/val/{city}/{sample_id}_leftImg8bit.png", f"image:{sample_id}".encode())
    with ZipFile(annotation_archive, "w") as archive:
        archive.writestr("gtFine/train/aachen/aachen_000000_000001_gtFine_labelIds.png", b"train mask")
        for sample_id in sample_ids:
            city = sample_id.split("_", maxsplit=1)[0]
            archive.writestr(f"gtFine/val/{city}/{sample_id}_gtFine_labelIds.png", f"mask:{sample_id}".encode())
            archive.writestr(f"gtFine/val/{city}/{sample_id}_gtFine_color.png", b"color")
            archive.writestr(f"gtFine/val/{city}/{sample_id}_gtFine_instanceIds.png", b"instance")
            archive.writestr(f"gtFine/val/{city}/{sample_id}_gtFine_polygons.json", b"{}")
            archive.writestr(f"gtFine/val/{city}/{sample_id}_gtFine_trainIds.png", b"train IDs")
    return image_archive, annotation_archive


def test_organize_cityscapes_materializes_lossless_validation_pairs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Copy only exactly paired validation images and label-ID masks without transcoding."""

    monkeypatch.setattr(organizer, "CITYSCAPES_VALIDATION_SAMPLE_COUNT", 2)
    sample_ids = ["frankfurt_000000_000294", "munster_000001_000019"]
    image_archive, annotation_archive = _write_cityscapes_archives(tmp_path, sample_ids)
    output_dir = tmp_path / "cityscapes"
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "annotations").mkdir()
    (output_dir / "images" / "stale.png").write_bytes(b"stale")
    (output_dir / "annotations" / "stale.png").write_bytes(b"stale")

    organizer.organize_cityscapes(str(image_archive), str(annotation_archive), str(output_dir))

    image_paths = sorted((output_dir / "images").glob("*.png"))
    annotation_paths = sorted((output_dir / "annotations").glob("*.png"))
    assert [path.name for path in image_paths] == [f"{sample_id}.png" for sample_id in sample_ids]
    assert [path.name for path in annotation_paths] == [f"{sample_id}.png" for sample_id in sample_ids]
    assert image_paths[0].read_bytes() == f"image:{sample_ids[0]}".encode()
    assert annotation_paths[0].read_bytes() == f"mask:{sample_ids[0]}".encode()
    assert not list(output_dir.rglob("*train*"))
    assert not list(output_dir.rglob("*color*"))
    assert not list(output_dir.rglob("*instance*"))
    assert not list(output_dir.rglob("*.json"))


def test_organize_cityscapes_enforces_validation_pair_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject incomplete validation sources before replacing existing data."""

    monkeypatch.setattr(organizer, "CITYSCAPES_VALIDATION_SAMPLE_COUNT", 2)
    image_archive, annotation_archive = _write_cityscapes_archives(tmp_path, ["lindau_000000_000019"])
    output_dir = tmp_path / "cityscapes"
    (output_dir / "images").mkdir(parents=True)
    marker = output_dir / "images" / "keep.png"
    marker.write_bytes(b"keep")

    with pytest.raises(ValueError, match="must contain 2 pairs"):
        organizer.organize_cityscapes(str(image_archive), str(annotation_archive), str(output_dir))

    assert marker.read_bytes() == b"keep"


def test_organize_cityscapes_rejects_mismatched_and_malformed_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Require exact official stems and reject malformed validation candidates."""

    monkeypatch.setattr(organizer, "CITYSCAPES_VALIDATION_SAMPLE_COUNT", 1)
    image_archive, annotation_archive = _write_cityscapes_archives(tmp_path, ["lindau_000000_000019"])
    with ZipFile(annotation_archive, "w") as archive:
        archive.writestr("gtFine/val/lindau/lindau_000000_000020_gtFine_labelIds.png", b"mask")
    with pytest.raises(ValueError, match="mismatch"):
        organizer.organize_cityscapes(str(image_archive), str(annotation_archive), str(tmp_path / "mismatched"))

    with ZipFile(annotation_archive, "w") as archive:
        archive.writestr("gtFine/val/lindau/frankfurt_000000_000019_gtFine_labelIds.png", b"mask")
    with pytest.raises(ValueError, match="Malformed Cityscapes annotation filename"):
        organizer.organize_cityscapes(str(image_archive), str(annotation_archive), str(tmp_path / "malformed"))


def test_organize_cityscapes_rejects_non_zip_duplicate_and_unsafe_inputs(tmp_path: Path) -> None:
    """Reject invalid ZIPs, duplicate members, and traversal paths before installation."""

    invalid_archive = tmp_path / "invalid.zip"
    invalid_archive.write_bytes(b"not a zip")
    valid_image_archive, valid_annotation_archive = _write_cityscapes_archives(
        tmp_path,
        ["lindau_000000_000019"],
    )
    with pytest.raises(ValueError, match="does not exist"):
        organizer.organize_cityscapes(
            str(tmp_path / "missing.zip"),
            str(valid_annotation_archive),
            str(tmp_path / "missing"),
        )
    with pytest.raises(ValueError, match="must be a valid ZIP"):
        organizer.organize_cityscapes(str(invalid_archive), str(valid_annotation_archive), str(tmp_path / "invalid"))

    duplicate_archive = tmp_path / "duplicate.zip"
    duplicate_member = "leftImg8bit/val/lindau/lindau_000000_000019_leftImg8bit.png"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with ZipFile(duplicate_archive, "w") as archive:
            archive.writestr(duplicate_member, b"first")
            archive.writestr(duplicate_member, b"second")
    with pytest.raises(ValueError, match="duplicate members"):
        organizer.organize_cityscapes(
            str(duplicate_archive), str(valid_annotation_archive), str(tmp_path / "duplicate")
        )

    unsafe_archive = tmp_path / "unsafe.zip"
    outside_marker = tmp_path / "outside.png"
    with ZipFile(unsafe_archive, "w") as archive:
        archive.writestr("../outside.png", b"outside")
    with pytest.raises(ValueError, match="Unsafe archive member path"):
        organizer.organize_cityscapes(str(unsafe_archive), str(valid_annotation_archive), str(tmp_path / "unsafe"))
    assert not outside_marker.exists()


def test_organize_cityscapes_rolls_back_failed_atomic_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Restore both previous directories if the staged installation fails."""

    monkeypatch.setattr(organizer, "CITYSCAPES_VALIDATION_SAMPLE_COUNT", 1)
    image_archive, annotation_archive = _write_cityscapes_archives(tmp_path, ["lindau_000000_000019"])
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
        organizer.organize_cityscapes(str(image_archive), str(annotation_archive), str(output_dir))

    assert (output_dir / "images" / "keep.png").read_bytes() == b"old image"
    assert (output_dir / "annotations" / "keep.png").read_bytes() == b"old annotation"
