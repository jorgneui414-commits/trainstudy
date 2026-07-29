"""Validate and split flat YOLO annotations into versioned datasets.

The source directory must contain ``classes.txt`` plus image files and their
same-stem ``.txt`` labels.  A successful build creates a new immutable version
under ``<dataset_dir>/versions`` and then atomically refreshes
``<dataset_dir>/data.yaml`` to point at that version.
"""

from __future__ import annotations

import math
import os
import random
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import yaml


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
SPLIT_NAMES = ("train", "val", "test")
_INTEGER_PATTERN = re.compile(r"[+-]?\d+\Z")


class DatasetValidationError(ValueError):
    """Raised when source data cannot form a valid YOLO dataset."""


@dataclass(frozen=True)
class Sample:
    """One validated image and its matching YOLO label."""

    image: Path
    label: Path


def _read_utf8(path: Path, description: str) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(f"{description} must be UTF-8 encoded: {path}") from exc


def _read_classes(classes_path: Path) -> list[str]:
    if not classes_path.is_file():
        raise FileNotFoundError(f"classes.txt not found: {classes_path}")

    names = [line.strip() for line in _read_utf8(classes_path, "classes.txt").splitlines()]
    names = [name for name in names if name]
    if not names:
        raise DatasetValidationError("classes.txt must contain at least one class name.")

    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise DatasetValidationError(
            "classes.txt contains duplicate class names: "
            + ", ".join(repr(name) for name in duplicates)
        )
    return names


def _index_by_stem(paths: Iterable[Path], kind: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    duplicate_groups: dict[str, list[Path]] = {}

    for path in paths:
        key = path.stem.casefold()
        if key in indexed:
            duplicate_groups.setdefault(key, [indexed[key]]).append(path)
        else:
            indexed[key] = path

    if duplicate_groups:
        details = []
        for group in duplicate_groups.values():
            details.append("/".join(sorted(path.name for path in group)))
        raise DatasetValidationError(f"Duplicate {kind} stems found: {', '.join(details)}")
    return indexed


def _find_samples(raw_dir: Path) -> list[Sample]:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw annotation directory not found: {raw_dir}")

    image_paths = [
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    label_paths = [
        path
        for path in raw_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".txt"
        and path.name.casefold() != "classes.txt"
    ]

    images = _index_by_stem(image_paths, "image")
    labels = _index_by_stem(label_paths, "label")
    if not images:
        supported = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise DatasetValidationError(
            f"No supported images found in {raw_dir}. Supported: {supported}"
        )

    missing_keys = sorted(images.keys() - labels.keys())
    orphan_keys = sorted(labels.keys() - images.keys())
    if missing_keys or orphan_keys:
        messages = []
        if missing_keys:
            messages.append(
                "missing labels for " + ", ".join(images[key].name for key in missing_keys)
            )
        if orphan_keys:
            messages.append(
                "orphan labels " + ", ".join(labels[key].name for key in orphan_keys)
            )
        raise DatasetValidationError("Image/label pairing failed: " + "; ".join(messages))

    ordered_keys = sorted(
        images, key=lambda key: (images[key].name.casefold(), images[key].name)
    )
    return [Sample(images[key], labels[key]) for key in ordered_keys]


def _parse_class_id(token: str, class_count: int, location: str) -> int:
    if not _INTEGER_PATTERN.fullmatch(token):
        raise DatasetValidationError(f"Class id must be an integer at {location}: {token!r}")
    class_id = int(token)
    if not 0 <= class_id < class_count:
        raise DatasetValidationError(
            f"Class id {class_id} is outside the valid range 0..{class_count - 1} at {location}."
        )
    return class_id


def _validate_labels(samples: Sequence[Sample], class_count: int) -> int:
    column_count: int | None = None
    annotation_count = 0

    for sample in samples:
        text = _read_utf8(sample.label, "Label file")
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            location = f"{sample.label}:{line_number}"
            if len(parts) not in (5, 9):
                raise DatasetValidationError(
                    f"Invalid YOLO label at {location}: expected 5 columns (Detection) "
                    f"or 9 columns (OBB), got {len(parts)}."
                )
            if column_count is None:
                column_count = len(parts)
            elif len(parts) != column_count:
                raise DatasetValidationError(
                    f"Detection and OBB labels cannot be mixed; {location} has {len(parts)} "
                    f"columns but earlier annotations have {column_count}."
                )

            _parse_class_id(parts[0], class_count, location)
            for token in parts[1:]:
                try:
                    value = float(token)
                except ValueError as exc:
                    raise DatasetValidationError(
                        f"Coordinate must be a number at {location}: {token!r}"
                    ) from exc
                if not math.isfinite(value):
                    raise DatasetValidationError(
                        f"Coordinate must be finite at {location}: {token!r}"
                    )
                if not 0.0 <= value <= 1.0:
                    raise DatasetValidationError(
                        f"Coordinate must be normalized to [0, 1] at {location}: {token!r}"
                    )
            annotation_count += 1

    if annotation_count == 0:
        raise DatasetValidationError(
            "All label files are empty. At least one annotated object is required."
        )
    assert column_count is not None
    return column_count


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in ratios):
        raise DatasetValidationError("Split ratios must be numeric values.")
    if any(not math.isfinite(float(value)) or value <= 0 for value in ratios):
        raise DatasetValidationError(
            "Train, validation, and test ratios must all be positive and finite."
        )
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise DatasetValidationError(
            f"Split ratios must sum to 1.0; received {sum(ratios):.12g}."
        )


def _split_samples(
    samples: Sequence[Sample],
    train_ratio: float,
    val_ratio: float,
    random_seed: int,
) -> dict[str, list[Sample]]:
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise DatasetValidationError("random_seed must be an integer.")

    shuffled = list(samples)
    random.Random(random_seed).shuffle(shuffled)
    train_count = math.floor(len(shuffled) * train_ratio)
    val_count = math.floor(len(shuffled) * val_ratio)
    test_count = len(shuffled) - train_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        raise DatasetValidationError(
            "The selected ratios leave an empty split: "
            f"train={train_count}, val={val_count}, test={test_count}. "
            "Add samples or change ratios."
        )

    train_end = train_count
    val_end = train_end + val_count
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def _next_version_dir(versions_dir: Path, timestamp: str) -> Path:
    candidate = versions_dir / timestamp
    suffix = 1
    while candidate.exists():
        candidate = versions_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    return candidate


def _yaml_text(version_dir: Path, class_names: Sequence[str]) -> str:
    data = {
        "path": version_dir.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


def _copy_version(
    staging_dir: Path,
    version_dir: Path,
    splits: dict[str, list[Sample]],
    classes_path: Path,
    class_names: Sequence[str],
) -> None:
    for split_name in SPLIT_NAMES:
        (staging_dir / "images" / split_name).mkdir(parents=True, exist_ok=False)
        (staging_dir / "labels" / split_name).mkdir(parents=True, exist_ok=False)

    for split_name, samples in splits.items():
        for sample in samples:
            shutil.copy2(sample.image, staging_dir / "images" / split_name / sample.image.name)
            shutil.copy2(sample.label, staging_dir / "labels" / split_name / sample.label.name)

    shutil.copy2(classes_path, staging_dir / "classes.txt")
    (staging_dir / "data.yaml").write_text(
        _yaml_text(version_dir, class_names), encoding="utf-8", newline="\n"
    )


def _replace_stable_yaml(dataset_dir: Path, yaml_text: str) -> None:
    temporary = dataset_dir / f".data.yaml.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(yaml_text, encoding="utf-8", newline="\n")
        os.replace(temporary, dataset_dir / "data.yaml")
    finally:
        temporary.unlink(missing_ok=True)


def build_yolo_dataset(
    raw_dir: str | Path,
    dataset_dir: str | Path,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Path:
    """Build one validated, reproducibly split YOLO dataset version.

    Args:
        raw_dir: Flat directory containing images, labels, and ``classes.txt``.
        dataset_dir: Root containing ``versions/`` and the stable ``data.yaml``.
        train_ratio: Fraction used to calculate the training count (floor).
        val_ratio: Fraction used to calculate the validation count (floor).
        test_ratio: Required test fraction; rounding remainder goes to test.
        random_seed: Seed for deterministic shuffling after filename sorting.

    Returns:
        Absolute path to the newly created version's ``data.yaml``.

    Raises:
        FileNotFoundError: When the source directory or classes file is missing.
        DatasetValidationError: When data or split settings are invalid.
    """

    raw_path = Path(raw_dir).expanduser().resolve()
    dataset_path = Path(dataset_dir).expanduser().resolve()
    if raw_path == dataset_path:
        raise DatasetValidationError("raw_dir and dataset_dir must be different directories.")

    _validate_ratios(train_ratio, val_ratio, test_ratio)
    if not raw_path.is_dir():
        raise FileNotFoundError(f"Raw annotation directory not found: {raw_path}")
    classes_path = raw_path / "classes.txt"
    class_names = _read_classes(classes_path)
    samples = _find_samples(raw_path)
    _validate_labels(samples, len(class_names))
    splits = _split_samples(samples, train_ratio, val_ratio, random_seed)

    versions_dir = dataset_path / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = _next_version_dir(versions_dir, timestamp)
    staging_dir = dataset_path / f".staging_{timestamp}_{uuid.uuid4().hex}"
    committed = False

    try:
        staging_dir.mkdir(parents=False, exist_ok=False)
        _copy_version(staging_dir, version_dir, splits, classes_path, class_names)
        staging_dir.rename(version_dir)
        committed = True

        version_yaml = version_dir / "data.yaml"
        _replace_stable_yaml(dataset_path, version_yaml.read_text(encoding="utf-8"))
        return version_yaml.resolve()
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if committed and version_dir.exists():
            shutil.rmtree(version_dir)
        raise


__all__ = [
    "DatasetValidationError",
    "IMAGE_EXTENSIONS",
    "Sample",
    "build_yolo_dataset",
]
