from __future__ import annotations

import csv
import hashlib
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .errors import DatasetValidationError
from .io_utils import clean_generated_dataset, ensure_dir, read_yaml, write_yaml

IMAGE_EXTENSIONS = {
    ".bmp",
    ".dng",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
BOX_BOUNDARY_TOLERANCE = 1e-6


def normalize_names(raw_names: object) -> list[str]:
    if isinstance(raw_names, list):
        names = [str(name).strip() for name in raw_names]
    elif isinstance(raw_names, dict):
        converted = {int(key): str(value).strip() for key, value in raw_names.items()}
        expected = list(range(len(converted)))
        if sorted(converted) != expected:
            raise DatasetValidationError(
                f"Class IDs must be sequential from 0; found {sorted(converted)}"
            )
        names = [converted[index] for index in expected]
    else:
        raise DatasetValidationError("data.yaml must define 'names' as a list or mapping")
    if not names or any(not name for name in names):
        raise DatasetValidationError("The class-name list cannot be empty")
    if len(set(names)) != len(names):
        raise DatasetValidationError("Class names in data.yaml must be unique")
    return names


def _resolve_root(yaml_path: Path, configured_path: object) -> Path:
    if configured_path in (None, ""):
        return yaml_path.parent
    raw = Path(str(configured_path)).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    candidates = [
        (Path.cwd() / raw).resolve(),
        (yaml_path.parent / raw).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if (yaml_path.parent / "images").exists():
        return yaml_path.parent
    return candidates[0]


@dataclass(frozen=True)
class DatasetSpec:
    yaml_path: Path
    root: Path
    names: list[str]
    splits: dict[str, Path]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetSpec":
        yaml_path = Path(path).expanduser().resolve()
        data = read_yaml(yaml_path)
        names = normalize_names(data.get("names"))
        root = _resolve_root(yaml_path, data.get("path"))
        splits: dict[str, Path] = {}
        for split in ("train", "val", "test"):
            configured = data.get(split)
            if configured in (None, ""):
                continue
            if isinstance(configured, list):
                raise DatasetValidationError(
                    f"This NestDetect version requires one directory for split '{split}'"
                )
            split_path = Path(str(configured)).expanduser()
            splits[split] = (
                split_path.resolve() if split_path.is_absolute() else (root / split_path).resolve()
            )
        if "train" not in splits or "val" not in splits:
            raise DatasetValidationError("data.yaml must define train and val splits")
        return cls(yaml_path=yaml_path, root=root, names=names, splits=splits)


@dataclass(frozen=True)
class Label:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_line(self, class_id: int | None = None) -> str:
        values = (
            self.class_id if class_id is None else class_id,
            self.x_center,
            self.y_center,
            self.width,
            self.height,
        )
        return f"{values[0]} " + " ".join(f"{value:.8g}" for value in values[1:])


@dataclass(frozen=True)
class Sample:
    split: str
    image_path: Path
    label_path: Path
    labels: tuple[Label, ...]

    @property
    def class_ids(self) -> set[int]:
        return {label.class_id for label in self.labels}


@dataclass
class ValidationIssue:
    severity: str
    code: str
    path: str
    message: str


@dataclass
class ValidationReport:
    dataset: str
    names: list[str]
    image_counts: dict[str, int] = field(default_factory=dict)
    instance_counts: dict[str, int] = field(default_factory=dict)
    class_image_counts: dict[str, int] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    @property
    def valid(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update(
            valid=self.valid,
            error_count=self.error_count,
            warning_count=self.warning_count,
        )
        return result


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def corresponding_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_indexes = [index for index, part in enumerate(parts) if part == "images"]
    if image_indexes:
        parts[image_indexes[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def parse_label_file(path: Path, class_count: int) -> tuple[list[Label], list[str]]:
    labels: list[Label] = []
    errors: list[str] = []
    if not path.exists():
        return labels, errors
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            errors.append(
                f"line {line_number}: expected 5 values, found {len(fields)}"
            )
            continue
        try:
            class_id_float = float(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"line {line_number}: all values must be numeric")
            continue
        class_id = int(class_id_float)
        if class_id_float != class_id:
            errors.append(f"line {line_number}: class_id must be an integer")
            continue
        if not 0 <= class_id < class_count:
            errors.append(
                f"line {line_number}: class_id {class_id} is outside 0..{class_count - 1}"
            )
            continue
        x_center, y_center, width, height = values
        if not all(0.0 <= value <= 1.0 for value in values):
            errors.append(f"line {line_number}: coordinates must be within 0..1")
            continue
        if width <= 0.0 or height <= 0.0:
            errors.append(f"line {line_number}: width and height must be greater than 0")
            continue
        if (
            x_center - width / 2 < -BOX_BOUNDARY_TOLERANCE
            or x_center + width / 2 > 1 + BOX_BOUNDARY_TOLERANCE
        ):
            errors.append(f"line {line_number}: bounding box exceeds horizontal bounds")
            continue
        if (
            y_center - height / 2 < -BOX_BOUNDARY_TOLERANCE
            or y_center + height / 2 > 1 + BOX_BOUNDARY_TOLERANCE
        ):
            errors.append(f"line {line_number}: bounding box exceeds vertical bounds")
            continue
        labels.append(Label(class_id, x_center, y_center, width, height))
    return labels, errors


def load_samples(spec: DatasetSpec, split: str) -> list[Sample]:
    directory = spec.splits.get(split)
    if directory is None:
        return []
    samples: list[Sample] = []
    for image_path in image_files(directory):
        label_path = corresponding_label_path(image_path)
        labels, errors = parse_label_file(label_path, len(spec.names))
        if errors:
            joined = "; ".join(errors)
            raise DatasetValidationError(f"Invalid label {label_path}: {joined}")
        samples.append(Sample(split, image_path, label_path, tuple(labels)))
    return samples


def validate_dataset(path: str | Path) -> ValidationReport:
    spec = DatasetSpec.from_yaml(path)
    report = ValidationReport(dataset=str(spec.yaml_path), names=spec.names)
    instances: Counter[int] = Counter()
    class_images: Counter[int] = Counter()
    for split, directory in spec.splits.items():
        if not directory.is_dir():
            report.issues.append(
                ValidationIssue(
                    "error",
                    "missing_split",
                    str(directory),
                    f"Split directory {split} does not exist",
                )
            )
            report.image_counts[split] = 0
            continue
        images = image_files(directory)
        report.image_counts[split] = len(images)
        if not images:
            report.issues.append(
                ValidationIssue(
                    "error",
                    "empty_split",
                    str(directory),
                    f"Split {split} contains no images",
                )
            )
        for image_path in images:
            label_path = corresponding_label_path(image_path)
            if not label_path.exists():
                report.issues.append(
                    ValidationIssue(
                        "warning",
                        "missing_label",
                        str(label_path),
                        "Label file is missing; the image is treated as background",
                    )
                )
                continue
            if not label_path.read_text(encoding="utf-8").strip():
                report.issues.append(
                    ValidationIssue(
                        "warning",
                        "empty_label",
                        str(label_path),
                        "Label file is empty",
                    )
                )
                continue
            labels, errors = parse_label_file(label_path, len(spec.names))
            for error in errors:
                report.issues.append(
                    ValidationIssue("error", "invalid_label", str(label_path), error)
                )
            class_ids = {label.class_id for label in labels}
            class_images.update(class_ids)
            instances.update(label.class_id for label in labels)
    report.instance_counts = {
        name: instances.get(class_id, 0) for class_id, name in enumerate(spec.names)
    }
    report.class_image_counts = {
        name: class_images.get(class_id, 0) for class_id, name in enumerate(spec.names)
    }
    for class_id, name in enumerate(spec.names):
        if instances.get(class_id, 0) == 0:
            report.issues.append(
                ValidationIssue(
                    "warning",
                    "class_without_instances",
                    str(spec.yaml_path),
                    f"Class '{name}' has no instances",
                )
            )
    return report


def merge_class_names(*name_groups: Sequence[str]) -> list[str]:
    merged: list[str] = []
    for names in name_groups:
        for name in names:
            if name not in merged:
                merged.append(name)
    return merged


def class_id_mapping(source_names: Sequence[str], target_names: Sequence[str]) -> dict[int, int]:
    target = {name: index for index, name in enumerate(target_names)}
    missing = [name for name in source_names if name not in target]
    if missing:
        raise DatasetValidationError(
            f"Classes are missing from the target names: {', '.join(missing)}"
        )
    return {source_id: target[name] for source_id, name in enumerate(source_names)}


def _sample_key(spec: DatasetSpec, sample: Sample, prefix: str) -> str:
    try:
        relative = sample.image_path.relative_to(spec.root)
    except ValueError:
        relative = sample.image_path
    digest = hashlib.sha1(str(relative).encode("utf-8")).hexdigest()[:10]
    safe_prefix = "".join(char if char.isalnum() else "_" for char in prefix).strip("_")
    return f"{safe_prefix}_{sample.image_path.stem}_{digest}"


def copy_sample(
    spec: DatasetSpec,
    sample: Sample,
    output_root: Path,
    target_names: Sequence[str],
    prefix: str,
    explicit_mapping: Mapping[int, int] | None = None,
) -> tuple[Path, Path]:
    mapping = (
        dict(explicit_mapping)
        if explicit_mapping is not None
        else class_id_mapping(spec.names, target_names)
    )
    missing_ids = sorted(sample.class_ids - mapping.keys())
    if missing_ids:
        missing_names = ", ".join(spec.names[class_id] for class_id in missing_ids)
        raise DatasetValidationError(f"Class mapping is missing for: {missing_names}")
    invalid_targets = sorted(
        target_id for target_id in mapping.values() if not 0 <= target_id < len(target_names)
    )
    if invalid_targets:
        raise DatasetValidationError(
            f"Target class IDs are outside 0..{len(target_names) - 1}: {invalid_targets}"
        )
    image_dir = ensure_dir(output_root / "images" / sample.split)
    label_dir = ensure_dir(output_root / "labels" / sample.split)
    key = _sample_key(spec, sample, prefix)
    destination_image = image_dir / f"{key}{sample.image_path.suffix.lower()}"
    destination_label = label_dir / f"{key}.txt"
    try:
        os.link(sample.image_path, destination_image)
    except OSError:
        shutil.copy2(sample.image_path, destination_image)
    with destination_label.open("w", encoding="utf-8") as handle:
        for label in sample.labels:
            handle.write(label.to_line(mapping[label.class_id]) + "\n")
    return destination_image, destination_label


def write_dataset_yaml(
    output_root: str | Path,
    names: Sequence[str],
    available_splits: Iterable[str] = ("train", "val"),
) -> Path:
    root = Path(output_root).resolve()
    splits = set(available_splits)
    data: dict[str, object] = {"path": str(root)}
    if "train" not in splits and "val" in splits:
        data["train"] = "images/val"
    for split in ("train", "val", "test"):
        if split in splits:
            data[split] = f"images/{split}"
    data["names"] = {index: name for index, name in enumerate(names)}
    return write_yaml(root / "data.yaml", data)


def materialize_dataset(
    source_yaml: str | Path,
    output_root: str | Path,
    target_names: Sequence[str] | None = None,
    splits: Sequence[str] = ("train", "val"),
    clean: bool = True,
    prefix: str = "source",
) -> Path:
    report = validate_dataset(source_yaml)
    if not report.valid:
        raise DatasetValidationError(
            f"The source dataset has {report.error_count} errors: {source_yaml}"
        )
    spec = DatasetSpec.from_yaml(source_yaml)
    names = list(target_names or spec.names)
    output = clean_generated_dataset(output_root) if clean else ensure_dir(output_root)
    manifest: list[dict[str, str]] = []
    copied_splits: set[str] = set()
    for split in splits:
        if split not in spec.splits:
            continue
        copied_splits.add(split)
        for sample in load_samples(spec, split):
            destination, _ = copy_sample(spec, sample, output, names, prefix)
            manifest.append(
                {
                    "split": split,
                    "source": str(sample.image_path),
                    "destination": str(destination),
                    "classes": ",".join(spec.names[index] for index in sorted(sample.class_ids)),
                }
            )
    _write_manifest(output / "manifest.csv", manifest)
    return write_dataset_yaml(output, names, copied_splits)


def materialize_dataset_with_id_mapping(
    source_yaml: str | Path,
    output_root: str | Path,
    target_names: Sequence[str],
    id_mapping: Mapping[int, int],
    splits: Sequence[str] = ("train", "val"),
    clean: bool = True,
    prefix: str = "source",
) -> Path:
    report = validate_dataset(source_yaml)
    if not report.valid:
        raise DatasetValidationError(
            f"The source dataset has {report.error_count} errors: {source_yaml}"
        )
    spec = DatasetSpec.from_yaml(source_yaml)
    expected_ids = set(range(len(spec.names)))
    missing_ids = sorted(expected_ids - set(id_mapping))
    if missing_ids:
        missing_names = ", ".join(spec.names[class_id] for class_id in missing_ids)
        raise DatasetValidationError(f"Class mapping is incomplete: {missing_names}")
    names = list(target_names)
    output = clean_generated_dataset(output_root) if clean else ensure_dir(output_root)
    manifest: list[dict[str, str]] = []
    copied_splits: set[str] = set()
    for split in splits:
        if split not in spec.splits:
            continue
        copied_splits.add(split)
        for sample in load_samples(spec, split):
            destination, _ = copy_sample(
                spec,
                sample,
                output,
                names,
                prefix,
                explicit_mapping=id_mapping,
            )
            manifest.append(
                {
                    "split": split,
                    "source": str(sample.image_path),
                    "destination": str(destination),
                    "classes": ",".join(
                        names[id_mapping[class_id]] for class_id in sorted(sample.class_ids)
                    ),
                }
            )
    _write_manifest(output / "manifest.csv", manifest)
    return write_dataset_yaml(output, names, copied_splits)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
    fieldnames = ["split", "source", "destination", "classes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_incremental_dataset(
    new_dataset_yaml: str | Path,
    output_root: str | Path,
    replay_dataset_yaml: str | Path | None = None,
    base_names: Sequence[str] | None = None,
    target_names: Sequence[str] | None = None,
) -> Path:
    new_report = validate_dataset(new_dataset_yaml)
    if not new_report.valid:
        raise DatasetValidationError(
            f"The new-class dataset has {new_report.error_count} errors"
        )
    if replay_dataset_yaml:
        replay_report = validate_dataset(replay_dataset_yaml)
        if not replay_report.valid:
            raise DatasetValidationError(
                f"Replay memory has {replay_report.error_count} errors"
            )
    new_spec = DatasetSpec.from_yaml(new_dataset_yaml)
    replay_spec = DatasetSpec.from_yaml(replay_dataset_yaml) if replay_dataset_yaml else None
    if target_names:
        names = list(target_names)
    else:
        names = merge_class_names(
            list(base_names or []),
            replay_spec.names if replay_spec else [],
            new_spec.names,
        )
    output = clean_generated_dataset(output_root)
    rows: list[dict[str, str]] = []
    copied_splits: set[str] = set()
    sources = [("new", new_spec)]
    if replay_spec:
        sources.append(("replay", replay_spec))
    for prefix, spec in sources:
        for split in ("train", "val", "test"):
            if split not in spec.splits:
                continue
            copied_splits.add(split)
            for sample in load_samples(spec, split):
                destination, _ = copy_sample(spec, sample, output, names, prefix)
                rows.append(
                    {
                        "split": split,
                        "source": str(sample.image_path),
                        "destination": str(destination),
                        "classes": ",".join(
                            spec.names[index] for index in sorted(sample.class_ids)
                        ),
                    }
                )
    _write_manifest(output / "manifest.csv", rows)
    return write_dataset_yaml(output, names, copied_splits)
