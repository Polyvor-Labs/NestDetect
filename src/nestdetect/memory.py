from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path

from .dataset import (
    DatasetSpec,
    Sample,
    copy_sample,
    load_samples,
    validate_dataset,
    write_dataset_yaml,
)
from .errors import DatasetValidationError
from .io_utils import clean_generated_dataset, ensure_dir


def select_balanced_samples(
    samples: list[Sample], class_count: int, per_class: int, seed: int
) -> list[Sample]:
    if per_class < 1:
        raise ValueError("per_class must be at least 1")
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    selected: dict[Path, Sample] = {}
    counts: Counter[int] = Counter()
    while True:
        progressed = False
        for class_id in range(class_count):
            if counts[class_id] >= per_class:
                continue
            candidate = next(
                (
                    sample
                    for sample in shuffled
                    if class_id in sample.class_ids and sample.image_path not in selected
                ),
                None,
            )
            if candidate is None:
                continue
            selected[candidate.image_path] = candidate
            counts.update(candidate.class_ids)
            progressed = True
        if not progressed or all(counts[class_id] >= per_class for class_id in range(class_count)):
            break
    return sorted(selected.values(), key=lambda sample: str(sample.image_path))


def build_replay_memory(
    source_yaml: str | Path,
    output_root: str | Path,
    train_per_class: int = 30,
    val_per_class: int | None = None,
    seed: int = 42,
) -> Path:
    report = validate_dataset(source_yaml)
    if not report.valid:
        raise DatasetValidationError(
            f"The source dataset has {report.error_count} errors; replay memory was not built"
        )
    spec = DatasetSpec.from_yaml(source_yaml)
    output = clean_generated_dataset(output_root)
    val_limit = val_per_class if val_per_class is not None else max(1, train_per_class // 5)
    rows: list[dict[str, str | int]] = []
    available_splits: set[str] = set()
    for split, per_class in (("train", train_per_class), ("val", val_limit)):
        if split not in spec.splits:
            continue
        available_splits.add(split)
        selected = select_balanced_samples(
            load_samples(spec, split), len(spec.names), per_class, seed + (split == "val")
        )
        represented = {class_id for sample in selected for class_id in sample.class_ids}
        missing = [
            spec.names[class_id]
            for class_id in range(len(spec.names))
            if class_id not in represented
        ]
        if missing:
            raise DatasetValidationError(
                f"Split {split} cannot represent replay classes: {', '.join(missing)}"
            )
        for sample in selected:
            destination, _ = copy_sample(spec, sample, output, spec.names, "memory")
            rows.append(
                {
                    "split": split,
                    "source": str(sample.image_path),
                    "destination": str(destination),
                    "classes": ",".join(spec.names[index] for index in sorted(sample.class_ids)),
                    "seed": seed,
                }
            )
    index_path = output / "memory_index.csv"
    ensure_dir(index_path.parent)
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["split", "source", "destination", "classes", "seed"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return write_dataset_yaml(output, spec.names, available_splits)
