from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import tempfile
import time
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import yaml

LABELS_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    "coco2017labels.zip"
)
IMAGE_URL = "http://images.cocodataset.org/{split}/{image_id}.jpg"
COCO_PAPER = "https://arxiv.org/abs/1405.0312"
COCO_TERMS = "https://cocodataset.org/#termsofuse"

CLASS_RULES = {
    0: {"select_min_area": 0.05, "select_max_area": 0.80, "label_min_area": 0.01},
    56: {"select_min_area": 0.05, "select_max_area": 0.75, "label_min_area": 0.01},
    60: {"select_min_area": 0.08, "select_max_area": 0.75, "label_min_area": 0.02},
    63: {"select_min_area": 0.08, "select_max_area": 0.75, "label_min_area": 0.02},
    73: {"select_min_area": 0.05, "select_max_area": 0.65, "label_min_area": 0.01},
}


@dataclass(frozen=True)
class Stage:
    name: str
    output: str
    classes: tuple[tuple[int, str], ...]
    excluded_ids: frozenset[int]

    @property
    def target_ids(self) -> frozenset[int]:
        return frozenset(class_id for class_id, _ in self.classes)


STAGES = (
    Stage(
        name="base",
        output="step1_base",
        classes=((0, "person"), (56, "chair"), (60, "table")),
        excluded_ids=frozenset({63, 73}),
    ),
    Stage(
        name="incremental",
        output="step2_new_classes",
        classes=((63, "laptop"), (73, "book")),
        excluded_ids=frozenset({0, 56, 60}),
    ),
)


def download(url: str, output: Path, retries: int = 4) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_size > 0:
        return output
    partial = output.with_suffix(output.suffix + ".part")
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "NestDetect/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                with partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            partial.replace(output)
            return output
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    return output


def prepare_labels(cache_dir: Path, provided_root: Path | None) -> Path:
    if provided_root:
        labels_root = provided_root.expanduser().resolve()
    else:
        archive = download(LABELS_URL, cache_dir / "coco2017labels.zip")
        extracted = cache_dir / "coco2017labels"
        marker = extracted / "coco" / "labels" / "train2017"
        if not marker.is_dir():
            extracted.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(extracted)
        labels_root = extracted / "coco" / "labels"
    if not (labels_root / "train2017").is_dir() or not (labels_root / "val2017").is_dir():
        raise FileNotFoundError(
            f"COCO train2017/val2017 label directories were not found in {labels_root}"
        )
    return labels_root


def parse_label(path: Path) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 5:
            result.append((int(fields[0]), " ".join(fields[1:])))
    return result


def select_balanced(
    labels_dir: Path,
    stage: Stage,
    per_class: int,
    seed: int,
) -> list[Path]:
    candidates: list[tuple[Path, frozenset[int]]] = []
    for label_path in labels_dir.glob("*.txt"):
        labels = parse_label(label_path)
        class_ids = frozenset(class_id for class_id, _ in labels)
        hit = class_ids & stage.target_ids
        eligible: set[int] = set()
        for class_id in hit:
            rule = CLASS_RULES[class_id]
            areas = [
                float(coordinates.split()[2]) * float(coordinates.split()[3])
                for label_id, coordinates in labels
                if label_id == class_id
            ]
            if any(
                rule["select_min_area"] <= area <= rule["select_max_area"]
                for area in areas
            ):
                eligible.add(class_id)
        if eligible and not class_ids & stage.excluded_ids:
            candidates.append((label_path, frozenset(eligible)))
    random.Random(seed).shuffle(candidates)
    selected: dict[Path, frozenset[int]] = {}
    counts: Counter[int] = Counter()
    while True:
        progressed = False
        for class_id in stage.target_ids:
            if counts[class_id] >= per_class:
                continue
            candidate = next(
                (
                    item
                    for item in candidates
                    if class_id in item[1] and item[0] not in selected
                ),
                None,
            )
            if candidate is None:
                continue
            selected[candidate[0]] = candidate[1]
            counts.update(candidate[1])
            progressed = True
        if not progressed or all(counts[class_id] >= per_class for class_id in stage.target_ids):
            break
    missing = {
        name: max(0, per_class - counts[class_id])
        for class_id, name in stage.classes
        if counts[class_id] < per_class
    }
    if missing:
        raise RuntimeError(f"Insufficient COCO samples for {stage.name}: {missing}")
    return sorted(selected)


def reset_output(root: Path) -> None:
    for relative in ("images", "labels"):
        target = root / relative
        if target.exists():
            shutil.rmtree(target)
    for relative in ("manifest.csv", "dataset_metadata.json"):
        (root / relative).unlink(missing_ok=True)


def write_data_yaml(root: Path, classes: tuple[tuple[int, str], ...], project_root: Path) -> None:
    relative_root = root.relative_to(project_root).as_posix()
    payload = {
        "path": relative_root,
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, (_, name) in enumerate(classes)},
    }
    (root / "data.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def materialize_stage(
    project_root: Path,
    labels_root: Path,
    cache_dir: Path,
    stage: Stage,
    train_per_class: int,
    val_per_class: int,
    seed: int,
    workers: int,
) -> dict[str, object]:
    output_root = project_root / "datasets" / stage.output
    reset_output(output_root)
    id_mapping = {
        source_id: target_id
        for target_id, (source_id, _) in enumerate(stage.classes)
    }
    rows: list[dict[str, object]] = []
    downloads: list[tuple[str, Path]] = []
    image_links: list[tuple[Path, Path]] = []
    selected_by_split: dict[str, list[Path]] = {}
    for offset, (source_split, target_split, limit) in enumerate(
        (("train2017", "train", train_per_class), ("val2017", "val", val_per_class))
    ):
        selected = select_balanced(
            labels_root / source_split,
            stage,
            limit,
            seed + offset,
        )
        selected_by_split[target_split] = selected
        image_dir = output_root / "images" / target_split
        label_dir = output_root / "labels" / target_split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for source_label in selected:
            image_id = source_label.stem
            url = IMAGE_URL.format(split=source_split, image_id=image_id)
            destination_image = image_dir / f"{image_id}.jpg"
            cached_image = cache_dir / "images" / source_split / f"{image_id}.jpg"
            downloads.append((url, cached_image))
            image_links.append((cached_image, destination_image))
            filtered = []
            for class_id, coordinates in parse_label(source_label):
                if class_id not in id_mapping:
                    continue
                _, _, width, height = map(float, coordinates.split())
                if width * height < CLASS_RULES[class_id]["label_min_area"]:
                    continue
                filtered.append(f"{id_mapping[class_id]} {coordinates}")
            (label_dir / f"{image_id}.txt").write_text(
                "\n".join(filtered) + "\n",
                encoding="utf-8",
            )
            present_ids = {
                class_id
                for class_id, _ in parse_label(source_label)
                if class_id in id_mapping
            }
            rows.append(
                {
                    "stage": stage.name,
                    "split": target_split,
                    "coco_split": source_split,
                    "coco_image_id": image_id,
                    "source_url": url,
                    "classes": ",".join(
                        name for class_id, name in stage.classes if class_id in present_ids
                    ),
                }
            )
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(download, url, destination): destination
            for url, destination in downloads
        }
        for index, future in enumerate(as_completed(futures), start=1):
            future.result()
            if index % 20 == 0 or index == len(futures):
                print(f"{stage.name}: downloaded {index}/{len(futures)} images", flush=True)
    for cached_image, destination_image in image_links:
        try:
            os.link(cached_image, destination_image)
        except OSError:
            shutil.copy2(cached_image, destination_image)
    write_data_yaml(output_root, stage.classes, project_root)
    with (output_root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "split",
                "coco_split",
                "coco_image_id",
                "source_url",
                "classes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "dataset": "Microsoft COCO 2017 curated lightweight subset",
        "stage": stage.name,
        "classes": [name for _, name in stage.classes],
        "coco_class_ids": {name: class_id for class_id, name in stage.classes},
        "excluded_coco_class_ids": sorted(stage.excluded_ids),
        "selection": {
            "train_images_per_class_minimum": train_per_class,
            "val_images_per_class_minimum": val_per_class,
            "seed": seed,
            "cross_stage_target_classes_excluded": True,
            "class_area_rules": {
                name: CLASS_RULES[class_id] for class_id, name in stage.classes
            },
        },
        "source": "https://cocodataset.org/",
        "terms": COCO_TERMS,
        "paper": COCO_PAPER,
        "labels_source": LABELS_URL,
        "image_count": {
            split: len(paths) for split, paths in selected_by_split.items()
        },
    }
    (output_root / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and build a lightweight real COCO subset for NestDetect"
    )
    parser.add_argument("--train-per-class", type=int, default=60)
    parser.add_argument("--val-per-class", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--labels-root",
        type=Path,
        help="Existing folder containing train2017/ and val2017/ YOLO labels",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "nestdetect-coco-cache",
    )
    args = parser.parse_args()
    if args.train_per_class < 1 or args.val_per_class < 1:
        raise ValueError("The number of samples per class must be at least 1")
    project_root = Path(__file__).resolve().parent.parent
    labels_root = prepare_labels(args.cache_dir, args.labels_root)
    summaries = []
    for index, stage in enumerate(STAGES):
        summaries.append(
            materialize_stage(
                project_root,
                labels_root,
                args.cache_dir,
                stage,
                args.train_per_class,
                args.val_per_class,
                args.seed + index * 100,
                args.workers,
            )
        )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
