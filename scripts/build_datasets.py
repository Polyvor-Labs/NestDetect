from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nestdetect.dataset import (
    build_incremental_dataset,
    materialize_dataset_with_id_mapping,
    validate_dataset,
)
from nestdetect.memory import build_replay_memory
from nestdetect.training import require_yolo


def main() -> None:
    YOLO = require_yolo()
    names_by_id = YOLO("yolo11n.pt").names
    names = [names_by_id[index] for index in range(len(names_by_id))]

    base = materialize_dataset_with_id_mapping(
        ROOT / "datasets/step1_base/data.yaml",
        ROOT / "datasets/base_coco80",
        names,
        {0: 0, 1: 56, 2: 60},
        prefix="base",
    )
    new = materialize_dataset_with_id_mapping(
        ROOT / "datasets/step2_new_classes/data.yaml",
        ROOT / "datasets/new_coco80",
        names,
        {0: 63, 1: 73},
        prefix="new",
    )
    baseline_memory = build_replay_memory(
        ROOT / "datasets/step1_base/data.yaml",
        ROOT / "memory/baseline",
        train_per_class=60,
        val_per_class=12,
        seed=42,
    )
    baseline_replay = materialize_dataset_with_id_mapping(
        baseline_memory,
        ROOT / "memory/baseline_coco80",
        names,
        {0: 0, 1: 56, 2: 60},
        prefix="baseline_memory",
    )
    hope_memory = build_replay_memory(
        ROOT / "datasets/step1_base/data.yaml",
        ROOT / "memory/hope",
        train_per_class=30,
        val_per_class=6,
        seed=42,
    )
    hope_replay = materialize_dataset_with_id_mapping(
        hope_memory,
        ROOT / "memory/hope_coco80",
        names,
        {0: 0, 1: 56, 2: 60},
        prefix="hope_memory",
    )
    no_replay = build_incremental_dataset(
        new,
        ROOT / "datasets/incremental_no_replay_coco80",
        target_names=names,
    )
    baseline_with_replay = build_incremental_dataset(
        new,
        ROOT / "datasets/incremental_replay_baseline_coco80",
        replay_dataset_yaml=baseline_replay,
        target_names=names,
    )
    hope_with_replay = build_incremental_dataset(
        new,
        ROOT / "datasets/incremental_replay_hope_coco80",
        replay_dataset_yaml=hope_replay,
        target_names=names,
    )
    datasets = {
        "base": base,
        "new": new,
        "baseline_replay_memory": baseline_replay,
        "hope_replay_memory": hope_replay,
        "no_replay": no_replay,
        "baseline_with_replay": baseline_with_replay,
        "hope_with_replay": hope_with_replay,
    }
    payload = {}
    for key, path in datasets.items():
        report = validate_dataset(path)
        payload[key] = {
            "path": str(path),
            "valid": report.valid,
            "images": report.image_counts,
            "target_instances": {
                name: count
                for name, count in report.instance_counts.items()
                if count > 0
            },
        }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
