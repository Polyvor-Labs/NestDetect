from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nestdetect.evaluation import evaluate_group, forgetting_score


def main() -> None:
    output = ROOT / "results/hope"
    output.mkdir(parents=True, exist_ok=True)
    common = {
        "imgsz": 640,
        "batch": 4,
        "workers": 0,
        "device": "cpu",
        "work_dir": ROOT / "artifacts/evaluations/hope",
    }
    old_data = ROOT / "datasets/base_coco80/data.yaml"
    new_data = ROOT / "datasets/new_coco80/data.yaml"
    old_classes = ["person", "chair", "dining table"]
    new_classes = ["laptop", "book"]
    base_model = ROOT / "artifacts/checkpoints/base_hope_coco80.pt"
    replay_model = ROOT / "models/nestdetect_hope.pt"
    no_replay_model = ROOT / "artifacts/checkpoints/incremental_hope_no_replay_coco80.pt"

    records = [
        evaluate_group(
            base_model,
            old_data,
            old_classes,
            "base_old",
            output,
            **common,
        ),
        evaluate_group(
            replay_model,
            old_data,
            old_classes,
            "replay_old",
            output,
            **common,
        ),
        evaluate_group(
            replay_model,
            new_data,
            new_classes,
            "replay_new",
            output,
            **common,
        ),
        evaluate_group(
            no_replay_model,
            old_data,
            old_classes,
            "no_replay_old",
            output,
            **common,
        ),
        evaluate_group(
            no_replay_model,
            new_data,
            new_classes,
            "no_replay_new",
            output,
            **common,
        ),
    ]
    by_group = {record.group: record for record in records}
    forgetting_with_replay = forgetting_score(
        by_group["base_old"].map50_95,
        by_group["replay_old"].map50_95,
    )
    forgetting_no_replay = forgetting_score(
        by_group["base_old"].map50_95,
        by_group["no_replay_old"].map50_95,
    )
    payload = {
        "architecture": "YOLO11n-HoPe",
        "records": [asdict(record) for record in records],
        "forgetting_no_replay": forgetting_no_replay,
        "forgetting_with_replay": forgetting_with_replay,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    comparison = [
        {
            "model": by_group["base_old"].model,
            "strategy": "base",
            "old_map50_95": by_group["base_old"].map50_95,
            "new_map50_95": "",
            "forgetting": "",
        },
        {
            "model": by_group["no_replay_old"].model,
            "strategy": "no_replay",
            "old_map50_95": by_group["no_replay_old"].map50_95,
            "new_map50_95": by_group["no_replay_new"].map50_95,
            "forgetting": forgetting_no_replay,
        },
        {
            "model": by_group["replay_old"].model,
            "strategy": "replay",
            "old_map50_95": by_group["replay_old"].map50_95,
            "new_map50_95": by_group["replay_new"].map50_95,
            "forgetting": forgetting_with_replay,
        },
    ]
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    with (output / "evaluation.csv").open("w", encoding="utf-8", newline="") as handle:
        rows = [asdict(record) for record in records]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
