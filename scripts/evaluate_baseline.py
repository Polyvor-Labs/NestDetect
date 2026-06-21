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
    output = ROOT / "results/baseline"
    output.mkdir(parents=True, exist_ok=True)
    common = {
        "imgsz": 640,
        "batch": 8,
        "workers": 0,
        "device": "cpu",
        "work_dir": ROOT / "artifacts/evaluations/baseline",
    }
    old_data = ROOT / "datasets/base_coco80/data.yaml"
    new_data = ROOT / "datasets/new_coco80/data.yaml"
    old_classes = ["person", "chair", "dining table"]
    new_classes = ["laptop", "book"]

    records = [
        evaluate_group(
            ROOT / "artifacts/checkpoints/base_coco80.pt",
            old_data,
            old_classes,
            "base_old",
            output,
            **common,
        ),
        evaluate_group(
            ROOT / "models/nestdetect_final.pt",
            old_data,
            old_classes,
            "replay_old",
            output,
            **common,
        ),
        evaluate_group(
            ROOT / "models/nestdetect_final.pt",
            new_data,
            new_classes,
            "replay_new",
            output,
            **common,
        ),
        evaluate_group(
            ROOT / "artifacts/checkpoints/incremental_no_replay_coco80.pt",
            old_data,
            old_classes,
            "no_replay_old",
            output,
            **common,
        ),
        evaluate_group(
            ROOT / "artifacts/checkpoints/incremental_no_replay_coco80.pt",
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
        "records": [asdict(record) for record in records],
        "forgetting_no_replay": forgetting_no_replay,
        "forgetting_with_replay": forgetting_with_replay,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = [
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "evaluation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "group",
                "classes",
                "precision",
                "recall",
                "map50",
                "map50_95",
            ],
        )
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row.pop("forgetting", None)
            writer.writerow(row)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
