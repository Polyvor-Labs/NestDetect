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
    output = ROOT / "results/cms"
    output.mkdir(parents=True, exist_ok=True)
    common = {
        "imgsz": 640,
        "batch": 4,
        "workers": 0,
        "device": "cpu",
        "work_dir": ROOT / "artifacts/evaluations/cms",
    }
    old_data = ROOT / "datasets/base_coco80/data.yaml"
    new_data = ROOT / "datasets/new_coco80/data.yaml"
    old_classes = ["person", "chair", "dining table"]
    new_classes = ["laptop", "book"]
    models = {
        "base": ROOT / "artifacts/checkpoints/base_hope_coco80.pt",
        "no_replay": (
            ROOT / "artifacts/checkpoints/incremental_hope_no_replay_coco80.pt"
        ),
        "cms_v1": ROOT / "models/nestdetect_hope_cms_only.pt",
        "cms_v2": ROOT / "models/nestdetect_hope_cms_only_v2.pt",
        "cms_v3": ROOT / "models/nestdetect_hope_cms_only_v3.pt",
        "cms_v4": ROOT / "models/nestdetect_hope_cms_only_v4.pt",
        "cms_v5": ROOT / "models/nestdetect_hope_cms_v5.pt",
        "replay_fusion": ROOT / "models/nestdetect_hope_cms_v4_replay_fusion.pt",
        "replay": ROOT / "models/nestdetect_hope.pt",
    }

    records = []
    for strategy, model in models.items():
        records.append(
            evaluate_group(
                model,
                old_data,
                old_classes,
                f"{strategy}_old",
                output,
                **common,
            )
        )
        if strategy != "base":
            records.append(
                evaluate_group(
                    model,
                    new_data,
                    new_classes,
                    f"{strategy}_new",
                    output,
                    **common,
                )
            )
    by_group = {record.group: record for record in records}
    base_old = by_group["base_old"].map50_95
    comparison = []
    for strategy in models:
        old_record = by_group[f"{strategy}_old"]
        new_record = by_group.get(f"{strategy}_new")
        comparison.append(
            {
                "strategy": strategy,
                "model": old_record.model,
                "old_map50_95": old_record.map50_95,
                "new_map50_95": (
                    new_record.map50_95 if new_record is not None else ""
                ),
                "forgetting": (
                    forgetting_score(base_old, old_record.map50_95)
                    if strategy != "base"
                    else ""
                ),
            }
        )
    payload = {
        "architecture": "YOLO11n-HoPe CMS ablation study",
        "records": [asdict(record) for record in records],
        "comparison": comparison,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "evaluation.csv").open("w", encoding="utf-8", newline="") as file:
        rows = [asdict(record) for record in records]
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
