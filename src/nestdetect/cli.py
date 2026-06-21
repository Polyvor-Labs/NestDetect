from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_config
from .dataset import DatasetSpec, build_incremental_dataset, validate_dataset
from .evaluation import evaluate_experiment
from .inference import detect_to_file
from .memory import build_replay_memory
from .training import train_from_config


def _csv_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nestdetect", description="NestDetect experiment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate a YOLO dataset")
    validate.add_argument("data_yaml")
    validate.add_argument("--json", dest="json_path")

    memory = subparsers.add_parser("build-memory", help="Build balanced replay memory")
    memory.add_argument("data_yaml")
    memory.add_argument("--output", default="memory")
    memory.add_argument("--per-class", type=int, default=30)
    memory.add_argument("--val-per-class", type=int)
    memory.add_argument("--seed", type=int, default=42)

    incremental = subparsers.add_parser(
        "build-incremental",
        help="Combine new-class data with optional replay data",
    )
    incremental.add_argument("new_data_yaml")
    incremental.add_argument("--replay-data-yaml")
    incremental.add_argument("--output", required=True)
    incremental.add_argument("--base-names", default="")
    incremental.add_argument("--target-names", default="")

    train = subparsers.add_parser("train", help="Train from a configuration file")
    train.add_argument("config")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate experimental forgetting",
    )
    evaluate.add_argument("--base-model", required=True)
    evaluate.add_argument("--no-replay-model")
    evaluate.add_argument("--with-replay-model")
    evaluate.add_argument("--old-data", required=True)
    evaluate.add_argument("--new-data", required=True)
    evaluate.add_argument("--old-classes", required=True)
    evaluate.add_argument("--new-classes", required=True)
    evaluate.add_argument("--output", default="results")
    evaluate.add_argument("--imgsz", type=int, default=640)
    evaluate.add_argument("--batch", type=int, default=8)
    evaluate.add_argument("--device")
    evaluate.add_argument("--workers", type=int, default=4)

    detect = subparsers.add_parser("detect", help="Detect objects in one image")
    detect.add_argument("--model", required=True)
    detect.add_argument("--source", required=True)
    detect.add_argument("--output", default="results/detection_outputs/detection.jpg")
    detect.add_argument("--conf", type=float, default=0.25)
    detect.add_argument("--iou", type=float, default=0.7)
    detect.add_argument("--imgsz", type=int, default=640)
    detect.add_argument("--device")
    detect.add_argument(
        "--all-classes",
        action="store_true",
        help="Do not filter predictions to the five NestDetect target classes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        report = validate_dataset(args.data_yaml)
        payload = report.to_dict()
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        if args.json_path:
            Path(args.json_path).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        return 0 if report.valid else 2
    if args.command == "build-memory":
        result = build_replay_memory(
            args.data_yaml,
            args.output,
            train_per_class=args.per_class,
            val_per_class=args.val_per_class,
            seed=args.seed,
        )
        print(result)
        return 0
    if args.command == "build-incremental":
        target_names = _csv_names(args.target_names) or None
        base_names = _csv_names(args.base_names) or None
        result = build_incremental_dataset(
            args.new_data_yaml,
            args.output,
            replay_dataset_yaml=args.replay_data_yaml,
            base_names=base_names,
            target_names=target_names,
        )
        print(result)
        return 0
    if args.command == "train":
        print(train_from_config(load_config(args.config)))
        return 0
    if args.command == "evaluate":
        models = {}
        if args.no_replay_model:
            models["no_replay"] = args.no_replay_model
        if args.with_replay_model:
            models["with_replay"] = args.with_replay_model
        if not models:
            raise ValueError(
                "Provide --no-replay-model and/or --with-replay-model"
            )
        records = evaluate_experiment(
            base_model=args.base_model,
            incremental_models=models,
            old_dataset_yaml=args.old_data,
            new_dataset_yaml=args.new_data,
            old_classes=_csv_names(args.old_classes),
            new_classes=_csv_names(args.new_classes),
            output_dir=args.output,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
        )
        print(json.dumps([record.__dict__ for record in records], indent=2))
        return 0
    if args.command == "detect":
        target_options = {}
        if not args.all_classes:
            target_options = {
                "classes": [0, 56, 60, 63, 73],
                "class_aliases": {"dining table": "table"},
            }
        output, detections = detect_to_file(
            args.model,
            args.source,
            args.output,
            confidence=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            **target_options,
        )
        print(json.dumps({"output": str(output), "detections": detections}, indent=2))
        return 0
    return 1
