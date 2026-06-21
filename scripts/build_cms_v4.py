from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nestdetect.hope import (  # noqa: E402
    ClassRoutedDetectionEnsemble,
    register_ultralytics_modules,
)
from nestdetect.io_utils import portable_path, write_json  # noqa: E402
from ultralytics.nn.tasks import load_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate persistent and plastic HoPe memories into CMS V4"
    )
    parser.add_argument(
        "--persistent",
        default="artifacts/checkpoints/base_hope_coco80.pt",
    )
    parser.add_argument(
        "--plastic",
        default="models/nestdetect_hope_cms_only.pt",
    )
    parser.add_argument(
        "--output",
        default="models/nestdetect_hope_cms_only_v4.pt",
    )
    parser.add_argument(
        "--plastic-classes",
        type=int,
        nargs="+",
        default=[63, 73],
    )
    parser.add_argument(
        "--fuse-persistent-plastic-classes",
        action="store_true",
        help="Keep persistent candidates for plastic classes and fuse by NMS",
    )
    parser.add_argument("--plastic-score-scale", type=float, default=1.0)
    args = parser.parse_args()

    def project_path(value: str) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

    register_ultralytics_modules()
    persistent_path = project_path(args.persistent)
    plastic_path = project_path(args.plastic)
    output_path = project_path(args.output)
    persistent, persistent_checkpoint = load_checkpoint(
        persistent_path,
        device="cpu",
    )
    plastic, _ = load_checkpoint(plastic_path, device="cpu")
    ensemble = ClassRoutedDetectionEnsemble(
        persistent_model=persistent.float(),
        plastic_model=plastic.float(),
        plastic_class_ids=args.plastic_classes,
        retain_persistent_plastic_classes=args.fuse_persistent_plastic_classes,
        plastic_score_scale=args.plastic_score_scale,
    ).half()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_args = dict(persistent_checkpoint.get("train_args", {}))
    train_args["model"] = str(output_path)
    torch.save(
        {
            "model": ensemble,
            "train_args": train_args,
            "cms_v4": {
                "variant": "independent_headwise_model_memory",
                "persistent_model": portable_path(persistent_path, ROOT),
                "plastic_model": portable_path(plastic_path, ROOT),
                "plastic_class_ids": list(args.plastic_classes),
                "fuse_persistent_plastic_classes": (
                    args.fuse_persistent_plastic_classes
                ),
                "plastic_score_scale": args.plastic_score_scale,
                "contains_images": False,
            },
        },
        output_path,
    )
    write_json(
        output_path.with_suffix(".metadata.json"),
        {
            "architecture": "HoPe CMS V4 class-routed model memory",
            "persistent_model": portable_path(persistent_path, ROOT),
            "plastic_model": portable_path(plastic_path, ROOT),
            "plastic_class_ids": list(args.plastic_classes),
            "fuse_persistent_plastic_classes": (
                args.fuse_persistent_plastic_classes
            ),
            "plastic_score_scale": args.plastic_score_scale,
            "output_model": portable_path(output_path, ROOT),
            "contains_images": False,
        },
    )
    print(output_path)


if __name__ == "__main__":
    main()
