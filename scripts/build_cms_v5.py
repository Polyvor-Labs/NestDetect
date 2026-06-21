from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nestdetect.conditional_memory import (  # noqa: E402
    ConditionalLowRankDetectionMemory,
)
from nestdetect.hope import register_ultralytics_modules  # noqa: E402
from nestdetect.io_utils import portable_path, write_json  # noqa: E402
from ultralytics.nn.tasks import load_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a conditionally regenerated low-rank HoPe CMS expert"
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
        default="models/nestdetect_hope_cms_v5.pt",
    )
    parser.add_argument("--plastic-classes", type=int, nargs="+", default=[63, 73])
    parser.add_argument("--max-rank", type=int, default=16)
    parser.add_argument("--minimum-energy", type=float, default=0.995)
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
    memory = ConditionalLowRankDetectionMemory(
        persistent_model=persistent.float(),
        plastic_model=plastic.float(),
        plastic_class_ids=args.plastic_classes,
        max_rank=args.max_rank,
        minimum_energy=args.minimum_energy,
    )
    report = asdict(memory.compression_report)
    report["compression_ratio"] = memory.compression_report.compression_ratio
    memory = memory.half()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_args = dict(persistent_checkpoint.get("train_args", {}))
    train_args["model"] = str(output_path)
    torch.save(
        {
            "model": memory,
            "train_args": train_args,
            "cms_v5": {
                "variant": "conditional_low_rank_parameter_memory",
                "persistent_model": portable_path(persistent_path, ROOT),
                "plastic_model": portable_path(plastic_path, ROOT),
                "plastic_class_ids": list(args.plastic_classes),
                "max_rank": args.max_rank,
                "minimum_energy": args.minimum_energy,
                "compression": report,
                "contains_images": False,
            },
        },
        output_path,
    )
    write_json(
        output_path.with_suffix(".metadata.json"),
        {
            "architecture": "HoPe CMS V5 conditional low-rank parameter memory",
            "persistent_model": portable_path(persistent_path, ROOT),
            "plastic_model": portable_path(plastic_path, ROOT),
            "plastic_class_ids": list(args.plastic_classes),
            "max_rank": args.max_rank,
            "minimum_energy": args.minimum_energy,
            "compression": report,
            "output_model": portable_path(output_path, ROOT),
            "contains_images": False,
        },
    )
    print(output_path)
    print(report)


if __name__ == "__main__":
    main()
