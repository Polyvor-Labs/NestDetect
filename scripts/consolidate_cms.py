from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nestdetect.consolidation import build_consolidation_artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress the base task into persistent CMS and importance tensors"
    )
    parser.add_argument(
        "--model",
        default="artifacts/checkpoints/base_hope_coco80.pt",
    )
    parser.add_argument(
        "--data",
        default="datasets/base_coco80/data.yaml",
    )
    parser.add_argument(
        "--output",
        default="artifacts/cms/base_hope_coco80_consolidation.pt",
    )
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--capture-batches", type=int)
    parser.add_argument("--importance-batches", type=int)
    args = parser.parse_args()

    def project_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    output = build_consolidation_artifact(
        model_path=project_path(args.model),
        data_yaml=project_path(args.data),
        output_path=project_path(args.output),
        project_dir=ROOT / "artifacts/runs",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        capture_batches=args.capture_batches,
        importance_batches=args.importance_batches,
    )
    print(output)


if __name__ == "__main__":
    main()
