from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image


@pytest.fixture
def make_dataset():
    def factory(
        root: Path,
        names: list[str],
        train_labels: list[list[tuple[int, float, float, float, float]]],
        val_labels: list[list[tuple[int, float, float, float, float]]],
    ) -> Path:
        for split, rows in (("train", train_labels), ("val", val_labels)):
            image_dir = root / "images" / split
            label_dir = root / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            for index, labels in enumerate(rows):
                Image.new("RGB", (32, 32), "white").save(image_dir / f"{split}_{index}.jpg")
                text = "\n".join(" ".join(map(str, label)) for label in labels)
                (label_dir / f"{split}_{index}.txt").write_text(
                    text + ("\n" if text else ""), encoding="utf-8"
                )
        data = {
            "path": str(root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {index: name for index, name in enumerate(names)},
        }
        yaml_path = root / "data.yaml"
        yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return yaml_path

    return factory
