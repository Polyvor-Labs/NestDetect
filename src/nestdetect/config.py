from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = read_yaml(config_path)
    configured_root = Path(str(config.get("project_root", config_path.parent.parent))).expanduser()
    project_root = (
        configured_root.resolve()
        if configured_root.is_absolute()
        else (config_path.parent / configured_root).resolve()
    )
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(project_root)
    return config


def resolve_project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(config["_project_root"]) / path).resolve()


def training_arguments(config: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "epochs",
        "imgsz",
        "batch",
        "device",
        "workers",
        "patience",
        "optimizer",
        "momentum",
        "lr0",
        "lrf",
        "weight_decay",
        "freeze",
        "classes",
        "cache",
        "amp",
        "seed",
        "deterministic",
        "plots",
        "save_period",
        "cos_lr",
        "close_mosaic",
        "warmup_epochs",
        "cls",
    }
    raw = config.get("training", {})
    return {key: value for key, value in raw.items() if key in allowed and value is not None}
