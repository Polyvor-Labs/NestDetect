from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def portable_path(path: str | Path, root: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(Path(root).expanduser().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def clean_generated_dataset(path: str | Path) -> Path:
    root = ensure_dir(path)
    for relative in ("images", "labels"):
        target = root / relative
        if target.exists():
            shutil.rmtree(target)
    for relative in ("data.yaml", "manifest.csv", "memory_index.csv"):
        target = root / relative
        if target.exists():
            target.unlink()
    return root


def read_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML content must be a mapping: {yaml_path}")
    return data


def write_yaml(path: str | Path, data: dict[str, Any]) -> Path:
    output = Path(path)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    return output


def write_json(path: str | Path, data: Any) -> Path:
    output = Path(path)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return output
