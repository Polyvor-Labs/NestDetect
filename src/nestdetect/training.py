from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .config import resolve_project_path, training_arguments
from .dataset import validate_dataset
from .errors import DatasetValidationError, DependencyError
from .hope import register_ultralytics_modules
from .io_utils import ensure_dir, portable_path, write_json


def require_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DependencyError(
            "Ultralytics is not installed. Run: python -m pip install -e ."
        ) from exc
    register_ultralytics_modules()
    return YOLO


def train_model(
    initial_model: str | Path,
    data_yaml: str | Path,
    output_model: str | Path,
    run_name: str,
    project_dir: str | Path = "runs/detect",
    exist_ok: bool = True,
    architecture: str | Path | None = None,
    optimizer_config: dict[str, Any] | None = None,
    continual_config: dict[str, Any] | None = None,
    **train_kwargs: Any,
) -> Path:
    report = validate_dataset(data_yaml)
    if not report.valid:
        raise DatasetValidationError(
            f"The dataset has {report.error_count} errors; fix them before training"
        )
    YOLO = require_yolo()
    output = Path(output_model).expanduser().resolve()
    run_project = Path(project_dir).expanduser().resolve()
    ensure_dir(output.parent)
    if architecture is not None:
        model = YOLO(str(Path(architecture).expanduser().resolve()))
        model.load(str(initial_model))
    else:
        model = YOLO(str(initial_model))
    train_options: dict[str, Any] = {
        "data": str(Path(data_yaml).expanduser().resolve()),
        "project": str(run_project),
        "name": run_name,
        "exist_ok": exist_ok,
        **train_kwargs,
    }
    if optimizer_config is not None or continual_config is not None:
        from .trainer import hope_trainer_factory

        train_options["trainer"] = hope_trainer_factory(
            optimizer_config,
            continual_config,
        )
    results = model.train(**train_options)
    save_dir = Path(getattr(results, "save_dir", run_project / run_name))
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        last = save_dir / "weights" / "last.pt"
        if last.is_file():
            best = last
        else:
            raise FileNotFoundError(f"No training checkpoint was found in {save_dir}")
    shutil.copy2(best, output)
    repository_root = Path(__file__).resolve().parents[2]
    write_json(
        output.with_suffix(".metadata.json"),
        {
            "initial_model": portable_path(initial_model, repository_root),
            "architecture": (
                portable_path(architecture, repository_root)
                if architecture
                else None
            ),
            "dataset": portable_path(data_yaml, repository_root),
            "output_model": portable_path(output, repository_root),
            "run_directory": portable_path(save_dir, repository_root),
            "training": train_kwargs,
            "optimizer": optimizer_config,
            "continual": continual_config,
        },
    )
    return output


def train_from_config(config: dict[str, Any]) -> Path:
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    output_cfg = config.get("output", {})
    initial = model_cfg.get("initial") or model_cfg.get("pretrained")
    architecture = model_cfg.get("architecture")
    data_yaml = dataset_cfg.get("data")
    output_model = output_cfg.get("model")
    if not initial or not data_yaml or not output_model:
        raise ValueError(
            "The config requires model.initial/pretrained, dataset.data, and output.model"
        )
    initial_path: str | Path = initial
    if "/" in str(initial) or "\\" in str(initial):
        initial_path = resolve_project_path(config, initial)
    architecture_path = (
        resolve_project_path(config, architecture) if architecture else None
    )
    continual_config = dict(config.get("continual") or {})
    for key in ("teacher", "consolidation_artifact"):
        if continual_config.get(key):
            continual_config[key] = str(
                resolve_project_path(config, continual_config[key])
            )
    headwise_cms = dict(continual_config.get("headwise_cms") or {})
    if headwise_cms.get("initialization_model"):
        headwise_cms["initialization_model"] = str(
            resolve_project_path(
                config,
                headwise_cms["initialization_model"],
            )
        )
        continual_config["headwise_cms"] = headwise_cms
    return train_model(
        initial_model=initial_path,
        data_yaml=resolve_project_path(config, data_yaml),
        output_model=resolve_project_path(config, output_model),
        run_name=str(output_cfg.get("run_name", Path(str(output_model)).stem)),
        project_dir=resolve_project_path(config, output_cfg.get("runs_dir", "runs/detect")),
        exist_ok=bool(output_cfg.get("exist_ok", True)),
        architecture=architecture_path,
        optimizer_config=config.get("optimizer"),
        continual_config=continual_config or None,
        **training_arguments(config),
    )
