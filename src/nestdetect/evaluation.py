from __future__ import annotations

import csv
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .dataset import materialize_dataset, normalize_names
from .io_utils import ensure_dir, write_json
from .training import require_yolo


def forgetting_score(map_old_before: float, map_old_after: float) -> float:
    return round(float(map_old_before) - float(map_old_after), 10)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def model_names(model: Any) -> list[str]:
    raw = getattr(model, "names", None)
    if raw is None and getattr(model, "model", None) is not None:
        raw = getattr(model.model, "names", None)
    return normalize_names(raw)


@dataclass
class MetricRecord:
    model: str
    group: str
    classes: str
    precision: float
    recall: float
    map50: float
    map50_95: float
    forgetting: float | None = None


def _copy_confusion_matrix(save_dir: Path, output_dir: Path, name: str) -> None:
    source = save_dir / "confusion_matrix.png"
    if source.is_file():
        figure_name = name.replace("_", "-")
        shutil.copy2(source, output_dir / f"confusion-matrix-{figure_name}.png")


def evaluate_group(
    model_path: str | Path,
    source_yaml: str | Path,
    group_names: Sequence[str],
    group: str,
    output_dir: str | Path,
    imgsz: int = 640,
    batch: int = 8,
    device: str | int | None = None,
    workers: int = 4,
    work_dir: str | Path | None = None,
) -> MetricRecord:
    YOLO = require_yolo()
    model_path = Path(model_path).expanduser().resolve()
    output = ensure_dir(Path(output_dir).expanduser().resolve())
    model = YOLO(str(model_path))
    names = model_names(model)
    missing = [name for name in group_names if name not in names]
    if missing:
        raise ValueError(
            f"Checkpoint {model_path.name} does not contain classes: {', '.join(missing)}"
        )
    safe_name = f"{model_path.stem}_{group}".replace(" ", "_")
    generated = ensure_dir(
        Path(work_dir).expanduser().resolve()
        if work_dir is not None
        else output / ".generated"
    )
    view_root = generated / "eval_views" / safe_name
    view_yaml = materialize_dataset(
        source_yaml=source_yaml,
        output_root=view_root,
        target_names=names,
        splits=("val",),
        prefix=group,
    )
    class_ids = [names.index(name) for name in group_names]
    kwargs: dict[str, Any] = {
        "data": str(view_yaml),
        "split": "val",
        "classes": class_ids,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "project": str(generated / "validation"),
        "name": safe_name,
        "exist_ok": True,
        "plots": True,
    }
    if device not in (None, ""):
        kwargs["device"] = device
    metrics = model.val(**kwargs)
    box = metrics.box
    record = MetricRecord(
        model=model_path.name,
        group=group,
        classes=", ".join(group_names),
        precision=_to_float(box.mp),
        recall=_to_float(box.mr),
        map50=_to_float(box.map50),
        map50_95=_to_float(box.map),
    )
    _copy_confusion_matrix(Path(metrics.save_dir), output, group)
    return record


def evaluate_experiment(
    base_model: str | Path,
    incremental_models: Mapping[str, str | Path],
    old_dataset_yaml: str | Path,
    new_dataset_yaml: str | Path,
    old_classes: Sequence[str],
    new_classes: Sequence[str],
    output_dir: str | Path = "results",
    **val_kwargs: Any,
) -> list[MetricRecord]:
    output = ensure_dir(Path(output_dir).expanduser().resolve())
    records: list[MetricRecord] = []
    comparison: list[dict[str, Any]] = []
    base_old = evaluate_group(
        base_model,
        old_dataset_yaml,
        old_classes,
        "old",
        output,
        **val_kwargs,
    )
    records.append(base_old)
    comparison.append(
        {
            "model": base_old.model,
            "strategy": "base",
            "old_map50_95": base_old.map50_95,
            "new_map50_95": None,
            "forgetting": None,
        }
    )
    for label, checkpoint in incremental_models.items():
        old_record = evaluate_group(
            checkpoint,
            old_dataset_yaml,
            old_classes,
            f"old_{label}",
            output,
            **val_kwargs,
        )
        old_record.forgetting = forgetting_score(base_old.map50_95, old_record.map50_95)
        records.append(old_record)
        records.append(
            evaluate_group(
                checkpoint,
                new_dataset_yaml,
                new_classes,
                f"new_{label}",
                output,
                **val_kwargs,
            )
        )
        new_record = records[-1]
        comparison.append(
            {
                "model": old_record.model,
                "strategy": label,
                "old_map50_95": old_record.map50_95,
                "new_map50_95": new_record.map50_95,
                "forgetting": old_record.forgetting,
            }
        )
    _write_evaluation_csv(output / "evaluation.csv", records)
    _write_comparison_csv(output / "comparison.csv", comparison)
    payload = {
        "records": [asdict(record) for record in records],
        "comparison": comparison,
    }
    write_json(output / "metrics.json", payload)
    create_forgetting_plot(records, output / "forgetting.png")
    return records


def _write_evaluation_csv(path: Path, records: Iterable[MetricRecord]) -> None:
    rows = [asdict(record) for record in records]
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
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
                "forgetting",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "strategy",
                "old_map50_95",
                "new_map50_95",
                "forgetting",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def create_forgetting_plot(records: Sequence[MetricRecord], output_path: str | Path) -> None:
    comparable = [record for record in records if record.forgetting is not None]
    if not comparable:
        return
    import matplotlib.pyplot as plt

    labels = [record.model for record in comparable]
    values = [record.forgetting * 100 for record in comparable]
    colors = ["#d95f02" if value > 0 else "#1b9e77" for value in values]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, values, color=colors)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Forgetting (mAP50-95 percentage points)")
    axis.set_title("Catastrophic Forgetting Comparison")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    output = Path(output_path)
    ensure_dir(output.parent)
    figure.savefig(output, dpi=160)
    plt.close(figure)
