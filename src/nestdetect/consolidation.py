from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

from .hope import (
    configure_continual_memory,
    consolidate_continual_memory,
    export_continual_memory,
    reset_continual_memory,
)
from .io_utils import ensure_dir
from .training import require_yolo


def _build_analysis_trainer(
    model_path: str | Path,
    data_yaml: str | Path,
    project_dir: str | Path,
    imgsz: int,
    batch: int,
    device: str,
) -> DetectionTrainer:
    require_yolo()
    trainer = DetectionTrainer(
        overrides={
            "model": str(Path(model_path).resolve()),
            "data": str(Path(data_yaml).resolve()),
            "project": str(Path(project_dir).resolve()),
            "name": "cms_consolidation",
            "exist_ok": True,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "workers": 0,
            "amp": False,
            "plots": False,
            "save": False,
            "verbose": False,
        }
    )
    trainer.setup_model()
    trainer.model = trainer.model.to(trainer.device)
    trainer.set_model_attributes()
    trainer.stride = max(int(trainer.model.stride.max()), 32)
    return trainer


@torch.no_grad()
def _capture_cms_state(
    trainer: DetectionTrainer,
    max_batches: int | None,
    persistent_config: dict[str, Any],
) -> int:
    model = unwrap_model(trainer.model)
    capture_config = {
        **persistent_config,
        "enabled": True,
        "capture": True,
        "lock_slowest": False,
    }
    configure_continual_memory(model, capture_config)
    reset_continual_memory(model)
    loader = trainer.get_dataloader(
        trainer.data["train"],
        batch_size=trainer.args.batch,
        rank=-1,
        mode="val",
    )
    model.eval()
    seen = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = trainer.preprocess_batch(batch)
        model.predict(batch["img"])
        seen += int(batch["img"].shape[0])
    consolidate_continual_memory(model)
    configure_continual_memory(
        model,
        {
            **persistent_config,
            "enabled": True,
            "capture": False,
            "lock_slowest": True,
        },
    )
    return seen


def _estimate_importance(
    trainer: DetectionTrainer,
    max_batches: int | None,
) -> tuple[dict[str, torch.Tensor], int]:
    model = unwrap_model(trainer.model)
    configure_continual_memory(
        model,
        {"enabled": False, "capture": False, "lock_slowest": False},
    )
    model.train()
    model.criterion = model.init_criterion()
    importance = {
        name: torch.zeros_like(parameter, dtype=torch.float32, device="cpu")
        for name, parameter in model.named_parameters()
        if parameter.dtype.is_floating_point
    }
    loader = trainer.get_dataloader(
        trainer.data["val"],
        batch_size=trainer.args.batch,
        rank=-1,
        mode="val",
    )
    batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = trainer.preprocess_batch(batch)
        model.zero_grad(set_to_none=True)
        loss, _ = model(batch)
        loss.sum().backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and name in importance:
                importance[name].add_(parameter.grad.detach().float().cpu().square())
        batches += 1
    if batches == 0:
        raise RuntimeError("No validation batches were available for consolidation")

    normalized: dict[str, torch.Tensor] = {}
    for name, value in importance.items():
        value.div_(batches).sqrt_()
        positive = value[value > 0]
        if positive.numel() == 0:
            continue
        scale = positive.mean().clamp_min(1e-12)
        normalized[name] = (value / scale).clamp_(max=20.0).half()
    model.zero_grad(set_to_none=True)
    return normalized, batches


def build_consolidation_artifact(
    model_path: str | Path,
    data_yaml: str | Path,
    output_path: str | Path,
    *,
    project_dir: str | Path = "artifacts/runs",
    imgsz: int = 480,
    batch: int = 4,
    device: str = "cpu",
    capture_batches: int | None = None,
    importance_batches: int | None = None,
    persistent_momentum: float = 0.98,
) -> Path:
    """Compress an old task into CMS state and gradient importance tensors.

    Images are consumed only during this consolidation pass. The resulting
    artifact stores tensors and metadata, never source images or annotations.
    """

    trainer = _build_analysis_trainer(
        model_path=model_path,
        data_yaml=data_yaml,
        project_dir=project_dir,
        imgsz=imgsz,
        batch=batch,
        device=device,
    )
    persistent_config = {
        "momentum": persistent_momentum,
        "consolidated_strength": 0.10,
        "working_strength": 0.05,
    }
    captured_images = _capture_cms_state(
        trainer,
        max_batches=capture_batches,
        persistent_config=persistent_config,
    )
    cms_state = export_continual_memory(unwrap_model(trainer.model))
    importance, fisher_batches = _estimate_importance(
        trainer,
        max_batches=importance_batches,
    )
    output = Path(output_path).expanduser().resolve()
    ensure_dir(output.parent)
    torch.save(
        {
            "format": "nestdetect-hope-cms-consolidation-v1",
            "source_model": str(Path(model_path).resolve()),
            "source_dataset": str(Path(data_yaml).resolve()),
            "imgsz": int(imgsz),
            "captured_images": captured_images,
            "importance_batches": fisher_batches,
            "contains_images": False,
            "persistent_config": persistent_config,
            "cms_state": cms_state,
            "importance": importance,
        },
        output,
    )
    return output
