from __future__ import annotations

from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model

from .hope import (
    HeadwiseCMSDetect,
    configure_continual_memory,
    initialize_headwise_cms_classes,
    install_headwise_cms,
    load_continual_memory,
    register_ultralytics_modules,
)
from .optim import M3, NestedAdamW


def _raw_detection_outputs(predictions: Any) -> dict[str, Any]:
    if isinstance(predictions, tuple):
        predictions = predictions[1]
    if not isinstance(predictions, dict):
        raise TypeError(
            "Continual distillation requires raw YOLO detection outputs"
        )
    if "one2many" in predictions:
        predictions = predictions["one2many"]
    return predictions


class ContinualDistillationCriterion:
    """Detection loss plus old-task distillation and importance anchoring."""

    def __init__(
        self,
        base_criterion: Any,
        student: nn.Module,
        teacher: nn.Module,
        importance: dict[str, torch.Tensor],
        config: dict[str, Any],
    ) -> None:
        self.base_criterion = base_criterion
        self.student = student
        self.teacher = teacher
        self.old_class_ids = tuple(int(value) for value in config["old_class_ids"])
        self.score_weight = float(config.get("score_weight", 0.5))
        self.feature_weight = float(config.get("feature_weight", 1.0))
        self.regularization_weight = float(
            config.get("regularization_weight", 0.05)
        )
        self.temperature = float(config.get("temperature", 2.0))
        if self.temperature <= 0:
            raise ValueError("Distillation temperature must be positive")

        teacher_parameters = dict(teacher.named_parameters())
        self.anchors: dict[str, torch.Tensor] = {}
        self.importance: dict[str, torch.Tensor] = {}
        for name, parameter in student.named_parameters():
            if not parameter.requires_grad or name not in importance:
                continue
            if name not in teacher_parameters:
                continue
            anchor = teacher_parameters[name].detach().to(parameter.device)
            weight = importance[name].detach().to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            if anchor.shape != parameter.shape or weight.shape != parameter.shape:
                raise ValueError(f"Consolidation shape mismatch for parameter {name}")
            self.anchors[name] = anchor
            self.importance[name] = weight
        self.last_components: dict[str, float] = {}

    def _score_distillation(
        self,
        student_raw: dict[str, Any],
        teacher_raw: dict[str, Any],
    ) -> torch.Tensor:
        student_scores = student_raw["scores"][:, self.old_class_ids]
        teacher_scores = teacher_raw["scores"][:, self.old_class_ids]
        temperature = self.temperature
        targets = torch.sigmoid(teacher_scores / temperature)
        return (
            F.binary_cross_entropy_with_logits(
                student_scores / temperature,
                targets,
            )
            * temperature**2
        )

    @staticmethod
    def _feature_distillation(
        student_raw: dict[str, Any],
        teacher_raw: dict[str, Any],
    ) -> torch.Tensor:
        losses = []
        for student_feature, teacher_feature in zip(
            student_raw["feats"],
            teacher_raw["feats"],
        ):
            student_normalized = F.normalize(
                student_feature.flatten(2),
                dim=1,
                eps=1e-6,
            )
            teacher_normalized = F.normalize(
                teacher_feature.flatten(2),
                dim=1,
                eps=1e-6,
            )
            losses.append(
                F.mse_loss(student_normalized, teacher_normalized)
            )
        return torch.stack(losses).mean()

    def _importance_regularization(self) -> torch.Tensor:
        terms = []
        for name, parameter in self.student.named_parameters():
            if name not in self.importance:
                continue
            importance = self.importance[name]
            denominator = importance.sum().clamp_min(1e-8)
            terms.append(
                (importance * (parameter - self.anchors[name]).square()).sum()
                / denominator
            )
        if terms:
            return torch.stack(terms).mean()
        return next(self.student.parameters()).new_zeros(())

    def __call__(
        self,
        predictions: Any,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        detection_loss, loss_items = self.base_criterion(predictions, batch)
        student_raw = _raw_detection_outputs(predictions)
        with torch.no_grad():
            teacher_predictions = self.teacher.predict(batch["img"])
            teacher_raw = _raw_detection_outputs(teacher_predictions)

        score_loss = self._score_distillation(student_raw, teacher_raw)
        feature_loss = self._feature_distillation(student_raw, teacher_raw)
        regularization_loss = self._importance_regularization()
        auxiliary = (
            self.score_weight * score_loss
            + self.feature_weight * feature_loss
            + self.regularization_weight * regularization_loss
        )
        scaled_auxiliary = auxiliary * batch["img"].shape[0]
        detection_loss = detection_loss.clone()
        loss_items = loss_items.clone()
        detection_loss[1] = detection_loss[1] + scaled_auxiliary
        loss_items[1] = loss_items[1] + auxiliary.detach()
        self.last_components = {
            "distill_scores": float(score_loss.detach()),
            "distill_features": float(feature_loss.detach()),
            "importance_regularization": float(regularization_loss.detach()),
            "auxiliary_total": float(auxiliary.detach()),
        }
        return detection_loss, loss_items


class HopeDetectionTrainer(DetectionTrainer):
    """Ultralytics detection trainer with HoPe multi-timescale optimizers."""

    def __init__(
        self,
        *args: Any,
        optimizer_config: dict[str, Any] | None = None,
        continual_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.hope_optimizer_config = dict(optimizer_config or {})
        self.continual_config = dict(continual_config or {})
        self.teacher_model: nn.Module | None = None
        super().__init__(*args, **kwargs)

    def setup_model(self):
        register_ultralytics_modules()
        checkpoint = super().setup_model()
        model = unwrap_model(self.model)
        distillation = dict(self.continual_config.get("distillation") or {})
        headwise_cms = dict(self.continual_config.get("headwise_cms") or {})
        if bool(headwise_cms.get("enabled", False)):
            if bool(distillation.get("freeze_shared_parameters", False)):
                raise ValueError(
                    "headwise_cms and strict classifier isolation are mutually exclusive"
                )
            head = install_headwise_cms(
                model,
                class_ids=headwise_cms.get("plastic_class_ids", ()),
                update_periods=headwise_cms.get(
                    "update_periods",
                    (16, 4, 1),
                ),
            )
            initialization_model = headwise_cms.get("initialization_model")
            if initialization_model:
                source, _ = load_checkpoint(
                    str(initialization_model),
                    device="cpu",
                )
                initialize_headwise_cms_classes(head, source.float())
            self._isolate_headwise_cms(model, head)
            LOGGER.info(
                "continual: head-wise CMS enabled, plastic classes=%s, "
                "update periods=%s",
                torch.nonzero(head.plastic_class_mask).flatten().tolist(),
                list(head.plastic_update_periods),
            )
        if bool(distillation.get("freeze_shared_parameters", False)):
            self._isolate_classifier_parameters(model)
        memory_config = self.continual_config.get("persistent_memory")
        if memory_config:
            configure_continual_memory(model, memory_config)
            state_path = self.continual_config.get("consolidation_artifact")
            if state_path:
                load_continual_memory(model, state_path)
        return checkpoint

    def _model_train(self) -> None:
        super()._model_train()
        distillation = dict(self.continual_config.get("distillation") or {})
        if bool(distillation.get("freeze_shared_parameters", False)):
            for module in unwrap_model(self.model).modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        headwise_cms = dict(self.continual_config.get("headwise_cms") or {})
        if bool(headwise_cms.get("enabled", False)):
            for module in unwrap_model(self.model).modules():
                parameters = tuple(module.parameters(recurse=True))
                if parameters and all(
                    bool(getattr(parameter, "_hope_frozen", False))
                    for parameter in parameters
                ):
                    module.eval()

    def _setup_train(self) -> None:
        super()._setup_train()
        model = unwrap_model(self.model)
        for parameter in model.parameters():
            if bool(getattr(parameter, "_hope_frozen", False)):
                parameter.requires_grad_(False)

        distillation = dict(self.continual_config.get("distillation") or {})
        if not distillation:
            return
        protected_class_ids = tuple(
            int(value)
            for value in distillation.get("protect_class_ids", ())
        )
        if protected_class_ids:
            protected_parameters = self._protect_classifier_channels(
                model,
                protected_class_ids,
            )
            LOGGER.info(
                "continual: protected classifier channels=%s across %d parameters",
                list(protected_class_ids),
                protected_parameters,
            )
        trainable_class_ids = tuple(
            int(value)
            for value in distillation.get("trainable_class_ids", ())
        )
        if trainable_class_ids:
            trainable_parameters = self._restrict_classifier_channels(
                model,
                trainable_class_ids,
            )
            LOGGER.info(
                "continual: trainable classifier channels=%s across %d parameters",
                list(trainable_class_ids),
                trainable_parameters,
            )
        teacher_path = self.continual_config.get("teacher")
        artifact_path = self.continual_config.get("consolidation_artifact")
        if not teacher_path or not artifact_path:
            raise ValueError(
                "Continual training requires teacher and consolidation_artifact"
            )
        teacher, _ = load_checkpoint(str(teacher_path), device=self.device)
        teacher = teacher.float().to(self.device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher_model = teacher

        artifact = torch.load(
            Path(artifact_path),
            map_location="cpu",
            weights_only=False,
        )
        importance = artifact.get("importance")
        if not isinstance(importance, dict):
            raise ValueError(
                "Consolidation artifact does not contain parameter importance"
            )
        if getattr(model, "criterion", None) is None:
            model.criterion = model.init_criterion()
        model.criterion = ContinualDistillationCriterion(
            base_criterion=model.criterion,
            student=model,
            teacher=teacher,
            importance=importance,
            config=distillation,
        )
        LOGGER.info(
            "continual: persistent CMS enabled, teacher=%s, old classes=%s, "
            "importance parameters=%d",
            Path(teacher_path).name,
            distillation.get("old_class_ids"),
            len(importance),
        )

    @staticmethod
    def _protect_classifier_channels(
        model: nn.Module,
        class_ids: tuple[int, ...],
    ) -> int:
        """Mask gradients for old-class rows in every YOLO classification head."""

        if not class_ids:
            return 0
        head = getattr(model, "model", [None])[-1]
        classification_branches = getattr(head, "cv3", None)
        class_count = int(getattr(head, "nc", 0))
        if classification_branches is None or class_count < 1:
            raise TypeError("Unable to locate YOLO classification branches")
        if min(class_ids) < 0 or max(class_ids) >= class_count:
            raise ValueError("Protected class id is outside the detector class range")

        registered = 0
        for branch in classification_branches:
            output = branch[-1]
            if not isinstance(output, nn.Conv2d) or output.out_channels != class_count:
                raise TypeError("Unexpected YOLO classification output layer")
            for parameter in (output.weight, output.bias):
                if parameter is None or not parameter.requires_grad:
                    continue
                mask = torch.ones_like(parameter)
                mask[list(class_ids)] = 0
                parameter.register_hook(
                    lambda gradient, gradient_mask=mask: gradient * gradient_mask
                )
                registered += 1
        return registered

    @staticmethod
    def _classification_outputs(model: nn.Module) -> list[nn.Conv2d]:
        head = getattr(model, "model", [None])[-1]
        branches = getattr(head, "cv3", None)
        class_count = int(getattr(head, "nc", 0))
        if branches is None or class_count < 1:
            raise TypeError("Unable to locate YOLO classification branches")
        outputs = [branch[-1] for branch in branches]
        if any(
            not isinstance(output, nn.Conv2d)
            or output.out_channels != class_count
            for output in outputs
        ):
            raise TypeError("Unexpected YOLO classification output layer")
        return outputs

    @classmethod
    def _isolate_classifier_parameters(cls, model: nn.Module) -> None:
        for parameter in model.parameters():
            parameter._hope_frozen = True  # type: ignore[attr-defined]
        for output in cls._classification_outputs(model):
            output.weight._hope_frozen = False  # type: ignore[attr-defined]
            if output.bias is not None:
                output.bias._hope_frozen = False  # type: ignore[attr-defined]

    @staticmethod
    def _isolate_headwise_cms(
        model: nn.Module,
        head: HeadwiseCMSDetect,
    ) -> None:
        for parameter in model.parameters():
            parameter._hope_frozen = True  # type: ignore[attr-defined]
        for parameter in head.plastic_cv2.parameters():
            parameter._hope_frozen = False  # type: ignore[attr-defined]
        for parameter in head.plastic_cv3.parameters():
            parameter._hope_frozen = False  # type: ignore[attr-defined]

    @classmethod
    def _restrict_classifier_channels(
        cls,
        model: nn.Module,
        class_ids: tuple[int, ...],
    ) -> int:
        outputs = cls._classification_outputs(model)
        class_count = outputs[0].out_channels
        if not class_ids or min(class_ids) < 0 or max(class_ids) >= class_count:
            raise ValueError("Trainable class id is outside the detector class range")
        registered = 0
        for output in outputs:
            for parameter in (output.weight, output.bias):
                if parameter is None or not parameter.requires_grad:
                    continue
                mask = torch.zeros_like(parameter)
                mask[list(class_ids)] = 1
                parameter.register_hook(
                    lambda gradient, gradient_mask=mask: gradient * gradient_mask
                )
                registered += 1
        return registered

    @staticmethod
    def _parameter_groups(model: nn.Module, decay: float) -> list[dict[str, Any]]:
        normalization_types = tuple(
            value for key, value in nn.__dict__.items() if "Norm" in key
        )
        grouped: dict[tuple[str, int], list[nn.Parameter]] = defaultdict(list)
        for module_name, module in unwrap_model(model).named_modules():
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if not parameter.requires_grad:
                    continue
                if bool(getattr(parameter, "_hope_frozen", False)):
                    continue
                full_name = (
                    f"{module_name}.{parameter_name}" if module_name else parameter_name
                )
                if "bias" in full_name:
                    kind = "bias"
                elif isinstance(module, normalization_types):
                    kind = "bn"
                else:
                    kind = "weight"
                update_every = max(
                    1,
                    int(getattr(parameter, "_hope_update_every", 1)),
                )
                grouped[(kind, update_every)].append(parameter)

        groups: list[dict[str, Any]] = []
        for (kind, update_every), parameters in sorted(
            grouped.items(), key=lambda item: (item[0][1], item[0][0])
        ):
            groups.append(
                {
                    "params": parameters,
                    "weight_decay": decay if kind == "weight" else 0.0,
                    "update_every": update_every,
                    "param_group": kind,
                }
            )
        return groups

    def build_optimizer(
        self,
        model,
        name: str = "auto",
        lr: float = 0.001,
        momentum: float = 0.9,
        decay: float = 1e-5,
        iterations: float = 1e5,
    ):
        del name, iterations
        config = self.hope_optimizer_config
        optimizer_name = str(config.get("name", "nested_adamw")).lower()
        groups = self._parameter_groups(model, decay=decay)
        if optimizer_name in {"nested_adamw", "adamw", "nested-adamw"}:
            optimizer = NestedAdamW(
                groups,
                lr=lr,
                betas=(momentum, float(config.get("beta2", 0.999))),
                eps=float(config.get("eps", 1e-8)),
            )
        elif optimizer_name in {"m3", "multi_scale_momentum_muon"}:
            optimizer = M3(
                groups,
                lr=lr,
                beta1=float(config.get("beta1", momentum)),
                beta2=float(config.get("beta2", 0.999)),
                beta3=float(config.get("beta3", 0.99)),
                alpha=float(config.get("alpha", 0.1)),
                eps=float(config.get("eps", 1e-8)),
                slow_frequency=int(config.get("slow_frequency", 8)),
                newton_schulz_steps=int(config.get("newton_schulz_steps", 5)),
            )
        else:
            raise ValueError(
                "HoPe optimizer must be 'nested_adamw' or 'm3'; "
                f"received {config.get('name')!r}"
            )

        frequencies = sorted({int(group["update_every"]) for group in groups})
        LOGGER.info(
            "optimizer: HoPe %s(lr=%g, momentum=%g), CMS update periods=%s",
            optimizer.__class__.__name__,
            lr,
            momentum,
            frequencies,
        )
        return optimizer


def hope_trainer_factory(
    optimizer_config: dict[str, Any] | None,
    continual_config: dict[str, Any] | None = None,
):
    """Create the trainer constructor expected by ``YOLO.train``."""

    return partial(
        HopeDetectionTrainer,
        optimizer_config=dict(optimizer_config or {}),
        continual_config=dict(continual_config or {}),
    )
