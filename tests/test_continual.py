from __future__ import annotations

import torch
from torch import nn

from nestdetect.hope import install_headwise_cms
from nestdetect.trainer import ContinualDistillationCriterion, HopeDetectionTrainer
from nestdetect.training import require_yolo


class ToyDetector(nn.Module):
    def __init__(self, predictions: dict[str, object]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.predictions = predictions

    def predict(self, images: torch.Tensor):
        del images
        return self.predictions


def test_continual_criterion_adds_distillation_and_regularization() -> None:
    teacher_predictions = {
        "scores": torch.zeros(1, 3, 4),
        "feats": [torch.ones(1, 2, 2, 2)],
    }
    student_predictions = {
        "scores": torch.ones(1, 3, 4, requires_grad=True),
        "feats": [torch.zeros(1, 2, 2, 2, requires_grad=True)],
    }
    student = ToyDetector(student_predictions)
    teacher = ToyDetector(teacher_predictions)
    teacher.weight.data.zero_()

    def base_criterion(predictions, batch):
        del predictions, batch
        return torch.zeros(3), torch.zeros(3)

    criterion = ContinualDistillationCriterion(
        base_criterion=base_criterion,
        student=student,
        teacher=teacher,
        importance={"weight": torch.ones(1)},
        config={
            "old_class_ids": [0, 1],
            "score_weight": 1.0,
            "feature_weight": 1.0,
            "regularization_weight": 1.0,
        },
    )
    loss, items = criterion(student_predictions, {"img": torch.zeros(1, 3, 8, 8)})
    assert loss[1] > 0
    assert items[1] > 0
    loss.sum().backward()
    assert student.weight.grad is not None
    assert student_predictions["scores"].grad is not None


def test_old_classifier_channels_are_gradient_protected() -> None:
    YOLO = require_yolo()
    model = YOLO("configs/models/yolo11n-hope.yaml").model
    count = HopeDetectionTrainer._protect_classifier_channels(model, (0, 56, 60))
    assert count == 6
    output = model.model[-1].cv3[0][-1]
    output.weight.sum().backward()
    assert torch.count_nonzero(output.weight.grad[[0, 56, 60]]) == 0
    assert torch.count_nonzero(output.weight.grad[63]) > 0


def test_classifier_isolation_only_trains_selected_rows() -> None:
    YOLO = require_yolo()
    model = YOLO("configs/models/yolo11n-hope.yaml").model
    HopeDetectionTrainer._isolate_classifier_parameters(model)
    groups = HopeDetectionTrainer._parameter_groups(model, decay=0.0)
    grouped = [parameter for group in groups for parameter in group["params"]]
    assert len(grouped) == 6
    HopeDetectionTrainer._restrict_classifier_channels(model, (63, 73))
    output = model.model[-1].cv3[0][-1]
    output.weight.sum().backward()
    assert torch.count_nonzero(output.weight.grad[0]) == 0
    assert torch.count_nonzero(output.weight.grad[63]) > 0
    assert torch.count_nonzero(output.weight.grad[73]) > 0


def test_headwise_cms_optimizer_only_contains_plastic_branch() -> None:
    YOLO = require_yolo()
    model = YOLO("configs/models/yolo11n-hope.yaml").model
    head = install_headwise_cms(model, (63, 73), (16, 4, 1))
    HopeDetectionTrainer._isolate_headwise_cms(model, head)
    groups = HopeDetectionTrainer._parameter_groups(model, decay=0.01)
    grouped_ids = {
        id(parameter)
        for group in groups
        for parameter in group["params"]
    }
    assert grouped_ids == {
        id(parameter)
        for branches in (head.plastic_cv2, head.plastic_cv3)
        for parameter in branches.parameters()
    }
    assert {group["update_every"] for group in groups} == {1, 4, 16}
