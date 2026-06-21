from __future__ import annotations

from pathlib import Path

import torch

from nestdetect.hope import (
    ClassRoutedDetectionEnsemble,
    HeadwiseCMSDetect,
    Hope2D,
    configure_continual_memory,
    export_continual_memory,
    install_headwise_cms,
    load_continual_memory,
)
from nestdetect.trainer import HopeDetectionTrainer
from nestdetect.training import require_yolo


def test_hope2d_preserves_shape_and_backpropagates() -> None:
    module = Hope2D(
        channels=16,
        memory_dimension=8,
        hidden_dimension=12,
        memory_chunk_size=4,
        projection_chunk_size=8,
        cms_hidden_dimension=12,
        cms_update_frequencies=(1, 2, 4),
    )
    inputs = torch.randn(2, 16, 4, 5, requires_grad=True)
    output = module(inputs)
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()

    output.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert any(parameter.grad is not None for parameter in module.titans.parameters())


def test_hope_fast_state_resets_for_each_forward() -> None:
    torch.manual_seed(7)
    module = Hope2D(
        channels=8,
        memory_dimension=8,
        hidden_dimension=8,
        memory_chunk_size=2,
        projection_chunk_size=4,
        cms_hidden_dimension=8,
        cms_update_frequencies=(1, 3),
    ).eval()
    inputs = torch.randn(1, 8, 3, 3)
    with torch.no_grad():
        first = module(inputs)
        second = module(inputs)
    assert torch.equal(first, second)


def test_cms_parameters_keep_their_update_periods() -> None:
    module = Hope2D(
        channels=8,
        memory_dimension=8,
        hidden_dimension=8,
        cms_hidden_dimension=8,
        cms_update_frequencies=(1, 4, 16),
    )
    periods = {
        int(getattr(parameter, "_hope_update_every", 1))
        for parameter in module.cms.parameters()
    }
    assert periods == {1, 4, 16}


def test_persistent_cms_survives_batches_and_can_be_exported() -> None:
    torch.manual_seed(11)
    module = Hope2D(
        channels=8,
        memory_dimension=8,
        hidden_dimension=8,
        memory_chunk_size=2,
        projection_chunk_size=4,
        cms_hidden_dimension=8,
        cms_update_frequencies=(1, 3),
    ).train()
    configure_continual_memory(
        module,
        {
            "enabled": True,
            "momentum": 0.5,
            "consolidated_strength": 0.2,
            "working_strength": 0.2,
            "lock_slowest": True,
        },
    )
    inputs = torch.randn(1, 8, 3, 3)
    first = module(inputs)
    second = module(inputs)
    assert not torch.equal(first, second)

    exported = {"cms_state": export_continual_memory(module)}
    restored = Hope2D(
        channels=8,
        memory_dimension=8,
        hidden_dimension=8,
        memory_chunk_size=2,
        projection_chunk_size=4,
        cms_hidden_dimension=8,
        cms_update_frequencies=(1, 3),
    )
    load_continual_memory(restored, exported)
    restored_state = export_continual_memory(restored)
    for name, state in exported["cms_state"].items():
        for key, value in state.items():
            assert torch.equal(value, restored_state[name][key])


def test_locked_slow_cms_is_excluded_from_optimizer_groups() -> None:
    module = Hope2D(
        channels=8,
        memory_dimension=8,
        hidden_dimension=8,
        cms_hidden_dimension=8,
        cms_update_frequencies=(1, 4, 16),
    )
    configure_continual_memory(module, {"enabled": True, "lock_slowest": True})
    groups = HopeDetectionTrainer._parameter_groups(module, decay=0.01)
    grouped_ids = {
        id(parameter)
        for group in groups
        for parameter in group["params"]
    }
    assert all(
        id(parameter) not in grouped_ids
        for parameter in module.cms.levels[-1].parameters()
    )


def test_ultralytics_builds_hope_model_yaml() -> None:
    root = Path(__file__).resolve().parent.parent
    YOLO = require_yolo()
    model = YOLO(root / "configs/models/yolo11n-hope.yaml")
    assert isinstance(model.model.model[10], Hope2D)


def test_headwise_cms_preserves_old_logits_and_updates_new_logits() -> None:
    root = Path(__file__).resolve().parent.parent
    YOLO = require_yolo()
    model = YOLO(root / "configs/models/yolo11n-hope.yaml").model
    head = install_headwise_cms(
        model,
        class_ids=(63, 73),
        update_periods=(8, 4, 1),
    )
    assert isinstance(head, HeadwiseCMSDetect)

    channels = [
        branch[0][0].conv.in_channels
        for branch in head.cv3
    ]
    features = [
        torch.randn(2, channel, size, size)
        for channel, size in zip(channels, (8, 4, 2))
    ]
    head.eval()
    before = head.forward_head(
        features,
        box_head=head.cv2,
        cls_head=head.cv3,
    )["scores"].detach()
    optimizer = torch.optim.SGD(
        [
            *head.plastic_cv2.parameters(),
            *head.plastic_cv3.parameters(),
        ],
        lr=0.01,
    )
    scores = head.forward_head(
        features,
        box_head=head.cv2,
        cls_head=head.cv3,
    )["scores"]
    scores[:, [63, 73]].mean().backward()
    optimizer.step()
    after = head.forward_head(
        features,
        box_head=head.cv2,
        cls_head=head.cv3,
    )["scores"].detach()

    assert torch.equal(before[:, [0, 56, 60]], after[:, [0, 56, 60]])
    assert not torch.equal(before[:, [63, 73]], after[:, [63, 73]])
    periods = {
        int(getattr(parameter, "_hope_update_every", 1))
        for branches in (head.plastic_cv2, head.plastic_cv3)
        for parameter in branches.parameters()
    }
    assert periods == {1, 4, 8}


class _ToyRoutedDetector(torch.nn.Module):
    def __init__(self, scores: torch.Tensor) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.names = {0: "old", 1: "new"}
        self.stride = torch.tensor([32.0])
        self.args = {}
        self.scores = scores

    def predict(self, inputs: torch.Tensor, **kwargs):
        del inputs, kwargs
        boxes = torch.zeros(self.scores.shape[0], 4, self.scores.shape[2])
        return torch.cat((boxes, self.scores), dim=1)


def test_class_routed_ensemble_selects_each_memory_by_class() -> None:
    persistent_scores = torch.tensor([[[0.8], [0.1]]])
    plastic_scores = torch.tensor([[[0.2], [0.9]]])
    ensemble = ClassRoutedDetectionEnsemble(
        _ToyRoutedDetector(persistent_scores),
        _ToyRoutedDetector(plastic_scores),
        plastic_class_ids=(1,),
    ).eval()
    output = ensemble(torch.zeros(1, 3, 8, 8))
    assert output.shape == (1, 6, 2)
    assert output[0, 4, 0] == 0.8
    assert output[0, 5, 0] == 0.0
    assert output[0, 4, 1] == 0.0
    assert output[0, 5, 1] == 0.9


def test_class_routed_ensemble_fuses_new_class_candidates() -> None:
    persistent_scores = torch.tensor([[[0.8], [0.6]]])
    plastic_scores = torch.tensor([[[0.2], [0.9]]])
    ensemble = ClassRoutedDetectionEnsemble(
        _ToyRoutedDetector(persistent_scores),
        _ToyRoutedDetector(plastic_scores),
        plastic_class_ids=(1,),
        retain_persistent_plastic_classes=True,
        plastic_score_scale=0.5,
    ).eval()
    output = ensemble(torch.zeros(1, 3, 8, 8))
    assert output[0, 5, 0] == 0.6
    assert output[0, 5, 1] == 0.45
    assert output[0, 4, 1] == 0.0


def test_legacy_class_routed_ensemble_uses_v4_defaults() -> None:
    persistent_scores = torch.tensor([[[0.8], [0.6]]])
    plastic_scores = torch.tensor([[[0.2], [0.9]]])
    ensemble = ClassRoutedDetectionEnsemble(
        _ToyRoutedDetector(persistent_scores),
        _ToyRoutedDetector(plastic_scores),
        plastic_class_ids=(1,),
    ).eval()
    del ensemble.retain_persistent_plastic_classes
    del ensemble.plastic_score_scale

    output = ensemble(torch.zeros(1, 3, 8, 8))

    assert output[0, 4, 0] == 0.8
    assert output[0, 5, 0] == 0.0
    assert output[0, 4, 1] == 0.0
    assert output[0, 5, 1] == 0.9
