from __future__ import annotations

import torch
from torch import nn

from nestdetect.conditional_memory import (
    ConditionalLowRankDetectionMemory,
    LowRankTensorDelta,
)


def test_low_rank_tensor_delta_reconstructs_exact_rank_two_matrix() -> None:
    left = torch.randn(8, 2)
    right = torch.randn(2, 6)
    delta = left @ right
    compressed = LowRankTensorDelta(
        delta,
        max_rank=2,
        minimum_energy=1.0,
    )
    reconstructed = compressed.reconstruct(delta)
    assert torch.allclose(delta, reconstructed, atol=1e-5, rtol=1e-5)
    assert compressed.stored_elements < delta.numel()


class _ToyConditionalDetector(nn.Module):
    def __init__(self, old_score: float, new_score: float) -> None:
        super().__init__()
        self.scores = nn.Parameter(torch.tensor([old_score, new_score]))
        self.register_buffer("scale", torch.ones(()))
        self.names = {0: "old", 1: "new"}
        self.stride = torch.tensor([32.0])
        self.args = {}
        self.yaml = {}

    def forward(self, inputs: torch.Tensor, **kwargs):
        del kwargs
        batch = inputs.shape[0]
        boxes = torch.zeros(batch, 4, 1, device=inputs.device)
        scores = (self.scores * self.scale)[None, :, None].expand(batch, -1, -1)
        return torch.cat((boxes, scores), dim=1)


def test_conditional_memory_regenerates_and_routes_expert() -> None:
    persistent = _ToyConditionalDetector(0.8, 0.1)
    plastic = _ToyConditionalDetector(0.2, 0.9)
    memory = ConditionalLowRankDetectionMemory(
        persistent,
        plastic,
        plastic_class_ids=(1,),
        max_rank=2,
        minimum_energy=1.0,
    ).eval()
    output = memory(torch.zeros(1, 3, 8, 8))
    assert output.shape == (1, 6, 2)
    assert output[0, 4, 0] == 0.8
    assert output[0, 5, 0] == 0.0
    assert output[0, 4, 1] == 0.0
    assert output[0, 5, 1] == 0.9
