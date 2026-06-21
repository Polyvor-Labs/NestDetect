from __future__ import annotations

import torch

from nestdetect.optim import M3, NestedAdamW


def test_nested_adamw_delays_and_accumulates_updates() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = NestedAdamW(
        [{"params": [parameter], "update_every": 2, "weight_decay": 0.0}],
        lr=0.1,
        betas=(0.0, 0.0),
    )

    parameter.grad = torch.tensor([1.0])
    optimizer.step()
    optimizer.zero_grad()
    assert parameter.item() == 1.0

    parameter.grad = torch.tensor([1.0])
    optimizer.step()
    optimizer.zero_grad()
    assert torch.isclose(parameter.detach(), torch.tensor([0.9])).all()


def test_m3_produces_finite_matrix_and_vector_updates() -> None:
    matrix = torch.nn.Parameter(torch.eye(2))
    vector = torch.nn.Parameter(torch.ones(2))
    optimizer = M3([matrix, vector], lr=1e-4, slow_frequency=2)
    matrix.grad = torch.ones_like(matrix)
    vector.grad = torch.ones_like(vector)
    optimizer.step()
    assert torch.isfinite(matrix).all()
    assert torch.isfinite(vector).all()
