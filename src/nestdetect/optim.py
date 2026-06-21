from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


class NestedAdamW(Optimizer):
    """AdamW with gradient accumulation at per-group CMS frequencies.

    Parameters tagged by a CMS level are updated every ``update_every`` outer
    optimizer steps. Gradients are accumulated between updates, matching the
    summed-error update in equation 71 of the HoPe paper.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("betas must be in [0, 1)")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "update_every": 1,
            "frequency_step": 0,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["frequency_step"] = int(group.get("frequency_step", 0)) + 1
            update_every = max(1, int(group.get("update_every", 1)))
            update_now = group["frequency_step"] % update_every == 0
            beta1, beta2 = group["betas"]

            for parameter in group["params"]:
                state = self.state[parameter]
                if parameter.grad is not None:
                    gradient = parameter.grad
                    if gradient.is_sparse:
                        raise RuntimeError("NestedAdamW does not support sparse gradients")
                    if "gradient_accumulator" not in state:
                        state["gradient_accumulator"] = torch.zeros_like(parameter)
                        state["accumulation_count"] = 0
                    state["gradient_accumulator"].add_(gradient)
                    state["accumulation_count"] += 1

                if not update_now or state.get("accumulation_count", 0) == 0:
                    continue

                gradient = state["gradient_accumulator"]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                state["exp_avg"].mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    gradient,
                    gradient,
                    value=1.0 - beta2,
                )

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                denominator = state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)
                denominator.add_(group["eps"])
                step_size = group["lr"] / bias_correction1
                if group["weight_decay"]:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.addcdiv_(state["exp_avg"], denominator, value=-step_size)

                gradient.zero_()
                state["accumulation_count"] = 0
        return loss


def _newton_schulz_orthogonalize(matrix: Tensor, steps: int, eps: float) -> Tensor:
    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz orthogonalization expects a matrix")
    original_dtype = matrix.dtype
    work = matrix.float()
    work = work / (work.norm() + eps)
    transposed = work.shape[0] > work.shape[1]
    if transposed:
        work = work.T
    for _ in range(steps):
        covariance = work @ work.T
        polynomial = -4.7750 * covariance + 2.0315 * covariance @ covariance
        work = 3.4445 * work + polynomial @ work
    if transposed:
        work = work.T
    return work.to(original_dtype)


def _orthogonalized_update(value: Tensor, steps: int, eps: float) -> Tensor:
    matrix = value.reshape(value.shape[0], -1)
    update = _newton_schulz_orthogonalize(matrix, steps=steps, eps=eps)
    update *= max(1.0, matrix.shape[-2] / matrix.shape[-1]) ** 0.5
    return update.reshape_as(value)


class M3(Optimizer):
    """Multi-scale Momentum Muon (M3) from algorithm 1 of the paper.

    The optimizer maintains fast and slow momentum memories, applies
    Newton-Schulz orthogonalization to matrix-like parameters, and uses an
    Adam-style second-moment memory for preconditioning.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        beta3: float = 0.99,
        alpha: float = 0.1,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        slow_frequency: int = 8,
        newton_schulz_steps: int = 5,
    ) -> None:
        if slow_frequency < 1:
            raise ValueError("slow_frequency must be positive")
        if newton_schulz_steps < 1:
            raise ValueError("newton_schulz_steps must be positive")
        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "beta3": beta3,
            "alpha": alpha,
            "eps": eps,
            "weight_decay": weight_decay,
            "slow_frequency": slow_frequency,
            "newton_schulz_steps": newton_schulz_steps,
            "update_every": 1,
            "frequency_step": 0,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["frequency_step"] = int(group.get("frequency_step", 0)) + 1
            update_every = max(1, int(group.get("update_every", 1)))
            update_now = group["frequency_step"] % update_every == 0
            for parameter in group["params"]:
                state = self.state[parameter]
                if parameter.grad is not None:
                    if parameter.grad.is_sparse:
                        raise RuntimeError("M3 does not support sparse gradients")
                    if "gradient_accumulator" not in state:
                        state["gradient_accumulator"] = torch.zeros_like(parameter)
                        state["accumulation_count"] = 0
                    state["gradient_accumulator"].add_(parameter.grad)
                    state["accumulation_count"] += 1

                if not update_now or state.get("accumulation_count", 0) == 0:
                    continue

                gradient = state["gradient_accumulator"]
                if "step" not in state:
                    state["step"] = 0
                    state["fast_momentum"] = torch.zeros_like(parameter)
                    state["slow_momentum"] = torch.zeros_like(parameter)
                    state["slow_accumulator"] = torch.zeros_like(parameter)
                    state["second_moment"] = torch.zeros_like(parameter)
                state["step"] += 1
                state["fast_momentum"].lerp_(gradient, 1.0 - group["beta1"])
                state["slow_accumulator"].add_(gradient)
                state["second_moment"].mul_(group["beta2"]).addcmul_(
                    gradient,
                    gradient,
                    value=1.0 - group["beta2"],
                )

                if state["step"] % group["slow_frequency"] == 0:
                    slow_gradient = state["slow_accumulator"] / group["slow_frequency"]
                    state["slow_momentum"].lerp_(slow_gradient, 1.0 - group["beta3"])
                    state["slow_accumulator"].zero_()

                if parameter.ndim >= 2:
                    fast_update = _orthogonalized_update(
                        state["fast_momentum"],
                        steps=group["newton_schulz_steps"],
                        eps=group["eps"],
                    )
                    slow_update = _orthogonalized_update(
                        state["slow_momentum"],
                        steps=group["newton_schulz_steps"],
                        eps=group["eps"],
                    )
                else:
                    fast_update = state["fast_momentum"]
                    slow_update = state["slow_momentum"]

                denominator = state["second_moment"].sqrt().add_(group["eps"])
                update = (fast_update + group["alpha"] * slow_update) / denominator
                if group["weight_decay"]:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])

                gradient.zero_()
                state["accumulation_count"] = 0
        return loss
