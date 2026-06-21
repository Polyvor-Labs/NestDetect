from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.func import functional_call


class FrozenTensor(nn.Module):
    """A checkpoint-friendly immutable tensor container."""

    def __init__(self, value: Tensor) -> None:
        super().__init__()
        self.register_buffer("value", value.detach().clone())


class LowRankTensorDelta(nn.Module):
    """SVD-compressed parameter delta used as a conditional expert memory."""

    def __init__(
        self,
        delta: Tensor,
        *,
        max_rank: int,
        minimum_energy: float,
    ) -> None:
        super().__init__()
        if max_rank < 1:
            raise ValueError("Low-rank expert max_rank must be positive")
        if not 0.0 < minimum_energy <= 1.0:
            raise ValueError("Low-rank expert minimum_energy must be in (0, 1]")
        self.original_shape = tuple(delta.shape)
        self.mode = "full"
        self.rank = 0
        work = delta.detach().float()
        if torch.count_nonzero(work) == 0:
            self.mode = "zero"
            self.register_buffer("full", torch.empty(0))
            return

        matrix = work.reshape(work.shape[0], -1) if work.ndim >= 2 else None
        if matrix is None or min(matrix.shape) <= 1:
            self.register_buffer("full", work)
            return

        left_vectors, singular_values, right_vectors = torch.linalg.svd(
            matrix,
            full_matrices=False,
        )
        energy = singular_values.square()
        cumulative = energy.cumsum(0)
        target = minimum_energy * energy.sum()
        energy_rank = int(torch.searchsorted(cumulative, target).item()) + 1
        rank = min(max_rank, energy_rank, min(matrix.shape))
        low_rank_storage = rank * (matrix.shape[0] + matrix.shape[1])
        if low_rank_storage >= matrix.numel():
            self.register_buffer("full", work)
            return

        self.mode = "low_rank"
        self.rank = rank
        self.register_buffer(
            "left",
            left_vectors[:, :rank] * singular_values[:rank],
        )
        self.register_buffer("right", right_vectors[:rank])

    def reconstruct(self, reference: Tensor) -> Tensor:
        if self.mode == "zero":
            return torch.zeros_like(reference)
        if self.mode == "full":
            return self.full.to(device=reference.device, dtype=reference.dtype)
        matrix = self.left.to(
            device=reference.device,
            dtype=reference.dtype,
        ) @ self.right.to(device=reference.device, dtype=reference.dtype)
        return matrix.reshape(self.original_shape)

    @property
    def stored_elements(self) -> int:
        if self.mode == "zero":
            return 0
        if self.mode == "full":
            return self.full.numel()
        return self.left.numel() + self.right.numel()


@dataclass(frozen=True)
class ExpertCompressionReport:
    parameter_count: int
    changed_parameter_count: int
    original_delta_elements: int
    stored_delta_elements: int
    relative_l2_error: float

    @property
    def compression_ratio(self) -> float:
        if self.stored_delta_elements == 0:
            return float("inf")
        return self.original_delta_elements / self.stored_delta_elements


class ConditionalLowRankDetectionMemory(nn.Module):
    """Replay-free detector memory with conditionally regenerated parameters.

    The base detector is immutable. A plastic expert is represented as compressed
    parameter deltas and exact non-parametric buffers. At inference, the expert
    parameter realization is generated with ``functional_call`` and class-routed
    against the persistent base prediction.
    """

    def __init__(
        self,
        persistent_model: nn.Module,
        plastic_model: nn.Module,
        plastic_class_ids: Sequence[int],
        *,
        max_rank: int = 16,
        minimum_energy: float = 0.995,
    ) -> None:
        super().__init__()
        self.persistent_model = persistent_model.eval()
        self.names = getattr(persistent_model, "names")
        self.task = "detect"
        self.stride = getattr(persistent_model, "stride", torch.tensor([32.0]))
        self.args = dict(getattr(persistent_model, "args", {}))
        self.yaml = getattr(persistent_model, "yaml", None)
        if getattr(plastic_model, "names", None) != self.names:
            raise ValueError("Conditional detector memories require identical classes")
        for model in (persistent_model, plastic_model):
            for module in model.modules():
                ensure_buffers = getattr(module, "_ensure_persistent_buffers", None)
                if callable(ensure_buffers):
                    ensure_buffers()

        class_count = len(self.names)
        selected = tuple(sorted({int(value) for value in plastic_class_ids}))
        if not selected or selected[0] < 0 or selected[-1] >= class_count:
            raise ValueError("Invalid plastic classes for conditional detector memory")
        mask = torch.zeros(class_count, dtype=torch.bool)
        mask[list(selected)] = True
        self.register_buffer("plastic_class_mask", mask)

        persistent_parameters = dict(persistent_model.named_parameters())
        plastic_parameters = dict(plastic_model.named_parameters())
        if persistent_parameters.keys() != plastic_parameters.keys():
            raise ValueError("Conditional detector memories have different parameters")
        self.expert_parameter_names: tuple[str, ...] = tuple(
            persistent_parameters
        )
        self.expert_deltas = nn.ModuleList()
        squared_error = 0.0
        squared_norm = 0.0
        changed = 0
        original_elements = 0
        stored_elements = 0
        for name in self.expert_parameter_names:
            persistent = persistent_parameters[name].detach().float()
            plastic = plastic_parameters[name].detach().float()
            if persistent.shape != plastic.shape:
                raise ValueError(f"Expert parameter shape mismatch for {name}")
            delta = plastic - persistent
            compressed = LowRankTensorDelta(
                delta,
                max_rank=max_rank,
                minimum_energy=minimum_energy,
            )
            self.expert_deltas.append(compressed)
            reconstruction = compressed.reconstruct(delta)
            squared_error += float((delta - reconstruction).square().sum())
            squared_norm += float(delta.square().sum())
            original_elements += delta.numel()
            stored_elements += compressed.stored_elements
            if compressed.mode != "zero":
                changed += 1

        persistent_buffers = dict(persistent_model.named_buffers())
        plastic_buffers = dict(plastic_model.named_buffers())
        if persistent_buffers.keys() != plastic_buffers.keys():
            raise ValueError("Conditional detector memories have different buffers")
        self.expert_buffer_names: tuple[str, ...] = tuple(plastic_buffers)
        self.expert_buffers = nn.ModuleList(
            [FrozenTensor(plastic_buffers[name]) for name in self.expert_buffer_names]
        )
        attribute_names = (
            "persistent_enabled",
            "persistent_capture",
            "persistent_momentum",
            "consolidated_strength",
            "working_strength",
            "lock_slowest",
        )
        persistent_modules = dict(persistent_model.named_modules())
        plastic_modules = dict(plastic_model.named_modules())
        self.expert_attribute_overrides: tuple[
            tuple[str, str, bool | float], ...
        ] = tuple(
            (module_name, attribute, getattr(plastic_modules[module_name], attribute))
            for module_name, module in persistent_modules.items()
            for attribute in attribute_names
            if hasattr(module, attribute)
            and hasattr(plastic_modules[module_name], attribute)
            and getattr(module, attribute)
            != getattr(plastic_modules[module_name], attribute)
        )
        self.compression_report = ExpertCompressionReport(
            parameter_count=len(self.expert_parameter_names),
            changed_parameter_count=changed,
            original_delta_elements=original_elements,
            stored_delta_elements=stored_elements,
            relative_l2_error=(squared_error / max(squared_norm, 1e-12)) ** 0.5,
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _decoded(prediction: Any) -> Tensor:
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        if not isinstance(prediction, Tensor) or prediction.ndim != 3:
            raise TypeError("Conditional detector returned an unsupported output")
        return prediction

    def _expert_state(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        persistent_parameters = dict(self.persistent_model.named_parameters())
        parameters = {
            name: persistent_parameters[name]
            + delta.reconstruct(persistent_parameters[name])
            for name, delta in zip(
                self.expert_parameter_names,
                self.expert_deltas,
            )
        }
        buffers = {
            name: state.value
            for name, state in zip(
                self.expert_buffer_names,
                self.expert_buffers,
            )
        }
        return parameters, buffers

    @contextmanager
    def _expert_attributes(self):
        modules = dict(self.persistent_model.named_modules())
        previous = []
        try:
            for module_name, attribute, value in self.expert_attribute_overrides:
                module = modules[module_name]
                previous.append((module, attribute, getattr(module, attribute)))
                setattr(module, attribute, value)
            yield
        finally:
            for module, attribute, value in reversed(previous):
                setattr(module, attribute, value)

    def forward(
        self,
        inputs: Tensor,
        profile: bool = False,
        visualize: bool = False,
        augment: bool = False,
        embed: list[int] | None = None,
        **_: Any,
    ) -> Tensor:
        if self.training:
            raise RuntimeError(
                "ConditionalLowRankDetectionMemory is a consolidated inference model"
            )
        prediction_kwargs = {
            "profile": profile,
            "visualize": visualize,
            "augment": augment,
            "embed": embed,
        }
        persistent = self._decoded(
            self.persistent_model(inputs, **prediction_kwargs)
        )
        parameters, buffers = self._expert_state()
        with self._expert_attributes():
            plastic = self._decoded(
                functional_call(
                    self.persistent_model,
                    (parameters, buffers),
                    (inputs,),
                    prediction_kwargs,
                    strict=True,
                )
            )
        class_count = self.plastic_class_mask.numel()
        mask = self.plastic_class_mask[None, :, None]
        persistent_scores = persistent[:, 4:].masked_fill(mask, 0.0)
        plastic_scores = plastic[:, 4:].masked_fill(~mask, 0.0)
        return torch.cat(
            (
                torch.cat((persistent[:, :4], persistent_scores), dim=1),
                torch.cat((plastic[:, :4], plastic_scores), dim=1),
            ),
            dim=2,
        )

    def fuse(self, verbose: bool = True):
        del verbose
        # Fusing would change parameter names and invalidate stored deltas.
        return self
