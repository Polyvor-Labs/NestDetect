from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ultralytics.nn.modules.head import Detect


def _silu_derivative(value: Tensor) -> Tensor:
    sigmoid = torch.sigmoid(value)
    return sigmoid * (1.0 + value * (1.0 - sigmoid))


def _clip_batched_matrices(value: Tensor, max_norm: float) -> Tensor:
    if max_norm <= 0:
        return value
    norm = value.flatten(1).norm(dim=1).clamp_min(1e-8)
    scale = (max_norm / norm).clamp(max=1.0)
    return value * scale[:, None, None]


@dataclass
class FastMemoryState:
    """Per-sample fast weights for a residual two-layer MLP memory."""

    down: Tensor
    up: Tensor
    down_momentum: Tensor
    up_momentum: Tensor


class ResidualFastMemory(nn.Module):
    """Two-layer residual MLP used as an associative memory.

    The learned parameters are the meta-learned initial state. A fresh fast state
    is created for every input image and modified in-context without mutating the
    checkpoint parameters.
    """

    def __init__(self, dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.hidden_dimension = int(hidden_dimension)
        self.down = nn.Parameter(torch.empty(self.hidden_dimension, self.dimension))
        self.up = nn.Parameter(torch.empty(self.dimension, self.hidden_dimension))
        nn.init.kaiming_uniform_(self.down, a=5**0.5)
        nn.init.normal_(self.up, std=1e-3)

    def initial_state(self, batch_size: int) -> FastMemoryState:
        down = self.down.unsqueeze(0).expand(batch_size, -1, -1)
        up = self.up.unsqueeze(0).expand(batch_size, -1, -1)
        return FastMemoryState(
            down=down,
            up=up,
            down_momentum=torch.zeros_like(down),
            up_momentum=torch.zeros_like(up),
        )

    @staticmethod
    def read_with_intermediates(
        values: Tensor, state: FastMemoryState
    ) -> tuple[Tensor, Tensor, Tensor]:
        pre_activation = torch.einsum("bnd,bhd->bnh", values, state.down)
        hidden = F.silu(pre_activation)
        output = values + torch.einsum("bnh,bdh->bnd", hidden, state.up)
        return output, pre_activation, hidden

    def read(self, values: Tensor, state: FastMemoryState) -> Tensor:
        return self.read_with_intermediates(values, state)[0]

    def dgd_update(
        self,
        state: FastMemoryState,
        keys: Tensor,
        targets: Tensor,
        learning_rate: Tensor,
        retention: Tensor,
        momentum: float,
        delta_decay: float,
        max_update_norm: float,
    ) -> FastMemoryState:
        """Apply the paper's L2/DGD-style in-context memory update.

        DGD contributes a state-dependent covariance decay in addition to the
        ordinary L2 mapping gradient. Layer-local inputs are used for each MLP
        matrix so the update remains well-defined for a nonlinear memory.
        """

        if keys.numel() == 0:
            return state
        predictions, pre_activation, hidden = self.read_with_intermediates(keys, state)
        token_count = max(keys.shape[1], 1)
        error = 2.0 * (predictions - targets) / token_count

        grad_up = torch.einsum("bnd,bnh->bdh", error, hidden)
        hidden_error = torch.einsum("bnd,bdh->bnh", error, state.up)
        hidden_error = hidden_error * _silu_derivative(pre_activation)
        grad_down = torch.einsum("bnh,bnd->bhd", hidden_error, keys)

        key_covariance = torch.einsum("bnd,bne->bde", keys, keys) / token_count
        hidden_covariance = torch.einsum("bnh,bnk->bhk", hidden, hidden) / token_count
        grad_down = grad_down + delta_decay * torch.bmm(state.down, key_covariance)
        grad_up = grad_up + delta_decay * torch.bmm(state.up, hidden_covariance)

        grad_down = _clip_batched_matrices(grad_down, max_update_norm)
        grad_up = _clip_batched_matrices(grad_up, max_update_norm)
        down_momentum = momentum * state.down_momentum + grad_down
        up_momentum = momentum * state.up_momentum + grad_up

        rate = learning_rate.reshape(-1, 1, 1)
        keep = retention.reshape(-1, 1, 1)
        return FastMemoryState(
            down=keep * state.down - rate * down_momentum,
            up=keep * state.up - rate * up_momentum,
            down_momentum=down_momentum,
            up_momentum=up_momentum,
        )


class SelfReferentialTitans(nn.Module):
    """Deep self-referential Titans from equations 83-96 of the paper.

    Spatial features are treated as a sequence. Key, value, query, learning-rate,
    retention, and main memory projections are all residual two-layer memories
    with meta-learned initial states and chunk-wise fast updates.
    """

    MEMORY_NAMES = ("key", "value", "query", "eta", "alpha", "memory")

    def __init__(
        self,
        dimension: int,
        hidden_dimension: int,
        memory_chunk_size: int = 16,
        projection_chunk_size: int = 64,
        inner_learning_rate: float = 0.05,
        minimum_retention: float = 0.95,
        momentum: float = 0.9,
        delta_decay: float = 1.0,
        max_update_norm: float = 5.0,
    ) -> None:
        super().__init__()
        if memory_chunk_size < 1 or projection_chunk_size < 1:
            raise ValueError("HoPe chunk sizes must be positive")
        if projection_chunk_size < memory_chunk_size:
            raise ValueError("projection_chunk_size must be >= memory_chunk_size")
        if not 0.0 <= minimum_retention < 1.0:
            raise ValueError("minimum_retention must be in [0, 1)")
        self.dimension = int(dimension)
        self.memory_chunk_size = int(memory_chunk_size)
        self.projection_chunk_size = int(projection_chunk_size)
        self.inner_learning_rate = float(inner_learning_rate)
        self.minimum_retention = float(minimum_retention)
        self.momentum = float(momentum)
        self.delta_decay = float(delta_decay)
        self.max_update_norm = float(max_update_norm)
        self.memories = nn.ModuleDict(
            {
                name: ResidualFastMemory(self.dimension, hidden_dimension)
                for name in self.MEMORY_NAMES
            }
        )

    def _rates(
        self,
        eta_values: Tensor,
        alpha_values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        eta = self.inner_learning_rate * torch.sigmoid(eta_values.mean(dim=(1, 2)))
        alpha_gate = torch.sigmoid(alpha_values.mean(dim=(1, 2)))
        retention = self.minimum_retention + (1.0 - self.minimum_retention) * alpha_gate
        return eta, retention

    def _update(
        self,
        name: str,
        state: FastMemoryState,
        keys: Tensor,
        targets: Tensor,
        learning_rate: Tensor,
        retention: Tensor,
    ) -> FastMemoryState:
        return self.memories[name].dgd_update(
            state,
            keys,
            targets,
            learning_rate,
            retention,
            momentum=self.momentum,
            delta_decay=self.delta_decay,
            max_update_norm=self.max_update_norm,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"SelfReferentialTitans expects [B, N, D], got {inputs.shape}")
        batch_size, token_count, dimension = inputs.shape
        if dimension != self.dimension:
            raise ValueError(f"Expected feature dimension {self.dimension}, got {dimension}")

        states = {
            name: memory.initial_state(batch_size)
            for name, memory in self.memories.items()
        }
        outputs: list[Tensor] = []
        buffered_keys: list[Tensor] = []
        buffered_targets: dict[str, list[Tensor]] = {
            name: [] for name in self.MEMORY_NAMES if name != "memory"
        }
        buffered_eta: list[Tensor] = []
        buffered_alpha: list[Tensor] = []

        for start in range(0, token_count, self.memory_chunk_size):
            stop = min(start + self.memory_chunk_size, token_count)
            chunk = inputs[:, start:stop]
            projected = {
                name: self.memories[name].read(chunk, states[name])
                for name in ("key", "value", "query", "eta", "alpha")
            }
            keys = F.normalize(projected["key"], dim=-1, eps=1e-6)
            values = projected["value"]
            queries = F.normalize(projected["query"], dim=-1, eps=1e-6)
            outputs.append(self.memories["memory"].read(queries, states["memory"]))

            generated_targets = {
                name: self.memories[name].read(values, states[name])
                for name in self.MEMORY_NAMES
            }
            learning_rate, retention = self._rates(
                projected["eta"],
                projected["alpha"],
            )
            states["memory"] = self._update(
                "memory",
                states["memory"],
                keys,
                generated_targets["memory"],
                learning_rate,
                retention,
            )

            buffered_keys.append(keys)
            buffered_eta.append(projected["eta"])
            buffered_alpha.append(projected["alpha"])
            for name in buffered_targets:
                buffered_targets[name].append(generated_targets[name])

            buffered_count = sum(item.shape[1] for item in buffered_keys)
            is_last = stop == token_count
            if buffered_count < self.projection_chunk_size and not is_last:
                continue

            projection_keys = torch.cat(buffered_keys, dim=1)
            projection_eta = torch.cat(buffered_eta, dim=1)
            projection_alpha = torch.cat(buffered_alpha, dim=1)
            projection_lr, projection_retention = self._rates(
                projection_eta,
                projection_alpha,
            )
            for name, chunks in buffered_targets.items():
                states[name] = self._update(
                    name,
                    states[name],
                    projection_keys,
                    torch.cat(chunks, dim=1),
                    projection_lr,
                    projection_retention,
                )
            buffered_keys.clear()
            buffered_eta.clear()
            buffered_alpha.clear()
            for chunks in buffered_targets.values():
                chunks.clear()

        return torch.cat(outputs, dim=1)


class CMSLevel(nn.Module):
    """One MLP memory level in a sequential Continuum Memory System."""

    def __init__(
        self,
        dimension: int,
        hidden_dimension: int,
        update_every: int,
    ) -> None:
        super().__init__()
        if update_every < 1:
            raise ValueError("CMS update frequency must be at least one")
        self.update_every = int(update_every)
        self.norm = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, hidden_dimension, bias=False)
        self.up = nn.Linear(hidden_dimension, dimension, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.normal_(self.up.weight, std=1e-3)
        for parameter in self.parameters():
            parameter._hope_update_every = self.update_every  # type: ignore[attr-defined]

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = self.norm(inputs)
        return inputs + self.up(F.silu(self.down(normalized)))


class ContinuumMemorySystem(nn.Module):
    """Sequential CMS corresponding to equations 70-73 of the paper."""

    def __init__(
        self,
        dimension: int,
        hidden_dimension: int,
        update_frequencies: Sequence[int] = (1, 4, 16),
    ) -> None:
        super().__init__()
        frequencies = tuple(int(value) for value in update_frequencies)
        if not frequencies:
            raise ValueError("CMS requires at least one memory level")
        if any(right < left for left, right in zip(frequencies, frequencies[1:])):
            raise ValueError("CMS update frequencies must be ordered from fast to slow")
        self.update_frequencies = frequencies
        self.levels = nn.ModuleList(
            [
                CMSLevel(dimension, hidden_dimension, update_every=frequency)
                for frequency in frequencies
            ]
        )

    @property
    def dimension(self) -> int:
        return int(self.levels[0].norm.normalized_shape[0])

    def _ensure_persistent_buffers(self) -> None:
        """Create persistent CMS state lazily for backward-compatible checkpoints."""

        shape = (len(self.levels), self.dimension)
        if "consolidated_state" not in self._buffers:
            self.register_buffer("consolidated_state", torch.zeros(shape))
        if "working_state" not in self._buffers:
            self.register_buffer("working_state", torch.zeros(shape))
        if "persistent_initialized" not in self._buffers:
            self.register_buffer(
                "persistent_initialized",
                torch.zeros(len(self.levels), dtype=torch.bool),
            )
        if not hasattr(self, "persistent_enabled"):
            self.persistent_enabled = False
        if not hasattr(self, "persistent_capture"):
            self.persistent_capture = False
        if not hasattr(self, "persistent_momentum"):
            self.persistent_momentum = 0.98
        if not hasattr(self, "consolidated_strength"):
            self.consolidated_strength = 0.10
        if not hasattr(self, "working_strength"):
            self.working_strength = 0.05
        if not hasattr(self, "lock_slowest"):
            self.lock_slowest = False

    def configure_persistence(
        self,
        *,
        enabled: bool = True,
        capture: bool = False,
        momentum: float = 0.98,
        consolidated_strength: float = 0.10,
        working_strength: float = 0.05,
        lock_slowest: bool = False,
    ) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("CMS persistent momentum must be in [0, 1)")
        self._ensure_persistent_buffers()
        self.persistent_enabled = bool(enabled)
        self.persistent_capture = bool(capture)
        self.persistent_momentum = float(momentum)
        self.consolidated_strength = float(consolidated_strength)
        self.working_strength = float(working_strength)
        self.lock_slowest = bool(lock_slowest)
        for parameter in self.levels[-1].parameters():
            parameter._hope_frozen = self.lock_slowest  # type: ignore[attr-defined]

    def reset_persistent_state(self) -> None:
        self._ensure_persistent_buffers()
        self.consolidated_state.zero_()
        self.working_state.zero_()
        self.persistent_initialized.zero_()

    def consolidate(self) -> None:
        """Promote the current task state into immutable slow memory."""

        self._ensure_persistent_buffers()
        initialized = self.persistent_initialized[:, None]
        self.consolidated_state.copy_(
            torch.where(initialized, self.working_state, self.consolidated_state)
        )

    def persistent_state_dict(self) -> dict[str, Tensor]:
        self._ensure_persistent_buffers()
        return {
            "consolidated_state": self.consolidated_state.detach().cpu().clone(),
            "working_state": self.working_state.detach().cpu().clone(),
            "persistent_initialized": self.persistent_initialized.detach().cpu().clone(),
        }

    def load_persistent_state_dict(self, state: Mapping[str, Tensor]) -> None:
        self._ensure_persistent_buffers()
        for name in (
            "consolidated_state",
            "working_state",
            "persistent_initialized",
        ):
            if name not in state:
                raise KeyError(f"CMS persistent state is missing {name!r}")
            target = getattr(self, name)
            source = state[name].to(device=target.device, dtype=target.dtype)
            if source.shape != target.shape:
                raise ValueError(
                    f"CMS state {name} has shape {source.shape}, expected {target.shape}"
                )
            target.copy_(source)

    def _persistent_context(self, level_index: int, output: Tensor) -> Tensor:
        if not bool(self.persistent_initialized[level_index]):
            return output
        context = (
            self.consolidated_strength * self.consolidated_state[level_index]
            + self.working_strength * self.working_state[level_index]
        )
        return output + context.to(dtype=output.dtype)[None, None, :]

    @torch.no_grad()
    def _update_working_state(self, level_index: int, output: Tensor) -> None:
        if self.lock_slowest and level_index == len(self.levels) - 1 and not self.persistent_capture:
            return
        summary = output.detach().float().mean(dim=(0, 1))
        target = self.working_state[level_index]
        if bool(self.persistent_initialized[level_index]):
            target.mul_(self.persistent_momentum).add_(
                summary.to(target.dtype),
                alpha=1.0 - self.persistent_momentum,
            )
        else:
            target.copy_(summary.to(target.dtype))
            self.persistent_initialized[level_index] = True

    def forward(self, inputs: Tensor) -> Tensor:
        self._ensure_persistent_buffers()
        output = inputs
        should_update = self.persistent_capture or (
            self.training and self.persistent_enabled
        )
        for level_index, level in enumerate(self.levels):
            if self.persistent_enabled:
                output = self._persistent_context(level_index, output)
            output = level(output)
            if should_update:
                self._update_working_state(level_index, output)
        return output


class Hope2D(nn.Module):
    """Vision adaptation of HoPe: self-referential Titans followed by CMS.

    A feature map is converted to spatial tokens, processed by the HoPe sequence
    module, and projected back to the detector with an outer residual connection.
    """

    def __init__(
        self,
        channels: int,
        memory_dimension: int = 64,
        hidden_dimension: int = 128,
        memory_chunk_size: int = 16,
        projection_chunk_size: int = 64,
        cms_hidden_dimension: int = 128,
        cms_update_frequencies: Sequence[int] = (1, 4, 16),
        local_kernel_size: int = 4,
        inner_learning_rate: float = 0.05,
        minimum_retention: float = 0.95,
        momentum: float = 0.9,
        delta_decay: float = 1.0,
    ) -> None:
        super().__init__()
        if channels < 1 or memory_dimension < 1:
            raise ValueError("HoPe channel dimensions must be positive")
        if local_kernel_size < 1:
            raise ValueError("local_kernel_size must be positive")
        self.channels = int(channels)
        self.memory_dimension = int(memory_dimension)
        self.local_kernel_size = int(local_kernel_size)
        self.input_projection = nn.Conv2d(channels, memory_dimension, kernel_size=1)
        self.local_convolution = nn.Conv2d(
            memory_dimension,
            memory_dimension,
            kernel_size=local_kernel_size,
            groups=memory_dimension,
            bias=False,
        )
        self.input_norm = nn.LayerNorm(memory_dimension)
        self.titans = SelfReferentialTitans(
            dimension=memory_dimension,
            hidden_dimension=hidden_dimension,
            memory_chunk_size=memory_chunk_size,
            projection_chunk_size=projection_chunk_size,
            inner_learning_rate=inner_learning_rate,
            minimum_retention=minimum_retention,
            momentum=momentum,
            delta_decay=delta_decay,
        )
        self.cms = ContinuumMemorySystem(
            dimension=memory_dimension,
            hidden_dimension=cms_hidden_dimension,
            update_frequencies=cms_update_frequencies,
        )
        self.output_norm = nn.LayerNorm(memory_dimension)
        self.output_projection = nn.Conv2d(memory_dimension, channels, kernel_size=1)
        nn.init.normal_(self.output_projection.weight, std=1e-3)
        nn.init.zeros_(self.output_projection.bias)

    def _local_mix(self, values: Tensor) -> Tensor:
        total_padding = self.local_kernel_size - 1
        left = total_padding // 2
        right = total_padding - left
        padded = F.pad(values, (left, right, left, right))
        return values + self.local_convolution(padded)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"Hope2D expects [B, C, H, W], got {inputs.shape}")
        batch_size, _, height, width = inputs.shape
        projected = self._local_mix(self.input_projection(inputs))
        tokens = projected.flatten(2).transpose(1, 2)
        tokens = self.titans(self.input_norm(tokens))
        tokens = self.cms(tokens)
        features = self.output_norm(tokens).transpose(1, 2).reshape(
            batch_size,
            self.memory_dimension,
            height,
            width,
        )
        return inputs + self.output_projection(features)


class HeadwiseCMSDetect(Detect):
    """Detection head with immutable and plastic class-memory branches.

    The original classification branch is the persistent memory. A cloned branch
    learns selected classes at multiple optimizer frequencies. The fixed mask is
    the independent/head-wise CMS aggregation from equation 74: old-class logits
    always come from persistent memory and plastic-class logits always come from
    the adaptive branch.
    """

    plastic_cv2: nn.ModuleList
    plastic_cv3: nn.ModuleList

    def forward_head(
        self,
        x: list[Tensor],
        box_head: nn.Module | None = None,
        cls_head: nn.Module | None = None,
    ) -> dict[str, Tensor]:
        if box_head is None or cls_head is None:
            return {}
        if not hasattr(self, "plastic_cv3"):
            return super().forward_head(x, box_head=box_head, cls_head=cls_head)
        if cls_head is not self.cv3:
            raise RuntimeError("HeadwiseCMSDetect does not support one-to-one heads")

        batch_size = x[0].shape[0]
        persistent_boxes = torch.cat(
            [
                self.cv2[index](x[index]).view(
                    batch_size,
                    4 * self.reg_max,
                    -1,
                )
                for index in range(self.nl)
            ],
            dim=-1,
        )
        plastic_boxes = torch.cat(
            [
                self.plastic_cv2[index](x[index]).view(
                    batch_size,
                    4 * self.reg_max,
                    -1,
                )
                for index in range(self.nl)
            ],
            dim=-1,
        )
        persistent_scores = torch.cat(
            [
                self.cv3[index](x[index]).view(batch_size, self.nc, -1)
                for index in range(self.nl)
            ],
            dim=-1,
        )
        plastic_scores = torch.cat(
            [
                self.plastic_cv3[index](x[index]).view(batch_size, self.nc, -1)
                for index in range(self.nl)
            ],
            dim=-1,
        )
        mask = self.plastic_class_mask.to(
            device=persistent_scores.device,
            dtype=torch.bool,
        )[None, :, None]
        disabled = torch.full_like(persistent_scores, -30.0)
        persistent_scores = torch.where(
            mask,
            disabled,
            persistent_scores,
        )
        plastic_scores = torch.where(mask, plastic_scores, disabled)
        return {
            "boxes": torch.cat((persistent_boxes, plastic_boxes), dim=-1),
            "scores": torch.cat((persistent_scores, plastic_scores), dim=-1),
            "feats": [*x, *x],
        }

    def fuse(self) -> None:
        self.cv2 = self.cv3 = self.plastic_cv2 = self.plastic_cv3 = None


class ClassRoutedDetectionEnsemble(nn.Module):
    """Inference-time head-wise CMS over two complete detector memories.

    The persistent detector contributes every non-plastic class. The plastic
    detector contributes only explicitly selected classes. Bounding-box
    candidates from both memories are retained so each task keeps its own
    localization representation.
    """

    def __init__(
        self,
        persistent_model: nn.Module,
        plastic_model: nn.Module,
        plastic_class_ids: Sequence[int],
        *,
        retain_persistent_plastic_classes: bool = False,
        plastic_score_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.persistent_model = persistent_model.eval()
        self.plastic_model = plastic_model.eval()
        self.names = getattr(persistent_model, "names")
        self.task = "detect"
        self.stride = getattr(persistent_model, "stride", torch.tensor([32.0]))
        self.args = dict(getattr(persistent_model, "args", {}))
        self.yaml = getattr(persistent_model, "yaml", None)
        if plastic_score_scale <= 0:
            raise ValueError("Plastic score scale must be positive")
        self.retain_persistent_plastic_classes = bool(
            retain_persistent_plastic_classes
        )
        self.plastic_score_scale = float(plastic_score_scale)

        class_count = len(self.names)
        selected = tuple(sorted({int(value) for value in plastic_class_ids}))
        if not selected or selected[0] < 0 or selected[-1] >= class_count:
            raise ValueError("Invalid plastic classes for routed detector ensemble")
        if getattr(plastic_model, "names", None) != self.names:
            raise ValueError("Routed detector memories must use identical class names")
        mask = torch.zeros(class_count, dtype=torch.bool)
        mask[list(selected)] = True
        self.register_buffer("plastic_class_mask", mask)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _decoded(prediction: Any) -> Tensor:
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        if not isinstance(prediction, Tensor) or prediction.ndim != 3:
            raise TypeError("Routed detector memory returned an unsupported output")
        return prediction

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
                "ClassRoutedDetectionEnsemble is a consolidated inference model"
            )
        persistent = self._decoded(
            self.persistent_model.predict(
                inputs,
                profile=profile,
                visualize=visualize,
                augment=augment,
                embed=embed,
            )
        )
        plastic = self._decoded(
            self.plastic_model.predict(
                inputs,
                profile=profile,
                visualize=visualize,
                augment=augment,
                embed=embed,
            )
        )
        class_count = self.plastic_class_mask.numel()
        if persistent.shape[1] != 4 + class_count or plastic.shape[1] != 4 + class_count:
            raise ValueError("Routed detector memories have incompatible outputs")

        mask = self.plastic_class_mask[None, :, None]
        persistent_scores = persistent[:, 4:]
        retain_persistent_plastic_classes = bool(
            getattr(self, "retain_persistent_plastic_classes", False)
        )
        plastic_score_scale = float(getattr(self, "plastic_score_scale", 1.0))
        if not retain_persistent_plastic_classes:
            persistent_scores = persistent_scores.masked_fill(mask, 0.0)
        plastic_scores = (
            plastic[:, 4:].masked_fill(~mask, 0.0)
            * plastic_score_scale
        )
        persistent = torch.cat((persistent[:, :4], persistent_scores), dim=1)
        plastic = torch.cat((plastic[:, :4], plastic_scores), dim=1)
        return torch.cat((persistent, plastic), dim=2)

    def fuse(self, verbose: bool = True):
        del verbose
        if hasattr(self.persistent_model, "fuse"):
            self.persistent_model.fuse()
        if hasattr(self.plastic_model, "fuse"):
            self.plastic_model.fuse()
        return self


def install_headwise_cms(
    model: nn.Module,
    class_ids: Sequence[int],
    update_periods: Sequence[int] = (16, 4, 1),
) -> HeadwiseCMSDetect:
    """Convert a YOLO Detect head to a class-routed head-wise CMS in place."""

    head = getattr(model, "model", [None])[-1]
    if not isinstance(head, Detect):
        raise TypeError("Head-wise CMS requires an Ultralytics Detect head")
    if head.end2end:
        raise ValueError("Head-wise CMS currently requires a one-to-many Detect head")

    selected = tuple(sorted({int(value) for value in class_ids}))
    if not selected:
        raise ValueError("Head-wise CMS requires at least one plastic class")
    if selected[0] < 0 or selected[-1] >= int(head.nc):
        raise ValueError("Plastic class id is outside the detector class range")

    periods = tuple(int(value) for value in update_periods)
    stage_count = len(head.cv3[0])
    if len(periods) != stage_count or any(value < 1 for value in periods):
        raise ValueError(
            f"Head-wise CMS requires {stage_count} positive update periods"
        )
    if isinstance(head, HeadwiseCMSDetect):
        existing = tuple(
            torch.nonzero(head.plastic_class_mask, as_tuple=False)
            .flatten()
            .tolist()
        )
        if existing != selected:
            raise ValueError(
                f"Checkpoint plastic classes {existing} do not match {selected}"
            )
    else:
        head.__class__ = HeadwiseCMSDetect
        head.plastic_cv2 = copy.deepcopy(head.cv2)
        head.plastic_cv3 = copy.deepcopy(head.cv3)
        mask = torch.zeros(int(head.nc), dtype=torch.bool)
        mask[list(selected)] = True
        head.register_buffer("plastic_class_mask", mask)
        head.stride = head.stride.repeat(2)

    head.plastic_update_periods = periods
    for branches in (head.plastic_cv2, head.plastic_cv3):
        for branch in branches:
            if len(branch) != len(periods):
                raise ValueError("Inconsistent detection branch depth")
            for stage, update_every in zip(branch, periods):
                for parameter in stage.parameters():
                    parameter._hope_update_every = update_every  # type: ignore[attr-defined]
    return head


@torch.no_grad()
def initialize_headwise_cms_classes(
    target: HeadwiseCMSDetect,
    source: nn.Module,
) -> None:
    """Restore plastic output rows from a compatible pretrained detector."""

    source_head = getattr(source, "model", [None])[-1]
    if not isinstance(source_head, Detect):
        raise TypeError("Plastic initialization source must contain a Detect head")
    class_ids = (
        torch.nonzero(target.plastic_class_mask, as_tuple=False)
        .flatten()
        .tolist()
    )
    if int(source_head.nc) != int(target.nc) or len(source_head.cv3) != len(
        target.plastic_cv3
    ):
        raise ValueError("Plastic initialization source has an incompatible head")
    for target_branch, source_branch in zip(
        target.plastic_cv3,
        source_head.cv3,
    ):
        target_output = target_branch[-1]
        source_output = source_branch[-1]
        if (
            not isinstance(target_output, nn.Conv2d)
            or not isinstance(source_output, nn.Conv2d)
            or target_output.weight.shape != source_output.weight.shape
        ):
            raise ValueError("Plastic classifier output tensors are incompatible")
        target_output.weight[class_ids].copy_(
            source_output.weight[class_ids].to(
                device=target_output.weight.device,
                dtype=target_output.weight.dtype,
            )
        )
        if target_output.bias is not None and source_output.bias is not None:
            target_output.bias[class_ids].copy_(
                source_output.bias[class_ids].to(
                    device=target_output.bias.device,
                    dtype=target_output.bias.dtype,
                )
            )


def register_ultralytics_modules() -> None:
    """Register custom modules in Ultralytics' YAML parser namespace."""

    import ultralytics.nn.tasks as tasks

    tasks.Hope2D = Hope2D
    tasks.HeadwiseCMSDetect = HeadwiseCMSDetect
    tasks.ClassRoutedDetectionEnsemble = ClassRoutedDetectionEnsemble


def iter_continuum_memories(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, ContinuumMemorySystem):
            yield name, module


def configure_continual_memory(
    model: nn.Module,
    config: Mapping[str, Any] | None = None,
) -> None:
    settings = dict(config or {})
    for _, cms in iter_continuum_memories(model):
        cms.configure_persistence(
            enabled=bool(settings.get("enabled", True)),
            capture=bool(settings.get("capture", False)),
            momentum=float(settings.get("momentum", 0.98)),
            consolidated_strength=float(
                settings.get("consolidated_strength", 0.10)
            ),
            working_strength=float(settings.get("working_strength", 0.05)),
            lock_slowest=bool(settings.get("lock_slowest", True)),
        )


def reset_continual_memory(model: nn.Module) -> None:
    for _, cms in iter_continuum_memories(model):
        cms.reset_persistent_state()


def consolidate_continual_memory(model: nn.Module) -> None:
    for _, cms in iter_continuum_memories(model):
        cms.consolidate()


def export_continual_memory(model: nn.Module) -> dict[str, dict[str, Tensor]]:
    return {
        name: cms.persistent_state_dict()
        for name, cms in iter_continuum_memories(model)
    }


def load_continual_memory(
    model: nn.Module,
    source: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    payload = (
        torch.load(Path(source), map_location="cpu", weights_only=False)
        if isinstance(source, (str, Path))
        else dict(source)
    )
    states = payload.get("cms_state", payload)
    modules = dict(iter_continuum_memories(model))
    missing = [name for name in states if name not in modules]
    if missing:
        raise KeyError(
            "CMS modules from the consolidation artifact were not found: "
            + ", ".join(missing)
        )
    for name, state in states.items():
        modules[name].load_persistent_state_dict(state)
    return payload
