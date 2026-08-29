"""Slot-local views over the fixed Multi-LoRA optimizer topology."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
    from megatron.core.optimizer.optimizer import MegatronOptimizer

_SLOT_TAG = "miles_multi_lora_slot"
_MOMENT_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")


def _reset_step(container: dict[str, Any]) -> None:
    if "step" not in container:
        return
    step = container["step"]
    if isinstance(step, torch.Tensor):
        step.zero_()
    else:
        container["step"] = 0


class SlotOptimizerHandle:
    """A non-owning view of the optimizer children assigned to one slot."""

    def __init__(self, optimizer: LayerWiseDistributedOptimizer, slot: int) -> None:
        indices_by_slot = getattr(optimizer, "miles_slot_child_indices", None)
        if not isinstance(indices_by_slot, dict) or slot not in indices_by_slot:
            raise ValueError(f"optimizer has no child mapping for slot {slot}")

        indices = tuple(indices_by_slot[slot])
        if not indices or len(indices) != len(set(indices)):
            raise ValueError(f"slot {slot} must map to unique optimizer children")
        if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
            raise TypeError(f"slot {slot} child indices must be integers")

        all_children = optimizer.chained_optimizers
        if any(index < 0 or index >= len(all_children) for index in indices):
            raise ValueError(f"slot {slot} child index is out of range")
        children = tuple(all_children[index] for index in indices)
        for child in children:
            for group in child.param_groups:
                if group.get(_SLOT_TAG) != slot:
                    raise ValueError(f"optimizer child is not tagged for slot {slot}")

        self._optimizer = optimizer
        self.slot = slot
        self.children: tuple[MegatronOptimizer, ...] = children

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [group for child in self.children for group in child.param_groups]

    def get_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for child in self.children for parameter in child.get_parameters()]

    def get_main_grads_for_grad_norm(self) -> list[torch.Tensor]:
        return [gradient for child in self.children for gradient in child.get_main_grads_for_grad_norm()]

    def prepare_grads(self) -> bool:
        found_inf = False
        for child in self.children:
            found_inf |= bool(child.prepare_grads())
        return found_inf

    def step(self) -> bool:
        """Step selected children without running root-level collectives."""
        success = True
        for child in self.children:
            success &= bool(child.step_with_ready_grads())
        return success

    def state_dict(self) -> list[dict[str, Any]]:
        """Return a stable list envelope even when the slot has one child."""
        return [child.state_dict() for child in self.children]

    def load_state_dict(self, states: Sequence[dict[str, Any]]) -> None:
        if len(states) != len(self.children):
            raise ValueError(f"slot {self.slot} expected {len(self.children)} optimizer states, got {len(states)}")
        for child, state in zip(self.children, states, strict=True):
            child.load_state_dict(deepcopy(state))
            for group in child.param_groups:
                group[_SLOT_TAG] = self.slot

    @torch.no_grad()
    def reset(self) -> None:
        """Clear selected gradients, Adam moments, and step clocks in place."""
        for child in self.children:
            child.zero_grad(set_to_none=True)
            for group in child.param_groups:
                _reset_step(group)
            for state in child.state.values():
                _reset_step(state)
                for key in _MOMENT_KEYS:
                    value = state.get(key)
                    if isinstance(value, torch.Tensor):
                        value.zero_()

    def reload_model_params(self) -> None:
        """Refresh selected FP32 masters after model-slot initialization."""
        for child in self.children:
            child.reload_model_params()

    def allgather_params(self) -> None:
        """Run the root collective after all slot-local post-step masking."""
        self._optimizer.allgather_params()
