"""Tests for selected-slot optimizer views."""

from copy import deepcopy

import pytest
import torch

from miles.backends.multi_lora.megatron.optimizer import SlotOptimizerHandle

SLOT_TAG = "miles_multi_lora_slot"


class FakeChild:
    def __init__(self, slot, *, prepare_found_inf=False, step_success=True):
        self.parameter = torch.nn.Parameter(torch.tensor([float(slot + 1)]))
        self.parameter.grad = torch.tensor([4.0])
        self.param_groups = [{SLOT_TAG: slot, "params": [self.parameter], "step": torch.tensor(7.0)}]
        self.state = {
            self.parameter: {
                "step": 9,
                "exp_avg": torch.tensor([2.0]),
                "exp_avg_sq": torch.tensor([3.0]),
                "max_exp_avg_sq": torch.tensor([5.0]),
            }
        }
        self.prepare_found_inf = prepare_found_inf
        self.step_success = step_success
        self.prepare_calls = 0
        self.step_calls = 0
        self.reload_calls = 0
        self.loaded_state = None

    def get_parameters(self):
        return [self.parameter]

    def get_main_grads_for_grad_norm(self):
        return [self.parameter.grad]

    def prepare_grads(self):
        self.prepare_calls += 1
        return self.prepare_found_inf

    def step_with_ready_grads(self):
        self.step_calls += 1
        return self.step_success

    def state_dict(self):
        return {"state": self.state, "param_groups": self.param_groups}

    def load_state_dict(self, state):
        self.loaded_state = state

    def zero_grad(self, *, set_to_none):
        assert set_to_none
        self.parameter.grad = None

    def reload_model_params(self):
        self.reload_calls += 1


class FakeRoot:
    def __init__(self, children, mapping):
        self.chained_optimizers = children
        self.miles_slot_child_indices = mapping
        self.allgather_calls = 0

    def allgather_params(self):
        self.allgather_calls += 1


def _root():
    children = [FakeChild(0), FakeChild(1), FakeChild(0)]
    return FakeRoot(children, {0: (0, 2), 1: (1,)})


@pytest.mark.parametrize(
    ("mapping", "error"),
    [
        ({}, ValueError),
        ({0: ()}, ValueError),
        ({0: (0, 0)}, ValueError),
        ({0: (True,)}, TypeError),
        ({0: (3,)}, ValueError),
    ],
)
def test_constructor_rejects_invalid_child_mapping(mapping, error):
    root = FakeRoot([FakeChild(0)], mapping)
    with pytest.raises(error):
        SlotOptimizerHandle(root, 0)


def test_constructor_rejects_mistagged_children():
    root = FakeRoot([FakeChild(1)], {0: (0,)})
    with pytest.raises(ValueError, match="not tagged"):
        SlotOptimizerHandle(root, 0)


def test_view_exposes_only_selected_children_in_mapping_order():
    root = _root()
    handle = SlotOptimizerHandle(root, 0)

    assert handle.children == (root.chained_optimizers[0], root.chained_optimizers[2])
    assert handle.param_groups == [child.param_groups[0] for child in handle.children]
    assert handle.get_parameters() == [child.parameter for child in handle.children]
    assert handle.get_main_grads_for_grad_norm() == [child.parameter.grad for child in handle.children]


def test_prepare_and_step_visit_every_selected_child():
    children = [FakeChild(0, prepare_found_inf=False, step_success=False), FakeChild(0, prepare_found_inf=True)]
    handle = SlotOptimizerHandle(FakeRoot(children, {0: (0, 1)}), 0)

    assert handle.prepare_grads()
    assert not handle.step()
    assert [child.prepare_calls for child in children] == [1, 1]
    assert [child.step_calls for child in children] == [1, 1]


def test_state_load_validates_count_and_deepcopies_nested_state():
    root = _root()
    handle = SlotOptimizerHandle(root, 1)
    source = {"state": {0: {"exp_avg": torch.tensor([4.0])}}, "param_groups": [{SLOT_TAG: 99}]}

    with pytest.raises(ValueError, match="expected 1"):
        handle.load_state_dict([])
    handle.load_state_dict([source])
    source["state"][0]["exp_avg"].fill_(11.0)

    child = root.chained_optimizers[1]
    assert child.loaded_state["state"][0]["exp_avg"].item() == 4.0
    assert child.param_groups[0][SLOT_TAG] == 1


def test_reset_clears_selected_state_without_touching_inactive_child():
    root = _root()
    inactive = root.chained_optimizers[1]
    inactive_before = deepcopy(inactive.state[inactive.parameter])
    handle = SlotOptimizerHandle(root, 0)

    handle.reset()

    for child in handle.children:
        assert child.parameter.grad is None
        assert child.param_groups[0]["step"].item() == 0
        state = child.state[child.parameter]
        assert state["step"] == 0
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            assert torch.count_nonzero(state[key]).item() == 0
    for key, value in inactive_before.items():
        current = inactive.state[inactive.parameter][key]
        assert torch.equal(current, value) if isinstance(value, torch.Tensor) else current == value


def test_selected_sync_and_root_allgather_are_separate_operations():
    root = _root()
    handle = SlotOptimizerHandle(root, 0)

    handle.reload_model_params()
    assert [child.reload_calls for child in root.chained_optimizers] == [1, 0, 1]
    assert root.allgather_calls == 0

    handle.allgather_params()
    assert root.allgather_calls == 1
