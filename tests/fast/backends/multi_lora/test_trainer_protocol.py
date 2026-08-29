"""Tests for the backend-neutral trainer protocol."""

import inspect

import miles.backends.multi_lora.trainer as trainer_module
from miles.backends.multi_lora.trainer import MultiLoraTrainer

OPERATIONS = {
    "create_model",
    "forward",
    "forward_backward",
    "optim_step",
    "clear_gradients",
    "save_state",
    "load_state",
    "export_for_sampler",
    "release_adapter",
}


def test_protocol_exposes_exactly_nine_async_operations():
    operations = {
        name
        for name, member in MultiLoraTrainer.__dict__.items()
        if not name.startswith("_") and inspect.iscoroutinefunction(member)
    }
    assert operations == OPERATIONS
    for name in operations:
        assert list(inspect.signature(getattr(MultiLoraTrainer, name)).parameters) == ["self", "request"]


def test_protocol_has_no_runtime_or_frontend_dependencies():
    source = inspect.getsource(trainer_module)
    for dependency in ("import ray", "import tinker", "megatron_utils", "Publication"):
        assert dependency not in source
