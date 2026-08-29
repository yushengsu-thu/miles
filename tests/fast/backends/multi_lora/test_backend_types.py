"""Tests for backend-neutral Multi-LoRA shapes."""

from dataclasses import FrozenInstanceError, fields

import pytest

from miles.backends.multi_lora.types import (
    AdamParams,
    ExportArtifact,
    LossSpec,
    OperationKind,
    OperationOutcome,
    OperationRequest,
    OptimStepPolicy,
    OptimStepSpec,
    PreflightRejected,
    RankResult,
    RuntimeFenced,
    ShardDescriptor,
)
from miles.multi_lora.types import AdapterIdentity


def _field_names(shape):
    return [field.name for field in fields(shape)]


def test_operation_and_policy_enums_are_complete():
    assert [kind.value for kind in OperationKind] == [
        "create_model",
        "forward",
        "forward_backward",
        "optim_step",
        "clear_gradients",
        "save_state",
        "load_state",
        "export_for_sampler",
        "release_adapter",
    ]
    assert [policy.value for policy in OptimStepPolicy] == ["explicit_adam_sum", "scheduled_mean"]


def test_neutral_shapes_have_only_backend_fields():
    assert _field_names(LossSpec) == ["name", "config"]
    assert _field_names(AdamParams) == ["learning_rate", "beta1", "beta2", "eps", "weight_decay", "grad_clip_norm"]
    assert _field_names(OptimStepSpec) == ["policy", "adam_params", "gradient_denominator", "scheduler_increment"]
    assert _field_names(OperationRequest) == ["operation_id", "identity", "kind", "payload"]
    assert _field_names(RankResult) == ["operation_id", "identity", "rank", "result"]
    assert _field_names(OperationOutcome) == ["operation_id", "identity", "kind", "result"]
    assert _field_names(ShardDescriptor) == ["rank", "path", "size", "checksum"]
    assert _field_names(ExportArtifact) == ["operation_id", "identity", "tensors", "config", "checksums"]


def test_operation_shapes_are_frozen_and_preserve_identity():
    identity = AdapterIdentity("adapter-a", "registration-a", 1)
    request = OperationRequest("operation-a", identity, OperationKind.FORWARD, {"rows": 2})
    result = RankResult("operation-a", identity, 0, {"loss": 1.0})
    outcome = OperationOutcome("operation-a", identity, OperationKind.FORWARD, result.result)

    assert request.identity is result.identity is outcome.identity
    with pytest.raises(FrozenInstanceError):
        request.operation_id = "operation-b"


def test_backend_failures_have_distinct_recovery_meaning():
    assert issubclass(PreflightRejected, RuntimeError)
    assert issubclass(RuntimeFenced, RuntimeError)
    assert not issubclass(PreflightRejected, RuntimeFenced)
