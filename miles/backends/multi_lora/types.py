"""Backend-neutral Multi-LoRA operation types."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

from miles.multi_lora.types import AdapterIdentity

_PayloadT = TypeVar("_PayloadT")
_ResultT = TypeVar("_ResultT")


class OperationKind(str, Enum):
    CREATE_MODEL = "create_model"
    FORWARD = "forward"
    FORWARD_BACKWARD = "forward_backward"
    OPTIM_STEP = "optim_step"
    CLEAR_GRADIENTS = "clear_gradients"
    SAVE_STATE = "save_state"
    LOAD_STATE = "load_state"
    EXPORT_FOR_SAMPLER = "export_for_sampler"
    RELEASE_ADAPTER = "release_adapter"


class OptimStepPolicy(str, Enum):
    EXPLICIT_ADAM_SUM = "explicit_adam_sum"
    SCHEDULED_MEAN = "scheduled_mean"


@dataclass(frozen=True)
class LossSpec:
    name: str
    config: Mapping[str, object] | None = None


@dataclass(frozen=True)
class AdamParams:
    learning_rate: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    grad_clip_norm: float


@dataclass(frozen=True)
class OptimStepSpec:
    policy: OptimStepPolicy
    adam_params: AdamParams | None = None
    gradient_denominator: float | None = None
    scheduler_increment: int | None = None


@dataclass(frozen=True)
class OperationRequest(Generic[_PayloadT]):
    operation_id: str
    identity: AdapterIdentity
    kind: OperationKind
    payload: _PayloadT


@dataclass(frozen=True)
class RankResult(Generic[_ResultT]):
    operation_id: str
    identity: AdapterIdentity
    rank: int
    result: _ResultT


@dataclass(frozen=True)
class OperationOutcome(Generic[_ResultT]):
    operation_id: str
    identity: AdapterIdentity
    kind: OperationKind
    result: _ResultT


@dataclass(frozen=True)
class ShardDescriptor:
    rank: int
    path: str
    size: int
    checksum: str


@dataclass(frozen=True)
class ExportArtifact:
    operation_id: str
    identity: AdapterIdentity
    tensors: Mapping[str, object]
    config: Mapping[str, object]
    checksums: Mapping[str, str]


class PreflightRejected(RuntimeError):
    """The request failed validation before backend mutation."""


class RuntimeFenced(RuntimeError):
    """Backend state is uncertain and the runtime must restart."""
