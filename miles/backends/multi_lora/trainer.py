"""Backend-neutral distributed Multi-LoRA trainer contract."""

from typing import Any, Protocol

from miles.backends.multi_lora.types import ExportArtifact, OperationOutcome, OperationRequest

_Request = OperationRequest[Any]
_Outcome = OperationOutcome[Any]


class MultiLoraTrainer(Protocol):
    """Distributed operations for one exact adapter identity."""

    async def create_model(self, request: _Request) -> _Outcome: ...

    async def forward(self, request: _Request) -> _Outcome: ...

    async def forward_backward(self, request: _Request) -> _Outcome: ...

    async def optim_step(self, request: _Request) -> _Outcome: ...

    async def clear_gradients(self, request: _Request) -> _Outcome:
        """Run internal gradient cleanup."""
        ...

    async def save_state(self, request: _Request) -> _Outcome: ...

    async def load_state(self, request: _Request) -> _Outcome: ...

    async def export_for_sampler(self, request: _Request) -> ExportArtifact: ...

    async def release_adapter(self, request: _Request) -> _Outcome:
        """Run internal adapter teardown."""
        ...
