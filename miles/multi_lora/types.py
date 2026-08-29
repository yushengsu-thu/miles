"""Shared immutable identities for Multi-LoRA components."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AdapterIdentity:
    """Stable identity for one adapter registration occupying a slot."""

    name: str
    registration_id: str
    slot: int


@dataclass(frozen=True)
class ServingRef:
    """Reference to one published version of an adapter registration."""

    identity: AdapterIdentity
    version: int
