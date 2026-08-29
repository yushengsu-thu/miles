"""Tests for shared Multi-LoRA identities."""

from dataclasses import FrozenInstanceError, fields

import pytest

from miles.multi_lora.types import AdapterIdentity, ServingRef


def test_identity_contract_has_only_shared_fields():
    assert [field.name for field in fields(AdapterIdentity)] == ["name", "registration_id", "slot"]
    assert [field.name for field in fields(ServingRef)] == ["identity", "version"]


def test_identity_contract_is_frozen_and_hashable():
    identity = AdapterIdentity(name="adapter-a", registration_id="registration-a", slot=2)
    serving_ref = ServingRef(identity=identity, version=7)

    next_registration = AdapterIdentity(name="adapter-a", registration_id="registration-b", slot=2)
    assert len({identity, next_registration}) == 2
    assert len({serving_ref, ServingRef(identity=identity, version=8)}) == 2

    with pytest.raises(FrozenInstanceError):
        identity.slot = 3
    with pytest.raises(FrozenInstanceError):
        serving_ref.version = 8
