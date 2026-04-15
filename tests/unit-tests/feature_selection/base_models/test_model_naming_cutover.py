"""Legacy export surface and member naming helper."""

from __future__ import annotations

import pytest

from feature_selection.base_models import ContinuousBinningModel, RuleBasedModel


def build_member_model_name(base_model_name: str, member_identity: str) -> str:
    """Build deterministic model names for flattened multi-member outputs."""
    return f"{base_model_name}::{member_identity}" if member_identity else base_model_name


def test_legacy_models_are_exported_and_raise_on_init() -> None:
    assert ContinuousBinningModel.__name__ == "ContinuousBinningModel"
    assert RuleBasedModel.__name__ == "RuleBasedModel"
    assert ContinuousBinningModel.model_type == "removed"
    assert RuleBasedModel.model_type == "removed"
    with pytest.raises(RuntimeError, match="signed-signal"):
        ContinuousBinningModel(n_bins=3, strategy="long")
    with pytest.raises(RuntimeError, match="signed-signal"):
        RuleBasedModel(strategy="long")


def test_member_model_name_is_deterministic() -> None:
    assert (
        build_member_model_name("ewmac", "spanFast=32,spanSlow=128")
        == "ewmac::spanFast=32,spanSlow=128"
    )


def test_member_model_name_without_member_identity_keeps_single_name() -> None:
    assert build_member_model_name("ewmac", "") == "ewmac"
