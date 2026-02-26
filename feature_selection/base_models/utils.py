from __future__ import annotations


def build_member_model_name(base_model_name: str, member_identity: str) -> str:
    """Build deterministic model names for flattened multi-member outputs."""
    return f"{base_model_name}::{member_identity}" if member_identity else base_model_name

