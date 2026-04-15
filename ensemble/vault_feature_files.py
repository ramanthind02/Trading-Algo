"""Compatibility shim; prefer imports from ``ensemble.vault.feature_files``."""
from __future__ import annotations

from importlib import import_module
from typing import Any

_m = import_module("ensemble.vault.feature_files")


def __getattr__(name: str) -> Any:
    return getattr(_m, name)


def __dir__() -> list[str]:
    return sorted(dir(_m))
