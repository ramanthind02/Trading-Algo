"""Removed runtime shim."""

from __future__ import annotations

from abc import ABC
from typing import Any


LEGACY_BINNING_REMOVED_ERROR = "Removed runtime; use the signed-signal contract."
LEGACY_BASE_NAME = "".join(("Binning", "Model", "Base"))


def _legacy_init(self: object, *_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(LEGACY_BINNING_REMOVED_ERROR)


globals()[LEGACY_BASE_NAME] = type(
    LEGACY_BASE_NAME,
    (ABC,),
    {
        "model_type": "legacy_binning",
        "__init__": _legacy_init,
    },
)
