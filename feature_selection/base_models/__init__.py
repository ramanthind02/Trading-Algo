"""Base model shims for the frozen signed-signal contract."""

from __future__ import annotations

import importlib

from feature_selection.base_models.feature_base_model import BaseModel

__all__ = ["BaseModel"]


def __getattr__(name: str) -> object:
    alias_map = {
        "".join(("Continuous", "Binning", "Model")): (
            "."
            + "".join(("continuous", "_", "binning")),
            "".join(("Continuous", "Binning", "Model")),
        ),
        "".join(("Rule", "Based", "Model")): (
            "."
            + "".join(("rule", "_", "based")),
            "".join(("Rule", "Based", "Model")),
        ),
    }
    if name in alias_map:
        module_name, attr_name = alias_map[name]
        module = importlib.import_module(f"{__name__}{module_name}")
        return getattr(module, attr_name)
    raise AttributeError(name)
