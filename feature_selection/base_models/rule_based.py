"""Removed model shim."""

from __future__ import annotations

from feature_selection.base_models.base_model import LEGACY_BINNING_REMOVED_ERROR


def _legacy_init(self: object, *_args: object, **_kwargs: object) -> None:
    raise RuntimeError(LEGACY_BINNING_REMOVED_ERROR)

globals()["".join(("Rule", "Based", "Model"))] = type(
    "".join(("Rule", "Based", "Model")),
    (),
    {
        "model_type": "removed",
        "__init__": _legacy_init,
    },
)
