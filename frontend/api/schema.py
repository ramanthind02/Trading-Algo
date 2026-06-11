"""Form schema for the Spec Builder: enum option lists, vault sleeves, and the node catalog.

The React form is data-driven — it renders selects from these option lists rather than
hard-coding the enum members, so adding a ``TimeFrame`` or a vault sleeve in Python flows to the
UI automatically.
"""

from __future__ import annotations

import importlib
import inspect
from enum import Enum
from typing import Any

from ensemble.vault.constants import valid_weight_hierarchy_groups
from lib.core.enums import Direction, Ticker, TimeFrame
from nodes._taxonomy import CANONICAL_MODULE_CLASSES, CANONICAL_MODULE_IMPORTS
from research.spec.strategy_spec import (
    FillFeed,
    Holding,
    OrderPolicy,
    StrategyMode,
    UnfilledLimitPolicy,
    VolScaling,
    VolScalingModel,
)


def _humanize(token: str) -> str:
    """``"market_on_open"`` -> ``"Market On Open"`` (labels for selects)."""

    return token.replace("_", " ").title()


def _enum_token(member: Enum) -> str:
    return member.value if isinstance(member.value, str) else member.name


def _options(enum_cls: type[Enum]) -> list[dict[str, str]]:
    return [{"value": _enum_token(m), "label": _humanize(_enum_token(m))} for m in enum_cls]


# Daily vs intraday admissible timeframes (mirrors StrategySpec._validate_mode_timeframe).
_DAILY_TF = ("D", "W", "M")
_INTRADAY_TF = ("H1", "H4")


def form_schema() -> dict[str, Any]:
    """Every option list the Spec Builder form needs, keyed by field."""

    return {
        "tickers": [t.name for t in Ticker],
        "timeframes": {
            "daily": list(_DAILY_TF),
            "intraday": list(_INTRADAY_TF),
            "all": [t.name for t in TimeFrame],
        },
        "modes": _options(StrategyMode),
        # ``data_feeds`` dropdown removed: signals are always additive futures and both result
        # lanes (ratio-futures + CFD) are produced on every run, so there is no feed to select.
        "directions": _options(Direction),
        "vol_scalings": _options(VolScaling),
        "vol_scaling_models": _options(VolScalingModel),
        "order_policies": _options(OrderPolicy),
        "unfilled_limit_policies": _options(UnfilledLimitPolicy),
        "holdings": _options(Holding),
        "fill_feeds": _options(FillFeed),
        "vault_sleeves": sorted(valid_weight_hierarchy_groups()),
        "max_grid_combos": _max_grid_combos(),
    }


def _max_grid_combos() -> int:
    from research.spec.strategy_spec import MAX_GRID_COMBOS

    return MAX_GRID_COMBOS


def module_catalog() -> list[dict[str, str]]:
    """The bias-node modules a signal can use, deduped to canonical names, grouped by category.

    Category is the node folder (``nodes.<category>...``). Concatenated aliases that resolve to
    the same class are collapsed (keep the first, underscore-style, canonical name).
    """

    seen_classes: set[str] = set()
    catalog: list[dict[str, str]] = []
    for name, class_name in CANONICAL_MODULE_CLASSES.items():
        import_path = CANONICAL_MODULE_IMPORTS.get(name)
        if import_path is None or class_name in seen_classes:
            continue
        seen_classes.add(class_name)
        parts = import_path.split(".")
        category = parts[1] if len(parts) > 1 else "other"
        catalog.append(
            {
                "name": name,
                "class_name": class_name,
                "category": category,
                "import_path": import_path,
            }
        )
    return sorted(catalog, key=lambda item: (item["category"], item["name"]))


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _enum_token(value)
    return str(value)


def module_detail(name: str) -> dict[str, Any]:
    """Best-effort constructor signature for a module, to suggest ``param_grid`` keys.

    These are the node class ``__init__`` parameters — a *suggestion* for the param-grid editor,
    not a strict contract (the pipeline builds nodes from the ``params`` dict). Standard plumbing
    params (timeframe, ticker, etc.) are included; the author picks which to sweep.
    """

    import_path = CANONICAL_MODULE_IMPORTS.get(name)
    class_name = CANONICAL_MODULE_CLASSES.get(name)
    if import_path is None or class_name is None:
        raise KeyError(f"Unknown module: {name}")

    module = importlib.import_module(import_path)
    klass = getattr(module, class_name)
    signature = inspect.signature(klass.__init__)
    node_param_choices: dict[str, list[str]] = getattr(klass, "param_choices", {})

    params: list[dict[str, Any]] = []
    for pname, parameter in signature.parameters.items():
        if pname == "self" or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        has_default = parameter.default is not inspect.Parameter.empty
        entry: dict[str, Any] = {
            "name": pname,
            "required": not has_default,
            "default": _jsonable(parameter.default) if has_default else None,
            "annotation": (
                None
                if parameter.annotation is inspect.Parameter.empty
                else str(parameter.annotation)
            ),
        }
        if pname in node_param_choices:
            entry["choices"] = node_param_choices[pname]
        params.append(entry)
    return {
        "name": name,
        "class_name": class_name,
        "import_path": import_path,
        "doc": inspect.getdoc(klass) or "",
        "params": params,
    }
