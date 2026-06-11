"""Agent-driven research pipeline — the ``StrategySpec`` and its adapter.

A strategy is described by one flat, self-validating :class:`StrategySpec`; the adapter
translates it into the existing feature / portfolio research configs and the Nautilus
execution engine without ever mutating the canonical configs.

See ``docs/library/Strategy_research/`` (README + the five methodology docs) for the design.
"""

from __future__ import annotations

from research.spec.adapter import (
    apply_vol_scaling,
    build_bias_spec,
    default_windows,
    ewsd_blend_for,
    feed_literal,
    resolve_windows,
    to_feature_config,
    to_pnl_engine,
    to_portfolio_config,
    validate,
)
from research.spec.serialization import (
    load_spec,
    save_spec,
    spec_from_dict,
    spec_from_json,
    spec_to_dict,
    spec_to_json,
)
from research.spec.strategy_spec import (
    MAX_GRID_COMBOS,
    AccountSpec,
    DataFeed,
    Direction,
    ExecutionSpec,
    FillFeed,
    Holding,
    OrderPolicy,
    PropConstraints,
    ResearchWindows,
    RiskSpec,
    SignalSpec,
    StrategyMode,
    StrategySpec,
    Ticker,
    TimeFrame,
    UnfilledLimitPolicy,
    VaultTarget,
    VolScaling,
    VolScalingModel,
    combo_count,
)

__all__ = [
    # core
    "StrategySpec",
    "MAX_GRID_COMBOS",
    "combo_count",
    # sub-objects
    "SignalSpec",
    "ResearchWindows",
    "RiskSpec",
    "PropConstraints",
    "AccountSpec",
    "ExecutionSpec",
    "VaultTarget",
    # spec enums
    "StrategyMode",
    "DataFeed",
    "VolScaling",
    "VolScalingModel",
    "OrderPolicy",
    "Holding",
    "UnfilledLimitPolicy",
    "FillFeed",
    # reused domain enums (re-exported for convenience)
    "Ticker",
    "TimeFrame",
    "Direction",
    # adapter
    "to_feature_config",
    "to_portfolio_config",
    "to_pnl_engine",
    "validate",
    "default_windows",
    "resolve_windows",
    "build_bias_spec",
    "feed_literal",
    "apply_vol_scaling",
    "ewsd_blend_for",
    # serialization (the round-trippable JSON contract)
    "spec_to_dict",
    "spec_from_dict",
    "spec_to_json",
    "spec_from_json",
    "load_spec",
    "save_spec",
]
