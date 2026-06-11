"""The ``StrategySpec`` object — the flat, self-validating description of one strategy.

This is the contract an **agent** (or a human) writes against. The agent never edits the
sprawling ``research/feature/config.py`` / ``research/portfolio/config.py``; it emits one
:class:`StrategySpec`, and the thin adapter in :mod:`research.spec.adapter` translates that
spec into the existing pipeline configs and the Nautilus execution engine.

The spec is intentionally small, declarative, and **self-validating** (every sub-object
checks its own invariants in ``__post_init__``) so a complete, correct strategy definition
can be constructed without internalising the whole research stack.

Source of truth: ``docs/library/Strategy_research/strategy_spec.md``. The dataclass is
implemented to match that doc.

Design notes
------------
* **No metrics fields.** Metrics are computed by the pipeline and presented by the agent;
  they are never goalposts (no auto accept/reject gates — README cross-cutting principle 1).
* **Dependency-light.** This module imports only the domain enums (:mod:`lib.core.enums`)
  and the vault sleeve constants. It deliberately does **not** import the Nautilus engine
  (which pulls ``nautilus_trader``); the adapter owns that translation.
* **Parsimony is enforced structurally.** The signal grid is capped at 300 combinations at
  construction time (README principle 7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Mapping, Sequence

from ensemble.vault.constants import valid_weight_hierarchy_groups
from lib.core.enums import Direction, Ticker, TimeFrame

# Maximum number of parameter combinations a single signal grid may expand to. Caps both
# the overfitting surface and compute (README principle 7 / strategy_spec.md §4).
MAX_GRID_COMBOS: int = 300

# Which ``TimeFrame`` members each strategy mode admits. ``TimeFrame`` currently exposes
# H1/H4/D/W/M; M15 (and finer) intraday members are a future addition.
_DAILY_TIMEFRAMES: frozenset[TimeFrame] = frozenset({TimeFrame.D, TimeFrame.W, TimeFrame.M})
_INTRADAY_TIMEFRAMES: frozenset[TimeFrame] = frozenset({TimeFrame.H1, TimeFrame.H4})


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StrategyMode(Enum):
    """Daily (D/W/M) vs intraday (H1/H4/…) research mode. Selects the window defaults."""

    DAILY = "daily"
    INTRADAY = "intraday"


class DataFeed(Enum):
    """Which price feed the research candle loaders use.

    Maps onto the existing ``data_feed`` literal consumed by the pipeline configs and
    :func:`lib.core.research_feed.set_research_feed` (``"futures"`` / ``"cfd"``).
    """

    NORGATE_FUTURES = "norgate_futures"  # continuous back-adjusted futures
    DARWINEX_CFD = "darwinex_cfd"        # MT5 CFD feed (faithful % returns)


class VolScaling(Enum):
    """Volatility-scaling choice. Collapses the two real decisions (scale? estimator?).

    See ``strategy_spec.md`` §5 — the daily model is EWSD, a 70 % short / 30 % long blend.
    """

    OFF = "off"            # raw all-in/all-out signal, no F = tau/sigma scaling
    BLENDED = "blended"    # sigma = 0.7*short(32d EWMA) + 0.3*long(2520d) — the default
    LONG_ONLY = "long_only"  # sigma = long-run only (blend 0.0 / 1.0)


class VolScalingModel(Enum):
    """Which return series feeds sigma for intraday strategies (``strategy_spec.md`` §5)."""

    INHERIT_DAILY = "inherit_daily"      # sigma from DAILY returns (default, recommended)
    INTRADAY_CUSTOM = "intraday_custom"  # sigma from intraday-bar returns (future plug-in)


class OrderPolicy(Enum):
    """How an order leg is worked. Mirrors the engine's ``ExecutionPolicy`` values; the
    adapter maps spec → engine so this module need not import the Nautilus engine.
    """

    MARKET_ON_OPEN = "market_on_open"  # cross the spread (taker); urgent
    LIMIT_AT_TOUCH = "limit_at_touch"  # rest a limit at the near touch (maker); fill risk
    LIMIT_IMPROVE = "limit_improve"    # rest inside the spread by improve_ticks


class Holding(Enum):
    """The alpha's holding intent — *not* the realised return convention.

    ``OVERNIGHT`` wants multi-day exposure; the c2c-vs-rollover-bounded-o2c convention is
    set by the portfolio swap policy, not the spec. ``INTRADAY`` wants session-only exposure.
    """

    OVERNIGHT = "overnight"
    INTRADAY = "intraday"


class UnfilledLimitPolicy(Enum):
    """What to do when a passive limit does not fill while the signal still wants the target."""

    CROSS_AFTER = "cross_after"  # work the limit, then cross with MARKET (guarantees the move)
    CARRY = "carry"              # leave the position to the next signal bar (no chase)


class FillFeed(Enum):
    """Which data lane simulates fills. ``DERIVED`` resolves from the leg policies."""

    DERIVED = "derived"
    SIGNAL_BAR = "signal_bar"  # fast vectorized bar-spine lane (both legs MARKET)
    BARS_M1 = "bars_m1"        # fine 1-minute bars (a leg uses a limit)
    TICKS = "ticks"            # tick lane (a leg uses a limit; highest fidelity)


_FINE_FILL_FEEDS: frozenset[FillFeed] = frozenset({FillFeed.BARS_M1, FillFeed.TICKS})
_LIMIT_POLICIES: frozenset[OrderPolicy] = frozenset(
    {OrderPolicy.LIMIT_AT_TOUCH, OrderPolicy.LIMIT_IMPROVE}
)


# ---------------------------------------------------------------------------
# Sub-objects
# ---------------------------------------------------------------------------


def combo_count(param_grid: Mapping[str, Sequence[object]]) -> int:
    """Number of combinations the cartesian expansion of ``param_grid`` produces."""

    return math.prod(len(values) for values in param_grid.values())


@dataclass(frozen=True)
class SignalSpec:
    """A bias-node taxonomy key plus a list-valued parameter grid.

    ``param_grid`` uses **list-valued params** that expand to a cartesian grid (the same
    shape the pipeline consumes as ``bias_spec['params']``). A *fixed* (non-swept) parameter
    is a single-element list, e.g. ``{"period": [252]}`` — parsimony favours fixing what need
    not be optimised. The product of the list lengths must not exceed
    :data:`MAX_GRID_COMBOS` (300); this is validated at construction.
    """

    module_name: str
    param_grid: Mapping[str, Sequence[object]]

    def __post_init__(self) -> None:
        if not self.module_name or not self.module_name.strip():
            raise ValueError("SignalSpec.module_name must be a non-empty string.")
        if not self.param_grid:
            raise ValueError("SignalSpec.param_grid must contain at least one parameter.")
        for name, values in self.param_grid.items():
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise ValueError(
                    f"SignalSpec.param_grid['{name}'] must be a list/tuple of values "
                    f"(fixed params use a 1-element list); got {type(values).__name__}."
                )
            if len(values) == 0:
                raise ValueError(
                    f"SignalSpec.param_grid['{name}'] must be a non-empty list."
                )
        combos = combo_count(self.param_grid)
        if combos > MAX_GRID_COMBOS:
            raise ValueError(
                f"SignalSpec param grid expands to {combos} combinations, exceeding the "
                f"cap of {MAX_GRID_COMBOS}. Shrink the grid (parsimony: fix what need not "
                "be swept, use smaller ranges)."
            )

    @property
    def num_combos(self) -> int:
        return combo_count(self.param_grid)


@dataclass(frozen=True)
class ResearchWindows:
    """Train / validation / test windows. The **test window is locked** — never used for
    selection, scored once at the end for the approval decision.
    """

    train: tuple[datetime, datetime]
    validation: tuple[datetime, datetime]
    test: tuple[datetime, datetime]

    def __post_init__(self) -> None:
        for label, window in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            start, end = window
            if start >= end:
                raise ValueError(
                    f"ResearchWindows.{label}: start must be before end "
                    f"(got start={start!s}, end={end!s})."
                )
        if self.train[1] >= self.validation[0]:
            raise ValueError(
                "ResearchWindows: train must end before validation starts "
                f"(train_end={self.train[1]!s}, validation_start={self.validation[0]!s})."
            )
        if self.validation[1] >= self.test[0]:
            raise ValueError(
                "ResearchWindows: validation must end before test starts "
                f"(validation_end={self.validation[1]!s}, test_start={self.test[0]!s})."
            )


@dataclass(frozen=True)
class RiskSpec:
    """Sizing parameters. ``F = target_vol / sigma`` clamped to +/- ``forecast_cap``."""

    target_vol: float = 0.15       # tau in F = tau/sigma
    forecast_cap: float = 2.0      # +/- clamp on F
    max_position_pct: float = 3.5  # per-instrument notional cap (fraction of capital)
    buffer_fraction: float = 0.0   # Carver no-trade band (0 = always rebalance to target)

    def __post_init__(self) -> None:
        if self.target_vol <= 0:
            raise ValueError(f"RiskSpec.target_vol must be > 0, got {self.target_vol}.")
        if self.forecast_cap <= 0:
            raise ValueError(f"RiskSpec.forecast_cap must be > 0, got {self.forecast_cap}.")
        if self.max_position_pct <= 0:
            raise ValueError(
                f"RiskSpec.max_position_pct must be > 0, got {self.max_position_pct}."
            )
        if self.buffer_fraction < 0:
            raise ValueError(
                f"RiskSpec.buffer_fraction must be >= 0, got {self.buffer_fraction}."
            )


@dataclass(frozen=True)
class PropConstraints:
    """Prop-firm limits carried for context. **Informational only — never an auto-gate.**

    Mirrors ``data_platform.providers.mt5.brokers.RiskRules``; all fields optional.
    """

    max_daily_loss_pct: float | None = None
    max_total_loss_pct: float | None = None
    profit_target_pct: float | None = None
    min_trading_days: int | None = None
    max_leverage: int | None = None
    news_trading_restricted: bool = False


@dataclass(frozen=True)
class AccountSpec:
    """Account capital and (informational) prop constraints."""

    capital: float = 50_000.0
    prop_constraints: PropConstraints | None = None

    def __post_init__(self) -> None:
        if self.capital <= 0:
            raise ValueError(f"AccountSpec.capital must be > 0, got {self.capital}.")


@dataclass(frozen=True)
class ExecutionSpec:
    """Engine-A execution: a level signal reached by market or passive limit, no per-trade
    stops. Per-leg policies plus the unfilled-limit fallback and the holding intent.

    ``fill_feed`` is **derived** from the leg policies unless set explicitly: any leg using a
    limit requires a fine feed (M1/tick) or the limit fill rots; both-market resolves to the
    fast vectorized bar-spine lane.
    """

    entry_policy: OrderPolicy = OrderPolicy.MARKET_ON_OPEN   # urgent: take the signal
    exit_policy: OrderPolicy = OrderPolicy.MARKET_ON_OPEN    # patient: a limit can scrape spread
    unfilled_limit: UnfilledLimitPolicy = UnfilledLimitPolicy.CROSS_AFTER
    holding: Holding = Holding.OVERNIGHT
    fill_feed: FillFeed = FillFeed.DERIVED

    def uses_limit(self) -> bool:
        """True if either leg rests a passive limit (so a fine feed is required)."""

        return self.entry_policy in _LIMIT_POLICIES or self.exit_policy in _LIMIT_POLICIES

    def resolved_fill_feed(self) -> FillFeed:
        """The concrete feed: explicit value if set, else derived from the leg policies."""

        if self.fill_feed is not FillFeed.DERIVED:
            return self.fill_feed
        return FillFeed.BARS_M1 if self.uses_limit() else FillFeed.SIGNAL_BAR

    def __post_init__(self) -> None:
        if self.uses_limit() and self.fill_feed is FillFeed.SIGNAL_BAR:
            raise ValueError(
                "ExecutionSpec: a limit leg requires a fine fill_feed (BARS_M1/TICKS) or "
                "DERIVED; SIGNAL_BAR would assume fills that never happened. Leave fill_feed "
                "as DERIVED or set BARS_M1/TICKS."
            )


@dataclass(frozen=True)
class VaultTarget:
    """Where the strategy lands in the vault on approval (used only on promotion)."""

    weight_hierarchy_group: str  # one of the 13 manual sleeves
    ensemble_name: str

    def __post_init__(self) -> None:
        groups = valid_weight_hierarchy_groups()
        if self.weight_hierarchy_group not in groups:
            valid = ", ".join(sorted(groups))
            raise ValueError(
                f"VaultTarget.weight_hierarchy_group '{self.weight_hierarchy_group}' is not "
                f"a configured sleeve. Valid: {valid}."
            )
        if not self.ensemble_name or not self.ensemble_name.strip():
            raise ValueError("VaultTarget.ensemble_name must be a non-empty string.")


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class StrategySpec:
    """The single, flat description of one strategy to research and (if approved) store.

    Constructed keyword-only (``kw_only=True``) so the field order mirrors
    ``strategy_spec.md`` exactly and construction is unambiguous. ``windows=None`` is allowed:
    the adapter fills the per-mode defaults (the test window stays locked out of selection).
    """

    # identity
    name: str
    hypothesis: str
    author: str
    created: datetime
    spec_version: str = "1.0"

    # universe & timeframe
    tickers: tuple[Ticker, ...]
    # ``data_feed`` is an internal label only, kept for back-compat. Signals are always generated
    # on additive futures; the dual-lane (ratio-futures + CFD) RETURN feeds are driven by the
    # executor, not this field. Optional so specs constructed without it (and the legacy JSONs)
    # stay valid.
    data_feed: DataFeed = DataFeed.NORGATE_FUTURES
    mode: StrategyMode
    timeframe: TimeFrame

    # windows (defaulted per mode by the adapter when None)
    windows: ResearchWindows | None = None

    # signal & direction
    signal: SignalSpec
    direction: Direction

    # volatility scaling
    vol_scaling: VolScaling = VolScaling.BLENDED
    vol_scaling_model: VolScalingModel = VolScalingModel.INHERIT_DAILY

    # risk / sizing
    risk: RiskSpec = field(default_factory=RiskSpec)

    # account
    account: AccountSpec = field(default_factory=AccountSpec)

    # execution
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)

    # vault target (used only on approval)
    vault: VaultTarget

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("StrategySpec.name must be a non-empty string.")
        if not self.tickers:
            raise ValueError("StrategySpec.tickers must contain at least one ticker.")
        self._validate_mode_timeframe()

    def _validate_mode_timeframe(self) -> None:
        if self.mode is StrategyMode.DAILY and self.timeframe not in _DAILY_TIMEFRAMES:
            allowed = ", ".join(tf.name for tf in TimeFrame if tf in _DAILY_TIMEFRAMES)
            raise ValueError(
                f"StrategySpec: DAILY mode requires a daily timeframe ({allowed}); "
                f"got {self.timeframe.name}."
            )
        if self.mode is StrategyMode.INTRADAY and self.timeframe not in _INTRADAY_TIMEFRAMES:
            allowed = ", ".join(tf.name for tf in TimeFrame if tf in _INTRADAY_TIMEFRAMES)
            raise ValueError(
                f"StrategySpec: INTRADAY mode requires an intraday timeframe ({allowed}); "
                f"got {self.timeframe.name}. (M15 and finer are not yet TimeFrame members.)"
            )
