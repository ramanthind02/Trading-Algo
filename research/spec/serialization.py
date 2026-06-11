"""JSON (de)serialization for :class:`~research.spec.strategy_spec.StrategySpec`.

This is the **round-trippable contract** shared by the two authors of a strategy spec:

* an **agent** (in a chat) writes a ``research/specs/<name>.json`` file, and
* the **frontend Spec Builder** loads that JSON into a form, edits it, and saves it back.

A spec written by either path deserializes to the identical :class:`StrategySpec`, so the
``__post_init__`` validation (grid cap, window ordering, sleeve membership, fill-feed
consistency, mode/timeframe) runs on load regardless of who produced the file.

Design
------
* **Dependency-light**, like :mod:`research.spec.strategy_spec`: only stdlib + the spec module +
  the domain enums. No pipeline / Nautilus imports.
* **Human-readable tokens.** Enums serialize to their string ``value`` where that is a string
  (e.g. ``"blended"``, ``"long_short"``); :class:`~lib.core.enums.Ticker` and
  :class:`~lib.core.enums.TimeFrame` serialize to their ``name`` (Ticker names are the canonical
  symbols used everywhere via ``Ticker[name]``; TimeFrame values are integers).
* **Tolerant on read.** :func:`spec_from_dict` accepts either the value or the name for every
  enum, so a hand-edited file is forgiving.
* **Stable field order** mirrors ``strategy_spec.md`` / the dataclass, so a saved file reads
  top-to-bottom like the spec doc and diffs cleanly.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Type, TypeVar

from lib.core.enums import Direction, Ticker, TimeFrame
from research.spec.strategy_spec import (
    AccountSpec,
    DataFeed,
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
    UnfilledLimitPolicy,
    VaultTarget,
    VolScaling,
    VolScalingModel,
)

__all__ = [
    "spec_to_dict",
    "spec_from_dict",
    "spec_to_json",
    "spec_from_json",
    "load_spec",
    "save_spec",
]

_EnumT = TypeVar("_EnumT", bound=Enum)


# ---------------------------------------------------------------------------
# Enum / scalar helpers
# ---------------------------------------------------------------------------


def _enum_token(member: Enum) -> str:
    """The serialized token for an enum member.

    Uses the ``value`` when it is a string (human-readable, e.g. ``"blended"``); otherwise the
    ``name`` (TimeFrame values are integer seconds, so they serialize by name).
    """

    return member.value if isinstance(member.value, str) else member.name


def _enum_from(enum_cls: Type[_EnumT], token: object, *, field: str) -> _EnumT:
    """Resolve an enum member from a token, accepting either its value or its name."""

    if isinstance(token, enum_cls):
        return token
    try:
        return enum_cls(token)  # by value
    except ValueError:
        pass
    if isinstance(token, str):
        try:
            return enum_cls[token]  # by name
        except KeyError:
            pass
    valid = ", ".join(_enum_token(member) for member in enum_cls)
    raise ValueError(
        f"{field}: {token!r} is not a valid {enum_cls.__name__} (expected one of: {valid})."
    )


def _require(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"StrategySpec JSON is missing required field '{key}'.")
    return data[key]


def _data_feed_from(token: object) -> DataFeed:
    """Resolve the (now optional, label-only) ``data_feed`` token.

    Signals are always generated on additive futures now; ``data_feed`` is kept on the spec only
    as a back-compat label and never drives feed selection (the dual-lane return feeds are driven
    by the executor). So reading tolerates the field being present **or absent**, and an
    unknown/legacy token degrades to the futures default rather than raising.
    """

    if token is None:
        return DataFeed.NORGATE_FUTURES
    try:
        return _enum_from(DataFeed, token, field="data_feed")
    except ValueError:
        return DataFeed.NORGATE_FUTURES


def _parse_dt(value: object, *, field: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field}: expected an ISO date/datetime string, got {type(value).__name__}.")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field}: {value!r} is not an ISO date/datetime.") from exc


# ---------------------------------------------------------------------------
# Sub-object (de)serializers
# ---------------------------------------------------------------------------


def _window_to_dict(window: tuple[datetime, datetime]) -> dict[str, str]:
    start, end = window
    return {"start": start.isoformat(), "end": end.isoformat()}


def _window_from(data: object, *, field: str) -> tuple[datetime, datetime]:
    if isinstance(data, Mapping):
        return (
            _parse_dt(_require(data, "start"), field=f"{field}.start"),
            _parse_dt(_require(data, "end"), field=f"{field}.end"),
        )
    if isinstance(data, (list, tuple)) and len(data) == 2:
        return (_parse_dt(data[0], field=f"{field}[0]"), _parse_dt(data[1], field=f"{field}[1]"))
    raise ValueError(f"{field}: expected {{start, end}} or a [start, end] pair.")


def _windows_to_dict(windows: ResearchWindows | None) -> dict[str, Any] | None:
    if windows is None:
        return None
    return {
        "train": _window_to_dict(windows.train),
        "validation": _window_to_dict(windows.validation),
        "test": _window_to_dict(windows.test),
    }


def _windows_from(data: object) -> ResearchWindows | None:
    if data is None:
        return None
    if not isinstance(data, Mapping):
        raise ValueError("windows: expected an object with train/validation/test or null.")
    return ResearchWindows(
        train=_window_from(_require(data, "train"), field="windows.train"),
        validation=_window_from(_require(data, "validation"), field="windows.validation"),
        test=_window_from(_require(data, "test"), field="windows.test"),
    )


def _signal_to_dict(signal: SignalSpec) -> dict[str, Any]:
    return {
        "module_name": signal.module_name,
        "param_grid": {name: list(values) for name, values in signal.param_grid.items()},
    }


def _signal_from(data: object) -> SignalSpec:
    if not isinstance(data, Mapping):
        raise ValueError("signal: expected an object with module_name and param_grid.")
    grid = _require(data, "param_grid")
    if not isinstance(grid, Mapping):
        raise ValueError("signal.param_grid: expected an object of {param: [values]}.")
    return SignalSpec(
        module_name=str(_require(data, "module_name")),
        param_grid={str(name): list(values) for name, values in grid.items()},
    )


def _risk_to_dict(risk: RiskSpec) -> dict[str, Any]:
    return {
        "target_vol": risk.target_vol,
        "forecast_cap": risk.forecast_cap,
        "max_position_pct": risk.max_position_pct,
        "buffer_fraction": risk.buffer_fraction,
    }


def _risk_from(data: object) -> RiskSpec:
    if data is None:
        return RiskSpec()
    if not isinstance(data, Mapping):
        raise ValueError("risk: expected an object or null.")
    return RiskSpec(
        target_vol=float(data.get("target_vol", RiskSpec.target_vol)),
        forecast_cap=float(data.get("forecast_cap", RiskSpec.forecast_cap)),
        max_position_pct=float(data.get("max_position_pct", RiskSpec.max_position_pct)),
        buffer_fraction=float(data.get("buffer_fraction", RiskSpec.buffer_fraction)),
    )


def _prop_to_dict(prop: PropConstraints | None) -> dict[str, Any] | None:
    if prop is None:
        return None
    return {
        "max_daily_loss_pct": prop.max_daily_loss_pct,
        "max_total_loss_pct": prop.max_total_loss_pct,
        "profit_target_pct": prop.profit_target_pct,
        "min_trading_days": prop.min_trading_days,
        "max_leverage": prop.max_leverage,
        "news_trading_restricted": prop.news_trading_restricted,
    }


def _prop_from(data: object) -> PropConstraints | None:
    if data is None:
        return None
    if not isinstance(data, Mapping):
        raise ValueError("account.prop_constraints: expected an object or null.")
    return PropConstraints(
        max_daily_loss_pct=data.get("max_daily_loss_pct"),
        max_total_loss_pct=data.get("max_total_loss_pct"),
        profit_target_pct=data.get("profit_target_pct"),
        min_trading_days=data.get("min_trading_days"),
        max_leverage=data.get("max_leverage"),
        news_trading_restricted=bool(data.get("news_trading_restricted", False)),
    )


def _account_to_dict(account: AccountSpec) -> dict[str, Any]:
    return {
        "capital": account.capital,
        "prop_constraints": _prop_to_dict(account.prop_constraints),
    }


def _account_from(data: object) -> AccountSpec:
    if data is None:
        return AccountSpec()
    if not isinstance(data, Mapping):
        raise ValueError("account: expected an object or null.")
    return AccountSpec(
        capital=float(data.get("capital", AccountSpec.capital)),
        prop_constraints=_prop_from(data.get("prop_constraints")),
    )


def _execution_to_dict(execution: ExecutionSpec) -> dict[str, Any]:
    return {
        "entry_policy": _enum_token(execution.entry_policy),
        "exit_policy": _enum_token(execution.exit_policy),
        "unfilled_limit": _enum_token(execution.unfilled_limit),
        "holding": _enum_token(execution.holding),
        "fill_feed": _enum_token(execution.fill_feed),
    }


def _execution_from(data: object) -> ExecutionSpec:
    if data is None:
        return ExecutionSpec()
    if not isinstance(data, Mapping):
        raise ValueError("execution: expected an object or null.")
    return ExecutionSpec(
        entry_policy=_enum_from(
            OrderPolicy, data.get("entry_policy", OrderPolicy.MARKET_ON_OPEN), field="execution.entry_policy"
        ),
        exit_policy=_enum_from(
            OrderPolicy, data.get("exit_policy", OrderPolicy.MARKET_ON_OPEN), field="execution.exit_policy"
        ),
        unfilled_limit=_enum_from(
            UnfilledLimitPolicy,
            data.get("unfilled_limit", UnfilledLimitPolicy.CROSS_AFTER),
            field="execution.unfilled_limit",
        ),
        holding=_enum_from(Holding, data.get("holding", Holding.OVERNIGHT), field="execution.holding"),
        fill_feed=_enum_from(FillFeed, data.get("fill_feed", FillFeed.DERIVED), field="execution.fill_feed"),
    )


def _vault_to_dict(vault: VaultTarget) -> dict[str, Any]:
    return {
        "weight_hierarchy_group": vault.weight_hierarchy_group,
        "ensemble_name": vault.ensemble_name,
    }


def _vault_from(data: object) -> VaultTarget:
    if not isinstance(data, Mapping):
        raise ValueError("vault: expected an object with weight_hierarchy_group and ensemble_name.")
    return VaultTarget(
        weight_hierarchy_group=str(_require(data, "weight_hierarchy_group")),
        ensemble_name=str(_require(data, "ensemble_name")),
    )


# ---------------------------------------------------------------------------
# Top-level (de)serialization
# ---------------------------------------------------------------------------


def spec_to_dict(spec: StrategySpec) -> dict[str, Any]:
    """Serialize a :class:`StrategySpec` to a JSON-compatible dict (stable field order)."""

    return {
        "spec_version": spec.spec_version,
        "name": spec.name,
        "hypothesis": spec.hypothesis,
        "author": spec.author,
        "created": spec.created.isoformat(),
        "tickers": [ticker.name for ticker in spec.tickers],
        "data_feed": _enum_token(spec.data_feed),
        "mode": _enum_token(spec.mode),
        "timeframe": spec.timeframe.name,
        "windows": _windows_to_dict(spec.windows),
        "signal": _signal_to_dict(spec.signal),
        "direction": _enum_token(spec.direction),
        "vol_scaling": _enum_token(spec.vol_scaling),
        "vol_scaling_model": _enum_token(spec.vol_scaling_model),
        "risk": _risk_to_dict(spec.risk),
        "account": _account_to_dict(spec.account),
        "execution": _execution_to_dict(spec.execution),
        "vault": _vault_to_dict(spec.vault),
    }


def spec_from_dict(data: Mapping[str, Any]) -> StrategySpec:
    """Reconstruct a :class:`StrategySpec` from a dict, running all spec validation."""

    if not isinstance(data, Mapping):
        raise ValueError("StrategySpec JSON must be an object.")

    tickers_raw = _require(data, "tickers")
    if not isinstance(tickers_raw, (list, tuple)):
        raise ValueError("tickers: expected a list of ticker names.")
    tickers = tuple(_enum_from(Ticker, token, field="tickers[]") for token in tickers_raw)

    return StrategySpec(
        name=str(_require(data, "name")),
        hypothesis=str(data.get("hypothesis", "")),
        author=str(data.get("author", "")),
        created=_parse_dt(data.get("created", datetime.now().isoformat()), field="created"),
        spec_version=str(data.get("spec_version", "1.0")),
        tickers=tickers,
        data_feed=_data_feed_from(data.get("data_feed")),
        mode=_enum_from(StrategyMode, _require(data, "mode"), field="mode"),
        timeframe=_enum_from(TimeFrame, _require(data, "timeframe"), field="timeframe"),
        windows=_windows_from(data.get("windows")),
        signal=_signal_from(_require(data, "signal")),
        direction=_enum_from(Direction, _require(data, "direction"), field="direction"),
        vol_scaling=_enum_from(VolScaling, data.get("vol_scaling", VolScaling.BLENDED), field="vol_scaling"),
        vol_scaling_model=_enum_from(
            VolScalingModel,
            data.get("vol_scaling_model", VolScalingModel.INHERIT_DAILY),
            field="vol_scaling_model",
        ),
        risk=_risk_from(data.get("risk")),
        account=_account_from(data.get("account")),
        execution=_execution_from(data.get("execution")),
        vault=_vault_from(_require(data, "vault")),
    )


def spec_to_json(spec: StrategySpec, *, indent: int = 2) -> str:
    """Serialize a :class:`StrategySpec` to a JSON string."""

    return json.dumps(spec_to_dict(spec), indent=indent)


def spec_from_json(text: str) -> StrategySpec:
    """Parse a JSON string into a validated :class:`StrategySpec`."""

    return spec_from_dict(json.loads(text))


def load_spec(path: str | Path) -> StrategySpec:
    """Load + validate a :class:`StrategySpec` from a JSON file (UTF-8, BOM-tolerant)."""

    return spec_from_json(Path(path).read_text(encoding="utf-8-sig"))


def save_spec(spec: StrategySpec, path: str | Path, *, indent: int = 2) -> Path:
    """Write a :class:`StrategySpec` to a JSON file (creating parent dirs). Returns the path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(spec_to_json(spec, indent=indent) + "\n", encoding="utf-8")
    return target
