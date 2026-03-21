from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prop_firms.base.models import ContractLimit, DayInput


_OPTIONAL_EXPOSURE_COLUMNS: frozenset[str] = frozenset(
    {"mini_contracts", "micro_contracts", "notional_exposure"}
)


@dataclass(frozen=True)
class ExposureCheck:
    """Derived exposure metrics for a single day."""

    mini_contracts: int
    micro_contracts: int
    mini_equivalent_contracts: float
    leverage: float | None
    contract_limit_breached: bool
    leverage_limit_breached: bool


def normalize_held_contracts(
    held_contracts: pd.DataFrame | None,
    returns_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Normalize optional held-contract data onto the returns index."""

    if held_contracts is None:
        return pd.DataFrame(
            {
                "mini_contracts": [0] * len(returns_index),
                "micro_contracts": [0] * len(returns_index),
                "notional_exposure": [np.nan] * len(returns_index),
            },
            index=returns_index,
        )

    source = held_contracts.copy()
    if "trading_day" in source.columns:
        source["trading_day"] = pd.to_datetime(source["trading_day"])
        source = source.set_index("trading_day")

    if not isinstance(source.index, pd.DatetimeIndex):
        raise TypeError(
            "held_contracts must use a DatetimeIndex or contain a 'trading_day' column"
        )

    unknown_columns = set(source.columns) - _OPTIONAL_EXPOSURE_COLUMNS
    if unknown_columns:
        raise ValueError(
            "held_contracts contains unsupported columns: "
            f"{sorted(unknown_columns)}"
        )

    aligned = source.reindex(returns_index).fillna(0.0)
    for column in ("mini_contracts", "micro_contracts"):
        if column in aligned.columns:
            aligned[column] = aligned[column].round().astype(int)

    if "mini_contracts" not in aligned.columns:
        aligned["mini_contracts"] = 0
    if "micro_contracts" not in aligned.columns:
        aligned["micro_contracts"] = 0
    if "notional_exposure" not in aligned.columns:
        aligned["notional_exposure"] = np.nan

    return aligned.loc[:, ["mini_contracts", "micro_contracts", "notional_exposure"]]


def build_day_inputs(
    returns: pd.Series,
    held_contracts: pd.DataFrame,
) -> tuple[DayInput, ...]:
    """Align returns and held contracts into immutable day inputs."""

    return tuple(
        DayInput(
            trading_day=pd.Timestamp(trading_day),
            daily_return=float(daily_return),
            mini_contracts=int(held_contracts.loc[trading_day, "mini_contracts"]),
            micro_contracts=int(held_contracts.loc[trading_day, "micro_contracts"]),
            notional_exposure=_coerce_optional_float(
                held_contracts.loc[trading_day, "notional_exposure"]
            ),
        )
        for trading_day, daily_return in returns.items()
    )


def evaluate_exposure(
    day_input: DayInput,
    contract_limit: ContractLimit,
    account_balance: float,
    max_leverage: float | None,
) -> ExposureCheck:
    """Compute daily contract and leverage constraints."""

    mini_contracts = abs(day_input.mini_contracts)
    micro_contracts = abs(day_input.micro_contracts)
    mini_equivalent = float(mini_contracts) + (float(micro_contracts) / 10.0)
    leverage = _compute_leverage(day_input.notional_exposure, account_balance)

    contract_limit_breached = any(
        (
            mini_contracts > contract_limit.mini_contracts,
            micro_contracts > contract_limit.micro_contracts,
            mini_equivalent > contract_limit.mini_equivalent_limit,
        )
    )
    leverage_limit_breached = (
        leverage is not None
        and max_leverage is not None
        and leverage > max_leverage
    )

    return ExposureCheck(
        mini_contracts=mini_contracts,
        micro_contracts=micro_contracts,
        mini_equivalent_contracts=mini_equivalent,
        leverage=leverage,
        contract_limit_breached=contract_limit_breached,
        leverage_limit_breached=leverage_limit_breached,
    )


def _coerce_optional_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _compute_leverage(
    notional_exposure: float | None,
    account_balance: float,
) -> float | None:
    if notional_exposure is None:
        return None
    if account_balance <= 0.0:
        return float("inf")
    return abs(notional_exposure) / account_balance
