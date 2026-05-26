from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, cast

from prop_firms.base.models import (
    AccountDefinition,
    AccountFees,
    ConsistencyRule,
    ContractLimit,
    EodTrailingDrawdownRule,
    EvaluationRules,
    FundedRules,
    PayoutCycleRule,
    PayoutRule,
    ProfitDayRule,
    ScalingTier,
)


def load_account_definitions(config_path: Path) -> dict[str, AccountDefinition]:
    """Load provider account definitions from a JSON config file."""

    raw = _load_json_mapping(config_path)
    provider_name = _read_str(raw, "provider_name")
    program_name = _read_str(raw, "program_name")
    accounts = _read_mapping(raw, "accounts")

    return {
        account_code: _build_account_definition(
            provider_name=provider_name,
            program_name=program_name,
            account_code=account_code,
            account_payload=_read_mapping(accounts, account_code),
        )
        for account_code in accounts
    }


def _build_account_definition(
    provider_name: str,
    program_name: str,
    account_code: str,
    account_payload: Mapping[str, object],
) -> AccountDefinition:
    metadata = {
        key: str(value)
        for key, value in _read_optional_mapping(account_payload, "metadata").items()
    }
    return AccountDefinition(
        provider_name=provider_name,
        program_name=program_name,
        account_code=account_code,
        account_size=_read_float(account_payload, "account_size"),
        fees=_build_fees(_read_mapping(account_payload, "fees")),
        evaluation=_build_evaluation_rules(_read_mapping(account_payload, "evaluation")),
        funded=_build_funded_rules(_read_mapping(account_payload, "funded")),
        verification=_build_optional_evaluation_rules(account_payload, "verification"),
        metadata=metadata,
    )


def _build_optional_evaluation_rules(
    payload: Mapping[str, object],
    key: str,
) -> EvaluationRules | None:
    rules_payload = _read_optional_mapping(payload, key)
    return None if not rules_payload else _build_evaluation_rules(rules_payload)


def _build_fees(payload: Mapping[str, object]) -> AccountFees:
    return AccountFees(
        challenge_fee=_read_float(payload, "challenge_fee"),
        reset_fee=_read_float(payload, "reset_fee"),
        activation_fee=_read_float(payload, "activation_fee", 0.0),
        refundable_on_first_payout=_read_bool(payload, "refundable_on_first_payout", False),
    )


def _build_contract_limit(payload: Mapping[str, object]) -> ContractLimit:
    return ContractLimit(
        mini_contracts=_read_int(payload, "mini_contracts"),
        micro_contracts=_read_int(payload, "micro_contracts"),
    )


def _build_drawdown_rule(payload: Mapping[str, object]) -> EodTrailingDrawdownRule:
    return EodTrailingDrawdownRule(
        loss_limit_amount=_read_float(payload, "loss_limit_amount"),
        initial_trail_balance=_read_float(payload, "initial_trail_balance"),
        locked_mll_balance=_read_float(payload, "locked_mll_balance"),
    )


def _build_consistency_rule(payload: Mapping[str, object]) -> ConsistencyRule:
    return ConsistencyRule(
        max_ratio=_read_float(payload, "max_ratio"),
        cushion_multiplier=_read_float(payload, "cushion_multiplier", 1.04),
        inclusive_limit=_read_bool(payload, "inclusive_limit", True),
    )


def _build_profit_day_rule(payload: Mapping[str, object]) -> ProfitDayRule:
    return ProfitDayRule(
        required_days=_read_int(payload, "required_days"),
        minimum_profit_amount=_read_float(payload, "minimum_profit_amount"),
    )


def _build_payout_cycle_rule(payload: Mapping[str, object]) -> PayoutCycleRule:
    return PayoutCycleRule(
        first_cycle_calendar_days=_read_int(payload, "first_cycle_calendar_days"),
        subsequent_cycle_calendar_days=_read_int(payload, "subsequent_cycle_calendar_days"),
    )


def _build_payout_rule(payload: Mapping[str, object]) -> PayoutRule:
    payout_cap_schedule = tuple(
        float(item) for item in _read_optional_list(payload, "payout_cap_schedule")
    )
    cycle_payload = _read_optional_mapping(payload, "payout_cycle_rule")
    return PayoutRule(
        trader_profit_split=_read_float(payload, "trader_profit_split"),
        min_request_amount=_read_float(payload, "min_request_amount"),
        max_profit_share=_read_float(payload, "max_profit_share"),
        max_payout_amount=_read_float(payload, "max_payout_amount"),
        max_requests_per_account=_read_int(payload, "max_requests_per_account"),
        profit_day_rule=_build_profit_day_rule(_read_mapping(payload, "profit_day_rule")),
        protected_balance=_read_optional_float(payload, "protected_balance"),
        consistency_rule=_build_optional_consistency_rule(payload, "consistency_rule"),
        payout_cap_schedule=payout_cap_schedule,
        payout_cycle_rule=(
            None if not cycle_payload else _build_payout_cycle_rule(cycle_payload)
        ),
    )


def _build_scaling_tier(payload: Mapping[str, object]) -> ScalingTier:
    max_profit_raw = payload.get("max_profit")
    max_profit = None if max_profit_raw is None else float(max_profit_raw)
    return ScalingTier(
        min_profit=_read_float(payload, "min_profit"),
        max_profit=max_profit,
        contract_limit=_build_contract_limit(_read_mapping(payload, "contract_limit")),
        daily_loss_limit=_read_optional_float(payload, "daily_loss_limit"),
    )


def _build_evaluation_rules(payload: Mapping[str, object]) -> EvaluationRules:
    return EvaluationRules(
        profit_target_amount=_read_float(payload, "profit_target_amount"),
        drawdown_rule=_build_drawdown_rule(_read_mapping(payload, "drawdown_rule")),
        max_contract_limit=_build_contract_limit(_read_mapping(payload, "max_contract_limit")),
        consistency_rule=_build_optional_consistency_rule(payload, "consistency_rule"),
        minimum_trading_days=_read_int(payload, "minimum_trading_days", 0),
        evaluation_expiry_calendar_days=_read_optional_int(
            payload,
            "evaluation_expiry_calendar_days",
        ),
        daily_loss_limit=_read_optional_float(payload, "daily_loss_limit"),
        max_leverage=_read_optional_float(payload, "max_leverage"),
    )


def _build_funded_rules(payload: Mapping[str, object]) -> FundedRules:
    scaling_plan = tuple(
        _build_scaling_tier(cast(Mapping[str, object], item))
        for item in _read_list(payload, "scaling_plan")
    )
    return FundedRules(
        drawdown_rule=_build_drawdown_rule(_read_mapping(payload, "drawdown_rule")),
        max_contract_limit=_build_contract_limit(_read_mapping(payload, "max_contract_limit")),
        payout_rule=_build_payout_rule(_read_mapping(payload, "payout_rule")),
        scaling_plan=scaling_plan,
        daily_loss_limit=_read_optional_float(payload, "daily_loss_limit"),
        max_leverage=_read_optional_float(payload, "max_leverage"),
    )


def _build_optional_consistency_rule(
    payload: Mapping[str, object],
    key: str,
) -> ConsistencyRule | None:
    rule_payload = _read_optional_mapping(payload, key)
    return None if not rule_payload else _build_consistency_rule(rule_payload)


def _load_json_mapping(config_path: Path) -> Mapping[str, object]:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {config_path}")
    return cast(Mapping[str, object], payload)


def _read_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping for '{key}'")
    return cast(Mapping[str, object], value)


def _read_optional_mapping(
    payload: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping for '{key}'")
    return cast(Mapping[str, object], value)


def _read_list(payload: Mapping[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Expected list for '{key}'")
    return value


def _read_optional_list(payload: Mapping[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Expected list for '{key}'")
    return value


def _read_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Expected string for '{key}'")
    return value


def _read_float(
    payload: Mapping[str, object],
    key: str,
    default: float | None = None,
) -> float:
    value = payload.get(key, default)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected number for '{key}'")
    return float(value)


def _read_optional_float(
    payload: Mapping[str, object],
    key: str,
) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected number for '{key}'")
    return float(value)


def _read_int(
    payload: Mapping[str, object],
    key: str,
    default: int | None = None,
) -> int:
    value = payload.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"Expected integer for '{key}'")
    return value


def _read_optional_int(
    payload: Mapping[str, object],
    key: str,
) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"Expected integer for '{key}'")
    return value


def _read_bool(
    payload: Mapping[str, object],
    key: str,
    default: bool | None = None,
) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean for '{key}'")
    return value
