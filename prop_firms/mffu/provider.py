from __future__ import annotations

from pathlib import Path

from prop_firms.base.config_loader import load_account_definitions
from prop_firms.base.models import AccountDefinition


DEFAULT_MFFU_CONFIG_PATH = Path(__file__).with_name("config.json")


def load_mffu_accounts(
    config_path: Path | None = None,
) -> dict[str, AccountDefinition]:
    """Load all MFFU account definitions from config."""

    resolved_path = DEFAULT_MFFU_CONFIG_PATH if config_path is None else config_path
    return load_account_definitions(resolved_path)


def load_mffu_account(
    account_code: str,
    config_path: Path | None = None,
) -> AccountDefinition:
    """Load one MFFU account definition by account code."""

    accounts = load_mffu_accounts(config_path=config_path)
    if account_code not in accounts:
        raise KeyError(
            f"Unknown MFFU account '{account_code}'. "
            f"Available accounts: {sorted(accounts)}"
        )
    return accounts[account_code]


def create_mffu_simulator(
    config_path: Path | None = None,
) -> "MffuRuleEngine":
    """Factory for the MFFU simulator."""

    from prop_firms.mffu.rules import MffuRuleEngine

    return MffuRuleEngine(account_definitions=load_mffu_accounts(config_path=config_path))


def create_mffu_portfolio_simulator(
    config_path: Path | None = None,
) -> "MffuPortfolioSimulator":
    """Factory for the MFFU multi-account portfolio simulator."""

    from prop_firms.mffu.portfolio_simulator import MffuPortfolioSimulator

    return MffuPortfolioSimulator(
        account_definitions=load_mffu_accounts(config_path=config_path)
    )
