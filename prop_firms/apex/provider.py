from __future__ import annotations

from pathlib import Path

from prop_firms.base.config_loader import load_account_definitions
from prop_firms.base.models import AccountDefinition


DEFAULT_APEX_CONFIG_PATH = Path(__file__).with_name("config.json")


def load_apex_accounts(
    config_path: Path | None = None,
) -> dict[str, AccountDefinition]:
    """Load all Apex account definitions from config."""

    resolved_path = DEFAULT_APEX_CONFIG_PATH if config_path is None else config_path
    return load_account_definitions(resolved_path)


def load_apex_account(
    account_code: str,
    config_path: Path | None = None,
) -> AccountDefinition:
    """Load one Apex account definition by account code."""

    accounts = load_apex_accounts(config_path=config_path)
    if account_code not in accounts:
        raise KeyError(
            f"Unknown Apex account '{account_code}'. "
            f"Available accounts: {sorted(accounts)}"
        )
    return accounts[account_code]


def create_apex_simulator(
    config_path: Path | None = None,
) -> "ApexRuleEngine":
    """Factory for the Apex simulator."""

    from prop_firms.apex.rules import ApexRuleEngine

    return ApexRuleEngine(account_definitions=load_apex_accounts(config_path=config_path))


def create_apex_portfolio_simulator(
    config_path: Path | None = None,
) -> "ApexPortfolioSimulator":
    """Factory for the Apex multi-account portfolio simulator."""

    from prop_firms.apex.portfolio_simulator import ApexPortfolioSimulator

    return ApexPortfolioSimulator(
        account_definitions=load_apex_accounts(config_path=config_path)
    )
