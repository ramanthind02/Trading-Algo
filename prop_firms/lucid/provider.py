from __future__ import annotations

from pathlib import Path

from prop_firms.base.config_loader import load_account_definitions
from prop_firms.base.models import AccountDefinition


DEFAULT_LUCID_CONFIG_PATH = Path(__file__).with_name("config.json")


def load_lucid_accounts(
    config_path: Path | None = None,
) -> dict[str, AccountDefinition]:
    """Load all Lucid account definitions from config."""

    resolved_path = DEFAULT_LUCID_CONFIG_PATH if config_path is None else config_path
    return load_account_definitions(resolved_path)


def load_lucid_account(
    account_code: str,
    config_path: Path | None = None,
) -> AccountDefinition:
    """Load one Lucid account definition by account code."""

    accounts = load_lucid_accounts(config_path=config_path)
    if account_code not in accounts:
        raise KeyError(
            f"Unknown Lucid account '{account_code}'. "
            f"Available accounts: {sorted(accounts)}"
        )
    return accounts[account_code]


def create_lucid_simulator(
    config_path: Path | None = None,
) -> "LucidRuleEngine":
    """Factory for the Lucid simulator."""

    from prop_firms.lucid.rules import LucidRuleEngine

    return LucidRuleEngine(
        account_definitions=load_lucid_accounts(config_path=config_path)
    )


def create_lucid_portfolio_simulator(
    config_path: Path | None = None,
) -> "LucidPortfolioSimulator":
    """Factory for the Lucid multi-account portfolio simulator."""

    from prop_firms.lucid.portfolio_simulator import LucidPortfolioSimulator

    return LucidPortfolioSimulator(
        account_definitions=load_lucid_accounts(config_path=config_path)
    )
