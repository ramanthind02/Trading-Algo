from __future__ import annotations

from pathlib import Path

from prop_firms.base.config_loader import load_account_definitions
from prop_firms.base.models import AccountDefinition


DEFAULT_TOPSTEP_CONFIG_PATH = Path(__file__).with_name("config.json")


def load_topstep_accounts(
    config_path: Path | None = None,
) -> dict[str, AccountDefinition]:
    """Load all Topstep account definitions from config."""

    resolved_path = DEFAULT_TOPSTEP_CONFIG_PATH if config_path is None else config_path
    return load_account_definitions(resolved_path)


def load_topstep_account(
    account_code: str,
    config_path: Path | None = None,
) -> AccountDefinition:
    """Load one Topstep account definition by account code."""

    accounts = load_topstep_accounts(config_path=config_path)
    if account_code not in accounts:
        raise KeyError(
            f"Unknown Topstep account '{account_code}'. "
            f"Available accounts: {sorted(accounts)}"
        )
    return accounts[account_code]


def create_topstep_simulator(
    config_path: Path | None = None,
) -> "TopstepRuleEngine":
    """Factory for the Topstep simulator."""

    from prop_firms.topstep.rules import TopstepRuleEngine

    return TopstepRuleEngine(account_definitions=load_topstep_accounts(config_path=config_path))


def create_topstep_portfolio_simulator(
    config_path: Path | None = None,
) -> "TopstepPortfolioSimulator":
    """Factory for the Topstep multi-account portfolio simulator."""

    from prop_firms.topstep.portfolio_simulator import TopstepPortfolioSimulator

    return TopstepPortfolioSimulator(
        account_definitions=load_topstep_accounts(config_path=config_path)
    )
