from __future__ import annotations

import pytest
from lib.core.enums import Ticker
from data_platform.providers.norgate.backadjust.roll_rules import (
    RollRule,
    get_roll_rule,
    get_all_roll_rules,
)


class TestRollRules:

    def test_all_tickers_have_rules(self) -> None:
        # Only futures tickers have roll rules; FX/CFD tickers (e.g. AUDNZD)
        # are intentionally excluded — they have no contract expiration.
        rules = get_all_roll_rules()
        for ticker, rule in rules.items():
            assert isinstance(rule, RollRule), f"Bad rule for {ticker.name}"

    def test_get_roll_rule_lookup(self) -> None:
        rule = get_roll_rule(Ticker.ES)
        assert rule.ticker == Ticker.ES
        assert rule.rollover_offset == -5
        assert rule.reference_point == "expiration"
        assert isinstance(rule.description, str)
        assert len(rule.description) > 0

    def test_roll_rule_immutability(self) -> None:
        rule = get_roll_rule(Ticker.ES)
        with pytest.raises(AttributeError):
            rule.rollover_offset = 99

    def test_get_roll_rule_invalid_ticker(self) -> None:
        with pytest.raises(ValueError, match="No roll rule defined"):
            get_roll_rule("NOT_A_TICKER")

    def test_reference_point_values(self) -> None:
        for rule in get_all_roll_rules().values():
            assert rule.reference_point in ("expiration", "month_end"), (
                f"{rule.ticker.name} has invalid reference_point: {rule.reference_point}"
            )

    def test_rollover_offset_sign_convention(self) -> None:
        for rule in get_all_roll_rules().values():
            if rule.reference_point == "expiration":
                assert rule.rollover_offset < 0, (
                    f"{rule.ticker.name}: expiration rule should have negative offset"
                )
            else:
                assert rule.rollover_offset > 0, (
                    f"{rule.ticker.name}: month_end rule should have positive offset"
                )
