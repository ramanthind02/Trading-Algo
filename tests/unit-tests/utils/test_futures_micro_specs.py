"""Unit tests for canonical micro-futures specs and contract sizing."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from execution.position_sizer import PositionSizer
from scripts import enigma_live_forecast as enigma_live
from lib.core.futures_micro_specs import (
    ListedMicroFuturesSpec,
    canonical_listed_micro_futures,
    listed_micro_futures_row,
    micro_contract_fractional_and_whole,
)


class TestCanonicalTable:
    def test_keys_cover_prop_universe(self) -> None:
        table = canonical_listed_micro_futures()
        for k in ("ES", "NQ", "YM", "RTY", "GC", "TLT"):
            assert k in table

    def test_micro_dollars_per_point_match_cme_micros(self) -> None:
        table = canonical_listed_micro_futures()
        assert table["ES"].micro_symbol == "MES" and table["ES"].micro_dollars_per_point == 5.0
        assert table["NQ"].micro_symbol == "MNQ" and table["NQ"].micro_dollars_per_point == 2.0
        assert table["GC"].micro_symbol == "MGC" and table["GC"].micro_dollars_per_point == 10.0
        assert table["YM"].micro_symbol == "MYM" and table["YM"].micro_dollars_per_point == 0.5
        assert table["RTY"].micro_symbol == "M2K" and table["RTY"].micro_dollars_per_point == 5.0

    def test_mini_is_tenfold_where_applicable(self) -> None:
        table = canonical_listed_micro_futures()
        for t in ("ES", "NQ", "YM", "RTY", "GC"):
            row = table[t]
            assert row.mini_dollars_per_point == pytest.approx(10.0 * row.micro_dollars_per_point)

    def test_listed_row_unknown_returns_none(self) -> None:
        assert listed_micro_futures_row("CL") is None

    def test_invalid_dataclass_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ListedMicroFuturesSpec(
                micro_symbol="X",
                mini_symbol="Y",
                micro_dollars_per_point=0.0,
                mini_dollars_per_point=1.0,
                exchange="CME",
                illustrative_margin_long_usd=0.0,
                illustrative_margin_short_usd=0.0,
            )


class TestMicroContractFractionalAndWhole:
    def test_mes_one_contract(self) -> None:
        # 5000 * 5 = 25_000 notional; 25% of 100k = 25k → 1 MES
        frac, whole = micro_contract_fractional_and_whole(
            futures_index_price=5_000.0,
            position_fraction=0.25,
            capital_usd=100_000.0,
            micro_dollars_per_point=5.0,
        )
        assert frac == pytest.approx(1.0)
        assert whole == 1

    def test_mnq_two_contracts(self) -> None:
        # 20_000 * 2 = 40_000; 80k target → 2 MNQ
        frac, whole = micro_contract_fractional_and_whole(
            futures_index_price=20_000.0,
            position_fraction=0.80,
            capital_usd=100_000.0,
            micro_dollars_per_point=2.0,
        )
        assert frac == pytest.approx(2.0)
        assert whole == 2

    def test_short_negative_rounds_toward_zero(self) -> None:
        frac, whole = micro_contract_fractional_and_whole(
            futures_index_price=5_000.0,
            position_fraction=-0.25,
            capital_usd=100_000.0,
            micro_dollars_per_point=5.0,
        )
        assert frac == pytest.approx(-1.0)
        assert whole == -1

    def test_zero_price_returns_zero_contracts(self) -> None:
        frac, whole = micro_contract_fractional_and_whole(
            futures_index_price=0.0,
            position_fraction=1.0,
            capital_usd=100_000.0,
            micro_dollars_per_point=5.0,
        )
        assert frac == 0.0 and whole == 0


class TestPositionSizerFromListedMicro:
    def test_matches_core_helper(self) -> None:
        prices = {"ES": 5_000.0, "NQ": 20_000.0}
        sizer = PositionSizer.from_listed_micro(100_000.0, prices)
        df = pd.DataFrame(
            {
                "ticker": ["ES", "NQ"],
                "forecast_score": [1.0, 1.0],
                "position_fraction": [0.25, 0.80],
            }
        )
        out = sizer.calculate_positions(df)
        es = int(out.loc[out["ticker"] == "ES", "contracts"].iloc[0])
        nq = int(out.loc[out["ticker"] == "NQ", "contracts"].iloc[0])
        assert es == micro_contract_fractional_and_whole(
            futures_index_price=5_000.0,
            position_fraction=0.25,
            capital_usd=100_000.0,
            micro_dollars_per_point=5.0,
        )[1]
        assert nq == micro_contract_fractional_and_whole(
            futures_index_price=20_000.0,
            position_fraction=0.80,
            capital_usd=100_000.0,
            micro_dollars_per_point=2.0,
        )[1]

    def test_research_tickers_filter(self) -> None:
        prices = {"ES": 5_000.0, "NQ": 20_000.0}
        sizer = PositionSizer.from_listed_micro(
            100_000.0, prices, research_tickers=["ES"]
        )
        assert set(sizer.contract_specs.keys()) == {"ES"}

    def test_empty_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="from_listed_micro"):
            PositionSizer.from_listed_micro(100_000.0, {"CL": 70.0})


class TestEnigmaCalculateFuturesContracts:
    def test_prop_table_parity_with_helper(self) -> None:
        positions_df = pd.DataFrame(
            {
                "ticker": ["ES", "NQ"],
                "datetime": [datetime(2026, 1, 1)] * 2,
                "forecast_score": [0.5, 0.5],
                "position_fraction": [0.25, 0.80],
            }
        )
        prices = {"ES": 5_000.0, "NQ": 20_000.0}
        capital = 100_000.0
        result = enigma_live.calculate_futures_contracts(positions_df, prices, capital)
        assert len(result) == 2
        es_row = result[result["ticker"] == "ES"].iloc[0]
        assert es_row["contract_symbol"] == "MES"
        assert es_row["contracts_whole"] == 1
        assert es_row["point_value"] == 5.0
        nq_row = result[result["ticker"] == "NQ"].iloc[0]
        assert nq_row["contract_symbol"] == "MNQ"
        assert nq_row["contracts_whole"] == 2
