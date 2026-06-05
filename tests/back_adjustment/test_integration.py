"""Integration test: run full pipeline on real ES data if available."""
from __future__ import annotations

import pytest
import pandas as pd
from pathlib import Path

from utils.core.enums import Ticker
from data_platform.providers.norgate.backadjust.orchestrator import process_ticker
from data_platform.providers.norgate.backadjust.validator import compare_price_levels

_INTRADAY_DIR = Path("data/intraday_original")
_NORGATE_ADJ = Path("data/norgate/continuous_futures/adjusted")
_HAS_REAL_DATA = (_INTRADAY_DIR / "ES").is_dir() and any(
    (_INTRADAY_DIR / "ES").glob("M1_ES.parquet")
)
_HAS_NORGATE = (_NORGATE_ADJ / "ES.parquet").exists()


@pytest.mark.skipif(not _HAS_REAL_DATA, reason="No real intraday data available")
class TestEndToEndES:

    def test_process_es_all_timeframes(self, tmp_path: Path) -> None:
        """Process real ES data through the pipeline for all timeframes."""
        output_dir = tmp_path / "adjusted"
        metadata_dir = tmp_path / "metadata"

        meta = process_ticker(Ticker.ES, _INTRADAY_DIR, output_dir, metadata_dir)

        assert meta.num_rolls_detected > 0
        assert (output_dir / "ES").is_dir()
        assert (metadata_dir / "ES.json").exists()
        # ES should have ~60 quarterly rolls
        assert meta.num_rolls_detected > 40
        assert meta.num_rolls_detected < 100
        # Should have adjusted multiple timeframes
        assert len(meta.timeframes_adjusted) > 1

    @pytest.mark.skipif(not _HAS_NORGATE, reason="No Norgate data available")
    def test_compare_es_with_norgate(self, tmp_path: Path) -> None:
        """Compare pipeline-adjusted ES M1 against Norgate daily data."""
        output_dir = tmp_path / "adjusted"
        metadata_dir = tmp_path / "metadata"
        process_ticker(Ticker.ES, _INTRADAY_DIR, output_dir, metadata_dir)

        # Resample adjusted M1 to daily for comparison
        adj_m1 = pd.read_parquet(output_dir / "ES" / "M1_ES.parquet")
        adj_m1["_date"] = pd.to_datetime(adj_m1["datetime"]).dt.normalize()
        daily = adj_m1.groupby("_date").agg(
            close=("close", "last"),
        ).reset_index().rename(columns={"_date": "datetime"})

        norgate = pd.read_parquet(_NORGATE_ADJ / "ES.parquet")

        stats = compare_price_levels(daily, norgate)
        assert stats["correlation"] > 0.99, (
            f"Correlation too low: {stats['correlation']:.4f}"
        )
