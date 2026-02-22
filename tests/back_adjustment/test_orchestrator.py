from __future__ import annotations

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from utils.enums import Ticker
from data_cleaning.back_adjustment.orchestrator import (
    AdjustmentMetadata,
    process_ticker,
    build_arg_parser,
)


def _write_synthetic_parquet(path: Path, n_days: int = 200) -> None:
    """Write a synthetic daily-style parquet with a roll gap at day 100."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    closes: list[float] = []
    for i in range(n_days):
        price = 4000.0 + i * 1.0
        if i >= 100:
            price += 50.0
        closes.append(price + rng.randn() * 0.3)
    closes_arr = np.array(closes)
    df = pd.DataFrame({
        "datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": (dates.astype(np.int64) // 10**9).astype(np.uint32),
        "open": closes_arr - rng.rand(n_days) * 0.5,
        "high": closes_arr + rng.rand(n_days),
        "low": closes_arr - rng.rand(n_days),
        "close": closes_arr,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _setup_multi_tf(input_dir: Path, ticker: str = "ES") -> None:
    """Create synthetic files for multiple timeframes."""
    ticker_dir = input_dir / ticker
    for tf in ("M1", "M5", "M15", "H1"):
        _write_synthetic_parquet(ticker_dir / f"{tf}_{ticker}.parquet")


class TestOrchestrator:

    def test_process_single_ticker_multi_tf(self, tmp_path: Path) -> None:
        """process_ticker adjusts all timeframes in the ticker directory."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        metadata_dir = tmp_path / "metadata"

        _setup_multi_tf(input_dir, "ES")

        meta = process_ticker(
            Ticker.ES, input_dir, output_dir, metadata_dir,
            norgate_dir=tmp_path / "norgate",
        )
        assert isinstance(meta, AdjustmentMetadata)
        assert meta.ticker == Ticker.ES
        assert len(meta.timeframes_adjusted) == 4
        assert set(meta.timeframes_adjusted) == {"M1", "M5", "M15", "H1"}

        for tf in ("M1", "M5", "M15", "H1"):
            assert (output_dir / "ES" / f"{tf}_ES.parquet").exists()

    def test_metadata_saved_with_timeframes(self, tmp_path: Path) -> None:
        """Metadata JSON contains timeframes_adjusted list."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        metadata_dir = tmp_path / "metadata"

        _setup_multi_tf(input_dir, "ES")
        process_ticker(
            Ticker.ES, input_dir, output_dir, metadata_dir,
            norgate_dir=tmp_path / "norgate",
        )

        data = json.loads((metadata_dir / "ES.json").read_text())
        assert data["ticker"] == "ES"
        assert "timeframes_adjusted" in data
        assert len(data["timeframes_adjusted"]) == 4
        assert "num_rolls_detected" in data

    def test_same_adjustments_across_timeframes(self, tmp_path: Path) -> None:
        """All timeframes get the same adjustment (0 after last roll)."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        metadata_dir = tmp_path / "metadata"

        _setup_multi_tf(input_dir, "ES")
        process_ticker(
            Ticker.ES, input_dir, output_dir, metadata_dir,
            norgate_dir=tmp_path / "norgate",
        )

        for tf in ("M1", "H1"):
            orig = pd.read_parquet(input_dir / "ES" / f"{tf}_ES.parquet")
            adj = pd.read_parquet(output_dir / "ES" / f"{tf}_ES.parquet")
            # After most recent roll: adjustment = 0
            last_diff = abs(adj["close"].iloc[-1] - orig["close"].iloc[-1])
            assert last_diff < 1e-6, f"{tf}: last row should have 0 adjustment"

    def test_cli_single_ticker(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["--ticker", "ES"])
        assert args.ticker == "ES"
        assert args.all is False

    def test_cli_all_tickers(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["--all"])
        assert args.all is True

    def test_determinism(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        _setup_multi_tf(input_dir, "ES")
        norgate = tmp_path / "norgate"

        out_a, meta_a = tmp_path / "out_a", tmp_path / "meta_a"
        out_b, meta_b = tmp_path / "out_b", tmp_path / "meta_b"

        process_ticker(Ticker.ES, input_dir, out_a, meta_a, norgate)
        process_ticker(Ticker.ES, input_dir, out_b, meta_b, norgate)

        for tf in ("M1", "M5", "M15", "H1"):
            df_a = pd.read_parquet(out_a / "ES" / f"{tf}_ES.parquet")
            df_b = pd.read_parquet(out_b / "ES" / f"{tf}_ES.parquet")
            pd.testing.assert_frame_equal(df_a, df_b)

    def test_missing_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            process_ticker(
                Ticker.ES,
                tmp_path / "nonexistent",
                tmp_path / "output",
                tmp_path / "metadata",
            )
