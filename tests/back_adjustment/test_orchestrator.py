from __future__ import annotations

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from utils.enums import Ticker
from data_cleaning.back_adjustment.orchestrator import (
    AdjustmentMetadata,
    process_ticker,
    build_arg_parser,
)


def _write_synthetic_parquet(path: Path, n_days: int = 200) -> None:
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


class TestOrchestrator:

    def test_process_single_ticker(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        metadata_dir = tmp_path / "metadata"
        _write_synthetic_parquet(input_dir / "ES.parquet")
        meta = process_ticker(Ticker.ES, input_dir, output_dir, metadata_dir)
        assert isinstance(meta, AdjustmentMetadata)
        assert meta.ticker == Ticker.ES
        assert (output_dir / "ES.parquet").exists()
        assert meta.num_rolls_detected >= 0

    def test_metadata_saved(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        metadata_dir = tmp_path / "metadata"
        _write_synthetic_parquet(input_dir / "ES.parquet")
        process_ticker(Ticker.ES, input_dir, output_dir, metadata_dir)
        meta_path = metadata_dir / "ES.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert data["ticker"] == "ES"
        assert "num_rolls_detected" in data
        assert "processing_date" in data

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
        _write_synthetic_parquet(input_dir / "ES.parquet")
        out_a = tmp_path / "out_a"
        meta_a = tmp_path / "meta_a"
        out_b = tmp_path / "out_b"
        meta_b = tmp_path / "meta_b"
        process_ticker(Ticker.ES, input_dir, out_a, meta_a)
        process_ticker(Ticker.ES, input_dir, out_b, meta_b)
        df_a = pd.read_parquet(out_a / "ES.parquet")
        df_b = pd.read_parquet(out_b / "ES.parquet")
        pd.testing.assert_frame_equal(df_a, df_b)

    def test_missing_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            process_ticker(
                Ticker.ES,
                tmp_path / "nonexistent",
                tmp_path / "output",
                tmp_path / "metadata",
            )
