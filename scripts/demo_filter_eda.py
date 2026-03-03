"""
Filter Framework EDA Verification Script
=========================================

Demonstrates that filtered bias nodes produce outputs compatible with the
existing EDA and testing pipeline.  Tests four cases:

1. Continuous bias node (RSI) — unfiltered
2. Rule-based bias node (SimpleMomentum) — unfiltered
3. Continuous bias node + VolatilityFilter
4. Rule-based bias node + VolatilityFilter

For each case, prints: column names, output shape, value distribution,
NaN counts, and unique-value summary — so you can verify filtered nodes
behave identically to raw nodes (with expected neutral-value masking).

Usage
-----
    python scripts/demo_filter_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

from nodes import BiasNode
from nodes.filtered import FilteredBiasNode
from filters import FilterSpec
from filters.volatility import VolatilityFilter
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle
from utils.core.helpers import create_bias_node, create_filtered_bias_node


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TICKER = Ticker.ES
TF = TimeFrame.D
DATA_PATH = PROJECT_ROOT / "data" / "ohlc_data" / "ES" / "D_ES.parquet"


def load_candles(path: Path, limit: int = 1000) -> pd.DataFrame:
    """Load OHLCV data and return the most recent *limit* rows."""
    df = pd.read_parquet(path)
    df = df.sort_values("datetime").tail(limit).reset_index(drop=True)
    return df


def stream_node(node: BiasNode, df: pd.DataFrame) -> pd.DataFrame:
    """Stream candles through a node and collect outputs as a DataFrame."""
    records = []
    for _, row in df.iterrows():
        candle = Candle(
            datetime=pd.to_datetime(row["datetime"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0)),
            ticker=TICKER,
            tf=TF,
        )
        result = node.add_candle(candle)
        records.append({"datetime": candle.datetime, **{f"out_{i}": v for i, v in enumerate(result)}})
    return pd.DataFrame(records)


def summarise(label: str, node: BiasNode, out_df: pd.DataFrame) -> None:
    """Print a summary block for one node's output."""
    cols = node.get_column_names()
    signal = out_df["out_0"]

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Node class   : {node.__class__.__name__}")
    print(f"  Column names : {cols}")
    print(f"  Output shape : {out_df.shape}")
    print(f"  NaN count    : {signal.isna().sum()}")
    print(f"  Zero count   : {(signal == 0).sum()}")
    print(f"  Unique values: {signal.nunique()} (first 10: {sorted(signal.dropna().unique()[:10])})")
    print(f"  Descriptive stats:")
    print(f"    mean={signal.mean():.4f}  std={signal.std():.4f}  "
          f"min={signal.min():.4f}  max={signal.max():.4f}")

    if isinstance(node, FilteredBiasNode):
        total = len(signal)
        front_bad = node.front_bad
        post_warmup = signal.iloc[front_bad:]
        neutral_count = (post_warmup == node.neutral_value).sum()
        active_count = len(post_warmup) - neutral_count
        pct = neutral_count / len(post_warmup) * 100 if len(post_warmup) > 0 else 0
        print(f"  Filter stats (post-warmup):")
        print(f"    warmup={front_bad}  active={active_count}  "
              f"filtered_out={neutral_count} ({pct:.1f}%)")


def main() -> None:
    # Clear singleton cache to get fresh instances
    BiasNode._instances.clear()

    print(f"Loading data from {DATA_PATH} ...")
    df = load_candles(DATA_PATH, limit=1000)
    print(f"Loaded {len(df)} candles: {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

    # ---------------------------------------------------------------
    # Case 1: Continuous node (RSI) — unfiltered
    # ---------------------------------------------------------------
    rsi_node = create_bias_node("rsi", TICKER, TF, {"lookback": 14})
    rsi_out = stream_node(rsi_node, df)
    summarise("Case 1: RSI (continuous, unfiltered)", rsi_node, rsi_out)

    # ---------------------------------------------------------------
    # Case 2: Rule-based node (SimpleMomentum) — unfiltered
    # ---------------------------------------------------------------
    BiasNode._instances.clear()
    mom_node = create_bias_node("simple_momentum", TICKER, TF, {"lookback": 5})
    mom_out = stream_node(mom_node, df)
    summarise("Case 2: SimpleMomentum (rule-based, unfiltered)", mom_node, mom_out)

    # ---------------------------------------------------------------
    # Case 3: Continuous node + VolatilityFilter
    # ---------------------------------------------------------------
    BiasNode._instances.clear()
    vol_spec = FilterSpec(
        filter_name="vol",
        params={"atr_period": 14, "rank_period": 252, "regime": "high", "threshold": 0.5},
    )
    rsi_filtered = create_filtered_bias_node(
        "rsi", TICKER, TF, {"lookback": 14}, [vol_spec],
    )
    rsi_filt_out = stream_node(rsi_filtered, df)
    summarise("Case 3: RSI + VolatilityFilter(high) (continuous, filtered)", rsi_filtered, rsi_filt_out)

    # ---------------------------------------------------------------
    # Case 4: Rule-based node + VolatilityFilter
    # ---------------------------------------------------------------
    BiasNode._instances.clear()
    mom_filtered = create_filtered_bias_node(
        "simple_momentum", TICKER, TF, {"lookback": 5}, [vol_spec],
    )
    mom_filt_out = stream_node(mom_filtered, df)
    summarise("Case 4: SimpleMomentum + VolatilityFilter(high) (rule-based, filtered)", mom_filtered, mom_filt_out)

    # ---------------------------------------------------------------
    # Side-by-side comparison
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  Side-by-side comparison")
    print(f"{'='*70}")

    compare = pd.DataFrame({
        "datetime": df["datetime"].values[:20],
        "rsi_raw": rsi_out["out_0"].values[:20],
        "rsi_filtered": rsi_filt_out["out_0"].values[:20],
        "mom_raw": mom_out["out_0"].values[:20],
        "mom_filtered": mom_filt_out["out_0"].values[:20],
    })
    print("\nFirst 20 rows (showing where filter replaces with neutral 0.0):")
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print(compare.to_string(index=False))

    print("\nAll 4 cases ran successfully. Filtered nodes produce the same "
          "interface as raw nodes with neutral-value masking where filters block.")


if __name__ == "__main__":
    main()
