# Data Migration: Back-Adjustment Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a full back-adjustment pipeline (T001-T006) for Kibot 1-minute futures data, then validate against Norgate continuous futures.

**Architecture:** Six sequential modules under `data_cleaning/back_adjustment/`, each with frozen dataclasses and pure functions. Pipeline: roll rules → detect gaps → calculate adjustments → apply adjustments → orchestrate → validate vs Norgate. All modules follow functional-core pattern with TDD.

**Tech Stack:** Python 3.10, pandas, numpy, norgatedata, pytest, dataclasses, argparse

---

## Task 0: Project Setup — Extract Data & Create Directory Structure

**Files:**
- Create: `data_cleaning/__init__.py`
- Create: `data_cleaning/back_adjustment/__init__.py`
- Create: `tests/back_adjustment/__init__.py`
- Create: `tests/back_adjustment/conftest.py`
- Extract: Kibot M1 + D files from zip

**Step 1: Create package structure**

```bash
mkdir -p data_cleaning/back_adjustment
mkdir -p tests/back_adjustment
mkdir -p data/intraday_1min_original
mkdir -p data/intraday_1min_adjusted
mkdir -p data/adjustment_metadata
mkdir -p data/norgate/continuous_futures
mkdir -p docs/library/Data/comparisons
```

Create `data_cleaning/__init__.py`:
```python
```

Create `data_cleaning/back_adjustment/__init__.py`:
```python
```

Create `tests/back_adjustment/__init__.py`:
```python
```

**Step 2: Extract Kibot M1 and D files from zip**

```python
import zipfile
z = zipfile.ZipFile(r'data/kibot_data.zip')
for entry in z.infolist():
    name = entry.filename
    # Extract M1 files to data/intraday_1min_original/{TICKER}.parquet
    if '/M1_' in name and name.endswith('.parquet'):
        ticker = name.split('/')[-1].replace('M1_', '').replace('.parquet', '')
        with z.open(name) as src:
            with open(f'data/intraday_1min_original/{ticker}.parquet', 'wb') as dst:
                dst.write(src.read())
    # Extract D files to data/ohlc_data/{TICKER}/D_{TICKER}.parquet (update existing)
    if '/D_' in name and name.endswith('.parquet'):
        parts = name.split('/')
        ticker = parts[1]  # e.g., "ES"
        import os
        os.makedirs(f'data/ohlc_data/{ticker}', exist_ok=True)
        with z.open(name) as src:
            with open(f'data/ohlc_data/{ticker}/D_{ticker}.parquet', 'wb') as dst:
                dst.write(src.read())
```

**Step 3: Create shared test fixtures**

Create `tests/back_adjustment/conftest.py`:
```python
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_m1_dataframe() -> pd.DataFrame:
    """
    Synthetic 1-minute OHLCV DataFrame spanning 60 trading days.
    Includes one deliberate price gap (roll) at day 30.
    """
    rows = []
    base_price = 4000.0
    minutes_per_day = 390  # 6.5 hours
    gap_day = 30
    gap_size = 50.0  # points

    for day in range(60):
        date = datetime(2023, 1, 2) + timedelta(days=day)
        if date.weekday() >= 5:
            continue
        price = base_price + day * 2.0
        if day >= gap_day:
            price += gap_size  # simulate roll gap

        for minute in range(minutes_per_day):
            ts = date + timedelta(hours=9, minutes=30) + timedelta(minutes=minute)
            noise = np.random.RandomState(day * 1000 + minute).randn() * 0.5
            o = price + noise
            h = o + abs(noise) * 0.5
            l = o - abs(noise) * 0.5
            c = o + noise * 0.3
            rows.append({
                "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp": int(ts.timestamp()),
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def sample_daily_dataframe() -> pd.DataFrame:
    """Synthetic daily OHLC DataFrame with a known gap at row 30."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-02", periods=60)
    base = 4000.0
    gap_size = 50.0

    closes = []
    for i in range(60):
        price = base + i * 2.0
        if i >= 30:
            price += gap_size
        closes.append(price + np.random.randn() * 0.5)

    closes = np.array(closes)
    return pd.DataFrame({
        "datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": (dates.astype(np.int64) // 10**9).astype(np.uint32),
        "open": closes - np.random.rand(60) * 2,
        "high": closes + np.random.rand(60) * 2,
        "low": closes - np.random.rand(60) * 3,
        "close": closes,
    })
```

**Step 4: Fetch Norgate data for all tickers**

Create a one-time script `scripts/fetch_norgate_data.py`:
```python
"""Fetch Norgate continuous futures data for all mapped tickers."""
from __future__ import annotations

import norgatedata
import pandas as pd
from pathlib import Path

# Kibot ticker -> Norgate back-adjusted continuous symbol
TICKER_TO_NORGATE: dict[str, str] = {
    "ES": "&ES_CCB",
    "NQ": "&NQ_CCB",
    "YM": "&YM_CCB",
    "RTY": "&RTY_CCB",
    "CL": "&CL_CCB",
    "HO": "&HO_CCB",
    "GC": "&GC_CCB",
    "HG": "&HG_CCB",
    "SI": "&SI_CCB",
    "PL": "&PL_CCB",
    "EU": "&6E_CCB",
    "JY": "&6J_CCB",
    "BP": "&6B_CCB",
    "CD": "&6C_CCB",
    "SF": "&6S_CCB",
    "C": "&ZC_CCB",
    "S": "&ZS_CCB",
    "W": "&ZW_CCB",
    "GF": "&GF_CCB",
    "TY": "&ZN_CCB",
    "FV": "&ZF_CCB",
    "US": "&ZB_CCB",
    "TU": "&ZT_CCB",
}

# Also fetch unadjusted for roll-date detection via Delivery Month
TICKER_TO_NORGATE_RAW: dict[str, str] = {
    k: v.replace("_CCB", "") for k, v in TICKER_TO_NORGATE.items()
}


def fetch_and_save(
    ticker: str,
    norgate_symbol: str,
    output_dir: Path,
    start_date: str = "2005-01-01",
) -> pd.DataFrame | None:
    try:
        df = norgatedata.price_timeseries(
            norgate_symbol,
            start_date=start_date,
            interval="D",
            timeseriesformat="pandas-dataframe",
        )
        if df is None or df.empty:
            print(f"  WARN: No data for {ticker} ({norgate_symbol})")
            return None

        df.index.name = "Date"
        df = df.reset_index()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{ticker}.parquet"
        df.to_parquet(path, index=False)
        print(f"  OK: {ticker} -> {path} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"  ERR: {ticker} ({norgate_symbol}): {e}")
        return None


def main() -> None:
    if not norgatedata.status():
        raise RuntimeError("Norgate Data Updater is not running")

    adj_dir = Path("data/norgate/continuous_futures/adjusted")
    raw_dir = Path("data/norgate/continuous_futures/unadjusted")

    print("=== Fetching back-adjusted continuous futures ===")
    for ticker, sym in TICKER_TO_NORGATE.items():
        fetch_and_save(ticker, sym, adj_dir)

    print("\n=== Fetching unadjusted continuous futures ===")
    for ticker, sym in TICKER_TO_NORGATE_RAW.items():
        fetch_and_save(ticker, sym, raw_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
```

Run: `python scripts/fetch_norgate_data.py`

**Step 5: Verify setup**

```bash
ls data/intraday_1min_original/   # Should show {TICKER}.parquet files
ls data/norgate/continuous_futures/adjusted/  # Should show {TICKER}.parquet files
ls data/norgate/continuous_futures/unadjusted/  # Should show {TICKER}.parquet files
python -c "import data_cleaning.back_adjustment"  # Should succeed (empty package)
```

**Step 6: Commit**

```bash
git add data_cleaning/__init__.py data_cleaning/back_adjustment/__init__.py \
    tests/back_adjustment/__init__.py tests/back_adjustment/conftest.py \
    scripts/fetch_norgate_data.py
git commit -m "chore: scaffold back-adjustment package and test fixtures"
```

---

## Task 1: Roll Rules Dataclass and Lookup (T001)

**Files:**
- Create: `data_cleaning/back_adjustment/roll_rules.py`
- Create: `tests/back_adjustment/test_roll_rules.py`

**Step 1: Write the failing tests**

Create `tests/back_adjustment/test_roll_rules.py`:
```python
from __future__ import annotations

import pytest
from utils.enums import Ticker
from data_cleaning.back_adjustment.roll_rules import (
    RollRule,
    get_roll_rule,
    get_all_roll_rules,
)


class TestRollRules:

    def test_all_tickers_have_rules(self) -> None:
        """Every Ticker enum member must have a corresponding RollRule."""
        rules = get_all_roll_rules()
        for ticker in Ticker:
            assert ticker in rules, f"Missing roll rule for {ticker.name}"
            assert isinstance(rules[ticker], RollRule)

    def test_get_roll_rule_lookup(self) -> None:
        """get_roll_rule returns the correct RollRule for known tickers."""
        rule = get_roll_rule(Ticker.ES)
        assert rule.ticker == Ticker.ES
        assert rule.rollover_offset == -5
        assert rule.reference_point == "expiration"
        assert isinstance(rule.description, str)
        assert len(rule.description) > 0

    def test_roll_rule_immutability(self) -> None:
        """RollRule must be frozen (immutable)."""
        rule = get_roll_rule(Ticker.ES)
        with pytest.raises(AttributeError):
            rule.rollover_offset = 99  # type: ignore[misc]

    def test_get_roll_rule_invalid_ticker(self) -> None:
        """get_roll_rule raises ValueError for unknown ticker."""
        with pytest.raises(ValueError, match="No roll rule defined"):
            get_roll_rule("NOT_A_TICKER")  # type: ignore[arg-type]

    def test_reference_point_values(self) -> None:
        """All reference_point values must be 'expiration' or 'month_end'."""
        for rule in get_all_roll_rules().values():
            assert rule.reference_point in ("expiration", "month_end"), (
                f"{rule.ticker.name} has invalid reference_point: {rule.reference_point}"
            )

    def test_rollover_offset_sign_convention(self) -> None:
        """Expiration-based rules use negative offset, month_end use positive."""
        for rule in get_all_roll_rules().values():
            if rule.reference_point == "expiration":
                assert rule.rollover_offset < 0, (
                    f"{rule.ticker.name}: expiration rule should have negative offset"
                )
            else:
                assert rule.rollover_offset > 0, (
                    f"{rule.ticker.name}: month_end rule should have positive offset"
                )
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/back_adjustment/test_roll_rules.py -v`
Expected: ImportError — module not found

**Step 3: Write the implementation**

Create `data_cleaning/back_adjustment/roll_rules.py`:
```python
"""Fixed-date roll rules for futures continuous contracts.

Each ticker has a RollRule defining when the continuous contract rolls
from the expiring front-month to the next contract. These rules
replicate the legacy (Kibot) fixed-date schedule.

References:
    - Kibot rollover rules: https://www.kibot.com/rollover_rules.aspx
    - CME/ICE contract specifications for expiration dates
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

from utils.enums import Ticker


@dataclass(frozen=True)
class RollRule:
    """Immutable roll rule for a single futures ticker.

    Parameters
    ----------
    ticker : Ticker
        The instrument this rule applies to.
    rollover_offset : int
        Days relative to reference_point.
        Negative = days *before* expiration.
        Positive = days *from* month-end (used for month_end reference).
    reference_point : Literal["expiration", "month_end"]
        What the offset is measured from.
    description : str
        Human-readable explanation of the rule.
    """

    ticker: Ticker
    rollover_offset: int
    reference_point: Literal["expiration", "month_end"]
    description: str


ROLL_RULES: Dict[Ticker, RollRule] = {
    # --- Equity Indices: roll ~5 trading days before quarterly expiration ---
    Ticker.ES: RollRule(Ticker.ES, -5, "expiration", "5 days before quarterly expiration"),
    Ticker.NQ: RollRule(Ticker.NQ, -5, "expiration", "5 days before quarterly expiration"),
    Ticker.YM: RollRule(Ticker.YM, -5, "expiration", "5 days before quarterly expiration"),
    Ticker.RTY: RollRule(Ticker.RTY, -5, "expiration", "5 days before quarterly expiration"),

    # --- Energy: roll ~3 trading days before expiration ---
    Ticker.CL: RollRule(Ticker.CL, -3, "expiration", "3 days before monthly expiration"),
    Ticker.HO: RollRule(Ticker.HO, -3, "expiration", "3 days before monthly expiration"),

    # --- Metals: roll ~2 days from month end ---
    Ticker.GC: RollRule(Ticker.GC, 2, "month_end", "2 days from month end"),
    Ticker.HG: RollRule(Ticker.HG, 2, "month_end", "2 days from month end"),
    Ticker.SI: RollRule(Ticker.SI, 2, "month_end", "2 days from month end"),
    Ticker.PL: RollRule(Ticker.PL, 2, "month_end", "2 days from month end"),

    # --- FX Currencies: roll ~2 trading days before quarterly expiration ---
    Ticker.EU: RollRule(Ticker.EU, -2, "expiration", "2 days before quarterly expiration"),
    Ticker.JY: RollRule(Ticker.JY, -2, "expiration", "2 days before quarterly expiration"),
    Ticker.BP: RollRule(Ticker.BP, -2, "expiration", "2 days before quarterly expiration"),
    Ticker.CD: RollRule(Ticker.CD, -2, "expiration", "2 days before quarterly expiration"),
    Ticker.SF: RollRule(Ticker.SF, -2, "expiration", "2 days before quarterly expiration"),

    # --- Agricultural: roll ~8 days from prior month-end ---
    Ticker.C: RollRule(Ticker.C, 8, "month_end", "8 days from prior month end"),
    Ticker.S: RollRule(Ticker.S, 8, "month_end", "8 days from prior month end"),
    Ticker.W: RollRule(Ticker.W, 8, "month_end", "8 days from prior month end"),
    Ticker.GF: RollRule(Ticker.GF, 8, "month_end", "8 days from prior month end"),

    # --- Fixed Income: roll ~3 days from month end ---
    Ticker.TY: RollRule(Ticker.TY, 3, "month_end", "3 days from month end"),
    Ticker.FV: RollRule(Ticker.FV, 3, "month_end", "3 days from month end"),
    Ticker.US: RollRule(Ticker.US, 3, "month_end", "3 days from month end"),
    Ticker.TU: RollRule(Ticker.TU, 3, "month_end", "3 days from month end"),
    Ticker.TLT: RollRule(Ticker.TLT, 3, "month_end", "3 days from month end"),
}


def get_roll_rule(ticker: Ticker) -> RollRule:
    """Look up the roll rule for a ticker.

    Raises
    ------
    ValueError
        If no rule is defined for the given ticker.
    """
    if ticker not in ROLL_RULES:
        raise ValueError(f"No roll rule defined for {ticker}")
    return ROLL_RULES[ticker]


def get_all_roll_rules() -> Dict[Ticker, RollRule]:
    """Return a copy of the complete roll rules dictionary."""
    return dict(ROLL_RULES)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/back_adjustment/test_roll_rules.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add data_cleaning/back_adjustment/roll_rules.py tests/back_adjustment/test_roll_rules.py
git commit -m "feat(T001): add roll rules dataclass and lookup"
```

---

## Task 2: Roll Event Detection (T002)

**Files:**
- Create: `data_cleaning/back_adjustment/roll_detector.py`
- Create: `tests/back_adjustment/test_roll_detector.py`

**Step 1: Write the failing tests**

Create `tests/back_adjustment/test_roll_detector.py`:
```python
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_cleaning.back_adjustment.roll_detector import RollEvent, detect_roll_dates
from data_cleaning.back_adjustment.roll_rules import RollRule
from utils.enums import Ticker


def _make_daily_series(
    n_days: int = 120,
    base_price: float = 4000.0,
    gap_day: int = 60,
    gap_size: float = 50.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic daily DataFrame with one gap at gap_day."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    closes: list[float] = []
    for i in range(n_days):
        price = base_price + i * 1.0
        if i >= gap_day:
            price += gap_size
        closes.append(price + rng.randn() * 0.3)
    closes_arr = np.array(closes)
    return pd.DataFrame({
        "datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": (dates.astype(np.int64) // 10**9).astype(np.uint32),
        "open": closes_arr - rng.rand(n_days) * 0.5,
        "high": closes_arr + rng.rand(n_days) * 1.0,
        "low": closes_arr - rng.rand(n_days) * 1.5,
        "close": closes_arr,
    })


# Use a permissive rule: 30 days before expiration, so the synthetic gap
# on business-day ~60 falls within the ±3 day tolerance window.
_PERMISSIVE_RULE = RollRule(
    ticker=Ticker.ES,
    rollover_offset=-30,
    reference_point="expiration",
    description="test rule: wide window",
)


class TestRollDetector:

    def test_detect_known_roll(self) -> None:
        """A single 50-point gap should produce exactly one RollEvent."""
        df = _make_daily_series(gap_day=60, gap_size=50.0)
        events = detect_roll_dates(df, _PERMISSIVE_RULE)
        assert len(events) >= 1
        # The gap should be close to 50 points
        assert any(abs(e.gap_points - 50.0) < 5.0 for e in events)

    def test_no_rolls_found(self) -> None:
        """Data with no significant gaps should return empty list."""
        df = _make_daily_series(gap_day=999, gap_size=0.0)  # no gap
        events = detect_roll_dates(df, _PERMISSIVE_RULE)
        assert events == []

    def test_roll_event_ordering(self) -> None:
        """Multiple roll events must be sorted chronologically."""
        # Create data with two gaps
        rng = np.random.RandomState(42)
        n_days = 200
        dates = pd.bdate_range("2023-01-02", periods=n_days)
        closes: list[float] = []
        for i in range(n_days):
            price = 4000.0 + i * 1.0
            if i >= 60:
                price += 50.0
            if i >= 130:
                price += 40.0
            closes.append(price + rng.randn() * 0.3)
        closes_arr = np.array(closes)
        df = pd.DataFrame({
            "datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": (dates.astype(np.int64) // 10**9).astype(np.uint32),
            "open": closes_arr,
            "high": closes_arr + 1.0,
            "low": closes_arr - 1.0,
            "close": closes_arr,
        })
        events = detect_roll_dates(df, _PERMISSIVE_RULE)
        if len(events) >= 2:
            for i in range(len(events) - 1):
                assert events[i].roll_date <= events[i + 1].roll_date

    def test_determinism(self) -> None:
        """Same input must produce identical output."""
        df = _make_daily_series()
        result_a = detect_roll_dates(df, _PERMISSIVE_RULE)
        result_b = detect_roll_dates(df, _PERMISSIVE_RULE)
        assert len(result_a) == len(result_b)
        for a, b in zip(result_a, result_b):
            assert a.roll_date == b.roll_date
            assert a.gap_points == b.gap_points

    def test_roll_event_fields(self) -> None:
        """RollEvent must have all required fields with correct types."""
        df = _make_daily_series(gap_day=60, gap_size=50.0)
        events = detect_roll_dates(df, _PERMISSIVE_RULE)
        assert len(events) >= 1
        e = events[0]
        assert isinstance(e.roll_date, datetime)
        assert isinstance(e.old_contract_close, float)
        assert isinstance(e.new_contract_close, float)
        assert isinstance(e.gap_points, float)
        assert e.gap_points == e.new_contract_close - e.old_contract_close

    def test_roll_event_immutability(self) -> None:
        """RollEvent must be frozen."""
        df = _make_daily_series(gap_day=60, gap_size=50.0)
        events = detect_roll_dates(df, _PERMISSIVE_RULE)
        assert len(events) >= 1
        with pytest.raises(AttributeError):
            events[0].gap_points = 0.0  # type: ignore[misc]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/back_adjustment/test_roll_detector.py -v`
Expected: ImportError

**Step 3: Write the implementation**

Create `data_cleaning/back_adjustment/roll_detector.py`:
```python
"""Detect roll events in historical futures data by finding price gaps.

The detector resamples intraday (or daily) data to daily close prices,
computes an adaptive threshold, and flags days where the close-to-close
gap exceeds that threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from data_cleaning.back_adjustment.roll_rules import RollRule


@dataclass(frozen=True)
class RollEvent:
    """A single detected roll event.

    Attributes
    ----------
    roll_date : datetime
        The date the gap was observed (first bar of the new contract).
    old_contract_close : float
        Closing price on the last day of the old contract.
    new_contract_close : float
        Closing price on the first day of the new contract.
    gap_points : float
        new_contract_close - old_contract_close (signed).
    """

    roll_date: datetime
    old_contract_close: float
    new_contract_close: float
    gap_points: float


def _resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a DataFrame (any frequency) to daily OHLC bars.

    Uses the last bar's close as the daily close, max high, min low, first open.
    """
    tmp = df.copy()
    tmp["_dt"] = pd.to_datetime(tmp["datetime"])
    tmp["_date"] = tmp["_dt"].dt.normalize()

    daily = tmp.groupby("_date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).reset_index().rename(columns={"_date": "date"})

    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def detect_roll_dates(
    df: pd.DataFrame,
    rule: RollRule,
    trailing_window: int = 252,
    sigma_multiplier: float = 3.0,
    pct_floor: float = 0.01,
) -> List[RollEvent]:
    """Detect roll dates by finding abnormal close-to-close gaps.

    Algorithm
    ---------
    1. Resample to daily close prices.
    2. Compute daily close-to-close changes.
    3. Threshold = max(pct_floor * prev_close, sigma_multiplier * trailing_std).
    4. Flag days where abs(change) >= threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with columns: datetime, open, high, low, close.
        Can be any frequency (1-min, daily, etc.).
    rule : RollRule
        The roll rule for this ticker (used for documentation; gap detection
        is purely price-based).
    trailing_window : int
        Number of trading days for trailing std calculation.
    sigma_multiplier : float
        Multiplier for the std-based threshold component.
    pct_floor : float
        Minimum threshold as a fraction of the previous close.

    Returns
    -------
    List[RollEvent]
        Detected roll events, sorted chronologically (oldest first).
    """
    if df.empty:
        return []

    daily = _resample_to_daily(df)
    if len(daily) < 2:
        return []

    closes = daily["close"].values.astype(np.float64)
    dates = pd.to_datetime(daily["date"]).values

    changes = np.diff(closes)
    abs_changes = np.abs(changes)

    events: list[RollEvent] = []

    for i in range(len(changes)):
        # Trailing std window
        start_idx = max(0, i - trailing_window)
        window = abs_changes[start_idx:i] if i > 0 else abs_changes[:1]
        trailing_std = float(np.std(window)) if len(window) > 1 else 0.0

        prev_close = closes[i]
        threshold = max(pct_floor * abs(prev_close), sigma_multiplier * trailing_std)

        if threshold > 0 and abs(changes[i]) >= threshold:
            roll_date = pd.Timestamp(dates[i + 1]).to_pydatetime()
            old_close = float(closes[i])
            new_close = float(closes[i + 1])
            gap = new_close - old_close

            events.append(RollEvent(
                roll_date=roll_date,
                old_contract_close=old_close,
                new_contract_close=new_close,
                gap_points=gap,
            ))

    events.sort(key=lambda e: e.roll_date)
    return events
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/back_adjustment/test_roll_detector.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add data_cleaning/back_adjustment/roll_detector.py tests/back_adjustment/test_roll_detector.py
git commit -m "feat(T002): add roll event detection from price gaps"
```

---

## Task 3: Gap Calculator and Adjustment Factors (T003)

**Files:**
- Create: `data_cleaning/back_adjustment/gap_calculator.py`
- Create: `tests/back_adjustment/test_gap_calculator.py`

**Step 1: Write the failing tests**

Create `tests/back_adjustment/test_gap_calculator.py`:
```python
from __future__ import annotations

import pytest
from datetime import datetime

from data_cleaning.back_adjustment.gap_calculator import (
    AdjustmentFactor,
    calculate_adjustments,
)
from data_cleaning.back_adjustment.roll_detector import RollEvent


def _make_events(*gaps: float) -> list[RollEvent]:
    """Create RollEvent list with given gap sizes at monthly intervals."""
    events: list[RollEvent] = []
    for i, gap in enumerate(gaps):
        month = i + 1
        events.append(RollEvent(
            roll_date=datetime(2023, month, 15),
            old_contract_close=4000.0 + i * 100,
            new_contract_close=4000.0 + i * 100 + gap,
            gap_points=gap,
        ))
    return events


class TestGapCalculator:

    def test_cumulative_calculation(self) -> None:
        """3 rolls with gaps [+10, -5, +3] -> cumulative [-2, 3, 0]."""
        events = _make_events(10.0, -5.0, 3.0)
        adjustments = calculate_adjustments(events)

        assert len(adjustments) == 3
        assert adjustments[0].cumulative_adjustment == pytest.approx(-2.0)
        assert adjustments[1].cumulative_adjustment == pytest.approx(3.0)
        assert adjustments[2].cumulative_adjustment == pytest.approx(0.0)

    def test_most_recent_roll_zero(self) -> None:
        """The most recent roll always has cumulative_adjustment = 0."""
        for n in range(1, 5):
            events = _make_events(*[float(i + 1) for i in range(n)])
            adjustments = calculate_adjustments(events)
            assert adjustments[-1].cumulative_adjustment == pytest.approx(0.0)

    def test_future_gap_sum(self) -> None:
        """Each adjustment equals the sum of all later gaps."""
        events = _make_events(10.0, 20.0, 30.0, 5.0)
        adjustments = calculate_adjustments(events)

        gaps = [10.0, 20.0, 30.0, 5.0]
        for i, adj in enumerate(adjustments):
            expected = sum(gaps[i + 1:])
            assert adj.cumulative_adjustment == pytest.approx(expected), (
                f"Index {i}: expected {expected}, got {adj.cumulative_adjustment}"
            )

    def test_determinism(self) -> None:
        """Same input -> same output."""
        events = _make_events(10.0, -5.0, 3.0)
        a = calculate_adjustments(events)
        b = calculate_adjustments(events)
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert x.cumulative_adjustment == y.cumulative_adjustment
            assert x.roll_date == y.roll_date

    def test_empty_input(self) -> None:
        """Empty roll events returns empty list."""
        assert calculate_adjustments([]) == []

    def test_single_roll(self) -> None:
        """Single roll has cumulative_adjustment = 0."""
        events = _make_events(25.0)
        adjustments = calculate_adjustments(events)
        assert len(adjustments) == 1
        assert adjustments[0].cumulative_adjustment == pytest.approx(0.0)
        assert adjustments[0].gap_points == pytest.approx(25.0)

    def test_adjustment_factor_immutability(self) -> None:
        """AdjustmentFactor must be frozen."""
        events = _make_events(10.0)
        adjustments = calculate_adjustments(events)
        with pytest.raises(AttributeError):
            adjustments[0].cumulative_adjustment = 99.0  # type: ignore[misc]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/back_adjustment/test_gap_calculator.py -v`
Expected: ImportError

**Step 3: Write the implementation**

Create `data_cleaning/back_adjustment/gap_calculator.py`:
```python
"""Calculate cumulative back-adjustment factors from detected roll events.

Arithmetic back-adjustment: for each roll, sum all *later* gaps to determine
how much to shift prices in the period before that roll.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

from data_cleaning.back_adjustment.roll_detector import RollEvent


@dataclass(frozen=True)
class AdjustmentFactor:
    """Cumulative adjustment for a single roll boundary.

    Attributes
    ----------
    roll_date : datetime
        The date this roll occurred.
    gap_points : float
        The gap at this specific roll (new_close - old_close).
    cumulative_adjustment : float
        Sum of gap_points for all rolls *after* this one.
        Added to OHLC prices for all bars before this roll_date.
    """

    roll_date: datetime
    gap_points: float
    cumulative_adjustment: float


def calculate_adjustments(
    roll_events: List[RollEvent],
) -> List[AdjustmentFactor]:
    """Compute cumulative adjustment factors from chronological roll events.

    Parameters
    ----------
    roll_events : List[RollEvent]
        Roll events sorted chronologically (oldest first).

    Returns
    -------
    List[AdjustmentFactor]
        Same length as input, sorted chronologically.
        The most recent event has cumulative_adjustment = 0.
        Each earlier event accumulates the sum of all later gaps.
    """
    if not roll_events:
        return []

    n = len(roll_events)
    gaps = [e.gap_points for e in roll_events]

    # Build suffix sums: cumulative[i] = sum(gaps[i+1:])
    suffix_sum = 0.0
    cumulative = [0.0] * n
    for i in range(n - 2, -1, -1):
        suffix_sum += gaps[i + 1]
        cumulative[i] = suffix_sum

    return [
        AdjustmentFactor(
            roll_date=roll_events[i].roll_date,
            gap_points=gaps[i],
            cumulative_adjustment=cumulative[i],
        )
        for i in range(n)
    ]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/back_adjustment/test_gap_calculator.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add data_cleaning/back_adjustment/gap_calculator.py tests/back_adjustment/test_gap_calculator.py
git commit -m "feat(T003): add cumulative gap calculator"
```

---

## Task 4: Back Adjuster Implementation (T004)

**Files:**
- Create: `data_cleaning/back_adjustment/back_adjuster.py`
- Create: `tests/back_adjustment/test_back_adjuster.py`

**Step 1: Write the failing tests**

Create `tests/back_adjustment/test_back_adjuster.py`:
```python
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from data_cleaning.back_adjustment.back_adjuster import (
    apply_back_adjustment,
    validate_adjusted_data,
)
from data_cleaning.back_adjustment.gap_calculator import AdjustmentFactor


def _make_ohlc(n: int = 100, base: float = 4000.0) -> pd.DataFrame:
    """Simple daily OHLC without gaps."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2023-01-02", periods=n)
    closes = base + np.arange(n, dtype=float) + rng.randn(n) * 0.5
    return pd.DataFrame({
        "datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": (dates.astype(np.int64) // 10**9).astype(np.uint32),
        "open": closes - rng.rand(n),
        "high": closes + rng.rand(n) * 2,
        "low": closes - rng.rand(n) * 2,
        "close": closes,
    })


def _make_adjustments() -> list[AdjustmentFactor]:
    """Two rolls: day ~30 (gap +20) and day ~60 (gap +10)."""
    return [
        AdjustmentFactor(
            roll_date=datetime(2023, 2, 14),  # ~day 30
            gap_points=20.0,
            cumulative_adjustment=10.0,  # sum of later gaps
        ),
        AdjustmentFactor(
            roll_date=datetime(2023, 3, 22),  # ~day 60
            gap_points=10.0,
            cumulative_adjustment=0.0,
        ),
    ]


class TestBackAdjuster:

    def test_apply_adjustment_correctness(self) -> None:
        """OHLC prices before first roll get +10, between rolls +10, after second roll +0."""
        df = _make_ohlc()
        adjustments = _make_adjustments()
        result = apply_back_adjustment(df, adjustments)

        dt = pd.to_datetime(result["datetime"])

        # Pre-first-roll segment: adjustment = sum of ALL gaps = 20 + 10 = 30
        pre_mask = dt < datetime(2023, 2, 14)
        if pre_mask.any():
            diff = result.loc[pre_mask, "close"].values - df.loc[pre_mask, "close"].values
            np.testing.assert_allclose(diff, 30.0, atol=1e-6)

        # Between rolls: cumulative_adjustment of first roll = 10
        mid_mask = (dt >= datetime(2023, 2, 14)) & (dt < datetime(2023, 3, 22))
        if mid_mask.any():
            diff = result.loc[mid_mask, "close"].values - df.loc[mid_mask, "close"].values
            np.testing.assert_allclose(diff, 10.0, atol=1e-6)

        # After most recent roll: adjustment = 0
        post_mask = dt >= datetime(2023, 3, 22)
        if post_mask.any():
            diff = result.loc[post_mask, "close"].values - df.loc[post_mask, "close"].values
            np.testing.assert_allclose(diff, 0.0, atol=1e-6)

    def test_volume_unchanged(self) -> None:
        """If a volume column exists, it must not be modified."""
        df = _make_ohlc()
        df["volume"] = np.arange(len(df), dtype=float) * 100
        adjustments = _make_adjustments()
        result = apply_back_adjustment(df, adjustments)
        pd.testing.assert_series_equal(result["volume"], df["volume"])

    def test_no_negative_prices(self) -> None:
        """validate_adjusted_data must raise if adjustment creates negatives."""
        df = _make_ohlc(base=5.0)  # low base price
        huge_negative = [
            AdjustmentFactor(
                roll_date=datetime(2023, 2, 14),
                gap_points=-1000.0,
                cumulative_adjustment=-1000.0,
            ),
        ]
        result = apply_back_adjustment(df, huge_negative)
        with pytest.raises(ValueError, match="negative"):
            validate_adjusted_data(df, result, huge_negative)

    def test_immutability(self) -> None:
        """Input DataFrame must not be modified."""
        df = _make_ohlc()
        original = df.copy()
        adjustments = _make_adjustments()
        apply_back_adjustment(df, adjustments)
        pd.testing.assert_frame_equal(df, original)

    def test_determinism(self) -> None:
        """Same input -> same output."""
        df = _make_ohlc()
        adjustments = _make_adjustments()
        a = apply_back_adjustment(df, adjustments)
        b = apply_back_adjustment(df, adjustments)
        pd.testing.assert_frame_equal(a, b)

    def test_empty_adjustments(self) -> None:
        """No adjustments -> output equals input."""
        df = _make_ohlc()
        result = apply_back_adjustment(df, [])
        pd.testing.assert_frame_equal(result, df)

    def test_row_count_preserved(self) -> None:
        """Output must have same number of rows as input."""
        df = _make_ohlc()
        result = apply_back_adjustment(df, _make_adjustments())
        assert len(result) == len(df)

    def test_all_ohlc_columns_adjusted(self) -> None:
        """Open, high, low, close all get the same adjustment per row."""
        df = _make_ohlc()
        adjustments = _make_adjustments()
        result = apply_back_adjustment(df, adjustments)
        for col in ("open", "high", "low", "close"):
            diff = result[col].values - df[col].values
            # All OHLC diffs should be the same per row
            np.testing.assert_allclose(
                diff,
                result["close"].values - df["close"].values,
                atol=1e-6,
            )
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/back_adjustment/test_back_adjuster.py -v`
Expected: ImportError

**Step 3: Write the implementation**

Create `data_cleaning/back_adjustment/back_adjuster.py`:
```python
"""Apply arithmetic back-adjustment to OHLC price data.

Given a set of cumulative adjustment factors (from gap_calculator),
shift open/high/low/close prices so that roll gaps are eliminated.
Volume and timestamp columns are never modified.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from data_cleaning.back_adjustment.gap_calculator import AdjustmentFactor

OHLC_COLS = ("open", "high", "low", "close")


def apply_back_adjustment(
    df: pd.DataFrame,
    adjustments: List[AdjustmentFactor],
) -> pd.DataFrame:
    """Apply arithmetic back-adjustment to OHLC columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with at least: datetime, open, high, low, close.
    adjustments : List[AdjustmentFactor]
        Sorted chronologically (oldest first). Each has a roll_date
        and cumulative_adjustment.

    Returns
    -------
    pd.DataFrame
        New DataFrame with adjusted OHLC prices. Input is not modified.

    Notes
    -----
    - Rows *before* the earliest roll_date: adjusted by sum of ALL gaps
      (cumulative_adjustment of first roll + first roll's own gap_points).
    - Rows *on or after* roll_date[i] and *before* roll_date[i+1]:
      adjusted by adjustments[i].cumulative_adjustment.
    - Rows *on or after* the most recent roll_date: adjustment = 0.
    """
    result = df.copy()

    if not adjustments:
        return result

    dt = pd.to_datetime(result["datetime"])

    # Build adjustment vector (one value per row)
    adj_values = np.zeros(len(result), dtype=np.float64)

    # Pre-first-roll: sum of ALL gaps
    total_gaps = sum(a.gap_points for a in adjustments)
    first_roll = adjustments[0].roll_date
    adj_values[dt < first_roll] = total_gaps

    # Between consecutive rolls
    for i in range(len(adjustments)):
        roll_dt = adjustments[i].roll_date
        next_dt = adjustments[i + 1].roll_date if i + 1 < len(adjustments) else None

        if next_dt is not None:
            mask = (dt >= roll_dt) & (dt < next_dt)
        else:
            mask = dt >= roll_dt

        adj_values[mask] = adjustments[i].cumulative_adjustment

    # Apply to OHLC columns
    for col in OHLC_COLS:
        if col in result.columns:
            result[col] = result[col].astype(np.float64) + adj_values

    return result


def validate_adjusted_data(
    original: pd.DataFrame,
    adjusted: pd.DataFrame,
    adjustments: List[AdjustmentFactor],
) -> bool:
    """Validate back-adjusted data integrity.

    Raises
    ------
    ValueError
        If any validation check fails.

    Returns
    -------
    bool
        True if all checks pass.
    """
    if len(original) != len(adjusted):
        raise ValueError(
            f"Row count mismatch: original={len(original)}, adjusted={len(adjusted)}"
        )

    # Check no negative prices in OHLC
    for col in OHLC_COLS:
        if col in adjusted.columns:
            min_val = adjusted[col].min()
            if min_val < 0:
                raise ValueError(
                    f"Adjusted '{col}' contains negative prices (min={min_val:.4f})"
                )

    # Check volume unchanged (if present)
    if "volume" in original.columns and "volume" in adjusted.columns:
        if not original["volume"].equals(adjusted["volume"]):
            raise ValueError("Volume column was modified during adjustment")

    # Check datetime/timestamp unchanged
    for col in ("datetime", "timestamp"):
        if col in original.columns and col in adjusted.columns:
            if not original[col].equals(adjusted[col]):
                raise ValueError(f"'{col}' column was modified during adjustment")

    return True
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/back_adjustment/test_back_adjuster.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add data_cleaning/back_adjustment/back_adjuster.py tests/back_adjustment/test_back_adjuster.py
git commit -m "feat(T004): add back-adjustment application and validation"
```

---

## Task 5: Orchestrator and CLI (T005)

**Files:**
- Create: `data_cleaning/back_adjustment/orchestrator.py`
- Create: `tests/back_adjustment/test_orchestrator.py`

**Step 1: Write the failing tests**

Create `tests/back_adjustment/test_orchestrator.py`:
```python
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
    """Write a synthetic M1-style parquet with a roll gap."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    closes: list[float] = []
    for i in range(n_days):
        price = 4000.0 + i * 1.0
        if i >= 100:
            price += 50.0  # roll gap at day 100
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
        """End-to-end: process_ticker reads input, writes output + metadata."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        metadata_dir = tmp_path / "metadata"

        _write_synthetic_parquet(input_dir / "ES.parquet")

        meta = process_ticker(
            Ticker.ES, input_dir, output_dir, metadata_dir,
        )
        assert isinstance(meta, AdjustmentMetadata)
        assert meta.ticker == Ticker.ES
        assert (output_dir / "ES.parquet").exists()
        assert meta.num_rolls_detected >= 0

    def test_metadata_saved(self, tmp_path: Path) -> None:
        """Metadata JSON is written and contains expected keys."""
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
        """CLI parser accepts --ticker ES."""
        parser = build_arg_parser()
        args = parser.parse_args(["--ticker", "ES"])
        assert args.ticker == "ES"
        assert args.all is False

    def test_cli_all_tickers(self) -> None:
        """CLI parser accepts --all."""
        parser = build_arg_parser()
        args = parser.parse_args(["--all"])
        assert args.all is True

    def test_determinism(self, tmp_path: Path) -> None:
        """Same input -> same output."""
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
        """process_ticker raises FileNotFoundError for missing input."""
        with pytest.raises(FileNotFoundError):
            process_ticker(
                Ticker.ES,
                tmp_path / "nonexistent",
                tmp_path / "output",
                tmp_path / "metadata",
            )
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/back_adjustment/test_orchestrator.py -v`
Expected: ImportError

**Step 3: Write the implementation**

Create `data_cleaning/back_adjustment/orchestrator.py`:
```python
"""Orchestrate the full back-adjustment pipeline for one or all tickers.

Usage:
    python -m data_cleaning.back_adjustment.orchestrator --ticker ES
    python -m data_cleaning.back_adjustment.orchestrator --all
    python -m data_cleaning.back_adjustment.orchestrator --all --parallel
"""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from utils.enums import Ticker
from data_cleaning.back_adjustment.roll_rules import get_roll_rule
from data_cleaning.back_adjustment.roll_detector import RollEvent, detect_roll_dates
from data_cleaning.back_adjustment.gap_calculator import (
    AdjustmentFactor,
    calculate_adjustments,
)
from data_cleaning.back_adjustment.back_adjuster import (
    apply_back_adjustment,
    validate_adjusted_data,
)

logger = logging.getLogger(__name__)

_DEFAULT_INPUT = Path("data/intraday_1min_original")
_DEFAULT_OUTPUT = Path("data/intraday_1min_adjusted")
_DEFAULT_METADATA = Path("data/adjustment_metadata")


@dataclass(frozen=True)
class AdjustmentMetadata:
    """Audit trail for one ticker's back-adjustment run."""

    ticker: Ticker
    processing_date: str
    num_rolls_detected: int
    total_adjustment_range: float
    source_file: str
    output_file: str
    roll_dates: List[str]
    gap_points: List[float]
    cumulative_adjustments: List[float]


def _serialize_metadata(meta: AdjustmentMetadata) -> dict:
    """Convert metadata to JSON-serializable dict."""
    d = asdict(meta)
    d["ticker"] = meta.ticker.name
    return d


def process_ticker(
    ticker: Ticker,
    input_dir: Path = _DEFAULT_INPUT,
    output_dir: Path = _DEFAULT_OUTPUT,
    metadata_dir: Path = _DEFAULT_METADATA,
) -> AdjustmentMetadata:
    """Run the full back-adjustment pipeline for a single ticker.

    Parameters
    ----------
    ticker : Ticker
        The futures ticker to process.
    input_dir : Path
        Directory containing {TICKER}.parquet input files.
    output_dir : Path
        Directory to write adjusted parquet files.
    metadata_dir : Path
        Directory to write JSON metadata files.

    Returns
    -------
    AdjustmentMetadata
        Summary of the adjustment run.

    Raises
    ------
    FileNotFoundError
        If the input parquet file does not exist.
    """
    source_path = input_dir / f"{ticker.name}.parquet"
    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {source_path}")

    logger.info("Processing %s from %s", ticker.name, source_path)

    # 1. Load data
    df = pd.read_parquet(source_path)

    # 2. Get roll rule
    rule = get_roll_rule(ticker)

    # 3. Detect rolls
    roll_events = detect_roll_dates(df, rule)
    logger.info("%s: %d rolls detected", ticker.name, len(roll_events))

    # 4. Calculate adjustments
    adjustments = calculate_adjustments(roll_events)

    # 5. Apply adjustments
    adjusted = apply_back_adjustment(df, adjustments)

    # 6. Validate
    if adjustments:
        validate_adjusted_data(df, adjusted, adjustments)

    # 7. Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker.name}.parquet"
    adjusted.to_parquet(output_path, index=False)

    # 8. Build and save metadata
    adj_range = 0.0
    if adjustments:
        all_cum = [a.cumulative_adjustment for a in adjustments]
        total_gaps = sum(a.gap_points for a in adjustments)
        adj_range = max(abs(total_gaps), max(abs(c) for c in all_cum))

    meta = AdjustmentMetadata(
        ticker=ticker,
        processing_date=datetime.utcnow().isoformat(),
        num_rolls_detected=len(roll_events),
        total_adjustment_range=adj_range,
        source_file=str(source_path),
        output_file=str(output_path),
        roll_dates=[e.roll_date.isoformat() for e in roll_events],
        gap_points=[e.gap_points for e in roll_events],
        cumulative_adjustments=[a.cumulative_adjustment for a in adjustments],
    )

    metadata_dir.mkdir(parents=True, exist_ok=True)
    meta_path = metadata_dir / f"{ticker.name}.json"
    meta_path.write_text(json.dumps(_serialize_metadata(meta), indent=2))

    logger.info(
        "%s: done. rolls=%d, adj_range=%.2f",
        ticker.name,
        meta.num_rolls_detected,
        meta.total_adjustment_range,
    )
    return meta


def _process_ticker_safe(
    ticker_name: str,
    input_dir: str,
    output_dir: str,
    metadata_dir: str,
) -> str:
    """Wrapper for parallel execution (picklable arguments)."""
    try:
        ticker = Ticker[ticker_name]
        process_ticker(ticker, Path(input_dir), Path(output_dir), Path(metadata_dir))
        return f"{ticker_name}: OK"
    except Exception as e:
        logger.error("%s: FAILED - %s", ticker_name, e)
        return f"{ticker_name}: FAILED - {e}"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Back-adjust futures OHLC data",
    )
    parser.add_argument("--ticker", type=str, help="Single ticker to process (e.g. ES)")
    parser.add_argument("--all", action="store_true", help="Process all tickers")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    parser.add_argument(
        "--input-dir", type=str, default=str(_DEFAULT_INPUT),
        help="Input directory for parquet files",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(_DEFAULT_OUTPUT),
        help="Output directory for adjusted files",
    )
    parser.add_argument(
        "--metadata-dir", type=str, default=str(_DEFAULT_METADATA),
        help="Directory for metadata JSON files",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = build_arg_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    metadata_dir = Path(args.metadata_dir)

    if args.all:
        tickers = [t for t in Ticker if (input_dir / f"{t.name}.parquet").exists()]
        logger.info("Processing %d tickers", len(tickers))

        if args.parallel:
            with ProcessPoolExecutor() as pool:
                futures = {
                    pool.submit(
                        _process_ticker_safe,
                        t.name, str(input_dir), str(output_dir), str(metadata_dir),
                    ): t.name
                    for t in tickers
                }
                for future in as_completed(futures):
                    print(future.result())
        else:
            for t in tickers:
                try:
                    process_ticker(t, input_dir, output_dir, metadata_dir)
                except Exception as e:
                    logger.error("%s: FAILED - %s", t.name, e)

    elif args.ticker:
        ticker = Ticker[args.ticker.upper()]
        process_ticker(ticker, input_dir, output_dir, metadata_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/back_adjustment/test_orchestrator.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add data_cleaning/back_adjustment/orchestrator.py tests/back_adjustment/test_orchestrator.py
git commit -m "feat(T005): add orchestrator with CLI for back-adjustment pipeline"
```

---

## Task 6: Validator and Comparison Tools (T006)

**Files:**
- Create: `data_cleaning/back_adjustment/validator.py`
- Create: `tests/back_adjustment/test_validator.py`

**Step 1: Write the failing tests**

Create `tests/back_adjustment/test_validator.py`:
```python
from __future__ import annotations

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from data_cleaning.back_adjustment.validator import (
    compare_price_levels,
    compare_roll_dates,
    generate_comparison_report,
)
from utils.enums import Ticker


def _make_aligned_data(
    n: int = 100,
    offset: float = 5.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create two daily DataFrames with a known offset."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    base = 4000.0 + np.arange(n, dtype=float) + rng.randn(n) * 0.5

    old_df = pd.DataFrame({
        "datetime": dates.strftime("%Y-%m-%d"),
        "close": base,
    })
    new_df = pd.DataFrame({
        "Date": dates,
        "Close": base + offset + rng.randn(n) * 0.1,
    })
    return old_df, new_df


class TestValidator:

    def test_price_level_comparison(self) -> None:
        """compare_price_levels returns correlation, mean_diff, etc."""
        old_df, new_df = _make_aligned_data(offset=5.0)
        stats = compare_price_levels(old_df, new_df)

        assert "correlation" in stats
        assert "mean_diff" in stats
        assert "max_divergence" in stats
        assert "rmse" in stats
        assert stats["correlation"] > 0.99
        assert abs(stats["mean_diff"] - 5.0) < 1.0

    def test_roll_date_comparison(self, tmp_path: Path) -> None:
        """compare_roll_dates returns DataFrame with expected columns."""
        # Create metadata with known roll dates
        meta = {
            "ticker": "ES",
            "roll_dates": ["2023-03-15T00:00:00", "2023-06-15T00:00:00"],
            "gap_points": [10.0, 5.0],
        }
        meta_path = tmp_path / "ES.json"
        meta_path.write_text(json.dumps(meta))

        # Norgate-style df with Delivery Month changes
        dates = pd.bdate_range("2023-01-02", periods=200)
        delivery = [202303.0] * 50 + [202306.0] * 70 + [202309.0] * 80
        new_df = pd.DataFrame({
            "Date": dates,
            "Close": np.arange(200, dtype=float) + 4000,
            "Delivery Month": delivery,
        })

        result = compare_roll_dates(meta_path, new_df)
        assert isinstance(result, pd.DataFrame)
        assert "roll_date_old" in result.columns
        assert "roll_date_new" in result.columns
        assert "delta_days" in result.columns

    def test_report_generation(self, tmp_path: Path) -> None:
        """generate_comparison_report creates a markdown string."""
        old_df, new_df = _make_aligned_data()

        meta = {
            "ticker": "ES",
            "roll_dates": ["2023-03-15T00:00:00"],
            "gap_points": [10.0],
            "num_rolls_detected": 1,
        }
        meta_path = tmp_path / "ES.json"
        meta_path.write_text(json.dumps(meta))

        old_path = tmp_path / "old_ES.parquet"
        old_df.to_parquet(old_path, index=False)

        new_path = tmp_path / "new_ES.parquet"
        new_df.to_parquet(new_path, index=False)

        report = generate_comparison_report(
            Ticker.ES, old_path, new_path, meta_path,
            output_dir=tmp_path / "reports",
        )
        assert isinstance(report, str)
        assert "ES" in report
        assert "correlation" in report.lower() or "Correlation" in report

    def test_determinism(self) -> None:
        """Same input -> same output."""
        old_df, new_df = _make_aligned_data()
        a = compare_price_levels(old_df, new_df)
        b = compare_price_levels(old_df, new_df)
        assert a == b
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/back_adjustment/test_validator.py -v`
Expected: ImportError

**Step 3: Write the implementation**

Create `data_cleaning/back_adjustment/validator.py`:
```python
"""Compare back-adjusted legacy data against Norgate continuous futures.

Produces price-level statistics, roll-date alignment tables, and
markdown comparison reports.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from utils.enums import Ticker

_DEFAULT_REPORT_DIR = Path("docs/library/Data/comparisons")


def _align_daily(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Align two DataFrames on their date index, returning close series."""
    # Normalize old data dates
    if "datetime" in old_df.columns:
        old_dates = pd.to_datetime(old_df["datetime"]).dt.normalize()
        old_close = old_df["close"].values if "close" in old_df.columns else old_df["Close"].values
    else:
        old_dates = pd.to_datetime(old_df.index).normalize()
        old_close = old_df["close"].values if "close" in old_df.columns else old_df["Close"].values

    old_s = pd.Series(old_close, index=old_dates, dtype=np.float64, name="old_close")
    old_s = old_s[~old_s.index.duplicated(keep="first")]

    # Normalize new data dates
    if "Date" in new_df.columns:
        new_dates = pd.to_datetime(new_df["Date"]).dt.normalize()
    elif "datetime" in new_df.columns:
        new_dates = pd.to_datetime(new_df["datetime"]).dt.normalize()
    else:
        new_dates = pd.to_datetime(new_df.index).normalize()

    new_close_col = "Close" if "Close" in new_df.columns else "close"
    new_s = pd.Series(new_df[new_close_col].values, index=new_dates, dtype=np.float64, name="new_close")
    new_s = new_s[~new_s.index.duplicated(keep="first")]

    # Intersect dates
    common = old_s.index.intersection(new_s.index)
    return old_s.loc[common], new_s.loc[common]


def compare_price_levels(
    old_data: pd.DataFrame,
    new_data: pd.DataFrame,
) -> Dict[str, float]:
    """Compare price levels between old (back-adjusted) and new (Norgate) data.

    Returns
    -------
    dict with keys: correlation, mean_diff, max_divergence, rmse
    """
    old_s, new_s = _align_daily(old_data, new_data)

    if len(old_s) == 0:
        return {"correlation": 0.0, "mean_diff": 0.0, "max_divergence": 0.0, "rmse": 0.0}

    diff = new_s.values - old_s.values
    return {
        "correlation": float(np.corrcoef(old_s.values, new_s.values)[0, 1]),
        "mean_diff": float(np.mean(diff)),
        "max_divergence": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
    }


def compare_roll_dates(
    old_metadata_path: Path,
    new_data: pd.DataFrame,
) -> pd.DataFrame:
    """Compare roll dates between legacy metadata and Norgate Delivery Month.

    Returns DataFrame with columns: roll_date_old, roll_date_new, delta_days.
    """
    meta = json.loads(old_metadata_path.read_text())
    old_roll_dates = [pd.Timestamp(d) for d in meta.get("roll_dates", [])]

    # Detect Norgate roll dates from Delivery Month column
    new_roll_dates: list[pd.Timestamp] = []
    if "Delivery Month" in new_data.columns:
        date_col = "Date" if "Date" in new_data.columns else "datetime"
        dates = pd.to_datetime(new_data[date_col])
        dm = new_data["Delivery Month"]
        changes = dm != dm.shift(1)
        new_roll_dates = [pd.Timestamp(d) for d in dates[changes].iloc[1:].values]

    # Pair up: for each old roll, find nearest new roll
    rows: list[dict] = []
    for old_dt in old_roll_dates:
        best_new = None
        best_delta = None
        for new_dt in new_roll_dates:
            delta = abs((new_dt - old_dt).days)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_new = new_dt
        rows.append({
            "roll_date_old": old_dt,
            "roll_date_new": best_new,
            "delta_days": best_delta,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["roll_date_old", "roll_date_new", "delta_days"]
    )


def generate_comparison_report(
    ticker: Ticker,
    old_data_path: Path,
    new_data_path: Path,
    metadata_path: Path,
    output_dir: Path = _DEFAULT_REPORT_DIR,
) -> str:
    """Generate a markdown comparison report for a ticker.

    Returns the report string and saves it to output_dir/{ticker}_comparison.md.
    """
    old_df = pd.read_parquet(old_data_path)
    new_df = pd.read_parquet(new_data_path)

    # Price statistics
    stats = compare_price_levels(old_df, new_df)

    # Roll date comparison
    roll_df = compare_roll_dates(metadata_path, new_df)

    # Build report
    lines = [
        f"# {ticker.name} — Back-Adjustment Comparison Report",
        "",
        f"Generated: {pd.Timestamp.now().isoformat()}",
        "",
        "## Price Level Statistics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Correlation | {stats['correlation']:.6f} |",
        f"| Mean Difference | {stats['mean_diff']:.2f} |",
        f"| Max Divergence | {stats['max_divergence']:.2f} |",
        f"| RMSE | {stats['rmse']:.2f} |",
        "",
    ]

    if not roll_df.empty:
        lines.extend([
            "## Roll Date Comparison",
            "",
            "| Old Roll Date | New Roll Date | Delta (days) |",
            "|---------------|---------------|--------------|",
        ])
        for _, row in roll_df.iterrows():
            old_str = str(row["roll_date_old"].date()) if pd.notna(row["roll_date_old"]) else "N/A"
            new_str = str(row["roll_date_new"].date()) if pd.notna(row["roll_date_new"]) else "N/A"
            delta_str = str(row["delta_days"]) if pd.notna(row["delta_days"]) else "N/A"
            lines.append(f"| {old_str} | {new_str} | {delta_str} |")
        lines.append("")

    lines.extend([
        "## Summary",
        "",
        f"- Correlation of {stats['correlation']:.4f} between legacy back-adjusted and Norgate data.",
        f"- Mean price difference: {stats['mean_diff']:.2f} points (expected due to different roll methods).",
        f"- {len(roll_df)} roll dates compared.",
        "",
    ])

    report = "\n".join(lines)

    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{ticker.name}_comparison.md"
    report_path.write_text(report)

    return report
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/back_adjustment/test_validator.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add data_cleaning/back_adjustment/validator.py tests/back_adjustment/test_validator.py
git commit -m "feat(T006): add validator and comparison tools for Norgate validation"
```

---

## Task 7: Integration Test — End-to-End Pipeline on Real Data

**Files:**
- Create: `tests/back_adjustment/test_integration.py`

**Step 1: Write integration test**

Create `tests/back_adjustment/test_integration.py`:
```python
"""Integration test: run full pipeline on real ES data if available."""
from __future__ import annotations

import pytest
from pathlib import Path

from utils.enums import Ticker
from data_cleaning.back_adjustment.orchestrator import process_ticker
from data_cleaning.back_adjustment.validator import (
    compare_price_levels,
    generate_comparison_report,
)

_M1_DIR = Path("data/intraday_1min_original")
_NORGATE_ADJ = Path("data/norgate/continuous_futures/adjusted")
_HAS_REAL_DATA = (_M1_DIR / "ES.parquet").exists()
_HAS_NORGATE = (_NORGATE_ADJ / "ES.parquet").exists()


@pytest.mark.skipif(not _HAS_REAL_DATA, reason="No real M1 data available")
class TestEndToEndES:

    def test_process_es_m1(self, tmp_path: Path) -> None:
        """Process real ES M1 data through the full pipeline."""
        output_dir = tmp_path / "adjusted"
        metadata_dir = tmp_path / "metadata"

        meta = process_ticker(Ticker.ES, _M1_DIR, output_dir, metadata_dir)

        assert meta.num_rolls_detected > 0
        assert (output_dir / "ES.parquet").exists()
        assert (metadata_dir / "ES.json").exists()

    @pytest.mark.skipif(not _HAS_NORGATE, reason="No Norgate data available")
    def test_compare_es_with_norgate(self, tmp_path: Path) -> None:
        """Compare pipeline-adjusted ES against Norgate back-adjusted data."""
        import pandas as pd

        output_dir = tmp_path / "adjusted"
        metadata_dir = tmp_path / "metadata"
        process_ticker(Ticker.ES, _M1_DIR, output_dir, metadata_dir)

        # Resample adjusted M1 to daily for comparison
        adj_m1 = pd.read_parquet(output_dir / "ES.parquet")
        adj_m1["_date"] = pd.to_datetime(adj_m1["datetime"]).dt.normalize()
        daily = adj_m1.groupby("_date").agg(
            close=("close", "last"),
        ).reset_index().rename(columns={"_date": "datetime"})

        norgate = pd.read_parquet(_NORGATE_ADJ / "ES.parquet")

        stats = compare_price_levels(daily, norgate)
        # Expect reasonable correlation (>0.95) despite different roll methods
        assert stats["correlation"] > 0.90, (
            f"Correlation too low: {stats['correlation']:.4f}"
        )
```

**Step 2: Run integration tests**

Run: `pytest tests/back_adjustment/test_integration.py -v`
Expected: PASS (or SKIP if real data not extracted yet)

**Step 3: Run ALL tests together**

Run: `pytest tests/back_adjustment/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/back_adjustment/test_integration.py
git commit -m "test: add integration test for full back-adjustment pipeline"
```

---

## Execution Checklist

After all tasks are complete, verify:

```bash
# 1. All unit tests pass
pytest tests/back_adjustment/ -v

# 2. Run pipeline on real data (if extracted)
python -m data_cleaning.back_adjustment.orchestrator --ticker ES

# 3. Check output files exist
ls data/intraday_1min_adjusted/ES.parquet
ls data/adjustment_metadata/ES.json

# 4. Run all tickers
python -m data_cleaning.back_adjustment.orchestrator --all

# 5. Generate comparison report
python -c "
from data_cleaning.back_adjustment.validator import generate_comparison_report
from utils.enums import Ticker
from pathlib import Path
report = generate_comparison_report(
    Ticker.ES,
    Path('data/intraday_1min_adjusted/ES.parquet'),
    Path('data/norgate/continuous_futures/adjusted/ES.parquet'),
    Path('data/adjustment_metadata/ES.json'),
)
print(report)
"
```
