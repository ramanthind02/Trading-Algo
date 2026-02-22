## Data Setup — Back-Adjusted Intraday Futures

Intraday data is too large for git (~700MB). Follow these steps to set up the data locally.

### Prerequisites

1. **Kibot data**: Place `kibot_data.zip` in `data/` (contains M1 parquet files for 10 tickers)
2. **Norgate Data Updater (NDU)**: Install and authenticate ([norgatedata.com](https://norgatedata.com)). Must be running on Windows.
3. **norgatedata package**: `pip install norgatedata`

### Quick Setup (existing tickers)

```bash
# 1. Extract M1 files from Kibot zip (D files are tracked in git — not touched)
python scripts/extract_kibot_data.py

# 2. Fetch Norgate reference data (NDU must be running)
python scripts/fetch_norgate_data.py

# 3. Run back-adjustment on all tickers
python -m data_cleaning.back_adjustment.orchestrator --all

# 4. Verify
ls data/intraday_1min_adjusted/   # Should show adjusted parquet files
ls data/adjustment_metadata/       # Should show JSON metadata per ticker
```

### Adding a New Ticker

When you have Kibot M1 data for a new ticker (e.g., `NG` for Natural Gas):

**Step 1: Add the ticker to the Ticker enum** (if not already there)

Edit `utils/enums.py` and add the ticker to the `Ticker` enum.

**Step 2: Add a roll rule**

Edit `data_cleaning/back_adjustment/roll_rules.py` and add an entry to `ROLL_RULES`:

```python
Ticker.NG: RollRule(Ticker.NG, -3, "expiration", "3 days before monthly expiration"),
```

Reference: [Kibot rollover rules](https://www.kibot.com/rollover_rules.aspx) for the correct offset and reference point. Use negative offsets for "days before expiration" and positive offsets for "days from month end".

**Step 3: Add the Norgate symbol mapping**

Edit `scripts/fetch_norgate_data.py` and add the ticker to `TICKER_TO_NORGATE`:

```python
"NG": "&NG_CCB",  # back-adjusted
```

The unadjusted mapping is auto-derived (strips `_CCB`). To find the right Norgate symbol, run:

```python
import norgatedata
syms = norgatedata.database_symbols('Continuous Futures')
[s for s in syms if 'NG' in s]  # find your symbol
```

**Step 4: Place the data and run**

```bash
# Place the M1 parquet in the input directory
cp /path/to/M1_NG.parquet data/intraday_1min_original/NG.parquet

# Re-fetch Norgate data (picks up new ticker)
python scripts/fetch_norgate_data.py

# Run back-adjustment for the new ticker
python -m data_cleaning.back_adjustment.orchestrator --ticker NG

# Check the comparison report
cat docs/library/Data/comparisons/NG_comparison.md
```

**Step 5: Run tests**

```bash
pytest tests/back_adjustment/ -v
```

The `test_all_tickers_have_rules` test will fail if you added a Ticker enum member without a corresponding roll rule.

### Architecture

```
data/kibot_data.zip                    # Source (gitignored)
  └─ ohlc_data/{TICKER}/M1_{TICKER}.parquet
       │
       ▼ extract
data/intraday_1min_original/           # Raw M1 data (gitignored)
  └─ {TICKER}.parquet
       │
       ▼ back-adjust (orchestrator)
data/intraday_1min_adjusted/           # Adjusted M1 data (gitignored)
  └─ {TICKER}.parquet
       │
       ▼ compare (validator)
data/norgate/continuous_futures/       # Norgate reference (gitignored)
  ├─ adjusted/{TICKER}.parquet         #   Back-adjusted daily
  └─ unadjusted/{TICKER}.parquet       #   Unadjusted with Delivery Month
```

### Current Tickers

| Ticker | Instrument | Rolls | Notes |
|--------|-----------|-------|-------|
| ES | E-mini S&P 500 | 60 | Quarterly |
| NQ | E-mini Nasdaq | 60 | Quarterly |
| CL | Crude Oil | 180 | Monthly, negative adjusted prices in 2020 |
| GC | Gold | 75 | Bi-monthly |
| BP | British Pound | 60 | Quarterly |
| EU | Euro FX | 60 | Quarterly |
| JY | Japanese Yen | 60 | Quarterly |
| FV | 5-Year T-Note | 60 | Quarterly |
| TY | 10-Year T-Note | 60 | Quarterly |
| US | 30-Year T-Bond | 60 | Quarterly |