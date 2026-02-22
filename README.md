## Data Setup — Back-Adjusted Intraday Futures

Intraday data is too large for git (~2GB across all timeframes). Follow these steps to set up the data locally.

### Prerequisites

1. **Kibot data**: `kibot_data.zip` in `data/` (contains intraday parquet files for 10 tickers across 15 timeframes)
2. **Norgate Data Updater (NDU)**: Install and authenticate ([norgatedata.com](https://norgatedata.com)). Must be running on Windows.
3. **norgatedata package**: `pip install norgatedata`

### Quick Setup (existing tickers)

```bash
# 1. Extract all intraday files from Kibot zip (M1-M30, H1-H4; skips seconds and D/W/M)
python scripts/extract_kibot_data.py

# 2. Fetch Norgate reference data (NDU must be running)
python scripts/fetch_norgate_data.py

# 3. Run back-adjustment on all tickers (adjusts every timeframe per ticker)
python -m data_cleaning.back_adjustment.orchestrator --all

# 4. Verify
ls data/intraday_adjusted/ES/     # Should show M1_ES.parquet, M5_ES.parquet, H1_ES.parquet, etc.
ls data/adjustment_metadata/       # Should show JSON metadata per ticker
```

### Adding a New Ticker

When you have Kibot intraday data for a new ticker (e.g., `NG` for Natural Gas):

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
# Place the intraday parquets in the input directory
# Structure: data/intraday_original/{TICKER}/{TF}_{TICKER}.parquet
mkdir -p data/intraday_original/NG
cp /path/to/M1_NG.parquet data/intraday_original/NG/M1_NG.parquet
cp /path/to/M5_NG.parquet data/intraday_original/NG/M5_NG.parquet
# ... (all available timeframes)

# Re-fetch Norgate data (picks up new ticker)
python scripts/fetch_norgate_data.py

# Run back-adjustment for the new ticker (adjusts all timeframes)
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
data/kibot_data.zip                    # Source (tracked in git)
  └─ ohlc_data/{TICKER}/{TF}_{TICKER}.parquet
       │
       ▼ extract (extract_kibot_data.py — M1-M30, H1-H4 only)
data/intraday_original/                # Raw intraday data (gitignored)
  └─ {TICKER}/{TF}_{TICKER}.parquet
       │
       ▼ back-adjust (orchestrator — rolls detected once, applied to all TFs)
data/intraday_adjusted/                # Adjusted intraday data (gitignored)
  └─ {TICKER}/{TF}_{TICKER}.parquet
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