# Prop vs Personal Forecast Workflows

> ⚠️ Slated for rewrite under the NautilusTrader migration (WP-4 live execution). See docs/refactor/nautilus/.

Two scripts run daily, both on top of the same pipeline
(`scripts/enigma_live_forecast.py`). They differ in vault, instrument type,
timing, and Telegram channel. Pick the workflow you care about below.

---

## At a glance

|                          | Prop firm                                  | Personal account                                     |
| ------------------------ | ------------------------------------------ | ---------------------------------------------------- |
| Entrypoint               | `scripts/enigma_prop_forecast.py`          | `scripts/enigma_personal_forecast.py`                |
| Batch script             | `deploy/run_prop_forecast.bat`             | `deploy/run_personal_forecast.bat`                   |
| Config                   | `configs/live_forecast_config_prop.json`   | `configs/live_forecast_config_personal.json`         |
| Vault                    | `vault/`                                   | `vault_personal/`                                    |
| Instruments              | Micro futures (MES, MNQ, MGC, M2K, MYM, ZN for TLT) | ETFs (SPY, QQQ, GLD, IWM, DIA, TLT)         |
| Broker                   | Tradecopia → prop firm (Apex / MFFU / TradeDay / Lucid / Topstep) | Interactive Brokers directly                 |
| Capital                  | $50,000 (per-account)                      | $1,000 (test) -- bump later                          |
| Tradeable filter         | `tradeable_tickers: ["ES", "NQ", "GC"]` (Apex subscription) | full set from `buy_hold_long`             |
| Daily candle for today   | Not needed -- script runs **after** close  | **Session-only** bar from 15-min data (not cached; see below) |
| Schedule                 | **6:00 PM ET** (3:00 PM PT)                | **3:45 PM ET** (12:45 PM PT)                         |
| Telegram channel         | Enigma Notifications (`-1002856645393`)    | Enigma Signals - Personal Account (`-1003955204069`) |

## Calendars and clocks (America/New_York)

Understanding **which “daily close”** matters avoids mixing equity cash hours with futures settlement hours.

| Concept | Time (ET) | Where it is defined |
| -------- | --------- | ------------------- |
| **US equity regular session** (NYSE/Nasdaq) | **9:30 AM – 4:00 PM** | Market calendar; used for fractional-share cutoffs in the personal workflow. |
| **Official cash “daily” for equities** | Bar dated **T** typically completes at **4:00 PM** on **T** | IB and most data vendors; the prop script does **not** use this for futures. |
| **CME-style daily for ES/NQ/RTY/YM/GC** (continuous futures in IB) | Prior session → **5:00 PM** boundary (ETH); vendor wording varies | `enigma_prop_forecast.py` docstring; run at **6:00 PM ET** so today’s official futures daily is available. |
| **Personal pre-close run** | **~3:45 PM** | `enigma_personal_forecast.py` docstring — before **4:00 PM** equity close for fractional ETF orders. |
| **Prop daily run** | **~6:00 PM** | Same module docstring — after the **5:00 PM** futures window and settlement buffer. |

### IB TWS “Financial instrument description” (reference)

Templates below come from **Interactive Brokers → TWS** (Financial instrument description). Exchanges occasionally change rules, contract rolls shift symbols, and holidays alter sessions — **re-open the Contract Description** for each live root before relying on these hours for execution.

#### CME equity index futures — example E-mini NASDAQ-100 (`NQM6`, CME)

| Item | As shown in TWS |
| ----- | ---------------- |
| Time zone on form | **US/Central** |
| **Regular trading session (RTH)** | **08:30 – 16:00** US/Central |
| **Total hours (includes overnight / extended)** | **17:00\*** – **16:00** US/Central — the asterisk means **17:00** is on the **calendar day before** the listed session date |

**US/Eastern shorthand (when US/Central is CDT and US/Eastern is EDT, Central + 1 hour):** RTH ≈ **09:30 – 17:00 US/Eastern**; the long “Globex-style” window runs from **18:00 US/Eastern on the prior calendar date** through **17:00 US/Eastern** on the session date.

**Extrapolate to:** Other **CME / CBOT equity index** products used for CONTFUT-style pulls in this repo (`ES`, `NQ`, `RTY`, `YM`, and micros such as `MES`, `MNQ`, `M2K`, `MYM`) **unless** that symbol’s own Contract Description disagrees. **COMEX** (`GC` / `MGC`) and **CBOT** rates (`ZN`, etc.) can differ — open each instrument in TWS.

#### US-listed ETFs — example SPY (SPDR S&P 500 ETF Trust, ARCA / SMART)

| Item | As shown in TWS |
| ----- | ---------------- |
| Time zone on form | **US/Eastern** |
| **Regular trading session (RTH)** | **09:30 – 16:00** US/Eastern |
| **Total available (premarket + regular + post)** | **04:00 – 20:00** US/Eastern |
| Other | IB lists **overnight trading** as available; SPY **trades in fractions** (relevant to the personal profile and IB order cutoffs). |

**Extrapolate to:** `QQQ`, `IWM`, `DIA`, `GLD`, `TLT`, and other **US equity ETFs** mapped off index futures in the personal config, until that ETF’s Contract Description differs.

**Important:** The prop pipeline is **not** tied to the **4:00 PM** equity cash close; it is timed for **CME daily futures** completion. If you ever moved the prop task to **4:00 PM ET**, you would likely fetch **incomplete** futures dailies for “today.”

**Migrating off cached partial dailies:** Before the pipeline change, the personal profile could persist a synthetic “today” row into ``.cache/trading_algo/central_cache/candles/*/D.parquet``. Later IB upserts usually overwrote the same calendar date with the official bar (``upsert`` keeps the last row per date). To drop **today’s** NY session row or bulk-trim from a cutoff date, run ``python scripts/purge_ephemeral_daily_candles.py --help`` (use ``--dry-run`` first).

**Prediction window:** `configs/live_forecast_config_*.json` → `data.prediction_daily_max_bars` (default **500**) trims the **daily** candle history fed into live `fit`/`predict` so bias features stay within a bounded lookback; **monthly** candles still use the full overlap window from the cache.

---

**What this does:** runs one forecast per day at 6:00 PM ET, converts the
position target into micro futures contracts, and posts to the prop firm
Telegram channel. A human (or Tradecopia copy-trader) places the resulting
trades on Apex / MFFU / TradeDay / Lucid / Topstep accounts.

**Why 6:00 PM ET:** the CME daily candle settles at 5:00 PM ET and the
settlement window ends at 6:00 PM ET. Waiting until 6:00 PM ET means we use
the official closed daily bar for today and can enter new positions
immediately after.

**Steps (daily):**

1. `run_prop_forecast.bat` runs (Windows Task Scheduler: `TradingAlgo\PropForecast`).
2. Script connects to TWS, fetches daily bars to fill any gap since last run.
3. Upserts **completed** daily bars into the central cache; refreshes any stale bias artifacts.
4. Builds a `GlobalPortfolio` from every ensemble under `vault/`.
5. Generates today's `position_fraction` for ES, NQ, GC, RTY, TLT.
6. Filters output to `tradeable_tickers` (ES, NQ, GC only) and converts to
   micro futures contracts.
7. Formats a Telegram message with the signal strength and fractional /
   rounded whole contract counts, and posts to **Enigma Notifications**.
8. Trader (or Tradecopia) executes the whole-contract trades on each prop
   firm account.

**Manual test run:**
```bash
# Dry run -- prints the Telegram message instead of sending
python scripts/enigma_prop_forecast.py --dry-run --port 7497

# Production -- actually sends
python scripts/enigma_prop_forecast.py --port 7497
```

---

## Personal account workflow

**What this does:** runs one forecast per day at 3:45 PM ET, converts the
position target into ETF fractional shares, and posts to the personal Telegram
channel. Orders are placed on Interactive Brokers directly (no prop firm copier).

**Why 3:45 PM ET:** the US equity market closes at 4:00 PM ET. IB disables
fractional-share entry outside regular trading hours, so the signal must
arrive with enough buffer to place orders before 4:00 PM ET. Running at
3:45 PM ET also means today's daily candle isn't closed yet -- so the
script synthesizes one from 15-minute intraday bars.

**Steps (daily):**

1. `run_personal_forecast.bat` runs (Windows Task Scheduler: `TradingAlgo\PersonalForecast`).
2. Script connects to TWS, fetches daily bars up to **completed** sessions; **drops** any row dated **today (NY calendar)** from the upsert payload so incomplete IB dailies are not cached.
3. Builds a **session-only** partial row for today from 15-min bars (**extended + overnight**, `use_rth=0`); merges it **in memory** for this run only (not persisted).
4. Upserts **completed** dailies into the central cache; refreshes stale bias artifacts.
5. Builds a `GlobalPortfolio` from every ensemble under `vault_personal/`.
6. Generates today's `position_fraction` for ES, NQ, GC, RTY.
7. Converts to ETF fractional shares via the `instruments[ticker].etf`
   mapping (ES→SPY, NQ→QQQ, GC→GLD, RTY→IWM, TLT→TLT, YM→DIA).
8. Fetches current ETF prices from TWS for the sizing table.
9. Formats a Telegram message with signal strength and fractional share
   counts, and posts to **Enigma Signals - Personal Account**.
10. Trader places the ETF orders in IB before 4:00 PM ET.

**Manual test run:**
```bash
# Dry run
python scripts/enigma_personal_forecast.py --dry-run --port 7497

# Production
python scripts/enigma_personal_forecast.py --port 7497
```

---

## One-time setup

### Prerequisites every day
- **TWS or IB Gateway running** with API enabled (port 7497 paper / 7496 live)
- Computer on and connected to the internet at the scheduled times

### Register the Windows scheduled tasks (one-time, as Administrator)
```bat
deploy\setup_scheduled_task.bat
```
This creates two daily tasks under `TradingAlgo\`:
- `TradingAlgo\PropForecast`     -- 3:00 PM PT (= 6:00 PM ET)
- `TradingAlgo\PersonalForecast` -- 12:45 PM PT (= 3:45 PM ET)

If you're in a timezone other than Pacific, edit the `/st` values in
`deploy/setup_scheduled_task.bat` before running.

### Useful commands
```bat
REM Check tasks exist
schtasks /query /tn "TradingAlgo\PropForecast"
schtasks /query /tn "TradingAlgo\PersonalForecast"

REM Run a task manually (e.g. to test the scheduled environment)
schtasks /run /tn "TradingAlgo\PropForecast"

REM Delete a task
schtasks /delete /tn "TradingAlgo\PropForecast" /f
```

Output from scheduled runs appends to `deploy/forecast.log`.

---

## Scheduling on your own PC (local automation)

### Windows

1. One-time: run `deploy\setup_scheduled_task.bat` **as Administrator** (or prefer `deploy\setup_scheduled_task.ps1` — it avoids some `schtasks` timezone quirks). Edit the script first so paths match this repo and your venv (repo convention: `.\.venv\Scripts\python.exe` from the repo root).
2. The tasks call `deploy\run_prop_forecast.bat` and `deploy\run_personal_forecast.bat`, which `cd` to the repo and run the wrappers. Update the hard-coded `cd` path inside those `.bat` files if needed.
3. Verify: `schtasks /query /tn "TradingAlgo\PropForecast"` and `schtasks /run /tn "TradingAlgo\PropForecast"` for a dry test.

### Linux (cron)

There is no installer script; use the system crontab or a user unit timer.

1. Pick the same interpreter you use for development (absolute path to the repo-root venv `python`).
2. `crontab -e` and add lines (adjust paths and log location), for example:

```cron
# Personal: Mon–Fri 3:45 PM America/New_York
45 15 * * 1-5 cd /path/to/Trading-Algo && TZ=America/New_York /path/to/Trading-Algo/.venv/bin/python scripts/enigma_personal_forecast.py >> /path/to/Trading-Algo/deploy/forecast.log 2>&1

# Prop: Mon–Fri 6:00 PM America/New_York
0 18 * * 1-5 cd /path/to/Trading-Algo && TZ=America/New_York /path/to/Trading-Algo/.venv/bin/python scripts/enigma_prop_forecast.py >> /path/to/Trading-Algo/deploy/forecast.log 2>&1
```

Cron’s `TZ=` affects the process environment; ensure TWS/Gateway is reachable from that host. Prefer a small wrapper shell script if you need `source`-style env vars.

---

## Replay recent prop forecasts (sanity check)

To tabulate **forecast_score** and **position_fraction** for the last *N* **business** days using the same vault + central cache as production (no TWS, no Telegram), run::

    python scripts/replay_prop_forecast_window.py --trading-days 10
    python scripts/replay_prop_forecast_window.py --trading-days 10 --output deploy/prop_forecast_replay.csv

This fits once like ``enigma_prop_forecast``, runs ``predict_from_cache`` on the full overlap, then **filters** rows to the replay window. It is **not** a causal walk-forward (the global weight layer is not re-fit per day). For strict as-of research, use ``portfolio_research`` walk-forward tooling instead.

---

## Changing the configs

| What to change                             | Where                                                            |
| ------------------------------------------ | ---------------------------------------------------------------- |
| Prop capital                               | `configs/live_forecast_config_prop.json` → `account.capital_usd` |
| Personal capital                           | `configs/live_forecast_config_personal.json` → `account.capital_usd` |
| Which tickers the prop firm can trade      | `configs/live_forecast_config_prop.json` → `tradeable_tickers`   |
| ETF mapping for personal                   | `configs/live_forecast_config_personal.json` → `instruments[T].etf` |
| Paper vs live port                         | `*.json` → `connection.port` (7497 paper / 7496 live)            |
| Schedule time                              | `deploy/setup_scheduled_task.bat` → `/st HH:MM`                  |
| Live daily lookback for fit/predict        | `configs/live_forecast_config_*.json` → `data.prediction_daily_max_bars` (default **500**; `0` = unlimited) |
| Telegram bot / channel                     | `deployment/telegram_notifier.py` → `_PROP_*` / `_PERSONAL_*`    |

> _Verified against commit a07b6bf on 2026-06-04 (docs Phase A)._
