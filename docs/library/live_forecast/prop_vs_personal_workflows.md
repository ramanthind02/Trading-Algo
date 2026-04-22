# Prop vs Personal Forecast Workflows

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
| Daily candle for today   | Not needed -- script runs **after** close  | **Synthesized** from today's 15-min bars             |
| Schedule                 | **6:00 PM ET** (3:00 PM PT)                | **3:45 PM ET** (12:45 PM PT)                         |
| Telegram channel         | Enigma Notifications (`-1002856645393`)    | Enigma Signals - Personal Account (`-1003955204069`) |

---

## Prop firm workflow

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
3. Upserts bars into the central cache; refreshes any stale bias artifacts.
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
2. Script connects to TWS, fetches daily bars up to yesterday.
3. **Synthesizes a partial daily candle for today** from 15-min intraday bars
   (O=first, H=max, L=min, C=latest, V=sum). This replaces any placeholder
   IB returned for today.
4. Upserts bars into the central cache; refreshes stale bias artifacts.
5. Builds a `GlobalPortfolio` from every ensemble under `vault_personal/`
   (currently just `buy_hold_long`).
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

## Changing the configs

| What to change                             | Where                                                            |
| ------------------------------------------ | ---------------------------------------------------------------- |
| Prop capital                               | `configs/live_forecast_config_prop.json` → `account.capital_usd` |
| Personal capital                           | `configs/live_forecast_config_personal.json` → `account.capital_usd` |
| Which tickers the prop firm can trade      | `configs/live_forecast_config_prop.json` → `tradeable_tickers`   |
| ETF mapping for personal                   | `configs/live_forecast_config_personal.json` → `instruments[T].etf` |
| Paper vs live port                         | `*.json` → `connection.port` (7497 paper / 7496 live)            |
| Schedule time                              | `deploy/setup_scheduled_task.bat` → `/st HH:MM`                  |
| Telegram bot / channel                     | `deployment/telegram_notifier.py` → `_PROP_*` / `_PERSONAL_*`    |
