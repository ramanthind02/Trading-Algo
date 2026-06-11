# Darwinex Live — Trading Universe Reference

Broker: **Darwinex** | Server: `liveUK-mt5.darwinex.com` | Account: `4000093084`  
Terminal build: `5836` | Total symbols: **844** | Measured: 2026-06-04

---

## Quick-reference card

| Category | Count | D1 history from | Tick history from | M1 storage (all syms) |
|----------|-------|----------------|------------------|----------------------|
| FX majors | 8 | **1971** | **2011** | 0.6 GB |
| FX crosses | 30 | **1993–2002** | **2011** | 2.3 GB |
| Indices (CFDs) | 10 | 2007–2012 | 2018 (NDX/metals) / 2023 (SP500) | 0.3 GB |
| Commodities | 4 | 1998–2010 | 2018–2020 | 0.1 GB |
| US Stocks | 692 | 2008 | 2018–2020 | 5–17 GB* |
| ETFs | 100 | 2010 | 2021 | 0.7–2.3 GB* |
| **Total (844)** | | | | **9–23 GB M1, ~80 GB ticks** |

> \* Range = realistic (M1 from 2021) to optimistic (M1 from D1 start). Actual stock M1 depth unknown until first fetch.  
> Tick storage is for a focused 9-symbol subset (FX + metals + oil + NDX); full-universe ticks would be ~400 GB.

**Key symbol name differences from standard conventions:**

| Standard name | Darwinex symbol | Note |
|---------------|----------------|------|
| NAS100 / US100 | **NDX** | Use NDX |
| GER40 / DAX40 / DE40 | **GDAXI** | Use GDAXI |
| Dow Jones | **WS30** | Use WS30 |
| BRK.B | **BRKb** | Different capitalisation |
| LIN (Linde) | — | Not in terminal |
| BTC/USD | — | No crypto on this account |

---

## Universe summary

| Category | Count | Description |
|----------|-------|-------------|
| US Stocks | 692 | NYSE, Nasdaq, DOW — roughly S&P 500 + S&P 400 mid-caps |
| ETFs | 100 | Broad market, sector SPDR, factor, international, thematic |
| Forex | 38 | G10 majors + crosses + MXN, NOK, SEK, SGD exotics |
| Indices (CFDs) | 10 | SP500, NDX, GDAXI, WS30, UK100, FCHI40, STOXX50E, NI225, AUS200, SPA35 |
| Commodities | 4 | XAUUSD, XTIUSD, XAGUSD, XNGUSD |

---

## Daily bar history depth

> Fetched via `copy_rates_range(sym, datetime(1970,1,1,UTC), now)`.  
> This is the correct API call — `copy_rates_from` and `copy_rates_from_pos` return 0 bars  
> due to MT5 IPC cache behaviour. On first call this triggers a broker download (~30-90s/symbol);  
> subsequent calls read from local terminal cache instantly.  
> `Yrs` = D1 bar count ÷ 252 trading days.

### Forex

| Symbol | Description | D1 bars | Oldest date | Yrs |
|--------|-------------|---------|-------------|-----|
| EURUSD | Euro / USD | 14,288 | **1971-01-04** | 56.7 |
| USDJPY | USD / Yen | 14,301 | **1971-01-04** | 56.8 |
| USDCHF | USD / Swiss Franc | 14,256 | **1971-01-04** | 56.6 |
| GBPUSD | GBP / USD | 8,594 | 1993-05-12 | 34.1 |
| AUDUSD | AUD / USD | 8,605 | 1993-04-27 | 34.1 |
| USDCAD | USD / CAD | 8,605 | 1993-04-28 | 34.1 |
| EURGBP | EUR / GBP | 8,602 | 1993-05-03 | 34.1 |
| EURJPY | EUR / JPY | 8,476 | 1993-04-27 | 33.6 |
| GBPJPY | GBP / JPY | 8,603 | 1993-04-19 | 34.1 |
| NZDUSD | NZD / USD | 8,410 | 1994-02-01 | 33.4 |
| EURCAD | EUR / CAD | 6,903 | 1999-08-02 | 27.4 |
| EURAUD | EUR / AUD | 6,337 | 2002-01-14 | 25.1 |

All 38 FX pairs available. The table above covers the most strategically relevant ones.  
Full pair list: `AUDCAD AUDCHF AUDJPY AUDNZD AUDUSD CADCHF CADJPY CHFJPY EURAUD EURCAD EURCHF EURGBP EURJPY EURMXN EURNOK EURNZD EURSEK EURUSD GBPAUD GBPCAD GBPCHF GBPJPY GBPMXN GBPNOK GBPNZD GBPSEK GBPUSD NZDCAD NZDCHF NZDJPY NZDUSD USDCAD USDCHF USDJPY USDMXN USDNOK USDSEK USDSGD`

### Indices (CFDs)

| Symbol | Description | D1 bars | Oldest date | Yrs |
|--------|-------------|---------|-------------|-----|
| SP500 | S&P 500 | 4,763 | 2008-08-07 | 18.9 |
| NDX | Nasdaq 100 | 4,764 | 2008-08-06 | 18.9 |
| WS30 | Dow Jones 30 | 4,763 | 2008-08-07 | 18.9 |
| NI225 | Nikkei 225 | 4,978 | 2007-03-15 | 19.8 |
| UK100 | FTSE 100 | 4,655 | 2008-03-06 | 18.5 |
| FCHI40 | CAC 40 | 4,669 | 2008-04-01 | 18.5 |
| AUS200 | ASX 200 | 4,727 | 2008-05-09 | 18.8 |
| GDAXI | DAX 40 | 3,545 | 2012-01-23 | 14.1 |
| STOXX50E | Euro Stoxx 50 | 3,500 | 2012-08-30 | 13.9 |
| SPA35 | IBEX 35 | — | — | — |

> **Important symbol names:** Darwinex uses non-standard names for indices.  
> Do NOT use NAS100, US100, GER40, DAX40, DE40 — they are not in this terminal.

### Commodities

| Symbol | Description | D1 bars | Oldest date | Yrs |
|--------|-------------|---------|-------------|-----|
| XAUUSD | Gold | 7,432 | **1998-04-22** | 29.5 |
| XAGUSD | Silver | 6,288 | 2002-09-15 | 25.0 |
| XNGUSD | Natural Gas | 4,978 | 2007-01-02 | 19.8 |
| XTIUSD | WTI Crude Oil | 4,308 | 2010-05-02 | 17.1 |

### US Stocks (representative sample)

All stocks share the same Darwinex history start date: **2008-05-05** (~18 years of daily data), except for names that IPO'd after that date.

| Symbol | Description | D1 bars | Oldest date | Yrs |
|--------|-------------|---------|-------------|-----|
| AAPL | Apple | 4,548 | 2008-05-05 | 18.0 |
| MSFT | Microsoft | 4,548 | 2008-05-05 | 18.0 |
| AMZN | Amazon | 4,548 | 2008-05-05 | 18.0 |
| NVDA | Nvidia | 4,545 | 2008-05-05 | 18.0 |
| GOOGL | Alphabet (A) | 4,548 | 2008-05-05 | 18.0 |
| JPM | JPMorgan | 4,548 | 2008-05-05 | 18.0 |
| V | Visa | 4,548 | 2008-05-05 | 18.0 |
| META | Meta | 3,530 | 2012-05-18 | 14.0 |
| TSLA | Tesla | 4,007 | 2010-06-29 | 15.9 |

> The 2008-05-05 start is a Darwinex data provider limit, not the actual listing date.  
> Stocks absent from the terminal: `BRK.A`, `LIN`. `BRK.B` appears as `BRKb`.  
> Stocks that IPO'd after 2008-05-05 start from their actual IPO date (e.g. META 2012-05-18, TSLA 2010-06-29).

### ETFs

> Darwinex provides a **different history limit for ETFs vs individual stocks**.  
> All ETFs tested start from **2010-01-11** (~16.4 years), while stocks start from **2008-05-05** (~18 years).  
> ETFs that launched after 2010-01-11 start from their actual inception date (e.g. ARKK Oct 2014).

**Broad market / fixed income / gold:**

| Symbol | Description | D1 bars | Oldest date | Yrs |
|--------|-------------|---------|-------------|-----|
| SPY | SPDR S&P 500 | 4,125 | 2010-01-11 | 16.4 |
| QQQ | Invesco Nasdaq 100 | 4,122 | 2010-01-11 | 16.4 |
| IWM | iShares Russell 2000 | 4,124 | 2010-01-11 | 16.4 |
| DIA | SPDR Dow Jones 30 | 4,124 | 2010-01-11 | 16.4 |
| TLT | iShares 20+ Yr Treasury | 4,122 | 2010-01-11 | 16.4 |
| GLD | SPDR Gold | 4,124 | 2010-01-11 | 16.4 |
| SLV | iShares Silver | 4,124 | 2010-01-11 | 16.4 |

**SPDR Sector ETFs (all 2010-01-11):**

| Symbol | Sector | D1 bars | Yrs |
|--------|--------|---------|-----|
| XLK | Technology | 4,122 | 16.4 |
| XLF | Financials | 4,121 | 16.3 |
| XLV | Health Care | 4,124 | 16.4 |
| XLI | Industrials | 4,124 | 16.4 |
| XLE | Energy | 4,120 | 16.3 |
| XLP | Consumer Staples | 4,124 | 16.4 |
| XLU | Utilities | 4,122 | 16.4 |
| XLB | Materials | 4,123 | 16.4 |
| XLC | Communication Services | ~4,120 | 16.4 |
| XLY | Consumer Discretionary | ~4,120 | 16.4 |

**ARK ETFs (inception date limited):**

| Symbol | Description | Approx oldest | Yrs |
|--------|-------------|---------------|-----|
| ARKK | ARK Innovation | ~2014-10-31 | ~11.6 |
| ARKG | ARK Genomic Revolution | ~2014-10-31 | ~11.6 |
| ARKW | ARK Next Generation Internet | ~2014-10-31 | ~11.6 |

Full ETF list (100 symbols):  
`AAXJ ARKG ARKK ARKW DGRO DIA DVY EEM EFA EFAV EFG EFV EMB ESGU EWJ EWT EWY EWZ EZU FDN FTEC FVD GDX GDXJ GLD GSLC HDV IBB ICLN IGV IHI IJH IJR IJS INDA ITOT IUSG IUSV IVE IVW IWB IWD IWF IWM IWN IWO IWS IYR IYW KRE MCHI MDY MTUM OEF PFF QQQ QUAL RSP SCHB SCHD SCHF SCHG SCHV SCZ SKYY SLV SMH SOXX SPDW SPLV SPY SPYG SPYV TIP TLT USMV VDE VEA VGT VHT VLUE VNQ VOE VPL VT VTI VWO VXF VXUS XBI XLB XLC XLE XLF XLI XLK XLP XLU XLV XLY`

---

## Tick history depth

> Fetched via `copy_ticks_from(sym, datetime(2000,1,1,UTC), 200_000, COPY_TICKS_ALL)`.  
> The 200k cap means the oldest date shown is where 200k ticks from year 2000 *ends*,  
> not necessarily the absolute oldest tick. True history may go further for low-volume symbols.

| Asset class | Symbols | Tick history starts |
|-------------|---------|-------------------|
| FX majors (EUR/GBP/JPY/AUD/CHF) | EURUSD GBPUSD USDJPY AUDUSD USDCHF | 2011-12-19 |
| Gold, DAX, FTSE, Dow | XAUUSD GDAXI UK100 WS30 | 2018-01 |
| Crude oil, AAPL | XTIUSD AAPL | 2020-01 |
| MSFT | MSFT | 2018-04-12 |
| AMZN, NVDA | AMZN NVDA | 2018-10 |
| S&P 500 index | SP500 | 2023-03-27 |
| NDX | NDX | no tick history |

---

## Data comparison: bars vs ticks

| Asset class | Tick history | Daily bar history | Gain from bars |
|-------------|-------------|-------------------|----------------|
| FX majors | 2011–2012 | **1971** | +40 years |
| Indices | 2023 (SP500) | **2007–2008** | +15 years |
| Gold | 2018 | **1998** | +20 years |
| Stocks/ETFs | 2018–2020 | **2008** | +10 years |

**Daily bars are the right foundation for all strategy backtesting on this universe.**  
Ticks are useful for execution research (spread, fill quality, intraday patterns) only.

---

## M1 bar storage — full 844-symbol universe

Computed analytically from confirmed history start dates and trading session minutes.
Bytes/bar: 14 (zstd parquet, 8 columns). See `scripts/dev/_mt5_m1_sizing.py`.

| Category | Symbols | Bars/sym | Total bars | GB |
|----------|---------|---------|-----------|-----|
| FX majors (from 2011-12-19, 1435 min/day) | 8 | 5,394,499 | 43M | 0.6 |
| FX crosses (from 2011-12-19, 1435 min/day) | 30 | 5,394,499 | 162M | 2.3 |
| Gold/Silver (from 2018-01-25, 1380 min/day) | 2 | 2,998,104 | 6M | 0.1 |
| WTI/NatGas (from 2020, 1380 min/day) | 2 | ~2,307,000 | 5M | 0.1 |
| Indices (from 2018, 1390 min/day) | 10 | ~3,020,000 | 30M | 0.4 |
| US stocks — **realistic** (from 2021, 390 min/day) | 692 | 532,770 | 369M | **5.2** |
| US stocks — **optimistic** (from 2008, 390 min/day) | 692 | 1,776,977 | 1,230M | **17.2** |
| ETFs — realistic (from 2021) | 100 | 532,770 | 53M | 0.7 |
| ETFs — optimistic (from 2010) | 100 | 1,611,226 | 161M | 2.3 |

**Total compressed storage:**

| Scenario | GB | Fits on 500 GB SSD? |
|----------|----|-------------------|
| Realistic (stocks/ETFs M1 from 2021) | **~9 GB** | Yes — trivially |
| Optimistic (stocks/ETFs M1 from D1 start) | **~23 GB** | Yes — trivially |

**Daily growth: ~5 MB/day → 1.4 GB/year** across all 844 symbols.

M1 bars are ~200× smaller than raw ticks for the same symbol/period (1440 bars/day
vs ~120k ticks/day for FX; 14 B/bar vs 25 B/tick compressed).

> **Confirmed (2026-06-05):** M1 reaches the same depth as D1 once `Max bars in
> chart` is set to Unlimited and the terminal is restarted. Measured: GBPUSD
> 9.78M M1 bars → 1993, XAUUSD → 1998, NDX/AAPL → 2008, NZDUSD 8.94M → 2000.
> Full-universe M1 backfill via `m1_backfill.py` (deep downloads serialise in the
> terminal → ~10 min/deep symbol, multi-night for all 844).

---

## API notes

> **Prerequisite for deep intraday bars:** set `Max bars in chart` = Unlimited
> (Tools → Options → Charts) and **restart the terminal**. Default 100,000 caps M1
> at ~70 days. After the lift, M1 reaches the same depth as D1 (FX → 1993, etc.).

### Working calls
```python
# Daily bars — full history in one call
rates = mt5.copy_rates_range(sym, datetime(1970,1,1,tzinfo=timezone.utc), datetime.now(timezone.utc))

# Deep M1 (and any intraday TF) — trigger the progressive download from NOW with a
# large count, then poll until the returned count stabilises (download landed).
rates = mt5.copy_rates_from(sym, mt5.TIMEFRAME_M1, datetime.now(timezone.utc), 30_000_000)

# Ticks — works without pre-loading charts
ticks = mt5.copy_ticks_from(sym, date_from, 200_000, mt5.COPY_TICKS_ALL)
```

### Gotchas
```python
# copy_rates_range over OLD dates returns empty on a cold base — it does not trigger
# the deep download. Use copy_rates_from(now, big_count) to populate, then range-read.
mt5.copy_rates_range(sym, old_from, old_to)       # empty until base is populated
mt5.copy_rates_from_pos(sym, timeframe, 0, count) # position-based; avoid for history
```

### IPC session rules
- `mt5.initialize()` must be called with **no arguments** to attach to the running terminal.  
  Passing path + credentials causes IPC conflicts and returns empty data.
- Only **one Python process** can hold the IPC channel at a time.
- After `mt5.shutdown()`, wait ~5 seconds before reinitialising.
- Do not run the scraper simultaneously with the live forecast server.

---

## Diagnostic scripts

| Script | Purpose |
|--------|---------|
| `data_platform/providers/mt5/probes/mt5_discovery.py` | Full symbol list + tick depth probe → `data/mt5_data/_discovery.json` |
| `data_platform/providers/mt5/probes/mt5_probe_daily.py` | D1/W1/MN1 depth for representative symbols → `data/mt5_data/_daily_history_depth.json` |
| `data_platform/providers/mt5/probes/mt5_probe_tf.py` | Multi-TF bar + tick depth table |
| `data_platform/providers/mt5/probes/mt5_minimal.py` | Bare connectivity sanity check |
| `data_platform/providers/mt5/scraper.py` | Incremental daily scraper (production use) |
| `data_platform/providers/mt5/probes/mt5_storage_estimate.py` | Storage sizing model |

> _Verified against current code via CodeGraph on 2026-06-07._
