# MT5 timezones — the broker-server-time trap

> **One-line rule:** Every timestamp that comes out of MetaTrader 5 — bars *and*
> ticks — is in the **broker server's local time (Darwinex = EET/EEST, UTC+2/+3)**,
> but our parquet stores label it `tz="UTC"`. It is **not** real UTC. Treat all
> `data/mt5_data/**` timestamps as **broker time**, and remember the daily
> financing rollover sits at **00:00 broker time** (= 17:00 New York).

This page exists because the offset is invisible — the data *looks* like UTC
(it's tz-aware, labelled UTC) but is silently 2–3 hours ahead of real UTC and
7 hours ahead of New York. Several pieces of the MT5 stack were built on the
wrong assumption; see [Affected code](#affected-code-known-bugs).

---

## The facts (empirically verified, 2026-06-05)

### 1. MT5 returns server time, not UTC

`mt5.copy_rates_*` and `mt5.copy_ticks_range` both return a `time` field as a
Unix-epoch integer **of the broker server wall-clock**, not of real UTC. The
Darwinex server runs on **EET/EEST** (Eastern European Time, UTC+2 in winter,
UTC+3 in summer — the standard MT5 "Cyprus" broker convention).

Our scraper does `pd.to_datetime(raw["time"], unit="s", utc=True)` in both
[`_write_bars`](../../../data_platform/providers/mt5/scraper.py) and
`_write_ticks`. That call **attaches** a UTC label to the server-wall-clock
integer — it does **not** convert. So a bar whose server time is `00:00 EET`
lands in the parquet as `2026-06-04 00:00:00+00:00`, which is a lie: the real
instant was `2026-06-03 21:00:00Z` (summer).

Tick `time_msc` (milliseconds) has the **same** server-time offset — it is *not*
a true epoch. Do not assume "ticks are UTC, bars are EET"; **both are EET.**

### 2. The conversion

| You have (stored label) | To get real UTC | To get New York (ET) |
|---|---|---|
| broker time `T` | `T − 3h` (summer) / `T − 2h` (winter) | `T − 7h` (all year) |

The cleanest correct conversion in code is to **re-interpret** the stored value
as EET and convert, rather than hand-rolling ±2/±3:

```python
# stored_ts is tz-aware but MIS-labelled UTC; it is really broker EET wall-clock
broker_naive = stored_ts.tz_localize(None)                  # drop the false UTC tag
real_utc     = broker_naive.tz_localize("Europe/Bucharest").tz_convert("UTC")
new_york     = real_utc.tz_convert("America/New_York")
```

`Europe/Bucharest` is a clean EET/EEST zone (handles the DST switch). Athens or
Helsinki work identically. **Do not** use a fixed `+2`/`+3` unless you also
handle the EU DST transition dates.

### 3. ET ↔ broker is a constant +7h; the rollover is pinned to broker midnight

New York and EET both observe DST and switch within ~2 weeks of each other, so
**ET → broker = +7h essentially year-round**. The broker pins the daily
financing rollover to its own **00:00**, which is why:

- the last M1 bar before the daily gap is at a **constant `23:58` broker time
  every month** (verified Jan 2025 → Jun 2026, no seasonal drift), and
- `17:00 America/New_York` (the FX/CFD financing rollover) `= 00:00 broker`.

> Edge case: during the ~2 mismatch weeks (mid-March, late-Oct/early-Nov) when
> US and EU DST are out of step, the broker's 00:00 rollover momentarily maps to
> 16:00 or 18:00 ET instead of 17:00 ET. Anchor windows to **broker 00:00**, not
> to a hard-coded ET hour, to stay correct through these weeks.

### 4. The daily rollover dead-zone, in broker time

Verified on XAUUSD (continuous tick fetch, Wed 2026-06-03→04) and confirmed on
SP500/NDX/XAGUSD M1 session gaps:

| Phase | Broker-time window | Real ET | Tick behaviour |
|---|---|---|---|
| **Exit / pre-rollover** | `23:00 – 23:59` | 16:00–16:59 | liquid until ~23:45, then spread blows out (XAUUSD 0.53→1.32) into the close |
| **Dead zone** | `00:00 – 01:00` | 17:00–18:00 | **zero ticks** — market closed, no quotes |
| **Entry / post-rollover** | `01:00 – 02:00` | 18:00–19:00 | reopens with a wide spread spike (XAUUSD ~1.29, max 2.45) that tightens to ~0.66 over the next hour |

So to study the rollover you want broker-time windows **`[23:00, 00:00)` (exit)**
and **`[01:00, 02:00)` (entry)** — *not* 20:00–21:00 / 22:00–23:00.

### 5. US-equity-ETF cross-check

The ETFs (SPY/QQQ/GLD/SLV) only trade US RTH, which gives an independent anchor:
their M1 sessions run **`16:32 → 22:58` broker time**, i.e. exactly
`09:30 → 16:00 ET` shifted +7h. This is the second, independent confirmation that
**broker = ET + 7h = EET.**

---

## Why the `copy_ticks_range` request also looks shifted

When you pass a tz-aware `datetime` to `copy_ticks_range(sym, start, end, ...)`,
the MT5 Python package effectively uses the **wall-clock numbers** you give it as
**server time** (the tz tag is not honoured as a real-UTC conversion). So
requesting `20:00Z–21:00Z` returns the **server 20:00–21:00** ticks (real
13:00–14:00 ET), stamped back at `20:00–21:00`. Net effect: **request and result
are both in broker time** — consistent, but 7h off ET if you *thought* you were
asking in ET-derived UTC.

**Correct pattern:** decide the window in **broker wall-clock** (rollover = 00:00)
and pass those hours. If you must think in ET, convert ET → broker (`+7h`) before
calling, *not* ET → UTC.

---

## Affected code — ✅ FIXED (as of 2026-06-06)

All three files that previously carried the EET-offset bug have been corrected:

| File | Fix |
|---|---|
| [`data_platform/providers/mt5/rollover_tick_scraper.py`](../../../data_platform/providers/mt5/rollover_tick_scraper.py) | `exit_window` / `entry_window` now build datetimes directly in broker wall-clock (no `astimezone` conversion). Exit = broker 23:00–00:00; entry = broker 01:00–02:00. |
| [`data_platform/providers/mt5/build_symbol_sessions.py`](../../../data_platform/providers/mt5/build_symbol_sessions.py) | Reads broker HH:MM directly — comment explicitly says "do NOT tz_convert (that re-adds the offset)". |
| [`data_platform/nautilus/ingest.py`](../../../data_platform/nautilus/ingest.py) | `ingest_mt5_intraday_windowed` now documents: pass `rollover=time(0, 0)` with `tz="UTC"` for MT5 data (broker midnight); warns explicitly against passing `"America/New_York"`. |

**The rule is now enforced in code:** place every rollover window in broker
wall-clock with the rollover at 00:00, and only convert to ET/UTC for
human-facing reporting.

---

## Quick reference

```
Darwinex MT5 server time = EET/EEST = UTC+2 (winter) / UTC+3 (summer)
data/mt5_data/**  timestamps (bars AND ticks) are BROKER TIME, mislabelled UTC.

Daily financing rollover:
    00:00 broker   =  17:00 America/New_York  =  21:00Z (summer) / 22:00Z (winter)
Exit  (pre-roll) window : 23:00–23:59 broker   (16:00–16:59 ET)
Dead  zone              : 00:00–01:00 broker   (17:00–18:00 ET)  ← no ticks
Entry (post-roll) window: 01:00–02:00 broker   (18:00–19:00 ET)

real_utc = broker_ts.tz_localize(None).tz_localize("Europe/Bucharest").tz_convert("UTC")
new_york = real_utc.tz_convert("America/New_York")        # == broker − 7h
```

---

## Daily-bar snapshot for cross-feed comparison

When building a **daily** series from MT5 M1 to compare against Norgate (or any
true-UTC/ET source), snapshot the **last M1 close ≤ broker 23:00 (= 16:00 ET US
cash close)** and label the day by the broker calendar date (= ET date, since
23:00 EET is 16:00 ET same day):

```python
t   = pd.to_datetime(df["time"], utc=True)          # stored broker tz, kept as-is
mod = t.dt.hour * 60 + t.dt.minute
keep = mod <= 23 * 60                                 # <= broker 23:00 == 16:00 ET
daily = (pd.DataFrame({"date": t[keep].dt.tz_convert(None).dt.normalize(),
                       "close": df["close"].values[keep.values]})
         .groupby("date")["close"].last())
```

A snapshot-hour sweep maximises daily-log-return correlation to the matching
Norgate close **at broker 23:00** — the same +7h fingerprint, measured on daily
returns rather than tick gaps:

| MT5 symbol | Norgate ref | corr @ broker-23:00 | corr if mis-read as "16:00 UTC" |
|---|---|---|---|
| SPY (ETF CFD)   | SPY ETF   | **0.995** | 0.71 |
| QQQ (ETF CFD)   | NDX/QQQ   | **0.998 / 0.996** | 0.73 |
| SP500 (idx CFD) | SPX cash  | **0.988** | 0.72 |
| XAUUSD (cmdty)  | GLD ETF   | **0.996** | — |
| XAGUSD (cmdty)  | SLV ETF   | **0.995** | — |

Harness: `research/feed_comparison/feed_tz_calibrate.py` (snapshot-hour sweep + session profiler),
`research/feed_comparison/feed_compare.py` (cross-feed daily comparison, `s_mt5_snapshot`).

---

## Related

- [[hybrid_tick_backtest]] — the bars-spine + on-demand-ticks design that consumes these stores
- [[mt5_data_scraper]] — the scraper that writes the mislabelled timestamps
- [[darwinex_universe]] — the Darwinex symbol set these timestamps belong to

> _Verified against current code via CodeGraph on 2026-06-07. (Original empirical
> verification: live Darwinex terminal and `data/mt5_data/`, 2026-06-05, commit
> `c3bd8a4`: XAUUSD continuous tick fetch located the dead-zone at 00:00–01:00
> broker time; SP500/NDX/XAUUSD/XAGUSD M1 gaps and SPY/QQQ/GLD/SLV RTH sessions
> confirm broker = ET + 7h = EET.)_
