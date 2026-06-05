"""Position sizing analysis for 100k account trading ETFs with 5:1 leverage."""
import MetaTrader5 as mt5, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ok = mt5.initialize()
if not ok:
    print("init failed"); sys.exit(1)

CAPITAL = 100_000

# Confirmed leverage / margin from order_calc_margin:
# 5:1  -> 20% margin  (68 ETFs)
# 3.3x -> 30% margin  (GLD)
# 1x   -> 100% margin (KRE)
# 30 ETFs have no live price (market closed / not streaming)
EXAMPLES = [
    # (sym, desc, price, margin_pct, daily_vol_pct)
    ("SPY",  "S&P 500 (benchmark)",  754.54, 0.20, 0.010),
    ("QQQ",  "Nasdaq 100",           744.46, 0.20, 0.013),
    ("IWM",  "Russell 2000",         287.78, 0.20, 0.014),
    ("DIA",  "Dow Jones",            508.54, 0.20, 0.009),
    ("TLT",  "Long Treasury",         85.29, 0.20, 0.006),
    ("GLD",  "Gold ETF",             407.93, 0.30, 0.009),  # 30% margin
    ("XLK",  "Technology SPDR",      196.36, 0.20, 0.012),
    ("XLF",  "Financials SPDR",       50.91, 0.20, 0.012),
    ("XLE",  "Energy SPDR",           58.82, 0.20, 0.013),
    ("XLV",  "Health Care SPDR",     147.63, 0.20, 0.009),
    ("XLU",  "Utilities SPDR",        43.77, 0.20, 0.009),
    ("ARKK", "ARK Innovation",        78.25, 0.20, 0.025),
    ("EEM",  "EM Equities",           69.89, 0.20, 0.011),
    ("KRE",  "Regional Banks",        67.90, 1.00, 0.015),  # NO leverage
]

# ── 1. Per-symbol leverage table ─────────────────────────────────────────────
print("Leverage and margin per ETF on Darwinex")
print()
print(f"  {'Symbol':<8} {'Price':>7}  {'Leverage':>9}  {'Margin/share':>13}  {'Max shares w/ 100k':>20}")
print("  " + "-" * 65)
for sym, desc, price, mp, _ in EXAMPLES:
    lev = 1 / mp
    margin_share = price * mp
    max_s = int(CAPITAL / margin_share)
    print(f"  {sym:<8} {price:>7.2f}  {lev:>8.1f}x  {margin_share:>13.2f}  {max_s:>20,}")

# ── 2. Equal-weight 10-ETF portfolio ─────────────────────────────────────────
N = 10
cap_per = CAPITAL / N
print()
print(f"Equal-weight {N}-ETF portfolio  (capital/position = {cap_per:,.0f})")
print()
print(f"  {'Symbol':<8} {'Shares':>7}  {'Notional':>11}  {'Margin used':>12}  {'1% move P&L':>13}  {'Daily vol P&L':>15}")
print("  " + "-" * 72)
total_notional = 0
total_margin   = 0
for sym, desc, price, mp, vol in EXAMPLES[:N]:
    margin_share = price * mp
    shares = int(cap_per / margin_share)
    notional = shares * price
    margin   = shares * margin_share
    move_1pct = notional * 0.01
    daily_vol  = notional * vol
    total_notional += notional
    total_margin   += margin
    print(f"  {sym:<8} {shares:>7}  {notional:>11,.0f}  {margin:>12,.0f}  {move_1pct:>13,.0f}  {daily_vol:>15,.0f}")

print("  " + "-" * 72)
print(f"  {'TOTAL':<8} {'':>7}  {total_notional:>11,.0f}  {total_margin:>12,.0f}")
print(f"  Leverage used: {total_notional/CAPITAL:.1f}x   Margin used: {total_margin/CAPITAL*100:.0f}% of capital")
print()

# ── 3. Key practical constraints ─────────────────────────────────────────────
print("Key constraints:")
print()
print(f"  Whole shares only (min lot = 1).  Fractional shares NOT available.")
print()
print("  High-price ETFs (SPY ~$755, QQQ ~$745, DIA ~$509):")
print(f"    Margin per share:  SPY={754.54*0.2:.0f}  QQQ={744.46*0.2:.0f}  DIA={508.54*0.2:.0f}")
print(f"    With {cap_per:,.0f}/position you get:")
print(f"      SPY: {int(cap_per/(754.54*0.2))} shares  = ${int(cap_per/(754.54*0.2))*754.54:,.0f} notional")
print(f"      QQQ: {int(cap_per/(744.46*0.2))} shares  = ${int(cap_per/(744.46*0.2))*744.46:,.0f} notional")
print()
print("  Low-price ETFs (XLF ~$51, XLU ~$44):")
print(f"    SPY: 1 share = $755 notional  (too coarse for small positions)")
print(f"    XLF: 1 share = $51 notional   (fine granularity)")
print(f"    XLU: 1 share = $44 notional   (fine granularity)")
print()
print("  GLD has 30% margin (3.3x leverage) — stricter than other ETFs.")
print("  KRE has 100% margin (1x, no leverage) — full cash required.")
print()
print("  30/100 ETFs show no live price — market closed or not streaming.")
print("  Those 30 cannot be traded/sized until a live price is available.")
print()

# ── 4. Carver vol-targeting approach ─────────────────────────────────────────
print("Carver vol-targeting: target 20% annualised portfolio vol")
ANN_VOL_TARGET = 0.20
daily_vol_target_portfolio = ANN_VOL_TARGET / (252 ** 0.5)
print(f"  Daily vol target: {daily_vol_target_portfolio*100:.2f}% of capital = ${CAPITAL*daily_vol_target_portfolio:,.0f}/day")
print()
print("  Per-instrument allocation at equal weight, 20% ann vol target:")
print(f"  {'Symbol':<8} {'Daily vol':>10}  {'$ risk/day':>11}  {'Target notional':>16}  {'Shares':>7}  {'Fits?'}")
print("  " + "-" * 65)
n_instruments = 10
alloc_per = CAPITAL / n_instruments
for sym, desc, price, mp, vol in EXAMPLES[:n_instruments]:
    daily_risk_target = alloc_per * daily_vol_target_portfolio
    target_notional   = daily_risk_target / vol
    shares = max(1, round(target_notional / price))
    actual_notional = shares * price
    margin_req = actual_notional * mp
    fits = "OK" if margin_req <= alloc_per else "OVER MARGIN"
    print(f"  {sym:<8} {vol*100:>9.1f}%  {daily_risk_target:>11,.0f}  {target_notional:>16,.0f}  {shares:>7}  {fits}")

mt5.shutdown()
