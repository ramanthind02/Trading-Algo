# Prop Firm & Trade Copier Recommendations

**Last updated:** 2026-04-12
**Strategy:** Systematic daily/monthly rebalancing across ES, NQ, GC, RTY, TLT, YM
**Execution:** Python pipeline sends signals via webhook/API to trade copier, which fans out to all prop firm accounts
**Position style:** Daily rebalance. For firms requiring end-of-day close, we close and re-enter next session.

**Note:** Pricing and rules change frequently. Always verify on the firm's website before purchasing. Numbers below were fact-checked April 2026.

---

## Chosen Prop Firms

### 1. My Funded Futures (MFFU)

- **Fee:** $227 one-time (50K Pro). No activation fee. Also: Rapid 50K at $125.60 one-time.
- **Profit target:** $3,000 | **Drawdown:** $2,000 (EOD). Eval: 3 contracts. Funded: 5 contracts.
- **Consistency rule:** 50% in eval. None in funded.
- **Overnight holds:** Yes, all plans
- **Algo/copier:** Semi-automated allowed, trade copier allowed
- **Payout:** Every 14 days. Pro: 80/20 split, $1K min, $2.1K buffer. Rapid: 90/10, $500 min, 1-day payout.
- **Trustpilot:** 4.9/5 (11,000+ reviews)
- **Platform:** Tradovate, Rithmic

Best overall for our strategy. Overnight holds from day one, highest trust rating. Pro recommended: no consistency in funded phase, EOD drawdown in funded, 5 contracts. Rapid is cheaper ($87-126) with 90/10 split and 1-day payouts, but funded drawdown switches to RealTime (stricter).

### 2. Lucid Trading

- **Fee:** ~$130-160 one-time (50K LucidPro)
- **Profit target:** $3,000 (6%) | **Drawdown:** $2,500 (5%)
- **Consistency rule:** 20% (lenient)
- **Overnight holds:** Only on LucidLive (after 5 payouts). Must close daily during sim phase.
- **Algo/copier:** Fully allowed, unlimited accounts
- **Payout:** Daily on LucidLive, 15-min processing. 100% of first $10K then 90/10.
- **Platform:** Tradovate

Best long-term option. One-time fee is cheap. Full algo support. Path: day-trade (close/re-enter daily) through 5 payouts on LucidPro, then deploy full swing strategy on LucidLive.

### 3. Apex Trader Funding

- **Fee:** ~$13-20 with 90% discount codes (frequent promos). One-time, 30-day eval.
- **Profit target:** $1,500 (25K) to $3,000 (50K) | **Drawdown:** $1,000-$2,500
- **Consistency rule:** 50% (PA phase only)
- **Overnight holds:** No (must close by 4:59 PM ET). We close and re-enter next day.
- **Algo/copier:** Fully allowed, up to 20 accounts
- **Payout:** Bi-weekly, payout ladder system. 100% of first $25K then 90/10.
- **Platform:** Tradovate, Rithmic
- **Trustpilot:** 4.4/5 (18,000+ reviews)

Cheapest entry with discount codes. Best scaling (20 accounts). Daily close/re-enter is manageable. Controversial payout history but 4.0 rules addressed most issues. 30-day eval expiry is tight for low-vol strategies.

### 4. TradeDay

- **Fee:** $175/mo (50K EOD) or ~$122 with promo. No activation fee. Resets: $104.
- **Profit target:** $3,000 (6%) | **Drawdown:** $2,000 (4%)
- **Consistency rule:** 30% in eval only (doesn't breach, just raises target). None in funded.
- **Overnight holds:** No (must close by 5:00 PM ET). We close and re-enter next day.
- **Algo/copier:** Allowed (NinjaTrader automation, TradeSyncer supported). No copying between TradeDay accounts.
- **Payout:** Weekly, 100% on first $10K cumulative, then 80%/90%/92.5%/95%
- **Platform:** Tradovate
- **Trustpilot:** 4.6/5 (1,347 reviews)

Proven firm (Chicago, since 2020). Best profit split tiers in the industry. No daily loss limit. No consistency rule once funded. Day-one payouts with next-business-day processing.

### 5. Topstep

- **Fee:** $49/mo (50K, Standard Path + $149 activation) or $109/mo (No Activation Fee Path)
- **Profit target:** $3,000 (6%) | **Drawdown:** $2,000 (4%)
- **Consistency rule:** 50%
- **Overnight holds:** No (must close by 3:10 PM CT). We close and re-enter next day.
- **Algo/copier:** Allowed with caveats, built-in TopstepX copier for 5 accounts
- **Payout:** 5 winning days ($150+ each) between payouts. After 30 benchmark trading days in Live Funded, unlocks 100% withdrawal + daily payouts. Profit split: 90/10.
- **Platform:** Tradovate
- **Trustpilot:** 3.4/5 (13,690 reviews)

Cheapest monthly. Longest track record ($1.1B+ paid). Tighter drawdown and lower trust rating.

### Firms to avoid

- **OneUp** -- 80% consistency rule (impossible for systematic strategies)
- **The Futures Desk** -- Only $1,000 drawdown (2%) on 50K
- **TopOneFutures** -- $389/mo, overpriced
- **FundingTicks** -- Winding down operations
- **Emerge Profit / Funded Futures Family** -- Prohibit algo trading entirely
- **Tradefundrr** -- Only 60 Trustpilot reviews, reports of retroactive account bans

---

## Chosen Trade Copier: Tradecopia (Primary)

- **URL:** https://tradecopia.com
- **Price:** $40/mo (Basic) or $60/mo (Pro with risk management). Flat fee, unlimited accounts.
- **Platforms:** Tradovate, Rithmic, ProjectX, NinjaTrader
- **Model:** Leader-follower. Tradovate demo acts as leader, all prop firm accounts are followers.
- **Local/Cloud:** Local desktop app (Tradovate only, no IB support). Runs on Raman's machine 24/7.
- **Trustpilot:** 4.7/5 (195 reviews)
- **Why chosen:** Cheapest for unlimited accounts ($40-60 flat vs $50/login for PickMyTrade). Local execution avoids potential prop firm bans for cloud-based copiers. Supports all major platforms.
- **Limitation:** No webhook/API. Tradovate only (no IB). Requires a Tradovate demo/sim as leader that our Python script trades on via Tradovate REST API.

### Alternatives (if needed later)

- **TradersPost** ($49-339/mo) -- best webhook API, no leader needed. Good for pure automation but expensive at scale and cloud-based.
- **PickMyTrade** ($50/mo per Tradovate login) -- webhook from Python, but costs add up with multiple prop firms.
- **Tradesyncer** ($49-99/mo) -- leader-follower, cloud-based, supports Rithmic + Tradovate.

---

## Recommended Architecture

```
Python pipeline (tws_live_forecast.py)
    ↓ places orders via Tradovate REST API
Tradovate demo/sim account (leader)
    ↓ Tradecopia watches leader
Tradecopia desktop app (runs 24/7 on Raman's machine)
    ├─→ Apex account (Tradovate follower)
    ├─→ MFFU account (Tradovate follower)
    ├─→ Lucid account (Tradovate follower)
    ├─→ TradeDay account (Tradovate follower)
    └─→ Topstep account (Tradovate follower)
```

For firms requiring daily close: the pipeline flattens all positions at 3:00 PM CT on the Tradovate leader, Tradecopia mirrors the flatten to all followers, then re-enters positions the next morning.

**Note:** IB TWS is still used for data fetching (OHLC candles). Only order execution moves to the Tradovate REST API for Tradecopia compatibility.

**Cost:** $40-60/mo total for unlimited prop firm accounts.

---

## Simulation Results

Monte Carlo backtest across all 5 firms (20 runs, Sharpe 1.5, 10% annual vol):

| Firm | Eval Pass Rate | Avg Payouts | Avg Paid Out | Eval Breach Rate |
|------|---------------|-------------|-------------|-----------------|
| **MFFU** | **85%** | 2.4 | **$2,966** | 15% |
| **TradeDay** | **70%** | **2.8** | $2,473 | 30% |
| Topstep | 70% | 1.6 | $1,963 | 30% |
| Lucid | 65% | 1.1 | $728 | 35% |
| Apex | 5% | 0.3 | $510 | 95% |

Simulators: `prop_firms/{apex,lucid,mffu,topstep,tradeday}/`

MFFU leads with 85% pass rate thanks to the generous 1:1 target-to-drawdown ratio ($3K/$3K). TradeDay has solid payouts despite tighter drawdown. Apex's 95% breach rate is due to the 30-day evaluation expiry -- a 10% vol strategy can't reliably hit $3K in 30 days.
