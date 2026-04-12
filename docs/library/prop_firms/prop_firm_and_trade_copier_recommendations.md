# Prop Firm & Trade Copier Recommendations

**Last updated:** 2026-04-12
**Strategy:** Systematic daily/monthly rebalancing across ES, NQ, GC, RTY, TLT, YM
**Execution:** Python pipeline sends signals via webhook/API to trade copier, which fans out to all prop firm accounts
**Position style:** Daily rebalance. For firms requiring end-of-day close, we close and re-enter next session.

---

## Chosen Prop Firms

### 1. My Funded Futures (MFFU)

- **Fee:** $127/mo (50K)
- **Profit target:** $3,000 (6%) | **Drawdown:** $3,000 (6%) -- 1:1 ratio, best available
- **Consistency rule:** 50% (best day < 50% of total profit)
- **Overnight holds:** Yes, all plans
- **Algo/copier:** Semi-automated allowed, trade copier allowed
- **Payout:** Weekly, 100% of first $10K then 90%
- **Trustpilot:** 4.9/5 (11,000+ reviews)
- **Platform:** Tradovate, Rithmic

Best overall for our strategy. Overnight holds from day one, generous drawdown, highest trust rating. Semi-automated is fine -- our pipeline runs daily with human oversight.

### 2. Lucid Trading

- **Fee:** $155 one-time (50K LucidPro)
- **Profit target:** $3,000 (6%) | **Drawdown:** $2,500 (5%)
- **Consistency rule:** 20% (lenient)
- **Overnight holds:** Only on LucidLive (after 5 payouts). Must close daily during sim phase.
- **Algo/copier:** Fully allowed, unlimited accounts
- **Payout:** Daily on LucidLive, 15-min processing
- **Platform:** Tradovate

Best long-term option. One-time fee is cheap. Full algo support. Path: day-trade (close/re-enter daily) through 5 payouts on LucidPro, then deploy full swing strategy on LucidLive.

### 3. Apex Trader Funding

- **Fee:** ~$13-20 with 90% discount codes (frequent promos)
- **Profit target:** $3,000 (6%) | **Drawdown:** $2,500 (5%)
- **Consistency rule:** 50% (PA phase only)
- **Overnight holds:** No (must close by 4:59 PM ET). We close and re-enter next day.
- **Algo/copier:** Fully allowed, up to 20 accounts
- **Payout:** Bi-weekly, payout ladder system
- **Platform:** Tradovate, Rithmic
- **Trustpilot:** 4.4/5 (18,000+ reviews)

Cheapest entry with discount codes. Best scaling (20 accounts). Daily close/re-enter is manageable for our strategy since we rebalance daily anyway. Controversial payout history but 4.0 rules addressed most issues.

### 4. Tradefundrr

- **Fee:** $149 one-time
- **Profit target:** $2,500 (5%) | **Drawdown:** $3,000 (6%) -- drawdown exceeds target
- **Consistency rule:** None
- **Overnight holds:** Verify before committing
- **Algo/copier:** Verify before committing
- **Payout:** Bi-weekly
- **Platform:** Rithmic

Best math -- only firm where drawdown > target. No consistency rule. One-time fee. Newer firm, verify overnight and algo policies directly before funding.

### 5. Topstep

- **Fee:** $49/mo (cheapest monthly)
- **Profit target:** $3,000 (6%) | **Drawdown:** $2,000 (4%)
- **Consistency rule:** 50%
- **Overnight holds:** No (must close by 3:10 PM CT). We close and re-enter next day.
- **Algo/copier:** Allowed with caveats, built-in copier for 5 accounts
- **Payout:** Daily (after 30 winning days)
- **Platform:** Tradovate
- **Trustpilot:** 3.4/5 (13,690 reviews)

Cheapest backup. Longest track record ($1.1B+ paid). Daily payouts once unlocked. Lower trust rating and tighter drawdown.

### Firms to avoid

- **OneUp** -- 80% consistency rule (impossible for systematic strategies)
- **The Futures Desk** -- Only $1,000 drawdown (2%) on 50K
- **TopOneFutures** -- $389/mo, overpriced
- **FundingTicks** -- Winding down operations
- **Emerge Profit / Funded Futures Family** -- Prohibit algo trading entirely

---

## Chosen Trade Copiers

### 1. PickMyTrade (Primary)

- **URL:** https://pickmytrade.io
- **Price:** $50/mo flat, unlimited accounts
- **Platforms:** Rithmic, Tradovate, IB, TradeStation, ProjectX
- **Webhook:** Yes, HTTP POST from Python
- **Why:** Best value for 5-10+ accounts. Broadest platform support covers all our chosen prop firms. Cloud-based, no VPS needed.

### 2. TradersPost (Alternative)

- **URL:** https://traderspost.io
- **Price:** ~$339/mo for 10 accounts
- **Platforms:** Tradovate, IB, TradeStation (no native Rithmic)
- **Webhook:** Best API -- supports position-level signals ("set position to +2 ES"), partial exits, sentiment flat for rebalancing
- **Why:** Best for position-aware rebalancing. Premium price but superior API design for our exact workflow.

### 3. Tradesyncer (Budget alternative)

- **URL:** https://tradesyncer.com
- **Price:** $49/mo for 10 accounts
- **Platforms:** NinjaTrader, Tradovate, Rithmic, ProjectX
- **Webhook:** Leader-follower model. We set up a demo account as leader, execute via Python, followers mirror.
- **Why:** Good value, sub-100ms execution. Leader model works well with our setup.

---

## Recommended Architecture

```
Python pipeline (tws_live_forecast.py)
    ↓ webhook HTTP POST
PickMyTrade (or TradersPost)
    ↓ fans out to all accounts
┌─ MFFU account (Tradovate)
├─ Lucid account (Tradovate)
├─ Apex account 1 (Tradovate)
├─ Apex account 2 (Tradovate)
└─ Topstep account (Tradovate)
```

For firms requiring daily close: the pipeline sends a "flatten all" signal at 3:00 PM CT, then re-enters positions the next morning at market open.

---

## Simulation Results

Monte Carlo backtest across all 5 firms (20 runs, Sharpe 1.5, 10% annual vol):

| Firm | Eval Pass Rate | Avg Payouts | Avg Paid Out | Eval Breach Rate |
|------|---------------|-------------|-------------|-----------------|
| **Tradefundrr** | **85%** | **9.9** | **$6,352** | 15% |
| **MFFU** | **85%** | 2.4 | $2,966 | 15% |
| Topstep | 70% | 1.6 | $1,985 | 30% |
| Lucid | 65% | 1.1 | $622 | 35% |
| Apex | 5% | 0.3 | $510 | 95% |

Simulators: `prop_firms/{apex,lucid,mffu,topstep,tradefundrr}/`

Apex's 95% breach rate is due to the 30-day evaluation expiry -- a 10% vol strategy can't reliably hit $3K in 30 days. Tradefundrr wins because drawdown ($3K) exceeds target ($2.5K), giving the most room to operate.
