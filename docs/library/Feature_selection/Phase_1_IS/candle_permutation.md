# Candle Permutation

> [!note] Used in [[permutation_testing]] Stage 2 (pipeline permutation). Destroys predictable temporal patterns in price history while preserving statistical structure needed for valid null comparisons.

Foundation: Timothy Masters, *Core Algorithms* (bar permutation chapter).

---

## Design Invariants (Both Variants)

| Invariant | Description |
|-----------|-------------|
| **Trend preservation** | First open and last close of the reconstructed series equal the original. Global market direction is unchanged. |
| **Relative quantities only** | No absolute-price shuffling. Only intra-bar triplets and inter-bar gaps are permuted and recombined. |
| **No gap mixing** | Weekday/overnight gaps are never placed where weekend-sized gaps belong, and vice versa. Prevents unnatural price artifacts. |

> [!warning] Known limitation — **volatility clumping**: shuffling day (or bar) order scatters volatility across time. Periods of unusually high or low ranges are redistributed randomly rather than appearing in natural clusters. This is a known, documented effect with no practical fix (Masters, p. 39).

---

## Version A: Daily (and Higher) Bar Permutation

**Applicability:** D, W, M bars — any timeframe where one bar = one full trading day or longer (24/5 schedule).

### Decomposition

For each bar, extract the **intra-bar triplet** relative to Open:
```
(H − O,  L − O,  C − O)
```

For each bar transition, extract the **inter-bar gap**:
```
gap_i = Open_i − Close_{i−1}
```

### Two Gap Vectors (24/5)

| Vector | Transitions | Typical size |
|--------|-------------|--------------|
| **A — Weekday** | Mon→Tue, Tue→Wed, Wed→Thu, Thu→Fri | Small |
| **B — Weekend** | Fri close → Mon open | Can be large |

Shuffle A and B **independently**. Never mix values between the two pools.

### Reconstruction

1. **Basis bar**: first bar of original series is fixed (anchor, unchanged).
2. For each subsequent bar: choose gap from the appropriate pool (weekday A or weekend B depending on position in the week), then `Open = previous Close + gap`.
3. Assign the next shuffled triplet: `High = Open + (H−O)`, `Low = Open + (L−O)`, `Close = Open + (C−O)`.
4. Triplets are drawn from a **single global shuffled pool** — preserves the distribution of bar shapes and daily volatility.

**Result:** First open = original first open. Last close = original last close.

---

## Version B: Intraday Bar Permutation

**Applicability:** Sub-daily bars (e.g. M15, H1) with known intraday session breaks.

**Core idea:** Treat days as modular blocks. Preserve each day's net change and internal structure while randomising which day appears in which position, and which gap connects consecutive days.

### Phase 1 — Decomposition

1. Identify the **basis day** (first day) — left unpermuted throughout.
2. Group all bars by calendar day (exchange timezone).
3. Compute intra-bar triplets `(H−O, L−O, C−O)` for every bar; within-day inter-bar gaps `Open − prev Close`; overnight between-day gaps.
4. Build two dedicated gap vectors (same principle as Version A):
   - **Vector A — Weekday overnight**: Mon→Tue, Tue→Wed, Wed→Thu, Thu→Fri.
   - **Vector B — Weekend gap**: Fri close → Mon open.
5. **Anchor days**: for each non-basis day, subtract that day's Open from all internal prices (day "opens at zero"). Enables within-day shuffling without changing the day's net move.
6. Create an **index vector** `(0, 1, 2, ...)` — one entry per non-basis day.

### Phase 2 — Permutation

Three independent shuffles:
- **Within each day** (except basis day): shuffle bars (triplets + within-day gaps) inside that day only. Preserves day net change; destroys intraday sequence.
- **Overnight gaps**: shuffle Vector A and Vector B independently.
- **Day order**: shuffle the index vector — randomises which day appears in each chronological slot.

### Phase 3 — Reconstruction

1. Output basis day unchanged (anchors the first open).
2. For each subsequent position in new chronology:
   - Day to append: given by the shuffled index vector.
   - Gap to apply: next value from **weekday Vector A** (positions 2–5 of each reconstructed week) or **weekend Vector B** (first position after each reconstructed weekend).
   - `New day Open = previous day Close + chosen gap`.
   - Append that day's shuffled (zero-anchored) bars, adding the computed Open to all prices.
3. Last close of reconstructed series equals original last close (trend preservation).

### Alternative: Weekly Block (Strategy A)

Treat the full Mon–Fri week as one block. Permute bars within the week using triplets and within-week gaps. Use only weekend gaps between blocks. Avoids weekday/weekend gap mixing by construction — only weekend gaps ever appear between blocks.

---

## Gap Taxonomy for ES M15

For instruments like ES (E-mini S&P) on M15 (24/5, CME schedule), from analysis of 2009–2023 (315,122 bars, 6,317 gaps):

| Gap type | When | Size |
|----------|------|------|
| **Maintenance** (dedicated vector) | Last bar at 17:xx — CME daily break 17:15→18:00 | Small, fixed |
| **Weekend** (dedicated vector) | Last bar on Friday, next bar on Monday | Large |
| **Regular** (one pool) | All other: weekday overnight, same-day 16:xx→16:30, early close (11–13xx), long weekends, one-offs | Variable |

Two dedicated vectors (maintenance, weekend) are the default. A third vector (early-close) can be added if holiday-sized gaps must never land on normal weekdays.

**Gap type classification:**
- Weekend: last bar on Friday, next bar on Monday.
- Maintenance: last bar at 17:xx (same calendar day).
- Regular: everything else.

---

## Datetime Handling

Datetimes are **not** shuffled with OHLC. After reconstructing OHLC from triplets and gaps, assign new timestamps so the order of bars matches the order of timestamps (first reconstructed bar gets the first timestamp, etc.). This prevents spurious correlations for time-based features.

---

## Limitations Summary

| Effect | Status |
|--------|--------|
| Volatility clumping (intraday) | Known, documented, no fix (Masters p. 39) |
| Basis bar/day sensitivity | Minimal but non-zero |
| Holiday gaps folded into regular pool | Acceptable simplification; use third vector if required |
