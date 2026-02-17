# RSI Signal Bias Node Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update `RSISignal` to cross-based long/short/long-short logic with fixed-exit support while keeping RSI calculation and cache behavior intact.

**Architecture:** Keep the existing RSI computation in `nodes/rsi_signal.py`, replace the signal generation with a small state machine that uses RSI threshold crosses and a bar-count exit. Store position state and bar counts on the node. Update tests to validate deterministic cross behavior via patched RSI outputs.

**Tech Stack:** Python, NumPy, `unittest` + `pytest` runner.

---

### Task 1: Create Kanban Task Contract

**Files:**
- Create: `docs/kanban/to-do/bias_nodes/T001_rsi_signal_rule_update.md`
- (If missing) Create: `docs/kanban/in-progress/`

**Step 1: Draft the task spec from template**

Use `docs/kanban/templates/feature.md` as the base. Fill with:
- Context: `docs/library/bias_nodes/to-do/rsi_signal.md`, `docs/library/bias_nodes/base_bias_node_specs.md`, `nodes/rsi_signal.py`, `docs/api/nodes.md`
- Scope: limit to `nodes/rsi_signal.py`, `tests/test_new_bias_nodes.py`, `docs/api/nodes.md`
- Interfaces: updated `RSISignal.__init__` signature and behavior
- Acceptance tests:
  1. `pytest tests/test_new_bias_nodes.py::TestRSISignal::test_long_mode_fixed_exit -v`
  2. `pytest tests/test_new_bias_nodes.py::TestRSISignal::test_short_mode_fixed_exit -v`
  3. `pytest tests/test_new_bias_nodes.py::TestRSISignal::test_long_short_mode_fixed_exit -v`
- Definition of done: add tests, update docs/api, run the tests above

**Step 2: Move task to in-progress before coding**

If `docs/kanban/in-progress/` does not exist, create it. Then move the task file into `docs/kanban/in-progress/` before code changes.

**Step 3: Commit**

```bash
git add docs/kanban/to-do/bias_nodes/T001_rsi_signal_rule_update.md docs/kanban/in-progress/
git commit -m "Add kanban task for RSI signal rule update"
```

---

### Task 2: Add Deterministic Unit Tests for RSISignal

**Files:**
- Modify: `tests/test_new_bias_nodes.py`

**Step 1: Write failing tests**

Add three tests to `TestRSISignal` using `unittest.mock.patch` to control RSI values:

```python
from unittest.mock import patch

class TestRSISignal(unittest.TestCase):
    def test_long_mode_fixed_exit(self):
        with patch("nodes.rsi_signal.compute_rsi_initial") as mock_init, \
             patch("nodes.rsi_signal.update_rsi") as mock_update:
            mock_init.return_value = (1.0, 1.0)  # rsi=50
            rsi_values = iter([25.0, 26.0, 27.0, 80.0])

            def update_side_effect(prev_close, curr_close, upsum, dnsum, rsi_period):
                rsi = next(rsi_values)
                return upsum, dnsum, rsi

            mock_update.side_effect = update_side_effect

            node = RSISignal(
                Ticker.ES,
                TimeFrame.D,
                rsi_period=2,
                oversold=30.0,
                overbought=70.0,
                strategy_mode="long",
                exit_policy="threshold_or_bars",
                exit_bars=2,
            )
            candles = generate_candles(6, volatility=0.0)
            signals = [node.add_candle(c)[0] for c in candles]

            self.assertEqual(signals, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def test_short_mode_fixed_exit(self):
        with patch("nodes.rsi_signal.compute_rsi_initial") as mock_init, \
             patch("nodes.rsi_signal.update_rsi") as mock_update:
            mock_init.return_value = (1.0, 1.0)
            rsi_values = iter([80.0, 79.0, 78.0, 25.0])

            def update_side_effect(prev_close, curr_close, upsum, dnsum, rsi_period):
                rsi = next(rsi_values)
                return upsum, dnsum, rsi

            mock_update.side_effect = update_side_effect

            node = RSISignal(
                Ticker.ES,
                TimeFrame.D,
                rsi_period=2,
                oversold=30.0,
                overbought=70.0,
                strategy_mode="short",
                exit_policy="threshold_or_bars",
                exit_bars=2,
            )
            candles = generate_candles(6, volatility=0.0)
            signals = [node.add_candle(c)[0] for c in candles]

            self.assertEqual(signals, [0.0, 0.0, -1.0, 0.0, 0.0, 0.0])

    def test_long_short_mode_fixed_exit(self):
        with patch("nodes.rsi_signal.compute_rsi_initial") as mock_init, \
             patch("nodes.rsi_signal.update_rsi") as mock_update:
            mock_init.return_value = (1.0, 1.0)
            rsi_values = iter([25.0, 26.0, 80.0, 81.0])

            def update_side_effect(prev_close, curr_close, upsum, dnsum, rsi_period):
                rsi = next(rsi_values)
                return upsum, dnsum, rsi

            mock_update.side_effect = update_side_effect

            node = RSISignal(
                Ticker.ES,
                TimeFrame.D,
                rsi_period=2,
                oversold=30.0,
                overbought=70.0,
                strategy_mode="long-short",
                exit_policy="threshold_or_bars",
                exit_bars=2,
            )
            candles = generate_candles(6, volatility=0.0)
            signals = [node.add_candle(c)[0] for c in candles]

            self.assertEqual(signals, [0.0, 0.0, 1.0, 0.0, -1.0, 0.0])
```

**Step 2: Run tests to verify failure**

Run:
`pytest tests/test_new_bias_nodes.py::TestRSISignal::test_long_mode_fixed_exit -v`

Expected: FAIL (unexpected keyword args / old behavior)

**Step 3: Commit failing tests**

```bash
git add tests/test_new_bias_nodes.py
git commit -m "Add RSISignal fixed-exit tests"
```

---

### Task 3: Implement RSISignal State Machine

**Files:**
- Modify: `nodes/rsi_signal.py`

**Step 1: Update constructor signature + validation**

Signature:
```python
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        strategy_mode: str = "long",
        exit_policy: str = "threshold_or_bars",
        exit_bars: int = 5,
    ):
```

Validation:
- `strategy_mode` in {"long", "short", "long-short"} (normalize hyphen to underscore if needed).
- `exit_policy` in {"threshold", "threshold_or_bars"}
- `exit_bars >= 1`

Metadata:
```python
self.module_name = "rsisignal"
self.output_features = ["signal"]
self.params = {
    "rsiPeriod": rsi_period,
    "oversold": oversold,
    "overbought": overbought,
    "strategyMode": self.strategy_mode,
    "exitPolicy": self.exit_policy,
    "exitBars": exit_bars,
}
```

State additions:
```python
self.position = 0
self.bars_in_position = 0
self.prev_rsi: Optional[float] = None
```

**Step 2: Replace signal logic**

Implement helper methods in `RSISignal`:

```python
def _crossed_below(self, prev_rsi: float, rsi: float) -> bool:
    return prev_rsi > self.oversold and rsi <= self.oversold

def _crossed_above(self, prev_rsi: float, rsi: float) -> bool:
    return prev_rsi < self.overbought and rsi >= self.overbought

def _apply_position_rules(self, rsi: float) -> int:
    if self.prev_rsi is None:
        return self.position

    crossed_below = self._crossed_below(self.prev_rsi, rsi)
    crossed_above = self._crossed_above(self.prev_rsi, rsi)

    next_position = self.position

    if self.strategy_mode == "long":
        if self.position == 0 and crossed_below:
            next_position = 1
        elif self.position == 1 and crossed_above:
            next_position = 0
    elif self.strategy_mode == "short":
        if self.position == 0 and crossed_above:
            next_position = -1
        elif self.position == -1 and crossed_below:
            next_position = 0
    else:  # long-short
        if crossed_below:
            next_position = 1
        elif crossed_above:
            next_position = -1

    return next_position
```

Fixed-exit handling in `_compute_candle` after `next_position` is computed:
- If `next_position` is same sign as `position`, increment `bars_in_position`.
- If `position` changes, reset `bars_in_position` to 1 (if entering) or 0 (if flat).
- If `exit_policy == "threshold_or_bars"` and `position != 0` and `bars_in_position >= exit_bars`, force `next_position = 0` and reset `bars_in_position = 0`.

Set `self.bias` based on final `position` (optional but recommended).

**Step 3: Run tests**

Run:
`pytest tests/test_new_bias_nodes.py::TestRSISignal::test_long_mode_fixed_exit -v`

Expected: PASS

**Step 4: Commit**

```bash
git add nodes/rsi_signal.py
git commit -m "Update RSISignal to cross-based strategy modes"
```

---

### Task 4: Update Nodes API Docs

**Files:**
- Modify: `docs/api/nodes.md`

**Step 1: Add RSISignal entry**

Add a new “Major bias-node entrypoint” section:

```markdown
### `nodes.rsi_signal.RSISignal`
Type: class

Signature:
```python
class RSISignal(BiasNode):
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        strategy_mode: str = "long",
        exit_policy: str = "threshold_or_bars",
        exit_bars: int = 5,
    )
```

Observable behavior:
- Outputs `-1`, `0`, or `1` based on RSI threshold **crosses**.
- `strategy_mode` controls long-only, short-only, or long-short behavior.
- `exit_policy="threshold_or_bars"` exits after `exit_bars` or threshold cross.
- Warmup outputs `0` until `rsi_period` candles.
```

**Step 2: Commit**

```bash
git add docs/api/nodes.md
git commit -m "Document RSISignal strategy modes and exits"
```

---

### Task 5: Verification & Kanban Result Block

**Files:**
- Modify: `docs/kanban/in-progress/T001_rsi_signal_rule_update.md` (move to complete or done folder as per repo convention)

**Step 1: Run full targeted tests**

Run:
`pytest tests/test_new_bias_nodes.py::TestRSISignal -v`

Expected: PASS

**Step 2: Update kanban Result block and move to complete/done**

Append the Result block with commit hash and test command. Move the task to `docs/kanban/complete/` if that is the repo’s “done” location.

**Step 3: Commit**

```bash
git add docs/kanban/in-progress/T001_rsi_signal_rule_update.md docs/kanban/complete/
git commit -m "Complete RSI signal rule update task"
```
