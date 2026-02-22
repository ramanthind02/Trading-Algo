# RSI Left-Tail Variants Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add four standalone RSI-derived bias nodes optimized for short-term left-tail mean-reversion behavior, with deterministic unit coverage and node registry/docs updates.

**Architecture:** Reuse the existing incremental RSI state pattern (Cython-backed kernels via `utils.fast_nodes`) as the functional core for all new nodes. Each node adds one focused transformation (lag, tail pressure, streak, rebound velocity) with explicit warmup defaults and standardized metadata. Surface nodes through canonical wrappers plus taxonomy mappings so dynamic creation and legacy imports both work.

**Tech Stack:** Python, NumPy, existing `BiasNode` framework, `utils.fast_nodes`, pytest.

---

### Task 1: Add shared RSI stream helper mixin (optional local helper pattern inside each file)

**Files:**
- Modify: `nodes/mean_reversion/rsi/lagged_rsi.py`
- Modify: `nodes/mean_reversion/rsi/rsi_left_tail_pressure.py`
- Modify: `nodes/mean_reversion/rsi/rsi_left_tail_streak.py`
- Modify: `nodes/mean_reversion/rsi/rsi_rebound_velocity.py`
- Test: `tests/unit-tests/nodes/test_rsi_left_tail_variants.py`

**Step 1: Write failing test**

```python
def test_rsi_left_tail_nodes_emit_one_value_per_candle(sample_candles):
    outputs = [node.add_candle(c)[0] for c in sample_candles]
    assert len(outputs) == len(sample_candles)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit-tests/nodes/test_rsi_left_tail_variants.py::test_rsi_left_tail_nodes_emit_one_value_per_candle -v`
Expected: FAIL (files/classes not found)

**Step 3: Write minimal implementation**

```python
# per node file: class <NodeName>(BiasNode):
# - init RSI state (close buffer, prev_close, upsum/dnsum)
# - _compute_base_rsi(candle) helper returning current RSI
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit-tests/nodes/test_rsi_left_tail_variants.py::test_rsi_left_tail_nodes_emit_one_value_per_candle -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nodes/mean_reversion/rsi/*.py tests/unit-tests/nodes/test_rsi_left_tail_variants.py
git commit -m "feat: add base left-tail RSI node scaffolds"
```

### Task 2: Implement LaggedRSI node

**Files:**
- Add: `nodes/mean_reversion/rsi/lagged_rsi.py`
- Add: `nodes/lagged_rsi.py`
- Modify: `nodes/_taxonomy.py`
- Test: `tests/unit-tests/nodes/test_rsi_left_tail_variants.py`

**Step 1: Write failing tests**

```python
def test_lagged_rsi_warmup_and_range(sample_candles):
    node = LaggedRSI(ticker=Ticker.ES, tf=TimeFrame.D, rsiPeriod=4, lag=2)
    values = [node.add_candle(c)[0] for c in sample_candles]
    assert all(0.0 <= v <= 100.0 for v in values[6:])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit-tests/nodes/test_rsi_left_tail_variants.py::test_lagged_rsi_warmup_and_range -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
lagged_value = rsi_history[-1 - lag] if len(rsi_history) > lag else 50.0
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit-tests/nodes/test_rsi_left_tail_variants.py::test_lagged_rsi_warmup_and_range -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nodes/mean_reversion/rsi/lagged_rsi.py nodes/lagged_rsi.py nodes/_taxonomy.py tests/unit-tests/nodes/test_rsi_left_tail_variants.py
git commit -m "feat: add lagged RSI node for short-horizon mean reversion"
```

### Task 3: Implement RSILeftTailPressure and RSILeftTailStreak

**Files:**
- Add: `nodes/mean_reversion/rsi/rsi_left_tail_pressure.py`
- Add: `nodes/rsi_left_tail_pressure.py`
- Add: `nodes/mean_reversion/rsi/rsi_left_tail_streak.py`
- Add: `nodes/rsi_left_tail_streak.py`
- Modify: `nodes/_taxonomy.py`
- Test: `tests/unit-tests/nodes/test_rsi_left_tail_variants.py`

**Step 1: Write failing tests**

```python
def test_pressure_increases_during_selloff(downtrend_candles):
    node = RSILeftTailPressure(...)
    values = [node.add_candle(c)[0] for c in downtrend_candles]
    assert values[-1] >= values[-5]

def test_streak_ratio_bounds(sample_candles):
    node = RSILeftTailStreak(...)
    values = [node.add_candle(c)[0] for c in sample_candles]
    assert all(0.0 <= v <= 1.0 for v in values)
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/unit-tests/nodes/test_rsi_left_tail_variants.py -k "pressure or streak" -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
pressure = np.mean(np.maximum(0.0, tail_level - np.asarray(recent_rsi)))
streak = min(max_streak, streak + 1) if rsi <= tail_level else 0
ratio = streak / max_streak
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/unit-tests/nodes/test_rsi_left_tail_variants.py -k "pressure or streak" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nodes/mean_reversion/rsi/rsi_left_tail_pressure.py nodes/rsi_left_tail_pressure.py nodes/mean_reversion/rsi/rsi_left_tail_streak.py nodes/rsi_left_tail_streak.py nodes/_taxonomy.py tests/unit-tests/nodes/test_rsi_left_tail_variants.py
git commit -m "feat: add RSI left-tail pressure and streak variants"
```

### Task 4: Implement RSIReboundVelocity

**Files:**
- Add: `nodes/mean_reversion/rsi/rsi_rebound_velocity.py`
- Add: `nodes/rsi_rebound_velocity.py`
- Modify: `nodes/_taxonomy.py`
- Test: `tests/unit-tests/nodes/test_rsi_left_tail_variants.py`

**Step 1: Write failing test**

```python
def test_rebound_velocity_rises_after_bounce(down_then_up_candles):
    node = RSIReboundVelocity(...)
    values = [node.add_candle(c)[0] for c in down_then_up_candles]
    assert values[-1] > values[-6]
```

**Step 2: Run test to verify fail**

Run: `pytest tests/unit-tests/nodes/test_rsi_left_tail_variants.py::test_rebound_velocity_rises_after_bounce -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
recent_min = float(np.min(np.asarray(rsi_history)))
velocity = max(0.0, current_rsi - recent_min)
```

**Step 4: Run test to verify pass**

Run: `pytest tests/unit-tests/nodes/test_rsi_left_tail_variants.py::test_rebound_velocity_rises_after_bounce -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nodes/mean_reversion/rsi/rsi_rebound_velocity.py nodes/rsi_rebound_velocity.py nodes/_taxonomy.py tests/unit-tests/nodes/test_rsi_left_tail_variants.py
git commit -m "feat: add RSI rebound velocity node"
```

### Task 5: Docs + full targeted verification

**Files:**
- Modify: `docs/api/nodes.md`
- Test: `tests/unit-tests/nodes/test_rsi_left_tail_variants.py`

**Step 1: Write failing doc expectation check (manual checklist)**

```python
# N/A (docs task): ensure new nodes are documented in public entrypoints
```

**Step 2: Run verification commands**

Run: `pytest tests/unit-tests/nodes/test_rsi_left_tail_variants.py -v`
Expected: PASS

Run: `pytest tests/unit-tests/nodes/test_rsi_left_tail_variants.py::test_rsi_left_tail_nodes_are_deterministic -v`
Expected: PASS

Run: `pytest tests/unit-tests/nodes/test_rsi_left_tail_variants.py::test_rsi_left_tail_behavior_on_down_then_rebound_path -v`
Expected: PASS

**Step 3: Write minimal docs update**

```markdown
### `nodes.lagged_rsi.LaggedRSI`
### `nodes.rsi_left_tail_pressure.RSILeftTailPressure`
### `nodes.rsi_left_tail_streak.RSILeftTailStreak`
### `nodes.rsi_rebound_velocity.RSIReboundVelocity`
```

**Step 4: Re-run verification**

Run: `pytest tests/unit-tests/nodes/test_rsi_left_tail_variants.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/api/nodes.md tests/unit-tests/nodes/test_rsi_left_tail_variants.py
git commit -m "docs: document new RSI left-tail variant nodes"
```
