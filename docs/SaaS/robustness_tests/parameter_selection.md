# Parameter Selection Methods

## 1. Purpose

This document specifies the parameter selection methods supported in QuantFoundry. The scope is deliberately narrow for the MVP: two methods, both simple and interpretable. Ensemble-based selection (Carver's portfolio optimisation approach) is a known extension and is deferred to a later milestone.

Related documents:
- `docs/SaaS/robustness_tests/in_sample.md` — search-bias tests; run before selecting
- `docs/SaaS/robustness_tests/parameter_sensitivity.md` — perturbation test and plots; use to inspect the chosen combination after selection

---

## 2. Method 1 — Manual Selection

The researcher explicitly picks a parameter combination from the sweep results. No algorithm is involved.

**When to use it:**
- The researcher has a prior economic view on which parameter value is appropriate (e.g. a lookback period that matches the expected market cycle length)
- The researcher wants to lock in a specific combination and use the robustness tests purely as confirmation
- A single combination has been used in live trading and needs to be formally registered

**Platform behaviour:** The researcher clicks any row in the sweep results table to select it. The robustness test suite runs against that combination. The selection method is stored as `MANUAL` in the strategy metadata.

---

## 3. Method 2 — Best by Metric

Select the combination with the highest configured selection metric (typically t-stat or Sharpe).

**When it is defensible:**
- The IS period is reasonably long
- The parameter space explored was narrow and motivated by economic logic before seeing the data
- The perturbation test shows a small optimism bias (peak ≈ median), confirming the best combination sits on a stable region rather than an isolated spike

**When it is not sufficient:** If the perturbation test reveals a large peak-to-median gap, the best combination is a fragile noise peak. In this case the researcher should either narrow the search space, obtain more data, or wait for the ensemble selection method (future milestone).

**Implementation:** Rank all combinations by the configured selection metric descending. Select rank 1. This is the default selection if the researcher does not manually choose a combination.

---

## 4. Deferred — Ensemble Selection

Forming a weighted ensemble of multiple parameter variants (Carver's approach) is the most statistically robust selection method and a natural fit for this system's existing ensemble architecture. It is deferred to a later milestone due to added complexity in both the UX and signal pipeline.

When implemented, the ensemble approach will sit alongside Methods 1 and 2 as a third option. The parameter sensitivity and perturbation infrastructure built for the MVP will feed directly into it.

---

## 5. Selection Method as Strategy Metadata

Whichever method is used, the selection method and the chosen combination are stored as immutable metadata on the strategy object:

```python
@dataclass(frozen=True)
class ParameterSelection:
    method: Literal["MANUAL", "BEST_METRIC"]
    selection_metric: str              # e.g. "nw_tstat", "sharpe"
    chosen_combination: dict[str, Any]
    chosen_metric_value: float
    perturbation_test_passed: bool     # from parameter_sensitivity.md
    n_combinations_evaluated: int
```

This is required to reproduce the selection decision at any future point and to audit it against later validation and portfolio-addition results.
