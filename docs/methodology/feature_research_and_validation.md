# Feature Research And Validation

**Purpose:** Define the research process for turning durable market effects into production-ready features without turning the pipeline into a data-mining exercise.

## Core Philosophy

The starting point is not arbitrary indicators. It is a small set of structural or risk-premia effects that have existed for years and have already been traded successfully in some form.

Typical sources of edge include:

- diversified buy-and-hold exposure,
- seasonality, calendar, and flow effects,
- mean reversion,
- breakouts,
- momentum.

The task is to build features that capture these effects in a way that is robust, interpretable, and stable enough to survive outside the research sample.

This is an Occam's razor workflow:

- start with known structural behavior,
- express it with simple feature families,
- test robustness directly,
- lock what works,
- avoid unnecessary adaptation.

The target production contract is simple: research can inspect both continuous and discrete bias nodes, but production only ships native discrete bias nodes that emit `-1/0/+1`.

## Data Split Policy

Research uses three distinct data sets:

1. **Training set**
   Used for feature design, parameter exploration, and robustness analysis.
2. **Validation set**
   Used as the first real out-of-sample check after the feature has been locked from the training stage.
3. **Test set**
   A pure final out-of-sample period. It is touched only when the researcher is fully satisfied with the feature and intends to make a final go/no-go decision.

The test set is not part of exploration. Once it is used, it stops being a clean final check.

## Research Sequence

The expected sequence is:

1. Start with a structural hypothesis.
2. Build a feature family that should express that hypothesis.
3. Explore the parameter landscape on the training set.
4. Run robustness checks on the training set.
5. Manually select a strong and stable parameter region or parameter combo.
6. Freeze the feature definition as a native signed-signal bias node.
7. Evaluate it on the validation set.
8. If validation behavior is acceptable, run the final test-set check.
9. If the test-set result is acceptable, move to production unchanged.

This framework is designed to filter serious ideas, not to search over a large feature universe.

## What Counts As A Good Feature Candidate

A feature is worth researching when:

- it is tied to a known structural, behavioral, or market-microstructure effect,
- the mechanism is understandable in plain language,
- the parameter family is broad enough to inspect stability,
- the feature can be frozen after research,
- the signal is simple enough to explain and monitor.

A feature is not worth researching when it exists only because it backtested well after broad experimentation.

## Training-Set Research

The training set is where discovery happens.

This includes:

- designing the feature family,
- surveying the parameter landscape,
- checking whether performance is isolated or exists across a region,
- running permutation tests,
- running parameter sensitivity checks,
- rejecting fragile features before they ever see validation.

The objective is not to find the single best point. The objective is to find a parameter region that appears structurally sensible and reasonably stable.

Manual selection is allowed here, but it must be disciplined. Looking at too many variants, reformulating the feature repeatedly, or continuing to refine the idea after weak evidence is just another form of overfitting.

## Robustness Testing

Robustness work should happen before validation and test.

Useful checks include:

- parameter-landscape inspection,
- parameter sensitivity around the chosen setting,
- permutation tests at the vector, pipeline, or feature level,
- stability across nearby parameter values,
- sanity checks on turnover, concentration, and path dependence.

No single test proves robustness. The goal is to build converging evidence that the feature is not just a narrow in-sample accident.

## Validation And Test

The validation set is the first real decision gate after the feature is locked.

The role of validation is to answer a practical question:

> Does the feature still behave acceptably once it leaves the training sample?

If the answer is no, the feature is rejected. Validation is not a license to keep iterating until something works.

The test set is reserved for the final confirmation. It is the closest thing to a clean pre-production out-of-sample result, so it should be used sparingly and only after the feature has already earned that review.

## Parameter Policy

Feature parameters are locked after research.

This is deliberate:

- the research process validates a specific feature definition,
- repeated retuning weakens the meaning of out-of-sample evaluation,
- stable parameter neighborhoods are more credible than a moving target,
- adaptive features often end up fitting recent noise.

For the feature layer, the standard is discover on the training set and freeze for future use.

## Selecting Parameters Within A Feature Family

The default goal is not to find the single best parameter point. It is to find a parameter choice that is strong, stable, and explainable.

When several nearby parameter settings work well, prefer a small coherent parameter region over a single isolated winner.

In practice:

- use a single parameter combo when the feature is naturally discrete or the surrounding region is weak,
- use a small fixed family ensemble when nearby settings express the same mechanism and behave similarly,
- avoid selecting a large collection of decent variants just to smooth the backtest.

For example, if `RSI(2)` through `RSI(5)` with the same binning logic all show similar mean-reversion behavior, a fixed mini-ensemble may be more robust than choosing only the top point. If `RSI(2)` and `RSI(5)` behave differently enough to imply different trade horizons or mechanisms, they should be treated as separate candidates rather than merged automatically.

The key test is whether the selected variants belong to one stable region of the landscape or whether they are distinct strategies hiding inside one feature family.

## Feature Enhancements And Filters

Adding a filter to an existing feature changes the feature definition. A trend filter, volatility filter, sector filter, or time filter should therefore be treated as a new variant of the base feature.

That does not always mean restarting from absolute zero, but it does mean the old validation no longer transfers automatically.

Recommended process:

1. Start from a base feature that is already credible.
2. Add one filter at a time.
3. State clearly why the filter should improve the feature.
4. Re-explore the filtered version on the training set.
5. Re-run robustness checks on the filtered version.
6. Freeze the filtered definition.
7. Evaluate it again on validation and then test.

Use the lighter enhancement framing when the filter is small, interpretable, and clearly attached to an existing mechanism. Use the stricter new-feature framing when the filter materially changes turnover, regime exposure, or the economic story of the signal.

The main danger is combinatorial growth. Once multiple parameter choices are mixed with multiple filters, the process becomes hidden feature mining. To prevent that:

- make one change at a time,
- require one clear rationale per change,
- validate each enhanced version as its own candidate,
- reject small gains that add complexity without improving robustness.

## Walkforward Optimization

This methodology does **not** use walkforward optimization for feature discovery.

Walkforward optimization is acceptable only in the **weight layer**, where the task is to combine already-validated signals rather than discover new ones.

That distinction matters:

- **Feature layer:** search for durable market effects, then lock them.
- **Weight layer:** adapt the combination of existing signals to changing portfolio conditions.

The weight layer is still a model and must be validated with the same seriousness as any other adaptive component. Overfitting risk has not disappeared just because it happens after feature generation.

## Main Failure Modes

This process is robust only if the following risks are controlled:

- testing too many feature ideas,
- using structural-edge language to justify repeated variant mining,
- manually overfitting by inspecting too many landscapes and reformulations,
- treating validation as another tuning set,
- letting the weight layer become an uncontrolled adaptive optimizer.

The main defense is process discipline, not just statistical tooling.

## Operating Rules

- Keep feature throughput low and hypothesis quality high.
- Log all tested feature families, not just the winners.
- Prefer stable parameter regions over isolated peaks.
- Advance to validation only after the feature is frozen.
- Use the test set once, late, and for a real decision.
- Do not retune failed features into passing ones.
- Treat the adaptive weight layer as a separate validation problem.

## Decision Summary

This methodology is robust when it stays narrow and explicit:

- start from durable structural edges,
- discover on the training set,
- lock features before validation,
- use validation and test as real out-of-sample gates,
- reserve adaptation for the signal-weighting layer,
- prefer simple, stable structures over clever optimization.
