# Portfolio Addition

> [!important]
> The source of truth for this stage is [[SaaS/robustness_tests/portfolio_addition]].

## Role in the workflow

Portfolio addition comes **after validation**.

It answers:

> Does the validated strategy improve the portfolio before the project holdout is opened?

That is why `portfolio_addition` is now the preferred stage name. It is more specific than the old generic `oos` label.

## Canonical meaning

In the active workflow:

```text
exploration -> validation -> portfolio_addition
```

Portfolio addition is a portfolio-admission decision, not a general-purpose OOS bucket.

The primary gate is **composite** (sleeve ΔSR plus max drawdown, ulcer index, and stress-period drawdown). See [[SaaS/robustness_tests/portfolio_addition]] §5.2.

When validation runs with the portfolio addition gate enabled, sleeve-scoped QuantStats HTML tearsheets are written under the validation visualization folder at `portfolio_gate_tearsheets/sleeve_<label>/` (sanitized `<asset>_<style>` sleeve label; with/without candidate on train, validation, and train+validation). Toggle via `portfolio_addition_gate.emit_sleeve_tearsheets` (default `True`). The full-book with/without and candidate-standalone tearsheets are written under the same `portfolio_gate_tearsheets/` root and toggle via `portfolio_addition_gate.emit_tearsheets` (default `True`).

## Local command surface

Preferred command:

```bash
python -m research.feature portfolio_addition
```

Compatibility alias still present:

```bash
python -m research.feature oos
```

Use the first name in docs and discussion. Keep the second only as a migration note.

## Local artifact surface

Some local artifact paths still use legacy naming:

- `oos_window`
- `visualization/oos/`
- `oos-permutation`

Those labels should be read as **portfolio-addition-era outputs**, not as the preferred workflow vocabulary.

The current local visualization area is:

```text
output_root/visualization/oos/
```

That folder name is legacy, but the conceptual stage is portfolio addition.

## What to read from the SaaS source-of-truth doc

Use the SaaS page for the actual gate logic:

- analytical hurdle
- empirical portfolio Sharpe comparison
- weight assessment
- IDM improvement
- contamination rules after a failed gate

This library page exists only to keep the local code/docs naming aligned with that workflow.

## What this stage is not

Portfolio addition is not:

- the project holdout
- live monitoring
- permission to change parameters after seeing the gate result

Those downstream workflows live in:

- [[SaaS/robustness_tests/portfolio_holdout]]
- [[SaaS/robustness_tests/monitoring]]

## Related

- [[Feature_selection/pipeline]]
- [[Feature_selection/validation]]
- [[SaaS/robustness_tests/portfolio_addition]]

> _Verified against commit a07b6bf->197221e on 2026-06-04 (docs Phase A; WP-8 restructure repoint)._
