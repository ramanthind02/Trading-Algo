#+#+#+#+## Design: Consolidate Param Selection Docs

**Date:** 2026-02-22

### Goal
Consolidate the parameter-selection documentation under a single concise, DRY library spec, using `docs/library/Feature_selection/Parameter Sensitivity/top_k_ensemble_selection.md` as the source of truth.

### Non-Goals
- No algorithm changes.
- No code changes.
- No renaming of the existing files.

### Constraints
- Preserve existing references and tests that read `docs/library/Feature_selection/Parameter Sensitivity/top_k_ensemble_selection.md`.
- Keep the canonical doc explicit that selection is selection-only (no forecast averaging at selection stage).

### Decision
**Approach 1:** Keep existing filenames. Make `top_k_ensemble_selection.md` the canonical spec. Convert `param_selection_rule.md` into a short redirect stub to the canonical spec.

### Planned Edits
1. Update `docs/library/Feature_selection/Parameter Sensitivity/top_k_ensemble_selection.md`:
   - Incorporate only the minimal, non-redundant context from `param_selection_rule.md` (pre-committed, identical in walkforward + production, no-position behavior when too few params).
   - Keep stable-region algorithm and config semantics as authoritative.
   - Keep any exact strings relied on by `tests/unit-tests/docs/test_walkforward_selection_weightlayer_docs.py`.

2. Replace the body of `docs/library/Feature_selection/Parameter Sensitivity/param_selection_rule.md` with a concise pointer:
   - State that it is superseded.
   - Link to `top_k_ensemble_selection.md` (canonical) and `grid_search_parameter_stability.md` (smoothing theory).

3. Remove obvious contradictions in high-level library docs (docs-only):
   - Update `docs/library/Feature_selection/pipeline_overview.md` section 5 to align with the stable-region selection spec.
   - Update cross-validation docs to link to the canonical spec (or rely on the stub).

### Verification
Run:
- `pytest tests/unit-tests/docs/test_walkforward_selection_weightlayer_docs.py`

### Git Hygiene
No commits are created unless explicitly requested.
