# T010 — Update Docs for Multi-TF Portfolio Architecture

## Goal
Update `docs/library/Ensemble/portfolio.md`, `docs/library/Ensemble/weight_layer.md`, and `CLAUDE.md` to reflect the new two-level portfolio hierarchy, the `TFPortfolio`/`GlobalPortfolio` split, and the `GlobalWeightLayer` cross-TF FDM.

## Context / References
- `docs/library/Ensemble/portfolio.md`
- `docs/library/Ensemble/weight_layer.md`
- `CLAUDE.md` — architecture section pipeline description
- T003/T004/T005 specs (source of truth for the new interfaces)

## Scope
In scope:
- Update `portfolio.md` pipeline diagram to show `TFPortfolio` → `GlobalWeightLayer` → `GlobalPortfolio`.
- Update multiplier comparison table to add cross-TF FDM row.
- Update usage example in `portfolio.md` to show multi-TF construction.
- Update `weight_layer.md` to add a section for `GlobalWeightLayer` (input schema, resampling contract, cross-TF FDM).
- Update `CLAUDE.md` architecture section pipeline description.

Out of scope:
- API reference generation.
- Deployment pipeline docs.

## Dependencies
- T008 and T009 complete (architecture finalised before docs are written).

## Acceptance Tests
- No code tests.
- All doc links and class names in docs match the actual implemented classes.

## Definition of Done
- [ ] `portfolio.md` updated with new pipeline diagram and multi-TF usage example
- [ ] `weight_layer.md` updated with `GlobalWeightLayer` section
- [ ] `CLAUDE.md` architecture pipeline updated
- [ ] No references to old single-level `Portfolio` as the top-level class (except as alias note)

## Notes
- Keep docs concise — update existing sections in place rather than appending. This is not a changelog.
