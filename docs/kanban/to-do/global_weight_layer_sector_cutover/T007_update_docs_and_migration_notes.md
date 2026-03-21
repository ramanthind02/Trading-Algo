# T007 - Update Docs and Migration Notes

## Goal
Update documentation to describe the new global orchestration model while stating clearly that the current `WeightLayer` implementation remains the same and sector-allocation JSON is no longer part of sizing.

## Context / References
- `docs/api/ensemble.md`
- `docs/library/Ensemble/portfolio.md`
- `docs/library/Ensemble/weight_layer.md`
- `docs/methodology/sector_allocation.md`

## Scope
In scope:
- Update API and library docs for adapter-based global weighting.
- Clarify that `WeightLayer` is reused unchanged.
- Replace sector-allocation methodology doc with a migration note.

Out of scope:
- Broader methodology cleanup unrelated to this cutover

## Required Doc Changes
- `docs/api/ensemble.md`
  clarify that global diversification is achieved by portfolio-layer adaptation around `WeightLayer`
- `docs/library/Ensemble/portfolio.md`
  update pipeline diagrams and responsibility split
- `docs/library/Ensemble/weight_layer.md`
  state explicitly that the implementation is unchanged and now receives adapter-encoded global streams
- `docs/methodology/sector_allocation.md`
  replace with a short deprecation/migration note

## Invariants / Constraints
- Docs must not imply a new `WeightLayer` algorithm exists.
- Docs must state that sector config is removed from active sizing flows.

## Acceptance Tests
1. Manual doc review for consistency across API and library docs.
2. Search check: updated docs no longer describe sector JSON as an active portfolio sizing input.

## Definition of Done
- [ ] API docs updated
- [ ] Library docs updated
- [ ] Sector-allocation doc replaced with migration note
