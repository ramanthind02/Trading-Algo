# Ticket 03: Replace Phase-2 Binning With Freeze Workflow

## Why

The current research storyline still assumes learned bin geometry for continuous features.
That conflicts with the new architecture, where cutpoints are chosen in research and never fit in production.

## Goal

Replace Phase-2 binning analysis with a Phase-1 freeze workflow that validates distributional stability and materializes a frozen node spec for downstream validation.

## Scope

- Remove Phase-2 learned binning from the preferred research path.
- Define the Phase-1 freeze workflow:
  - inspect feature time series
  - inspect occupancy across absolute bins
  - inspect rolling occupancy/drift behavior
  - confirm chosen cutpoints and selected bins
  - materialize the frozen `bias_node_spec`
- Add explicit diagnostics for:
  - occupancy concentration
  - rolling occupancy drift
  - obvious distributional instability
- Define the researcher handoff artifact:
  - one frozen `bias_node_spec` using the domain-discrete contract
- Update the research narrative so validation and test reuse the exact frozen spec with no fit step.

## Out Of Scope

- Rewriting every EDA plot in one pass.
- Changing portfolio or weighting methodology.

## Acceptance Criteria

- The research workflow no longer requires Phase-2 quantile-bin selection.
- The output of research is a frozen domain-discrete spec, not fitted edges.
- The docs and implementation notes make the train/validation/test contract explicit: same spec, no adaptive geometry.

## Risks

- Researchers may overfit cutpoints manually if diagnostics are not disciplined and explicit.

## Dependencies

- Tickets 01 and 02.
