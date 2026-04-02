# Domain Discrete Refactor Ticket Backlog

## Purpose

This backlog tracks the cutover from fitted binning models to one frozen signed-signal feature path.

Guiding principle: feature behavior is chosen in research, frozen into a node-backed spec, and executed deterministically in validation, test, and production.

## Decisions Locked

- Big-bang cutover: no coexistence window for legacy fitted feature artifacts.
- Canonical runtime output is signed discrete signal `-1/0/+1`.
- Per-feature multiplier fitting is removed from the feature layer.
- One generic wrapper node plus frozen specs is preferred over generated per-feature Python modules.

## Tickets

- [01-domain-discrete-contract-and-schema.md](./01-domain-discrete-contract-and-schema.md)
- [02-domain-discrete-node-framework.md](./02-domain-discrete-node-framework.md)
- [03-research-eda-and-freeze-workflow.md](./03-research-eda-and-freeze-workflow.md)
- [04-feature-layer-cutover.md](./04-feature-layer-cutover.md)
- [05-vault-ensemble-and-control-file-cutover.md](./05-vault-ensemble-and-control-file-cutover.md)
- [06-validation-walkforward-and-deployment-cutover.md](./06-validation-walkforward-and-deployment-cutover.md)
- [07-legacy-removal-docs-and-tests.md](./07-legacy-removal-docs-and-tests.md)

## Suggested Execution Order

1. `01-domain-discrete-contract-and-schema.md`
2. `02-domain-discrete-node-framework.md`
3. `03-research-eda-and-freeze-workflow.md`
4. `04-feature-layer-cutover.md`
5. `05-vault-ensemble-and-control-file-cutover.md`
6. `06-validation-walkforward-and-deployment-cutover.md`
7. `07-legacy-removal-docs-and-tests.md`

## Definition Of Done

- There is one canonical feature path: frozen domain-discrete node spec -> signed signal.
- No runtime path learns, restores, or persists bin geometry.
- Control files and vault entries no longer store base-model fitted payloads.
- Research, validation, walkforward, ensemble loading, and deployment all consume the same signed-signal contract.
- Legacy fitted feature artifacts fail fast with an explicit migration error.
