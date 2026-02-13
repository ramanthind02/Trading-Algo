---
active: true
iteration: 140
max_iterations: 0
completion_promise: null
started_at: "2026-02-13T07:19:41Z"
---

Goal:
Scan the entire repository and generate/update API docs into docs/api/.

Hard constraints:
- Follow docs/api/_template.md for formatting.
- Follow docs/api/_scope.md for include/exclude and public API rules.
- Keep changes limited to docs/api/* only.

Process:
1) Inventory pass:
   - Scan included code paths (per docs/api/_scope.md).
   - Build docs/api/_inventory.md:
     - list major packages/modules
     - list key entrypoints (scripts/CLIs)
     - list cross-module surfaces (symbols imported across packages)

2) Parallel doc pass (use sub-agents):
   - Group the repo into ~5–12 coherent doc targets (avoid tiny per-file docs).
   - Spawn one sub-agent per target group.
   - Each agent writes/updates docs/api/<group>.md using docs/api/_template.md:
     - Purpose
     - Public API reference (signatures + behavior)
     - Data contracts
     - Errors/logging
     - Minimal examples

3) Update docs/api/index.md:
   - Link each generated module page with a one-line description.

## Docs Landscape
- `docs/api/` contains the generated API documentation; follow `_template.md`/`_scope.md` and keep this tree in sync whenever you touch exported interfaces.
- `docs/kanban/` is the task queue: `README.md` defines the workflow and policy, and `templates/feature|bugfix|docs.md` provide the enforced “one file = one task” contract for future coding efforts.
- `docs/library/`, `docs/methodology/`, `docs/plans/`, and `docs/complete/` host research notes, playbooks, plans, and validation philosophy—cite or update them when your change touches domain assumptions or methodology.

Output rules:
- If a doc exists, update it without unnecessary rewriting.
- If behavior is unclear, document observable behavior and add Open questions.
- Document time-alignment/no-lookahead constraints where relevant.
