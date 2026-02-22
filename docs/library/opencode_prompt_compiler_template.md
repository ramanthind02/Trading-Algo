# Prompt Compiler Template (Meta -> Engineered Prompt)

Use this file when you want a consistent workflow:

meta prompt (you) -> engineered prompt (from `@plan`) -> execution (in `@build` / `@orchestrator`).

This is designed for this repo’s stack:

- OpenCode
- omos (oh-my-opencode-slim) roles (explorer/librarian/oracle/fixer)
- local RAG + cognitive memory (`/recall`, `/remember`, hook injection)
- Superpowers skills (invoke on-demand)

## How To Use

1) OpenCode: talk to `@plan`.
2) Paste the “Meta Prompt” template below and fill it in.
3) `@plan` returns an “Engineered Prompt Pack”.
4) You copy the pack into:

- `@orchestrator` for research/planning, OR
- `@build` for implementation.

## Meta Prompt (Paste Into @plan)

```text
PHASE: research | planning | execute
TASK TYPE: feature | refactor | bugfix | docs | experiment
GOAL: <one sentence>

SCOPE:
- In-scope:
- Out-of-scope:

CONSTRAINTS:
- Token priority: low | medium | high
- Tools allowed: local-only | allow-web

KNOWN AREAS (optional):
- Suspect dirs/files: nodes/, ensemble/, execution/, deployment/, utils/, docs/

MUST-NOT-CHANGE (invariants):
- <schemas/semantics that must remain stable>

ACCEPTANCE CRITERIA:
1) <observable behavior/result>
2) <observable behavior/result>

VERIFICATION:
- Target tests to run (python -m pytest ...):
- Integration tests (if cross-layer):

MEMORY:
- /remember candidates (only if truly stable preferences/decisions):
- /recall queries to run first:
```

## Engineered Prompt Pack (What @plan Should Output)

Tell `@plan` to output exactly these sections:

1) Run Mode

- Which preset to use:
  - `OH_MY_OPENCODE_SLIM_PRESET=research` or `OH_MY_OPENCODE_SLIM_PRESET=execute`
- Recommended local retrieval controls (only if needed):
  - `RAG_MODE=execute|research|off`
  - `RAG_MAX_CHARS=...`

2) Role Delegation

- Which omos roles to use and why:
  - explorer: repo mapping
  - oracle: architecture risk checks / tricky debugging
  - librarian: external references (only if Tools allowed = allow-web)

3) Context Bootstrap (Explicit)

- `/recall <query>` list
- `Read` list: exact file paths to open before coding
- `Grep` targets: symbols/strings to search

4) Execution Prompt (Paste-This)

- A single prompt tailored for:
  - `@orchestrator` (research/planning), OR
  - `@build` (execute)

It must include:

- the acceptance criteria
- the invariants
- the verification commands
- the context bootstrap

## Examples

### Example A: Research Phase (Architecture)

Paste into `@plan`:

```text
PHASE: research
TASK TYPE: refactor
GOAL: Decide how to refactor the weight layer without breaking vault/deployment readers.

SCOPE:
- In-scope: design options, migration plan, compatibility strategy
- Out-of-scope: implementing code changes

CONSTRAINTS:
- Token priority: medium
- Tools allowed: local-only

MUST-NOT-CHANGE (invariants):
- Do not silently change control-file schema semantics used by ensembles/vault/deployment.

ACCEPTANCE CRITERIA:
1) We pick 1 approach and list trade-offs.
2) We define a migration plan + verification steps.

VERIFICATION:
- Target tests to run (python -m pytest ...): <leave blank for now>
- Integration tests (if cross-layer): identify candidates

MEMORY:
- /remember candidates: <none>
- /recall queries to run first: weight layer schema, vault artifacts, deployment readers
```

### Example B: Planning Phase (Write A Plan File)

Paste into `@plan`:

```text
PHASE: planning
TASK TYPE: feature
GOAL: Create a plan to add a new bias node and validate it properly.

SCOPE:
- In-scope: plan doc with steps, exact files, tests, and acceptance criteria
- Out-of-scope: implementation

CONSTRAINTS:
- Token priority: low
- Tools allowed: local-only

MUST-NOT-CHANGE (invariants):
- Preserve feature naming consistency for bias-node generated columns.
- Keep unit vs integration test taxonomy strict.

ACCEPTANCE CRITERIA:
1) Plan includes exact files and new column naming.
2) Plan includes targeted unit tests + any required integration tests.

VERIFICATION:
- Target tests to run (python -m pytest ...): tests/...
- Integration tests (if cross-layer): tests/integration/...

MEMORY:
- /remember candidates: Prefer frozen dataclasses.
- /recall queries to run first: bias node naming convention, BaseBiasNode patterns
```

### Example C: Execute Phase (Implement The Approved Plan)

Paste into `@plan`:

```text
PHASE: execute
TASK TYPE: refactor
GOAL: Implement the approved plan in docs/plans/XXXX.md.

SCOPE:
- In-scope: implement exactly what the plan says + tests
- Out-of-scope: expanding scope, adding extra features

CONSTRAINTS:
- Token priority: low
- Tools allowed: local-only

MUST-NOT-CHANGE (invariants):
- No silent schema changes.

ACCEPTANCE CRITERIA:
1) Tests pass.
2) Behavior matches plan.

VERIFICATION:
- Target tests to run (python -m pytest ...): <from plan>
- Integration tests (if cross-layer): <from plan>

MEMORY:
- /remember candidates: <none>
- /recall queries to run first: <from plan>
```

## Notes On Token Cost

- In execute phase, the goal is to keep the plan + the exact files you `Read` as the dominant context.
- If you need more repo context, use targeted `/recall` and explicit `Read` instead of raising automatic injection for every turn.
