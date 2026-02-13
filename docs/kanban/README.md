# Docs Kanban Policy

## Goal
Keep the task queue in sync with our “one file = one task” promise: each entry in `docs/kanban/` is a contract that can be handed to an implementation agent and executed with minimal back-and-forth.

## Workflow
- `docs/kanban/todo/`: Task drafts needing more detail or approval before coding.
- `docs/kanban/in-progress/`: At most one file per developer/branch; spec is complete enough for Codex to start coding. Move here once acceptance tests + interfaces are locked down.
- `docs/kanban/done/`: Tests executed, docs updated, and the `Result` block appended so reviewers can see what happened.

## Task Contract
Every task document must include:

- **Context / References** — Link to the relevant `docs/api/` page, module, or external spec.
- **Scope** — Clearly define what is in and out of scope (max 1–2 modules; split along architectural boundaries if more work is needed).
- **Interfaces (must match)** — Exact files, classes, functions, or config objects being added/modified, including signatures.
- **Invariants / Constraints** — Determinism, no-lookahead, units, timing rules, resource limits.
- **Acceptance tests** — 2+ executable checks (at least one deterministic test: same seed => same result, no-lookahead guard, or alignment test). Use `pytest` tests or deterministic scripts; avoid vague statements like “performance improves.”
- **Definition of done** — Checklist covering tests, docs, and required verification commands.
- **Notes** — Reservations, optional follow-ups, or signals that the task may spawn parallel workstreams/sub-agents.

## Policy

- **One task = one change** — Every code change must map back to a kanban task. Create or update the task before touching the code it describes.
- **Interface changes** must simultaneously update the appropriate `docs/api/*` page (even if the change is small) and note that in the task’s `Definition of done` checklist.
- **Acceptance tests** must be executable and deterministic (exact assertions, exact pytest commands, or simple scripts with a fixed dataset). Agents should not invent their own acceptance tests.
- **No unrelated edits** — Do not touch files outside the task’s stated scope.
- **One active task per agent** — Keep `docs/kanban/in-progress/` limited to a single file per developer/branch; otherwise, move spec back to `todo/` until it has an owner.

## Result Block (IN-PROGRESS → DONE)
When a task moves to `docs/kanban/done/`, append the following block to document how it landed:

```markdown
## Result
- Implemented in: <commit hash / PR link>
- Tests: `pytest path::test_name` ✅
- Notes: <unexpected findings>
```

This keeps the kanban directory an audit trail and makes it easy to see whether the implementation met the intent.
