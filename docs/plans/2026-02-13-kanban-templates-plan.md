# Kanban Templates Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Define the new `docs/kanban/` workflow, policy, and task templates so every queued item is self-contained, executable, and honors deterministic tests.

**Architecture:** A policy README enforces the “one file per task” contract and workflow rules; a template folder hosts feature/bugfix/docs task shells with baked-in interfaces, acceptance tests, constraints, and done criteria. This keeps Codex-aligned agents from scope-drifting.

**Tech Stack:** Markdown, git (status/diff), shell (`ls`, `cat`).

---

### Task 1: Policy README

**Files:**
- Create: `docs/kanban/README.md`

**Step 1: Write the README**
```markdown
# Docs Kanban Policy

## Goal
Keep the task queue in sync with our “one file = one task” promise: every file under `docs/kanban/*` is a contract with scope, interfaces, acceptance tests, and a done checklist.

## Workflow
- `docs/kanban/todo/`: Drafts needing more detail.
- `docs/kanban/in-progress/`: Only one active file per developer/branch; spec is review-ready and Codex can start.
- `docs/kanban/done/`: Tests passed, docs updated, and the Result block appended.

## Task Contract
- Every task document must surface:
  - Context (refs to `docs/api/` or code files)
  - Explicit interfaces (module paths, signatures)
  - Invariants (no lookahead, determinism, units)
  - Acceptance tests that are numeric, assertion-based, or reproducible scripts (at least one determinism/no-lookahead/alignment test).
  - Definition of done checklist covering tests, docs, and tests executed.
- Keep scope tight: touch 1–2 modules max. If it spans more, split into additional tasks.

## Policy
- Code changes must map to a kanban file. Update the corresponding task before touching code.
- Interface changes must include updates to `docs/api/*` (note them in the task). Do not modify unrelated files.
- Acceptance tests must be executable via `pytest` or deterministic CLI; avoid vague descriptors like “make it fast.”
- Task authors should note when new parallel workstreams (sub-agents) are required and call them out in the task.

## Result Block (done files only)
- When a task moves to `done`, append:
  ```md
  ## Result
  - Implemented in: <commit hash / PR link>
  - Tests: `pytest ...` ✅
  - Notes: <surprises>
  ```
  This creates an audit trail for reviewers.
```

**Step 2: Verify directory listing**
- Run: `ls docs/kanban`
- Expect: the new README appears, and there are no other files yet (just the README).

**Step 3: Spot-check contents**
- Run: `cat docs/kanban/README.md` to ensure the policy text renders and references the workflow sections faithfully.

**Step 4: Stage the README for later commit**
- Run: `git add docs/kanban/README.md`

**Step 5: No commit yet** (hold until Task 2 completes so templates + README ship together).

### Task 2: Task Templates

**Files:**
- Create: `docs/kanban/templates/feature.md`
- Create: `docs/kanban/templates/bugfix.md`
- Create: `docs/kanban/templates/docs.md`

**Step 1: Write template contents**
- Feature template:
```markdown
# TXXX — <Short Feature Title>

## Goal
One-sentence description of the feature and why it matters.

## Context / References
- `docs/api/...` or relevant modules
- Tickets/design docs

## Scope
- In scope:
  - Minimal list of responsibilities (1–2 modules)
- Out of scope:
  - Clarify adjacent work to prevent scope creep

## Interfaces (must match)
- Modify: `<path>` — signature and behavior
- Add: `<config/class>` — expected inputs/outputs

## Data Contracts
- Schema, DataFrame columns, dataclasses, events, enums

## Dependencies
- Modules/packages touched (keep to 1–2)

## Invariants / Constraints
- No lookahead
- Deterministic identical-input behavior
- Units explicit (bps vs decimals, timezones)

## Acceptance tests
1. `<exact pytest test or script>` – deterministic check (e.g., same seed -> same outputs).
2. `<alignment test>` – features align with trades/no lookahead.
3. `<smoke test>` – `pytest tests/…` passes.

## Definition of done
- [ ] Tests added under `tests/...`
- [ ] Docs updated under `docs/api/...`
- [ ] `pytest path::test -q` passes

## Notes
- Edge cases, performance knobs, optional follow-ups
```
- Bugfix template adds sections:
  - Observed behavior
  - Expected behavior
  - Reproduction steps (commands or parameters)
  - Regression window (e.g., “Last good commit XYZ”)
  - Risk statement (what can break without warning)
- Docs template adds sections:
  - Target docs/files to update
  - Source of truth (code locations validating content)
  - Examples/snippets to add or revise

**Step 2: Verify templates directory**
- Run: `ls docs/kanban/templates`
- Expect: the three template files exist.

**Step 3: Spot-check template text**
- Run: `cat docs/kanban/templates/feature.md`
- Repeat for `bugfix.md` and `docs.md` to confirm each section is present.

**Step 4: Stage the templates**
- Run: `git add docs/kanban/templates/*.md`

**Step 5: Commit all changes**
- Run:
  ```bash
  git commit -m "chore: add kanban workflow templates"
  ```
  Includes `docs/kanban/README.md` plus the templates.
