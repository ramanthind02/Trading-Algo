# Norgate Data Docs Update Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update kanban data migration tasks and add a comprehensive Norgate data reference under the library docs.

**Architecture:** Pure documentation updates. Add one canonical library spec and link/update existing kanban tasks with data source context and contracts.

**Tech Stack:** Markdown docs only.

---

### Task 1: Create Norgate library reference

**Files:**
- Create: `docs/library/Data/Norgate.md`

**Step 1: Draft the library doc content**

Populate sections for overview, operational requirements, time series formats, datetime/timezone options, price/volume schema, identifiers (symbol vs assetid), futures metadata, error handling, and migration implications.

**Step 2: Review for completeness and consistency**

Manually scan for missing fields (open interest, delivery month, unadjusted close, dividend/padding status) and confirm language matches repository conventions.

**Step 3: Verification**

Docs-only change: no automated tests. Confirm file renders as plain Markdown and links use repository-relative paths.

**Step 4: Commit**

```bash
git add docs/library/Data/Norgate.md
git commit -m "docs: add Norgate data reference"
```

### Task 2: Update kanban data migration tasks

**Files:**
- Modify: `docs/kanban/to-do/data_migration/T001_roll_rules_dataclass.md`
- Modify: `docs/kanban/to-do/data_migration/T002_roll_detector.md`
- Modify: `docs/kanban/to-do/data_migration/T003_gap_calculator.md`
- Modify: `docs/kanban/to-do/data_migration/T004_back_adjuster.md`
- Modify: `docs/kanban/to-do/data_migration/T005_orchestrator.md`
- Modify: `docs/kanban/to-do/data_migration/T006_validator_comparison.md`

**Step 1: Add Data Source references**

Insert a short “Data Source” paragraph in each task referencing `docs/library/Data/Norgate.md` and noting roll methodology differences where relevant.

**Step 2: Update Context/References and data contracts**

Add Norgate to Context/References and clarify comparison inputs (especially T006). Provide a concrete placeholder path for Norgate data used in validation.

**Step 3: Review for scope alignment**

Ensure changes stay within the current task scope and do not alter acceptance tests unless a path needs to be clarified.

**Step 4: Commit**

```bash
git add docs/kanban/to-do/data_migration/*.md
git commit -m "docs: align data migration tasks with Norgate source"
```

### Task 3: Final review

**Files:**
- Verify: `docs/library/Data/Norgate.md`
- Verify: `docs/kanban/to-do/data_migration/*.md`

**Step 1: Cross-link verification**

Ensure each task references `docs/library/Data/Norgate.md` and that any placeholder paths are clearly labeled for later updates.

**Step 2: Verification**

Docs-only change: no automated tests. Confirm no unrelated files were modified.

**Step 3: Commit (if needed)**

If any edits were made during the review:

```bash
git add docs/library/Data/Norgate.md docs/kanban/to-do/data_migration/*.md
git commit -m "docs: refine Norgate migration references"
```
