# Param Selection Doc Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate parameter-selection docs into a single concise canonical spec while keeping existing filenames and references working.

**Architecture:** Make `docs/library/Feature_selection/Parameter Sensitivity/top_k_ensemble_selection.md` the single source of truth. Convert `docs/library/Feature_selection/Parameter Sensitivity/param_selection_rule.md` into a redirect stub. Update higher-level docs that embed the old rule so they match the canonical spec.

**Tech Stack:** Markdown docs + pytest doc-string assertions.

---

### Task 1: Lock down doc-test constraints

**Files:**
- Read: `tests/unit-tests/docs/test_walkforward_selection_weightlayer_docs.py`

**Step 1: Identify required strings/structure**
- Ensure the canonical doc continues to contain:
  - `Selection is selection-only: no forecast averaging is performed at the selection stage.`
  - A configuration table row of the form `| `selection_method` | `<default>` |`
  - The phrase `upstream walkforward prefilter` and the token `WalkforwardResearchConfig`

**Step 2: Decide whether any wording must be exact**
- Treat the strings asserted by the test as exact-match requirements.

---

### Task 2: Update the canonical spec (KISS + DRY)

**Files:**
- Modify: `docs/library/Feature_selection/Parameter Sensitivity/top_k_ensemble_selection.md`

**Step 1: Rewrite top matter**
- Keep the title and status/date block.
- Add a short "pre-committed" context section (1-2 paragraphs): identical in walkforward + production; selection outputs params only; no position if fewer than 2.

**Step 2: Keep the stable-region algorithm as authoritative**
- Retain the algorithm steps, hard filters, floor computation, connected components, min region size, k caps.
- Keep references to `grid_search_parameter_stability.md` for neighbor smoothing / adjacency.

**Step 3: Make configuration section consistent**
- Ensure `selection_method` row includes all enum values and the default from `WalkforwardResearchConfig`.
- Keep the upstream trade frequency note and explicitly mention `WalkforwardResearchConfig`.

---

### Task 3: Replace legacy spec with a redirect stub

**Files:**
- Modify: `docs/library/Feature_selection/Parameter Sensitivity/param_selection_rule.md`

**Step 1: Replace content**
- Keep a short header + status/date.
- State: this doc is superseded; canonical spec is `top_k_ensemble_selection.md`.
- Link to `grid_search_parameter_stability.md` for smoothing theory.

---

### Task 4: Remove contradictions in high-level docs (docs-only)

**Files:**
- Modify: `docs/library/Feature_selection/pipeline_overview.md` (Section 5)
- Modify (optional but preferred): `docs/library/Feature_selection/Cross_Validation/kfold_cv.md`
- Modify (optional but preferred): `docs/library/Feature_selection/Cross_Validation/cpcv.md`

**Step 1: Update pipeline overview Section 5**
- Replace the older stability_ratio/raw-threshold sketch with a short description aligned to the stable-region spec:
  - Fit all params; neighbor smoothing; upstream trade-frequency prefilter; stable-region selection by relative floor; no position if fewer than 2.
- Link to the canonical doc for full details.

**Step 2: Update CV doc references**
- Point reference rows to `top_k_ensemble_selection.md` (or keep pointing to the stub if minimizing churn).

---

### Task 5: Verify

**Step 1: Run doc unit tests**
Run:
`source venv/bin/activate && pytest tests/unit-tests/docs/test_walkforward_selection_weightlayer_docs.py -v`

Expected: PASS.

---

### Task 6: Commit (optional)

Only if explicitly requested:
`git add docs/library/Feature_selection/Parameter\ Sensitivity/top_k_ensemble_selection.md docs/library/Feature_selection/Parameter\ Sensitivity/param_selection_rule.md docs/library/Feature_selection/pipeline_overview.md docs/library/Feature_selection/Cross_Validation/kfold_cv.md docs/library/Feature_selection/Cross_Validation/cpcv.md docs/plans/2026-02-22-parameter-selection-doc-consolidation-*.md`

Commit message suggestion:
`docs: consolidate param selection specs under stable-region selector`
