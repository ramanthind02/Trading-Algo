# Feature Validation Pipeline — Documentation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write comprehensive documentation capturing the full feature validation pipeline as designed in conversation, establishing it as the single source of truth.

**Architecture:** Two-tier design — IS screening (coarse filter, full IS period) feeds into walkforward validation (definitive test). Features graduate (not individual param combos). All param combos remain in the grid throughout; the pre-committed param selection rule selects the ensemble within each fold.

**Tech Stack:** Markdown docs in `docs/library/Feature_selection/`

---

## Task 1: Create `pipeline_overview.md` (new master document)

**Files:**
- Create: `docs/library/Feature_selection/pipeline_overview.md`

Content must cover (in order):
1. Philosophy & Design Goals
2. Data Split Strategy
3. Pre-Committed Param Selection Rule (the algorithm)
4. Phase 1: IS EDA (informational)
5. Phase 2: IS Permutation Screening (coarse filter)
6. Phase 3: Walkforward Validation (definitive)
7. Phase 4: Walkforward Permutation Test (statistical gate)
8. Phase 5: Graduation & Production
9. What informs what (decision boundary clarity)
10. Training window size requirements
11. Cross-references to existing specs

---

## Task 2: Create `param_selection_rule.md`

**Files:**
- Create: `docs/library/Feature_selection/Parameter Sensitivity/param_selection_rule.md`

Content: The pre-committed param selection algorithm — identical in walkforward and production. Steps, thresholds, ensemble formation, minimum member requirements.

---

## Task 3: Update `feature_validator.md`

**Files:**
- Modify: `docs/library/Feature_selection/feature_validator.md`

Changes:
- Add header note pointing to `pipeline_overview.md` as the master pipeline reference
- Clarify this doc covers Phase 1 (EDA) and Phase 2 (IS permutation screening) only
- Note that feature-level graduation (not param-level) is the correct interpretation
- Update "Implementation Scope" to reflect the two-tier pipeline

---

## Task 4: Update `in-sample_pt.md`

**Files:**
- Modify: `docs/library/Feature_selection/Permutation Testing/in-sample_pt.md`

Changes:
- Add preamble clarifying this is Phase 2 (IS Screening), a coarse filter
- Clarify that Stage 1-2 per-param results are diagnostic only; feature-level verdict is the gate
- Note that all params enter walkforward regardless of IS permutation result
- Add reference to `pipeline_overview.md`
