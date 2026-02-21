# Walkforward Top-K Selection Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove robustness from enhanced walkforward top-k selection and fix selected-params reporting to emit exact selected combo rows including bin count.

**Architecture:** Keep the existing fold scoring flow but simplify enhanced selection quality to smoothed-objective-first scoring with configurable concentration and softer diversity penalty. Build selected-params export from selected top-k rows, and include `bin_count` in canonical labels when provided by context.

**Tech Stack:** Python, pandas, pytest, project walkforward modules.

---

### Task 1: Add failing tests for no-robustness selection and schema
- Update walkforward top-k tests to remove robustness expectations and assert quality is smoothed-driven.
- Update runner/io tests to assert `robustness_score` is absent and selected-params rows come from top-k selections.
- Run targeted tests and verify RED.

### Task 2: Implement walkforward selection/config/runner/io changes
- Remove robustness computation and fields from selection result and fold outputs.
- Update config fields/defaults for stronger smoothed-performance emphasis.
- Include `bin_count` in canonical labels when passed in research context.
- Update selected-params builder to use `selected_by_diversity=True` rows.

### Task 3: Update docs and verify
- Update library spec and API docs for algorithm/schema changes.
- Run targeted pytest commands until green.
- Summarize exact changed files and remaining risks.
