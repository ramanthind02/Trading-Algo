# Pre-Committed Parameter Selection Rule (Superseded)

> **Status:** Superseded library specification (redirect)
> **Date:** 2026-02-22

This doc is kept for historical context and existing links.

The canonical, current specification for parameter-combo selection is:
- [`top_k_ensemble_selection.md`](top_k_ensemble_selection.md)

Neighbour smoothing theory and grid adjacency definitions are documented in:
- [`grid_search_parameter_stability.md`](grid_search_parameter_stability.md)

Contractual invariants (pre-committed):
- The selection rule runs identically in walkforward training folds and in production refits.
- Selection outputs a selected param set only; any forecast combination is downstream.
- If `|selected| < 2`, no position is taken.
