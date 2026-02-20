# Parameter Sensitivity 2D Plot UX — Design Doc

**Date:** 2026-02-20
**Status:** Approved

## Problem

`plot_2d_stability_heatmap` overlays smoothed objective and stability contours in one dense view. Interpreting raw vs smoothed objective vs stability ratio is difficult in a single static composition.

## Goal

Keep one plot while improving interpretability and research utility via interactive toggles and broader metric-layer support.

## Design

1. Keep a single Plotly figure from `plot_2d_stability_heatmap`.
2. Add selectable base layers (dropdown) with defaults:
   - `smoothed` (`smoothed_{metric}` when available)
   - `raw` (`metric`)
   - `stability_ratio`
   - `n_neighbors`
   - `delta` (`raw - smoothed`)
3. Add overlay control buttons for:
   - no overlay
   - stable-region markers only
   - stability contours only
   - both markers and contours
4. Allow additional research metrics by accepting layer keys that match DataFrame column names (beyond predefined aliases).
5. Keep backward compatibility for existing calls and report generation.

## Out of Scope

- New multi-figure API in report dataclass.
- Reworking 1D/3D plot behavior.
