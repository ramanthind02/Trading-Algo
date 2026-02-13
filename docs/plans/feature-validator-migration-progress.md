# Feature Validator Migration Progress

**Date Started:** 2026-02-12
**Goal:** Implement the new FeatureValidator class and migrate away from FeatureExplorer

## Verification Commands

Run after each implementation slice:

```bash
# 1. Tests
pytest tests/validators/ -v

# 2. Type checking
mypy feature_selection/validators/ --strict

# 3. Linting (optional)
ruff check feature_selection/validators/
```

## Spec Requirements Checklist

### Core Components

- [ ] ValidationConfig dataclass
- [ ] FeatureValidator main class
- [ ] Report data structures
  - [ ] DescriptiveStats
  - [ ] ADFTestResult, KPSSTestResult
  - [ ] MonotonicityTestResult
  - [ ] EDAReport
  - [ ] ContinuousEDAReport
  - [ ] RuleEDAReport
  - [ ] PermutationReport
  - [ ] StabilityReport
  - [ ] FoldResult
  - [ ] ValidationReport (top-level)

### EDA Methods

- [ ] Common EDA
  - [ ] Distribution statistics
  - [ ] Correlation analysis (Pearson, Spearman, Kendall)
  - [ ] Stationarity tests (ADF, KPSS)
  - [ ] Lagged correlations
  - [ ] Rolling correlation
  - [ ] Temporal stability
- [ ] Continuous-specific EDA
  - [ ] Decile analysis
  - [ ] Monotonicity testing
  - [ ] Outlier detection
  - [ ] Non-linearity tests (polynomial regression)
  - [ ] Binning diagnostics
- [ ] Rule-based-specific EDA
  - [ ] Level distribution
  - [ ] Per-level statistics
  - [ ] Transition matrix
  - [ ] Average duration
  - [ ] Confidence intervals
  - [ ] Grid report integration

### Permutation Testing

- [ ] Stage 1: Vector Shuffle
  - [ ] Shuffle feature vector
  - [ ] Compute permuted Sharpe/t-stat
  - [ ] p-value calculation
  - [ ] Pass/fail verdict
- [ ] Stage 2: Pipeline Permutation
  - [ ] Continuous: Feature shuffle
  - [ ] Continuous: Candle shuffle
  - [ ] Rule-based: Candle shuffle
  - [ ] Integration with binning pipeline
- [ ] Integration with existing PermutationEngine

### Stability Analysis (Stage 3)

- [ ] Walkforward fold structure
- [ ] Per-fold parameter evaluation
- [ ] Grid-aware neighbor smoothing
- [ ] Top-K parameter selection per fold
- [ ] Temporal consistency metrics
- [ ] Stability report generation

### Parameter Sensitivity

- [ ] 1D parameter analysis
- [ ] 2D parameter analysis (heatmaps)
- [ ] 3D parameter analysis (interactive)
- [ ] 4D parameter analysis (interactive)
- [ ] Robustness metrics
- [ ] Integration with existing ParameterAnalyzer

### High-Level Workflows

- [ ] `run_full_validation()` - complete pipeline
- [ ] `run_progressive_validation()` - stage-by-stage
- [ ] Early exit on permutation failure
- [ ] ValidationReport accumulation

### Report Export

- [ ] HTML export (with embedded Plotly)
- [ ] PDF export
- [ ] Markdown export
- [ ] JSON export
- [ ] Researcher summary generation

### Testing & Migration

- [ ] Unit tests for all components
- [ ] Integration tests (end-to-end pipeline)
- [ ] Test continuous features
- [ ] Test rule-based features
- [ ] FeatureExplorer compatibility layer (if needed)
- [ ] Migrate existing FeatureExplorer usage

## Implementation Tasks (from plan)

### Phase 1: Report Data Structures ✅ / ⏳ / ❌

- [ ] Task 1: Foundation - Base Report Structures
  - [ ] Step 1: Test for DescriptiveStats ❌
  - [ ] Step 2: Run test (expect fail) ❌
  - [ ] Step 3: Create directory structure ❌
  - [ ] Step 4: Implement DescriptiveStats ❌
  - [ ] Step 5: Run test (expect pass) ❌
  - [ ] Step 6: Add tests for ADF/KPSS results ❌
  - [ ] Step 7: Run all tests (expect pass) ❌
  - [ ] Step 8: Commit ❌

- [ ] Task 2: EDA Report Structures
  - [ ] Step 1: Test for EDAReport ❌
  - [ ] Step 2: Run test (expect fail) ❌
  - [ ] Step 3: Implement EDAReport ❌
  - [ ] Step 4: Run test (expect pass) ❌
  - [ ] Step 5: Commit ❌

- [ ] Task 3: Permutation & Stability Reports
  - [ ] Step 1: Test for PermutationReport ❌
  - [ ] Step 2: Run test (expect fail) ❌
  - [ ] Step 3: Implement reports ❌
  - [ ] Step 4: Run test (expect pass) ❌
  - [ ] Step 5: Commit ❌

### Phase 2: Configuration

- [ ] Task 4: ValidationConfig
  - [ ] Step 1: Test for ValidationConfig ❌
  - [ ] Step 2: Run test (expect fail) ❌
  - [ ] Step 3: Implement ValidationConfig ❌
  - [ ] Step 4: Run tests (expect pass) ❌
  - [ ] Step 5: Commit ❌

### Phase 3: EDA Methods

- [ ] Task 5: Common EDA Methods
- [ ] Task 6: Continuous-Specific EDA
- [ ] Task 7: Rule-Based-Specific EDA

### Phase 4: Permutation Testing

- [ ] Task 8: Permutation Testing Integration

### Phase 5: Core Orchestration

- [ ] Task 9: FeatureValidator Init + EDA Orchestration
- [ ] Task 10: Stability Analysis Methods
- [ ] Task 11: Permutation Stages
- [ ] Task 12: Full Pipeline

### Phase 6: Report Export

- [ ] Task 13: Report Export Functions

### Phase 7: Integration & Testing

- [ ] Task 14: Integration Tests

## Current Status

**Working on:** Creating progress ledger
**Last completed:** N/A
**Next up:** Task 1 - Base Report Structures

## Notes

- Following TDD approach: write test → fail → implement → pass → commit
- Small, verifiable slices
- Run verification commands after each slice
- Preserve existing FeatureExplorer behavior where needed
