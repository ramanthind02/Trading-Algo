# T009 — Delete Legacy feature_research Folders (Hard Cutover)

## Goal
Remove the legacy directories `feature_research/rule_based/` and `feature_research/continuous_binning/` after migration, ensuring a hard cutoff.

## Context / References
- Hard cutoff requirement from T001.

## Scope
In scope:
- Delete the old directories.
- Ensure no stray references remain (code, tests, docs).

Out of scope:
- Any behavior refactors beyond what is required to keep imports working.

## Acceptance Tests
- `python -c "import pathlib; assert not pathlib.Path('feature_research/rule_based').exists()"`
- `python -c "import pathlib; assert not pathlib.Path('feature_research/continuous_binning').exists()"`
- `python -c "import subprocess; import sys; import os"` (sanity)
- Repo grep check (pick one):
  - `python -c "import pathlib, re; import sys"` (or use `git grep`): ensure no matches for `feature_research.rule_based` / `feature_research.continuous_binning`

## Definition of Done
- Legacy folders deleted.
- No references to legacy module paths remain.
