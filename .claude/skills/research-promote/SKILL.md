---
name: research-promote
description: The single human-gated step that writes an approved researched strategy into the vault. Invoke as `/research-promote <experiment-dir>` only AFTER a human has read the memo and approved. Dry-runs first, then performs the canonical vault save. Never auto-invoked by /research.
---

# /research-promote — the one human-gated vault write

Promotion is **human-only** (see `docs/library/Strategy_research/agent_harness.md`). `/research`
never promotes; it stops at the memo. This command is the *separate, explicit* action a human takes
after reading `memo.md` and deciding to keep the strategy.

> The `PreToolUse` guardrail blocks the `Edit`/`Write` **tools** on the vault trees, so the agent can
> never hand-edit a vault JSON. It does **not** block the canonical save *pipeline* run via Bash
> (ordinary Python file I/O) — that is exactly this audited, human-triggered path.

## Inputs

`$ARGUMENTS` is the experiment dir, e.g. `research/agent_experiments/2026-06-06_turnaround_tuesday/`.

## Gate (do not skip)

1. Confirm `memo.md` exists in the experiment dir and that **the human has explicitly approved** in
   this session. If approval was not stated, **stop and ask** — do not promote on inference.
2. Read `spec.py` → `SPEC`; confirm `SPEC.vault` (the sleeve ∈ the 13 and the `ensemble_name`).

## Promote (dry-run, then write)

Use the experiment's spec — **not** `feature_research.config.load_config()` — as the source of the
vault target. Build the config through the adapter and flip the save to live:

```python
import dataclasses
from research.spec import to_feature_config           # builds a FRESH ResearchConfig
# load SPEC from the experiment's spec.py, then:
cfg = to_feature_config(SPEC)
vault_save = dataclasses.replace(cfg.vault_save, dry_run=False)   # promotion writes
```

1. **Dry-run first.** Resolve and print the destination
   (`<vault_root>/<TF>/<weight_hierarchy_group>/<ensemble_name>_<direction>/`) and the feature JSON
   that would be written. Reuse the canonical save machinery in
   `research/feature/save_feature_to_vault.py` (`_resolve_ensemble_dir`, the eval bias-spec
   normalization). Show the human the exact paths.
2. **Write** only after the human confirms the dry-run looks right. Run the canonical save with
   `dry_run=False`. The save uses the scalar `eval_bias_spec` (a single param combo), not the grid.
3. Report the written path(s) and remind the human to commit the vault change deliberately.

## Constraints

- Promote exactly one approved strategy per invocation; never batch-promote.
- Never promote without explicit human approval in-session.
- Do not edit `research/feature/config.py` / `research/portfolio/config.py` — build the config from
  the spec via the adapter (the guardrail enforces this regardless).
