# Research pipeline — user guide

The primary research workflow is the **frontend app** in `frontend/`. Build or edit a
`StrategySpec` in the Spec Builder, launch a run, and inspect results — all without touching
`research/feature/config.py`. An agent (in a Claude Code chat) and the UI share the same
round-trippable JSON in `research/specs/`, so either can author or edit a spec.

---

## Quick start

### 1. Start the app (two processes from the repo root)

```powershell
# API — port 5057
.\.venv\Scripts\python.exe -m uvicorn frontend.api.server:app --reload --port 5057

# Web dev server — port 5173, proxies /api → 5057
cd frontend\web
npm install   # first time only
npm run dev
```

Open http://localhost:5173.

---

## 2. Write a spec (two paths)

### Path A — Spec Builder (UI)

In the browser: go to **Spec Library → New Spec**. The form is live-validated (combo count,
window ordering, sleeve membership, fill-feed consistency). Fill in:

| Field | Notes |
|-------|-------|
| **Name / hypothesis / author** | Identity; name becomes the JSON filename stem |
| **Tickers** | Multi-select from the canonical ticker list |
| **Data feed** | `darwinex_cfd` (default) or `norgate_futures` |
| **Mode / timeframe** | DAILY (D/W/M) or INTRADAY (H1/…) |
| **Signal** | Select a bias-node module from the catalog; set param grid (list per param) |
| **Direction** | LONG / SHORT / LONG_SHORT |
| **Vol scaling** | BLENDED (default 70/30), LONG_ONLY, or OFF |
| **Execution** | entry/exit policy, unfilled-limit policy, holding, fill_feed |
| **Vault target** | One of the 13 sleeves + ensemble name |

The form shows derived values live (combo count, resolved fill feed, window dates). Save writes to
`research/specs/<name>.json`.

### Path B — Agent co-author

In a Claude Code chat, ask the agent to design a spec and write it to `research/specs/`. The agent
follows [[Strategy_research/strategy_engineering]] (category taxonomy, parsimony rules) and emits
valid JSON. On next app load the spec appears in Spec Library — edit it in the form if you like.

```python
# Python access
from research.spec import load_spec, save_spec, to_feature_config

spec = load_spec("research/specs/es_double7s_mr.json")  # validates on load
cfg  = to_feature_config(spec)                           # adapter → pipeline config
```

---

## 3. Write a brief (optional, for agent path)

If using the agent path, a brief is a plain markdown file anywhere in the repo (convention:
`research/briefs/`). Include: idea + economic rationale, instrument(s), timeframe, signal hint,
vol scaling preference, direction, constraints. Keep it 10–20 lines.

The agent reads the brief and makes all translation decisions (node selection, grid size, vault
sleeve) per [[Strategy_research/strategy_engineering]].

---

## 4. Launch a run

In the UI, open a spec → click **Run → Exploration** (or **Validation**). The app calls
`POST /api/runs` with the spec JSON and the chosen phase, shows live log output, and polls until
complete.

| Phase | What it runs | When to use |
|-------|-------------|-------------|
| `exploration` | In-sample parameter sweep (`execute_exploration_phase`) | First pass — evaluates all combos, emits robustness metrics |
| `validation` | Walk-forward + portfolio-addition gate | After exploration looks good; scores OOS + gate |

One run at a time (the backend serializes). Runs are in-memory; page refresh loses the list
(artifacts on disk persist).

---

## 5. Inspect results

After a run completes, the **Results** tab shows:

- **Plateau** — param-grid metric surface (heatmap for 2-param grids)
- **Equity** — returns/equity curve
- **Grid** — per-combo metrics table
- **Headline** — aggregated Sharpe / Sortino / Calmar / drawdown

Raw CSV/JSON artifacts land under `feature_research/shared_results/` (exploration) or the
spec's configured output root. The run log is always visible in the UI.

---

## 6. Promote to vault

After validation passes, go to **Vault → Preview** in the UI (calls `POST /api/vault/preview`).
Review the dry-run, then **Commit** (calls `POST /api/vault/commit`). Nothing is written
automatically — promotion is always a deliberate, human-initiated step.

---

## 7. Legacy `/research` agent orchestrator (superseded)

The older `research` slash command (`/research research/briefs/my_strategy.md`) is still in
place (`.claude/skills/research/`) but is **superseded by the frontend app**. It ran a
brief → spec → pipeline → memo loop through three subagents. See
[[Strategy_research/agent_harness]] for the design reference; use the frontend app for new work.

---

## Tips

- **Combo count**: aim for 20–60. Wide grids run slowly in the Nautilus lane.
- **Fixed params**: say so in the brief or set a single-element list in the grid. Every fixed
  param removes a degree of freedom.
- **Windows**: leave `windows: null` and the adapter fills the per-mode defaults (daily:
  train 2000–2018, val 2019–2022, test 2023+; intraday: train 2018–2022, val 2023–2024,
  test 2025+).
- **Windows note**: the pipeline prints unicode; if you see encoding errors, run with
  `PYTHONUTF8=1`.
- **Production build**: `cd frontend\web && npm run build` emits `frontend/web/dist/`; the
  FastAPI app then serves the whole thing from a single `uvicorn` process.

> _Verified against current code via CodeGraph on 2026-06-07._
