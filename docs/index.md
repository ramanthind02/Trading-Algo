# docs/ — four-area map

This vault contains four areas. Pick the right one before opening any file.

---

## library/ — implementation reference

Everything that describes **how the code works today**: architecture notes, data
contracts, operator guides, and the canonical robustness-test workflow.

- **Start here:** [[library/index]] — full topic index with links to every sub-area
- **Data hub:** [[library/Data/data_platform_migration_plan]] — storage & registry
  plan (storage/, registry/, live ingest), the single authoritative source for the
  data-platform roadmap and the registry CLI

Key sub-areas:

| Folder | Covers |
|---|---|
| `library/Data/` | Data platform, providers, registry DB, live-state file contract |
| `library/bias_nodes/` | Node authoring, architecture, composition |
| `library/Ensemble/` | Base models, weight layer, portfolio layer |
| `library/Vault/` | Feature vault ownership, save/load, monitoring |
| `library/Strategy_research/` | StrategySpec, research pipeline, execution architecture |
| `library/live_forecast/` | IB legacy live driver + MT5/Nautilus vault runtime |
| `library/Deployment/` | Forecast server, production training, Cython |
| `library/Cache/` | Central cache design and lifecycle |

---

## SaaS/ — product specs and robustness tests (source of truth)

Canonical **product-level** documents: position sizing, weight-layer spec,
transaction costs, strategy spec format, and — most importantly — the
robustness-test workflow that all features must pass before promotion.

- **Robustness workflow entry point:** [[SaaS/robustness_tests/index]]
- [[SaaS/position_sizing]] — PositionSizer contract, micro/mini specs
- [[SaaS/weight_layer_spec]] — WeightLayer and FDM formula spec
- [[SaaS/transaction_costs]] — cost model used in validation

---

## refactor/nautilus/ — executed work-package plans (historical)

Completed NautilusTrader migration work packages. These are **historical records**
of what was built and why — not operational guides. Consult `library/` for current
behaviour.

---

## nautilustrader/ — vendored NT docs mirror (read-only)

Local mirror of the NautilusTrader source-repo Markdown docs, fetched by
`scripts/scrape_nautilus_docs.py`. **Read-only** — do not edit; refresh with:

```powershell
.\.venv\Scripts\python.exe scripts\scrape_nautilus_docs.py --force
```

Key pages: `concepts/architecture.md`, `concepts/strategies.md`,
`concepts/backtesting.md`, `concepts/live.md`, `integrations/ib.md`.

---

> _Verified against the working tree on 2026-06-10._
