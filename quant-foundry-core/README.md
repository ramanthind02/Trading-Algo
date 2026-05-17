# QuantFoundry-Core

Shared Python library for QuantFoundry: zone configuration, validation, data slicing (UTC, inclusive bounds), and an **in-repo** QuantStats-aligned performance metrics table (Apache-2.0 derived; no PyPI `quantstats` dependency).

## Install (editable, from repo root)

```bash
pip install -e ./quant-foundry-core
```

Runtime deps include **`tabulate`** (used only if `metrics(..., display=True)`).

Design contract for inputs and conventions: [`docs/SaaS/metrics_library.md`](../docs/SaaS/metrics_library.md).

```python
import pandas as pd
from quantfoundry_core.metrics import (
    ReportMode,
    ReturnsCompounding,
    compute_aligned_performance_metrics,
)

# strategy_returns: pd.Series, DatetimeIndex (UTC-aware allowed)
report = compute_aligned_performance_metrics(
    strategy_returns,
    benchmark_returns=None,
    mode=ReportMode.FULL,
    compounding=ReturnsCompounding.SIMPLE,
    periods_per_year=252,
    prepare_returns=False,
    match_dates=False,
)
wide = report.to_dataframe()
```

## Zone manager

Design contract: [`docs/SaaS/zone_manager.md`](../docs/SaaS/zone_manager.md) (in parent Trading-Algo repo).

- **Tier 1:** `project_test_start` / `project_test_end` — single project holdout (inclusive UTC).
- **Tier 2:** `strategy_zones` — only **Train** and **Validation**; each must satisfy `end_at_utc < project_test_start`.
- Bounds: **UTC** datetimes; inclusive slice semantics for materialization.
- DataFrame input: **`DatetimeIndex`**, monotonic increasing; **tz-naive index is interpreted as UTC**.

```python
from quantfoundry_core.zone_manager import ZoneManager, ZoneSnapshot, ZoneSpec, ZoneType
from datetime import datetime, timezone
from uuid import uuid4

snapshot = ZoneSnapshot(
    project_test_start=datetime(2021, 1, 1, tzinfo=timezone.utc),
    project_test_end=datetime(2021, 12, 31, tzinfo=timezone.utc),
    strategy_zones=[
        ZoneSpec(
            zone_id=uuid4(),
            name="train",
            zone_type=ZoneType.TRAIN,
            start_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end_at_utc=datetime(2020, 6, 30, tzinfo=timezone.utc),
        ),
    ],
)
mgr = ZoneManager()
assert mgr.validate(snapshot).is_success
```
