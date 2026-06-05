from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from research.feature.ui.combo_analysis import build_combo_diagnostics


def test_build_combo_diagnostics_prefers_existing_robustness_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "research.feature.ui.combo_analysis.expand_bias_specs",
        lambda _bias_spec: [{"params": {"period_1": 8}} for _ in range(6)],
    )
    (tmp_path / "robustness_report.json").write_text(
        json.dumps(
            {
                "n_combinations": 6,
                "n_effective": {"n_effective": 2.5, "surface_label": "Smooth surface."},
            }
        ),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        bias_spec={"module_name": "stacked_sma_long_only", "params": {"period_1": [8, 16]}},
        eval_bias_spec={"module_name": "stacked_sma_long_only", "params": {"period_1": 8}},
        reports_dir=tmp_path,
    )

    diagnostics = build_combo_diagnostics(config)

    assert diagnostics.raw_combo_count == 6
    assert diagnostics.loaded_combo_count == 6
    assert diagnostics.effective_combo_count == 2.5
    assert diagnostics.surface_interpretation == "Smooth surface."
    assert diagnostics.notes == ("Effective combo count loaded from the current exploration robustness report.",)
