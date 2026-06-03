from __future__ import annotations

from pathlib import Path

from feature_research.inclusion_gates import gate_tearsheet_panel_title


def test_gate_tearsheet_panel_title_full_portfolio() -> None:
    path = Path(
        "validation/visualization/portfolio_gate_tearsheets/full_portfolio/"
        "with_candidate/train_plus_validation/tearsheet.html"
    )
    assert gate_tearsheet_panel_title(path) == "Full portfolio — train+validation (with candidate)"


def test_gate_tearsheet_panel_title_candidate_strategy() -> None:
    path = Path("portfolio_gate_tearsheets/candidate_strategy/train/tearsheet.html")
    assert gate_tearsheet_panel_title(path) == "Candidate strategy — train"


def test_gate_tearsheet_panel_title_sleeve_without() -> None:
    path = Path(
        "portfolio_gate_tearsheets/sleeve_equity_indices__mean_reversion/"
        "without_candidate/validation/tearsheet.html"
    )
    title = gate_tearsheet_panel_title(path)
    assert title == "Sleeve (equity indices / mean reversion) — validation (without candidate)"


def test_gate_tearsheet_panel_title_ignores_legacy_walkforward_html() -> None:
    path = Path("signed_signal/module/validation/tearsheets/train_ensemble_tearsheet.html")
    assert gate_tearsheet_panel_title(path) is None
