"""Tests for Portfolio vault auto-load behavior."""

from pathlib import Path

from ensemble.portfolio import Portfolio


def _make_vault_tree(root: Path) -> None:
    (root / "D" / "mean-reversion_indices_long").mkdir(parents=True, exist_ok=True)
    (root / "W" / "carry_indices_short").mkdir(parents=True, exist_ok=True)
    (root / "M").mkdir(parents=True, exist_ok=True)


def test_portfolio_autoloads_all_when_ensembles_none(tmp_path, monkeypatch) -> None:
    from ensemble import vault_manager

    vault_root = tmp_path / "vault"
    _make_vault_tree(vault_root)
    loaded = []

    def _fake_load(path: str, refit: bool = False, target_volatility: float = 0.20):
        loaded.append(Path(path).name)
        return {"ensemble_dir": Path(path).name}

    monkeypatch.setattr(vault_manager, "load_ensemble_from_vault", _fake_load)
    portfolio = Portfolio(ensembles=None, vault_root=str(vault_root))

    assert len(portfolio.ensembles) == 2
    assert set(loaded) == {"mean-reversion_indices_long", "carry_indices_short"}


def test_portfolio_autoload_filters_by_ensemble_names(tmp_path, monkeypatch) -> None:
    from ensemble import vault_manager

    vault_root = tmp_path / "vault"
    _make_vault_tree(vault_root)
    loaded = []

    def _fake_load(path: str, refit: bool = False, target_volatility: float = 0.20):
        loaded.append(Path(path).name)
        return {"ensemble_dir": Path(path).name}

    monkeypatch.setattr(vault_manager, "load_ensemble_from_vault", _fake_load)
    portfolio = Portfolio(
        ensembles=None,
        ensemble_names=["mean-reversion_indices_long"],
        vault_root=str(vault_root),
    )

    assert len(portfolio.ensembles) == 1
    assert loaded == ["mean-reversion_indices_long"]


def test_portfolio_explicit_ensembles_bypass_autoload(tmp_path, monkeypatch) -> None:
    from ensemble import vault_manager

    vault_root = tmp_path / "vault"
    _make_vault_tree(vault_root)
    called = {"n": 0}

    def _fake_load(path: str, refit: bool = False, target_volatility: float = 0.20):
        called["n"] += 1
        return {"ensemble_dir": Path(path).name}

    monkeypatch.setattr(vault_manager, "load_ensemble_from_vault", _fake_load)
    portfolio = Portfolio(ensembles=[], vault_root=str(vault_root))

    assert portfolio.ensembles == []
    assert called["n"] == 0
