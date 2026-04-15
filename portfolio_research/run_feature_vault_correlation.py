"""Run feature-research OOS once, then export feature–vault correlation CSV (Power BI).

Loads ``ResearchConfig`` from ``feature_research.config.load_config`` and
``PortfolioResearchConfig`` from ``portfolio_research.config.load_config``.
Correlation export runs only when ``PortfolioResearchConfig.feature_vault_correlation.enabled``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from feature_research.config import load_config as load_feature_research_config
from feature_research.pipeline import run_oos_pipeline_with_bundle
from portfolio_research.config import load_config as load_portfolio_research_config
from portfolio_research.correlation_export import write_vault_correlation_powerbi_long
from portfolio_research.vault_correlation import (
    build_vault_correlation_long_rows,
    resolve_default_vault_root,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run signed-signal OOS walkforward (feature_research.config.load_config) "
            "and write vault vs research correlation CSV under portfolio output_root."
        )
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="INFO logging to stderr.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    portfolio_cfg = load_portfolio_research_config()
    fvc = portfolio_cfg.feature_vault_correlation
    if not fvc.enabled:
        logger.info(
            "feature_vault_correlation is disabled; set "
            "FeatureVaultCorrelationConfig(enabled=True) in portfolio_research.config.load_config()."
        )
        return 0

    research_cfg = load_feature_research_config()
    vault_root: Path = (
        fvc.vault_root if fvc.vault_root is not None else resolve_default_vault_root()
    )
    out_dir = portfolio_cfg.output_root / fvc.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    _report, bundle = run_oos_pipeline_with_bundle(research_cfg)
    if bundle is None:
        logger.warning(
            "OOS correlation bundle is None (e.g. config.oos_window unset); no CSV written."
        )
        return 0

    rows = build_vault_correlation_long_rows(
        combo_signal_target=bundle.combo_signal_target,
        selection_summary_df=bundle.selection_summary_df,
        module_name=bundle.module_name,
        research_timeframe=bundle.eval_tf,
        research_eval_bias_spec=bundle.research_eval_bias_spec,
        target_col=bundle.target_col,
        extended_start=bundle.extended_start,
        extended_end=bundle.extended_end,
        vault_root=vault_root,
    )
    csv_path = write_vault_correlation_powerbi_long(rows, out_dir / "vault_correlation_long")
    logger.info("Wrote %s", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
