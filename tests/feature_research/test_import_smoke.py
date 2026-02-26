def test_import_in_sample_packages() -> None:
    import feature_research.in_sample.config as is_config  # noqa: F401
    import feature_research.in_sample.data_loader as is_loader  # noqa: F401
    import feature_research.pipeline as research_pipeline  # noqa: F401

    from feature_research.in_sample.config import ResearchConfig, load_config  # noqa: F401
    from feature_research.pipeline import (  # noqa: F401
        run_eda_pipeline,
        run_walkforward_pipeline,
        run_continuous_walkforward_pipeline,
        run_rule_based_walkforward_pipeline,
    )


def test_import_walkforward_entrypoints() -> None:
    from feature_research.walkforward.runner import run_walkforward_research  # noqa: F401
    from feature_research.walkforward.config import (  # noqa: F401
        WalkforwardResearchConfig,
        WalkforwardSelectionMethod,
        WeightLayerAlgorithm,
    )
