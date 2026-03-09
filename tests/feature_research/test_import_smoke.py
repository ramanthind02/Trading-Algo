def test_import_in_sample_packages() -> None:
    import feature_research.in_sample.config as is_config  # noqa: F401
    import feature_research.in_sample.data_loader as is_loader  # noqa: F401
    import feature_research.pipeline as research_pipeline  # noqa: F401

    from feature_research.in_sample.config import ResearchConfig, load_config  # noqa: F401
    from feature_research.pipeline import (  # noqa: F401
        run_eda_pipeline,
        run_validation_pipeline,
        run_oos_pipeline,
    )


def test_import_validation_and_utils_entrypoints() -> None:
    from utils.evaluation.walkforward.runner import run_walkforward_research  # noqa: F401
    from utils.evaluation.walkforward.config import (  # noqa: F401
        WalkforwardResearchConfig,
        WalkforwardSelectionMethod,
        WeightLayerAlgorithm,
    )
