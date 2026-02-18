import py_compile


def test_binning_analysis_entrypoint_compiles() -> None:
    py_compile.compile(
        "feature_research/continuous_binning/run_binning_analysis.py", doraise=True
    )
