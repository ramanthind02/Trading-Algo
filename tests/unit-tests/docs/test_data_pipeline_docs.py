from pathlib import Path


def test_data_pipeline_docs_mentions_binning_analysis() -> None:
    content = Path("docs/api/data_pipeline.md").read_text(encoding="utf-8")
    assert "run_binning_analysis.py" in content
