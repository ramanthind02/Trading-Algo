from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from matplotlib.figure import Figure

from feature_research.walkforward.runner import WalkforwardRunReport


@dataclass(frozen=True)
class WalkforwardArtifactPaths:
    output_dir: Path
    folds_csv: Path
    fold_scores_csv: Path
    selection_summary_csv: Path
    report_json: Path
    walkforward_stability_png: Path
    fold_timeline_png: Path


def _validate_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def resolve_walkforward_output_dir(
    feature_type: str,
    module_name: str,
    root_dir: Path = Path("feature_research/shared_results"),
) -> Path:
    validated_feature_type = _validate_identifier(feature_type, "feature_type")
    validated_module_name = _validate_identifier(module_name, "module_name")
    return Path(root_dir) / validated_feature_type / validated_module_name / "walkforward"


def write_walkforward_artifacts(
    report: WalkforwardRunReport,
    walkforward_stability_figure: Figure,
    fold_timeline_figure: Figure,
    feature_type: str,
    module_name: str,
    root_dir: Path = Path("feature_research/shared_results"),
) -> WalkforwardArtifactPaths:
    validated_feature_type = _validate_identifier(feature_type, "feature_type")
    validated_module_name = _validate_identifier(module_name, "module_name")
    output_dir = resolve_walkforward_output_dir(
        feature_type=validated_feature_type,
        module_name=validated_module_name,
        root_dir=root_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = WalkforwardArtifactPaths(
        output_dir=output_dir,
        folds_csv=output_dir / "folds.csv",
        fold_scores_csv=output_dir / "fold_scores.csv",
        selection_summary_csv=output_dir / "selection_summary.csv",
        report_json=output_dir / "report.json",
        walkforward_stability_png=output_dir / "walkforward_stability.png",
        fold_timeline_png=output_dir / "fold_timeline.png",
    )

    report.folds_df[
        [
            "fold_id",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "train_samples",
            "test_samples",
        ]
    ].to_csv(paths.folds_csv, index=False, lineterminator="\n")
    report.fold_scores_df[
        [
            "fold_id",
            "param_label",
            "raw_objective",
            "smoothed_objective",
            "rank",
            "selected_feature",
        ]
    ].to_csv(paths.fold_scores_csv, index=False, lineterminator="\n")
    report.selection_summary_df[
        [
            "fold_id",
            "selected_feature",
            "selected_raw_objective",
            "selected_smoothed_objective",
            "top_k_features",
        ]
    ].to_csv(paths.selection_summary_csv, index=False, lineterminator="\n")

    walkforward_stability_figure.savefig(paths.walkforward_stability_png)
    fold_timeline_figure.savefig(paths.fold_timeline_png)

    report_payload = {
        "feature_type": validated_feature_type,
        "module_name": validated_module_name,
        "output_dir": str(output_dir),
        "artifact_files": {
            "folds_csv": str(paths.folds_csv),
            "fold_scores_csv": str(paths.fold_scores_csv),
            "selection_summary_csv": str(paths.selection_summary_csv),
            "report_json": str(paths.report_json),
            "walkforward_stability_png": str(paths.walkforward_stability_png),
            "fold_timeline_png": str(paths.fold_timeline_png),
        },
        "row_counts": {
            "folds": int(len(report.folds_df)),
            "fold_scores": int(len(report.fold_scores_df)),
            "selection_summary": int(len(report.selection_summary_df)),
        },
    }
    paths.report_json.write_text(
        json.dumps(report_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        encoding="utf-8",
    )

    return paths
