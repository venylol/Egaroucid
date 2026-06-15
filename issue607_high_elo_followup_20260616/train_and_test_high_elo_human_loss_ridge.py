"""Train raw Ridge on recent5 high-Elo rows and test on loss-bucket sample."""

import argparse
import csv
import importlib.util
import json
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SCRIPT_DIR = Path(__file__).resolve().parent
PUBLISH_DIR = SCRIPT_DIR / "reports" / "human_loss_ridge_publish_ready_20260615"
TRAIN_CODE = PUBLISH_DIR / "code" / "train_human_loss_ridge_clean.py"

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reports" / "human_loss_ridge_high_elo_test_20260616"
DEFAULT_RECENT5_LOWLEVEL = SCRIPT_DIR / "reports" / "recent5_lowlevel_match_rates_threads5"
DEFAULT_RECENT5_CHILD = SCRIPT_DIR / "reports" / "recent5_candidate_child_losses_level18" / "candidate_child_losses_level18.csv"
DEFAULT_TEST_LOWLEVELS = [
    SCRIPT_DIR / "reports" / "recent5_selected_lowlevel_elo_ge1750_20260616",
    SCRIPT_DIR / "reports" / "full_supplement_lowlevel_elo_ge1750_20260616",
]
DEFAULT_TEST_CHILDREN = [
    SCRIPT_DIR / "reports" / "recent5_selected_candidate_child_losses_level18_elo_ge1750_20260616" / "candidate_child_losses_level18.csv",
    SCRIPT_DIR / "reports" / "full_supplement_candidate_child_losses_level18_elo_ge1750_20260616" / "candidate_child_losses_level18.csv",
]
DEFAULT_TEST_MANIFEST = SCRIPT_DIR / "reports" / "loss_bucket_testset_recent5_first_elo_ge1750_20260616" / "task_manifest.csv"


def import_clean_module():
    spec = importlib.util.spec_from_file_location("human_loss_clean", TRAIN_CODE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def union_fieldnames(rows: Iterable[dict]) -> List[str]:
    names: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                names.append(key)
    return names


def merge_level_dirs(level_dirs: Sequence[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    task_rows: List[dict] = []
    for level_dir in level_dirs:
        task_rows.extend(read_csv(level_dir / "task_manifest.csv"))
    write_csv(output_dir / "task_manifest.csv", union_fieldnames(task_rows), task_rows)

    for level in range(1, 11):
        rows: List[dict] = []
        for level_dir in level_dirs:
            rows.extend(read_csv(level_dir / f"level{level}_bestmoves.csv"))
        write_csv(output_dir / f"level{level}_bestmoves.csv", union_fieldnames(rows), rows)


def merge_child_loss_csvs(paths: Sequence[Path], output_path: Path) -> None:
    rows: List[dict] = []
    for path in paths:
        rows.extend(read_csv(path))
    write_csv(output_path, union_fieldnames(rows), rows)


def build_package_from_existing(lowlevel_dir: Path, child_loss_csv: Path, output_package: Path) -> None:
    data_dir = output_package / "data"
    low_out = data_dir / "recent5_lowlevel_match_rates_threads5"
    child_out = data_dir / "recent5_candidate_child_losses_level18"
    if low_out.exists():
        shutil.rmtree(low_out)
    if child_out.exists():
        shutil.rmtree(child_out)
    low_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lowlevel_dir / "task_manifest.csv", low_out / "task_manifest.csv")
    for level in range(1, 11):
        shutil.copy2(lowlevel_dir / f"level{level}_bestmoves.csv", low_out / f"level{level}_bestmoves.csv")
    child_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(child_loss_csv, child_out / "candidate_child_losses_level18.csv")


def build_test_package(level_dirs: Sequence[Path], child_loss_csvs: Sequence[Path], output_package: Path) -> None:
    data_dir = output_package / "data"
    low_out = data_dir / "recent5_lowlevel_match_rates_threads5"
    child_out = data_dir / "recent5_candidate_child_losses_level18"
    if low_out.exists():
        shutil.rmtree(low_out)
    if child_out.exists():
        shutil.rmtree(child_out)
    merge_level_dirs(level_dirs, low_out)
    child_out.mkdir(parents=True, exist_ok=True)
    merge_child_loss_csvs(child_loss_csvs, child_out / "candidate_child_losses_level18.csv")


def metrics_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "split": name,
        "count": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "bias_pred_minus_true": float(np.mean(err)),
        "true_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(y_pred)),
        "true_min": float(np.min(y_true)),
        "true_max": float(np.max(y_true)),
        "pred_min": float(np.min(y_pred)),
        "pred_max": float(np.max(y_pred)),
    }


def bucket_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket, group in pred_df.groupby("loss_bucket", sort=False):
        rows.append(metrics_row(str(bucket), group["actual_child_loss"].to_numpy(), group["predicted_child_loss"].to_numpy()))
        rows[-1]["loss_bucket"] = bucket
    return pd.DataFrame(rows)


def plot_actual_vs_pred(pred_df: pd.DataFrame, output_path: Path) -> None:
    x = pred_df["actual_child_loss"].to_numpy(dtype=float)
    y = pred_df["predicted_child_loss"].to_numpy(dtype=float)
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    pad = max(1.0, (hi - lo) * 0.04)
    lo -= pad
    hi += pad

    plt.figure(figsize=(8, 7), dpi=150)
    plt.scatter(x, y, s=8, alpha=0.28, edgecolors="none")
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1.2, linestyle="--", label="y = x")
    plt.xlabel("Actual child loss")
    plt.ylabel("Predicted expected child loss")
    plt.title("High-Elo loss-bucket test: actual vs predicted child loss")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.grid(True, linewidth=0.4, alpha=0.35)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    view = df[list(columns)].copy()
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        vals = []
        for col in view.columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train high-Elo raw Ridge and evaluate on high-Elo loss-bucket test set.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-elo", type=float, default=1750.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean = import_clean_module()
    train_package = args.output_dir / "train_recent5_high_elo_package"
    test_package = args.output_dir / "test_loss_bucket_high_elo_package"
    build_package_from_existing(DEFAULT_RECENT5_LOWLEVEL, DEFAULT_RECENT5_CHILD, train_package)
    build_test_package(DEFAULT_TEST_LOWLEVELS, DEFAULT_TEST_CHILDREN, test_package)

    train_df = clean.build_feature_frame(train_package, cache_path=args.output_dir / "train_feature_frame_cache.pkl", use_cache=False)
    test_df = clean.build_feature_frame(test_package, cache_path=args.output_dir / "test_feature_frame_cache.pkl", use_cache=False)
    train_df = train_df[train_df["target_elo"].astype(float) >= args.min_elo].copy()
    test_df = test_df[test_df["target_elo"].astype(float) >= args.min_elo].copy()

    feature_names = clean.get_feature_specs()["clean_compact"].features
    X_train = train_df[feature_names].astype(float)
    y_train = train_df[clean.TARGET].astype(float).to_numpy()
    X_test = test_df[feature_names].astype(float)
    y_test = test_df[clean.TARGET].astype(float).to_numpy()

    model = clean.make_model(alpha=10.0)
    model.fit(X_train, y_train)
    pred_test = np.clip(model.predict(X_test), -128.0, 128.0)
    pred_train = np.clip(model.predict(X_train), -128.0, 128.0)

    test_manifest = pd.read_csv(DEFAULT_TEST_MANIFEST, usecols=["task_uid", "source_dataset", "loss_bucket"])
    pred_df = test_df[
        ["task_uid", "task_index", "year", "target_elo", "raw_name", "chunk_index", "empties_before", "board_str", clean.TARGET]
    ].copy()
    pred_df = pred_df.rename(columns={clean.TARGET: "actual_child_loss"})
    pred_df["predicted_child_loss"] = pred_test
    pred_df["error_pred_minus_true"] = pred_df["predicted_child_loss"] - pred_df["actual_child_loss"]
    pred_df["abs_error"] = pred_df["error_pred_minus_true"].abs()
    pred_df = pred_df.merge(test_manifest, on="task_uid", how="left", validate="one_to_one")

    overall = pd.DataFrame([
        metrics_row("train_recent5_elo_ge1750", y_train, pred_train),
        metrics_row("test_loss_bucket_elo_ge1750", y_test, pred_test),
    ])
    by_bucket = bucket_metrics(pred_df)

    outputs = args.output_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    overall.to_csv(outputs / "metrics_overall.csv", index=False)
    by_bucket.to_csv(outputs / "metrics_by_loss_bucket.csv", index=False)
    pred_df.to_csv(outputs / "predictions.csv", index=False)
    with (outputs / "human_loss_ridge_high_elo_model.pkl").open("wb") as f:
        pickle.dump({"model": model, "feature_names": feature_names, "min_elo": args.min_elo, "target": clean.TARGET}, f)
    plot_actual_vs_pred(pred_df, outputs / "actual_vs_predicted_child_loss.png")

    report = [
        "# High-Elo Raw Ridge Test",
        "",
        f"- Created: {datetime.now().isoformat(timespec='seconds')}",
        f"- Train rows: {len(train_df):,} recent5 rows with target_elo >= {args.min_elo:g}",
        f"- Test rows: {len(test_df):,} loss-bucket rows with target_elo >= {args.min_elo:g}",
        f"- Feature set: clean_compact ({len(feature_names)} features)",
        "",
        "## Overall Metrics",
        markdown_table(overall, ["split", "count", "mae", "mse", "rmse", "r2", "bias_pred_minus_true", "true_mean", "pred_mean"]),
        "",
        "## Metrics By Actual-Loss Bucket",
        markdown_table(by_bucket, ["loss_bucket", "count", "mae", "mse", "rmse", "r2", "bias_pred_minus_true", "true_mean", "pred_mean"]),
        "",
        "## Plot",
        "",
        "![Actual vs predicted child loss](actual_vs_predicted_child_loss.png)",
    ]
    (outputs / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "min_elo_inclusive": args.min_elo,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_names": feature_names,
        "outputs": {
            "metrics_overall": str(outputs / "metrics_overall.csv"),
            "metrics_by_loss_bucket": str(outputs / "metrics_by_loss_bucket.csv"),
            "predictions": str(outputs / "predictions.csv"),
            "plot": str(outputs / "actual_vs_predicted_child_loss.png"),
            "report": str(outputs / "REPORT.md"),
        },
    }
    (outputs / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "done", **run_config}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
