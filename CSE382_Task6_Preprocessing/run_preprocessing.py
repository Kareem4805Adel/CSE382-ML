"""CLI runner to execute Task 6: Data Understanding, EDA, and Task-Specific Preprocessing."""

from __future__ import annotations

from src.data.eda import run_full_eda
from src.data.loader import load_raw_h2
from src.preprocessing.pipeline import prepare_task_datasets


def main() -> None:
    print("1. Loading and verifying H2.csv (City Hotel)...")
    raw_df = load_raw_h2("data/H2.csv", auto_download=True)
    print(f"   Loaded shape: {raw_df.shape}")

    print("\n2. Running Exploratory Data Analysis (saving tables & plots to outputs/eda/)...")
    audit = run_full_eda(raw_df, out_dir="outputs/eda")
    for key, val in audit.items():
        print(f"   - {key}: {val}")

    print("\n3. Building leakage-free preprocessors across tasks & model families...")
    configs = [
        ("regression", "linear"),
        ("regression", "distance"),
        ("regression", "tree"),
        ("classification", "linear"),
        ("classification", "distance"),
        ("classification", "tree"),
        ("clustering", "distance"),
    ]

    for task, family in configs:
        bundle = prepare_task_datasets(
            raw_df,
            task=task,  # type: ignore[arg-type]
            model_or_family=family,
            drop_duplicates=True,
            save_artifact=True,
            artifact_dir="artifacts",
        )
        if bundle.X_val is not None and bundle.X_test is not None:
            print(
                f"   [{task.upper():14s} | {family:8s}] "
                f"Train: {bundle.X_train.shape}, Val: {bundle.X_val.shape}, "
                f"Test: {bundle.X_test.shape} -> {bundle.artifact_path}"
            )
        else:
            print(
                f"   [{task.upper():14s} | {family:8s}] "
                f"Full Unsupervised Matrix: {bundle.X_train.shape} -> {bundle.artifact_path}"
            )

    print("\nDone! EDA outputs saved in 'outputs/eda/' and artifacts saved in 'artifacts/'.")


if __name__ == "__main__":
    main()
