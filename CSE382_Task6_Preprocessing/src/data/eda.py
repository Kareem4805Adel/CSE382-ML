"""Exploratory Data Analysis (EDA), anomaly auditing, and visualization generator for H2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.loader import NUMERIC_SOURCE_COLS

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def compute_quality_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect dimensions, duplicates, missing values, and unusual/invalid domain records."""
    total_guests = (
        df["Adults"].fillna(0) + df["Children"].fillna(0) + df["Babies"].fillna(0)
    )
    total_nights = df["StaysInWeekendNights"].fillna(0) + df["StaysInWeekNights"].fillna(0)

    parking_mask = df["RequiredCarParkingSpaces"] > 0
    parking_cancel = (
        round(float(df.loc[parking_mask, "IsCanceled"].mean()), 4)
        if parking_mask.any()
        else 0.0
    )

    audit: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "exact_duplicate_pct": round(float(df.duplicated().mean() * 100), 2),
        "zero_guest_rows": int((total_guests <= 0).sum()),
        "zero_stay_night_rows": int((total_nights <= 0).sum()),
        "negative_adr_rows": int((df["ADR"] < 0).sum()),
        "zero_adr_rows": int((df["ADR"] == 0).sum()),
        "extreme_adr_gt_1000_rows": int((df["ADR"] > 1000).sum()),
        "max_adr_value": float(df["ADR"].max()),
        "meal_undefined_rows": int((df["Meal"] == "Undefined").sum()),
        "market_segment_undefined_rows": int((df["MarketSegment"] == "Undefined").sum()),
        "dist_channel_undefined_rows": int((df["DistributionChannel"] == "Undefined").sum()),
        "parking_gt0_cancel_rate": parking_cancel,
        "overall_cancel_rate": round(float(df["IsCanceled"].mean()), 4),
    }
    return audit


def generate_summary_tables(df: pd.DataFrame, out_dir: Path) -> Dict[str, pd.DataFrame]:
    """Generate and save numeric summary stats, skewness, IQR outliers, and cardinality tables."""
    out_dir.mkdir(parents=True, exist_ok=True)

    num_df = df[NUMERIC_SOURCE_COLS]
    desc = num_df.describe().T
    desc["skewness"] = num_df.skew()
    q1 = num_df.quantile(0.25)
    q3 = num_df.quantile(0.75)
    iqr = q3 - q1
    outliers = ((num_df < (q1 - 1.5 * iqr)) | (num_df > (q3 + 1.5 * iqr))).sum()
    desc["iqr_outliers"] = outliers
    desc["missing_count"] = num_df.isna().sum()
    desc.to_csv(out_dir / "summary_numeric.csv")

    records = []
    for col in df.columns:
        miss_cnt = int(df[col].isna().sum())
        n_uniq = int(df[col].nunique(dropna=True))
        top_vals = df[col].dropna().value_counts().head(4).index.tolist()
        records.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "role_group": "numeric" if col in NUMERIC_SOURCE_COLS else "categorical/date",
            "missing_count": miss_cnt,
            "missing_pct": round(miss_cnt / len(df) * 100, 3),
            "n_unique": n_uniq,
            "sample_top_values": ", ".join(map(str, top_vals)),
        })
    card_df = pd.DataFrame(records).sort_values(
        by=["missing_count", "n_unique"], ascending=False
    )
    card_df.to_csv(out_dir / "missing_and_cardinality.csv", index=False)

    return {"numeric_summary": desc, "cardinality_summary": card_df}


def generate_eda_plots(df: pd.DataFrame, out_dir: Path) -> None:
    """Create and save exploratory plots for Data Understanding."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    miss_pct = (df.isna().mean() * 100).loc[lambda s: s > 0].sort_values(ascending=True)
    if not miss_pct.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        miss_pct.plot(kind="barh", color="#2b5c8f", ax=ax)
        for idx, val in enumerate(miss_pct):
            ax.text(val + 0.5, idx, f"{val:.2f}%", va="center", fontsize=9)
        ax.set_title("Columns with Missing Values in Raw H2 Dataset (%)")
        ax.set_xlabel("Missing Percentage (%)")
        ax.set_xlim(0, max(105, miss_pct.max() + 12))
        fig.tight_layout()
        fig.savefig(out_dir / "01_missing_values.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cancel_counts = df["IsCanceled"].value_counts().reindex([0, 1], fill_value=0)
    axes[0].bar(["Not Canceled (0)", "Canceled (1)"], cancel_counts.values, color=["#3b8ea5", "#e06d53"])
    axes[0].set_title(f"IsCanceled Balance (Cancel Rate: {df['IsCanceled'].mean()*100:.1f}%)")
    axes[0].set_ylabel("Bookings")

    sns.boxplot(x=df["ADR"], ax=axes[1], color="#f4a261")
    axes[1].set_title("Raw ADR Boxplot (Reveals Extreme Outlier)")

    valid_adr = df.loc[(df["ADR"] > 0) & (df["ADR"] <= 500), "ADR"]
    sns.histplot(valid_adr, bins=50, kde=True, ax=axes[2], color="#2a9d8f")
    axes[2].set_title("Cleaned ADR Distribution (0 < ADR <= 500)")
    fig.tight_layout()
    fig.savefig(out_dir / "02_target_distributions.png", dpi=150)
    plt.close(fig)

    skew_cols = ["LeadTime", "StaysInWeekNights", "DaysInWaitingList", "PreviousCancellations"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.ravel(), skew_cols):
        series = df[col].dropna()
        sns.histplot(series, bins=45, ax=ax, color="#457b9d")
        ax.set_title(f"{col} (Skewness = {series.skew():.2f})")
        ax.set_yscale("log")
        ax.set_ylabel("Count (Log Scale)")
    fig.tight_layout()
    fig.savefig(out_dir / "03_skewed_features.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 9))
    corr = df[NUMERIC_SOURCE_COLS].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
    ax.set_title("Pearson Correlation Matrix (H2 Numerical Variables)")
    fig.tight_layout()
    fig.savefig(out_dir / "04_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    sub_adr = df[(df["ADR"] > 0) & (df["ADR"] <= 1000)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    month_means = sub_adr.groupby("ArrivalDateMonth")["ADR"].mean().reindex(MONTH_ORDER)
    axes[0].plot(MONTH_ORDER, month_means.values, marker="o", linewidth=2, color="#1d3557")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_title("Average ADR Seasonality by Arrival Month")
    axes[0].set_ylabel("Mean ADR")

    sns.boxplot(
        data=sub_adr, x="MarketSegment", y="ADR", ax=axes[1], color="#2a9d8f"
    )
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_title("ADR Distribution by Market Segment")
    fig.tight_layout()
    fig.savefig(out_dir / "05_adr_by_month_and_segment.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    dep_cancel = df.groupby("DepositType")["IsCanceled"].mean().sort_values(ascending=False)
    dep_cancel.plot(kind="bar", color="#e76f51", ax=axes[0])
    axes[0].set_title("Cancellation Rate by DepositType")
    axes[0].set_ylabel("Cancellation Probability")
    axes[0].tick_params(axis="x", rotation=0)

    lead_bins = pd.qcut(df["LeadTime"], q=5, duplicates="drop")
    lead_cancel = df.groupby(lead_bins, observed=False)["IsCanceled"].mean()
    lead_cancel.plot(kind="bar", color="#264653", ax=axes[1])
    axes[1].set_title("Cancellation Rate Across LeadTime Quintiles")
    axes[1].set_ylabel("Cancellation Probability")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_dir / "06_cancellation_by_deposit_and_leadtime.png", dpi=150)
    plt.close(fig)


def run_full_eda(df: pd.DataFrame, out_dir: str | Path = "outputs/eda") -> Dict[str, Any]:
    """Execute complete Data Understanding audit, save CSV/JSON tables, and generate plots."""
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    audit = compute_quality_audit(df)
    with open(target_dir / "data_quality_audit.json", "w", encoding="utf-8") as fp:
        json.dump(audit, fp, indent=2)

    generate_summary_tables(df, target_dir)
    generate_eda_plots(df, target_dir)
    return audit
