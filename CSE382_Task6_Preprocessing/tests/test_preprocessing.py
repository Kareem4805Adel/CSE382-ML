"""Automated unit and leakage-integrity tests (works with both unittest and pytest)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.eda import run_full_eda
from src.data.loader import SOURCE_SCHEMA, sanitize_dataframe_types
from src.preprocessing.pipeline import (
    LEAKAGE_AUDIT_LOG,
    load_saved_preprocessor,
    prepare_task_datasets,
)


def make_synthetic_h2(n: int = 300) -> pd.DataFrame:
    """Create a synthetic dataset matching raw H2 quirks (padded strings, NULL, NA, outliers)."""
    rng = np.random.default_rng(42)
    months = ["January", "March", "July", "August", "October", "December"]
    df = pd.DataFrame({
        "IsCanceled": rng.integers(0, 2, size=n).astype(str),
        "LeadTime": rng.integers(0, 350, size=n).astype(str),
        "ArrivalDateYear": rng.choice(["2015", "2016", "2017"], size=n),
        "ArrivalDateMonth": rng.choice(months, size=n),
        "ArrivalDateWeekNumber": rng.integers(1, 53, size=n).astype(str),
        "ArrivalDateDayOfMonth": rng.integers(1, 29, size=n).astype(str),
        "StaysInWeekendNights": rng.integers(0, 4, size=n).astype(str),
        "StaysInWeekNights": rng.integers(1, 7, size=n).astype(str),
        "Adults": rng.choice(["1", "2", "3"], size=n),
        "Children": rng.choice(["0", "1", "NA"], p=[0.85, 0.13, 0.02], size=n),
        "Babies": np.zeros(n, dtype=int).astype(str),
        "Meal": rng.choice(["BB       ", "SC", "HB", "Undefined"], size=n),
        "Country": rng.choice(["PRT", "FRA", "DEU", "GBR", "       NULL"], size=n),
        "MarketSegment": rng.choice(["Online TA", "Offline TA/TO", "Direct", "Groups", "Undefined"], size=n),
        "DistributionChannel": rng.choice(["TA/TO", "Direct", "Corporate"], size=n),
        "IsRepeatedGuest": rng.choice(["0", "1"], p=[0.95, 0.05], size=n),
        "PreviousCancellations": rng.choice(["0", "1", "2"], p=[0.9, 0.08, 0.02], size=n),
        "PreviousBookingsNotCanceled": rng.integers(0, 5, size=n).astype(str),
        "ReservedRoomType": rng.choice(["A               ", "D", "E"], size=n),
        "AssignedRoomType": rng.choice(["A               ", "D", "F"], size=n),
        "BookingChanges": rng.integers(0, 4, size=n).astype(str),
        "DepositType": rng.choice(["No Deposit     ", "Non Refund", "Refundable"], size=n),
        "Agent": rng.choice(["9", "240", "       NULL"], size=n),
        "Company": rng.choice(["       NULL", "40"], p=[0.94, 0.06], size=n),
        "DaysInWaitingList": rng.choice(["0", "15"], p=[0.9, 0.1], size=n),
        "CustomerType": rng.choice(["Transient", "Transient-Party", "Contract"], size=n),
        "ADR": rng.uniform(55.0, 220.0, size=n).round(2).astype(str),
        "RequiredCarParkingSpaces": rng.choice(["0", "1"], p=[0.95, 0.05], size=n),
        "TotalOfSpecialRequests": rng.integers(0, 4, size=n).astype(str),
        "ReservationStatus": rng.choice(["Check-Out", "Canceled"], size=n),
        "ReservationStatusDate": ["2016-08-01"] * n,
    })
    df.loc[0, ["Adults", "Children", "Babies"]] = ["0", "0", "0"]
    df.loc[1, "ADR"] = "5400.0"
    return sanitize_dataframe_types(df)


class TestTask6Preprocessing(unittest.TestCase):
    def test_schema_and_null_sanitization(self) -> None:
        df = make_synthetic_h2(100)
        self.assertEqual(list(df.columns), list(SOURCE_SCHEMA.keys()))
        self.assertEqual(int(df["Meal"].isin(["BB       "]).sum()), 0)
        self.assertGreater(int(df["Agent"].isna().sum()), 0)

    def test_eda_outputs_created(self) -> None:
        df = make_synthetic_h2(150)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audit = run_full_eda(df, out_dir=tmp_path)
            self.assertEqual(audit["n_rows"], 150)
            self.assertTrue((tmp_path / "summary_numeric.csv").exists())
            self.assertTrue((tmp_path / "missing_and_cardinality.csv").exists())
            self.assertTrue((tmp_path / "02_target_distributions.png").exists())

    def test_no_leakage_columns_present(self) -> None:
        df = make_synthetic_h2(250)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for task in ("regression", "classification", "clustering"):
                bundle = prepare_task_datasets(
                    df, task=task, model_or_family="distance", artifact_dir=tmp_path
                )
                forbidden = set(LEAKAGE_AUDIT_LOG[task]["excluded_columns"].keys())
                for col in bundle.feature_names:
                    for leak_col in forbidden:
                        self.assertFalse(
                            col.startswith(leak_col),
                            f"Leakage column {leak_col} found in {col} for {task}!",
                        )

    def test_finite_outputs_and_saved_single_row_inference(self) -> None:
        df = make_synthetic_h2(260)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for family in ("linear", "distance", "tree"):
                bundle = prepare_task_datasets(
                    df, task="classification", model_or_family=family, artifact_dir=tmp_path
                )
                self.assertFalse(np.isnan(bundle.X_train.to_numpy()).any())
                self.assertFalse(np.isinf(bundle.X_train.to_numpy()).any())
                assert bundle.X_val is not None and bundle.X_test is not None
                self.assertFalse(np.isnan(bundle.X_val.to_numpy()).any())
                self.assertFalse(np.isnan(bundle.X_test.to_numpy()).any())

                loaded = load_saved_preprocessor(
                    "classification", model_or_family=family, artifact_dir=tmp_path
                )
                assert bundle.X_test_raw is not None
                single_row = bundle.X_test_raw.iloc[[0]].copy()
                single_row["Country"] = "UNSEEN_ISO"
                pred_features = loaded.transform(single_row)
                self.assertEqual(pred_features.shape, (1, len(bundle.feature_names)))
                self.assertFalse(np.isnan(pred_features).any())


if __name__ == "__main__":
    unittest.main()
