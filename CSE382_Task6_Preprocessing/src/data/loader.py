"""Dataset acquisition, schema verification, and raw string sanitization for H2.csv."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

H2_URL = (
    "https://raw.githubusercontent.com/afaq-ahmad/"
    "Hotel-booking-demand-Case-Study/master/H2.csv"
)
EXPECTED_ROWS = 79_330

SOURCE_SCHEMA: Dict[str, str] = {
    "IsCanceled": "binary",
    "LeadTime": "integer",
    "ArrivalDateYear": "integer",
    "ArrivalDateMonth": "categorical",
    "ArrivalDateWeekNumber": "integer",
    "ArrivalDateDayOfMonth": "integer",
    "StaysInWeekendNights": "integer",
    "StaysInWeekNights": "integer",
    "Adults": "integer",
    "Children": "numeric",
    "Babies": "integer",
    "Meal": "categorical",
    "Country": "categorical",
    "MarketSegment": "categorical",
    "DistributionChannel": "categorical",
    "IsRepeatedGuest": "binary",
    "PreviousCancellations": "integer",
    "PreviousBookingsNotCanceled": "integer",
    "ReservedRoomType": "categorical",
    "AssignedRoomType": "categorical",
    "BookingChanges": "integer",
    "DepositType": "categorical",
    "Agent": "categorical_id",
    "Company": "categorical_id",
    "DaysInWaitingList": "integer",
    "CustomerType": "categorical",
    "ADR": "numeric",
    "RequiredCarParkingSpaces": "integer",
    "TotalOfSpecialRequests": "integer",
    "ReservationStatus": "categorical",
    "ReservationStatusDate": "date",
}

NUMERIC_SOURCE_COLS: List[str] = [
    col
    for col, kind in SOURCE_SCHEMA.items()
    if kind in {"binary", "integer", "numeric"}
]

STRING_SOURCE_COLS: List[str] = [
    col
    for col, kind in SOURCE_SCHEMA.items()
    if kind in {"categorical", "categorical_id", "date"}
]

MISSING_TOKENS = {"NULL", "null", "NaN", "nan", "None", "NA", ""}


def download_h2_dataset(csv_path: str | Path = "data/H2.csv", url: str = H2_URL) -> Path:
    """Download H2.csv from the official public mirror if not already cached locally."""
    target = Path(csv_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        urllib.request.urlretrieve(url, target)
    return target


def sanitize_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    """Strip SQL fixed-width whitespace, standardize NULL tokens to np.nan, and cast numerics."""
    clean = df.copy()
    clean.columns = [str(c).strip().replace(" ", "") for c in clean.columns]

    for col in STRING_SOURCE_COLS:
        if col in clean.columns:
            raw_na = clean[col].isna()
            stripped = clean[col].astype(str).str.strip()
            is_missing = raw_na | stripped.isin(MISSING_TOKENS)
            clean[col] = pd.Series(
                np.where(is_missing, np.nan, stripped),
                index=clean.index,
                dtype=object,
            )

    for col in NUMERIC_SOURCE_COLS:
        if col in clean.columns:
            if clean[col].dtype == object or pd.api.types.is_string_dtype(clean[col]):
                stripped_num = clean[col].astype(str).str.strip()
                clean[col] = pd.to_numeric(stripped_num, errors="coerce")
            else:
                clean[col] = pd.to_numeric(clean[col], errors="coerce")

    return clean


def verify_h2_schema(df: pd.DataFrame, check_row_count: bool = True) -> None:
    """Validate exact H2 column names and expected row count."""
    expected_cols = list(SOURCE_SCHEMA.keys())
    missing_cols = [c for c in expected_cols if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected_cols]

    if missing_cols or extra_cols:
        raise ValueError(
            f"H2 schema mismatch. Missing: {missing_cols}, Unexpected: {extra_cols}"
        )
    if check_row_count and len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows in raw H2.csv, found {len(df)}."
        )


def load_raw_h2(
    csv_path: str | Path = "data/H2.csv",
    auto_download: bool = True,
    strict_row_check: bool = True,
) -> pd.DataFrame:
    """Load, sanitize raw whitespace/NULL tokens, and validate H2.csv."""
    path = Path(csv_path)
    if not path.exists():
        if auto_download:
            download_h2_dataset(path)
        else:
            raise FileNotFoundError(f"Dataset not found at {path}")

    raw_df = pd.read_csv(path, keep_default_na=False, dtype=str)
    clean_df = sanitize_dataframe_types(raw_df)
    verify_h2_schema(clean_df, check_row_count=strict_row_check)
    return clean_df
