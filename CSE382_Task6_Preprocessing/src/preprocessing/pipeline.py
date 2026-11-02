"""Leakage-audited task feature selection, custom transformers, and model-family pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42

TaskType = Literal["regression", "classification", "clustering"]
ModelFamily = Literal["linear", "distance", "tree"]

MODEL_TO_FAMILY: Dict[str, ModelFamily] = {
    "linear": "linear",
    "linear_regression": "linear",
    "ridge": "linear",
    "lasso": "linear",
    "logistic": "linear",
    "logistic_regression": "linear",
    "lda": "linear",
    "distance": "distance",
    "knn": "distance",
    "svm": "distance",
    "svr": "distance",
    "svc": "distance",
    "mlp": "distance",
    "neural_network": "distance",
    "naive_bayes": "distance",
    "pca": "distance",
    "kmeans": "distance",
    "agglomerative": "distance",
    "dbscan": "distance",
    "gmm": "distance",
    "clustering": "distance",
    "tree": "tree",
    "decision_tree": "tree",
    "random_forest": "tree",
    "xgboost": "tree",
}

MONTH_TO_INT: Dict[str, int] = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

LEAKAGE_AUDIT_LOG: Dict[str, Dict[str, object]] = {
    "regression": {
        "target": "ADR",
        "prediction_point": "Initial Booking Creation (Rate Quotation Time)",
        "excluded_columns": {
            "IsCanceled": "Future booking outcome unknown at initial rate quotation.",
            "ReservationStatus": "Final post-stay/post-cancellation status (direct outcome leakage).",
            "ReservationStatusDate": "Timestamp of final status change (temporal outcome leakage).",
            "AssignedRoomType": "Assigned at check-in or operational overbooking (lifecycle leakage).",
            "BookingChanges": "Accumulates post-booking amendments up until check-in/cancellation.",
            "DaysInWaitingList": "Waiting duration occurs after initial PMS entry before confirmation.",
            "RequiredCarParkingSpaces": "Recorded primarily at physical check-in (~0% cancel rate artifact).",
        },
    },
    "classification": {
        "target": "IsCanceled",
        "prediction_point": "Booking Confirmation Time",
        "excluded_columns": {
            "ReservationStatus": "Deterministic 1-to-1 encoding of target IsCanceled (direct leakage).",
            "ReservationStatusDate": "Precedes ArrivalDate iff booking canceled (severe temporal leakage).",
            "AssignedRoomType": "Updated at check-in for arriving guests (indirect lifecycle leakage).",
            "BookingChanges": "Mutates throughout lifecycle after confirmation until arrival/cancellation.",
            "RequiredCarParkingSpaces": "Logged upon physical arrival at reception (indirect check-in leakage).",
        },
    },
    "clustering": {
        "target": None,
        "prediction_point": "Unsupervised Pre-Arrival Behavioral Demand Segmentation",
        "excluded_columns": {
            "IsCanceled": "Supervised outcome variable; clustering must remain genuinely unsupervised.",
            "ReservationStatus": "Encodes final supervised outcome.",
            "ReservationStatusDate": "Post-outcome timestamp.",
            "AssignedRoomType": "Operational post-booking assignment.",
            "BookingChanges": "Post-booking lifecycle counter.",
            "RequiredCarParkingSpaces": "Check-in operational variable.",
            "ArrivalDateYear": "Calendar year index (excluded to discover year-invariant segments).",
            "ArrivalDateWeekNumber": "Redundant with cyclical month encoding.",
            "ArrivalDateDayOfMonth": "Calendar day noise unsuitable for Euclidean clustering.",
            "Country": "High-cardinality nationality code that fragments distance-based clusters.",
        },
    },
}

RAW_INPUT_COLUMNS: Dict[TaskType, List[str]] = {
    "regression": [
        "LeadTime", "ArrivalDateYear", "ArrivalDateMonth", "ArrivalDateWeekNumber",
        "ArrivalDateDayOfMonth", "StaysInWeekendNights", "StaysInWeekNights",
        "Adults", "Children", "Babies", "Meal", "Country", "MarketSegment",
        "DistributionChannel", "IsRepeatedGuest", "PreviousCancellations",
        "PreviousBookingsNotCanceled", "ReservedRoomType", "DepositType",
        "Agent", "Company", "CustomerType", "TotalOfSpecialRequests",
    ],
    "classification": [
        "LeadTime", "ArrivalDateYear", "ArrivalDateMonth", "ArrivalDateWeekNumber",
        "ArrivalDateDayOfMonth", "StaysInWeekendNights", "StaysInWeekNights",
        "Adults", "Children", "Babies", "Meal", "Country", "MarketSegment",
        "DistributionChannel", "IsRepeatedGuest", "PreviousCancellations",
        "PreviousBookingsNotCanceled", "ReservedRoomType", "DepositType",
        "Agent", "Company", "DaysInWaitingList", "CustomerType", "ADR",
        "TotalOfSpecialRequests",
    ],
    "clustering": [
        "LeadTime", "ArrivalDateMonth", "StaysInWeekendNights", "StaysInWeekNights",
        "Adults", "Children", "Babies", "Meal", "MarketSegment",
        "DistributionChannel", "IsRepeatedGuest", "PreviousCancellations",
        "PreviousBookingsNotCanceled", "ReservedRoomType", "DepositType",
        "Agent", "Company", "DaysInWaitingList", "CustomerType", "ADR",
        "TotalOfSpecialRequests",
    ],
}


def clean_valid_records(
    df: pd.DataFrame,
    task: TaskType,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Apply deterministic row-level validity rules before train/val/test splitting."""
    work = df.copy()

    if "Meal" in work.columns:
        work["Meal"] = work["Meal"].replace({"Undefined": "SC"})
    for col in ("MarketSegment", "DistributionChannel"):
        if col in work.columns:
            work[col] = pd.Series(
                np.where(work[col] == "Undefined", np.nan, work[col]),
                index=work.index,
                dtype=object,
            )

    total_guests = (
        work["Adults"].fillna(0) + work["Children"].fillna(0) + work["Babies"].fillna(0)
    )
    work = work.loc[total_guests > 0].copy()

    if drop_duplicates:
        work = work.drop_duplicates().copy()

    if task == "regression":
        total_nights = work["StaysInWeekendNights"].fillna(0) + work["StaysInWeekNights"].fillna(0)
        work = work.loc[
            (work["ADR"] > 0) & (work["ADR"] <= 1000) & (total_nights > 0)
        ].copy()
    else:
        work = work.loc[(work["ADR"] >= 0) & (work["ADR"] <= 1000)].copy()

    return work.reset_index(drop=True)


class BookingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Deterministic, leakage-safe domain feature engineer compatible with single-row inference."""

    def __init__(self, task: TaskType = "classification") -> None:
        self.task = task

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BookingFeatureEngineer":
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        sample_out = self._engineer(X.iloc[: min(5, len(X))])
        self.output_columns_: List[str] = list(sample_out.columns)
        return self

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        required = RAW_INPUT_COLUMNS[self.task]
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required input columns for {self.task}: {missing}")

        df = X[required].copy()

        if "Meal" in df.columns:
            df["Meal"] = df["Meal"].replace({"Undefined": "SC"})
        for col in ("MarketSegment", "DistributionChannel"):
            if col in df.columns:
                df[col] = pd.Series(
                    np.where(df[col] == "Undefined", np.nan, df[col]),
                    index=df.index,
                    dtype=object,
                )

        adults = pd.to_numeric(df["Adults"], errors="coerce").fillna(2.0)
        children = pd.to_numeric(df["Children"], errors="coerce").fillna(0.0)
        babies = pd.to_numeric(df["Babies"], errors="coerce").fillna(0.0)
        wknd = pd.to_numeric(df["StaysInWeekendNights"], errors="coerce").fillna(0.0)
        wkday = pd.to_numeric(df["StaysInWeekNights"], errors="coerce").fillna(0.0)
        prev_c = pd.to_numeric(df["PreviousCancellations"], errors="coerce").fillna(0.0)
        prev_nc = pd.to_numeric(df["PreviousBookingsNotCanceled"], errors="coerce").fillna(0.0)

        total_nights = wknd + wkday
        total_guests = adults + children + babies

        df["TotalStayNights"] = total_nights
        df["TotalGuests"] = total_guests
        df["IsFamily"] = ((children + babies) > 0).astype(float)
        df["WeekendRatio"] = wknd / np.maximum(total_nights, 1.0)
        df["PrevCancelRatio"] = prev_c / np.maximum(prev_c + prev_nc, 1.0)

        df["HasAgent"] = (~df["Agent"].isna() & (df["Agent"].astype(str).str.strip() != "NULL")).astype(float)
        df["HasCompany"] = (~df["Company"].isna() & (df["Company"].astype(str).str.strip() != "NULL")).astype(float)

        month_num = df["ArrivalDateMonth"].map(MONTH_TO_INT).fillna(7).astype(float)
        df["ArrivalMonthSin"] = np.sin(2.0 * np.pi * month_num / 12.0)
        df["ArrivalMonthCos"] = np.cos(2.0 * np.pi * month_num / 12.0)

        drop_cols = ["Agent", "Company"]
        if self.task == "clustering":
            drop_cols.append("ArrivalDateMonth")
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        return df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._engineer(X)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        return np.asarray(self.output_columns_, dtype=object)


class WinsorLogTransformer(BaseEstimator, TransformerMixin):
    """Learns upper quantile bounds strictly on X_train and optionally applies log1p."""

    def __init__(self, upper_quantile: float = 0.995, apply_log: bool = True) -> None:
        self.upper_quantile = upper_quantile
        self.apply_log = apply_log

    def fit(self, X: np.ndarray | pd.DataFrame, y: Optional[np.ndarray] = None) -> "WinsorLogTransformer":
        arr = np.asarray(X, dtype=float)
        self.n_features_in_ = arr.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.upper_bounds_ = np.nanquantile(arr, self.upper_quantile, axis=0)
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        clipped = np.minimum(arr, self.upper_bounds_)
        if self.apply_log:
            clipped = np.log1p(np.maximum(clipped, 0.0))
        return clipped

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


def _get_engineered_column_groups(task: TaskType) -> Tuple[List[str], List[str], List[str], List[str]]:
    skewed_numeric = [
        "LeadTime", "StaysInWeekendNights", "StaysInWeekNights", "TotalStayNights",
        "PreviousCancellations", "PreviousBookingsNotCanceled", "TotalOfSpecialRequests",
    ]
    if task in {"classification", "clustering"}:
        skewed_numeric.extend(["DaysInWaitingList", "ADR"])

    regular_numeric = [
        "Adults", "Children", "Babies", "TotalGuests",
        "WeekendRatio", "PrevCancelRatio", "ArrivalMonthSin", "ArrivalMonthCos",
    ]
    if task in {"regression", "classification"}:
        regular_numeric.extend(["ArrivalDateYear", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth"])

    binary_cols = ["IsRepeatedGuest", "IsFamily", "HasAgent", "HasCompany"]

    categorical_cols = [
        "Meal", "MarketSegment", "DistributionChannel",
        "ReservedRoomType", "DepositType", "CustomerType",
    ]
    if task in {"regression", "classification"}:
        categorical_cols.extend(["ArrivalDateMonth", "Country"])

    return skewed_numeric, regular_numeric, binary_cols, categorical_cols


def resolve_model_family(model_or_family: str) -> ModelFamily:
    key = model_or_family.strip().lower()
    if key not in MODEL_TO_FAMILY:
        raise ValueError(
            f"Unknown model/family '{model_or_family}'. Valid options: {sorted(MODEL_TO_FAMILY.keys())}"
        )
    return MODEL_TO_FAMILY[key]


def build_preprocessor(task: TaskType, model_or_family: str = "distance") -> Pipeline:
    """Construct an unfitted, leakage-safe sklearn Pipeline tailored to task and model family."""
    family = resolve_model_family(model_or_family)
    skewed_cols, regular_cols, binary_cols, cat_cols = _get_engineered_column_groups(task)

    if family in {"linear", "distance"}:
        skewed_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("winsor_log", WinsorLogTransformer(upper_quantile=0.995, apply_log=True)),
            ("scaler", StandardScaler()),
        ])
        regular_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        skewed_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("winsor", WinsorLogTransformer(upper_quantile=0.999, apply_log=False)),
        ])
        regular_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    ohe_drop = "first" if family == "linear" else "if_binary"
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "ohe",
            OneHotEncoder(
                drop=ohe_drop,
                min_frequency=0.01,
                handle_unknown="infrequent_if_exist",
                sparse_output=False,
            ),
        ),
    ])

    col_transformer = ColumnTransformer(
        transformers=[
            ("skew_num", skewed_pipe, skewed_cols),
            ("reg_num", regular_pipe, regular_cols),
            ("bin", binary_pipe, binary_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline([
        ("feature_engineer", BookingFeatureEngineer(task=task)),
        ("column_transformer", col_transformer),
    ])


@dataclass
class PreparedTaskData:
    task: TaskType
    model_family: ModelFamily
    X_train: pd.DataFrame
    X_val: Optional[pd.DataFrame]
    X_test: Optional[pd.DataFrame]
    y_train: Optional[pd.Series]
    y_val: Optional[pd.Series]
    y_test: Optional[pd.Series]
    X_train_raw: pd.DataFrame
    X_val_raw: Optional[pd.DataFrame]
    X_test_raw: Optional[pd.DataFrame]
    preprocessor: Pipeline
    feature_names: List[str]
    artifact_path: Optional[Path] = None


def prepare_task_datasets(
    raw_df: pd.DataFrame,
    task: TaskType,
    model_or_family: str = "distance",
    drop_duplicates: bool = True,
    save_artifact: bool = True,
    artifact_dir: str | Path = "artifacts",
    random_state: int = RANDOM_SEED,
) -> PreparedTaskData:
    """Clean valid records, split 70/15/15, fit strictly on X_train, and save .joblib artifact."""
    family = resolve_model_family(model_or_family)
    clean_df = clean_valid_records(raw_df, task=task, drop_duplicates=drop_duplicates)
    input_cols = RAW_INPUT_COLUMNS[task]

    preprocessor = build_preprocessor(task=task, model_or_family=family)
    saved_path: Optional[Path] = None

    if task == "clustering":
        X_raw = clean_df[input_cols].copy()
        X_mat = preprocessor.fit_transform(X_raw)
        feat_names = list(preprocessor.named_steps["column_transformer"].get_feature_names_out())
        X_proc = pd.DataFrame(X_mat, columns=feat_names, index=X_raw.index)

        if save_artifact:
            out_dir = Path(artifact_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            saved_path = out_dir / f"preprocessor_{task}_{family}.joblib"
            joblib.dump(preprocessor, saved_path)

        return PreparedTaskData(
            task=task,
            model_family=family,
            X_train=X_proc,
            X_val=None,
            X_test=None,
            y_train=None,
            y_val=None,
            y_test=None,
            X_train_raw=X_raw,
            X_val_raw=None,
            X_test_raw=None,
            preprocessor=preprocessor,
            feature_names=feat_names,
            artifact_path=saved_path,
        )

    target_col = "ADR" if task == "regression" else "IsCanceled"
    X_all = clean_df[input_cols].copy()
    y_all = clean_df[target_col].copy()
    if task == "classification":
        y_all = y_all.astype(int)

    stratify_all = y_all if task == "classification" else None

    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_all,
        y_all,
        test_size=0.30,
        random_state=random_state,
        stratify=stratify_all,
    )

    stratify_temp = y_temp if task == "classification" else None

    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=stratify_temp,
    )

    X_train_mat = preprocessor.fit_transform(X_train_raw, y_train)
    X_val_mat = preprocessor.transform(X_val_raw)
    X_test_mat = preprocessor.transform(X_test_raw)

    feat_names = list(preprocessor.named_steps["column_transformer"].get_feature_names_out())

    X_train_df = pd.DataFrame(X_train_mat, columns=feat_names, index=X_train_raw.index)
    X_val_df = pd.DataFrame(X_val_mat, columns=feat_names, index=X_val_raw.index)
    X_test_df = pd.DataFrame(X_test_mat, columns=feat_names, index=X_test_raw.index)

    if save_artifact:
        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_path = out_dir / f"preprocessor_{task}_{family}.joblib"
        joblib.dump(preprocessor, saved_path)

    return PreparedTaskData(
        task=task,
        model_family=family,
        X_train=X_train_df,
        X_val=X_val_df,
        X_test=X_test_df,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        preprocessor=preprocessor,
        feature_names=feat_names,
        artifact_path=saved_path,
    )


def load_saved_preprocessor(
    task: TaskType,
    model_or_family: str = "distance",
    artifact_dir: str | Path = "artifacts",
) -> Pipeline:
    """Load a previously fitted preprocessing artifact for inference without retraining."""
    family = resolve_model_family(model_or_family)
    path = Path(artifact_dir) / f"preprocessor_{task}_{family}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Saved preprocessor not found at {path}. Run preprocessing first.")
    return joblib.load(path)
