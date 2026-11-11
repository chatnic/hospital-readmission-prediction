"""Feature definitions and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "readmitted_30d"
ID_COLS = ["encounter_id"]

NUMERIC = [
    "age",
    "length_of_stay_days",
    "charlson_index",
    "prior_admissions_12m",
    "ed_visits_6m",
    "num_medications",
    "abnormal_lab_count",
    "hba1c_last",
    "lives_alone",
    "followup_scheduled",
]
CATEGORICAL = ["sex", "admission_type", "primary_diagnosis", "discharge_disposition", "insurance"]
ENGINEERED = ["hba1c_missing", "utilization_score", "los_extreme", "high_burden_no_followup"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically motivated features. Pure function: returns a copy."""
    out = df.copy()
    out["hba1c_missing"] = out["hba1c_last"].isna().astype(int)
    # Combined recent utilization — the single strongest readmission signal in the literature.
    out["utilization_score"] = np.log1p(out["prior_admissions_12m"]) * 2 + np.log1p(out["ed_visits_6m"])
    # Very short or very long stays both carry extra risk.
    out["los_extreme"] = ((out["length_of_stay_days"] <= 1.5) | (out["length_of_stay_days"] >= 12)).astype(int)
    out["high_burden_no_followup"] = ((out["charlson_index"] >= 3) & (out["followup_scheduled"] == 0)).astype(int)
    return out


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = add_features(df)
    X = df[NUMERIC + ENGINEERED + CATEGORICAL]
    y = df[TARGET].astype(int)
    return X, y


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC + ENGINEERED),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )
