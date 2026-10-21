"""
Shared feature definitions + the sklearn transformer chain.

Both `train.py` (fitting) and `app.py` (serving) import from here so the
feature contract can never drift between the two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "price"

NUMERIC_FEATURES = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "age",
    "floors",
    "garage",
    "lot_size_sqft",
    "has_basement",
    "has_pool",
    "air_conditioning",
    "furnished",
    "main_road",
    "overall_quality",
    "crime_rate",
    "school_rating",
    "distance_to_cbd",
]

CATEGORICAL_FEATURES = ["neighborhood", "renovation"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DERIVED_FEATURES = ["area_per_bedroom", "is_new_build", "has_amenity_score"]

NEIGHBORHOODS = [
    "Northwood",
    "Riverside",
    "Oakridge",
    "Downtown",
    "Hillcrest",
    "Lakeview",
    "Maplewood",
    "Sunnyvale",
    "Brookfield",
    "Cedar Park",
]

RENOVATION_STATES = ["Not_Renovated", "Partial", "Full"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns. Pure function - safe to pickle inside a pipeline."""
    out = df.copy()
    bedrooms = pd.to_numeric(out["bedrooms"], errors="coerce").clip(lower=1)
    out["area_per_bedroom"] = pd.to_numeric(out["area_sqft"], errors="coerce") / bedrooms
    out["is_new_build"] = (pd.to_numeric(out["age"], errors="coerce") <= 5).astype(int)
    out["has_amenity_score"] = (
        pd.to_numeric(out["has_pool"], errors="coerce").fillna(0)
        + pd.to_numeric(out["has_basement"], errors="coerce").fillna(0)
        + pd.to_numeric(out["air_conditioning"], errors="coerce").fillna(0)
        + pd.to_numeric(out["furnished"], errors="coerce").fillna(0)
    ).astype(int)
    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """sklearn transformer wrapper around :func:`engineer_features`."""

    def fit(self, X, y=None):  # noqa: N803 - sklearn API
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):  # noqa: N803 - sklearn API
        return engineer_features(X)

    def get_feature_names_out(self, input_features=None):
        base = list(self.feature_names_in_) if input_features is None else list(input_features)
        return base + [c for c in DERIVED_FEATURES if c not in base]


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("scale", StandardScaler())])
    categorical = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES + DERIVED_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def feature_order() -> list[str]:
    """Column order the API expects, used to build request frames."""
    return list(FEATURES)


def coerce_frame(payload) -> pd.DataFrame:
    """Build a DataFrame with the exact dtypes the model was trained on.

    ``payload`` may be a single mapping, a sequence of mappings, or a
    DataFrame (extra columns such as ``price`` are dropped).
    """
    if isinstance(payload, pd.DataFrame):
        df = payload.reindex(columns=FEATURES).copy()
    else:
        records = [payload] if isinstance(payload, dict) else list(payload)
        df = pd.DataFrame(
            [{name: rec.get(name) for name in FEATURES} for rec in records]
        )

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("has_basement", "has_pool", "air_conditioning", "furnished", "main_road"):
        df[col] = df[col].fillna(0).astype(int)

    df["bedrooms"] = df["bedrooms"].fillna(3).round().clip(1, 8).astype(int)
    df["floors"] = df["floors"].fillna(2).round().clip(1, 4).astype(int)
    df["garage"] = df["garage"].fillna(1).round().clip(0, 4).astype(int)
    df["overall_quality"] = df["overall_quality"].fillna(5).round().clip(1, 10).astype(int)
    df["school_rating"] = df["school_rating"].fillna(6).round().clip(1, 10).astype(int)
    df["bathrooms"] = df["bathrooms"].fillna(2.0).clip(1, 6).astype(float)
    df["age"] = df["age"].fillna(20).clip(0, 155).astype(int)
    df["area_sqft"] = df["area_sqft"].fillna(1800).clip(100, 12000).astype(int)
    df["lot_size_sqft"] = df["lot_size_sqft"].fillna(7000).clip(500, 60000).astype(int)
    df["crime_rate"] = df["crime_rate"].fillna(0.2).clip(0, 1).astype(float)
    df["distance_to_cbd"] = df["distance_to_cbd"].fillna(8.0).clip(0, 60).astype(float)

    df["neighborhood"] = df["neighborhood"].fillna("Maplewood")
    df["renovation"] = df["renovation"].fillna("Not_Renovated")
    return df


def linear_space(minimum: float, maximum: float, steps: int = 120) -> np.ndarray:
    return np.linspace(minimum, maximum, steps)
