"""
Train the house-price model.

Pipeline
--------
1. Load data/houses.csv
2. Feature engineering (drop the redundant year_built, add derived columns)
3. Train / test split
4. Compare candidate regressors with 5-fold cross-validation
5. Refit the winner, evaluate on the hold-out test set
6. Persist the sklearn pipeline + metrics + metadata used by the web app

Artifacts
---------
model/house_price_model.pkl   fitted pipeline
model/metrics.json           scores, feature importances, market stats
model/feature_meta.json      ranges / categories for the UI controls
static/img/actual_vs_pred.png
static/img/residuals.png
static/img/feature_importance.png
"""

import json
import os
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NEIGHBORHOODS,
    RENOVATION_STATES,
    TARGET,
    FeatureEngineer,
    build_preprocessor,
)

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "houses.csv")
MODEL_DIR = os.path.join(ROOT, "model")
IMG_DIR = os.path.join(ROOT, "static", "img")
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
META_PATH = os.path.join(MODEL_DIR, "feature_meta.json")

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise SystemExit(
            "data/houses.csv not found - run `python make_dataset.py` first."
        )
    df = pd.read_csv(DATA_PATH)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise SystemExit(f"Dataset is missing columns: {sorted(missing)}")
    return df


def build_pipeline(regressor) -> Pipeline:
    """Full chain: engineer -> preprocess -> regress."""
    return Pipeline(
        [
            ("engineer", FeatureEngineer()),
            ("prep", build_preprocessor()),
            ("reg", regressor),
        ]
    )


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def candidate_models() -> dict:
    return {
        "RandomForest": build_pipeline(
            RandomForestRegressor(
                n_estimators=320,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
        "GradientBoosting": build_pipeline(
            TransformedTargetRegressor(
                regressor=HistGradientBoostingRegressor(
                    learning_rate=0.06,
                    max_iter=600,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
                func=np.log,
                inverse_func=np.exp,
            )
        ),
        "RidgeLog": build_pipeline(
            TransformedTargetRegressor(
                regressor=Ridge(alpha=1.0),
                func=np.log,
                inverse_func=np.exp,
            )
        ),
    }


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "mape": float(np.mean(np.abs((y - pred) / y)) * 100),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _style_axes(ax):
    ax.set_facecolor("#0f1420")
    ax.tick_params(colors="#8ea0c0", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#243049")
    ax.grid(color="#1d2740", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def save_plots(y_true, y_pred, importances, feature_names) -> None:
    os.makedirs(IMG_DIR, exist_ok=True)

    # 1. Actual vs predicted
    fig, ax = plt.subplots(figsize=(6, 4.4), dpi=140)
    _style_axes(ax)
    lim = max(y_true.max(), y_pred.max()) * 1.02
    ax.scatter(y_true, y_pred, s=9, alpha=0.35, color="#5b8cff", edgecolors="none")
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.6, color="#ff6b6b")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual price ($)", color="#c9d6f0", fontsize=10)
    ax.set_ylabel("Predicted price ($)", color="#c9d6f0", fontsize=10)
    ax.set_title("Actual vs Predicted", color="#eaf1ff", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "actual_vs_pred.png"), facecolor="#0b1020")
    plt.close(fig)

    # 2. Residuals
    resid = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4.4), dpi=140)
    _style_axes(ax)
    ax.scatter(y_pred, resid, s=9, alpha=0.35, color="#48d1a4", edgecolors="none")
    ax.axhline(0, linestyle="--", linewidth=1.4, color="#ff6b6b")
    ax.set_xlabel("Predicted price ($)", color="#c9d6f0", fontsize=10)
    ax.set_ylabel("Residual ($)", color="#c9d6f0", fontsize=10)
    ax.set_title("Residuals", color="#eaf1ff", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "residuals.png"), facecolor="#0b1020")
    plt.close(fig)

    # 3. Feature importance
    order = np.argsort(importances)[::-1][:14][::-1]
    labels = [feature_names[i] for i in order]
    values = [importances[i] for i in order]
    fig, ax = plt.subplots(figsize=(6, 5.2), dpi=140)
    _style_axes(ax)
    colors = plt.get_cmap("viridis")(np.linspace(0.25, 0.95, len(values)))
    ax.barh(labels, values, color=colors, height=0.68)
    ax.set_xlabel("Importance", color="#c9d6f0", fontsize=10)
    ax.set_title("Top feature importances", color="#eaf1ff", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "feature_importance.png"), facecolor="#0b1020")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Metadata for the UI
# --------------------------------------------------------------------------- #
def build_feature_meta(df: pd.DataFrame) -> dict:
    ranges = {
        "area_sqft": {
            "label": "Living area",
            "unit": "sq ft",
            "min": int(df["area_sqft"].min()),
            "max": int(df["area_sqft"].max()),
            "step": 10,
            "default": 1800,
        },
        "bedrooms": {
            "label": "Bedrooms",
            "unit": "",
            "min": int(df["bedrooms"].min()),
            "max": int(df["bedrooms"].max()),
            "step": 1,
            "default": 3,
        },
        "bathrooms": {
            "label": "Bathrooms",
            "unit": "",
            "min": float(df["bathrooms"].min()),
            "max": float(df["bathrooms"].max()),
            "step": 0.5,
            "default": 2.0,
        },
        "age": {
            "label": "Age of house",
            "unit": "yrs",
            "min": 0,
            "max": int(df["age"].max()),
            "step": 1,
            "default": 20,
        },
        "floors": {
            "label": "Floors",
            "unit": "",
            "min": 1,
            "max": 4,
            "step": 1,
            "default": 2,
        },
        "garage": {
            "label": "Garage spaces",
            "unit": "",
            "min": 0,
            "max": 4,
            "step": 1,
            "default": 1,
        },
        "lot_size_sqft": {
            "label": "Lot size",
            "unit": "sq ft",
            "min": int(df["lot_size_sqft"].min()),
            "max": int(df["lot_size_sqft"].max()),
            "step": 50,
            "default": 7000,
        },
        "overall_quality": {
            "label": "Build quality",
            "unit": "/10",
            "min": 1,
            "max": 10,
            "step": 1,
            "default": 5,
        },
        "crime_rate": {
            "label": "Area crime rate",
            "unit": "0-1",
            "min": 0.0,
            "max": float(df["crime_rate"].max()),
            "step": 0.01,
            "default": 0.2,
        },
        "school_rating": {
            "label": "Schools",
            "unit": "/10",
            "min": 1,
            "max": 10,
            "step": 1,
            "default": 6,
        },
        "distance_to_cbd": {
            "label": "Distance to city centre",
            "unit": "km",
            "min": 0.0,
            "max": float(df["distance_to_cbd"].max()),
            "step": 0.5,
            "default": 8.0,
        },
    }
    toggles = {
        "has_pool": "Swimming pool",
        "has_basement": "Basement",
        "air_conditioning": "Air conditioning",
        "furnished": "Furnished",
        "main_road": "On a main road",
    }
    return {
        "ranges": ranges,
        "toggles": toggles,
        "choices": {
            "neighborhood": NEIGHBORHOODS,
            "renovation": RENOVATION_STATES,
        },
    }


def neighborhood_stats(df: pd.DataFrame) -> dict:
    g = df.groupby("neighborhood")["price"]
    stats = g.agg(["count", "mean", "median", "min", "max"]).round(0)
    return {
        str(k): {
            "count": int(v["count"]),
            "mean": float(v["mean"]),
            "median": float(v["median"]),
            "min": float(v["min"]),
            "max": float(v["max"]),
        }
        for k, v in stats.iterrows()
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    raw = load_data()
    print(f"Dataset: {len(raw):,} rows x {raw.shape[1]} columns\n")

    X = raw[FEATURES]
    y = raw[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("Cross-validating candidates ...")
    results = {}
    for name, model in candidate_models().items():
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        rmse = -scores.mean()
        results[name] = {"cv_rmse": float(rmse), "cv_std": float(scores.std())}
        print(f"  {name:<20} CV-RMSE = ${rmse:,.0f}  (+/- {scores.std():,.0f})")

    best_name = min(results, key=lambda k: results[k]["cv_rmse"])
    print(f"\nWinner: {best_name}\n")

    best = candidate_models()[best_name]
    best.fit(X_train, y_train)

    test_metrics = evaluate(best, X_test, y_test)
    train_metrics = evaluate(best, X_train, y_train)
    print("Test set:")
    for k, v in test_metrics.items():
        print(f"  {k.upper():<5} {v:,.4f}" if k != "r2" else f"  {k.upper():<5} {v:.4f}")

    # Empirical relative-error band -> used for the "likely price range" in the UI.
    test_pred = best.predict(X_test)
    rel_err = (y_test.values - test_pred) / test_pred
    log_rel = np.log(y_test.values) - np.log(np.clip(test_pred, 1, None))
    band = {
        "rel_q10": float(np.percentile(rel_err, 10)),
        "rel_q90": float(np.percentile(rel_err, 90)),
        "rel_p05": float(np.percentile(rel_err, 5)),
        "rel_p95": float(np.percentile(rel_err, 95)),
        "log_std": float(np.std(log_rel)),
    }
    print(
        "80% band: {rel_q10:+.1%} .. {rel_q90:+.1%}".format(
            rel_q10=band["rel_q10"], rel_q90=band["rel_q90"]
        )
    )

    # Permutation importance on the held-out set (model-agnostic)
    perm = permutation_importance(
        best, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    order = np.argsort(perm.importances_mean)[::-1]
    importances = {
        FEATURES[i]: float(perm.importances_mean[i]) for i in order
    }
    top = list(importances.items())[:14]
    save_plots(
        y_test.values,
        best.predict(X_test),
        np.array([importances[f] for f in FEATURES]),
        FEATURES,
    )

    joblib.dump(best, MODEL_PATH, compress=3)
    print(f"\nSaved model  -> {MODEL_PATH}")

    metrics = {
        "model": best_name,
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_samples": int(len(raw)),
        "n_features": len(FEATURES),
        "test": test_metrics,
        "train": train_metrics,
        "cv": results,
        "feature_importance": importances,
        "price_stats": {
            "min": float(y.min()),
            "max": float(y.max()),
            "mean": float(y.mean()),
            "median": float(y.median()),
        },
        "prediction_band": band,
        "neighborhood_stats": neighborhood_stats(raw),
        "images": {
            "actual_vs_pred": "static/img/actual_vs_pred.png",
            "residuals": "static/img/residuals.png",
            "feature_importance": "static/img/feature_importance.png",
        },
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Saved metrics -> {METRICS_PATH}")

    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(build_feature_meta(raw), fh, indent=2)
    print(f"Saved meta    -> {META_PATH}")

    print("\nTop 10 features:")
    for name, val in top[:10]:
        print(f"  {name:<20} {val:,.0f}")


if __name__ == "__main__":
    main()
