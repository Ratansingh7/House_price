"""
House Price Predictor - Flask application.

Routes
------
GET  /                    the single-page UI
GET  /api/meta            feature metadata + model metrics (for building the form)
POST /api/predict         price prediction for one house
POST /api/sensitivity     price curve when one feature is swept
GET  /api/samples         a few real rows from the dataset with predictions
GET  /api/health          liveness + model status
POST /api/retrain         re-runs train.py in a background thread
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from features import FEATURES, coerce_frame

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
META_PATH = os.path.join(MODEL_DIR, "feature_meta.json")
DATA_PATH = os.path.join(ROOT, "data", "houses.csv")

app = Flask(__name__)
app.json.sort_keys = False

_MODEL = None
_METRICS: dict = {}
_FEATURE_META: dict = {}
_RETRAIN_STATE = {"running": False, "log": [], "finished_at": None}


# --------------------------------------------------------------------------- #
# Artifact loading
# --------------------------------------------------------------------------- #
def load_artifacts() -> None:
    global _MODEL, _METRICS, _FEATURE_META
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            "Model not found. Run `python make_dataset.py` then `python train.py`."
        )
    _MODEL = joblib.load(MODEL_PATH)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding="utf-8") as fh:
            _METRICS = json.load(fh)
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as fh:
            _FEATURE_META = json.load(fh)


def metrics() -> dict:
    return _METRICS


# --------------------------------------------------------------------------- #
# Prediction helpers
# --------------------------------------------------------------------------- #
def predict_frame(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(_MODEL.predict(frame), dtype=float).ravel()


def predict_one(payload: dict) -> dict:
    frame = coerce_frame(payload)
    price = float(predict_frame(frame)[0])
    area = float(frame["area_sqft"].iloc[0])
    hood = str(frame["neighborhood"].iloc[0])

    band = metrics().get("prediction_band", {})
    rel_q10 = band.get("rel_q10", -0.12)
    rel_q90 = band.get("rel_q90", 0.12)
    lower = max(1000.0, price * (1 + rel_q10))
    upper = price * (1 + rel_q90)

    hood_stats = metrics().get("neighborhood_stats", {}).get(hood, {})
    hood_mean = hood_stats.get("mean")
    hood_median = hood_stats.get("median")

    r2 = metrics().get("test", {}).get("r2", 0.0)
    confidence = float(np.clip(r2 * 100, 0, 99.9))

    return {
        "price": round(price, -1),
        "lower": round(lower, -1),
        "upper": round(upper, -1),
        "price_per_sqft": round(price / max(1.0, area), 2),
        "confidence": round(confidence, 1),
        "neighborhood": hood,
        "neighborhood_mean": hood_mean,
        "neighborhood_median": hood_median,
        "delta_vs_median": (
            round(price - hood_median, -1) if hood_median is not None else None
        ),
        "model": metrics().get("model", "unknown"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/meta")
def api_meta():
    test = metrics().get("test", {})
    return jsonify(
        {
            "ranges": _FEATURE_META.get("ranges", {}),
            "toggles": _FEATURE_META.get("toggles", {}),
            "choices": _FEATURE_META.get("choices", {}),
            "features": FEATURES,
            "model": metrics().get("model"),
            "trained_at": metrics().get("trained_at"),
            "n_samples": metrics().get("n_samples"),
            "metrics": test,
            "price_stats": metrics().get("price_stats", {}),
            "neighborhood_stats": metrics().get("neighborhood_stats", {}),
            "feature_importance": metrics().get("feature_importance", {}),
            "images": metrics().get("images", {}),
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or request.form.to_dict() or {}
    try:
        return jsonify({"ok": True, "result": predict_one(payload)})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/sensitivity", methods=["POST"])
def api_sensitivity():
    """Sweep one numeric feature and return the resulting price curve."""
    payload = request.get_json(silent=True) or {}
    feature = payload.get("feature", "area_sqft")
    steps = int(payload.get("steps", 24) or 24)
    steps = max(6, min(steps, 60))

    spec = _FEATURE_META.get("ranges", {}).get(feature)
    if not spec:
        return jsonify({"ok": False, "error": f"Unknown numeric feature: {feature}"}), 400

    lo, hi = float(spec["min"]), float(spec["max"])
    if hi < lo:
        lo, hi = hi, lo
    step = float(spec.get("step", 1) or 1)
    n_steps = max(6, min(steps, 60))

    # Build the natural grid at the feature's own granularity, then subsample
    # it evenly so the whole [min, max] range is always covered.
    if step >= 0.1:
        grid = np.round(np.arange(lo, hi + step * 0.5, step), 6)
    else:
        grid = np.round(np.linspace(lo, hi, n_steps), 6)

    if len(grid) > n_steps:
        picks = np.unique(
            np.linspace(0, len(grid) - 1, n_steps).round().astype(int)
        )
        grid = grid[picks]

    base = {k: v for k, v in payload.items() if k not in ("feature", "steps")}
    rows = []
    for value in grid:
        row = dict(base)
        row[feature] = float(value)
        frame = coerce_frame(row)
        price = float(predict_frame(frame)[0])
        rows.append({"x": float(value), "price": round(price, -1)})

    baseline = None
    if feature in base and base[feature] is not None:
        try:
            baseline = float(base[feature])
        except (TypeError, ValueError):
            baseline = None

    return jsonify(
        {
            "ok": True,
            "feature": feature,
            "label": spec.get("label", feature),
            "unit": spec.get("unit", ""),
            "points": rows,
            "baseline": baseline,
        }
    )


@app.route("/api/samples")
def api_samples():
    """A handful of real dataset rows plus the model's prediction for each."""
    if not os.path.exists(DATA_PATH):
        return jsonify({"ok": False, "error": "dataset missing"}), 404
    df = pd.read_csv(DATA_PATH)
    df = df.sample(n=min(6, len(df)), random_state=7)
    frame = coerce_frame(df.to_dict(orient="records"))
    preds = predict_frame(frame)
    out = []
    for (_, row), pred in zip(df.iterrows(), preds):
        out.append(
            {
                "area_sqft": int(row["area_sqft"]),
                "bedrooms": int(row["bedrooms"]),
                "bathrooms": float(row["bathrooms"]),
                "age": int(row["age"]),
                "neighborhood": str(row["neighborhood"]),
                "actual": round(float(row["price"]), -1),
                "predicted": round(float(pred), -1),
            }
        )
    return jsonify({"ok": True, "samples": out})


@app.route("/api/health")
def api_health():
    return jsonify(
        {
            "status": "ok" if _MODEL is not None else "no-model",
            "model_loaded": _MODEL is not None,
            "model": metrics().get("model"),
            "trained_at": metrics().get("trained_at"),
            "retraining": _RETRAIN_STATE["running"],
        }
    )


def _run_retrain() -> None:
    log: list[str] = []
    _RETRAIN_STATE["log"] = log
    try:
        proc = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "train.py")],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.append(line.rstrip())
            log[:] = log[-400:]
        proc.wait()
        if proc.returncode == 0:
            load_artifacts()
            log.append("OK - model reloaded")
    except Exception as exc:  # noqa: BLE001
        log.append(f"FAILED: {exc}")
    finally:
        _RETRAIN_STATE["running"] = False
        _RETRAIN_STATE["finished_at"] = datetime.now().isoformat(timespec="seconds")


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    if _RETRAIN_STATE["running"]:
        return jsonify({"ok": False, "error": "Training already in progress"}), 409
    _RETRAIN_STATE["running"] = True
    _RETRAIN_STATE["finished_at"] = None
    threading.Thread(target=_run_retrain, daemon=True).start()
    return jsonify({"ok": True, "message": "Retraining started"})


@app.route("/api/retrain/status")
def api_retrain_status():
    return jsonify(
        {
            "ok": True,
            "running": _RETRAIN_STATE["running"],
            "finished_at": _RETRAIN_STATE["finished_at"],
            "log": _RETRAIN_STATE["log"][-60:],
        }
    )


if __name__ == "__main__":
    load_artifacts()
    print(f"Model: {metrics().get('model')} | R2 = {metrics().get('test', {}).get('r2')}")
    print("Serving on http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000, threaded=True)
