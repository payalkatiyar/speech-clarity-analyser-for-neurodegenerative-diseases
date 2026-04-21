"""
Calibration of regression outputs on a validation set (never fit on test).
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_linear_calibration(y_pred: np.ndarray, y_true: np.ndarray):
    """Fit y_true ≈ a * y_pred + b (least squares)."""
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    X = np.column_stack([y_pred, np.ones_like(y_pred)])
    coef, _, _, _ = np.linalg.lstsq(X, y_true, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return a, b


def apply_linear_calibration(y_pred: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.clip(a * np.asarray(y_pred) + b, 0.0, 1.0)


def fit_isotonic_calibration(y_pred: np.ndarray, y_true: np.ndarray):
    """Monotonic map from predictions to targets (reduces systematic bias)."""
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_pred, y_true)
    return iso


def apply_isotonic_calibration(y_pred: np.ndarray, iso: IsotonicRegression) -> np.ndarray:
    out = iso.predict(np.asarray(y_pred, dtype=np.float64).ravel())
    return np.clip(out, 0.0, 1.0)


def save_calibration_bundle(
    out_dir: str,
    linear_a: float,
    linear_b: float,
    iso: Optional[IsotonicRegression],
):
    os.makedirs(out_dir, exist_ok=True)
    linear_path = os.path.join(out_dir, "calibration_linear.json")
    with open(linear_path, "w", encoding="utf-8") as f:
        json.dump({"a": linear_a, "b": linear_b}, f, indent=2)
    if iso is not None:
        iso_path = os.path.join(out_dir, "calibration_isotonic.pkl")
        with open(iso_path, "wb") as f:
            pickle.dump(iso, f)


def load_calibration_bundle(out_dir: str) -> Tuple[Optional[Tuple[float, float]], Optional[IsotonicRegression]]:
    linear_path = os.path.join(out_dir, "calibration_linear.json")
    iso_path = os.path.join(out_dir, "calibration_isotonic.pkl")
    linear = None
    iso = None
    if os.path.exists(linear_path):
        with open(linear_path, "r", encoding="utf-8") as f:
            d = json.load(f)
            linear = (float(d["a"]), float(d["b"]))
    if os.path.exists(iso_path):
        with open(iso_path, "rb") as f:
            iso = pickle.load(f)
    return linear, iso


def apply_best_calibration(y_pred: np.ndarray, out_dir: str) -> np.ndarray:
    """Prefer linear (stable OOD); else isotonic; else identity."""
    linear, iso = load_calibration_bundle(out_dir)
    if linear is not None:
        a, b = linear
        return apply_linear_calibration(y_pred, a, b)
    if iso is not None:
        return apply_isotonic_calibration(y_pred, iso)
    return np.asarray(y_pred, dtype=np.float64)
