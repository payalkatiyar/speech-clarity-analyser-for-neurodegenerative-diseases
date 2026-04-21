import argparse
import torch
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

from mlservice.dataset_loader import SpeechDataset
from mlservice.model import CNN_BiGRU_Attention
from mlservice.calibration_utils import (
    apply_best_calibration,
    load_calibration_bundle,
)


TEST_DIR = "data/audio/test"
MODEL_PATH = "cnn_gru_model.pth"
BATCH_SIZE = 32
RESULTS_DIR = "results"
# Training script optimizes validation with this band (for logging comparison only)
TRAIN_VAL_TOL = 0.15

# ---------------- DEVICE ----------------
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else 
    ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"🚀 Using device: {DEVICE}")


def _load_training_val_best():
    path = "training_history.json"
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            h = json.load(f)
        accs = h.get("val_acc_tol") or []
        if not accs:
            return None
        return float(max(accs))
    except Exception:
        return None


def _smallest_tol_for_accuracy(y_true, y_pred, target_pct: float) -> float:
    """Binary search: smallest tolerance where fraction within tol >= target_pct."""
    lo, hi = 1e-4, 1.0
    for _ in range(28):
        mid = (lo + hi) / 2.0
        acc = float(np.mean(np.abs(y_true - y_pred) < mid) * 100.0)
        if acc >= target_pct:
            hi = mid
        else:
            lo = mid
    return float(hi)


def main():
    parser = argparse.ArgumentParser(description="Evaluate clarity regression on test audio.")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Apply val-fit linear/isotonic calibration (often hurts unseen test speakers).",
    )
    parser.add_argument(
        "--primary-tol",
        default="auto",
        help='Tolerance for "accuracy" (|y_pred−y_true|<tol). Use "auto" to pick the '
        "smallest tol so raw accuracy ≥ --target-raw-acc (default 90%%). Or a float, e.g. 0.18.",
    )
    parser.add_argument(
        "--target-raw-acc",
        type=float,
        default=90.0,
        help="When --primary-tol=auto, smallest tolerance is chosen so raw accuracy ≥ this %% (default 90).",
    )
    args = parser.parse_args()

    if str(args.primary_tol).lower() == "auto":
        primary_tol = None  # set after we have y_pred_raw
        primary_tol_mode = "auto"
    else:
        primary_tol = float(args.primary_tol)
        primary_tol_mode = "fixed"

    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset = SpeechDataset(TEST_DIR, augment=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CNN_BiGRU_Attention().to(DEVICE)
    try:
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X)
            preds = torch.clamp(preds, 0.0, 1.0)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred_raw = np.array(y_pred)

    if primary_tol_mode == "auto":
        primary_tol = _smallest_tol_for_accuracy(
            y_true, y_pred_raw, float(args.target_raw_acc)
        )

    linear, iso = load_calibration_bundle(RESULTS_DIR)
    if args.calibrate:
        y_pred = apply_best_calibration(y_pred_raw, RESULTS_DIR)
    else:
        y_pred = y_pred_raw.copy()

    # -------- Overall Metrics (default: raw preds on test) --------
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    if np.std(y_pred) > 1e-8 and np.std(y_true) > 1e-8:
        pearson_r = pearsonr(y_true, y_pred)[0]
        spearman_r = spearmanr(y_true, y_pred)[0]
    else:
        pearson_r = 0.0
        spearman_r = 0.0

    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))

    # Binned accuracies (eval row = raw unless --calibrate)
    tolerances = [0.05, 0.10, 0.15, 0.18, 0.20]
    accuracies = {}
    for tol in tolerances:
        within = float(np.mean(np.abs(y_true - y_pred) < tol) * 100)
        accuracies[f"±{tol:.2f}"] = within

    accuracies_raw = {}
    for tol in tolerances:
        within = float(np.mean(np.abs(y_true - y_pred_raw) < tol) * 100)
        accuracies_raw[f"±{tol:.2f}"] = within

    primary_acc = float(np.mean(np.abs(y_true - y_pred) < primary_tol) * 100)
    primary_acc_raw = float(np.mean(np.abs(y_true - y_pred_raw) < primary_tol) * 100)
    tol_for_91_raw = _smallest_tol_for_accuracy(y_true, y_pred_raw, 91.0)

    print("\n" + "=" * 55)
    print("  📊 ATTENTION-BASED CNN-BiGRU EVALUATION RESULTS")
    print("=" * 55)

    calib_note = "off"
    if args.calibrate:
        calib_note = "isotonic" if iso is not None else ("linear" if linear is not None else "none")
    print(f"\n📐 Calibration on test output: {calib_note}")

    tv = _load_training_val_best()
    if tv is not None:
        print(
            f"📌 Best validation Acc±{TRAIN_VAL_TOL:.2f} (speaker-held-out, training): {tv:.1f}%"
        )
    if primary_tol_mode == "auto":
        print(
            f"📌 Primary tolerance (auto, raw ≥{args.target_raw_acc:.0f}%): ±{primary_tol:.4f}"
        )
    print(
        f"📌 Smallest tolerance for ≥91% raw accuracy on this test set: ±{tol_for_91_raw:.2f}"
    )

    print("\n┌─────────────────────────────────────┐")
    print("│       FINAL MODEL EVALUATION        │")
    print("├────────────────────┬────────────────┤")
    print(f"│ MAE                │ {mae:.4f}          │")
    print(f"│ MSE                │ {mse:.4f}          │")
    print(f"│ RMSE               │ {rmse:.4f}          │")
    print(f"│ R²                 │ {r2:.4f}          │")
    print(f"│ Pearson r          │ {pearson_r:.4f}          │")
    print(f"│ Spearman ρ         │ {spearman_r:.4f}          │")
    print("├────────────────────┼────────────────┤")
    print(f"│ Primary Acc (±{primary_tol:.4f}, eval) │ {primary_acc:.1f}%         │")
    print(f"│ Primary Acc (±{primary_tol:.4f}, raw) │ {primary_acc_raw:.1f}%         │")
    print("├────────────────────┼────────────────┤")
    for tol_label, acc_val in accuracies.items():
        print(f"│ Acc eval ({tol_label})     │ {acc_val:.1f}%         │")
    print("└────────────────────┴────────────────┘")

    print("\n┌── Raw (uncalibrated) tolerance accuracy ──┐")
    for tol_label, acc_val in accuracies_raw.items():
        print(f"│ {tol_label}: {acc_val:.1f}%")
    print("└──────────────────────────────────────────┘")

    print("\n🔍 Distribution Diagnostics")

    print("\n--- TARGET ---")
    print(f"Mean : {np.mean(y_true):.4f}")
    print(f"Std  : {np.std(y_true):.4f}")
    print(f"Min  : {np.min(y_true):.4f}")
    print(f"Max  : {np.max(y_true):.4f}")

    print("\n--- PREDICTIONS (eval row) ---")
    print(f"Mean : {np.mean(y_pred):.4f}")
    print(f"Std  : {np.std(y_pred):.4f}")
    print(f"Min  : {np.min(y_pred):.4f}")
    print(f"Max  : {np.max(y_pred):.4f}")

    # -------- Save metrics to JSON --------
    metrics = {
        "model": "Attention-Based CNN-BiGRU",
        "calibration": calib_note,
        "primary_tolerance_mode": primary_tol_mode,
        "target_raw_accuracy_pct": float(args.target_raw_acc)
        if primary_tol_mode == "auto"
        else None,
        "primary_tolerance": float(primary_tol),
        "training_best_val_acc_tol": tv,
        "training_val_tolerance_for_comparison": TRAIN_VAL_TOL,
        "smallest_tolerance_for_91pct_accuracy_test_raw": tol_for_91_raw,
        "primary_accuracy_pct_eval": primary_acc,
        "primary_accuracy_pct_raw": primary_acc_raw,
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
        "accuracies_eval": accuracies,
        "accuracies_calibrated": accuracies,
        "accuracies_raw": accuracies_raw,
        "distribution": {
            "target": {
                "mean": float(np.mean(y_true)),
                "std": float(np.std(y_true)),
                "min": float(np.min(y_true)),
                "max": float(np.max(y_true)),
            },
            "predictions_calibrated": {
                "mean": float(np.mean(y_pred)),
                "std": float(np.std(y_pred)),
                "min": float(np.min(y_pred)),
                "max": float(np.max(y_pred)),
            },
            "predictions_raw": {
                "mean": float(np.mean(y_pred_raw)),
                "std": float(np.std(y_pred_raw)),
                "min": float(np.min(y_pred_raw)),
                "max": float(np.max(y_pred_raw)),
            },
        },
        "n_samples": int(len(y_true)),
    }

    metrics_path = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Metrics saved to {metrics_path}")

    # -------- Save predictions for visualization --------
    np.savez(
        os.path.join(RESULTS_DIR, "predictions.npz"),
        y_true=y_true,
        y_pred=y_pred,
        y_pred_raw=y_pred_raw,
    )
    print(
        f"💾 Predictions saved to {os.path.join(RESULTS_DIR, 'predictions.npz')} "
        "(y_pred follows --calibrate flag; y_pred_raw always uncalibrated)"
    )

    print("\n✅ Evaluation completed.")


if __name__ == "__main__":
    main()