import json
import os

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = "results"
PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "predictions.npz")
METRICS_PATH = os.path.join(RESULTS_DIR, "evaluation_metrics.json")


def _load_artifacts():
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(
            f"Missing {PREDICTIONS_PATH}. Run `python3 -m mlservice.evaluate_regression` first."
        )

    data = np.load(PREDICTIONS_PATH)
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return y_true, y_pred, metrics


def _plot_results(y_true, y_pred, metrics):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) Predicted vs true scatter
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.35, s=12)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5)
    ax.set_title("Predicted vs True Clarity")
    ax.set_xlabel("True Score")
    ax.set_ylabel("Predicted Score")
    ax.grid(alpha=0.25)

    # 2) Error distribution
    ax = axes[0, 1]
    ax.hist(errors, bins=40, alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title("Residual Distribution (Pred - True)")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)

    # 3) Target vs prediction distributions
    ax = axes[1, 0]
    ax.hist(y_true, bins=30, alpha=0.55, label="True")
    ax.hist(y_pred, bins=30, alpha=0.55, label="Predicted")
    ax.set_title("Score Distribution Comparison")
    ax.set_xlabel("Clarity Score")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.25)

    # 4) Absolute error CDF + key tolerances
    ax = axes[1, 1]
    sorted_abs = np.sort(abs_errors)
    cdf = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs)
    ax.plot(sorted_abs, cdf, linewidth=2)
    for tol in [0.05, 0.10, 0.15, 0.20]:
        within = float(np.mean(abs_errors < tol) * 100)
        ax.axvline(tol, linestyle="--", linewidth=1, alpha=0.6)
        ax.text(tol + 0.003, 0.08 + tol, f"{tol:.2f}: {within:.1f}%")
    ax.set_title("Absolute Error CDF")
    ax.set_xlabel("|Prediction Error|")
    ax.set_ylabel("Cumulative Fraction")
    ax.grid(alpha=0.25)

    if metrics:
        fig.suptitle(
            (
                "Model Evaluation\n"
                f"MAE={metrics.get('mae', 0):.4f}  "
                f"RMSE={metrics.get('rmse', 0):.4f}  "
                f"R2={metrics.get('r2', 0):.4f}  "
                f"Pearson={metrics.get('pearson_r', 0):.4f}"
            ),
            fontsize=12,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(RESULTS_DIR, "evaluation_visualizations.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    y_true, y_pred, metrics = _load_artifacts()
    out_path = _plot_results(y_true, y_pred, metrics)
    print(f"Saved visualization: {out_path}")


if __name__ == "__main__":
    main()
