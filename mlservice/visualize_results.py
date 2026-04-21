import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr


RESULTS_DIR = "results"
HISTORY_PATH = "training_history.json"


def load_predictions():
    """Load saved predictions from evaluation."""
    pred_path = os.path.join(RESULTS_DIR, "predictions.npz")
    if os.path.exists(pred_path):
        data = np.load(pred_path)
        return data["y_true"], data["y_pred"]
    else:
        raise FileNotFoundError(
            f"No predictions found at {pred_path}. Run evaluate_regression.py first."
        )


def load_metrics():
    """Load saved metrics from evaluation."""
    metrics_path = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


def load_history():
    """Load training history."""
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    return None


# ================= STYLE CONFIG =================
STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#e6edf3",
    "text.color": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
    "font.size": 11,
}


COLORS = {
    "primary": "#58a6ff",
    "secondary": "#f78166",
    "accent": "#3fb950",
    "purple": "#bc8cff",
    "gold": "#d29922",
    "true_hist": "#58a6ff",
    "pred_hist": "#f78166",
    "scatter": "#58a6ff",
    "line": "#f78166",
    "train": "#58a6ff",
    "val": "#f78166",
}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.rcParams.update(STYLE)

    y_true, y_pred = load_predictions()
    metrics = load_metrics()
    history = load_history()

    mae = mean_absolute_error(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)[0] if np.std(y_pred) > 1e-8 else 0.0

    print(f"MAE: {mae:.4f}")
    print(f"Pearson: {pearson:.4f}")

    # =====================================================
    # 1️⃣ COMPREHENSIVE DASHBOARD (4 subplots)
    # =====================================================
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # --- Scatter: Predicted vs True ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true, y_pred, alpha=0.35, s=20, c=COLORS["scatter"], edgecolors="none")
    ax1.plot([0, 1], [0, 1], "--", color=COLORS["line"], linewidth=2, label="Perfect")
    # Tolerance band
    x_line = np.linspace(0, 1, 100)
    ax1.fill_between(x_line, x_line - 0.15, x_line + 0.15, alpha=0.08, color=COLORS["accent"], label="±0.15 band")
    ax1.set_xlabel("True Clarity Score")
    ax1.set_ylabel("Predicted Clarity Score")
    ax1.set_title("Predicted vs True Clarity", fontweight="bold", fontsize=13)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    # Add metrics text
    textstr = f"r = {pearson:.3f}\nMAE = {mae:.3f}"
    ax1.text(0.97, 0.03, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment="bottom", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262d", edgecolor="#30363d", alpha=0.9))

    # --- Distribution Comparison ---
    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, 1, 35)
    ax2.hist(y_true, bins=bins, alpha=0.6, color=COLORS["true_hist"], label="True", edgecolor="none")
    ax2.hist(y_pred, bins=bins, alpha=0.6, color=COLORS["pred_hist"], label="Predicted", edgecolor="none")
    ax2.set_xlabel("Clarity Score")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Score Distribution Comparison", fontweight="bold", fontsize=13)
    ax2.legend(fontsize=10, framealpha=0.3)
    ax2.grid(True, alpha=0.3)

    # --- Residual Distribution ---
    ax3 = fig.add_subplot(gs[1, 0])
    residuals = y_true - y_pred
    ax3.hist(residuals, bins=40, color=COLORS["purple"], alpha=0.7, edgecolor="none")
    ax3.axvline(0, color=COLORS["line"], linestyle="--", linewidth=1.5)
    ax3.axvline(np.mean(residuals), color=COLORS["accent"], linestyle="-", linewidth=1.5,
                label=f"Mean = {np.mean(residuals):.4f}")
    ax3.set_xlabel("Residual (True - Predicted)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Residual Error Distribution", fontweight="bold", fontsize=13)
    ax3.legend(fontsize=10, framealpha=0.3)
    ax3.grid(True, alpha=0.3)

    # --- Bland-Altman Plot ---
    ax4 = fig.add_subplot(gs[1, 1])
    means = (y_true + y_pred) / 2
    diffs = y_true - y_pred
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    ax4.scatter(means, diffs, alpha=0.35, s=20, c=COLORS["gold"], edgecolors="none")
    ax4.axhline(mean_diff, color=COLORS["accent"], linestyle="-", linewidth=1.5,
                label=f"Mean = {mean_diff:.4f}")
    ax4.axhline(mean_diff + 1.96 * std_diff, color=COLORS["line"], linestyle="--",
                linewidth=1, label=f"+1.96 SD = {mean_diff + 1.96 * std_diff:.4f}")
    ax4.axhline(mean_diff - 1.96 * std_diff, color=COLORS["line"], linestyle="--",
                linewidth=1, label=f"-1.96 SD = {mean_diff - 1.96 * std_diff:.4f}")
    ax4.set_xlabel("Mean of True & Predicted")
    ax4.set_ylabel("Difference (True - Predicted)")
    ax4.set_title("Bland-Altman Plot", fontweight="bold", fontsize=13)
    ax4.legend(fontsize=8, framealpha=0.3, loc="upper right")
    ax4.grid(True, alpha=0.3)

    # Suptitle
    fig.suptitle(
        "Attention-Based CNN-BiGRU · Speech Clarity Estimation in ALS",
        fontsize=16, fontweight="bold", color=COLORS["primary"], y=0.98
    )

    plt.savefig(os.path.join(RESULTS_DIR, "evaluation_dashboard.png"), dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("✅ Saved: evaluation_dashboard.png")

    # =====================================================
    # 2️⃣ TRAINING CURVES
    # =====================================================
    if history:
        fig2, axes = plt.subplots(1, 3, figsize=(18, 5))

        epochs = range(1, len(history["train_loss"]) + 1)

        # Loss curves
        axes[0].plot(epochs, history["train_loss"], color=COLORS["train"], linewidth=1.5, label="Train Loss")
        axes[0].plot(epochs, history["val_loss"], color=COLORS["val"], linewidth=1.5, label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss", fontweight="bold")
        axes[0].legend(framealpha=0.3)
        axes[0].grid(True, alpha=0.3)

        # Validation MAE
        axes[1].plot(epochs, history["val_mae"], color=COLORS["accent"], linewidth=1.5)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].set_title("Validation MAE", fontweight="bold")
        axes[1].grid(True, alpha=0.3)
        # Target line
        axes[1].axhline(0.083, color=COLORS["gold"], linestyle="--", linewidth=1, label="Target MAE (0.083)")
        axes[1].legend(framealpha=0.3)

        # Learning Rate
        axes[2].plot(epochs, history["lr"], color=COLORS["purple"], linewidth=1.5)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_title("Learning Rate Schedule", fontweight="bold")
        axes[2].grid(True, alpha=0.3)
        axes[2].ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

        fig2.suptitle(
            "Training Progress · Attention-Based CNN-BiGRU",
            fontsize=14, fontweight="bold", color=COLORS["primary"], y=1.02
        )

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=300, bbox_inches="tight",
                    facecolor="#0d1117")
        plt.close()
        print("✅ Saved: training_curves.png")

    # =====================================================
    # 3️⃣ INDIVIDUAL HIGH-RES PLOTS (for paper/report)
    # =====================================================
    # Scatter
    fig3, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.35, s=20, c=COLORS["scatter"], edgecolors="none")
    ax.plot([0, 1], [0, 1], "--", color=COLORS["line"], linewidth=2)
    x_line = np.linspace(0, 1, 100)
    ax.fill_between(x_line, x_line - 0.15, x_line + 0.15, alpha=0.08, color=COLORS["accent"])
    ax.set_xlabel("True Clarity Score", fontsize=12)
    ax.set_ylabel("Predicted Clarity Score", fontsize=12)
    ax.set_title("Predicted vs True Clarity", fontweight="bold", fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    textstr = f"Pearson r = {pearson:.3f}\nMAE = {mae:.3f}"
    if metrics:
        textstr += f"\nR² = {metrics['r2']:.3f}"
    ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d", edgecolor="#30363d", alpha=0.9))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "scatter_pred_vs_true.png"), dpi=300, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close()
    print("✅ Saved: scatter_pred_vs_true.png")

    # Distribution
    fig4, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 35)
    ax.hist(y_true, bins=bins, alpha=0.6, color=COLORS["true_hist"], label="True", edgecolor="none")
    ax.hist(y_pred, bins=bins, alpha=0.6, color=COLORS["pred_hist"], label="Predicted", edgecolor="none")
    ax.set_xlabel("Clarity Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Clarity Score Distribution Comparison", fontweight="bold", fontsize=14)
    ax.legend(fontsize=11, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "distribution_comparison.png"), dpi=300, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close()
    print("✅ Saved: distribution_comparison.png")

    # Residuals
    fig5, ax = plt.subplots(figsize=(9, 5))
    ax.hist(residuals, bins=40, color=COLORS["purple"], alpha=0.7, edgecolor="none")
    ax.axvline(0, color=COLORS["line"], linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(residuals), color=COLORS["accent"], linestyle="-", linewidth=1.5,
               label=f"Mean = {np.mean(residuals):.4f}")
    ax.set_xlabel("Residual (True - Predicted)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Residual Error Distribution", fontweight="bold", fontsize=14)
    ax.legend(fontsize=11, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "residual_distribution.png"), dpi=300, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close()
    print("✅ Saved: residual_distribution.png")

    # =====================================================
    # 4️⃣ METRICS SUMMARY TABLE IMAGE
    # =====================================================
    if metrics:
        fig6, ax = plt.subplots(figsize=(8, 5))
        ax.axis("off")

        table_data = [
            ["Metric", "Value"],
            ["Mean Absolute Error (MAE)", f"{metrics['mae']:.4f}"],
            ["Mean Squared Error (MSE)", f"{metrics['mse']:.4f}"],
            ["Root MSE (RMSE)", f"{metrics['rmse']:.4f}"],
            ["R² Score", f"{metrics['r2']:.4f}"],
            ["Pearson Correlation (r)", f"{metrics['pearson_r']:.4f}"],
            ["Spearman Correlation (ρ)", f"{metrics['spearman_r']:.4f}"],
        ]
        acc_dict = (
            metrics.get("accuracies_eval")
            or metrics.get("accuracies_calibrated")
            or metrics.get("accuracies", {})
        )
        for tol_label, acc_val in acc_dict.items():
            table_data.append([f"Accuracy ({tol_label} cal.)", f"{acc_val:.1f}%"])

        table = ax.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        # Style the table
        for i in range(len(table_data)):
            for j in range(2):
                cell = table[i, j] if i > 0 else table[0, j]
                if i == 0:
                    cell = table[0, j]
                    cell.set_facecolor(COLORS["primary"])
                    cell.set_text_props(color="white", fontweight="bold")
                else:
                    cell = table[i, j]
                    cell.set_facecolor("#161b22" if i % 2 == 0 else "#0d1117")
                    cell.set_text_props(color="#e6edf3")
                cell.set_edgecolor("#30363d")

        ax.set_title(
            "Final Model Evaluation Metrics\nAttention-Based CNN-BiGRU",
            fontweight="bold", fontsize=14, color=COLORS["primary"], pad=20
        )

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "metrics_table.png"), dpi=300, bbox_inches="tight",
                    facecolor="#0d1117")
        plt.close()
        print("✅ Saved: metrics_table.png")

    print("\n✅ All visualizations saved inside 'results/' folder.")


if __name__ == "__main__":
    main()