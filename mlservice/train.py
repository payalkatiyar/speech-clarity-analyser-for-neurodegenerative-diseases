import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader
import numpy as np
import random
import json
import os

from mlservice.dataset_loader import SpeechDataset
from mlservice.model import CNN_BiGRU_Attention
from mlservice.calibration_utils import fit_linear_calibration, save_calibration_bundle


# ---------------- REPRODUCIBILITY ----------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ---------------- CONFIG ----------------
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else 
    ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"🚀 Using device: {DEVICE}")

TRAIN_DIR = "data/audio/train"
BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 2        # effective batch = 64
EPOCHS = 20                 # target exactly 20
INITIAL_LR = 1.5e-3         # higher LR for wider model
WARMUP_EPOCHS = 3           # longer warmup for stability
PATIENCE = 6                # more patience for larger model
MIXUP_ALPHA = 0.05          # light mixup for diagonal smoothness
TOLERANCE_TARGET = 0.15     # tighter target band
SOFT_TOL_SHARPNESS = 50.0
LOSS_TOL_WEIGHT = 0.25      # strongly prioritize band accuracy
LOSS_VAR_WEIGHT = 0.12      # maintain spread
LOSS_PEARSON_WEIGHT = 0.35  # primary driver for diagonal fit
LOSS_MEAN_WEIGHT = 0.05     # centering
# Maximize composite = ACC_WEIGHT * val_acc@tol + PEARSON_WEIGHT * val_pearson
COMPOSITE_ACC_WEIGHT = 0.45
COMPOSITE_PEARSON_WEIGHT = 0.55
RESULTS_DIR = "results"
HISTORY_PATH = "training_history.json"


# ================= MIXUP =================
def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation: blends pairs of samples and labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


# ================= COMBINED LOSS =================
class CombinedRegressionLoss(nn.Module):
    """SmoothL1 + MSE combined loss for better gradient signal."""

    def __init__(self, smooth_weight=0.6, mse_weight=0.4):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.mse = nn.MSELoss()
        self.sw = smooth_weight
        self.mw = mse_weight

    def forward(self, pred, target):
        return self.sw * self.smooth_l1(pred, target) + self.mw * self.mse(pred, target)


def soft_tolerance_loss(pred, target, tol: float, sharpness: float) -> torch.Tensor:
    """Differentiable surrogate: high when |pred-target| < tol."""
    err = torch.abs(pred - target)
    soft = torch.sigmoid(sharpness * (tol - err))
    return 1.0 - soft.mean()


def batch_std_match_loss(pred, target) -> torch.Tensor:
    """Encourage prediction spread to match target spread within a batch."""
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    sp = pred.std(unbiased=False)
    st = target.std(unbiased=False)
    return torch.abs(sp - st)


def batch_pearson_loss(pred, target) -> torch.Tensor:
    """1 - Pearson r on batch (minimize to align pred/target on diagonal)."""
    if pred.numel() < 4:
        return pred.new_tensor(0.0)
    p = pred - pred.mean()
    t = target - target.mean()
    denom = p.std(unbiased=False) * t.std(unbiased=False) + 1e-5
    r = (p * t).mean() / denom
    return (1.0 - r).clamp(min=0.0, max=2.0)


def batch_mean_match_loss(pred, target) -> torch.Tensor:
    """Reduce batch-level bias (helps move cloud toward y=x)."""
    return (pred.mean() - target.mean()) ** 2


def _numpy_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred).ravel()
    true = np.asarray(true).ravel()
    if len(pred) < 3 or np.std(pred) < 1e-9 or np.std(true) < 1e-9:
        return 0.0
    r = np.corrcoef(pred, true)[0, 1]
    if np.isnan(r):
        return 0.0
    return float(np.clip(r, -1.0, 1.0))


# ================= WARMUP + COSINE SCHEDULER =================
class WarmupCosineScheduler:
    """Linear warmup for `warmup_epochs`, then cosine annealing."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


def _speaker_id_from_path(path: str) -> str:
    return os.path.basename(path).split("_")[0]


# ================= MAIN TRAINING =================
def train():
    # ---------------- DATA (speaker-disjoint val: matches test generalization) ----------------
    base_no_aug = SpeechDataset(TRAIN_DIR, augment=False)
    speaker_to_idx = defaultdict(list)
    for i, (path, _) in enumerate(base_no_aug.samples):
        speaker_to_idx[_speaker_id_from_path(path)].append(i)

    speakers = list(speaker_to_idx.keys())
    random.shuffle(speakers)
    n_val_spk = max(1, int(round(0.2 * len(speakers))))
    val_speakers = set(speakers[:n_val_spk])
    train_speakers = set(speakers[n_val_spk:])

    train_indices = [i for sp in train_speakers for i in speaker_to_idx[sp]]
    val_indices = [i for sp in val_speakers for i in speaker_to_idx[sp]]
    random.shuffle(train_indices)

    train_ds_aug = SpeechDataset(TRAIN_DIR, augment=True)
    train_ds_noaug = SpeechDataset(TRAIN_DIR, augment=False)
    train_dataset = torch.utils.data.Subset(train_ds_aug, train_indices)
    val_subset = torch.utils.data.Subset(train_ds_noaug, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(
        f"📊 Speaker split — train spk: {len(train_speakers)}, val spk: {len(val_speakers)} | "
        f"samples train: {len(train_indices)}, val: {len(val_indices)}"
    )

    # ---------------- MODEL ----------------
    model = CNN_BiGRU_Attention().to(DEVICE)

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model: {trainable_params:,} trainable params ({total_params:,} total)")

    loss_fn = CombinedRegressionLoss(smooth_weight=0.6, mse_weight=0.4)
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=5e-4)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, EPOCHS, INITIAL_LR)

    # ---------------- TRAINING LOOP ----------------
    best_composite = -1.0
    best_val_acc = -1.0
    best_val_pearson = -1.0
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "val_mae": [],
        "val_acc_tol": [],
        "val_pearson": [],
        "val_composite": [],
    }

    for epoch in range(EPOCHS):

        # ----- Update LR -----
        current_lr = scheduler.step(epoch)

        # ----- Train -----
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()

        for step, (X, y) in enumerate(train_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            # Mixup augmentation
            if MIXUP_ALPHA > 0:
                X, y = mixup_data(X, y, alpha=MIXUP_ALPHA)

            preds = model(X)
            base = loss_fn(preds, y)
            tol_l = soft_tolerance_loss(
                preds, y, TOLERANCE_TARGET, SOFT_TOL_SHARPNESS
            )
            var_l = batch_std_match_loss(preds, y)
            pear_l = batch_pearson_loss(preds, y)
            mean_l = batch_mean_match_loss(preds, y)
            loss = (
                base
                + LOSS_TOL_WEIGHT * tol_l
                + LOSS_VAR_WEIGHT * var_l
                + LOSS_PEARSON_WEIGHT * pear_l
                + LOSS_MEAN_WEIGHT * mean_l
            ) / GRAD_ACCUM_STEPS

            loss.backward()

            # Gradient accumulation
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * GRAD_ACCUM_STEPS

        avg_train_loss = total_train_loss / len(train_loader)

        # ----- Validate -----
        model.eval()
        total_val_loss = 0.0
        val_preds_all = []
        val_true_all = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds = model(X)
                preds = torch.clamp(preds, 0.0, 1.0)
                loss = loss_fn(preds, y)
                total_val_loss += loss.item()

                val_preds_all.extend(preds.cpu().numpy())
                val_true_all.extend(y.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)

        # Validation MAE
        val_mae = float(np.mean(np.abs(np.array(val_true_all) - np.array(val_preds_all))))
        val_acc_tol = float(
            np.mean(
                np.abs(np.array(val_true_all) - np.array(val_preds_all))
                < TOLERANCE_TARGET
            )
            * 100.0
        )
        val_pearson = _numpy_pearson(
            np.array(val_preds_all), np.array(val_true_all)
        )
        acc_frac = val_acc_tol / 100.0
        composite = (
            COMPOSITE_ACC_WEIGHT * acc_frac
            + COMPOSITE_PEARSON_WEIGHT * max(0.0, val_pearson)
        )

        # Record history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)
        history["val_mae"].append(val_mae)
        history["val_acc_tol"].append(val_acc_tol)
        history["val_pearson"].append(val_pearson)
        history["val_composite"].append(composite)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"MAE: {val_mae:.4f} | "
            f"Acc±{TOLERANCE_TARGET:.2f}: {val_acc_tol:.1f}% | "
            f"ρ_val: {val_pearson:.3f} | "
            f"composite: {composite:.3f} | "
            f"LR: {current_lr:.6f}"
        )

        # ----- Early stopping: maximize acc@tol + Pearson (tighter diagonal + band accuracy) -----
        if composite > best_composite:
            best_composite = composite
            best_val_acc = val_acc_tol
            best_val_pearson = val_pearson
            patience_counter = 0
            torch.save(model.state_dict(), "cnn_gru_model.pth")
            print(
                f"  💾 Best model saved "
                f"(composite={best_composite:.3f}, "
                f"Acc±{TOLERANCE_TARGET:.2f}={val_acc_tol:.1f}%, "
                f"ρ={val_pearson:.3f})"
            )
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(
                    f"\n⏹ Early stopping at epoch {epoch+1} "
                    f"(no composite improvement for {PATIENCE} epochs)"
                )
                break

    history["meta"] = {
        "epochs_max": EPOCHS,
        "early_stopping_patience": PATIENCE,
        "mixup_alpha": MIXUP_ALPHA,
        "tolerance_target": TOLERANCE_TARGET,
        "composite_weights": {
            "acc": COMPOSITE_ACC_WEIGHT,
            "pearson": COMPOSITE_PEARSON_WEIGHT,
        },
        "best_composite": best_composite,
        "best_val_pearson": best_val_pearson,
    }

    # Save training history
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n📈 Training history saved to {HISTORY_PATH}")

    print(
        f"\n✅ Training complete. Best composite={best_composite:.3f} | "
        f"Acc±{TOLERANCE_TARGET:.2f}={best_val_acc:.1f}% | ρ={best_val_pearson:.3f}"
    )

    # ----- Fit calibration on validation (for evaluation on test) -----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        state = torch.load("cnn_gru_model.pth", map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load("cnn_gru_model.pth", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    v_pred, v_true = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(DEVICE)
            p = torch.clamp(model(X), 0.0, 1.0)
            v_pred.extend(p.cpu().numpy())
            v_true.extend(y.numpy())
    v_pred = np.array(v_pred)
    v_true = np.array(v_true)
    a, b = fit_linear_calibration(v_pred, v_true)
    save_calibration_bundle(RESULTS_DIR, a, b, None)
    print(
        f"📐 Linear calibration fit on val (n={len(v_true)}): "
        f"y ≈ {a:.4f} * x + {b:.4f}"
    )


if __name__ == "__main__":
    train()