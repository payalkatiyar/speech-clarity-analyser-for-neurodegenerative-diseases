import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

from mlservice.dataset_loader import SpeechDataset
from mlservice.model import CNN_GRU


TEST_DIR = "data/audio/test"
MODEL_PATH = "cnn_gru_model.pth"

dataset = SpeechDataset(TEST_DIR)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

model = CNN_GRU()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for X, y in loader:
        preds = model(X)
        preds = torch.clamp(preds, 0.0, 1.0)

        y_true.extend(y.numpy())
        y_pred.extend(preds.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\n📊 CLARITY REGRESSION RESULTS")
print(f"MAE       : {mean_absolute_error(y_true, y_pred):.4f}")
print(f"MSE       : {mean_squared_error(y_true, y_pred):.4f}")

pearson_r, _ = pearsonr(y_true, y_pred)
spearman_r, _ = spearmanr(y_true, y_pred)

print(f"Pearson r : {pearson_r:.4f}")
print(f"Spearman r: {spearman_r:.4f}")
