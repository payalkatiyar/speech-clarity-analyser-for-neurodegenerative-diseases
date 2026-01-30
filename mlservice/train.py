import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mlservice.dataset_loader import SpeechDataset
from mlservice.model import CNN_GRU


TRAIN_DIR = "data/audio/train"
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 5e-5

dataset = SpeechDataset(TRAIN_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = CNN_GRU()
loss_fn = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        preds = model(X)
        preds = torch.clamp(preds, 0.0, 1.0)

        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "cnn_gru_model.pth")
print("✅ Regression model trained and saved")
