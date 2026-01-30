import torch
import torch.nn as nn


class CNN_GRU(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )

        self.gru = nn.GRU(
            input_size=32,
            hidden_size=64,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)

        _, h = self.gru(x)
        h = self.dropout(h[-1])

        out = self.fc(h)
        return out.squeeze()
