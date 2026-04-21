import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Channel attention: learns to re-weight frequency/feature channels."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * w


class ResidualCNNBlock(nn.Module):
    """Conv → BN → GELU → Conv → BN + SE + residual, then pool."""

    def __init__(self, in_ch, out_ch, pool_freq=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SqueezeExcitation(out_ch)

        # 1x1 conv for channel matching in residual
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

        self.pool = nn.MaxPool2d((2, 1)) if pool_freq else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        out = F.gelu(out + identity)
        out = self.pool(out)
        return out


class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-head self-attention over the temporal dimension.
    Allows the model to attend to multiple temporal patterns
    (pauses, tremor bursts, articulation transitions).

    Uses a learned pooling query to aggregate the attended sequence
    into a single vector.
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Learned pooling: project each time step to a scalar weight
        self.pool_query = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: (batch, time, features)
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.layer_norm(x + attn_out)  # residual + LayerNorm

        # Learned temporal pooling
        # pool_query maps (batch, time, features) → (batch, time, 1)
        weights = torch.softmax(self.pool_query(x), dim=1)  # (batch, time, 1)
        pooled = (x * weights).sum(dim=1)  # (batch, features)

        return pooled, attn_weights


class CNN_BiGRU_Attention(nn.Module):
    """
    Attention-Based CNN-BiGRU for Speech Clarity Estimation in ALS.

    Architecture (stronger capacity for better pred vs. target alignment):
      1. Wider residual CNN + SE → richer spectro-temporal features
      2. 2-layer BiGRU → longer-range temporal context
      3. Multi-head temporal attention
      4. Deeper regression head → [0, 1] sigmoid output
    """

    def __init__(self):
        super().__init__()

        cnn_hidden = 256
        gru_hidden = 128
        gru_layers = 2

        # ---------- Residual CNN (wider) ----------
        self.cnn = nn.Sequential(
            ResidualCNNBlock(1, 64, pool_freq=True),
            nn.Dropout2d(0.10),
            ResidualCNNBlock(64, 128, pool_freq=True),
            nn.Dropout2d(0.15),
            ResidualCNNBlock(128, cnn_hidden, pool_freq=True),
            nn.Dropout2d(0.20),
        )

        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # ---------- Bidirectional GRU (stacked) ----------
        self.gru = nn.GRU(
            input_size=cnn_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if gru_layers > 1 else 0.0,
        )

        gru_out_dim = gru_hidden * 2

        # ---------- Multi-Head Temporal Attention ----------
        self.attention = MultiHeadTemporalAttention(
            embed_dim=gru_out_dim,
            num_heads=16,
            dropout=0.25,
        )

        # ---------- Regression head (Multi-Sample Dropout) ----------
        self.ln = nn.LayerNorm(gru_out_dim)
        self.fc1 = nn.Linear(gru_out_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        
        # Multi-Sample Dropout: uses multiple masks to stabilize gradients
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.4) for _ in range(8)
        ])
        self.inner_dropout = nn.Dropout(0.25)

    def forward(self, x):
        # x: (batch, 1, 120, 200)
        x = self.cnn(x)
        x = self.freq_pool(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru(x)
        attended, _ = self.attention(gru_out)
        
        # Head with Multi-Sample Dropout
        x = self.ln(attended)
        
        # First linear layer
        x = F.gelu(self.fc1(x))
        x = self.inner_dropout(x)
        
        # Second linear layer
        x = F.gelu(self.fc2(x))
        
        # Final layer with multiple dropout paths
        out = 0
        for dropout in self.dropouts:
            out += self.fc3(dropout(x))
        out /= len(self.dropouts)
        
        return torch.sigmoid(out).squeeze(-1)


# Backward-compatible alias
CNN_GRU = CNN_BiGRU_Attention