"""
Task 1a — Genre Classification CNN
===================================
A compact 2-D CNN that operates on log-mel spectrograms.
Architecture inspired by VGG-style blocks with batch normalisation.

Input : (B, 1, 128, T)   — log-mel spectrogram
Output: (B, 10)           — logits for 10 GTZAN genres
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → (optional second conv) → MaxPool → Dropout."""

    def __init__(self, in_ch, out_ch, pool_size=(2, 2), drop=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
            nn.Dropout2d(drop),
        )

    def forward(self, x):
        return self.block(x)


class GenreCNN(nn.Module):
    """
    Genre classification CNN.

    Architecture:
        4 × ConvBlock  (32 → 64 → 128 → 256 filters)
        Global Average Pooling
        FC(256) → FC(128) → FC(num_genres)
    """

    def __init__(self, num_classes=10, in_channels=1, dropout_fc=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels,  32,  pool_size=(2, 2), drop=0.1),
            ConvBlock(32,           64,  pool_size=(2, 2), drop=0.2),
            ConvBlock(64,          128,  pool_size=(2, 2), drop=0.25),
            ConvBlock(128,         256,  pool_size=(2, 2), drop=0.25),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)

    def get_embedding(self, x):
        """Return 128-dim embedding (before final classification layer)."""
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier[1](x)   # Linear(256)
        x = self.classifier[2](x)   # ReLU
        x = self.classifier[3](x)   # Dropout
        x = self.classifier[4](x)   # Linear(128)
        return x                    # (B, 128)


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == '__main__':
    model = GenreCNN(num_classes=10)
    x = torch.randn(4, 1, 128, 130)   # batch=4, 3s clip at SR=22050
    out = model(x)
    print(f'GenreCNN output shape : {out.shape}')   # (4, 10)
    emb = model.get_embedding(x)
    print(f'Embedding shape       : {emb.shape}')   # (4, 128)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters  : {total_params:,}')
