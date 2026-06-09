"""
Task 1b — Instrument Recognition CRNN
=======================================
A Convolutional Recurrent Neural Network (CRNN) that captures both
local spectral patterns (CNN) and temporal dependencies (BiGRU).

Input : (B, 1, 128, T)   — log-mel spectrogram
Output: (B, 11)           — logits for 11 IRMAS instruments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstrumentCRNN(nn.Module):
    """
    CRNN for instrument recognition.

    Architecture:
        3 × (Conv2d → BN → ELU → MaxPool) — local pattern extraction
        Reshape to sequence
        2-layer Bidirectional GRU          — temporal modelling
        FC head
    """

    def __init__(self, num_classes=11, dropout=0.3):
        super().__init__()

        # --- CNN front-end ---
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d((4, 1)),   # pool more aggressively in freq dim
            nn.Dropout2d(0.2),
        )

        # After 3 blocks with pooling (2,2),(2,2),(4,1):
        # freq: 128 → 64 → 32 → 8      → 8 * 128 = 1024 per time step
        self.rnn_input_size = 8 * 128   # freq_bins * channels after CNN

        # --- Recurrent layers ---
        self.bigru = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),   # 256 = 128 * 2 (bidirectional)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, 128, T)
        x = self.cnn(x)            # (B, 128, 8, T)

        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)          # (B, T, C, F)
        x = x.reshape(B, T, C * F)         # (B, T, rnn_input_size)

        x, _ = self.bigru(x)                # (B, T, 256)
        x = x[:, -1, :]                     # last time step (B, 256)

        return self.classifier(x)

    def get_embedding(self, x):
        """Return 128-dim embedding for fusion."""
        x = self.cnn(x)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)
        x, _ = self.bigru(x)
        x = x[:, -1, :]
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        return x    # (B, 128)


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model = InstrumentCRNN(num_classes=11)
    x = torch.randn(4, 1, 128, 130)
    out = model(x)
    print(f'InstrumentCRNN output shape : {out.shape}')  # (4, 11)
    emb = model.get_embedding(x)
    print(f'Embedding shape             : {emb.shape}')  # (4, 128)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters        : {total_params:,}')
