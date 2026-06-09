"""
Task 3 — Joint Fusion Model
=============================
Combines genre classification and instrument recognition into a single
multi-task deep learning model.

Three fusion strategies are implemented and can be compared:
  1. LateFusion   — independent CNN/CRNN branches, concatenate embeddings
  2. EarlyFusion  — shared convolutional backbone, task-specific heads
  3. MTL_Attention — multi-task learning with cross-attention between tasks

All models accept (B, 1, 128, T) mel spectrograms and output BOTH tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.genre_cnn import GenreCNN, ConvBlock
from models.instrument_crnn import InstrumentCRNN


# ─── Strategy 1: Late Fusion ──────────────────────────────────────────────────

class LateFusionModel(nn.Module):
    """
    Two independent pre-trained branches.
    Embeddings are concatenated and fed to a shared fusion layer,
    then to separate classification heads.

    This preserves each model's specialisation while allowing them
    to inform each other in the final prediction.
    """

    def __init__(self, num_genres=10, num_instruments=11,
                 genre_ckpt=None, instr_ckpt=None, freeze_backbones=False):
        super().__init__()

        self.genre_branch = GenreCNN(num_classes=num_genres)
        self.instr_branch = InstrumentCRNN(num_classes=num_instruments)

        if genre_ckpt:
            self.genre_branch.load_state_dict(torch.load(genre_ckpt, map_location='cpu'))
        if instr_ckpt:
            self.instr_branch.load_state_dict(torch.load(instr_ckpt, map_location='cpu'))

        if freeze_backbones:
            for p in self.genre_branch.parameters():
                p.requires_grad = False
            for p in self.instr_branch.parameters():
                p.requires_grad = False

        # Fusion: 128 + 128 → 256 → task heads
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.genre_head  = nn.Linear(128, num_genres)
        self.instr_head  = nn.Linear(128, num_instruments)

    def forward(self, x):
        genre_emb = self.genre_branch.get_embedding(x)   # (B, 128)
        instr_emb = self.instr_branch.get_embedding(x)   # (B, 128)
        fused = torch.cat([genre_emb, instr_emb], dim=1) # (B, 256)
        fused = self.fusion(fused)                         # (B, 128)
        return self.genre_head(fused), self.instr_head(fused)


# ─── Strategy 2: Early Fusion (Shared Backbone) ───────────────────────────────

class EarlyFusionModel(nn.Module):
    """
    Shared convolutional backbone followed by task-specific heads.
    Forces the network to learn a joint representation from the start.
    """

    def __init__(self, num_genres=10, num_instruments=11, dropout=0.4):
        super().__init__()

        # Shared backbone (4 ConvBlocks identical to GenreCNN)
        self.backbone = nn.Sequential(
            ConvBlock(1,   32,  pool_size=(2, 2), drop=0.1),
            ConvBlock(32,  64,  pool_size=(2, 2), drop=0.2),
            ConvBlock(64,  128, pool_size=(2, 2), drop=0.25),
            ConvBlock(128, 256, pool_size=(2, 2), drop=0.25),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Shared intermediate layer
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Task-specific branches
        self.genre_branch = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_genres)
        )
        self.instr_branch = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_instruments)
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.gap(feat)
        shared = self.shared_fc(feat)
        return self.genre_branch(shared), self.instr_branch(shared)


# ─── Strategy 3: MTL with Cross-Task Attention ────────────────────────────────

class CrossTaskAttention(nn.Module):
    """Scaled dot-product attention between two task embeddings."""

    def __init__(self, embed_dim=128):
        super().__init__()
        self.scale = embed_dim ** -0.5
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key_value):
        """query attends to key_value."""
        Q = self.Wq(query)
        K = self.Wk(key_value)
        V = self.Wv(key_value)
        # For batch vectors (B, D) treat each as a 1-step sequence
        attn = torch.sigmoid(torch.sum(Q * K, dim=-1, keepdim=True) * self.scale)
        out = attn * V
        return self.out_proj(out)


class MTLAttentionModel(nn.Module):
    """
    Multi-task model where genre and instrument branches attend to each other.
    Genre classifier can attend to instrument features and vice versa,
    capturing the prior knowledge that instruments correlate with genres.
    """

    def __init__(self, num_genres=10, num_instruments=11, dropout=0.4):
        super().__init__()

        # Independent feature extractors
        self.genre_cnn  = GenreCNN(num_classes=num_genres)
        self.instr_crnn = InstrumentCRNN(num_classes=num_instruments)

        # Cross-task attention modules
        self.genre_attends_to_instr = CrossTaskAttention(128)
        self.instr_attends_to_genre = CrossTaskAttention(128)

        # Final classification heads (after attention)
        self.genre_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_genres)
        )
        self.instr_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_instruments)
        )

    def forward(self, x):
        genre_emb = self.genre_cnn.get_embedding(x)   # (B, 128)
        instr_emb = self.instr_crnn.get_embedding(x)  # (B, 128)

        # Cross-task information exchange
        genre_enriched = genre_emb + self.genre_attends_to_instr(genre_emb, instr_emb)
        instr_enriched = instr_emb + self.instr_attends_to_genre(instr_emb, genre_emb)

        return self.genre_head(genre_enriched), self.instr_head(instr_enriched)


# ─── Multi-task Loss ──────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Learnable multi-task loss weighting (Kendall et al., 2018).
    log σ² terms act as learned task uncertainty weights.
    """

    def __init__(self):
        super().__init__()
        # Log variance parameters (one per task)
        self.log_var_genre = nn.Parameter(torch.zeros(1))
        self.log_var_instr = nn.Parameter(torch.zeros(1))

    def forward(self, loss_genre, loss_instr):
        precision_genre = torch.exp(-self.log_var_genre)
        precision_instr = torch.exp(-self.log_var_instr)
        total = (precision_genre * loss_genre + self.log_var_genre +
                 precision_instr * loss_instr + self.log_var_instr)
        return total, precision_genre.item(), precision_instr.item()


# ─── Sanity checks ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    x = torch.randn(4, 1, 128, 130)

    print('=== Late Fusion ===')
    m1 = LateFusionModel()
    g, i = m1(x)
    print(f'  Genre logits : {g.shape}, Instr logits : {i.shape}')

    print('=== Early Fusion ===')
    m2 = EarlyFusionModel()
    g, i = m2(x)
    print(f'  Genre logits : {g.shape}, Instr logits : {i.shape}')

    print('=== MTL Attention ===')
    m3 = MTLAttentionModel()
    g, i = m3(x)
    print(f'  Genre logits : {g.shape}, Instr logits : {i.shape}')

    print('\nAll models OK ✓')
