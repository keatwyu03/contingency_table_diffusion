"""Cell-token Transformer for h_theta(t, X_t).

Inputs are exactly (X_t, t) -- the current (unnormalized-count-recoverable)
table and the current time. No target row/column margins, no margin
errors, and no S_2(X_t) are ever given to this network; those quantities
live exclusively in the reward R(X_0) = exp(-gamma * S_2(X_0)) used to
build the training targets (see h_dataset.py).

Architecture: 144 cell tokens (learned integer-count embedding + learned
row/column position embeddings + a shared Fourier time embedding, each of
dimension 128), 4 pre-LN transformer encoder layers (4 heads, feedforward
dim 512, GELU, dropout 0.05), mean-pooled over all 144 tokens (no CLS
token), then a 128->128->1 MLP head with SiLU. Output is
sigmoid(logit) in (0, 1); log h is computed stably via logsigmoid(logit).

A count embedding (not a scalar Linear(1, d_model) projection) is used for
cell values: a scalar linear projection of a single normalized count is
easily dominated by the larger-magnitude row/col position embeddings once
summed, causing the model to lose sensitivity to count differences between
tables. An integer-count embedding gives each count 0..total_count its own
independently-learned, non-collinear vector.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

D_MODEL = 128
NUM_LAYERS = 4
NUM_HEADS = 4
FF_DIM = 512
DROPOUT = 0.05
NUM_FOURIER_FREQS = 32


class FourierTimeEmbedding(nn.Module):
    """Fourier features of t in [0,1] with log-spaced frequencies in [1, 1000],
    followed by a small 64 -> 128 -> 128 MLP with SiLU."""

    def __init__(self, d_model: int = D_MODEL, num_freqs: int = NUM_FOURIER_FREQS):
        super().__init__()
        freqs = torch.logspace(
            math.log10(1.0), math.log10(1000.0), steps=num_freqs, dtype=torch.float32
        )
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_freqs, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: Tensor) -> Tensor:
        # t: (B,) normalized time in [0,1]
        args = 2.0 * math.pi * t.unsqueeze(-1) * self.freqs.unsqueeze(0)  # (B, num_freqs)
        phi = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, 2*num_freqs)
        return self.mlp(phi)  # (B, d_model)


class TransformerBlock(nn.Module):
    """Pre-LN transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)
        h = self.ln2(x)
        x = x + self.dropout(self.ff(h))
        return x


class HModel(nn.Module):
    """Cell-token Transformer mapping (X_t, t) -> logit, with h = sigmoid(logit)."""

    def __init__(self, m: int, n: int, total_count: int):
        super().__init__()
        self.m = m
        self.n = n
        self.total_count = total_count

        self.count_embedding = nn.Embedding(total_count + 1, D_MODEL)
        self.row_embedding = nn.Embedding(m, D_MODEL)
        self.column_embedding = nn.Embedding(n, D_MODEL)
        self.time_embedding = FourierTimeEmbedding(D_MODEL)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(D_MODEL, NUM_HEADS, FF_DIM, DROPOUT)
                for _ in range(NUM_LAYERS)
            ]
        )
        self.final_ln = nn.LayerNorm(D_MODEL)

        self.head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, 1),
        )

        row_idx = torch.arange(m).unsqueeze(1).expand(m, n).reshape(-1)  # (m*n,)
        col_idx = torch.arange(n).unsqueeze(0).expand(m, n).reshape(-1)  # (m*n,)
        self.register_buffer("row_idx", row_idx)
        self.register_buffer("col_idx", col_idx)

    def encode(self, table_norm: Tensor, time_norm: Tensor) -> Tensor:
        """E_phi: embed + transformer-encode + pool (X_t, t) -> a D_MODEL vector.

        This is exactly the "encoder" half of forward_logits, split out so it
        can be pretrained standalone (see pretrain_score.py) and so its
        weights can be transferred into a fresh HModel before h-training.
        H_omega (self.head) is deliberately excluded from this method.

        Args:
            table_norm: (B, m, n), already divided by total_count.
            time_norm: (B,), already divided by terminal_time.

        Returns:
            (B, D_MODEL) pooled encoder representation.
        """
        B = table_norm.shape[0]
        device = table_norm.device

        counts = torch.round(table_norm * self.total_count).long()
        counts = counts.clamp(0, self.total_count)
        cell_tokens = self.count_embedding(counts.reshape(B, self.m * self.n))  # (B, d, D_MODEL)
        cell_tokens = (
            cell_tokens
            + self.row_embedding(self.row_idx.to(device)).unsqueeze(0)
            + self.column_embedding(self.col_idx.to(device)).unsqueeze(0)
        )

        time_embed = self.time_embedding(time_norm).unsqueeze(1)  # (B, 1, D_MODEL)
        tokens = cell_tokens + time_embed  # broadcast over the 144 cell tokens

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_ln(tokens)

        pooled = tokens.mean(dim=1)  # (B, D_MODEL) -- mean pool over all 144 cell tokens
        return pooled

    def encoder_state_dict(self) -> dict:
        """State dict of exactly the E_phi submodules (excludes self.head)."""
        encoder_modules = (
            "count_embedding", "row_embedding", "column_embedding",
            "time_embedding", "blocks", "final_ln",
        )
        return {
            k: v for k, v in self.state_dict().items()
            if k.split(".", 1)[0] in encoder_modules
        }

    def load_encoder_state_dict(self, encoder_state: dict) -> None:
        """Load a state dict produced by encoder_state_dict (E_phi weights only)."""
        self.load_state_dict(encoder_state, strict=False)

    def forward_logits(self, table_norm: Tensor, time_norm: Tensor) -> Tensor:
        """Compute the raw scalar logit for a batch of (normalized) inputs.

        Args:
            table_norm: (B, m, n), already divided by total_count.
            time_norm: (B,), already divided by terminal_time.

        Returns:
            (B,) tensor of logits.
        """
        pooled = self.encode(table_norm, time_norm)  # E_phi
        logits = self.head(pooled).squeeze(-1)  # H_omega
        return logits

    def forward(self, table_norm: Tensor, time_norm: Tensor) -> Tensor:
        """Compute h_theta(t, X_t) = sigmoid(logit) in (0, 1)."""
        return torch.sigmoid(self.forward_logits(table_norm, time_norm))

    def forward_log_h(self, table_norm: Tensor, time_norm: Tensor) -> Tensor:
        """Compute log h_theta(t, X_t) stably via F.logsigmoid(logit)."""
        return F.logsigmoid(self.forward_logits(table_norm, time_norm))
