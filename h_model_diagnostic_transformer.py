"""Capacity-stress-test model: a cell-token Transformer for h_theta(t, x).

DIAGNOSTIC ONLY -- not part of the main pipeline. h_model.py (the spec-
compliant small MLP) remains the model used by train_h.py and
sample_guided.py. This module exists only to answer one question: does a
much larger, structurally stronger h_theta produce materially better
guided-sampling results, holding the CTMC, reward, dataset, rejection
sampler, and guided jump-rate formula fixed?

Like h_model.py, this model sees only the normalized table X_t/N and
normalized time t/T -- no target row/column margins, no margin-error
features. This isolates architecture/capacity as the only variable being
tested, matching h_model.py's input contract exactly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FourierTimeEmbedding(nn.Module):
    """Sinusoidal time embedding, standard transformer/diffusion-style."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, dim // 2, dtype=torch.float32) / (dim // 2)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: Tensor) -> Tensor:
        # t: (B,) normalized time in [0,1]. Rescaled to 1000*t before mixing
        # with frequencies spanning ~1..1e-4: at the raw [0,1] scale nearly
        # all sine/cosine components barely move across the full time range,
        # so the network gets almost no usable time signal. The 1000x
        # rescale is the standard positional-encoding convention (as used
        # for integer token positions / diffusion timesteps) applied here to
        # normalized time.
        t_scaled = 1000.0 * t
        args = t_scaled.unsqueeze(-1) * self.freqs.unsqueeze(0)  # (B, dim//2)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


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


class HModelTransformer(nn.Module):
    """Cell-token Transformer for h_theta(t, x), used as a capacity ceiling test.

    144 cell tokens (learned integer-count embedding, one embedding vector
    per possible count 0..total_count, plus row/col position embeddings)
    plus a learned [CLS] token. A Fourier time embedding is injected
    additively into every token (broadcast). No target-margin information
    is used anywhere -- same raw-input contract as h_model.py (X_t/N and
    t/T only; the count embedding recovers the integer count from X_t/N).

    Uses a count embedding rather than a scalar Linear(1, d_model)
    projection: a scalar linear projection of a single normalized count
    mostly reduces to "one learned direction scaled by a small number,"
    which is easily dominated by the (larger-magnitude, higher-rank)
    row/col position embeddings once summed -- an integer-count embedding
    gives every count its own independently-learned, non-collinear vector.
    """

    def __init__(
        self,
        m: int,
        n: int,
        total_count: int,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.m = m
        self.n = n
        self.d_model = d_model
        self.total_count = total_count

        self.count_embedding = nn.Embedding(total_count + 1, d_model)
        self.row_pos_emb = nn.Embedding(m, d_model)
        self.col_pos_emb = nn.Embedding(n, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.time_embed = FourierTimeEmbedding(d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

        row_idx = torch.arange(m).unsqueeze(1).expand(m, n).reshape(-1)  # (m*n,)
        col_idx = torch.arange(n).unsqueeze(0).expand(m, n).reshape(-1)  # (m*n,)
        self.register_buffer("row_idx", row_idx)
        self.register_buffer("col_idx", col_idx)

    def forward_logits(self, table_norm: Tensor, time_norm: Tensor) -> Tensor:
        """Compute the raw scalar logit for a batch of (normalized) inputs.

        Args:
            table_norm: (B, m, n), already divided by N.
            time_norm: (B,), already divided by T.
        """
        B = table_norm.shape[0]
        device = table_norm.device

        counts = torch.round(table_norm * self.total_count).long()
        counts = counts.clamp(0, self.total_count)
        cell_tokens = self.count_embedding(counts.reshape(B, self.m * self.n))  # (B, d, d_model)
        cell_tokens = (
            cell_tokens
            + self.row_pos_emb(self.row_idx.to(device)).unsqueeze(0)
            + self.col_pos_emb(self.col_idx.to(device)).unsqueeze(0)
        )

        time_embed = self.time_embed(time_norm).unsqueeze(1)  # (B, 1, d_model)

        cls = self.cls_token.expand(B, 1, self.d_model)
        tokens = torch.cat([cls, cell_tokens], dim=1)  # (B, 1+d, d_model)
        tokens = tokens + time_embed  # broadcast over sequence dim

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_ln(tokens)

        cls_out = tokens[:, 0, :]  # (B, d_model)
        logits = self.head(cls_out).squeeze(-1)  # (B,)
        return logits

    def forward(self, table_norm: Tensor, time_norm: Tensor) -> Tensor:
        """Compute h_theta(t, X_t) = sigmoid(logit) in (0, 1)."""
        return torch.sigmoid(self.forward_logits(table_norm, time_norm))

    def forward_log_h(self, table_norm: Tensor, time_norm: Tensor) -> Tensor:
        """Compute log h_theta(t, X_t) stably via F.logsigmoid(logit)."""
        return F.logsigmoid(self.forward_logits(table_norm, time_norm))
