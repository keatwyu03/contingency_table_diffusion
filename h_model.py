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

    def __init__(self, d_model: int=D_MODEL, num_freqs: int=NUM_FOURIER_FREQS):
        super().__init__()
        freqs = torch.logspace(math.log10(1.0), math.log10(1000.0), steps=num_freqs, dtype=torch.float32)
        self.register_buffer('freqs', freqs)
        self.mlp = nn.Sequential(nn.Linear(2 * num_freqs, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

    def forward(self, t: Tensor) -> Tensor:
        args = 2.0 * math.pi * t.unsqueeze(-1) * self.freqs.unsqueeze(0)
        phi = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(phi)

class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)
        h = self.ln2(x)
        x = x + self.dropout(self.ff(h))
        return x

class HModel(nn.Module):

    def __init__(self, m: int, n: int, total_count: int):
        super().__init__()
        self.m = m
        self.n = n
        self.total_count = total_count
        self.count_embedding = nn.Embedding(total_count + 1, D_MODEL)
        self.row_embedding = nn.Embedding(m, D_MODEL)
        self.column_embedding = nn.Embedding(n, D_MODEL)
        self.time_embedding = FourierTimeEmbedding(D_MODEL)
        self.blocks = nn.ModuleList([TransformerBlock(D_MODEL, NUM_HEADS, FF_DIM, DROPOUT) for _ in range(NUM_LAYERS)])
        self.final_ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.SiLU(), nn.Linear(D_MODEL, 1))
        row_idx = torch.arange(m).unsqueeze(1).expand(m, n).reshape(-1)
        col_idx = torch.arange(n).unsqueeze(0).expand(m, n).reshape(-1)
        self.register_buffer('row_idx', row_idx)
        self.register_buffer('col_idx', col_idx)

    def forward_logits(self, table_norm: Tensor, time_norm: Tensor) -> Tensor:
        B = table_norm.shape[0]
        device = table_norm.device
        counts = torch.round(table_norm * self.total_count).long()
        counts = counts.clamp(0, self.total_count)
        cell_tokens = self.count_embedding(counts.reshape(B, self.m * self.n))
        cell_tokens = cell_tokens + self.row_embedding(self.row_idx.to(device)).unsqueeze(0) + self.column_embedding(self.col_idx.to(device)).unsqueeze(0)
        time_embed = self.time_embedding(time_norm).unsqueeze(1)
        tokens = cell_tokens + time_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_ln(tokens)
        pooled = tokens.mean(dim=1)
        logits = self.head(pooled).squeeze(-1)
        return logits

    def forward(self, table_norm: Tensor, time_norm: Tensor) -> Tensor:
        return torch.sigmoid(self.forward_logits(table_norm, time_norm))

    def forward_log_h(self, table_norm: Tensor, time_norm: Tensor) -> Tensor:
        return F.logsigmoid(self.forward_logits(table_norm, time_norm))
