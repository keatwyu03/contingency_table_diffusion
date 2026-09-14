"""MLP model for h_theta(t, X_t), the soft-terminal-reward predictor.

Input: the *already normalized* table X_t / N (flattened) concatenated with
normalized time t / T (a single scalar). Normalization is performed
upstream (see h_dataset.py) -- this module expects pre-normalized inputs and
does not renormalize.

Architecture (fixed by spec):
    input dim = m*n + 1
    4 hidden layers of width 126, each followed by SiLU
    output dim = 1 (a scalar logit)

h_theta(t, x) = sigmoid(logit) in (0, 1). Use forward_logits(...) to get the
raw logit (for numerically stable log(h) via F.logsigmoid), and forward(...)
to get h itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HModel(nn.Module):
    """MLP mapping (X_t/N, t/T) -> logit, with h = sigmoid(logit)."""

    def __init__(self, m: int, n: int, hidden_width: int = 126, num_hidden_layers: int = 4):
        super().__init__()
        self.m = m
        self.n = n
        self.input_dim = m * n + 1

        layers = []
        in_dim = self.input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_width))
            layers.append(nn.SiLU())
            in_dim = hidden_width
        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 1)

    def forward_logits(self, table_norm: Tensor, time_norm: Tensor) -> Tensor:
        """Compute the raw scalar logit for a batch of (normalized) inputs.

        Args:
            table_norm: (B, m, n) tensor, already divided by N.
            time_norm: (B,) tensor, already divided by T.

        Returns:
            (B,) tensor of logits.
        """
        batch_size = table_norm.shape[0]
        flat_table = table_norm.reshape(batch_size, self.m * self.n)
        time_col = time_norm.reshape(batch_size, 1).to(flat_table.dtype)
        x = torch.cat([flat_table, time_col], dim=1)
        h = self.hidden(x)
        logits = self.output_layer(h).squeeze(-1)
        return logits

    def forward(self, table_norm: Tensor, time_norm: Tensor) -> Tensor:
        """Compute h_theta(t, X_t) = sigmoid(logit) in (0, 1)."""
        return torch.sigmoid(self.forward_logits(table_norm, time_norm))

    def forward_log_h(self, table_norm: Tensor, time_norm: Tensor) -> Tensor:
        """Compute log h_theta(t, X_t) stably via F.logsigmoid(logit)."""
        return F.logsigmoid(self.forward_logits(table_norm, time_norm))
