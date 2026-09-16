"""Optional CTMC-pretraining stage for the h-function encoder E_phi.

This does NOT introduce a new noising process: trajectories are generated
with the existing unconditional CTMC (ctmc.simulate_trajectory), the same
sampler used everywhere else in this project. Pretraining only changes how
the encoder used by h_theta is initialized before h-training (see
train_h.py / main.py).

Task: for two times t < s sampled along a trajectory, predict X_s from
(X_t, t, delta) where delta = s - t.

    (X_t, t, delta) -> E_phi -> G_psi -> predicted X_s

E_phi (PretrainEncoder) reuses HModel's own encoder stack (count/row/column
embeddings + transformer blocks + pooling, via HModel.encode) so the
pretrained weights transfer directly into an HModel's identically-shaped
submodules. delta is injected with a second instance of HModel's Fourier
time embedding (added into the token stream alongside t's), since
HModel.encode only takes a single time scalar.

G_psi is a thin per-cell classification head (D_MODEL -> total_count+1
logits per cell, applied to the pooled representation broadcast over m*n
cells) predicting each cell's count as a class in {0, ..., total_count}.
This mirrors HModel's own count_embedding representation of a table (an
integer class per cell) and is the simplest structured objective
compatible with that representation. G_psi is discarded after pretraining;
only E_phi's weights are kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from config import Config
from ctmc import simulate_trajectory, table_at_time
from h_model import D_MODEL, FourierTimeEmbedding, HModel
from table_space import sample_uniform_tables


class PretrainEncoder(nn.Module):
    """E_phi: encodes (X_t, t, delta) into a pooled D_MODEL representation.

    Architecturally identical to HModel's own encoder (same embeddings,
    same transformer blocks, same pooling) plus one extra Fourier embedding
    for delta, so that count_embedding / row_embedding / column_embedding /
    time_embedding / blocks / final_ln can be copied verbatim into an
    HModel via HModel.load_encoder_state_dict.
    """

    def __init__(self, m: int, n: int, total_count: int):
        super().__init__()
        # A plain HModel supplies every E_phi submodule we need (and has the
        # exact state-dict keys HModel.load_encoder_state_dict expects) --
        # its unused `head` (H_omega) is simply never called here.
        self.base = HModel(m, n, total_count)
        self.delta_embedding = FourierTimeEmbedding(D_MODEL)

    def forward(self, table_norm: Tensor, time_norm: Tensor, delta_norm: Tensor) -> Tensor:
        pooled_with_t = self.base.encode(table_norm, time_norm)
        delta_embed = self.delta_embedding(delta_norm)
        return pooled_with_t + delta_embed  # (B, D_MODEL)

    def encoder_state_dict(self) -> dict:
        """E_phi weights in the exact layout HModel.load_encoder_state_dict expects.

        The delta_embedding is intentionally excluded: HModel.encode has no
        delta input, so there is nowhere in HModel for those weights to go.
        """
        return self.base.encoder_state_dict()


class ScorePredictionHead(nn.Module):
    """G_psi: per-cell count classifier over pooled E_phi(X_t, t, delta).

    Broadcasts the single pooled (B, D_MODEL) vector to all m*n cells (each
    cell also gets its row/column embedding so the head can distinguish
    cells) and predicts a class in {0, ..., total_count} per cell.
    """

    def __init__(self, m: int, n: int, total_count: int):
        super().__init__()
        self.m = m
        self.n = n
        self.total_count = total_count
        self.row_embedding = nn.Embedding(m, D_MODEL)
        self.column_embedding = nn.Embedding(n, D_MODEL)
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, total_count + 1),
        )
        row_idx = torch.arange(m).unsqueeze(1).expand(m, n).reshape(-1)
        col_idx = torch.arange(n).unsqueeze(0).expand(m, n).reshape(-1)
        self.register_buffer("row_idx", row_idx)
        self.register_buffer("col_idx", col_idx)

    def forward(self, pooled: Tensor) -> Tensor:
        """Returns (B, m*n, total_count+1) per-cell class logits."""
        device = pooled.device
        cell_pos = (
            self.row_embedding(self.row_idx.to(device))
            + self.column_embedding(self.col_idx.to(device))
        )  # (m*n, D_MODEL)
        tokens = pooled.unsqueeze(1) + cell_pos.unsqueeze(0)  # (B, m*n, D_MODEL)
        return self.mlp(tokens)  # (B, m*n, total_count+1)


@dataclass
class PretrainHistory:
    losses: List[float]


def _sample_t_s_pairs(
    terminal_time: float, batch_n: int, generator: Optional[torch.Generator] = None
) -> Tuple[Tensor, Tensor]:
    """Sample t < s independently ~ Uniform(0, terminal_time) per trajectory."""
    a = torch.rand((batch_n,), generator=generator) * terminal_time
    b = torch.rand((batch_n,), generator=generator) * terminal_time
    t = torch.minimum(a, b)
    s = torch.maximum(a, b)
    # Avoid the (measure-zero) degenerate delta=0 case.
    s = torch.where(s <= t, t + 1e-6, s).clamp(max=terminal_time)
    return t, s


def generate_pretrain_batch(
    cfg: Config, batch_n: int, generator: Optional[torch.Generator] = None
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Simulate ``batch_n`` independent CTMC trajectories and extract one
    (X_t, t, delta, X_s) example from each, via the existing unconditional
    CTMC sampler (ctmc.simulate_trajectory) -- no new noising process.

    Returns:
        x_t: (B, m, n) unnormalized tables at time t.
        t: (B,) normalized in [0, 1].
        delta: (B,) normalized in [0, 1], = (s - t) / terminal_time.
        x_s: (B, m, n) unnormalized tables at time s -- the target.
    """
    x0_batch = sample_uniform_tables(
        batch_n, cfg.m, cfg.n, cfg.total_count, generator=generator
    )
    t_batch, s_batch = _sample_t_s_pairs(cfg.terminal_time, batch_n, generator=generator)

    x_t_list: List[Tensor] = []
    x_s_list: List[Tensor] = []
    for i in range(batch_n):
        t_i = float(t_batch[i].item())
        s_i = float(s_batch[i].item())
        result = simulate_trajectory(
            x0_batch[i], s_i, cfg.ctmc_rate, snapshot_times=[t_i, s_i], generator=generator
        )
        x_t_list.append(result.snapshots[0].table.float())
        x_s_list.append(result.snapshots[1].table.float())

    x_t = torch.stack(x_t_list, dim=0)
    x_s = torch.stack(x_s_list, dim=0)
    delta = (s_batch - t_batch) / cfg.terminal_time
    t_norm = t_batch / cfg.terminal_time
    return x_t, t_norm, delta, x_s


def compute_pretrain_loss(
    encoder: PretrainEncoder,
    head: ScorePredictionHead,
    x_t: Tensor,
    t_norm: Tensor,
    delta_norm: Tensor,
    x_s: Tensor,
    total_count: int,
) -> Tensor:
    """Per-cell cross-entropy between predicted and actual X_s cell counts.

    X_s is the LABEL (via its per-cell integer class); it is never fed into
    the encoder, only used as the classification target.
    """
    B, m, n = x_s.shape
    x_t_norm = x_t / total_count
    pooled = encoder(x_t_norm, t_norm, delta_norm)
    logits = head(pooled)  # (B, m*n, total_count+1)

    target_classes = torch.round(x_s).long().clamp(0, total_count).reshape(B, m * n)
    loss = F.cross_entropy(logits.reshape(B * m * n, -1), target_classes.reshape(-1))
    return loss


def pretrain_score(cfg: Config, verbose: bool = True) -> PretrainEncoder:
    """Pretrain E_phi (+ discardable G_psi) on the CTMC-trajectory scoring task.

    Returns:
        The trained PretrainEncoder holding E_phi's weights. G_psi
        (ScorePredictionHead) is used only during this function and is not
        returned -- callers should transfer the encoder into an HModel via
        HModel.load_encoder_state_dict(encoder.encoder_state_dict()).
    """
    device = torch.device(cfg.device)
    generator = torch.Generator().manual_seed(cfg.seed)

    encoder = PretrainEncoder(cfg.m, cfg.n, cfg.total_count).to(device)
    head = ScorePredictionHead(cfg.m, cfg.n, cfg.total_count).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=cfg.pretrain_learning_rate,
        weight_decay=cfg.weight_decay,
    )

    num_batches_per_epoch = max(1, cfg.pretrain_num_trajectories // cfg.pretrain_batch_size)
    history = PretrainHistory(losses=[])

    epoch_bar = tqdm(range(cfg.pretrain_epochs), desc="pretrain_score[epochs]", disable=not verbose)
    for epoch in epoch_bar:
        encoder.train()
        head.train()
        epoch_loss_sum = 0.0
        for _ in tqdm(
            range(num_batches_per_epoch), desc=f"epoch {epoch + 1} pretrain",
            leave=False, disable=not verbose,
        ):
            x_t, t_norm, delta_norm, x_s = generate_pretrain_batch(
                cfg, cfg.pretrain_batch_size, generator=generator
            )
            x_t = x_t.to(device)
            t_norm = t_norm.to(device)
            delta_norm = delta_norm.to(device)
            x_s = x_s.to(device)

            optimizer.zero_grad()
            loss = compute_pretrain_loss(
                encoder, head, x_t, t_norm, delta_norm, x_s, cfg.total_count
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()), cfg.grad_clip_norm
            )
            optimizer.step()
            epoch_loss_sum += float(loss.detach().item())

        epoch_loss = epoch_loss_sum / num_batches_per_epoch
        history.losses.append(epoch_loss)
        if verbose:
            epoch_bar.set_postfix(loss=epoch_loss)
            tqdm.write(f"[pretrain_score] epoch {epoch + 1}/{cfg.pretrain_epochs} loss={epoch_loss:.6f}")

    return encoder
