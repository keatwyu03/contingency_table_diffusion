"""Optional CTMC-pretraining stage for the h-function encoder E_phi.

This does NOT introduce a new noising process and does NOT simulate any
separate set of CTMC trajectories: it reuses the exact same
num_original_samples trajectories already simulated for the h-training
dataset (see h_dataset.generate_h_dataset), which stores an extra later
snapshot (X_s, s) per trajectory alongside the (X_tau, tau, R(X_0)) row
used for h-training. Pretraining only changes how the encoder used by
h_theta is initialized before h-training (see train_h.py / main.py).

Task: for the two times tau < s recorded per trajectory, predict X_s from
(X_tau, tau, delta) where delta = s - tau.

    (X_tau, tau, delta) -> E_phi -> G_psi -> predicted X_s

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
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import Config
from h_dataset import HDataset
from h_model import D_MODEL, FourierTimeEmbedding, HModel


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


class PretrainPairDataset(Dataset):
    """Wraps an HDataset's (X_tau, tau, X_s, s) fields for the pretraining task.

    Reads the later_table/later_time fields that h_dataset.generate_h_dataset
    stores alongside each (X_tau, tau, R(X_0)) row -- from the SAME
    trajectory simulation used for h-training -- rather than simulating any
    new trajectories.
    """

    def __init__(self, dataset: HDataset):
        self.samples = dataset.samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        delta = s.later_time - s.time
        return (
            s.current_table,
            torch.tensor(s.time, dtype=torch.float32),
            torch.tensor(delta, dtype=torch.float32),
            s.later_table,
        )


def compute_pretrain_loss(
    encoder: PretrainEncoder,
    head: ScorePredictionHead,
    x_t_norm: Tensor,
    t_norm: Tensor,
    delta_norm: Tensor,
    x_s_norm: Tensor,
    total_count: int,
) -> Tensor:
    """Per-cell cross-entropy between predicted and actual X_s cell counts.

    X_s is the LABEL (via its per-cell integer class); it is never fed into
    the encoder, only used as the classification target. Inputs are already
    normalized by total_count, matching HDataset's convention (see
    h_dataset.py) -- x_s_norm is un-normalized back to integer counts here
    only to build the classification target.
    """
    B, m, n = x_s_norm.shape
    pooled = encoder(x_t_norm, t_norm, delta_norm)
    logits = head(pooled)  # (B, m*n, total_count+1)

    target_classes = torch.round(x_s_norm * total_count).long().clamp(0, total_count).reshape(B, m * n)
    loss = F.cross_entropy(logits.reshape(B * m * n, -1), target_classes.reshape(-1))
    return loss


def pretrain_score(cfg: Config, dataset: HDataset, verbose: bool = True) -> PretrainEncoder:
    """Pretrain E_phi (+ discardable G_psi) on the CTMC-trajectory scoring task.

    Reuses the SAME trajectories already simulated for ``dataset`` (see
    h_dataset.generate_h_dataset) -- no new CTMC trajectories are simulated
    here. Each row's (X_tau, tau, X_s, s) fields yield one
    (X_tau, tau, delta=s-tau) -> X_s pretraining example.

    Args:
        cfg: resolved Config.
        dataset: the HDataset produced by generate_h_dataset (or loaded via
            HDataset.load), whose later_table/later_time fields this
            function reads.

    Returns:
        The trained PretrainEncoder holding E_phi's weights. G_psi
        (ScorePredictionHead) is used only during this function and is not
        returned -- callers should transfer the encoder into an HModel via
        HModel.load_encoder_state_dict(encoder.encoder_state_dict()).
    """
    device = torch.device(cfg.device)

    encoder = PretrainEncoder(cfg.m, cfg.n, cfg.total_count).to(device)
    head = ScorePredictionHead(cfg.m, cfg.n, cfg.total_count).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=cfg.pretrain_learning_rate,
        weight_decay=cfg.weight_decay,
    )

    pair_dataset = PretrainPairDataset(dataset)
    loader = DataLoader(pair_dataset, batch_size=cfg.pretrain_batch_size, shuffle=True)
    history = PretrainHistory(losses=[])

    epoch_bar = tqdm(range(cfg.pretrain_epochs), desc="pretrain_score[epochs]", disable=not verbose)
    for epoch in epoch_bar:
        encoder.train()
        head.train()
        epoch_loss_sum = 0.0
        num_batches = 0
        for x_t_norm, t_norm, delta_norm, x_s_norm in tqdm(
            loader, desc=f"epoch {epoch + 1} pretrain", leave=False, disable=not verbose,
        ):
            x_t_norm = x_t_norm.to(device)
            t_norm = t_norm.to(device)
            delta_norm = delta_norm.to(device)
            x_s_norm = x_s_norm.to(device)

            optimizer.zero_grad()
            loss = compute_pretrain_loss(
                encoder, head, x_t_norm, t_norm, delta_norm, x_s_norm, cfg.total_count
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()), cfg.grad_clip_norm
            )
            optimizer.step()
            epoch_loss_sum += float(loss.detach().item())
            num_batches += 1

        epoch_loss = epoch_loss_sum / max(num_batches, 1)
        history.losses.append(epoch_loss)
        if verbose:
            epoch_bar.set_postfix(loss=epoch_loss)
            tqdm.write(f"[pretrain_score] epoch {epoch + 1}/{cfg.pretrain_epochs} loss={epoch_loss:.6f}")

    return encoder
