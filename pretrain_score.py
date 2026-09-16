"""Optional CTMC-pretraining stage for the h-function encoder E_phi.

This does NOT introduce a new noising process and does NOT simulate any
separate set of CTMC trajectories: it reuses the exact same
num_original_samples trajectories already simulated for the h-training
dataset (see h_dataset.generate_h_dataset), which stores an extra later
snapshot (X_s, s) per trajectory alongside the (X_tau, tau, R(X_0)) row
used for h-training. Pretraining only changes how the encoder used by
h_theta is initialized before h-training (see train_h.py / main.py).

Task: for the two times tau < s recorded per trajectory, predict the
FUTURE MARGINS of X_s (its row sums r_s = X_s @ 1 and column sums
c_s = X_s^T @ 1) from (X_tau, tau, delta), delta = s - tau:

    (X_tau, tau) -> E_phi -> z_tau
    (z_tau, delta) -> G_psi -> predicted (r_s, c_s)

E_phi (PretrainEncoder) is now EXACTLY HModel.encode -- z_tau = E_phi(X_tau,
tau) has no dependence on delta, so the representation being pretrained is
identical to the representation later transferred into HModel (no
delta-conditioned information leaks into the transferable weights). This
is important because E_phi's weights (not G_psi's) are what gets copied
into a fresh HModel before h-training; if delta were mixed into E_phi's
output (e.g. via addition into the pooled vector), the transferred encoder
would have been trained to produce a representation that depended on an
input (delta) it will never receive at h-training/sampling time.

delta is instead injected inside the discardable G_psi head: G_psi embeds
delta with a Fourier time embedding and concatenates it with z_tau (not
adds -- concatenation keeps state and horizon information in separate
coordinates rather than forcing them to share the same subspace), then an
MLP maps the concatenated features to two length-m/length-n logit vectors
a_r, a_c, converted into predicted margins via total_count * softmax(.) so
each predicted margin vector sums exactly to total_count by construction --
matching the true row/column sums of any valid contingency table. This is
a much lighter-weight, structurally-matched pretraining target than
per-cell classification: predicting future margins forces E_phi to
represent whatever information about the CTMC's future row/column drift is
useful for later h-training, without requiring it to reconstruct every
cell. G_psi (including its delta embedding) is discarded after
pretraining; only E_phi's weights are kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import Config
from h_dataset import HDataset
from h_model import D_MODEL, FourierTimeEmbedding, HModel
from table_space import col_sums, row_sums


class PretrainEncoder(nn.Module):
    """E_phi: encodes (X_t, t) into a pooled D_MODEL representation z_t.

    Architecturally identical to HModel's own encoder (same embeddings,
    same transformer blocks, same pooling) and, critically, takes NO delta
    input -- z_t = E_phi(X_t, t) is exactly the representation that will
    later be transferred into a fresh HModel, so nothing delta-dependent
    can leak into these weights. count_embedding / row_embedding /
    column_embedding / time_embedding / blocks / final_ln can be copied
    verbatim into an HModel via HModel.load_encoder_state_dict.
    """

    def __init__(self, m: int, n: int, total_count: int):
        super().__init__()
        # A plain HModel supplies every E_phi submodule we need (and has the
        # exact state-dict keys HModel.load_encoder_state_dict expects) --
        # its unused `head` (H_omega) is simply never called here.
        self.base = HModel(m, n, total_count)

    def forward(self, table_norm: Tensor, time_norm: Tensor) -> Tensor:
        return self.base.encode(table_norm, time_norm)  # (B, D_MODEL)

    def encoder_state_dict(self) -> dict:
        """E_phi weights in the exact layout HModel.load_encoder_state_dict expects."""
        return self.base.encoder_state_dict()


class MarginPredictionHead(nn.Module):
    """G_psi: predicts future (X_s) row/column margins from (z_t, delta).

    delta is embedded with its own Fourier time embedding and CONCATENATED
    with the pooled encoder representation z_t (not added -- concatenation
    keeps state and horizon information in separate coordinates instead of
    forcing them into the same subspace). The concatenated features feed
    two MLPs producing length-m/length-n logit vectors a_r, a_c; callers
    convert these into margins summing exactly to total_count via
    total_count * softmax(.) (see predict_margins below). This whole module
    -- including delta_embedding -- is discarded after pretraining; delta
    never touches E_phi's own weights.
    """

    def __init__(self, m: int, n: int, total_count: int):
        super().__init__()
        self.delta_embedding = FourierTimeEmbedding(D_MODEL)
        in_dim = 2 * D_MODEL
        self.row_mlp = nn.Sequential(
            nn.Linear(in_dim, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, m),
        )
        self.col_mlp = nn.Sequential(
            nn.Linear(in_dim, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, n),
        )

    def forward(self, z_t: Tensor, delta_norm: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (a_r, a_c): raw logits of shape (B, m) and (B, n)."""
        delta_embed = self.delta_embedding(delta_norm)
        features = torch.cat([z_t, delta_embed], dim=-1)  # (B, 2*D_MODEL)
        return self.row_mlp(features), self.col_mlp(features)


def predict_margins(
    head: MarginPredictionHead, z_t: Tensor, delta_norm: Tensor, total_count: int
) -> Tuple[Tensor, Tensor]:
    """Converts G_psi's logits into margins each summing exactly to total_count.

    r_hat = total_count * softmax(a_r), c_hat = total_count * softmax(a_c).
    """
    a_r, a_c = head(z_t, delta_norm)
    r_hat = total_count * torch.softmax(a_r, dim=-1)
    c_hat = total_count * torch.softmax(a_c, dim=-1)
    return r_hat, c_hat


@dataclass
class PretrainHistory:
    losses: List[float]
    row_losses: List[float]
    col_losses: List[float]


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


def _assert_normalized_table(table_norm: Tensor, total_count: int, atol: float = 1e-4) -> None:
    """Guards the x_s = x_s_norm * total_count un-normalization below.

    h_dataset.generate_h_dataset stores later_table pre-divided by
    total_count (same convention as current_table -- see h_dataset.py's
    HDatasetSample docstring and generate_h_dataset's
    `later_table=x_s / cfg.total_count`). If later_table were instead raw
    counts summing to total_count, multiplying by total_count again would
    silently produce margin targets off by a factor of total_count, so this
    checks each table's normalized entries sum to 1 rather than assuming it.
    """
    totals = table_norm.sum(dim=(-2, -1))
    assert torch.allclose(
        totals, torch.ones_like(totals), atol=atol
    ), (
        "later_table does not appear to be normalized by total_count "
        f"(expected each table's normalized entries to sum to 1, got {totals})"
    )


def compute_pretrain_loss(
    encoder: PretrainEncoder,
    head: MarginPredictionHead,
    x_t_norm: Tensor,
    t_norm: Tensor,
    delta_norm: Tensor,
    x_s_norm: Tensor,
    total_count: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Normalized squared-error loss between predicted and true future margins.

    x_s_norm (X_s, normalized by total_count) is the LABEL -- it is never
    fed into the encoder, only un-normalized here to build the row/column
    margin targets r_s = X_s @ 1, c_s = X_s^T @ 1. Inputs are already
    normalized by total_count / terminal_time, matching HDataset's
    convention (see h_dataset.py).

    Returns (total_loss, row_loss, col_loss), where
        total_loss = row_loss + col_loss
        row_loss = mean_batch( ||r_hat - r_s||_2^2 ) / total_count^2
        col_loss = mean_batch( ||c_hat - c_s||_2^2 ) / total_count^2
    """
    _assert_normalized_table(x_s_norm, total_count)

    z_t = encoder(x_t_norm, t_norm)
    r_hat, c_hat = predict_margins(head, z_t, delta_norm, total_count)

    x_s = x_s_norm * total_count
    r_s = row_sums(x_s)
    c_s = col_sums(x_s)

    n_sq = float(total_count) ** 2
    row_loss = ((r_hat - r_s) ** 2).sum(dim=-1).mean() / n_sq
    col_loss = ((c_hat - c_s) ** 2).sum(dim=-1).mean() / n_sq
    total_loss = row_loss + col_loss
    return total_loss, row_loss, col_loss


@torch.no_grad()
def check_margins_sum_to_total_count(
    encoder: PretrainEncoder,
    head: MarginPredictionHead,
    loader: DataLoader,
    total_count: int,
    device: torch.device,
    atol: float = 1e-3,
) -> None:
    """Asserts every predicted row/column margin vector sums to total_count.

    Sanity check for the total_count * softmax(.) construction: raises
    AssertionError if any batch's predicted margins fail to sum to
    total_count within atol.
    """
    encoder.eval()
    head.eval()
    for x_t_norm, t_norm, delta_norm, _ in loader:
        x_t_norm = x_t_norm.to(device)
        t_norm = t_norm.to(device)
        delta_norm = delta_norm.to(device)
        z_t = encoder(x_t_norm, t_norm)
        r_hat, c_hat = predict_margins(head, z_t, delta_norm, total_count)
        row_totals = r_hat.sum(dim=-1)
        col_totals = c_hat.sum(dim=-1)
        assert torch.allclose(
            row_totals, torch.full_like(row_totals, float(total_count)), atol=atol
        ), f"row margins do not sum to total_count={total_count}: {row_totals}"
        assert torch.allclose(
            col_totals, torch.full_like(col_totals, float(total_count)), atol=atol
        ), f"column margins do not sum to total_count={total_count}: {col_totals}"


@torch.no_grad()
def evaluate_pretrain(
    encoder: PretrainEncoder,
    head: MarginPredictionHead,
    loader: DataLoader,
    total_count: int,
    device: torch.device,
) -> dict:
    """Computes mean total/row/col loss and mean absolute margin error over a loader."""
    encoder.eval()
    head.eval()
    total_sum = 0.0
    row_sum = 0.0
    col_sum = 0.0
    mae_sum = 0.0
    num_batches = 0
    for x_t_norm, t_norm, delta_norm, x_s_norm in loader:
        x_t_norm = x_t_norm.to(device)
        t_norm = t_norm.to(device)
        delta_norm = delta_norm.to(device)
        x_s_norm = x_s_norm.to(device)

        total_loss, row_loss, col_loss = compute_pretrain_loss(
            encoder, head, x_t_norm, t_norm, delta_norm, x_s_norm, total_count
        )

        z_t = encoder(x_t_norm, t_norm)
        r_hat, c_hat = predict_margins(head, z_t, delta_norm, total_count)
        x_s = x_s_norm * total_count
        r_s = row_sums(x_s)
        c_s = col_sums(x_s)
        mae = (
            (r_hat - r_s).abs().mean(dim=-1) + (c_hat - c_s).abs().mean(dim=-1)
        ).mean() / 2.0

        total_sum += float(total_loss.item())
        row_sum += float(row_loss.item())
        col_sum += float(col_loss.item())
        mae_sum += float(mae.item())
        num_batches += 1

    num_batches = max(num_batches, 1)
    return {
        "total_loss": total_sum / num_batches,
        "row_loss": row_sum / num_batches,
        "col_loss": col_sum / num_batches,
        "mean_abs_margin_error": mae_sum / num_batches,
    }


def pretrain_score(cfg: Config, dataset: HDataset, verbose: bool = True) -> PretrainEncoder:
    """Pretrain E_phi (+ discardable G_psi) on the CTMC multi-horizon margin task.

    Reuses the SAME trajectories already simulated for ``dataset`` (see
    h_dataset.generate_h_dataset) -- no new CTMC trajectories are simulated
    here. Each row's (X_tau, tau, X_s, s) fields yield one
    (X_tau, tau, delta=s-tau) -> (r_s, c_s) pretraining example, where
    r_s, c_s are X_s's row/column margins.

    Args:
        cfg: resolved Config.
        dataset: the HDataset produced by generate_h_dataset (or loaded via
            HDataset.load), whose later_table/later_time fields this
            function reads.

    Returns:
        The PretrainEncoder holding E_phi's weights FROM THE EPOCH WITH THE
        LOWEST VALIDATION total_loss (not necessarily the final epoch --
        with a limited number of independent trajectories, later epochs can
        overfit), analogous to train_h_model's best-checkpoint selection.
        G_psi (MarginPredictionHead) is used only during this function and
        is not returned -- callers should transfer the encoder into an
        HModel via HModel.load_encoder_state_dict(encoder.encoder_state_dict()).
    """
    device = torch.device(cfg.device)

    encoder = PretrainEncoder(cfg.m, cfg.n, cfg.total_count).to(device)
    head = MarginPredictionHead(cfg.m, cfg.n, cfg.total_count).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=cfg.pretrain_learning_rate,
        weight_decay=cfg.weight_decay,
    )

    pair_dataset = PretrainPairDataset(dataset)
    val_size = max(1, int(len(pair_dataset) * cfg.val_fraction))
    train_size = len(pair_dataset) - val_size
    generator = torch.Generator().manual_seed(cfg.seed)
    train_subset, val_subset = torch.utils.data.random_split(
        pair_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_subset, batch_size=cfg.pretrain_batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=cfg.pretrain_batch_size, shuffle=False)

    history = PretrainHistory(losses=[], row_losses=[], col_losses=[])
    best_val_loss = float("inf")
    best_encoder_state = None
    best_head_state = None

    epoch_bar = tqdm(range(cfg.pretrain_epochs), desc="pretrain_score[epochs]", disable=not verbose)
    for epoch in epoch_bar:
        encoder.train()
        head.train()
        epoch_loss_sum = 0.0
        epoch_row_sum = 0.0
        epoch_col_sum = 0.0
        num_batches = 0
        for x_t_norm, t_norm, delta_norm, x_s_norm in tqdm(
            train_loader, desc=f"epoch {epoch + 1} pretrain", leave=False, disable=not verbose,
        ):
            x_t_norm = x_t_norm.to(device)
            t_norm = t_norm.to(device)
            delta_norm = delta_norm.to(device)
            x_s_norm = x_s_norm.to(device)

            optimizer.zero_grad()
            loss, row_loss, col_loss = compute_pretrain_loss(
                encoder, head, x_t_norm, t_norm, delta_norm, x_s_norm, cfg.total_count
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()), cfg.grad_clip_norm
            )
            optimizer.step()
            epoch_loss_sum += float(loss.detach().item())
            epoch_row_sum += float(row_loss.detach().item())
            epoch_col_sum += float(col_loss.detach().item())
            num_batches += 1

        num_batches = max(num_batches, 1)
        epoch_loss = epoch_loss_sum / num_batches
        history.losses.append(epoch_loss)
        history.row_losses.append(epoch_row_sum / num_batches)
        history.col_losses.append(epoch_col_sum / num_batches)

        val_metrics = evaluate_pretrain(encoder, head, val_loader, cfg.total_count, device)

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_encoder_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
            best_head_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

        if verbose:
            epoch_bar.set_postfix(loss=epoch_loss, val_loss=val_metrics["total_loss"])
            tqdm.write(
                f"[pretrain_score] epoch {epoch + 1}/{cfg.pretrain_epochs} "
                f"train_loss={epoch_loss:.6f} "
                f"val_row_loss={val_metrics['row_loss']:.6f} "
                f"val_col_loss={val_metrics['col_loss']:.6f} "
                f"val_total_loss={val_metrics['total_loss']:.6f} "
                f"val_mean_abs_margin_error={val_metrics['mean_abs_margin_error']:.6f}"
            )

    # Restore best-val-loss weights (not necessarily the final epoch's) for
    # both encoder and head before the final validation check, mirroring
    # train_h_model's best-checkpoint selection.
    if best_encoder_state is not None:
        encoder.load_state_dict(best_encoder_state)
        head.load_state_dict(best_head_state)
        if verbose:
            tqdm.write(f"[pretrain_score] restored best-val-loss weights (val_total_loss={best_val_loss:.6f})")

    check_margins_sum_to_total_count(encoder, head, val_loader, cfg.total_count, device)
    if verbose:
        tqdm.write(
            "[pretrain_score] validated: all predicted row/column margins sum to "
            f"total_count={cfg.total_count}"
        )

    return encoder
