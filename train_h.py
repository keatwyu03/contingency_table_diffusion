from __future__ import annotations

import os
import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.nn.attention import SDPBackend, sdpa_kernel

from config import Config
from ctmc import num_ordered_pairs
from h_dataset import HDataset
from h_model import HModel


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainHistory:
    """Tracks per-epoch train/validation loss history."""

    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = -1


def split_dataset(
    dataset: HDataset, val_fraction: float, seed: int
) -> Tuple[HDataset, HDataset]:
    """Split a dataset into train/val subsets by original_sample_id.

    All (tau, X_tau) observations sharing the same original_sample_id (i.e.
    drawn by forward-noising the same X_0) are kept together in either train
    or val -- splitting at the flat sample level would leak the same X_0's
    R(X_0) target across the split via correlated snapshots.
    """
    original_ids = sorted({s.original_sample_id for s in dataset.samples})
    n_val_ids = max(1, int(len(original_ids) * val_fraction))
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(original_ids), generator=generator).tolist()
    val_id_set = {original_ids[i] for i in perm[:n_val_ids]}
    train_id_set = {original_ids[i] for i in perm[n_val_ids:]}

    train_samples = [s for s in dataset.samples if s.original_sample_id in train_id_set]
    val_samples = [s for s in dataset.samples if s.original_sample_id in val_id_set]
    return HDataset(train_samples), HDataset(val_samples)


@dataclass
class BKDiagnostics:
    """Running (sum, count) accumulators for BK-loss diagnostics over an epoch.

    Kept separate from TrainHistory (which only tracks scalar per-epoch
    losses) since these are finer-grained internals of the BK term specific
    to the hybrid direct-h + backward-Kolmogorov objective.
    """

    sum_mc_loss: float = 0.0
    sum_terminal_loss: float = 0.0
    sum_bk_loss: float = 0.0
    sum_total_loss: float = 0.0
    sum_mc_weighted: float = 0.0
    sum_terminal_weighted: float = 0.0
    sum_bk_weighted: float = 0.0
    sum_abs_bk_residual: float = 0.0
    max_abs_bk_residual: float = 0.0
    sum_h: float = 0.0
    sum_u: float = 0.0
    min_u: float = float("inf")
    max_u: float = float("-inf")
    sum_log_ratio: float = 0.0
    min_log_ratio: float = float("inf")
    max_log_ratio: float = float("-inf")
    num_log_ratio_clipped: int = 0
    num_log_ratio_total: int = 0
    sum_exit_intensity: float = 0.0
    num_batches: int = 0
    num_bk_rows: int = 0

    def update(
        self,
        mc_loss: float,
        terminal_loss: float,
        bk_loss: float,
        total_loss: float,
        bk_stats: Optional[dict] = None,
        mc_weighted: float = 0.0,
        terminal_weighted: float = 0.0,
        bk_weighted: float = 0.0,
    ) -> None:
        self.sum_mc_loss += mc_loss
        self.sum_terminal_loss += terminal_loss
        self.sum_bk_loss += bk_loss
        self.sum_total_loss += total_loss
        self.sum_mc_weighted += mc_weighted
        self.sum_terminal_weighted += terminal_weighted
        self.sum_bk_weighted += bk_weighted
        self.num_batches += 1
        if bk_stats is not None:
            self.sum_abs_bk_residual += bk_stats["sum_abs_residual"]
            self.max_abs_bk_residual = max(self.max_abs_bk_residual, bk_stats["max_abs_residual"])
            self.sum_h += bk_stats["sum_h"]
            self.sum_u += bk_stats["sum_u"]
            self.min_u = min(self.min_u, bk_stats["min_u"])
            self.max_u = max(self.max_u, bk_stats["max_u"])
            self.sum_log_ratio += bk_stats["sum_log_ratio"]
            self.min_log_ratio = min(self.min_log_ratio, bk_stats["min_log_ratio"])
            self.max_log_ratio = max(self.max_log_ratio, bk_stats["max_log_ratio"])
            self.num_log_ratio_clipped += bk_stats["num_log_ratio_clipped"]
            self.num_log_ratio_total += bk_stats["num_log_ratio_total"]
            self.sum_exit_intensity += bk_stats["sum_exit_intensity"]
            self.num_bk_rows += bk_stats["num_rows"]

    def summary(self) -> str:
        nb = max(self.num_batches, 1)
        nr = max(self.num_bk_rows, 1)
        return (
            f"mc_loss={self.sum_mc_loss / nb:.6f} "
            f"terminal_loss={self.sum_terminal_loss / nb:.6f} "
            f"bk_loss={self.sum_bk_loss / nb:.6f} "
            f"total_loss={self.sum_total_loss / nb:.6f} "
            f"| weighted: mc={self.sum_mc_weighted / nb:.6f} "
            f"terminal={self.sum_terminal_weighted / nb:.6f} "
            f"bk={self.sum_bk_weighted / nb:.6f} "
            f"mean_abs_bk_residual={self.sum_abs_bk_residual / nr:.6f} "
            f"max_abs_bk_residual={self.max_abs_bk_residual:.6f} "
            f"mean_h={self.sum_h / nr:.6f} "
            f"mean_u={self.sum_u / nr:.6f} range_u=[{self.min_u:.4f},{self.max_u:.4f}] "
            f"mean_log_ratio={self.sum_log_ratio / max(self.num_log_ratio_total, 1):.6f} "
            f"range_log_ratio=[{self.min_log_ratio:.4f},{self.max_log_ratio:.4f}] "
            f"frac_clipped={self.num_log_ratio_clipped / max(self.num_log_ratio_total, 1):.6f} "
            f"mean_exit_intensity={self.sum_exit_intensity / nr:.6f}"
        )


def _sample_neighbors_direct(
    unnormalized_tables: torch.Tensor, num_neighbors: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample ``num_neighbors`` valid neighbor tables per row, WITHOUT ever
    calling all_neighbors (which materializes every one of a table's
    potentially thousands of valid moves -- for total_count=82, m=n=12, a
    typical table has on the order of 80 * 143 ~= 11,000 valid neighbors,
    so building all of them per anchor row is exactly the "prohibitively
    large" cost the BK term must avoid).

    Matches the proposal convention of ctmc.propose_move exactly: a move is
    (source cell, dest cell) with source uniform among POSITIVE cells (a
    move needs a unit to take from) and dest uniform among the other d-1
    cells. This is also the same convention as
    sample_guided._thinning_batch_step's proposal sampling, reused here via
    the same padded-positive-cell-index gather trick (built once per row,
    since the row's positive cells don't change while sampling its
    neighbors).

    Args:
        unnormalized_tables: (A, m, n) integer-valued raw-count tables.
        num_neighbors: number of neighbor samples to draw per row (with
            replacement across distinct (src, dst) pairs -- collisions
            across rows are impossible since each row's neighbors are
            sampled independently, and within a row, repeated (src, dst)
            draws are permitted and simply reweight that outcome, which the
            Monte Carlo generator-term estimator already accounts for).

    Returns:
        (neighbor_tables, positive_counts): neighbor_tables is
        (A, num_neighbors, m, n); positive_counts is (A,) float, the number
        of positive cells P_i per row (used by the caller to recover
        lambda_t(x) = P_i * (d-1) * base_rate, the total exit intensity).
    """
    A, m, n = unnormalized_tables.shape
    d = m * n
    flat_x = unnormalized_tables.reshape(A, d)

    positive_mask = flat_x > 0
    positive_counts = positive_mask.sum(dim=1).to(torch.float32)  # (A,)
    max_P = int(positive_counts.max().item()) if A > 0 else 0
    if max_P == 0:
        return torch.empty((A, 0, m, n), dtype=unnormalized_tables.dtype), positive_counts

    # Dense (A, max_P) padded positive-cell index table -- same trick as
    # sample_guided._thinning_batch_step's positive_padded.
    positive_padded = torch.zeros((A, max_P), dtype=torch.long)
    for i in range(A):
        idx = torch.nonzero(positive_mask[i], as_tuple=False).view(-1)
        positive_padded[i, : idx.shape[0]] = idx

    K = num_neighbors
    P_per_row = positive_counts.clamp(min=1.0)  # avoid div-by-zero for degenerate rows

    u_src = torch.rand(A, K)
    src_choice = torch.clamp((u_src * P_per_row.unsqueeze(1)).long(), max=(P_per_row.long() - 1).unsqueeze(1))
    src_cells = torch.gather(positive_padded, 1, src_choice)  # (A, K)

    dst_choice = torch.randint(0, d - 1, (A, K))
    dst_cells = dst_choice + (dst_choice >= src_cells).long()  # skip dst == src

    neighbor_flat = flat_x.unsqueeze(1).expand(A, K, d).clone()
    row_idx = torch.arange(A).unsqueeze(1).expand(A, K)
    k_idx = torch.arange(K).unsqueeze(0).expand(A, K)
    neighbor_flat[row_idx, k_idx, src_cells] -= 1
    neighbor_flat[row_idx, k_idx, dst_cells] += 1

    neighbor_tables = neighbor_flat.view(A, K, m, n)
    return neighbor_tables, positive_counts


def bk_residual(
    model: HModel,
    tables: torch.Tensor,
    times: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, dict]:
    """Backward-Kolmogorov residual, evaluated on a small ANCHOR subset of
    the given (X_tau, tau) batch (see Config.bk_anchor_batch_size) with
    neighbors sampled directly, never via all_neighbors.

    Sign convention (see train_h.py module docstring / audit): the guided
    sampler's sampling clock s = T - t increases from t=T toward the
    generated output at t=0 (sample_guided.py runs t: T -> 0). With
    u_theta(t,x) = log h_theta(t,x) parameterized directly in model time t,
    the chain rule ds = -dt gives d/ds u = -d/dt u, so the BK equation
    d/ds u + sum_y G(x,y)[exp(u(s,y)-u(s,x)) - 1] = 0 becomes, in model time:

        -d/dt u_theta(t,x) + sum_y G_t(x,y) [exp(u_theta(t,y)-u_theta(t,x)) - 1] = 0

    G_t(x,y) is the exact base (unguided) off-diagonal rate used by the
    guided sampler immediately before its h(t,y)/h(t,x) multiplier (see
    sample_guided.guided_step / _guided_batch_step): a constant
    ctmc_rate / K for every valid neighbor y of x, K = num_ordered_pairs(m,n).

    time_norm = t / terminal_time is the model's actual input (see
    h_model.py); autograd differentiates u_theta w.r.t. time_norm (the leaf
    requiring grad), and the result is converted to d/dt via the chain rule
    d/dt = (1/terminal_time) * d/d(time_norm).

    tables: (B, m, n) NORMALIZED tables (X_tau / total_count), matching the
    convention of every other tensor already flowing through compute_loss.
    times: (B,) NORMALIZED times (tau / terminal_time, in [0, 1]). Only
    cfg.bk_anchor_batch_size rows (sampled uniformly without replacement,
    or all B if B <= bk_anchor_batch_size) are used as BK anchors -- the PDE
    residual is a pointwise constraint on (t,x), so a small anchor subset
    per step gives unbiased-in-expectation coverage across steps without
    scaling BK cost with the (typically much larger) MC batch size B.

    Returns:
        (residual, stats): residual is (A,) the normalized BK residual
        R_tilde_theta(t,x) = R_theta(t,x) / (1 + lambda_t(x)) for each
        anchor row (A = min(B, cfg.bk_anchor_batch_size)); stats is a dict
        of running diagnostics (see BKDiagnostics.update).
    """
    device = next(model.parameters()).device
    B = tables.shape[0]

    anchor_n = min(B, cfg.bk_anchor_batch_size)
    anchor_idx = torch.randperm(B)[:anchor_n]
    tables = tables[anchor_idx]
    times = times[anchor_idx]
    A = anchor_n

    K = num_ordered_pairs(cfg.m, cfg.n)
    base_rate = cfg.ctmc_rate / K  # G_t(x,y), constant over every valid neighbor y
    d = cfg.m * cfg.n

    time_norm = times.to(device).detach().requires_grad_(True)
    table_norm = tables.to(device)

    needs_double_backward = model.training
    # The fused/efficient scaled-dot-product-attention kernel used inside
    # HModel's transformer blocks does not implement a second derivative
    # (backward-of-backward), which torch.autograd.grad(..., create_graph=
    # True) below requires so the eventual loss.backward() can differentiate
    # THROUGH du_dtime_norm into the model's own weights. Forcing the math
    # (unfused) SDPA backend only around this forward pass sidesteps that
    # gap; forward passes that are never double-differentiated (logit_y
    # below, and this same call in eval mode) can keep the fused kernel.
    # This only runs on the small anchor batch (A rows, not B), so the
    # MATH backend's extra memory cost no longer scales with the full
    # minibatch size.
    attention_backend = (
        sdpa_kernel([SDPBackend.MATH]) if needs_double_backward else nullcontext()
    )
    with torch.enable_grad(), attention_backend:
        logit_x = model.forward_logits(table_norm, time_norm)  # (A,)
        u_x = nn.functional.logsigmoid(logit_x)  # log h_theta(t,x), stable

        du_dtime_norm = torch.autograd.grad(
            u_x.sum(), time_norm,
            create_graph=needs_double_backward, retain_graph=needs_double_backward,
        )[0]
    du_dt = du_dtime_norm / cfg.terminal_time  # chain rule: d/dt = (1/T) d/d(t/T)
    du_ds = -du_dt  # ds = -dt (see docstring)

    unnormalized_tables = torch.round(tables * cfg.total_count).to(torch.float32).cpu()
    neighbor_tables, positive_counts = _sample_neighbors_direct(
        unnormalized_tables, cfg.bk_num_neighbors
    )  # (A, K, m, n), (A,)

    valid_rows = positive_counts > 0
    if not valid_rows.any():
        empty_stats = {
            "sum_abs_residual": 0.0, "max_abs_residual": 0.0, "sum_h": 0.0,
            "sum_u": 0.0, "min_u": float("inf"), "max_u": float("-inf"),
            "sum_log_ratio": 0.0, "min_log_ratio": float("inf"),
            "max_log_ratio": float("-inf"), "num_log_ratio_clipped": 0,
            "num_log_ratio_total": 0, "sum_exit_intensity": 0.0, "num_rows": 0,
        }
        return torch.zeros(0, device=device), empty_stats

    Knb = cfg.bk_num_neighbors
    row_time_norm = times.unsqueeze(1).expand(A, Knb).reshape(-1)  # (A*K,)
    all_neighbor_table_norm = (neighbor_tables.reshape(A * Knb, cfg.m, cfg.n).to(device)) / cfg.total_count
    all_neighbor_time_norm = row_time_norm.to(device)
    logit_y = model.forward_logits(all_neighbor_table_norm, all_neighbor_time_norm)
    u_y = nn.functional.logsigmoid(logit_y).view(A, Knb)  # (A, K)

    u_x_per_neighbor = u_x.unsqueeze(1).expand(A, Knb)  # (A, K)

    log_ratio_raw = u_y - u_x_per_neighbor
    clip = cfg.bk_log_ratio_clip
    log_ratio = torch.clamp(log_ratio_raw, min=-clip, max=clip)
    num_clipped = int(((log_ratio_raw < -clip) | (log_ratio_raw > clip)).sum().item())

    # Monte Carlo estimate of sum_y G(x,y)[e^{u(y)-u(x)}-1] via K neighbors
    # sampled uniformly over the P*(d-1) valid moves (pi(y|x) = 1/(P*(d-1))),
    # so lambda_t(x)/K per sampled neighbor is the correct importance weight
    # (see the unbiased-estimator derivation in the module/spec docstring).
    lambda_t = positive_counts.to(device) * (d - 1) * base_rate  # (A,) sum_y G_t(x,y)
    generator_term = (lambda_t.unsqueeze(1) / Knb) * (torch.exp(log_ratio) - 1.0)
    generator_term = generator_term.sum(dim=1)  # (A,)

    valid_idx = torch.nonzero(valid_rows.to(device), as_tuple=False).view(-1)
    residual = du_ds[valid_idx] + generator_term[valid_idx]
    normalized_residual = residual / (1.0 + lambda_t[valid_idx])

    with torch.no_grad():
        h_x = torch.sigmoid(logit_x[valid_idx])
        valid_log_ratio = log_ratio[valid_idx]
        stats = {
            "sum_abs_residual": normalized_residual.abs().sum().item(),
            "max_abs_residual": normalized_residual.abs().max().item(),
            "sum_h": h_x.sum().item(),
            "sum_u": u_x[valid_idx].sum().item(),
            "min_u": u_x[valid_idx].min().item(),
            "max_u": u_x[valid_idx].max().item(),
            "sum_log_ratio": valid_log_ratio.sum().item(),
            "min_log_ratio": valid_log_ratio.min().item() if valid_log_ratio.numel() > 0 else float("inf"),
            "max_log_ratio": valid_log_ratio.max().item() if valid_log_ratio.numel() > 0 else float("-inf"),
            "num_log_ratio_clipped": num_clipped,
            "num_log_ratio_total": log_ratio.numel(),
            "sum_exit_intensity": lambda_t[valid_idx].sum().item(),
            "num_rows": int(valid_idx.numel()),
        }

    return normalized_residual, stats


def compute_loss(
    model: HModel,
    tables: torch.Tensor,
    times: torch.Tensor,
    rewards: torch.Tensor,
    original_tables: torch.Tensor,
    boundary_loss_weight: float,
    cfg: Config,
) -> Tuple[torch.Tensor, dict]:
    """Compute the hybrid direct-h + backward-Kolmogorov loss for one batch.

    L = L_MC + terminal_loss_weight * L_terminal + bk_loss_weight * L_BK

    L_MC (unchanged, primary signal): MSE between h_theta(t, X_t) and the
    Monte Carlo target R(X_0) -- NEVER done in log space, since
    log E[R|X_t] != E[log R|X_t] in general (see module docstring).

    L_terminal: deterministic terminal-boundary identity at the actual
    generated-output endpoint t=0, X_0 (not an intermediate or later
    snapshot): u_theta(0, X_0) should equal log R(X_0) = -gamma * S_2(X_0).
    Evaluated in log space since this boundary condition is itself an exact
    (not Monte Carlo) identity, so no Jensen-gap issue applies here.

    L_BK: mean squared normalized backward-Kolmogorov residual (see
    bk_residual) enforcing the PDE u_theta must solve under the
    unconditional CTMC generator, evaluated at every (t, X_t) training row
    (not only the boundary).

    Returns:
        (total_loss, unweighted_parts): unweighted_parts has keys
        "mc_loss", "terminal_loss", "bk_loss" (each a python float, detached)
        plus, when use_bk_regularization is True, the bk_residual diagnostics
        dict under "bk_stats".
    """
    device = tables.device
    preds = model.forward(tables, times)
    mc_loss = nn.functional.mse_loss(preds, rewards)

    if boundary_loss_weight > 0.0:
        is_boundary = times <= 1e-6
        if is_boundary.any():
            boundary_preds = preds[is_boundary]
            boundary_targets = rewards[is_boundary]
            boundary_loss = nn.functional.mse_loss(boundary_preds, boundary_targets)
            mc_loss = mc_loss + boundary_loss_weight * boundary_loss

    total_loss = mc_loss
    terminal_loss_value = 0.0

    if cfg.terminal_loss_weight > 0.0:
        original_table_norm = original_tables.to(device)  # already normalized by N (see HDatasetSample)
        zero_time = torch.zeros(tables.shape[0], device=device)
        terminal_logit = model.forward_logits(original_table_norm, zero_time)
        u_terminal = nn.functional.logsigmoid(terminal_logit)
        log_r_terminal = torch.log(rewards.clamp(min=cfg.h_log_epsilon))
        terminal_loss = nn.functional.mse_loss(u_terminal, log_r_terminal)
        total_loss = total_loss + cfg.terminal_loss_weight * terminal_loss
        terminal_loss_value = terminal_loss.detach().item()

    unweighted_parts = {
        "mc_loss": mc_loss.detach().item(),
        "terminal_loss": terminal_loss_value,
        "bk_loss": 0.0,
        "mc_weighted": mc_loss.detach().item(),
        "terminal_weighted": cfg.terminal_loss_weight * terminal_loss_value,
        "bk_weighted": 0.0,
    }

    if cfg.use_bk_regularization:
        residual, bk_stats = bk_residual(model, tables, times, cfg)
        if residual.numel() > 0:
            bk_loss = (residual ** 2).mean()
            total_loss = total_loss + cfg.bk_loss_weight * bk_loss
            unweighted_parts["bk_loss"] = bk_loss.detach().item()
            unweighted_parts["bk_weighted"] = cfg.bk_loss_weight * bk_loss.detach().item()
            unweighted_parts["bk_stats"] = bk_stats

    unweighted_parts["total_loss"] = total_loss.detach().item()

    return total_loss, unweighted_parts


def train_h_model(
    cfg: Config,
    dataset: HDataset,
    model: Optional[HModel] = None,
    encoder_lr_scale: Optional[float] = None,
) -> Tuple[HModel, TrainHistory]:
    """Train h_theta on ``dataset`` per the Config, with checkpointing.

    Saves ``h_model_last.pt`` and ``h_model_best.pt`` (by validation loss) to
    cfg.checkpoint_dir, along with the loss history.

    Args:
        encoder_lr_scale: if given (e.g. cfg.pretrain_encoder_lr_scale after
            CTMC pretraining -- see pretrain_score.py), E_phi's submodules
            (model.encoder_state_dict()'s keys) are trained at
            cfg.learning_rate * encoder_lr_scale while H_omega (model.head)
            keeps the full cfg.learning_rate, via two AdamW param groups.
            Both still receive gradients from the same loss.backward() call
            below, so E_phi and H_omega are jointly fine-tuned. If None
            (default), every parameter uses a single group at
            cfg.learning_rate, identical to prior behavior.

    Returns:
        (trained model, TrainHistory)
    """
    set_seed(cfg.seed)
    cfg.ensure_dirs()

    device = torch.device(cfg.device)
    model = model or HModel(cfg.m, cfg.n, cfg.total_count)
    model = model.to(device)

    train_subset, val_subset = split_dataset(dataset, cfg.val_fraction, cfg.seed)
    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=False)

    if encoder_lr_scale is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
    else:
        encoder_param_names = set(model.encoder_state_dict().keys())
        encoder_params = [
            p for name, p in model.named_parameters() if name in encoder_param_names
        ]
        head_params = [
            p for name, p in model.named_parameters() if name not in encoder_param_names
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": cfg.learning_rate * encoder_lr_scale},
                {"params": head_params, "lr": cfg.learning_rate},
            ],
            weight_decay=cfg.weight_decay,
        )

    history = TrainHistory()
    best_state_dict = None
    epochs_without_improvement = 0

    epoch_bar = tqdm(range(cfg.num_epochs), desc="train_h[epochs]")
    for epoch in epoch_bar:
        model.train()
        train_loss_sum = torch.zeros((), device=device)
        train_count = 0
        train_diag = BKDiagnostics()
        batch_bar = tqdm(train_loader, desc=f"epoch {epoch + 1} train", leave=False)
        for tables, times, rewards, original_tables in batch_bar:
            tables = tables.to(device)
            times = times.to(device)
            rewards = rewards.to(device)
            original_tables = original_tables.to(device)

            optimizer.zero_grad()
            loss, parts = compute_loss(
                model, tables, times, rewards, original_tables, cfg.boundary_loss_weight, cfg
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            batch_size = tables.shape[0]
            # Accumulate as a tensor (no .item()) to avoid forcing a
            # CUDA sync on every step; only sync once per epoch below.
            train_loss_sum += loss.detach() * batch_size
            train_count += batch_size
            train_diag.update(
                parts["mc_loss"], parts["terminal_loss"], parts["bk_loss"],
                loss.detach().item(), parts.get("bk_stats"),
                parts["mc_weighted"], parts["terminal_weighted"], parts["bk_weighted"],
            )

        train_loss = (train_loss_sum / max(train_count, 1)).item()
        history.train_losses.append(train_loss)

        model.eval()
        val_loss_sum = torch.zeros((), device=device)
        val_count = 0
        val_diag = BKDiagnostics()
        for tables, times, rewards, original_tables in val_loader:
            tables = tables.to(device)
            times = times.to(device)
            rewards = rewards.to(device)
            original_tables = original_tables.to(device)
            # compute_loss's BK term needs autograd (d/dt via
            # torch.autograd.grad) even in eval, so this is NOT wrapped in
            # torch.no_grad(); we detach explicitly below instead.
            loss, parts = compute_loss(
                model, tables, times, rewards, original_tables, cfg.boundary_loss_weight, cfg
            )
            batch_size = tables.shape[0]
            val_loss_sum += loss.detach() * batch_size
            val_count += batch_size
            val_diag.update(
                parts["mc_loss"], parts["terminal_loss"], parts["bk_loss"],
                loss.detach().item(), parts.get("bk_stats"),
                parts["mc_weighted"], parts["terminal_weighted"], parts["bk_weighted"],
            )

        val_loss = (val_loss_sum / max(val_count, 1)).item()
        history.val_losses.append(val_loss)

        epoch_bar.set_postfix(train_loss=train_loss, val_loss=val_loss)
        tqdm.write(
            f"[train_h] epoch {epoch + 1}/{cfg.num_epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )
        if cfg.use_bk_regularization or cfg.terminal_loss_weight > 0.0:
            tqdm.write(f"[train_h][diagnostics][train] {train_diag.summary()}")
            tqdm.write(f"[train_h][diagnostics][val]   {val_diag.summary()}")

        last_ckpt_path = os.path.join(cfg.checkpoint_dir, "h_model_last.pt")
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, last_ckpt_path)

        if val_loss < history.best_val_loss - cfg.early_stop_min_delta:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            best_ckpt_path = os.path.join(cfg.checkpoint_dir, "h_model_best.pt")
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                best_ckpt_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stop_patience:
                tqdm.write(
                    f"[train_h] early stopping at epoch {epoch + 1}/{cfg.num_epochs}: "
                    f"val_loss did not improve by >= {cfg.early_stop_min_delta} for "
                    f"{cfg.early_stop_patience} consecutive epochs "
                    f"(best_val_loss={history.best_val_loss:.6f} at epoch {history.best_epoch + 1})"
                )
                break

    # Always leave `model` (and the returned model) at its best-val-loss
    # weights, not the last epoch run's weights, whether or not early
    # stopping triggered.
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    history_path = os.path.join(cfg.checkpoint_dir, "train_history.pt")
    torch.save(
        {
            "train_losses": history.train_losses,
            "val_losses": history.val_losses,
            "best_val_loss": history.best_val_loss,
            "best_epoch": history.best_epoch,
        },
        history_path,
    )

    plot_path = plot_loss_curve(history, cfg.results_dir)
    print(f"[train_h] Saved loss curve to {plot_path}")

    return model, history


def plot_loss_curve(history: TrainHistory, results_dir: str) -> str:
    """Plot train/val loss vs. epoch and save to <results_dir>/loss_curve.png."""
    os.makedirs(results_dir, exist_ok=True)
    epochs = range(1, len(history.train_losses) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history.train_losses, label="train_loss")
    ax.plot(epochs, history.val_losses, label="val_loss")
    ax.axvline(history.best_epoch + 1, color="gray", linestyle="--", alpha=0.5, label="best epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("h_theta training loss")
    ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(results_dir, "loss_curve.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def load_h_model(cfg: Config, checkpoint_path: Optional[str] = None) -> HModel:
    """Load a trained HModel from checkpoint (defaults to the best checkpoint)."""
    checkpoint_path = checkpoint_path or os.path.join(
        cfg.checkpoint_dir, "h_model_best.pt"
    )
    model = HModel(cfg.m, cfg.n, cfg.total_count)
    blob = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(blob["model_state_dict"])
    model = model.to(torch.device(cfg.device))
    return model
