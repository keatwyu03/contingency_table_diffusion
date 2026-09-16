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
from table_space import all_neighbors, soft_reward, squared_margin_error


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
    ) -> None:
        self.sum_mc_loss += mc_loss
        self.sum_terminal_loss += terminal_loss
        self.sum_bk_loss += bk_loss
        self.sum_total_loss += total_loss
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
            f"mean_abs_bk_residual={self.sum_abs_bk_residual / nr:.6f} "
            f"max_abs_bk_residual={self.max_abs_bk_residual:.6f} "
            f"mean_h={self.sum_h / nr:.6f} "
            f"mean_u={self.sum_u / nr:.6f} range_u=[{self.min_u:.4f},{self.max_u:.4f}] "
            f"mean_log_ratio={self.sum_log_ratio / max(self.num_log_ratio_total, 1):.6f} "
            f"range_log_ratio=[{self.min_log_ratio:.4f},{self.max_log_ratio:.4f}] "
            f"frac_clipped={self.num_log_ratio_clipped / max(self.num_log_ratio_total, 1):.6f} "
            f"mean_exit_intensity={self.sum_exit_intensity / nr:.6f}"
        )


def bk_residual(
    model: HModel,
    tables: torch.Tensor,
    times: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, dict]:
    """Backward-Kolmogorov residual for a batch of (X_tau, tau) rows.

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
    times: (B,) NORMALIZED times (tau / terminal_time, in [0, 1]).
    Un-normalized (integer-count) tables are reconstructed internally only
    for all_neighbors, which requires raw counts.

    Returns:
        (residual, stats): residual is (B,) the normalized BK residual
        R_tilde_theta(t,x) = R_theta(t,x) / (1 + lambda_t(x)) for each row
        that has at least one neighbor (rows with zero neighbors, which
        should not occur for total_count > 0, are excluded); stats is a
        dict of running diagnostics (see BKDiagnostics.update).
    """
    device = next(model.parameters()).device
    B = tables.shape[0]
    K = num_ordered_pairs(cfg.m, cfg.n)
    base_rate = cfg.ctmc_rate / K  # G_t(x,y), constant over every valid neighbor y

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
    attention_backend = (
        sdpa_kernel([SDPBackend.MATH]) if needs_double_backward else nullcontext()
    )
    with torch.enable_grad(), attention_backend:
        logit_x = model.forward_logits(table_norm, time_norm)  # (B,)
        u_x = nn.functional.logsigmoid(logit_x)  # log h_theta(t,x), stable

        du_dtime_norm = torch.autograd.grad(
            u_x.sum(), time_norm,
            create_graph=needs_double_backward, retain_graph=needs_double_backward,
        )[0]
    du_dt = du_dtime_norm / cfg.terminal_time  # chain rule: d/dt = (1/T) d/d(t/T)
    du_ds = -du_dt  # ds = -dt (see docstring)

    unnormalized_tables = torch.round(tables * cfg.total_count).to(torch.float32)

    neighbor_tables: List[torch.Tensor] = []
    neighbor_time_norms: List[float] = []
    segment_ids: List[int] = []
    neighbor_gate_weights: List[float] = []  # per-neighbor multiplier for the generator sum
    num_neighbors_per_row = torch.zeros(B, dtype=torch.float32)

    for i in range(B):
        x_i = unnormalized_tables[i].cpu()
        neighbors, _ = all_neighbors(x_i)
        c = neighbors.shape[0]
        num_neighbors_per_row[i] = c
        if c == 0:
            continue
        row_time_norm = float(times[i].item())
        if c <= cfg.bk_num_neighbors:
            # Exhaustive: cheap enough, and preferable for correctness (no
            # sampling variance) per the spec.
            neighbor_tables.append(neighbors)
            neighbor_time_norms.extend([row_time_norm] * c)
            segment_ids.extend([i] * c)
            neighbor_gate_weights.extend([1.0] * c)
        else:
            # Unbiased sampled-neighbor estimator: neighbors are uniform
            # over the c valid moves (matching the base CTMC's uniform
            # proposal, see ctmc.propose_move), so pi_t(y|x) = 1/c and the
            # importance weight lambda_t(x)/K_samples * (1/pi) simplifies to
            # (c * base_rate / bk_num_neighbors) per sampled neighbor, i.e.
            # lambda_t(x) / bk_num_neighbors.
            idx = torch.randint(0, c, (cfg.bk_num_neighbors,))
            sampled = neighbors[idx]
            neighbor_tables.append(sampled)
            neighbor_time_norms.extend([row_time_norm] * cfg.bk_num_neighbors)
            segment_ids.extend([i] * cfg.bk_num_neighbors)
            weight = float(c) / float(cfg.bk_num_neighbors)
            neighbor_gate_weights.extend([weight] * cfg.bk_num_neighbors)

    valid_rows = num_neighbors_per_row > 0
    if not valid_rows.any():
        empty_stats = {
            "sum_abs_residual": 0.0, "max_abs_residual": 0.0, "sum_h": 0.0,
            "sum_u": 0.0, "min_u": float("inf"), "max_u": float("-inf"),
            "sum_log_ratio": 0.0, "min_log_ratio": float("inf"),
            "max_log_ratio": float("-inf"), "num_log_ratio_clipped": 0,
            "num_log_ratio_total": 0, "sum_exit_intensity": 0.0, "num_rows": 0,
        }
        return torch.zeros(0, device=device), empty_stats

    all_neighbor_tables = torch.cat(neighbor_tables, dim=0)
    all_neighbor_table_norm = (all_neighbor_tables.to(device)) / cfg.total_count
    all_neighbor_time_norm = torch.tensor(
        neighbor_time_norms, dtype=torch.float32, device=device
    )
    logit_y = model.forward_logits(all_neighbor_table_norm, all_neighbor_time_norm)
    u_y = nn.functional.logsigmoid(logit_y)  # (sum_neighbors,)

    segment_ids_t = torch.tensor(segment_ids, dtype=torch.long, device=device)
    weights_t = torch.tensor(neighbor_gate_weights, dtype=torch.float32, device=device)
    u_x_per_neighbor = u_x[segment_ids_t]

    log_ratio_raw = u_y - u_x_per_neighbor
    clip = cfg.bk_log_ratio_clip
    log_ratio = torch.clamp(log_ratio_raw, min=-clip, max=clip)
    num_clipped = int(((log_ratio_raw < -clip) | (log_ratio_raw > clip)).sum().item())

    per_neighbor_term = weights_t * (torch.exp(log_ratio) - 1.0) * base_rate

    generator_term = torch.zeros(B, device=device)
    generator_term.scatter_add_(0, segment_ids_t, per_neighbor_term)

    lambda_t = num_neighbors_per_row.to(device) * base_rate  # sum_y G_t(x,y)

    valid_idx = torch.nonzero(valid_rows.to(device), as_tuple=False).view(-1)
    residual = du_ds[valid_idx] + generator_term[valid_idx]
    normalized_residual = residual / (1.0 + lambda_t[valid_idx])

    with torch.no_grad():
        h_x = torch.sigmoid(logit_x[valid_idx])
        stats = {
            "sum_abs_residual": normalized_residual.abs().sum().item(),
            "max_abs_residual": normalized_residual.abs().max().item(),
            "sum_h": h_x.sum().item(),
            "sum_u": u_x[valid_idx].sum().item(),
            "min_u": u_x[valid_idx].min().item(),
            "max_u": u_x[valid_idx].max().item(),
            "sum_log_ratio": log_ratio.sum().item(),
            "min_log_ratio": log_ratio.min().item() if log_ratio.numel() > 0 else float("inf"),
            "max_log_ratio": log_ratio.max().item() if log_ratio.numel() > 0 else float("-inf"),
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

    original_table_norm = original_tables.to(device)  # already normalized by N (see HDatasetSample)
    zero_time = torch.zeros(tables.shape[0], device=device)
    terminal_logit = model.forward_logits(original_table_norm, zero_time)
    u_terminal = nn.functional.logsigmoid(terminal_logit)
    log_r_terminal = torch.log(rewards.clamp(min=cfg.h_log_epsilon))
    terminal_loss = nn.functional.mse_loss(u_terminal, log_r_terminal)

    total_loss = mc_loss + cfg.terminal_loss_weight * terminal_loss

    unweighted_parts = {
        "mc_loss": mc_loss.detach().item(),
        "terminal_loss": terminal_loss.detach().item(),
        "bk_loss": 0.0,
    }

    if cfg.use_bk_regularization:
        residual, bk_stats = bk_residual(model, tables, times, cfg)
        if residual.numel() > 0:
            bk_loss = (residual ** 2).mean()
            total_loss = total_loss + cfg.bk_loss_weight * bk_loss
            unweighted_parts["bk_loss"] = bk_loss.detach().item()
            unweighted_parts["bk_stats"] = bk_stats

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
            )

        val_loss = (val_loss_sum / max(val_count, 1)).item()
        history.val_losses.append(val_loss)

        epoch_bar.set_postfix(train_loss=train_loss, val_loss=val_loss)
        tqdm.write(
            f"[train_h] epoch {epoch + 1}/{cfg.num_epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )
        if cfg.use_bk_regularization:
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
