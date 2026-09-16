from __future__ import annotations

import os
import random
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

from config import Config
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


def compute_loss(
    model: HModel,
    tables: torch.Tensor,
    times: torch.Tensor,
    rewards: torch.Tensor,
    boundary_loss_weight: float,
) -> torch.Tensor:
    """Compute the MSE loss (plus optional boundary term) for one batch.

    The optional boundary term targets t=0, where h_theta(0, x) = R(x)
    exactly by construction (X_0 = X_tau when tau=0) -- this is the
    boundary identity under the forward-noising orientation, distinct from
    the old scheme's t=T boundary.
    """
    preds = model.forward(tables, times)
    loss = nn.functional.mse_loss(preds, rewards)

    if boundary_loss_weight > 0.0:
        is_boundary = times <= 1e-6
        if is_boundary.any():
            boundary_preds = preds[is_boundary]
            boundary_targets = rewards[is_boundary]
            boundary_loss = nn.functional.mse_loss(boundary_preds, boundary_targets)
            loss = loss + boundary_loss_weight * boundary_loss

    return loss


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

    epoch_bar = tqdm(range(cfg.num_epochs), desc="train_h[epochs]")
    for epoch in epoch_bar:
        model.train()
        train_loss_sum = torch.zeros((), device=device)
        train_count = 0
        batch_bar = tqdm(train_loader, desc=f"epoch {epoch + 1} train", leave=False)
        for tables, times, rewards in batch_bar:
            tables = tables.to(device)
            times = times.to(device)
            rewards = rewards.to(device)

            optimizer.zero_grad()
            loss = compute_loss(model, tables, times, rewards, cfg.boundary_loss_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            batch_size = tables.shape[0]
            # Accumulate as a tensor (no .item()) to avoid forcing a
            # CUDA sync on every step; only sync once per epoch below.
            train_loss_sum += loss.detach() * batch_size
            train_count += batch_size

        train_loss = (train_loss_sum / max(train_count, 1)).item()
        history.train_losses.append(train_loss)

        model.eval()
        val_loss_sum = torch.zeros((), device=device)
        val_count = 0
        with torch.no_grad():
            for tables, times, rewards in val_loader:
                tables = tables.to(device)
                times = times.to(device)
                rewards = rewards.to(device)
                loss = compute_loss(
                    model, tables, times, rewards, cfg.boundary_loss_weight
                )
                batch_size = tables.shape[0]
                val_loss_sum += loss.detach() * batch_size
                val_count += batch_size

        val_loss = (val_loss_sum / max(val_count, 1)).item()
        history.val_losses.append(val_loss)

        epoch_bar.set_postfix(train_loss=train_loss, val_loss=val_loss)
        tqdm.write(
            f"[train_h] epoch {epoch + 1}/{cfg.num_epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        last_ckpt_path = os.path.join(cfg.checkpoint_dir, "h_model_last.pt")
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, last_ckpt_path)

        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_ckpt_path = os.path.join(cfg.checkpoint_dir, "h_model_best.pt")
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                best_ckpt_path,
            )

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
