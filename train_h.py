"""Training loop for h_theta(t, X_t) against the soft terminal reward.

Loss (per spec):
    L_h(theta) = E[ (h_theta(t, X_t) - exp(-gamma * S_2(X_T)))^2 ]

i.e. plain MSE regression against the continuous target in (0, 1]. BCE is
not used because the target is not a 0/1 label.

An optional boundary loss term (disabled by default via
cfg.boundary_loss_weight = 0.0) adds MSE at exactly t = T, where the
"prediction" target is exactly the terminal reward by construction; this
term is a redundant regularizer on the (t=T) samples already in the dataset
and can help the network match the boundary condition h_theta(T, x) = R(x).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

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
    """Split a dataset into train/val subsets using a fixed-seed generator."""
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        dataset, [n_train, n_val], generator=generator
    )
    return train_subset, val_subset


def compute_loss(
    model: HModel,
    tables: torch.Tensor,
    times: torch.Tensor,
    rewards: torch.Tensor,
    boundary_loss_weight: float,
) -> torch.Tensor:
    """Compute the MSE loss (plus optional boundary term) for one batch."""
    preds = model.forward(tables, times)
    loss = nn.functional.mse_loss(preds, rewards)

    if boundary_loss_weight > 0.0:
        is_boundary = times >= (1.0 - 1e-6)
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
) -> Tuple[HModel, TrainHistory]:
    """Train h_theta on ``dataset`` per the Config, with checkpointing.

    Saves ``h_model_last.pt`` and ``h_model_best.pt`` (by validation loss) to
    cfg.checkpoint_dir, along with the loss history.

    Returns:
        (trained model, TrainHistory)
    """
    set_seed(cfg.seed)
    cfg.ensure_dirs()

    device = torch.device(cfg.device)
    model = model or HModel(cfg.m, cfg.n, cfg.hidden_width, cfg.num_hidden_layers)
    model = model.to(device)

    train_subset, val_subset = split_dataset(dataset, cfg.val_fraction, cfg.seed)
    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    history = TrainHistory()

    for epoch in range(cfg.num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for tables, times, rewards in train_loader:
            tables = tables.to(device)
            times = times.to(device)
            rewards = rewards.to(device)

            optimizer.zero_grad()
            loss = compute_loss(model, tables, times, rewards, cfg.boundary_loss_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            batch_size = tables.shape[0]
            train_loss_sum += loss.item() * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)
        history.train_losses.append(train_loss)

        model.eval()
        val_loss_sum = 0.0
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
                val_loss_sum += loss.item() * batch_size
                val_count += batch_size

        val_loss = val_loss_sum / max(val_count, 1)
        history.val_losses.append(val_loss)

        print(
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

    return model, history


def load_h_model(cfg: Config, checkpoint_path: Optional[str] = None) -> HModel:
    """Load a trained HModel from checkpoint (defaults to the best checkpoint)."""
    checkpoint_path = checkpoint_path or os.path.join(
        cfg.checkpoint_dir, "h_model_best.pt"
    )
    model = HModel(cfg.m, cfg.n, cfg.hidden_width, cfg.num_hidden_layers)
    blob = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(blob["model_state_dict"])
    model = model.to(torch.device(cfg.device))
    return model
