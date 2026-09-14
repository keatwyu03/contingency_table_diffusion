"""Configuration for the guided-CTMC contingency table sampler.

All hyperparameters live in a single dataclass, ``Config``. Two rates are
used in this project and must not be confused:

* ``ctmc_rate`` (default 50000.0): the constant proposal rate of the
  *unconditional* CTMC over the table space (jumps per unit time).
* ``reward_gamma`` (default 0.02): the decay rate in the soft terminal
  reward ``R(x) = exp(-reward_gamma * S_2(x))``. This is intentionally
  named ``reward_gamma`` and not ``beta`` so it is never confused with
  ``ctmc_rate``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class Config:
    """Central configuration for the contingency-table CTMC pipeline."""

    # --- Table geometry -------------------------------------------------
    m: int = 12
    n: int = 12
    total_count: int = 82

    # --- CTMC parameters --------------------------------------------------
    terminal_time: float = 1.0
    ctmc_rate: float = 50000.0

    # --- Reward parameters --------------------------------------------------
    reward_gamma: float = 0.02

    # --- Network architecture --------------------------------------------
    hidden_width: int = 126
    num_hidden_layers: int = 4
    activation: str = "silu"

    # --- Target margins ----------------------------------------------------
    target_rows: List[int] = field(
        default_factory=lambda: [6, 5, 5, 12, 12, 3, 10, 7, 3, 7, 9, 3]
    )
    target_cols: List[int] = field(
        default_factory=lambda: [13, 4, 7, 10, 8, 4, 5, 3, 4, 9, 7, 8]
    )

    # --- Randomness / device -----------------------------------------------
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Dataset generation --------------------------------------------------
    batch_size: int = 64
    num_trajectories: int = 2000
    num_time_samples_per_trajectory: int = 4

    # --- Training ------------------------------------------------------------
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    val_fraction: float = 0.1
    boundary_loss_weight: float = 0.0  # disabled by default

    # --- Guided sampling -----------------------------------------------------
    num_guided_samples: int = 32
    max_time_step: float = 0.01
    log_ratio_clip: float = 20.0

    # --- Paths -----------------------------------------------------------------
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    dataset_path: str = "outputs/h_dataset.pt"

    def __post_init__(self) -> None:
        if self.m <= 0 or self.n <= 0:
            raise ValueError(f"m and n must be positive, got m={self.m}, n={self.n}")
        if self.total_count <= 0:
            raise ValueError(f"total_count must be positive, got {self.total_count}")
        if len(self.target_rows) != self.m:
            raise ValueError(
                f"len(target_rows)={len(self.target_rows)} must equal m={self.m}"
            )
        if len(self.target_cols) != self.n:
            raise ValueError(
                f"len(target_cols)={len(self.target_cols)} must equal n={self.n}"
            )
        if sum(self.target_rows) != self.total_count:
            raise ValueError(
                f"sum(target_rows)={sum(self.target_rows)} must equal "
                f"total_count={self.total_count}"
            )
        if sum(self.target_cols) != self.total_count:
            raise ValueError(
                f"sum(target_cols)={sum(self.target_cols)} must equal "
                f"total_count={self.total_count}"
            )
        if self.ctmc_rate <= 0:
            raise ValueError(f"ctmc_rate must be positive, got {self.ctmc_rate}")
        if self.reward_gamma <= 0:
            raise ValueError(f"reward_gamma must be positive, got {self.reward_gamma}")
        if self.terminal_time <= 0:
            raise ValueError(f"terminal_time must be positive, got {self.terminal_time}")
        if self.activation.lower() != "silu":
            raise ValueError(
                f"Only 'silu' activation is supported per spec, got {self.activation}"
            )

    def ensure_dirs(self) -> None:
        """Create checkpoint/output directories if they do not exist."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def summary(self) -> str:
        """Return a human-readable summary of the resolved configuration."""
        lines = ["Resolved configuration:"]
        for k, v in self.__dict__.items():
            lines.append(f"  {k} = {v}")
        return "\n".join(lines)
