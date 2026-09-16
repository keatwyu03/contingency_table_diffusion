from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List
import torch

@dataclass
class Config:
    """Central configuration for the contingency-table CTMC pipeline."""

    m: int = 12
    n: int = 12
    total_count: int = 82

    # --- CTMC parameters --------------------------------------------------
    terminal_time: float = 1.0
    ctmc_rate: float = 100.0

    # --- Reward parameters --------------------------------------------------
    reward_gamma: float = 0.01

    # --- Target margins ----------------------------------------------------
    target_rows: List[int] = field(default_factory=lambda: [6, 5, 5, 12, 12, 3, 10, 7, 3, 7, 9, 3])
    target_cols: List[int] = field(default_factory=lambda: [13, 4, 7, 10, 8, 4, 5, 3, 4, 9, 7, 8])

    # --- Randomness / device -----------------------------------------------
    seed: int = 0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Dataset generation --------------------------------------------------
    batch_size: int = 512
    num_original_samples: int = 200000

    # --- Training ------------------------------------------------------------
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    grad_clip_norm: float = 1.0
    val_fraction: float = 0.1
    boundary_loss_weight: float = 0.0
    # Early stopping: training halts once val_loss fails to improve by at
    # least early_stop_min_delta for early_stop_patience consecutive epochs.
    # The final returned/checkpointed model is always the best-val-loss one
    # (not necessarily the last epoch run), whether or not early stopping
    # actually triggers.
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-5

    # --- CTMC pretraining (E_phi encoder) -------------------------------------
    # Reuses the same num_original_samples CTMC trajectories generated for
    # the h-training dataset (see h_dataset.py) -- no separate trajectory
    # set is simulated for pretraining.
    use_pretraining: bool = False

    # --- Backward-Kolmogorov (BK) regularization for h-training --------------
    # Adds two auxiliary terms to the direct Monte Carlo h-regression loss:
    # a terminal-boundary term (u_theta(0, X_0) == log R(X_0)) and a dynamics
    # term enforcing the backward-Kolmogorov PDE that log h_theta must solve
    # under the unconditional CTMC generator (see train_h.compute_loss and
    # train_h.bk_residual for the derivation and exact sign convention).
    use_bk_regularization: bool = True
    bk_loss_weight: float = 0.01
    terminal_loss_weight: float = 0.1
    # BK is evaluated on a small random ANCHOR subset of each minibatch (not
    # every row) -- the PDE residual is a pointwise constraint, so a handful
    # of anchors per step is enough signal without scaling cost with the
    # full (often 512-row) MC batch size. Neighbors within that subset are
    # sampled directly (positive source cell, uniform destination cell), the
    # same proposal convention as ctmc.propose_move, WITHOUT ever calling
    # all_neighbors -- a table can have thousands of valid neighbors, and
    # materializing all of them just to keep bk_num_neighbors is wasted
    # compute/memory (this previously caused a CUDA OOM at batch_size=512).
    bk_anchor_batch_size: int = 16
    bk_num_neighbors: int = 32
    bk_log_ratio_clip: float = 10.0
    h_log_epsilon: float = 1e-8

    # --- Guided-sampler initial distribution ----------------------------------
    # p_T^R(x) \propto h_theta(T,x) is the theoretically correct law to draw
    # the reverse sampler's X_T from (see sample_guided.sample_x_start_reverse).
    # "uniform_fallback" skips the h(T,.)-reweighting rejection step and is
    # only a valid approximation when h_theta(T,.) is itself close to constant
    # across x -- judged here by its coefficient of variation (std/mean) over
    # cfg.init_check_num_probe_samples uniform probe tables (see
    # check_h_constant_at_T), falling back to the exact "rejection" mode
    # whenever that check fails, rather than hardcoding uniform_fallback
    # regardless of what the diagnostic finds.
    init_uniform_fallback_cv_threshold: float = 0.05
    init_check_num_probe_samples: int = 200
    pretrain_epochs: int = 50
    pretrain_learning_rate: float = 0.001
    pretrain_batch_size: int = 512
    # Scales pretrain_learning_rate for E_phi during the joint h-training
    # fine-tune stage (H_omega keeps the full cfg.learning_rate).
    pretrain_encoder_lr_scale: float = 0.1

    # --- Guided sampling -----------------------------------------------------
    num_guided_samples: int = 2000
    guided_batch_size: int = 512
    max_time_step: float = 0.005

    # --- Paths -----------------------------------------------------------------
    checkpoint_dir: str = 'checkpoints'
    output_dir: str = 'outputs'
    dataset_path: str = 'outputs/h_dataset.pt'
    results_dir: str = 'results'

    def __post_init__(self) -> None:
        if self.m <= 0 or self.n <= 0:
            raise ValueError(f'm and n must be positive, got m={self.m}, n={self.n}')
        if self.total_count <= 0:
            raise ValueError(f'total_count must be positive, got {self.total_count}')
        if len(self.target_rows) != self.m:
            raise ValueError(f'len(target_rows)={len(self.target_rows)} must equal m={self.m}')
        if len(self.target_cols) != self.n:
            raise ValueError(f'len(target_cols)={len(self.target_cols)} must equal n={self.n}')
        if sum(self.target_rows) != self.total_count:
            raise ValueError(f'sum(target_rows)={sum(self.target_rows)} must equal total_count={self.total_count}')
        if sum(self.target_cols) != self.total_count:
            raise ValueError(f'sum(target_cols)={sum(self.target_cols)} must equal total_count={self.total_count}')
        if self.ctmc_rate <= 0:
            raise ValueError(f'ctmc_rate must be positive, got {self.ctmc_rate}')
        if self.reward_gamma <= 0:
            raise ValueError(f'reward_gamma must be positive, got {self.reward_gamma}')
        if self.terminal_time <= 0:
            raise ValueError(f'terminal_time must be positive, got {self.terminal_time}')
        if self.guided_batch_size <= 0:
            raise ValueError(f'guided_batch_size must be positive, got {self.guided_batch_size}')

    def ensure_dirs(self) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def summary(self) -> str:
        lines = ['Resolved configuration:']
        for k, v in self.__dict__.items():
            lines.append(f'  {k} = {v}')
        return '\n'.join(lines)
