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
    terminal_loss_weight: float = 0.001
    # Like BK, the terminal-boundary term is evaluated on a small random
    # ANCHOR subset of each minibatch rather than every row: it is a
    # regularizer computed via a SECOND full transformer forward pass (at
    # t=0, over original_table instead of the current (X_tau, tau) batch),
    # so running it over the full (often 512-row) batch every step doubles
    # the per-step forward/backward memory and was observed to push a
    # 12GB GPU into OOM. A small anchor subset gives the same boundary
    # signal, averaged over steps, at a small fraction of the cost.
    terminal_anchor_batch_size: int = 32
    # BK is evaluated on a small random ANCHOR subset of each minibatch (not
    # every row) -- the PDE residual is a pointwise constraint, so a handful
    # of anchors per step is enough signal without scaling cost with the
    # full (often 512-row) MC batch size. Neighbors within that subset are
    # sampled directly (positive source cell, uniform destination cell), the
    # same proposal convention as ctmc.propose_move, WITHOUT ever calling
    # all_neighbors -- a table can have thousands of valid neighbors, and
    # materializing all of them just to keep bk_num_neighbors is wasted
    # compute/memory (this previously caused a CUDA OOM at batch_size=512).
    bk_anchor_batch_size: int = 8
    bk_num_neighbors: int = 32
    bk_log_ratio_clip: float = 10.0
    # BK's double derivative (autograd through time_norm with create_graph=
    # True) can only run on the MATH scaled-dot-product-attention backend on
    # GPUs that don't support a fused kernel with a second derivative (e.g.
    # sm_70/TITAN V and observed on the V100s used here too) -- measured
    # locally at ~2x the per-step cost of MC+terminal alone. bk_every_n_steps
    # computes BK on only every Nth training/validation batch (by a
    # monotonically increasing global_step counter) instead of every batch,
    # scaling the BK loss by bk_every_n_steps on the batches where it IS
    # computed so its contribution stays an unbiased estimate of
    # bk_loss_weight * L_BK in expectation over steps.
    bk_every_n_steps: int = 4
    h_log_epsilon: float = 1e-8

    # --- Guided-sampler initial distribution ----------------------------------
    # p_T^R(x) \propto h_theta(T,x) is the mathematically exact law to draw
    # the reverse sampler's X_T from (see sample_guided.sample_x_start_reverse
    # mode="rejection"). main.cmd_sample_guided always uses "rejection" for
    # this reason -- init_check_num_probe_samples only controls the size of
    # a purely informational h_theta(T,.) constancy probe (mean/std/CV
    # printed, via check_h_constant_at_T), which does NOT select the
    # sampling mode: skipping rejection in favor of uniform_fallback is an
    # approximation regardless of how constant h_theta(T,.) looks, so it is
    # never chosen automatically.
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
    results_dir: str = 'results/pretrain_kolmogorov_hfunction_100'

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
        """Create checkpoint_dir/output_dir/results_dir, auto-avoiding a
        collision with an IN-PROGRESS run in the same checkpoint_dir or
        results_dir.

        checkpoint_dir is written to every epoch (h_model_last.pt is
        overwritten each epoch, before results_dir gets anything -- that
        only gets a report/loss-curve at the very end), so it is checked
        for existing content in addition to results_dir; either one having
        files bumps BOTH dirs together with a "-run2", "-run3", ... suffix
        (they stay paired) rather than silently letting two concurrent
        processes overwrite each other's checkpoints. An empty (or
        not-yet-created) pair of dirs is used as-is -- this only triggers
        when there's actually something there to collide with.
        """
        def has_content(path: str) -> bool:
            return os.path.isdir(path) and len(os.listdir(path)) > 0

        original_results_dir = self.results_dir
        original_checkpoint_dir = self.checkpoint_dir
        suffix = 1
        while has_content(self.results_dir) or has_content(self.checkpoint_dir):
            suffix += 1
            self.results_dir = f"{original_results_dir}-run{suffix}"
            self.checkpoint_dir = f"{original_checkpoint_dir}-run{suffix}"
        if suffix > 1:
            print(
                f"[Config.ensure_dirs] {original_results_dir} or {original_checkpoint_dir} "
                f"already has files (another run in progress?) -- using "
                f"{self.results_dir} and {self.checkpoint_dir} instead."
            )

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def summary(self) -> str:
        lines = ['Resolved configuration:']
        for k, v in self.__dict__.items():
            lines.append(f'  {k} = {v}')
        return '\n'.join(lines)
