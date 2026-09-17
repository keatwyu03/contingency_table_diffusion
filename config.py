from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from typing import List
import torch

_RUN_DIR_RE = re.compile(r'^run_(\d+)$')


def _next_run_number(*dirs_to_scan: str) -> int:
    found = []
    for d in dirs_to_scan:
        if not os.path.isdir(d):
            continue
        for entry in os.listdir(d):
            match = _RUN_DIR_RE.match(entry)
            if match:
                found.append(int(match.group(1)))
    return max(found, default=0) + 1


def _claim_run_number(*base_dirs: str) -> int:
    candidate = _next_run_number(
        *base_dirs, *(d + '_archive' for d in base_dirs)
    )
    while True:
        created = []
        collided = False
        for d in base_dirs:
            path = os.path.join(d, f'run_{candidate}')
            try:
                os.makedirs(path, exist_ok=False)
                created.append(path)
            except FileExistsError:
                collided = True
                break
        if not collided:
            return candidate
        for path in created:
            os.rmdir(path)
        candidate += 1

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
    generate_new_data: bool = False

    # --- Training ------------------------------------------------------------
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    grad_clip_norm: float = 1.0
    val_fraction: float = 0.1
    boundary_loss_weight: float = 0.0
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-5

    # --- CTMC pretraining (E_phi encoder) -------------------------------------
    use_pretraining: bool = True

    # --- Backward-Kolmogorov (BK) regularization for h-training --------------
    use_bk_regularization: bool = True
    bk_loss_weight: float = 0.01
    terminal_loss_weight: float = 0.001
    terminal_anchor_batch_size: int = 32
    bk_anchor_batch_size: int = 8
    bk_num_neighbors: int = 32
    bk_log_ratio_clip: float = 10.0
    bk_every_n_steps: int = 4
    h_log_epsilon: float = 1e-8

    # --- Guided-sampler initial distribution ----------------------------------
    init_check_num_probe_samples: int = 200
    pretrain_epochs: int = 50
    pretrain_learning_rate: float = 0.001
    pretrain_batch_size: int = 512
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
        """Create checkpoint_dir/output_dir/results_dir, each numbered into
        a shared fresh run_N subfolder (never reusing a number already used
        in any of the three dirs or their _archive siblings), with a
        config.txt (full resolved config) written into each. Idempotent per
        instance, since this gets called twice per pipeline run
        (main.build_config, then train_h.train_h_model).
        """
        if getattr(self, '_dirs_resolved', False):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.results_dir, exist_ok=True)
            return

        original_checkpoint_dir = self.checkpoint_dir
        original_output_dir = self.output_dir
        original_results_dir = self.results_dir
        run_number = _claim_run_number(
            original_checkpoint_dir, original_output_dir, original_results_dir
        )
        self.checkpoint_dir = os.path.join(original_checkpoint_dir, f'run_{run_number}')
        self.output_dir = os.path.join(original_output_dir, f'run_{run_number}')
        self.results_dir = os.path.join(original_results_dir, f'run_{run_number}')

        config_text = self.summary()
        for run_dir in (self.checkpoint_dir, self.output_dir, self.results_dir):
            with open(os.path.join(run_dir, 'config.txt'), 'w') as f:
                f.write(config_text)

        print(
            f'[Config.ensure_dirs] run_{run_number}: checkpoint_dir={self.checkpoint_dir} '
            f'output_dir={self.output_dir} results_dir={self.results_dir}'
        )
        self._dirs_resolved = True

    def summary(self) -> str:
        lines = ['Resolved configuration:']
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            lines.append(f'  {k} = {v}')
        return '\n'.join(lines)
