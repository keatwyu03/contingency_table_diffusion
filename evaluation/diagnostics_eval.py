"""Statistics for comparing guided-sampler output against the oracle/unguided
reference distributions. Pure functions (no I/O) so new statistics can be
added here and wired into evaluation_main.py without touching the sampler
or main.py, which only runs the baseline pipeline and writes results/.

Every statistic that needs a reference distribution draws a FRESH pool of
uniform tables sized to match the number of guided samples being evaluated
(not a fixed constant), so the guided batch and its reference are on equal
Monte Carlo footing.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import torch
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from table_space import col_sums, row_sums, sample_uniform_tables, soft_reward, squared_margin_error
from tuning.gamma import oracle_mean_s2 as _oracle_mean_s2


def _uniform_pool(cfg: Config, num_samples: int, generator: torch.Generator) -> Tensor:
    return sample_uniform_tables(num_samples, cfg.m, cfg.n, cfg.total_count, generator=generator)


def standardized_margin_deviation(
    guided_tables: Tensor,
    cfg: Config,
    generator: torch.Generator,
) -> Tuple[float, float]:
    """Average and max |generated margin - target margin| in std units.

    The standard deviation per row/col sum is the EMPIRICAL std of that
    margin under Uniform(E_N), estimated from a fresh uniform pool sized to
    match guided_tables (see module docstring). Deviations for every row sum
    and every column sum (across all guided tables) are pooled together
    before taking the average/max, so D_avg/D_max are single scalars over
    all m+n margins x num_samples deviations.
    """
    n_samples = guided_tables.shape[0]
    pool = _uniform_pool(cfg, n_samples, generator)
    row_std = row_sums(pool).std(dim=0)  # (m,)
    col_std = col_sums(pool).std(dim=0)  # (n,)

    target_rows = torch.tensor(cfg.target_rows, dtype=torch.float32)
    target_cols = torch.tensor(cfg.target_cols, dtype=torch.float32)

    row_dev = (row_sums(guided_tables) - target_rows).abs() / row_std  # (N, m)
    col_dev = (col_sums(guided_tables) - target_cols).abs() / col_std  # (N, n)

    all_dev = torch.cat([row_dev.reshape(-1), col_dev.reshape(-1)])
    return float(all_dev.mean().item()), float(all_dev.max().item())


def per_margin_deviation_breakdown(
    guided_tables: Tensor,
    cfg: Config,
    generator: torch.Generator,
) -> Dict[str, Tensor]:
    """Standardized deviation averaged per individual row/col margin.

    Complements standardized_margin_deviation's pooled D_avg/D_max: this
    breaks the same deviations down per margin index so a systematic bias
    on one specific row/column (vs. uniform noise across all of them) is
    visible.
    """
    n_samples = guided_tables.shape[0]
    pool = _uniform_pool(cfg, n_samples, generator)
    row_std = row_sums(pool).std(dim=0)
    col_std = col_sums(pool).std(dim=0)

    target_rows = torch.tensor(cfg.target_rows, dtype=torch.float32)
    target_cols = torch.tensor(cfg.target_cols, dtype=torch.float32)

    row_dev = (row_sums(guided_tables) - target_rows).abs() / row_std  # (N, m)
    col_dev = (col_sums(guided_tables) - target_cols).abs() / col_std  # (N, n)

    return {
        "row_mean_deviation": row_dev.mean(dim=0),  # (m,)
        "col_mean_deviation": col_dev.mean(dim=0),  # (n,)
    }


def oracle_vs_guided_mean_s2(
    guided_tables: Tensor,
    cfg: Config,
    generator: torch.Generator,
) -> Tuple[float, float]:
    """Oracle mean S_2 (importance-weighted uniform pool) vs. guided empirical mean S_2.

    Oracle value is the true reward-tilted E[S_2] under p(x) \\propto
    exp(-gamma * S_2(x)), estimated via self-normalized importance weighting
    on a fresh uniform pool (same construction as tuning/gamma.py's
    oracle_mean_s2), at the pool size matching guided_tables. This is the
    ground-truth target the guided sampler is trying to hit -- comparing it
    directly against the guided sampler's empirical mean S_2 checks
    correctness, not just "better than unguided".
    """
    n_samples = guided_tables.shape[0]
    pool = _uniform_pool(cfg, n_samples, generator)

    target_rows = torch.tensor(cfg.target_rows, dtype=torch.float32)
    target_cols = torch.tensor(cfg.target_cols, dtype=torch.float32)

    pool_s2 = squared_margin_error(pool, target_rows, target_cols)
    oracle_s2 = _oracle_mean_s2(pool_s2, cfg.reward_gamma)

    guided_s2 = squared_margin_error(guided_tables, target_rows, target_cols)
    guided_mean_s2 = float(guided_s2.mean().item())

    return oracle_s2, guided_mean_s2


def mean_reward_unguided_vs_guided(
    guided_tables: Tensor,
    cfg: Config,
    generator: torch.Generator,
) -> Tuple[float, float]:
    """Mean soft reward R(x) under an unguided uniform pool vs. under guided sampling.

    Unguided baseline: mean_i exp(-gamma * S_2(x_i)) for x_i ~ Uniform(E_N),
    from a fresh pool sized to match guided_tables. Guided: the same
    quantity computed directly on guided_tables. The gap between the two is
    the standard off-policy sanity check that guidance is actually tilting
    samples toward higher reward.
    """
    n_samples = guided_tables.shape[0]
    pool = _uniform_pool(cfg, n_samples, generator)

    target_rows = torch.tensor(cfg.target_rows, dtype=torch.float32)
    target_cols = torch.tensor(cfg.target_cols, dtype=torch.float32)

    pool_s2 = squared_margin_error(pool, target_rows, target_cols)
    unguided_mean_reward = float(soft_reward(pool_s2, cfg.reward_gamma).mean().item())

    guided_s2 = squared_margin_error(guided_tables, target_rows, target_cols)
    guided_mean_reward = float(soft_reward(guided_s2, cfg.reward_gamma).mean().item())

    return unguided_mean_reward, guided_mean_reward


def diversity(guided_tables: Tensor) -> Tuple[int, int]:
    """Number of distinct generated tables and the total sample count."""
    distinct = len({tuple(t.reshape(-1).tolist()) for t in guided_tables})
    return distinct, guided_tables.shape[0]
