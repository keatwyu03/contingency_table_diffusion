"""Gamma sweep diagnostic.

For a fixed large pool of uniform tables x ~ Uniform(E_N), computes, for
each candidate gamma:

  - oracle mean S_2 under the reward-tilted distribution p(x) \\propto
    exp(-gamma * S_2(x)) via self-normalized importance weighting on the
    SAME uniform pool (no resampling needed):

        oracle_mean_S2(gamma) = sum_i S2_i * exp(-gamma*S2_i)
                                 / sum_i exp(-gamma*S2_i)

  - mean training reward = mean_i exp(-gamma * S2_i), i.e. the average
    label h_theta would be trained against at gamma -- collapses toward 0
    as gamma grows, starving the network of usable (nonzero) targets.

Run from the project directory:
    cd /sailhome/kadenwu/contingency_table_diffusion
    python tuning/gamma.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import Config
from table_space import sample_uniform_tables, squared_margin_error

GAMMAS = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02]
POOL_SIZE = 200_000


def oracle_mean_s2(s2: torch.Tensor, gamma: float) -> float:
    log_w = -gamma * s2
    log_w = log_w - log_w.max()  # numerically stable self-normalized weights
    w = torch.exp(log_w)
    return float((s2 * w).sum() / w.sum())


def mean_reward(s2: torch.Tensor, gamma: float) -> float:
    return float(torch.exp(-gamma * s2).mean())


def effective_sample_size(s2: torch.Tensor, gamma: float) -> float:
    """ESS of the self-normalized importance weights, as a fraction of pool size."""
    log_w = -gamma * s2
    log_w = log_w - log_w.max()
    w = torch.exp(log_w)
    return float((w.sum() ** 2) / (w ** 2).sum())


def main() -> None:
    cfg = Config()
    generator = torch.Generator().manual_seed(cfg.seed)

    target_rows = torch.tensor(cfg.target_rows, dtype=torch.float32)
    target_cols = torch.tensor(cfg.target_cols, dtype=torch.float32)

    print(f"Sampling pool of {POOL_SIZE} uniform tables ({cfg.m}x{cfg.n}, total_count={cfg.total_count})...")
    pool = sample_uniform_tables(POOL_SIZE, cfg.m, cfg.n, cfg.total_count, generator=generator)
    s2 = squared_margin_error(pool, target_rows, target_cols)
    print(f"Pool S_2: mean={s2.mean().item():.2f} min={s2.min().item():.2f} max={s2.max().item():.2f}\n")

    rows = []
    for gamma in GAMMAS:
        os2 = oracle_mean_s2(s2, gamma)
        mr = mean_reward(s2, gamma)
        ess = effective_sample_size(s2, gamma)
        ess_frac = ess / POOL_SIZE
        rows.append((gamma, os2, mr, ess_frac))

    header = f"{'Gamma':>8} | {'Oracle mean S_2':>16} | {'Mean reward':>12} | {'ESS fraction':>12}"
    print(header)
    print("-" * len(header))
    for gamma, os2, mr, ess_frac in rows:
        print(f"{gamma:>8.4f} | {os2:>16.2f} | {mr:>12.5f} | {ess_frac:>12.5f}")

    out_path = os.path.join(os.path.dirname(__file__), "gamma_sweep_results.pt")
    torch.save(
        {
            "gammas": torch.tensor([r[0] for r in rows]),
            "oracle_mean_s2": torch.tensor([r[1] for r in rows]),
            "mean_reward": torch.tensor([r[2] for r in rows]),
            "ess_fraction": torch.tensor([r[3] for r in rows]),
            "pool_size": POOL_SIZE,
        },
        out_path,
    )
    print(f"\nSaved sweep results to {out_path}")


if __name__ == "__main__":
    main()
