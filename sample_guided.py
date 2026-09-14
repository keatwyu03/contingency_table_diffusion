"""Guided CTMC sampler using the Doob-h transform.

For every valid neighboring table y of the current table x, the guided rate
is

    q_t^S(x, y) = q(x, y) * h_theta(t, y) / h_theta(t, x)

computed stably in log-space:

    log q_t^S(x, y) = log q(x, y) + log h_theta(t, y) - log h_theta(t, x)

where log q(x, y) = log(ctmc_rate / K) is constant across all valid
neighbors (K = d*(d-1), d = m*n), and log h_theta is obtained via
F.logsigmoid on the model's logits (see h_model.py).

IMPORTANT APPROXIMATION: h_theta(t, x) changes continuously with t, but
between jumps we hold the guided rates fixed (computed once, at the time of
the last jump or refresh). This makes the sampler a piecewise-constant
approximation to the true time-inhomogeneous guided CTMC. To control the
resulting error, ``max_time_step`` bounds how long the sampler will go
without refreshing the rates even if no jump occurs -- i.e. rates are
recomputed either at a jump or after at most ``max_time_step`` of simulated
time, whichever comes first.

The sampler performs a genuine weighted random draw over the actual
neighboring tables (proportional to their guided rates); it does NOT always
jump to the highest-rate neighbor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from config import Config
from ctmc import num_ordered_pairs, sample_exponential
from h_model import HModel
from table_space import (
    all_neighbors,
    col_sums,
    exact_margins_satisfied,
    row_sums,
    sample_uniform_tables,
    squared_margin_error,
    soft_reward,
)


@dataclass
class GuidedSampleResult:
    """Result of one guided-CTMC trajectory."""

    terminal_table: Tensor
    num_jumps: int
    trajectory: List[Tensor] = field(default_factory=list)
    jump_times: List[float] = field(default_factory=list)


def _log_h_batch(model: HModel, tables: Tensor, t: float, total_count: int, terminal_time: float) -> Tensor:
    """Evaluate log h_theta(t, x) in one batch for a stack of tables.

    Args:
        tables: (B, m, n) unnormalized tables.
        t: unnormalized current time (scalar, same for all rows).

    Returns:
        (B,) tensor of log h_theta values.
    """
    device = next(model.parameters()).device
    tables_norm = (tables.to(device)) / total_count
    time_norm = torch.full(
        (tables.shape[0],), t / terminal_time, dtype=torch.float32, device=device
    )
    with torch.no_grad():
        return model.forward_log_h(tables_norm, time_norm)


def guided_step(
    x: Tensor,
    t: float,
    model: HModel,
    cfg: Config,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, float, bool]:
    """Compute one guided-CTMC step from state x at time t.

    Returns:
        (next_table, dt, jumped): the (possibly unchanged) next table, the
        waiting time sampled, and whether a jump actually occurred (False
        only when there are no valid neighbors, e.g. a degenerate table).
    """
    neighbors, _cell_pairs = all_neighbors(x)
    if neighbors.shape[0] == 0:
        # No valid moves (should not occur for total_count > 0).
        return x.clone(), cfg.max_time_step, False

    d = cfg.m * cfg.n
    K = num_ordered_pairs(cfg.m, cfg.n)
    log_q_base = torch.log(torch.tensor(cfg.ctmc_rate / K, dtype=torch.float32))

    log_h_x = _log_h_batch(
        model, x.unsqueeze(0), t, cfg.total_count, cfg.terminal_time
    )[0]
    log_h_neighbors = _log_h_batch(
        model, neighbors, t, cfg.total_count, cfg.terminal_time
    )

    log_ratio = log_h_neighbors - log_h_x
    log_ratio = torch.clamp(log_ratio, -cfg.log_ratio_clip, cfg.log_ratio_clip)
    log_guided_rates = log_q_base + log_ratio
    guided_rates = torch.exp(log_guided_rates)

    total_rate = guided_rates.sum()
    if total_rate.item() <= 0:
        return x.clone(), cfg.max_time_step, False

    dt = sample_exponential(float(total_rate.item()), generator=generator)

    probs = guided_rates / total_rate
    idx = torch.multinomial(probs, num_samples=1, generator=generator).item()

    next_table = neighbors[idx]
    return next_table, dt, True


def simulate_guided_trajectory(
    x0: Tensor,
    model: HModel,
    cfg: Config,
    record_trajectory: bool = False,
    generator: Optional[torch.Generator] = None,
) -> GuidedSampleResult:
    """Simulate one guided-CTMC trajectory from x0 to cfg.terminal_time.

    Rates are refreshed at every jump, and additionally at least every
    ``cfg.max_time_step`` of simulated time even if no jump occurs, bounding
    the piecewise-constant-rate approximation error (h_theta varies
    continuously with t between refreshes).
    """
    x = x0.clone()
    t = 0.0
    num_jumps = 0
    trajectory: List[Tensor] = [x.clone()] if record_trajectory else []
    jump_times: List[float] = []

    while t < cfg.terminal_time:
        next_table, dt, jumped = guided_step(x, t, model, cfg, generator=generator)
        capped_dt = min(dt, cfg.max_time_step, cfg.terminal_time - t)

        if dt <= capped_dt + 1e-12 and jumped and t + dt <= cfg.terminal_time:
            # The jump happens before the next forced refresh and before T.
            x = next_table
            t = t + dt
            num_jumps += 1
            if record_trajectory:
                trajectory.append(x.clone())
                jump_times.append(t)
        else:
            # No jump within this window: advance time to the refresh point
            # (or T) without changing the table, then recompute rates.
            t = t + capped_dt

    return GuidedSampleResult(
        terminal_table=x, num_jumps=num_jumps, trajectory=trajectory, jump_times=jump_times
    )


def simulate_guided_batch(
    model: HModel,
    cfg: Config,
    num_samples: int,
    seed: Optional[int] = None,
) -> List[GuidedSampleResult]:
    """Generate ``num_samples`` independent guided trajectories.

    Each trajectory starts from an independent X_0 ~ Uniform(E_N), drawn via
    exact stars-and-bars sampling, matching the agreed procedure rather than
    a single shared deterministic starting table.
    """
    seed = cfg.seed if seed is None else seed
    generator = torch.Generator()
    generator.manual_seed(seed)

    results = []
    for _ in range(num_samples):
        x0 = sample_uniform_tables(
            1, cfg.m, cfg.n, cfg.total_count, generator=generator
        ).squeeze(0)
        results.append(
            simulate_guided_trajectory(x0, model, cfg, generator=generator)
        )
    return results


def summarize_guided_samples(results: List[GuidedSampleResult], cfg: Config) -> str:
    """Produce a human-readable report on a batch of guided samples.

    Reports terminal row/col sums, S_2, soft reward, exact-margin
    satisfaction, jump counts, and the number of distinct terminal tables.
    """
    target_rows = torch.tensor(cfg.target_rows, dtype=torch.float32)
    target_cols = torch.tensor(cfg.target_cols, dtype=torch.float32)

    terminal_tables = torch.stack([r.terminal_table for r in results], dim=0)
    s2 = squared_margin_error(terminal_tables, target_rows, target_cols)
    rewards = soft_reward(s2, cfg.reward_gamma)
    exact = exact_margins_satisfied(terminal_tables, target_rows, target_cols)
    num_jumps = torch.tensor([r.num_jumps for r in results], dtype=torch.float32)

    distinct = len(
        {tuple(t.reshape(-1).tolist()) for t in terminal_tables}
    )

    lines = ["Guided sampling summary", "=" * 40]
    lines.append(f"num_samples = {len(results)}")
    lines.append(f"distinct terminal tables = {distinct}")
    lines.append(f"S_2: mean={s2.mean().item():.4f} min={s2.min().item():.4f} max={s2.max().item():.4f}")
    lines.append(
        f"soft reward R(x): mean={rewards.mean().item():.4f} "
        f"min={rewards.min().item():.4f} max={rewards.max().item():.4f}"
    )
    lines.append(f"exact margin satisfaction rate: {exact.float().mean().item():.4f}")
    lines.append(f"num_jumps: mean={num_jumps.mean().item():.2f} min={num_jumps.min().item():.0f} max={num_jumps.max().item():.0f}")

    best_idx = int(torch.argmin(s2).item())
    best_table = terminal_tables[best_idx]
    lines.append("\nBest sample (lowest S_2):")
    lines.append(f"  row sums = {row_sums(best_table).tolist()}")
    lines.append(f"  col sums = {col_sums(best_table).tolist()}")
    lines.append(f"  S_2 = {s2[best_idx].item():.4f}")
    lines.append(f"  R(x) = {rewards[best_idx].item():.4f}")
    lines.append(f"  exact margins satisfied = {bool(exact[best_idx].item())}")

    return "\n".join(lines)
