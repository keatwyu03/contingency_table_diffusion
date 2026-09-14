"""Unconditional symmetric CTMC over the contingency-table space E_N.

State space: E_N = { x in Z_{>=0}^{m x n} : sum x_{ij} = N }.

Transition mechanism: at each proposal, an ordered pair of distinct cells
(source, destination) is chosen uniformly at random from all d*(d-1)
ordered pairs, where d = m*n. If the source cell is positive, one unit of
mass moves from source to destination. If the source cell is zero, the
proposal is *rejected* and the table is unchanged -- but simulated time
still advances by the sampled waiting time.

Proposals arrive at constant rate ``ctmc_rate``; waiting times between
proposals are drawn i.i.d. from Exponential(ctmc_rate), independent of the
proposal outcome. Consequently every valid off-diagonal transition x -> y
(y reachable from x by one such move) has the same rate

    q(x, y) = ctmc_rate / K,   K = d * (d - 1)

so the unconditional chain is symmetric: q(x, y) = q(y, x) whenever y is
reachable from x by moving mass from cell a to cell b, since the reverse
move (b -> a) is an equally likely ordered pair and is valid on y (because
y has positive mass at cell a, the destination that just received it... note
validity of the *reverse* move requires that x had positive mass at the
original destination is irrelevant; validity of moving b->a on y requires
y[b] > 0, which holds since y = x with one unit moved into b).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from table_space import (
    all_neighbors,
    col_sums,
    row_sums,
    sample_uniform_tables,
    validate_table,
)


@dataclass
class TrajectorySnapshot:
    """A single recorded (time, table) pair along a trajectory."""

    time: float
    table: Tensor


@dataclass
class SimulationResult:
    """Result of simulating one CTMC trajectory to terminal time T."""

    terminal_table: Tensor
    snapshots: List[TrajectorySnapshot]
    num_jumps: int


def num_cells(m: int, n: int) -> int:
    """Return d = m * n, the number of cells."""
    return m * n


def num_ordered_pairs(m: int, n: int) -> int:
    """Return K = d * (d - 1), the number of ordered (source, dest) pairs."""
    d = num_cells(m, n)
    return d * (d - 1)


def propose_move(
    d: int, generator: Optional[torch.Generator] = None, device: str = "cpu"
) -> Tuple[int, int]:
    """Sample one ordered pair of distinct cells (source, dest) uniformly.

    Returns:
        (source_flat_idx, dest_flat_idx), source != dest.
    """
    while True:
        pair = torch.randint(
            0, d, (2,), generator=generator, device=device
        ).tolist()
        if pair[0] != pair[1]:
            return pair[0], pair[1]


def apply_move(x: Tensor, source: int, dest: int) -> Tensor:
    """Apply a single proposed move to flat-indexed cells (source, dest).

    If x has positive mass at ``source``, returns a new table with one unit
    moved from source to dest. Otherwise (source is zero), the proposal is
    rejected and an unchanged copy of x is returned.
    """
    m, n = x.shape[0], x.shape[1]
    flat = x.reshape(-1).clone()
    if flat[source] > 0:
        flat[source] -= 1
        flat[dest] += 1
    return flat.view(m, n)


def simulate_trajectory(
    x0: Tensor,
    terminal_time: float,
    ctmc_rate: float,
    snapshot_times: Optional[Sequence[float]] = None,
    generator: Optional[torch.Generator] = None,
    device: str = "cpu",
) -> SimulationResult:
    """Simulate one unconditional CTMC trajectory from x0 to terminal_time.

    Waiting times between successive proposals are drawn i.i.d. from
    Exponential(ctmc_rate). Each proposal picks a uniformly random ordered
    pair of distinct cells; if the source is zero, the proposal is rejected
    (table unchanged) but simulated time still advances by the sampled
    waiting time, matching a genuine constant-proposal-rate CTMC with
    self-loops collapsed into rejections.

    Args:
        x0: starting table, shape (m, n).
        terminal_time: T, the time horizon to simulate to.
        ctmc_rate: constant proposal rate (Exponential rate parameter).
        snapshot_times: optional sorted sequence of times in [0, T] at which
            to record the table state (nearest proposal-time <= requested
            time, i.e. the state that is current at that time).
        generator: optional torch.Generator for reproducibility.
        device: torch device string.

    Returns:
        SimulationResult with terminal_table, snapshots (one per requested
        snapshot time, in order), and the number of accepted jumps.
    """
    m, n = x0.shape[0], x0.shape[1]
    d = num_cells(m, n)
    x = x0.clone().to(device)
    t = 0.0
    num_jumps = 0

    snapshot_times = list(snapshot_times) if snapshot_times is not None else []
    snapshots: List[TrajectorySnapshot] = []
    snap_idx = 0

    while t < terminal_time:
        dt = torch.distributions.Exponential(ctmc_rate).sample(
            generator=generator
        ).item() if generator is not None else float(
            torch.distributions.Exponential(ctmc_rate).sample()
        )
        next_t = t + dt

        # Record any snapshot times that fall within (t, next_t], i.e. whose
        # current state (before this proposal takes effect) is `x`, using
        # min(next_t, terminal_time) as the boundary.
        while snap_idx < len(snapshot_times) and snapshot_times[snap_idx] < min(
            next_t, terminal_time
        ):
            snapshots.append(TrajectorySnapshot(time=snapshot_times[snap_idx], table=x.clone()))
            snap_idx += 1

        if next_t >= terminal_time:
            t = terminal_time
            break

        source, dest = propose_move(d, generator=generator, device=device)
        flat = x.reshape(-1)
        if flat[source] > 0:
            new_x = apply_move(x, source, dest)
            x = new_x
            num_jumps += 1
        # else: rejected proposal, x unchanged, but time still advances.
        t = next_t

    # Any remaining requested snapshot times (including exactly T) get the
    # final state.
    while snap_idx < len(snapshot_times):
        snapshots.append(TrajectorySnapshot(time=snapshot_times[snap_idx], table=x.clone()))
        snap_idx += 1

    return SimulationResult(terminal_table=x, snapshots=snapshots, num_jumps=num_jumps)


def simulate_batch(
    x0: Tensor,
    terminal_time: float,
    ctmc_rate: float,
    batch_size: int,
    snapshot_times: Optional[Sequence[float]] = None,
    generator: Optional[torch.Generator] = None,
    device: str = "cpu",
) -> List[SimulationResult]:
    """Generate ``batch_size`` independent trajectories from the same x0.

    Trajectories are simulated independently (each has its own random
    proposal sequence and waiting times); this is "batched" in the sense of
    producing many independent trajectories per call, though each individual
    trajectory's event-driven simulation is inherently sequential.
    """
    results = []
    for _ in range(batch_size):
        results.append(
            simulate_trajectory(
                x0,
                terminal_time,
                ctmc_rate,
                snapshot_times=snapshot_times,
                generator=generator,
                device=device,
            )
        )
    return results


def validate_trajectory_invariants(
    result: SimulationResult, m: int, n: int, total_count: int
) -> None:
    """Validate nonnegativity and total-count preservation along a trajectory."""
    validate_table(result.terminal_table, m, n, total_count)
    for snap in result.snapshots:
        validate_table(snap.table, m, n, total_count)


def forward_rate(ctmc_rate: float, m: int, n: int) -> float:
    """Return q(x, y) = ctmc_rate / K for any valid off-diagonal transition."""
    K = num_ordered_pairs(m, n)
    return ctmc_rate / K


def enumerate_neighbors(x: Tensor):
    """Enumerate all valid neighboring tables of x and their (src, dst) pairs.

    Thin wrapper around table_space.all_neighbors, re-exported here since the
    CTMC module is the natural place callers look for "neighbor" utilities.
    """
    return all_neighbors(x)


# ---------------------------------------------------------------------------
# Mixing calibration
# ---------------------------------------------------------------------------


def calibrate_mixing(
    m: int,
    n: int,
    total_count: int,
    terminal_time: float,
    ctmc_rate: float,
    start_tables: Dict[str, Tensor],
    num_samples_per_start: int = 200,
    num_uniform_reference: int = 200,
    seed: int = 0,
) -> Dict[str, object]:
    """Compare terminal CTMC samples against exact uniform samples over E_N.

    For each named starting table in ``start_tables``, simulate
    ``num_samples_per_start`` independent trajectories to ``terminal_time``
    and collect terminal states. Also draw ``num_uniform_reference`` exact
    uniform samples from E_N via stars-and-bars. Report summary statistics
    to assess how close the CTMC's terminal distribution is to uniform
    (a proxy for mixing), and how much the terminal distribution still
    depends on the starting state.

    Returns:
        A dictionary with per-start and reference summaries:
          - cellwise_mean, cellwise_var: (m, n) tensors
          - row_sum_mean, row_sum_var: (m,) tensors
          - col_sum_mean, col_sum_var: (n,) tensors
          - zero_count_mean, zero_count_var: scalars (mean/var of number of
            zero cells per table)
        plus a "max_abs_cellwise_mean_diff_from_uniform" scalar per start,
        summarizing divergence from the uniform reference.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    def summarize(tables: Tensor) -> Dict[str, Tensor]:
        # tables: (S, m, n)
        cellwise_mean = tables.mean(dim=0)
        cellwise_var = tables.var(dim=0, unbiased=False)
        rs = row_sums(tables)  # (S, m)
        cs = col_sums(tables)  # (S, n)
        zero_counts = (tables == 0).float().sum(dim=(1, 2))  # (S,)
        return {
            "cellwise_mean": cellwise_mean,
            "cellwise_var": cellwise_var,
            "row_sum_mean": rs.mean(dim=0),
            "row_sum_var": rs.var(dim=0, unbiased=False),
            "col_sum_mean": cs.mean(dim=0),
            "col_sum_var": cs.var(dim=0, unbiased=False),
            "zero_count_mean": zero_counts.mean(),
            "zero_count_var": zero_counts.var(unbiased=False),
        }

    uniform_samples = sample_uniform_tables(
        num_uniform_reference, m, n, total_count, generator=generator
    )
    uniform_summary = summarize(uniform_samples)

    report: Dict[str, object] = {"uniform_reference": uniform_summary}

    for name, x0 in start_tables.items():
        terminal_tables = []
        for _ in range(num_samples_per_start):
            result = simulate_trajectory(
                x0, terminal_time, ctmc_rate, generator=generator
            )
            terminal_tables.append(result.terminal_table)
        terminal_tables = torch.stack(terminal_tables, dim=0)
        summary = summarize(terminal_tables)
        diff = (summary["cellwise_mean"] - uniform_summary["cellwise_mean"]).abs()
        summary["max_abs_cellwise_mean_diff_from_uniform"] = diff.max()
        report[name] = summary

    # Cross-start differences: max abs difference in cellwise mean between
    # any two starting states' terminal distributions.
    start_names = list(start_tables.keys())
    cross_diffs = {}
    for i in range(len(start_names)):
        for j in range(i + 1, len(start_names)):
            a, b = start_names[i], start_names[j]
            diff = (
                report[a]["cellwise_mean"] - report[b]["cellwise_mean"]
            ).abs().max()
            cross_diffs[f"{a}_vs_{b}"] = diff
    report["cross_start_max_abs_diff"] = cross_diffs

    return report


def format_calibration_report(report: Dict[str, object]) -> str:
    """Format a calibrate_mixing report dict as a human-readable string."""
    lines = ["CTMC mixing calibration report", "=" * 40]
    for key, val in report.items():
        if key == "cross_start_max_abs_diff":
            lines.append("\nCross-start max abs cellwise-mean differences:")
            for pair, diff in val.items():
                lines.append(f"  {pair}: {float(diff):.4f}")
            continue
        lines.append(f"\n[{key}]")
        for stat_name, stat_val in val.items():
            if isinstance(stat_val, Tensor) and stat_val.numel() > 1:
                lines.append(
                    f"  {stat_name}: mean={stat_val.mean().item():.4f} "
                    f"(shape={tuple(stat_val.shape)})"
                )
            else:
                v = stat_val.item() if isinstance(stat_val, Tensor) else stat_val
                lines.append(f"  {stat_name}: {v:.4f}")
    return "\n".join(lines)
