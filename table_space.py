"""Core utilities for the space of nonnegative integer contingency tables.

The state space of interest is

    E_N = { x in Z_{>=0}^{m x n} : sum_{i,j} x_{ij} = N }

i.e. all m x n nonnegative-integer tables with a fixed total count N. This
module provides validation, margin/reward computation, exact uniform
sampling over E_N via stars-and-bars, and deterministic starting tables.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor


def validate_table(x: Tensor, m: int, n: int, total_count: int) -> None:
    """Validate that ``x`` is a valid (m, n) table with the given total count.

    Raises:
        ValueError: if shape, nonnegativity, integrality, or total count fail.
    """
    if x.dim() != 2 or x.shape[0] != m or x.shape[1] != n:
        raise ValueError(f"Table must have shape ({m}, {n}), got {tuple(x.shape)}")
    if not torch.is_floating_point(x) and not torch.is_complex(x):
        rounded_ok = True
    else:
        rounded_ok = torch.allclose(x, torch.round(x))
    if not rounded_ok:
        raise ValueError("Table must contain integer values")
    if (x < 0).any():
        raise ValueError("Table must contain only nonnegative entries")
    total = int(x.sum().item())
    if total != total_count:
        raise ValueError(f"Table total count must be {total_count}, got {total}")


def row_sums(x: Tensor) -> Tensor:
    """Return the row sums of table ``x`` (shape (..., m, n) -> (..., m))."""
    return x.sum(dim=-1)


def col_sums(x: Tensor) -> Tensor:
    """Return the column sums of table ``x`` (shape (..., m, n) -> (..., n))."""
    return x.sum(dim=-2)


def squared_margin_error(
    x: Tensor, target_rows: Tensor, target_cols: Tensor
) -> Tensor:
    """Compute S_2(x) = sum_i (row_i - r_i)^2 + sum_j (col_j - c_j)^2.

    Supports a batch of tables with leading batch dimensions; ``target_rows``
    and ``target_cols`` are broadcast against the trailing row/col dims.

    Returns:
        Tensor of shape x.shape[:-2], the batch of S_2 values.
    """
    rs = row_sums(x).to(target_rows.dtype)
    cs = col_sums(x).to(target_cols.dtype)
    row_err = ((rs - target_rows) ** 2).sum(dim=-1)
    col_err = ((cs - target_cols) ** 2).sum(dim=-1)
    return row_err + col_err


def soft_reward(s2: Tensor, gamma: float) -> Tensor:
    """Compute the soft terminal reward R(x) = exp(-gamma * S_2(x))."""
    return torch.exp(-gamma * s2)


def exact_margins_satisfied(
    x: Tensor, target_rows: Tensor, target_cols: Tensor
) -> Tensor:
    """Return a boolean tensor: True where row AND col margins exactly match."""
    rs = row_sums(x)
    cs = col_sums(x)
    rows_ok = (rs == target_rows).all(dim=-1)
    cols_ok = (cs == target_cols).all(dim=-1)
    return rows_ok & cols_ok


def sample_uniform_tables(
    num_samples: int,
    m: int,
    n: int,
    total_count: int,
    generator: torch.Generator = None,
    device: str = "cpu",
) -> Tensor:
    """Exact uniform sampling over E_N via the stars-and-bars bijection.

    E_N is in exact bijection with the set of ways to place N indistinguishable
    balls into d = m*n distinguishable bins, i.e. with multisets of size N
    drawn from d bins. A uniform *composition* of N into d nonnegative parts
    can be sampled exactly and uniformly by choosing d-1 distinct "bar"
    positions uniformly at random (without replacement) from the N + d - 1
    slots that separate N stars and d-1 bars, then reading off gap sizes.

    Concretely: choose a uniformly random (d-1)-subset of {0, ..., N+d-2}
    (the bar positions among N+d-1 total star-or-bar slots). Sorting these
    positions and taking successive gaps (accounting for the bars
    themselves) yields the d nonnegative integers that sum to N, uniformly
    over all such compositions. This is equivalent to sampling a uniformly
    random combination and is implemented by sampling a random permutation
    of the N+d-1 slots and taking the first d-1 as bar positions.

    We deliberately do NOT generate N iid cell labels drawn uniformly from
    {1, ..., d} (equivalent to multinomial cell counts): that construction is
    uniform over *sequences of ball placements*, which induces a multinomial
    (non-uniform) distribution over cell-count vectors, not a uniform
    distribution over E_N.

    Returns:
        Tensor of shape (num_samples, m, n) with dtype float32.
    """
    d = m * n
    num_slots = total_count + d - 1
    num_bars = d - 1

    tables = torch.empty((num_samples, d), dtype=torch.float32, device=device)
    for i in range(num_samples):
        perm = torch.randperm(num_slots, generator=generator, device=device)
        bar_positions = perm[:num_bars]
        bar_positions, _ = torch.sort(bar_positions)
        bar_positions_list = bar_positions.tolist()
        boundaries = [-1] + bar_positions_list + [num_slots]
        counts = [
            boundaries[k + 1] - boundaries[k] - 1 for k in range(len(boundaries) - 1)
        ]
        tables[i] = torch.tensor(counts, dtype=torch.float32, device=device)

    return tables.view(num_samples, m, n)


def make_start_table_first_cell(m: int, n: int, total_count: int) -> Tensor:
    """Deterministic starting table with all mass in the first cell (0, 0)."""
    x = torch.zeros((m, n), dtype=torch.float32)
    x[0, 0] = float(total_count)
    return x


def make_start_table_even(m: int, n: int, total_count: int) -> Tensor:
    """Deterministic starting table with mass spread as evenly as possible.

    The base amount ``total_count // (m*n)`` is placed in every cell, and the
    remainder is distributed one unit at a time to the first cells in
    row-major order, so the result is fully deterministic.
    """
    d = m * n
    base = total_count // d
    remainder = total_count % d
    flat = torch.full((d,), float(base), dtype=torch.float32)
    flat[:remainder] += 1.0
    return flat.view(m, n)


def all_neighbors(x: Tensor) -> Tuple[Tensor, List[Tuple[int, int]]]:
    """Enumerate all valid neighboring tables of ``x`` under the CTMC moves.

    A move picks an ordered pair (source cell, destination cell) of distinct
    cells; it is *valid* (produces a table different from ``x``) exactly when
    the source cell is positive. Invalid proposals (source == 0) leave the
    table unchanged and are excluded here since they are not distinct
    neighbors.

    Args:
        x: a single table of shape (m, n).

    Returns:
        neighbors: Tensor of shape (K_valid, m, n), one neighboring table per
            valid ordered (source, dest) pair.
        cell_pairs: list of (source_flat_idx, dest_flat_idx) parallel to
            ``neighbors``.
    """
    m, n = x.shape[0], x.shape[1]
    d = m * n
    flat = x.reshape(-1)
    positive_idx = torch.nonzero(flat > 0, as_tuple=False).view(-1).tolist()

    neighbor_list = []
    cell_pairs: List[Tuple[int, int]] = []
    for src in positive_idx:
        for dst in range(d):
            if dst == src:
                continue
            new_flat = flat.clone()
            new_flat[src] -= 1
            new_flat[dst] += 1
            neighbor_list.append(new_flat.view(m, n))
            cell_pairs.append((src, dst))

    if len(neighbor_list) == 0:
        neighbors = torch.empty((0, m, n), dtype=x.dtype)
    else:
        neighbors = torch.stack(neighbor_list, dim=0)
    return neighbors, cell_pairs
