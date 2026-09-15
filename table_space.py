from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor
from tqdm import tqdm


def validate_table(x: Tensor, m: int, n: int, total_count: int) -> None:
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
    return x.sum(dim=-1)


def col_sums(x: Tensor) -> Tensor:
    return x.sum(dim=-2)


def squared_margin_error(
    x: Tensor, target_rows: Tensor, target_cols: Tensor
) -> Tensor:
    rs = row_sums(x).to(target_rows.dtype)
    cs = col_sums(x).to(target_cols.dtype)
    row_err = ((rs - target_rows) ** 2).sum(dim=-1)
    col_err = ((cs - target_cols) ** 2).sum(dim=-1)
    return row_err + col_err


def soft_reward(s2: Tensor, gamma: float) -> Tensor:
    return torch.exp(-gamma * s2)

def exact_margins_satisfied(
    x: Tensor, target_rows: Tensor, target_cols: Tensor
) -> Tensor:
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

    d = m * n
    num_slots = total_count + d - 1
    num_bars = d - 1

    tables = torch.empty((num_samples, d), dtype=torch.float32, device=device)
    iterator = tqdm(range(num_samples), desc="sample_uniform_tables", disable=num_samples < 50)
    for i in iterator:
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
    x = torch.zeros((m, n), dtype=torch.float32)
    x[0, 0] = float(total_count)
    return x


def make_start_table_even(m: int, n: int, total_count: int) -> Tensor:
    d = m * n
    base = total_count // d
    remainder = total_count % d
    flat = torch.full((d,), float(base), dtype=torch.float32)
    flat[:remainder] += 1.0
    return flat.view(m, n)


def all_neighbors(x: Tensor) -> Tuple[Tensor, List[Tuple[int, int]]]:
    """Enumerate every valid (src, dst) single-unit move's resulting table.

    Vectorized: builds all neighbor tables in one batched tensor op (no
    per-neighbor Python-level .clone()), which matters because a table can
    have thousands of valid (src, dst) pairs (up to positive_cells * (d-1)).
    Produces the exact same set of neighbor tables and (src, dst) ordering
    as the previous nested-Python-loop implementation.
    """
    m, n = x.shape[0], x.shape[1]
    d = m * n
    flat = x.reshape(-1)
    positive_idx = torch.nonzero(flat > 0, as_tuple=False).view(-1)

    if positive_idx.numel() == 0:
        return torch.empty((0, m, n), dtype=x.dtype), []

    dst_idx = torch.arange(d, dtype=positive_idx.dtype, device=x.device)
    # (num_src, d) grid of every (src, dst) pair with dst != src, in the
    # same row-major (src outer, dst inner) order as the old nested loop.
    src_grid = positive_idx.unsqueeze(1).expand(-1, d)
    dst_grid = dst_idx.unsqueeze(0).expand(positive_idx.numel(), -1)
    keep_mask = dst_grid != src_grid

    src_flat = src_grid[keep_mask]
    dst_flat = dst_grid[keep_mask]
    num_pairs = src_flat.shape[0]

    neighbors_flat = flat.unsqueeze(0).expand(num_pairs, d).clone()
    row_idx = torch.arange(num_pairs, device=x.device)
    neighbors_flat[row_idx, src_flat] -= 1
    neighbors_flat[row_idx, dst_flat] += 1

    neighbors = neighbors_flat.view(num_pairs, m, n)
    cell_pairs = list(zip(src_flat.tolist(), dst_flat.tolist()))
    return neighbors, cell_pairs
