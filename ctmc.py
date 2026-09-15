from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch
from torch import Tensor
from tqdm import tqdm
from table_space import all_neighbors, col_sums, row_sums, sample_uniform_tables, validate_table

@dataclass
class TrajectorySnapshot:
    time: float
    table: Tensor

@dataclass
class SimulationResult:
    terminal_table: Tensor
    snapshots: List[TrajectorySnapshot]
    num_jumps: int
    path: List[TrajectorySnapshot]

def table_at_time(path: List[TrajectorySnapshot], t: float, terminal_time: float) -> Tensor:
    if t <= 0.0:
        return path[0].table
    if t >= terminal_time:
        return path[-1].table
    times = [snap.time for snap in path]
    lo, hi = (0, len(times) - 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if times[mid] <= t:
            lo = mid
        else:
            hi = mid - 1
    return path[lo].table

def sample_exponential(rate: float, generator: Optional[torch.Generator]=None, device: str='cpu') -> float:
    u = torch.rand((), generator=generator, device=device)
    u = torch.clamp(u, max=1.0 - 1e-12)
    return float(-torch.log1p(-u) / rate)

def sample_exponential_batch(rate: float, batch_n: int, generator: Optional[torch.Generator]=None, device: str='cpu') -> Tensor:
    u = torch.rand((batch_n,), generator=generator, device=device)
    u = torch.clamp(u, max=1.0 - 1e-12)
    return -torch.log1p(-u) / rate

def num_cells(m: int, n: int) -> int:
    return m * n

def num_ordered_pairs(m: int, n: int) -> int:
    d = num_cells(m, n)
    return d * (d - 1)

def propose_move(d: int, generator: Optional[torch.Generator]=None, device: str='cpu') -> Tuple[int, int]:
    while True:
        pair = torch.randint(0, d, (2,), generator=generator, device=device).tolist()
        if pair[0] != pair[1]:
            return (pair[0], pair[1])

def propose_move_batch(d: int, batch_n: int, generator: Optional[torch.Generator]=None, device: str='cpu') -> Tuple[List[int], List[int]]:
    sources = torch.randint(0, d, (batch_n,), generator=generator, device=device)
    dests = torch.randint(0, d, (batch_n,), generator=generator, device=device)
    collision_mask = sources == dests
    while collision_mask.any():
        num_collisions = int(collision_mask.sum().item())
        dests[collision_mask] = torch.randint(0, d, (num_collisions,), generator=generator, device=device)
        collision_mask = sources == dests
    return (sources.tolist(), dests.tolist())

def apply_move(x: Tensor, source: int, dest: int) -> Tensor:
    m, n = (x.shape[0], x.shape[1])
    flat = x.reshape(-1).clone()
    if flat[source] > 0:
        flat[source] -= 1
        flat[dest] += 1
    return flat.view(m, n)

def simulate_trajectory(x0: Tensor, terminal_time: float, ctmc_rate: float, snapshot_times: Optional[Sequence[float]]=None, generator: Optional[torch.Generator]=None, device: str='cpu') -> SimulationResult:
    m, n = (x0.shape[0], x0.shape[1])
    d = num_cells(m, n)
    flat = x0.reshape(-1).clone().to(device).tolist()
    t = 0.0
    num_jumps = 0
    path: List[TrajectorySnapshot] = [TrajectorySnapshot(time=0.0, table=torch.tensor(flat, device=device).view(m, n))]
    chunk_size = 4096
    dts: List[float] = []
    sources: List[int] = []
    dests: List[int] = []
    chunk_idx = 0

    def refill() -> None:
        nonlocal dts, sources, dests, chunk_idx
        dts = sample_exponential_batch(ctmc_rate, chunk_size, generator=generator, device=device).tolist()
        sources, dests = propose_move_batch(d, chunk_size, generator=generator, device=device)
        chunk_idx = 0
    refill()
    while t < terminal_time:
        if chunk_idx == chunk_size:
            refill()
        dt = dts[chunk_idx]
        source = sources[chunk_idx]
        dest = dests[chunk_idx]
        chunk_idx += 1
        next_t = t + dt
        if next_t >= terminal_time:
            t = terminal_time
            break
        if flat[source] > 0:
            flat[source] -= 1
            flat[dest] += 1
            num_jumps += 1
            path.append(TrajectorySnapshot(time=next_t, table=torch.tensor(flat, device=device).view(m, n)))
        t = next_t
    terminal_table = torch.tensor(flat, dtype=torch.float32, device=device).view(m, n)
    snapshots: List[TrajectorySnapshot] = []
    if snapshot_times:
        for st in snapshot_times:
            snapshots.append(TrajectorySnapshot(time=st, table=table_at_time(path, st, terminal_time)))
    return SimulationResult(terminal_table=terminal_table, snapshots=snapshots, num_jumps=num_jumps, path=path)

def simulate_batch(x0: Tensor, terminal_time: float, ctmc_rate: float, batch_size: int, snapshot_times: Optional[Sequence[float]]=None, generator: Optional[torch.Generator]=None, device: str='cpu') -> List[SimulationResult]:
    results = []
    for _ in tqdm(range(batch_size), desc='simulate_batch'):
        results.append(simulate_trajectory(x0, terminal_time, ctmc_rate, snapshot_times=snapshot_times, generator=generator, device=device))
    return results

def validate_trajectory_invariants(result: SimulationResult, m: int, n: int, total_count: int) -> None:
    validate_table(result.terminal_table, m, n, total_count)
    for snap in result.snapshots:
        validate_table(snap.table, m, n, total_count)

def forward_rate(ctmc_rate: float, m: int, n: int) -> float:
    K = num_ordered_pairs(m, n)
    return ctmc_rate / K

def enumerate_neighbors(x: Tensor):
    return all_neighbors(x)

def calibrate_mixing(m: int, n: int, total_count: int, terminal_time: float, ctmc_rate: float, start_tables: Dict[str, Tensor], num_samples_per_start: int=200, num_uniform_reference: int=200, seed: int=0) -> Dict[str, object]:
    generator = torch.Generator()
    generator.manual_seed(seed)

    def summarize(tables: Tensor) -> Dict[str, Tensor]:
        cellwise_mean = tables.mean(dim=0)
        cellwise_var = tables.var(dim=0, unbiased=False)
        rs = row_sums(tables)
        cs = col_sums(tables)
        zero_counts = (tables == 0).float().sum(dim=(1, 2))
        return {'cellwise_mean': cellwise_mean, 'cellwise_var': cellwise_var, 'row_sum_mean': rs.mean(dim=0), 'row_sum_var': rs.var(dim=0, unbiased=False), 'col_sum_mean': cs.mean(dim=0), 'col_sum_var': cs.var(dim=0, unbiased=False), 'zero_count_mean': zero_counts.mean(), 'zero_count_var': zero_counts.var(unbiased=False)}
    uniform_samples = sample_uniform_tables(num_uniform_reference, m, n, total_count, generator=generator)
    uniform_summary = summarize(uniform_samples)
    report: Dict[str, object] = {'uniform_reference': uniform_summary}
    for name, x0 in start_tables.items():
        terminal_tables = []
        for _ in tqdm(range(num_samples_per_start), desc=f'calibrate_mixing[{name}]'):
            result = simulate_trajectory(x0, terminal_time, ctmc_rate, generator=generator)
            terminal_tables.append(result.terminal_table)
        terminal_tables = torch.stack(terminal_tables, dim=0)
        summary = summarize(terminal_tables)
        diff = (summary['cellwise_mean'] - uniform_summary['cellwise_mean']).abs()
        summary['max_abs_cellwise_mean_diff_from_uniform'] = diff.max()
        report[name] = summary
    start_names = list(start_tables.keys())
    cross_diffs = {}
    for i in range(len(start_names)):
        for j in range(i + 1, len(start_names)):
            a, b = (start_names[i], start_names[j])
            diff = (report[a]['cellwise_mean'] - report[b]['cellwise_mean']).abs().max()
            cross_diffs[f'{a}_vs_{b}'] = diff
    report['cross_start_max_abs_diff'] = cross_diffs
    return report

def format_calibration_report(report: Dict[str, object]) -> str:
    lines = ['CTMC mixing calibration report', '=' * 40]
    for key, val in report.items():
        if key == 'cross_start_max_abs_diff':
            lines.append('\nCross-start max abs cellwise-mean differences:')
            for pair, diff in val.items():
                lines.append(f'  {pair}: {float(diff):.4f}')
            continue
        lines.append(f'\n[{key}]')
        for stat_name, stat_val in val.items():
            if isinstance(stat_val, Tensor) and stat_val.numel() > 1:
                lines.append(f'  {stat_name}: mean={stat_val.mean().item():.4f} (shape={tuple(stat_val.shape)})')
            else:
                v = stat_val.item() if isinstance(stat_val, Tensor) else stat_val
                lines.append(f'  {stat_name}: {v:.4f}')
    return '\n'.join(lines)
