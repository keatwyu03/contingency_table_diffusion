from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
from torch import Tensor
from tqdm import tqdm
from config import Config
from ctmc import num_ordered_pairs, sample_exponential
from h_model import HModel
from table_space import all_neighbors, col_sums, exact_margins_satisfied, row_sums, sample_uniform_tables, squared_margin_error, soft_reward

@dataclass
class GuidedSampleResult:
    terminal_table: Tensor
    num_jumps: int
    trajectory: List[Tensor] = field(default_factory=list)
    jump_times: List[float] = field(default_factory=list)
_LOG_H_EVAL_BATCH_SIZE = 512

def _log_h_batch(model: HModel, tables: Tensor, t: float, total_count: int, terminal_time: float, eval_batch_size: int=_LOG_H_EVAL_BATCH_SIZE) -> Tensor:
    device = next(model.parameters()).device
    B = tables.shape[0]
    time_scalar = t / terminal_time
    chunks: List[Tensor] = []
    with torch.no_grad():
        for start in range(0, B, eval_batch_size):
            end = min(start + eval_batch_size, B)
            chunk_tables = tables[start:end].to(device) / total_count
            chunk_time = torch.full((end - start,), time_scalar, dtype=torch.float32, device=device)
            chunks.append(model.forward_log_h(chunk_tables, chunk_time).cpu())
    return torch.cat(chunks, dim=0)

def guided_step(x: Tensor, t: float, model: HModel, cfg: Config, generator: Optional[torch.Generator]=None) -> Tuple[Tensor, float, bool]:
    neighbors, _cell_pairs = all_neighbors(x)
    if neighbors.shape[0] == 0:
        return (x.clone(), cfg.max_time_step, False)
    d = cfg.m * cfg.n
    K = num_ordered_pairs(cfg.m, cfg.n)
    log_q_base = torch.log(torch.tensor(cfg.ctmc_rate / K, dtype=torch.float32))
    log_h_x = _log_h_batch(model, x.unsqueeze(0), t, cfg.total_count, cfg.terminal_time)[0]
    log_h_neighbors = _log_h_batch(model, neighbors, t, cfg.total_count, cfg.terminal_time)
    log_ratio = log_h_neighbors - log_h_x
    log_guided_rates = log_q_base + log_ratio
    guided_rates = torch.exp(log_guided_rates)
    total_rate = guided_rates.sum()
    if total_rate.item() <= 0:
        return (x.clone(), cfg.max_time_step, False)
    dt = sample_exponential(float(total_rate.item()), generator=generator)
    probs = guided_rates / total_rate
    idx = torch.multinomial(probs, num_samples=1, generator=generator).item()
    next_table = neighbors[idx]
    return (next_table, dt, True)

def simulate_guided_trajectory(x_start: Tensor, model: HModel, cfg: Config, record_trajectory: bool=False, generator: Optional[torch.Generator]=None) -> GuidedSampleResult:
    x = x_start.clone()
    t = cfg.terminal_time
    num_jumps = 0
    trajectory: List[Tensor] = [x.clone()] if record_trajectory else []
    jump_times: List[float] = []
    while t > 0.0:
        next_table, dt, jumped = guided_step(x, t, model, cfg, generator=generator)
        capped_dt = min(dt, cfg.max_time_step, t)
        if dt <= capped_dt + 1e-12 and jumped and (t - dt >= 0.0):
            x = next_table
            t = t - dt
            num_jumps += 1
            if record_trajectory:
                trajectory.append(x.clone())
                jump_times.append(t)
        else:
            t = t - capped_dt
    return GuidedSampleResult(terminal_table=x, num_jumps=num_jumps, trajectory=trajectory, jump_times=jump_times)

def _log_h_batch_multi_time(model: HModel, tables: Tensor, times: Tensor, total_count: int, terminal_time: float, eval_batch_size: int=_LOG_H_EVAL_BATCH_SIZE) -> Tensor:
    device = next(model.parameters()).device
    B = tables.shape[0]
    chunks: List[Tensor] = []
    with torch.no_grad():
        for start in range(0, B, eval_batch_size):
            end = min(start + eval_batch_size, B)
            chunk_tables = tables[start:end].to(device) / total_count
            chunk_time = times[start:end].to(device) / terminal_time
            chunks.append(model.forward_log_h(chunk_tables, chunk_time).cpu())
    return torch.cat(chunks, dim=0)

def _guided_batch_step(active_x: Tensor, active_t: Tensor, model: HModel, cfg: Config, generator: Optional[torch.Generator]=None) -> Tuple[Tensor, Tensor, Tensor]:
    A = active_x.shape[0]
    K = num_ordered_pairs(cfg.m, cfg.n)
    log_q_base = float(torch.log(torch.tensor(cfg.ctmc_rate / K)).item())
    neighbor_tables: List[Tensor] = []
    neighbor_times: List[float] = []
    segment_ids: List[int] = []
    neighbor_counts = torch.zeros(A, dtype=torch.long)
    for i in range(A):
        neighbors, _ = all_neighbors(active_x[i])
        c = neighbors.shape[0]
        neighbor_counts[i] = c
        if c > 0:
            neighbor_tables.append(neighbors)
            neighbor_times.extend([float(active_t[i].item())] * c)
            segment_ids.extend([i] * c)
    next_x = active_x.clone()
    dt = torch.full((A,), cfg.max_time_step, dtype=torch.float32)
    jumped = torch.zeros(A, dtype=torch.bool)
    has_neighbors = neighbor_counts > 0
    if not has_neighbors.any():
        return (next_x, dt, jumped)
    own_tables = active_x
    own_times = active_t
    all_tables = torch.cat([own_tables, torch.cat(neighbor_tables, dim=0)], dim=0)
    all_times = torch.cat([own_times, torch.tensor(neighbor_times, dtype=torch.float32)], dim=0)
    log_h_all = _log_h_batch_multi_time(model, all_tables, all_times, cfg.total_count, cfg.terminal_time)
    log_h_x = log_h_all[:A]
    log_h_neighbors = log_h_all[A:]
    segment_ids_t = torch.tensor(segment_ids, dtype=torch.long)
    log_h_x_per_neighbor = log_h_x[segment_ids_t]
    log_ratio = log_h_neighbors - log_h_x_per_neighbor
    guided_rates = torch.exp(log_q_base + log_ratio)
    total_rate = torch.zeros(A, dtype=torch.float32)
    total_rate.scatter_add_(0, segment_ids_t, guided_rates)
    all_neighbors_flat = torch.cat(neighbor_tables, dim=0)
    offset = 0
    for i in range(A):
        c = int(neighbor_counts[i].item())
        if c == 0:
            continue
        rate_i = float(total_rate[i].item())
        if rate_i > 0:
            dt[i] = sample_exponential(rate_i, generator=generator)
            probs_i = guided_rates[offset:offset + c] / rate_i
            idx = torch.multinomial(probs_i, num_samples=1, generator=generator).item()
            next_x[i] = all_neighbors_flat[offset + idx]
            jumped[i] = True
        offset += c
    return (next_x, dt, jumped)

def simulate_guided_trajectories_batched(x_start: Tensor, model: HModel, cfg: Config, generator: Optional[torch.Generator]=None) -> List[GuidedSampleResult]:
    S = x_start.shape[0]
    x = x_start.clone()
    t = torch.full((S,), cfg.terminal_time, dtype=torch.float32)
    num_jumps = torch.zeros(S, dtype=torch.long)
    active = torch.ones(S, dtype=torch.bool)
    pbar = tqdm(total=S, desc='simulate_guided_batch[vectorized]')
    done_count = 0
    while active.any():
        idx = torch.nonzero(active, as_tuple=False).view(-1)
        active_x = x[idx]
        active_t = t[idx]
        next_x, dt, jumped = _guided_batch_step(active_x, active_t, model, cfg, generator=generator)
        capped_dt = torch.minimum(torch.minimum(dt, torch.full_like(dt, cfg.max_time_step)), active_t)
        take_jump = (dt <= capped_dt + 1e-12) & jumped & (active_t - dt >= 0.0)
        new_x = torch.where(take_jump.unsqueeze(-1).unsqueeze(-1), next_x, active_x)
        new_t = torch.where(take_jump, active_t - dt, active_t - capped_dt)
        x[idx] = new_x
        t[idx] = new_t
        num_jumps[idx] += take_jump.long()
        newly_done = idx[new_t <= 0.0]
        if newly_done.numel() > 0:
            active[newly_done] = False
            done_count += newly_done.numel()
            pbar.update(newly_done.numel())
    pbar.close()
    return [GuidedSampleResult(terminal_table=x[i], num_jumps=int(num_jumps[i].item())) for i in range(S)]

@dataclass
class ThinningDiagnostics:
    num_proposed: int = 0
    num_evaluated: int = 0
    num_rejected: int = 0
    num_accepted_jumps: int = 0
    num_model_calls: int = 0

    def summary(self) -> str:
        return f'[thinning diagnostics] proposed={self.num_proposed} evaluated={self.num_evaluated} rejected={self.num_rejected} accepted_jumps={self.num_accepted_jumps} model_calls={self.num_model_calls}'

def _h_batch_multi_time(model: HModel, tables: Tensor, times: Tensor, total_count: int, terminal_time: float, eval_batch_size: int=_LOG_H_EVAL_BATCH_SIZE) -> Tensor:
    device = next(model.parameters()).device
    B = tables.shape[0]
    chunks: List[Tensor] = []
    with torch.no_grad():
        for start in range(0, B, eval_batch_size):
            end = min(start + eval_batch_size, B)
            chunk_tables = tables[start:end].to(device) / total_count
            chunk_time = times[start:end].to(device) / terminal_time
            chunks.append(model.forward(chunk_tables, chunk_time).cpu())
    return torch.cat(chunks, dim=0)

class ThinningSafetyLimitExceeded(RuntimeError):
    pass

def _thinning_batch_step(active_x: Tensor, active_t: Tensor, model: HModel, cfg: Config, diagnostics: ThinningDiagnostics, generator: Optional[torch.Generator]=None, proposals_per_chunk: int=256, max_chunks: int=10000) -> Tuple[Tensor, Tensor, Tensor]:
    A = active_x.shape[0]
    d = cfg.m * cfg.n
    K = num_ordered_pairs(cfg.m, cfg.n)
    q = cfg.ctmc_rate / K
    flat_x = active_x.reshape(A, d)
    positive_lists = [torch.nonzero(flat_x[i] > 0, as_tuple=False).view(-1) for i in range(A)]
    positive_counts = torch.tensor([p.shape[0] for p in positive_lists], dtype=torch.float32)
    max_P = int(positive_counts.max().item()) if A > 0 else 0
    positive_padded = torch.zeros((A, max_P), dtype=torch.long)
    for i in range(A):
        positive_padded[i, :positive_lists[i].shape[0]] = positive_lists[i]
    log_h_x = _log_h_batch_multi_time(model, active_x, active_t, cfg.total_count, cfg.terminal_time)
    diagnostics.num_evaluated += A
    diagnostics.num_model_calls += 1
    D = positive_counts * (d - 1)
    log_Dq = torch.log(D * q)
    lambda_bar = torch.exp(log_Dq - log_h_x)
    next_x = active_x.clone()
    dt = torch.full((A,), cfg.max_time_step, dtype=torch.float32)
    jumped = torch.zeros(A, dtype=torch.bool)
    remaining_window = torch.minimum(torch.full((A,), cfg.max_time_step, dtype=torch.float32), active_t)
    window_end = remaining_window.clone()
    resolved = torch.zeros(A, dtype=torch.bool)
    resolved[D <= 0] = True
    elapsed_offset = torch.zeros(A, dtype=torch.float32)
    for _chunk in range(max_chunks):
        pending = torch.nonzero(~resolved, as_tuple=False).view(-1)
        if pending.numel() == 0:
            break
        k = proposals_per_chunk
        rates_pending = lambda_bar[pending]
        u = torch.rand((pending.numel(), k), generator=generator)
        u = torch.clamp(u, max=1.0 - 1e-12)
        interarrival = -torch.log1p(-u) / rates_pending.unsqueeze(-1)
        arrival_times = torch.cumsum(interarrival, dim=1)
        abs_arrival_times = elapsed_offset[pending].unsqueeze(-1) + arrival_times
        diagnostics.num_proposed += k * pending.numel()
        crosses_boundary = abs_arrival_times[:, -1] > window_end[pending]
        traj_of_proposal = pending.repeat_interleave(k)
        proposal_abs_time = abs_arrival_times.reshape(-1)
        P_per_proposal = positive_counts[traj_of_proposal].long()
        M = traj_of_proposal.shape[0]
        u_src = torch.rand(M, generator=generator)
        src_choice = torch.clamp((u_src * P_per_proposal.to(torch.float32)).long(), max=P_per_proposal - 1)
        src_cells = positive_padded[traj_of_proposal, src_choice]
        dst_choice = torch.randint(0, d - 1, (M,), generator=generator)
        dst_cells = dst_choice + (dst_choice >= src_cells).long()
        candidate_tables = flat_x[traj_of_proposal].clone()
        row_arange = torch.arange(M)
        candidate_tables[row_arange, src_cells] -= 1
        candidate_tables[row_arange, dst_cells] += 1
        candidate_tables_2d = candidate_tables.view(M, cfg.m, cfg.n)
        candidate_times = active_t[traj_of_proposal]
        h_y = _h_batch_multi_time(model, candidate_tables_2d, candidate_times, cfg.total_count, cfg.terminal_time)
        diagnostics.num_evaluated += M
        diagnostics.num_model_calls += 1
        accept_u = torch.rand(M, generator=generator)
        accepted = accept_u <= h_y
        diagnostics.num_rejected += int((~accepted).sum().item())
        in_window = proposal_abs_time <= window_end[traj_of_proposal] + 1e-12
        for j_local in range(pending.numel()):
            traj_global = int(pending[j_local].item())
            mask = (traj_of_proposal == traj_global) & accepted & in_window
            if mask.any():
                times_this_traj = torch.where(mask, proposal_abs_time, torch.full_like(proposal_abs_time, float('inf')))
                best = torch.argmin(times_this_traj)
                next_x[traj_global] = candidate_tables_2d[best]
                dt[traj_global] = float(times_this_traj[best].item())
                jumped[traj_global] = True
                diagnostics.num_accepted_jumps += 1
                resolved[traj_global] = True
            elif bool(crosses_boundary[j_local].item()):
                resolved[traj_global] = True
        for local_i, traj_global in enumerate(pending.tolist()):
            if not resolved[traj_global]:
                elapsed_offset[traj_global] = float(abs_arrival_times[local_i, -1].item())
    else:
        unresolved = torch.nonzero(~resolved, as_tuple=False).view(-1)
        raise ThinningSafetyLimitExceeded(f"Thinning did not resolve {unresolved.numel()} of {A} trajectories within max_chunks={max_chunks} chunks of {proposals_per_chunk} proposals each. This should not happen under normal operation; investigate cfg.max_time_step, cfg.ctmc_rate, or the model's h_theta output range rather than silently continuing.")
    return (next_x, dt, jumped)

def simulate_guided_trajectories_thinning(x_start: Tensor, model: HModel, cfg: Config, generator: Optional[torch.Generator]=None, proposals_per_chunk: int=256) -> Tuple[List[GuidedSampleResult], ThinningDiagnostics]:
    S = x_start.shape[0]
    x = x_start.clone()
    t = torch.full((S,), cfg.terminal_time, dtype=torch.float32)
    num_jumps = torch.zeros(S, dtype=torch.long)
    active = torch.ones(S, dtype=torch.bool)
    diagnostics = ThinningDiagnostics()
    pbar = tqdm(total=S, desc='simulate_guided_batch[thinning]')
    while active.any():
        idx = torch.nonzero(active, as_tuple=False).view(-1)
        active_x = x[idx]
        active_t = t[idx]
        next_x, dt, jumped = _thinning_batch_step(active_x, active_t, model, cfg, diagnostics, generator=generator, proposals_per_chunk=proposals_per_chunk)
        capped_dt = torch.minimum(torch.minimum(dt, torch.full_like(dt, cfg.max_time_step)), active_t)
        take_jump = (dt <= capped_dt + 1e-12) & jumped & (active_t - dt >= 0.0)
        new_x = torch.where(take_jump.unsqueeze(-1).unsqueeze(-1), next_x, active_x)
        new_t = torch.where(take_jump, active_t - dt, active_t - capped_dt)
        x[idx] = new_x
        t[idx] = new_t
        num_jumps[idx] += take_jump.long()
        newly_done = idx[new_t <= 0.0]
        if newly_done.numel() > 0:
            active[newly_done] = False
            pbar.update(newly_done.numel())
    pbar.close()
    results = [GuidedSampleResult(terminal_table=x[i], num_jumps=int(num_jumps[i].item())) for i in range(S)]
    return (results, diagnostics)

@dataclass
class DoobHInitResult:
    x_start_samples: Tensor
    num_proposed: int
    num_passed: int
    num_retained: int
    mean_h_proposed: float
    mean_h_accepted: float
    mode: str

    @property
    def candidate_pass_rate(self) -> float:
        return self.num_passed / self.num_proposed if self.num_proposed > 0 else 0.0

    @property
    def retained_efficiency(self) -> float:
        return self.num_retained / self.num_proposed if self.num_proposed > 0 else 0.0

def check_h_constant_at_T(model: HModel, cfg: Config, num_probe_samples: int, generator: torch.Generator) -> Tuple[float, float]:
    device = next(model.parameters()).device
    model.eval()
    candidates = sample_uniform_tables(num_probe_samples, cfg.m, cfg.n, cfg.total_count, generator=generator)
    with torch.no_grad():
        h_vals = model.forward(candidates.to(device) / cfg.total_count, torch.full((num_probe_samples,), 1.0, dtype=torch.float32, device=device)).cpu()
    return (h_vals.mean().item(), h_vals.std().item())

def sample_x_start_reverse(model: HModel, cfg: Config, num_samples: int, generator: torch.Generator, mode: str='rejection', proposal_batch_size: int=256) -> DoobHInitResult:
    device = next(model.parameters()).device
    model.eval()
    T = cfg.terminal_time
    if mode == 'uniform_fallback':
        x_start_samples = sample_uniform_tables(num_samples, cfg.m, cfg.n, cfg.total_count, generator=generator)
        with torch.no_grad():
            h_vals = model.forward(x_start_samples.to(device) / cfg.total_count, torch.full((num_samples,), 1.0, dtype=torch.float32, device=device)).cpu()
        return DoobHInitResult(x_start_samples=x_start_samples, num_proposed=num_samples, num_passed=num_samples, num_retained=num_samples, mean_h_proposed=h_vals.mean().item(), mean_h_accepted=h_vals.mean().item(), mode='uniform_fallback')
    if mode != 'rejection':
        raise ValueError(f"Unknown mode {mode!r}, expected 'rejection' or 'uniform_fallback'")
    accepted: List[Tensor] = []
    num_proposed = 0
    num_passed = 0
    num_retained = 0
    sum_h_proposed = 0.0
    sum_h_accepted = 0.0
    while len(accepted) < num_samples:
        n_needed = num_samples - len(accepted)
        batch_n = max(n_needed, proposal_batch_size)
        candidates = sample_uniform_tables(batch_n, cfg.m, cfg.n, cfg.total_count, generator=generator)
        with torch.no_grad():
            h_vals = model.forward(candidates.to(device) / cfg.total_count, torch.full((batch_n,), 1.0, dtype=torch.float32, device=device)).cpu()
        assert (h_vals >= 0.0).all() and (h_vals <= 1.0).all(), f'h_theta output must lie in [0,1] (sigmoid guarantees this); got min={h_vals.min().item()}, max={h_vals.max().item()}'
        u = torch.rand(batch_n, generator=generator)
        accept_mask = u <= h_vals
        num_proposed += batch_n
        sum_h_proposed += h_vals.sum().item()
        passed_idx = torch.nonzero(accept_mask, as_tuple=False).view(-1)
        selected_idx = passed_idx[:n_needed]
        num_passed += len(passed_idx)
        num_retained += len(selected_idx)
        sum_h_accepted += h_vals[selected_idx].sum().item()
        for idx in selected_idx.tolist():
            accepted.append(candidates[idx])
    x_start_samples = torch.stack(accepted[:num_samples], dim=0)
    return DoobHInitResult(x_start_samples=x_start_samples, num_proposed=num_proposed, num_passed=num_passed, num_retained=num_retained, mean_h_proposed=sum_h_proposed / num_proposed, mean_h_accepted=sum_h_accepted / num_retained if num_retained > 0 else float('nan'), mode='rejection')

def simulate_guided_batch(model: HModel, cfg: Config, num_samples: int, seed: Optional[int]=None, init_mode: str='rejection', method: str='thinning') -> List[GuidedSampleResult]:
    seed = cfg.seed if seed is None else seed
    generator = torch.Generator()
    generator.manual_seed(seed)
    init_result = sample_x_start_reverse(model, cfg, num_samples, generator, mode=init_mode)
    print(f'[sample_x_start_reverse mode={init_result.mode}] proposed={init_result.num_proposed} passed={init_result.num_passed} retained={init_result.num_retained} candidate_pass_rate={init_result.candidate_pass_rate:.4f} retained_efficiency={init_result.retained_efficiency:.4f} mean_h_proposed={init_result.mean_h_proposed:.6f} mean_h_accepted={init_result.mean_h_accepted:.6f}')
    if method == 'thinning':
        results, diagnostics = simulate_guided_trajectories_thinning(init_result.x_start_samples, model, cfg, generator=generator)
        print(diagnostics.summary())
        return results
    elif method == 'exhaustive':
        return simulate_guided_trajectories_batched(init_result.x_start_samples, model, cfg, generator=generator)
    else:
        raise ValueError(f"Unknown method {method!r}, expected 'thinning' or 'exhaustive'")

def summarize_guided_samples(results: List[GuidedSampleResult], cfg: Config) -> str:
    target_rows = torch.tensor(cfg.target_rows, dtype=torch.float32)
    target_cols = torch.tensor(cfg.target_cols, dtype=torch.float32)
    generated_tables = torch.stack([r.terminal_table for r in results], dim=0)
    s2 = squared_margin_error(generated_tables, target_rows, target_cols)
    rewards = soft_reward(s2, cfg.reward_gamma)
    exact = exact_margins_satisfied(generated_tables, target_rows, target_cols)
    num_jumps = torch.tensor([r.num_jumps for r in results], dtype=torch.float32)
    distinct = len({tuple(t.reshape(-1).tolist()) for t in generated_tables})
    lines = ['Guided sampling summary (generated X_0)', '=' * 40]
    lines.append(f'num_samples = {len(results)}')
    lines.append(f'distinct generated tables = {distinct}')
    lines.append(f'S_2: mean={s2.mean().item():.4f} median={s2.median().item():.4f} min={s2.min().item():.4f} max={s2.max().item():.4f}')
    lines.append(f'soft reward R(x): mean={rewards.mean().item():.4f} min={rewards.min().item():.4f} max={rewards.max().item():.4f}')
    lines.append(f'exact margin satisfaction rate: {exact.float().mean().item():.4f}')
    lines.append(f'num_jumps: mean={num_jumps.mean().item():.2f} min={num_jumps.min().item():.0f} max={num_jumps.max().item():.0f}')
    best_idx = int(torch.argmin(s2).item())
    best_table = generated_tables[best_idx]
    lines.append('\nBest sample (lowest S_2):')
    lines.append(f'  row sums = {row_sums(best_table).tolist()}')
    lines.append(f'  col sums = {col_sums(best_table).tolist()}')
    lines.append(f'  S_2 = {s2[best_idx].item():.4f}')
    lines.append(f'  R(x) = {rewards[best_idx].item():.4f}')
    lines.append(f'  exact margins satisfied = {bool(exact[best_idx].item())}')
    return '\n'.join(lines)
