"""Guided CTMC sampler under the forward-noising / reverse-guidance orientation.

X_0 is the original (clean) table; X_t is its forward-noised version at
time t under the unconditional CTMC; X_T is the fully noised table. h_theta
is trained (see h_dataset.py, train_h.py) to approximate

    h_theta(t, X_t) ~= E[R(X_0) | X_t]

Guided sampling runs BACKWARD: it starts at t=T with a noisy table and
moves time toward t=0, at each step biasing the (reverse-time, but rate-
identical to forward since the forward chain is symmetric with uniform
stationary law) jump proposal by h_theta(t,y)/h_theta(t,x), producing an
approximate sample of the reward-conditioned X_0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

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
    """Result of one guided-CTMC trajectory (run backward from t=T to t=0).

    ``terminal_table`` holds the sampler's output table at t=0 -- i.e. the
    generated (approximate) X_0. The field name is kept as "terminal_table"
    for compatibility with the reporting code below, even though under this
    orientation it is the START of the modeled process (X_0), not the end.
    """

    terminal_table: Tensor
    num_jumps: int
    trajectory: List[Tensor] = field(default_factory=list)
    jump_times: List[float] = field(default_factory=list)


_LOG_H_EVAL_BATCH_SIZE = 512


def _log_h_batch(
    model: HModel,
    tables: Tensor,
    t: float,
    total_count: int,
    terminal_time: float,
    eval_batch_size: int = _LOG_H_EVAL_BATCH_SIZE,
) -> Tensor:
    """Evaluate log h_theta(t, x) for a stack of tables, chunked internally.

    Args:
        tables: (B, m, n) unnormalized tables. B can be large (a table with
            many nonzero cells can have thousands of valid neighbors), so
            the forward pass is split into chunks of at most
            ``eval_batch_size`` rows to bound peak GPU memory -- pushing
            all of B through the transformer at once can exhaust GPU memory
            (observed as CUDA OOM) for a large neighbor set.
        t: unnormalized current time (scalar, same for all rows).

    Returns:
        (B,) tensor of log h_theta values, on CPU.
    """
    device = next(model.parameters()).device
    B = tables.shape[0]
    time_scalar = t / terminal_time

    chunks: List[Tensor] = []
    with torch.no_grad():
        for start in range(0, B, eval_batch_size):
            end = min(start + eval_batch_size, B)
            chunk_tables = (tables[start:end].to(device)) / total_count
            chunk_time = torch.full(
                (end - start,), time_scalar, dtype=torch.float32, device=device
            )
            chunks.append(model.forward_log_h(chunk_tables, chunk_time).cpu())
    return torch.cat(chunks, dim=0)


def guided_step(
    x: Tensor,
    t: float,
    model: HModel,
    cfg: Config,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, float, bool]:
    """Compute one guided-CTMC step from state x at forward-time label t.

    The forward CTMC is symmetric and Uniform(E_N) is its stationary
    distribution, so the reverse-time proposal rate equals the forward rate:
    q_t^reverse(x,y) = q_t^F(y,x) = q_t^F(x,y) = ctmc_rate / K. The guided
    rate is therefore q_t^guided(x,y) = (ctmc_rate/K) * h_theta(t,y)/h_theta(t,x)
    for every valid neighbor y, exactly as in the forward-time case -- only
    the direction time moves in the caller (simulate_guided_trajectory)
    differs.

    Kept for single-trajectory use (e.g. tests); simulate_guided_batch below
    uses a vectorized batched version of this same logic instead of calling
    this per sample, since the model forward pass is by far the dominant
    cost and batching it across live samples is what makes many-sample runs
    tractable.

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

    # Neural network forward pass runs on the model's device (possibly cuda,
    # internally chunked -- see _log_h_batch), but everything else in this
    # function (the table state, RNG generator, and multinomial draw)
    # operates on CPU, which is what _log_h_batch already returns.
    log_h_x = _log_h_batch(
        model, x.unsqueeze(0), t, cfg.total_count, cfg.terminal_time
    )[0]
    log_h_neighbors = _log_h_batch(
        model, neighbors, t, cfg.total_count, cfg.terminal_time
    )

    log_ratio = log_h_neighbors - log_h_x
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
    x_start: Tensor,
    model: HModel,
    cfg: Config,
    record_trajectory: bool = False,
    generator: Optional[torch.Generator] = None,
) -> GuidedSampleResult:
    """Simulate one guided-CTMC trajectory backward from t=T to t=0.

    ``x_start`` is the initial noisy table at t=T (drawn from the guided
    initial law -- see sample_x0_doob_h_reverse). Every waiting-time
    increment moves time BACKWARD: t_new = t - dt. If t_new <= 0, the
    sampler stops without performing a jump beyond time zero, and the
    current table is returned as the generated (approximate) X_0.

    Rates are refreshed at every jump, and additionally at least every
    ``cfg.max_time_step`` of simulated time even if no jump occurs, bounding
    the piecewise-constant-rate approximation error (h_theta varies
    continuously with t between refreshes).

    Despite the "terminal_table" field name (kept for compatibility with
    the reporting/summary code), the returned table here is the sampler's
    OUTPUT at t=0, i.e. the generated X_0 -- not a t=T terminal state.

    This single-trajectory path (via guided_step) is O(1) model forward
    passes per sample per step; simulate_guided_batch batches these across
    samples instead and should be preferred whenever num_samples > 1.
    """
    x = x_start.clone()
    t = cfg.terminal_time
    num_jumps = 0
    trajectory: List[Tensor] = [x.clone()] if record_trajectory else []
    jump_times: List[float] = []

    while t > 0.0:
        next_table, dt, jumped = guided_step(x, t, model, cfg, generator=generator)
        capped_dt = min(dt, cfg.max_time_step, t)

        if dt <= capped_dt + 1e-12 and jumped and t - dt >= 0.0:
            # The jump happens before the next forced refresh and before t=0.
            x = next_table
            t = t - dt
            num_jumps += 1
            if record_trajectory:
                trajectory.append(x.clone())
                jump_times.append(t)
        else:
            # No jump within this window: advance time backward to the
            # refresh point (or 0) without changing the table, then
            # recompute rates.
            t = t - capped_dt

    return GuidedSampleResult(
        terminal_table=x, num_jumps=num_jumps, trajectory=trajectory, jump_times=jump_times
    )


def _log_h_batch_multi_time(
    model: HModel,
    tables: Tensor,
    times: Tensor,
    total_count: int,
    terminal_time: float,
    eval_batch_size: int = _LOG_H_EVAL_BATCH_SIZE,
) -> Tensor:
    """Like _log_h_batch, but each row of ``tables`` has its own time.

    Args:
        tables: (B, m, n) unnormalized tables.
        times: (B,) unnormalized times, one per row (not a shared scalar).

    Returns:
        (B,) tensor of log h_theta values, on CPU.
    """
    device = next(model.parameters()).device
    B = tables.shape[0]

    chunks: List[Tensor] = []
    with torch.no_grad():
        for start in range(0, B, eval_batch_size):
            end = min(start + eval_batch_size, B)
            chunk_tables = (tables[start:end].to(device)) / total_count
            chunk_time = (times[start:end].to(device)) / terminal_time
            chunks.append(model.forward_log_h(chunk_tables, chunk_time).cpu())
    return torch.cat(chunks, dim=0)


def _guided_batch_step(
    active_x: Tensor,
    active_t: Tensor,
    model: HModel,
    cfg: Config,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Vectorized version of guided_step over all currently-active samples.

    Builds every active sample's neighbor set, concatenates them into one
    (sum_of_neighbor_counts, m, n) tensor tagged with a per-row segment id,
    and evaluates h_theta with a SINGLE batched (chunked) model forward pass
    covering all active samples' current tables and all their neighbors
    together -- replacing what used to be ``len(active)`` separate forward
    passes per outer step.

    Args:
        active_x: (A, m, n) current tables of the A active samples.
        active_t: (A,) current times of the A active samples.

    Returns:
        (next_x, dt, jumped): (A, m, n) next tables (unchanged where no jump
        occurred), (A,) sampled waiting times, (A,) bool jump flags -- same
        per-sample semantics as guided_step, just batched.
    """
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
        return next_x, dt, jumped

    # One batched forward pass covers every active sample's own table (at
    # its own time) AND every active sample's neighbor tables -- this is the
    # single model call that replaces the old one-call-per-sample loop.
    own_tables = active_x
    own_times = active_t
    all_tables = torch.cat([own_tables, torch.cat(neighbor_tables, dim=0)], dim=0)
    all_times = torch.cat(
        [own_times, torch.tensor(neighbor_times, dtype=torch.float32)], dim=0
    )
    log_h_all = _log_h_batch_multi_time(
        model, all_tables, all_times, cfg.total_count, cfg.terminal_time
    )
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
            probs_i = guided_rates[offset : offset + c] / rate_i
            idx = torch.multinomial(probs_i, num_samples=1, generator=generator).item()
            next_x[i] = all_neighbors_flat[offset + idx]
            jumped[i] = True
        offset += c

    return next_x, dt, jumped


def simulate_guided_trajectories_batched(
    x_start: Tensor,
    model: HModel,
    cfg: Config,
    generator: Optional[torch.Generator] = None,
) -> List[GuidedSampleResult]:
    """Simulate many independent guided-CTMC trajectories together, backward T->0.

    Semantically equivalent to calling simulate_guided_trajectory once per
    row of x_start (each trajectory has its own clock and stopping time),
    but every model forward pass is batched across all samples that are
    still active at that outer-loop iteration -- see _guided_batch_step.
    This is what makes many-sample guided sampling fast: previously each of
    the num_samples trajectories issued its own sequence of ~terminal_time /
    max_time_step model calls one at a time.

    Args:
        x_start: (S, m, n) independent starting tables at t=T.

    Returns:
        list of S GuidedSampleResult, in the same order as x_start.
    """
    S = x_start.shape[0]
    x = x_start.clone()
    t = torch.full((S,), cfg.terminal_time, dtype=torch.float32)
    num_jumps = torch.zeros(S, dtype=torch.long)
    active = torch.ones(S, dtype=torch.bool)

    pbar = tqdm(total=S, desc="simulate_guided_batch[vectorized]")
    done_count = 0

    while active.any():
        idx = torch.nonzero(active, as_tuple=False).view(-1)
        active_x = x[idx]
        active_t = t[idx]

        next_x, dt, jumped = _guided_batch_step(
            active_x, active_t, model, cfg, generator=generator
        )
        capped_dt = torch.minimum(
            torch.minimum(dt, torch.full_like(dt, cfg.max_time_step)), active_t
        )

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

    return [
        GuidedSampleResult(
            terminal_table=x[i], num_jumps=int(num_jumps[i].item())
        )
        for i in range(S)
    ]


@dataclass
class ThinningDiagnostics:
    """Running counters for the batched thinning/rejection sampler.

    Accumulated across an entire simulate_guided_trajectories_thinning call
    (all trajectories, all steps) so the caller can sanity-check how much
    work thinning actually did relative to the exhaustive alternative.
    """

    num_proposed: int = 0  # total candidate proposals generated (all chunks)
    num_evaluated: int = 0  # total candidates actually scored by the model
    num_rejected: int = 0  # evaluated candidates whose Bernoulli test failed
    num_accepted_jumps: int = 0  # proposals that were the FIRST accept in their window (i.e. actual jumps taken)
    num_model_calls: int = 0  # number of batched model forward calls issued

    def summary(self) -> str:
        return (
            f"[thinning diagnostics] proposed={self.num_proposed} "
            f"evaluated={self.num_evaluated} rejected={self.num_rejected} "
            f"accepted_jumps={self.num_accepted_jumps} "
            f"model_calls={self.num_model_calls}"
        )


def _h_batch_multi_time(
    model: HModel,
    tables: Tensor,
    times: Tensor,
    total_count: int,
    terminal_time: float,
    eval_batch_size: int = _LOG_H_EVAL_BATCH_SIZE,
) -> Tensor:
    """Like _log_h_batch_multi_time, but returns h_theta = sigmoid(logit)
    directly (not log h) -- the thinning sampler's acceptance probability
    and dominating-rate normalization both need h_theta itself, per the
    exact-thinning construction (see simulate_guided_trajectories_thinning),
    not its log.
    """
    device = next(model.parameters()).device
    B = tables.shape[0]

    chunks: List[Tensor] = []
    with torch.no_grad():
        for start in range(0, B, eval_batch_size):
            end = min(start + eval_batch_size, B)
            chunk_tables = (tables[start:end].to(device)) / total_count
            chunk_time = (times[start:end].to(device)) / terminal_time
            chunks.append(model.forward(chunk_tables, chunk_time).cpu())
    return torch.cat(chunks, dim=0)


def _thinning_batch_step(
    active_x: Tensor,
    active_t: Tensor,
    model: HModel,
    cfg: Config,
    diagnostics: ThinningDiagnostics,
    generator: Optional[torch.Generator] = None,
    proposals_per_chunk: int = 256,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Exact batched thinning/rejection step, replacing exhaustive enumeration.

    For each active trajectory x at time t, let d = m*n, P = number of
    positive cells in x, D = P*(d-1) the number of valid neighbors, and
    q = ctmc_rate / K the existing base per-pair rate. Writing
    h_x = h_theta(t, x), the dominating proposal rate is

        Lambda_bar = D * q / h_x

    computed in log space as exp(log(D*q) - log_h_x), where log_h_x comes
    from the model's stable logsigmoid-based forward_log_h -- h_x itself is
    never computed or floored; only its log is used.

    Proposals arrive as a Poisson process at rate Lambda_bar; each proposal
    samples a candidate neighbor y uniformly among the D valid neighbors
    (src uniform over positive cells, dst uniform over the other d-1 cells)
    and is accepted with probability h_y = h_theta(t, y). This is exact
    relative to the piecewise-constant-rate approximation already used by
    the exhaustive sampler, because the accepted rate for landing on any
    particular neighbor y works out to

        Lambda_bar * (1/D) * h_y = (D*q/h_x) * (1/D) * h_y = q * h_y / h_x

    which is exactly the guided rate q * h_theta(t,y)/h_theta(t,x) used by
    the exhaustive path -- see guided_step's docstring.

    Window handling: proposals are generated in chunks per pending
    trajectory. Within each chunk, only the proposals whose arrival time
    falls within the window are actually constructed and evaluated
    (candidate table built, model call, Bernoulli test) -- proposals past
    window_end are never evaluated, only their arrival times are used, as
    proof the window has been fully covered. If a chunk contains at least
    one proposal whose arrival time is past window_end, then every
    in-window proposal for that trajectory has now been generated and
    evaluated -- so if none of them were accepted, the trajectory is
    definitively resolved with no jump (advanced to window_end). A
    trajectory whose whole chunk lands before window_end cannot be resolved
    yet -- whether a later, not-yet-drawn proposal within the window would
    accept is still unknown -- so another chunk is drawn for it, continuing
    the same Poisson process (exponential memorylessness makes fresh
    interarrival draws from the last-drawn arrival time exact, not an
    approximation).

    Never calls all_neighbors -- candidate tables are built directly from
    sampled (src, dst) pairs.

    Returns:
        (next_x, dt, jumped): same per-sample contract as _guided_batch_step.
    """
    A = active_x.shape[0]
    d = cfg.m * cfg.n
    K = num_ordered_pairs(cfg.m, cfg.n)
    q = cfg.ctmc_rate / K

    flat_x = active_x.reshape(A, d)
    positive_lists = [
        torch.nonzero(flat_x[i] > 0, as_tuple=False).view(-1) for i in range(A)
    ]
    positive_counts = torch.tensor(
        [p.shape[0] for p in positive_lists], dtype=torch.float32
    )
    # Dense (A, max_P) padded positive-cell index table, built once per call
    # (positive cells are fixed for the duration of this window -- a jump
    # resolves and exits its trajectory rather than changing them mid-loop)
    # so per-chunk src sampling below can be a single vectorized gather
    # instead of a Python loop over proposals.
    max_P = int(positive_counts.max().item()) if A > 0 else 0
    positive_padded = torch.zeros((A, max_P), dtype=torch.long)
    for i in range(A):
        positive_padded[i, : positive_lists[i].shape[0]] = positive_lists[i]

    # log h_x for every active trajectory's OWN current table, at its own
    # time, via the model's numerically stable logsigmoid path -- h_x
    # itself is never materialized or floored.
    log_h_x = _log_h_batch_multi_time(
        model, active_x, active_t, cfg.total_count, cfg.terminal_time
    )
    diagnostics.num_evaluated += A
    diagnostics.num_model_calls += 1

    D = positive_counts * (d - 1)
    log_Dq = torch.log(D * q)
    lambda_bar = torch.exp(log_Dq - log_h_x)  # (A,)

    next_x = active_x.clone()
    dt = torch.full((A,), cfg.max_time_step, dtype=torch.float32)
    jumped = torch.zeros(A, dtype=torch.bool)

    remaining_window = torch.minimum(
        torch.full((A,), cfg.max_time_step, dtype=torch.float32), active_t
    )
    # window_end[i]: elapsed-time boundary (from the start of this call) at
    # which trajectory i's rates must be refreshed regardless of acceptance.
    window_end = remaining_window.clone()

    resolved = torch.zeros(A, dtype=torch.bool)  # trajectory decided this call
    resolved[D <= 0] = True  # degenerate: no positive cells at all (shouldn't occur for total_count > 0)
    elapsed_offset = torch.zeros(A, dtype=torch.float32)  # time already generated-and-cleared, per trajectory

    while True:
        pending = torch.nonzero(~resolved, as_tuple=False).view(-1)
        if pending.numel() == 0:
            break

        k = proposals_per_chunk

        # Per-trajectory exponential interarrival times -> cumulative
        # arrival times (elapsed since the start of THIS chunk).
        rates_pending = lambda_bar[pending]
        u = torch.rand((pending.numel(), k), generator=generator)
        u = torch.clamp(u, max=1.0 - 1e-12)
        interarrival = -torch.log1p(-u) / rates_pending.unsqueeze(-1)
        arrival_times = torch.cumsum(interarrival, dim=1)  # (P, k), elapsed within this chunk
        abs_arrival_times = (
            elapsed_offset[pending].unsqueeze(-1) + arrival_times
        )  # elapsed since window start
        diagnostics.num_proposed += k * pending.numel()

        # A trajectory's window is FULLY covered by this chunk once its
        # chunk contains at least one proposal past window_end (arrival
        # times are increasing along dim=1, so the last column is the
        # latest arrival drawn) -- that later, out-of-window arrival is
        # never evaluated through the model; it only serves as PROOF that
        # every in-window proposal for the trajectory has now been drawn.
        crosses_boundary = abs_arrival_times[:, -1] > window_end[pending]

        traj_of_proposal_full = pending.repeat_interleave(k)  # (P*k,) trajectory-major order
        proposal_abs_time_full = abs_arrival_times.reshape(-1)  # matches trajectory-major order
        in_window_full = proposal_abs_time_full <= window_end[traj_of_proposal_full] + 1e-12

        # Only in-window proposals are actually constructed and evaluated --
        # everything past window_end is discarded here without ever
        # building a candidate table or calling the model.
        traj_of_proposal = traj_of_proposal_full[in_window_full]
        proposal_abs_time = proposal_abs_time_full[in_window_full]
        M = traj_of_proposal.shape[0]

        if M > 0:
            P_per_proposal = positive_counts[traj_of_proposal].long()  # (M,)

            # Vectorized "uniform integer in [0, P_i)" per proposal row: draw a
            # continuous uniform and floor-scale by that row's own P_i, then
            # gather from the padded positive-cell table -- avoids a Python
            # loop over proposals (M can be in the thousands per chunk).
            u_src = torch.rand(M, generator=generator)
            src_choice = torch.clamp(
                (u_src * P_per_proposal.to(torch.float32)).long(), max=P_per_proposal - 1
            )
            src_cells = positive_padded[traj_of_proposal, src_choice]

            dst_choice = torch.randint(0, d - 1, (M,), generator=generator)
            dst_cells = dst_choice + (dst_choice >= src_cells).long()

            candidate_tables = flat_x[traj_of_proposal].clone()
            row_arange = torch.arange(M)
            candidate_tables[row_arange, src_cells] -= 1
            candidate_tables[row_arange, dst_cells] += 1
            candidate_tables_2d = candidate_tables.view(M, cfg.m, cfg.n)

            candidate_times = active_t[traj_of_proposal]
            h_y = _h_batch_multi_time(
                model, candidate_tables_2d, candidate_times, cfg.total_count, cfg.terminal_time
            )
            diagnostics.num_evaluated += M
            diagnostics.num_model_calls += 1

            accept_u = torch.rand(M, generator=generator)
            accepted = accept_u <= h_y
            diagnostics.num_rejected += int((~accepted).sum().item())
        else:
            accepted = torch.zeros(0, dtype=torch.bool)
            candidate_tables_2d = torch.empty((0, cfg.m, cfg.n), dtype=flat_x.dtype)

        # For each pending trajectory, find its FIRST accepted proposal
        # (lowest abs arrival time among accepted==True, all already
        # guaranteed to be within the window since only in-window
        # proposals were evaluated above).
        for j_local in range(pending.numel()):
            traj_global = int(pending[j_local].item())
            mask = (traj_of_proposal == traj_global) & accepted
            if mask.any():
                times_this_traj = torch.where(
                    mask, proposal_abs_time, torch.full_like(proposal_abs_time, float("inf"))
                )
                best = torch.argmin(times_this_traj)
                next_x[traj_global] = candidate_tables_2d[best]
                dt[traj_global] = float(times_this_traj[best].item())
                jumped[traj_global] = True
                diagnostics.num_accepted_jumps += 1
                resolved[traj_global] = True
            elif bool(crosses_boundary[j_local].item()):
                # Every in-window proposal for this trajectory has been
                # generated and evaluated (this chunk's last drawn arrival
                # was past window_end) and none accepted -> definitively no
                # jump within the window. Do NOT restart or draw further
                # proposals for it.
                resolved[traj_global] = True

        # Trajectories that did not cross the boundary this chunk and had
        # no acceptance must continue: draw another chunk starting from
        # where this one left off (memoryless exponential -> exact, not an
        # approximation).
        for local_i, traj_global in enumerate(pending.tolist()):
            if not resolved[traj_global]:
                elapsed_offset[traj_global] = float(abs_arrival_times[local_i, -1].item())

    return next_x, dt, jumped


def simulate_guided_trajectories_thinning(
    x_start: Tensor,
    model: HModel,
    cfg: Config,
    generator: Optional[torch.Generator] = None,
    proposals_per_chunk: int = 256,
    pbar: Optional[tqdm] = None,
) -> Tuple[List[GuidedSampleResult], ThinningDiagnostics]:
    """Simulate many guided-CTMC trajectories via exact batched thinning.

    Drop-in alternative to simulate_guided_trajectories_batched: same
    per-trajectory semantics (independent clock, backward T->0, same
    GuidedSampleResult output, same piecewise-constant-rate refresh every
    cfg.max_time_step), but avoids ever enumerating the full neighbor set
    (all_neighbors) via exact thinning/rejection -- see
    _thinning_batch_step for the construction and its exactness argument.

    Returns:
        (results, diagnostics): results in the same order as x_start;
        diagnostics accumulates proposal/evaluation/rejection/jump/model-call
        counts across the whole run (see ThinningDiagnostics).

    If ``pbar`` is given, its progress is updated (by newly-completed
    trajectory count) instead of creating and closing a fresh bar -- lets a
    caller share one bar across multiple calls (e.g. one per sub-batch)
    without each call starting a new line.
    """
    S = x_start.shape[0]
    x = x_start.clone()
    t = torch.full((S,), cfg.terminal_time, dtype=torch.float32)
    num_jumps = torch.zeros(S, dtype=torch.long)
    active = torch.ones(S, dtype=torch.bool)
    diagnostics = ThinningDiagnostics()

    owns_pbar = pbar is None
    if pbar is None:
        pbar = tqdm(total=S, desc="simulate_guided_batch[thinning]")

    while active.any():
        idx = torch.nonzero(active, as_tuple=False).view(-1)
        active_x = x[idx]
        active_t = t[idx]

        next_x, dt, jumped = _thinning_batch_step(
            active_x,
            active_t,
            model,
            cfg,
            diagnostics,
            generator=generator,
            proposals_per_chunk=proposals_per_chunk,
        )
        capped_dt = torch.minimum(
            torch.minimum(dt, torch.full_like(dt, cfg.max_time_step)), active_t
        )

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
    if owns_pbar:
        pbar.close()

    results = [
        GuidedSampleResult(
            terminal_table=x[i], num_jumps=int(num_jumps[i].item())
        )
        for i in range(S)
    ]
    return results, diagnostics


@dataclass
class DoobHInitResult:
    """Diagnostics from rejection-sampling X_T ~ p_T^R(x) at the noisy boundary.

    num_passed counts every candidate whose U <= h(T,x) test passed, even if
    it was then discarded because enough samples had already been retained
    (this happens on the final proposal batch, which is over-sized relative
    to what's still needed). num_retained counts only the candidates that
    were actually kept and appear in x_start_samples. These differ exactly
    when a batch produces more passing candidates than are still needed.
    """

    x_start_samples: Tensor  # (num_samples, m, n), samples at t=T
    num_proposed: int
    num_passed: int
    num_retained: int
    mean_h_proposed: float
    mean_h_accepted: float  # mean h over RETAINED candidates only
    mode: str  # "rejection" or "uniform_fallback"

    @property
    def candidate_pass_rate(self) -> float:
        """Fraction of proposed candidates whose U <= h(T,x) test passed."""
        return self.num_passed / self.num_proposed if self.num_proposed > 0 else 0.0

    @property
    def retained_efficiency(self) -> float:
        """Fraction of proposed candidates that were actually retained."""
        return self.num_retained / self.num_proposed if self.num_proposed > 0 else 0.0


def check_h_constant_at_T(
    model: HModel,
    cfg: Config,
    num_probe_samples: int,
    generator: torch.Generator,
) -> Tuple[float, float]:
    """Estimate mean and std of h_theta(T, x) over uniform tables x.

    Used to decide whether the reverse sampler's exact-vs-uniform initial
    distribution should default to rejection sampling (h_theta(T,.) varies
    meaningfully across x) or plain uniform (h_theta(T,.) is already
    approximately constant, so p_T^R(x) \\propto h_theta(T,x) is itself
    approximately uniform and the rejection step would just add variance
    for no benefit).
    """
    device = next(model.parameters()).device
    model.eval()
    candidates = sample_uniform_tables(
        num_probe_samples, cfg.m, cfg.n, cfg.total_count, generator=generator
    )
    with torch.no_grad():
        h_vals = model.forward(
            (candidates.to(device)) / cfg.total_count,
            torch.full((num_probe_samples,), 1.0, dtype=torch.float32, device=device),
        ).cpu()
    return h_vals.mean().item(), h_vals.std().item()


def sample_x_start_reverse(
    model: HModel,
    cfg: Config,
    num_samples: int,
    generator: torch.Generator,
    mode: str = "rejection",
    proposal_batch_size: int = 2048,
) -> DoobHInitResult:
    """Sample the reverse sampler's starting distribution at t=T.

    Under the forward-noising / reverse-guidance orientation, the exact
    guided starting law lives at the NOISY boundary t=T, not at t=0:

        p_T^R(x) = p_T(x) h_theta(T,x) / sum_z p_T(z) h_theta(T,z)

    Since p_T (the forward chain's law at time T) is uniform over E_N, this
    simplifies to p_T^R(x) \\propto h_theta(T,x). Two modes:

    - mode="rejection" (default, correctness-preserving): rejection-sample
      against the uniform proposal, accepting candidate x with probability
      h_theta(T,x) in (0,1] (guaranteed by the sigmoid output, so it always
      upper-bounds the acceptance probability -- no rescaling constant
      needed). This targets p_T^R exactly (up to Monte Carlo error) without
      computing the intractable normalizing sum, and without any
      finite-pool self-normalized importance resampling.
    - mode="uniform_fallback": skip reweighting entirely and draw x ~
      Uniform(E_N) directly. Only an approximation of p_T^R, valid when
      h_theta(T,.) is empirically close to constant (see
      check_h_constant_at_T) -- in that regime p_T^R(x) \\propto h_theta(T,x)
      is itself close to uniform, so the rejection step contributes little
      beyond extra variance and compute.

    There is no rejection sampling here against h_theta(0,x) -- that
    belonged to the old (pre forward-noising) time orientation and has been
    removed.
    """
    device = next(model.parameters()).device
    model.eval()
    T = cfg.terminal_time

    if mode == "uniform_fallback":
        x_start_samples = sample_uniform_tables(
            num_samples, cfg.m, cfg.n, cfg.total_count, generator=generator
        )
        with torch.no_grad():
            h_vals = model.forward(
                (x_start_samples.to(device)) / cfg.total_count,
                torch.full((num_samples,), 1.0, dtype=torch.float32, device=device),
            ).cpu()
        return DoobHInitResult(
            x_start_samples=x_start_samples,
            num_proposed=num_samples,
            num_passed=num_samples,
            num_retained=num_samples,
            mean_h_proposed=h_vals.mean().item(),
            mean_h_accepted=h_vals.mean().item(),
            mode="uniform_fallback",
        )

    if mode != "rejection":
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

        candidates = sample_uniform_tables(
            batch_n, cfg.m, cfg.n, cfg.total_count, generator=generator
        )  # unnormalized, (batch_n, m, n) -- matches training's raw-table convention

        with torch.no_grad():
            h_vals = model.forward(
                (candidates.to(device)) / cfg.total_count,
                torch.full((batch_n,), 1.0, dtype=torch.float32, device=device),
            ).cpu()

        assert (h_vals >= 0.0).all() and (h_vals <= 1.0).all(), (
            "h_theta output must lie in [0,1] (sigmoid guarantees this); "
            f"got min={h_vals.min().item()}, max={h_vals.max().item()}"
        )

        u = torch.rand(batch_n, generator=generator)
        accept_mask = u <= h_vals

        num_proposed += batch_n
        sum_h_proposed += h_vals.sum().item()

        # passed_idx: every candidate whose U <= h(T,x) test passed. Only
        # the first n_needed of these are actually retained -- the rest are
        # discarded (this happens on the final, over-sized batch, where more
        # candidates can pass than are still needed). Diagnostics below are
        # computed from selected_idx (retained only), not passed_idx, so
        # they reflect what was actually kept in x_start_samples.
        passed_idx = torch.nonzero(accept_mask, as_tuple=False).view(-1)
        selected_idx = passed_idx[:n_needed]

        num_passed += len(passed_idx)
        num_retained += len(selected_idx)
        sum_h_accepted += h_vals[selected_idx].sum().item()

        for idx in selected_idx.tolist():
            accepted.append(candidates[idx])

    x_start_samples = torch.stack(accepted[:num_samples], dim=0)
    return DoobHInitResult(
        x_start_samples=x_start_samples,
        num_proposed=num_proposed,
        num_passed=num_passed,
        num_retained=num_retained,
        mean_h_proposed=sum_h_proposed / num_proposed,
        mean_h_accepted=sum_h_accepted / num_retained if num_retained > 0 else float("nan"),
        mode="rejection",
    )


def simulate_guided_batch(
    model: HModel,
    cfg: Config,
    num_samples: int,
    seed: Optional[int] = None,
    init_mode: str = "rejection",
    method: str = "thinning",
    pbar: Optional[tqdm] = None,
    verbose: bool = True,
) -> List[GuidedSampleResult]:
    """Generate ``num_samples`` independent guided trajectories, backward T->0.

    Each trajectory starts from an independent X_T drawn from p_T^R(x)
    \\propto h_theta(T,x) (via sample_x_start_reverse; default mode
    "rejection", the correctness-preserving exact initialization -- pass
    init_mode="uniform_fallback" only when h_theta(T,.) has been verified
    approximately constant, see check_h_constant_at_T).

    All num_samples trajectories are simulated together, batching the model
    forward pass across every sample still active at each step (rather than
    issuing one full sequence of model calls per sample, one sample at a
    time). Two interchangeable methods, selected via ``method``:

    - "thinning" (default): simulate_guided_trajectories_thinning, exact
      batched thinning/rejection that never enumerates the full neighbor
      set (see its docstring for the exactness argument). Much cheaper per
      step when a table has many positive cells, since it evaluates only as
      many candidates as needed to find an acceptance rather than all ~D
      neighbors every refresh.
    - "exhaustive": simulate_guided_trajectories_batched, the original
      reference implementation that enumerates and scores every valid
      neighbor each refresh. Kept for validation/comparison against the
      thinning path (see test_sample_guided.py).
    """
    seed = cfg.seed if seed is None else seed
    generator = torch.Generator()
    generator.manual_seed(seed)

    init_result = sample_x_start_reverse(model, cfg, num_samples, generator, mode=init_mode)
    if verbose:
        msg = (
            f"[sample_x_start_reverse mode={init_result.mode}] "
            f"proposed={init_result.num_proposed} "
            f"passed={init_result.num_passed} "
            f"retained={init_result.num_retained} "
            f"candidate_pass_rate={init_result.candidate_pass_rate:.4f} "
            f"retained_efficiency={init_result.retained_efficiency:.4f} "
            f"mean_h_proposed={init_result.mean_h_proposed:.6f} "
            f"mean_h_accepted={init_result.mean_h_accepted:.6f}"
        )
        tqdm.write(msg) if pbar is not None else print(msg)

    if method == "thinning":
        results, diagnostics = simulate_guided_trajectories_thinning(
            init_result.x_start_samples, model, cfg, generator=generator, pbar=pbar
        )
        if verbose:
            tqdm.write(diagnostics.summary()) if pbar is not None else print(diagnostics.summary())
        return results
    elif method == "exhaustive":
        return simulate_guided_trajectories_batched(
            init_result.x_start_samples, model, cfg, generator=generator
        )
    else:
        raise ValueError(f"Unknown method {method!r}, expected 'thinning' or 'exhaustive'")


def summarize_guided_samples(results: List[GuidedSampleResult], cfg: Config) -> str:
    """Produce a human-readable report on a batch of guided samples.

    Reports row/col sums, S_2 (mean/median/min/max), soft reward,
    exact-margin satisfaction, jump counts, and the number of distinct
    tables among the sampler's generated X_0 outputs (see
    GuidedSampleResult -- the guided sampler runs backward T->0, so these
    are the generated original tables, not terminal states of a forward run).
    """
    target_rows = torch.tensor(cfg.target_rows, dtype=torch.float32)
    target_cols = torch.tensor(cfg.target_cols, dtype=torch.float32)

    generated_tables = torch.stack([r.terminal_table for r in results], dim=0)
    s2 = squared_margin_error(generated_tables, target_rows, target_cols)
    rewards = soft_reward(s2, cfg.reward_gamma)
    exact = exact_margins_satisfied(generated_tables, target_rows, target_cols)
    num_jumps = torch.tensor([r.num_jumps for r in results], dtype=torch.float32)

    distinct = len(
        {tuple(t.reshape(-1).tolist()) for t in generated_tables}
    )

    lines = ["Guided sampling summary (generated X_0)", "=" * 40]
    lines.append(f"num_samples = {len(results)}")
    lines.append(f"distinct generated tables = {distinct}")
    lines.append(
        f"S_2: mean={s2.mean().item():.4f} median={s2.median().item():.4f} "
        f"min={s2.min().item():.4f} max={s2.max().item():.4f}"
    )
    lines.append(
        f"soft reward R(x): mean={rewards.mean().item():.4f} "
        f"min={rewards.min().item():.4f} max={rewards.max().item():.4f}"
    )
    lines.append(f"exact margin satisfaction rate: {exact.float().mean().item():.4f}")
    lines.append(f"num_jumps: mean={num_jumps.mean().item():.2f} min={num_jumps.min().item():.0f} max={num_jumps.max().item():.0f}")

    best_idx = int(torch.argmin(s2).item())
    best_table = generated_tables[best_idx]
    lines.append("\nBest sample (lowest S_2):")
    lines.append(f"  row sums = {row_sums(best_table).tolist()}")
    lines.append(f"  col sums = {col_sums(best_table).tolist()}")
    lines.append(f"  S_2 = {s2[best_idx].item():.4f}")
    lines.append(f"  R(x) = {rewards[best_idx].item():.4f}")
    lines.append(f"  exact margins satisfied = {bool(exact[best_idx].item())}")

    return "\n".join(lines)


def save_guided_samples(results: List[GuidedSampleResult], path: str) -> None:
    """Save every generated X_0 table (and its jump count) to a torch checkpoint.

    Written for downstream/offline computation on the full sample set (not
    for personal inspection) -- see format_best_samples / summarize_guided_samples
    for the human-facing views. Load back with:

        blob = torch.load(path)
        tables = blob["tables"]       # (num_samples, m, n)
        num_jumps = blob["num_jumps"]  # (num_samples,)
    """
    tables = torch.stack([r.terminal_table for r in results], dim=0)
    num_jumps = torch.tensor([r.num_jumps for r in results], dtype=torch.int64)
    torch.save({"tables": tables, "num_jumps": num_jumps}, path)


def format_best_samples(
    results: List[GuidedSampleResult], cfg: Config, top_k: int = 20
) -> str:
    """Format the top_k generated tables with the lowest S_2, for human inspection.

    Complements summarize_guided_samples (aggregate stats) and
    save_guided_samples (the full table set for later computation) -- this
    is the "look at a handful of the best tables" view.
    """
    target_rows = torch.tensor(cfg.target_rows, dtype=torch.float32)
    target_cols = torch.tensor(cfg.target_cols, dtype=torch.float32)

    generated_tables = torch.stack([r.terminal_table for r in results], dim=0)
    s2 = squared_margin_error(generated_tables, target_rows, target_cols)
    rewards = soft_reward(s2, cfg.reward_gamma)
    exact = exact_margins_satisfied(generated_tables, target_rows, target_cols)
    num_jumps = torch.tensor([r.num_jumps for r in results], dtype=torch.int64)

    top_k = min(top_k, len(results))
    best_indices = torch.argsort(s2)[:top_k]

    lines = [f"Best {top_k} generated samples (lowest S_2)", "=" * 40]
    for rank, idx_t in enumerate(best_indices, start=1):
        i = int(idx_t.item())
        table = generated_tables[i]
        lines.append(
            f"\nRank {rank}: sample {i}, S_2={s2[i].item():.4f}, "
            f"R(x)={rewards[i].item():.4f}, jumps={num_jumps[i].item()}, "
            f"exact_margins={bool(exact[i].item())}"
        )
        lines.append(f"  row sums = {row_sums(table).tolist()}")
        lines.append(f"  col sums = {col_sums(table).tolist()}")
        lines.append(f"{table.long().tolist()}")

    return "\n".join(lines)
