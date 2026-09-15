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
    """Compute one guided-CTMC step from state x at forward-time label t.

    The forward CTMC is symmetric and Uniform(E_N) is its stationary
    distribution, so the reverse-time proposal rate equals the forward rate:
    q_t^reverse(x,y) = q_t^F(y,x) = q_t^F(x,y) = ctmc_rate / K. The guided
    rate is therefore q_t^guided(x,y) = (ctmc_rate/K) * h_theta(t,y)/h_theta(t,x)
    for every valid neighbor y, exactly as in the forward-time case -- only
    the direction time moves in the caller (simulate_guided_trajectory)
    differs.

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

    # Neural network forward pass runs on the model's device (possibly cuda),
    # but everything else in this function (the table state, RNG generator,
    # and multinomial draw) operates on CPU -- so results are moved back to
    # CPU immediately after the forward pass to keep devices consistent for
    # sample_exponential/multinomial, which require generator and tensor
    # devices to match.
    log_h_x = _log_h_batch(
        model, x.unsqueeze(0), t, cfg.total_count, cfg.terminal_time
    )[0].cpu()
    log_h_neighbors = _log_h_batch(
        model, neighbors, t, cfg.total_count, cfg.terminal_time
    ).cpu()

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


@dataclass
class DoobHInitResult:
    """Diagnostics from rejection-sampling X_T ~ p_T^R(x) at the noisy boundary."""

    x_start_samples: Tensor  # (num_samples, m, n), samples at t=T
    num_proposed: int
    num_accepted: int
    mean_h_proposed: float
    mean_h_accepted: float
    mode: str  # "rejection" or "uniform_fallback"

    @property
    def acceptance_rate(self) -> float:
        return self.num_accepted / self.num_proposed if self.num_proposed > 0 else 0.0


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
    proposal_batch_size: int = 256,
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
            num_accepted=num_samples,
            mean_h_proposed=h_vals.mean().item(),
            mean_h_accepted=h_vals.mean().item(),
            mode="uniform_fallback",
        )

    if mode != "rejection":
        raise ValueError(f"Unknown mode {mode!r}, expected 'rejection' or 'uniform_fallback'")

    accepted: List[Tensor] = []
    num_proposed = 0
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
        sum_h_accepted += h_vals[accept_mask].sum().item()

        for idx in torch.nonzero(accept_mask, as_tuple=False).view(-1).tolist():
            if len(accepted) >= num_samples:
                break
            accepted.append(candidates[idx])

    x_start_samples = torch.stack(accepted[:num_samples], dim=0)
    return DoobHInitResult(
        x_start_samples=x_start_samples,
        num_proposed=num_proposed,
        num_accepted=num_samples,
        mean_h_proposed=sum_h_proposed / num_proposed,
        mean_h_accepted=sum_h_accepted / num_samples if num_samples > 0 else float("nan"),
        mode="rejection",
    )


def simulate_guided_batch(
    model: HModel,
    cfg: Config,
    num_samples: int,
    seed: Optional[int] = None,
    init_mode: str = "rejection",
) -> List[GuidedSampleResult]:
    """Generate ``num_samples`` independent guided trajectories, backward T->0.

    Each trajectory starts from an independent X_T drawn from p_T^R(x)
    \\propto h_theta(T,x) (via sample_x_start_reverse; default mode
    "rejection", the correctness-preserving exact initialization -- pass
    init_mode="uniform_fallback" only when h_theta(T,.) has been verified
    approximately constant, see check_h_constant_at_T).
    """
    seed = cfg.seed if seed is None else seed
    generator = torch.Generator()
    generator.manual_seed(seed)

    init_result = sample_x_start_reverse(model, cfg, num_samples, generator, mode=init_mode)
    print(
        f"[sample_x_start_reverse mode={init_result.mode}] "
        f"proposed={init_result.num_proposed} "
        f"accepted={init_result.num_accepted} "
        f"acceptance_rate={init_result.acceptance_rate:.4f} "
        f"mean_h_proposed={init_result.mean_h_proposed:.6f} "
        f"mean_h_accepted={init_result.mean_h_accepted:.6f}"
    )

    results = []
    for i in tqdm(range(num_samples), desc="simulate_guided_batch"):
        x_start = init_result.x_start_samples[i]
        results.append(
            simulate_guided_trajectory(x_start, model, cfg, generator=generator)
        )
    return results


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
