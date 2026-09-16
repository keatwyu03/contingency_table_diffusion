"""Dataset generation for training h_theta(t, X_t) under the forward-noising
/ reverse-guidance orientation.

Convention: X_0 is the original (clean) table, X_t is its forward-noised
version at time t (via the existing unconditional CTMC), and X_T is the
fully noised terminal table. Training runs forward in time (0 -> T);
guided sampling (see sample_guided.py) runs backward (T -> 0), recovering
an approximate X_0.

The network target is

    h_theta(t, X_t) ~= E[R(X_0) | X_t]

where R(X_0) = exp(-gamma * S_2(X_0)) is the soft reward of the ORIGINAL
table the forward noising started from -- NOT the reward of X_t itself, and
NOT the reward of some future continuation of the chain past t. This is a
different target than the old "terminal reward of a trajectory run forward
from a fixed start" scheme: here every training pair (t, X_t) is generated
by first drawing X_0 ~ Uniform(E_N), computing R(X_0) once, then running the
forward CTMC from X_0 for a SHORT time tau (not out to T), and pairing
X_tau with the already-known R(X_0).

Normalization convention (see also h_model.py): tables are normalized by
dividing by N = total_count, and time is normalized by dividing by T =
terminal_time. This normalization is performed HERE, in the dataset, and
NOT inside the model -- h_model.py expects already-normalized inputs. This
is the single place normalization happens; do not re-normalize in the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from config import Config
from ctmc import simulate_trajectory
from table_space import sample_uniform_tables, squared_margin_error, soft_reward


@dataclass
class HDatasetSample:
    """A single training example for h_theta.

    original_table and original_sample_id are diagnostics/bookkeeping only
    (used to verify the target and to split train/val by original sample so
    that no X_0's observations leak across the split) -- they are not model
    inputs. The model sees only (current_table, time).

    later_table/later_time hold a second snapshot (X_s, s), s > tau, from
    the SAME trajectory simulation used for (current_table, time) -- these
    exist purely so pretrain_score.py can build (X_tau, tau, delta=s-tau)
    -> X_s pretraining pairs without re-simulating the CTMC; they are not
    used by h-training (train_h.py) at all.
    """

    current_table: Tensor  # X_tau, (m, n), normalized by N
    time: float  # tau, normalized by T, in [0, 1]
    original_reward: float  # R(X_0) in (0, 1] -- the training target
    original_table: Tensor  # X_0, (m, n), normalized by N -- diagnostic only
    original_sample_id: int  # groups all (tau, X_tau) drawn from the same X_0
    later_table: Tensor  # X_s, (m, n), normalized by N -- pretraining target only
    later_time: float  # s, normalized by T, in [0, 1], s > tau -- pretraining only


class HDataset(Dataset):
    """Dataset of (X_tau, tau, R(X_0)) triples, grouped by original_sample_id."""

    def __init__(self, samples: List[HDatasetSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            s.current_table,
            torch.tensor(s.time, dtype=torch.float32),
            torch.tensor(s.original_reward, dtype=torch.float32),
            s.original_table,
        )

    def save(self, path: str) -> None:
        """Save the dataset to disk as a torch checkpoint."""
        tables = torch.stack([s.current_table for s in self.samples], dim=0)
        times = torch.tensor([s.time for s in self.samples], dtype=torch.float32)
        rewards = torch.tensor(
            [s.original_reward for s in self.samples], dtype=torch.float32
        )
        original_tables = torch.stack(
            [s.original_table for s in self.samples], dim=0
        )
        sample_ids = torch.tensor(
            [s.original_sample_id for s in self.samples], dtype=torch.int64
        )
        later_tables = torch.stack([s.later_table for s in self.samples], dim=0)
        later_times = torch.tensor(
            [s.later_time for s in self.samples], dtype=torch.float32
        )
        torch.save(
            {
                "tables": tables,
                "times": times,
                "rewards": rewards,
                "original_tables": original_tables,
                "sample_ids": sample_ids,
                "later_tables": later_tables,
                "later_times": later_times,
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "HDataset":
        """Load a dataset previously saved with .save()."""
        blob = torch.load(path)
        tables, times, rewards = blob["tables"], blob["times"], blob["rewards"]
        original_tables = blob["original_tables"]
        sample_ids = blob["sample_ids"]
        later_tables = blob["later_tables"]
        later_times = blob["later_times"]
        samples = [
            HDatasetSample(
                current_table=tables[i],
                time=float(times[i]),
                original_reward=float(rewards[i]),
                original_table=original_tables[i],
                original_sample_id=int(sample_ids[i]),
                later_table=later_tables[i],
                later_time=float(later_times[i]),
            )
            for i in range(tables.shape[0])
        ]
        return HDataset(samples)


def generate_h_dataset(
    cfg: Config,
    num_original_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> HDataset:
    """Generate an offline dataset of (tau, X_tau, R(X_0), s, X_s) rows.

    For each independent original sample:
      1. Draw X_0 ~ Uniform(E_N) via exact stars-and-bars.
      2. Compute S_2(X_0) and R(X_0) = exp(-gamma * S_2(X_0)) immediately --
         this is the training target for the (tau, X_tau) pair drawn below,
         and does NOT depend on any future continuation of the chain.
      3. Draw tau ~ Uniform(0, T) and a second, later time
         s ~ Uniform(tau, T), then run the unconditional forward CTMC ONCE
         from X_0 out to time s (not two separate simulations), recording
         snapshots at both tau and s to obtain X_tau and X_s from that
         single run. Store (X_tau, tau, R(X_0)) as before, plus (X_s, s) as
         extra fields used only by CTMC pretraining (see pretrain_score.py)
         to build (X_tau, tau, delta=s-tau) -> X_s pairs -- h-training
         (train_h.py) does not read later_table/later_time at all.

    We deliberately do NOT simulate from X_tau onward to T to derive an
    h-training label -- that was the previous ("terminal reward of a
    forward continuation") scheme and is removed here. The X_s snapshot
    added here is solely a pretraining target, not an h-training target.

    Exactly ONE (tau, s) pair is drawn per X_0 (not several per X_0
    including forced tau=0 and tau=T boundary values): forcing boundary
    observations into every group of K samples per X_0 would make a fixed
    fraction (e.g. 2/K) of the dataset sit exactly at the two boundaries
    instead of genuinely tau ~ Uniform(0, T), overweighting the boundaries
    relative to the interior. If boundary supervision is wanted, it should
    be added as separate, explicitly controlled-weight observations, not by
    forcing two of every four (tau, X_tau) pairs from each X_0 onto the
    boundary.

    Args:
        cfg: resolved Config.
        num_original_samples: overrides cfg.num_original_samples if given.
        seed: overrides cfg.seed if given.

    Returns:
        An HDataset with num_original_samples rows (one (tau, s) pair per
        X_0), each tagged with a distinct original_sample_id.
    """
    num_original_samples = num_original_samples or cfg.num_original_samples
    seed = cfg.seed if seed is None else seed

    generator = torch.Generator()
    generator.manual_seed(seed)

    target_rows = torch.tensor(cfg.target_rows, dtype=torch.float32)
    target_cols = torch.tensor(cfg.target_cols, dtype=torch.float32)

    samples: List[HDatasetSample] = []

    for original_sample_id in tqdm(
        range(num_original_samples), desc="generate_h_dataset[original samples]"
    ):
        x0 = sample_uniform_tables(
            1, cfg.m, cfg.n, cfg.total_count, generator=generator
        ).squeeze(0)

        s2_x0 = squared_margin_error(x0.unsqueeze(0), target_rows, target_cols).squeeze(0)
        r_x0 = float(soft_reward(s2_x0, cfg.reward_gamma).item())
        x0_normalized = x0 / cfg.total_count

        tau = float(torch.rand((), generator=generator).item()) * cfg.terminal_time
        # s ~ Uniform(tau, T); guard the (measure-zero) tau == T case.
        if tau >= cfg.terminal_time:
            s_time = cfg.terminal_time
        else:
            s_time = tau + float(torch.rand((), generator=generator).item()) * (
                cfg.terminal_time - tau
            )

        if s_time <= 0.0:
            x_tau = x0
            x_s = x0
        else:
            result = simulate_trajectory(
                x0, s_time, cfg.ctmc_rate,
                snapshot_times=[tau, s_time], generator=generator,
            )
            x_tau = result.snapshots[0].table
            x_s = result.snapshots[1].table

        samples.append(
            HDatasetSample(
                current_table=x_tau / cfg.total_count,
                time=tau / cfg.terminal_time,
                original_reward=r_x0,
                original_table=x0_normalized,
                original_sample_id=original_sample_id,
                later_table=x_s / cfg.total_count,
                later_time=s_time / cfg.terminal_time,
            )
        )

    return HDataset(samples)
