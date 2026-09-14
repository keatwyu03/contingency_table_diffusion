"""Dataset generation for training h_theta(t, X_t).

For each unconditional CTMC trajectory started from a fixed starting table,
we simulate from 0 to T, record several intermediate (t, X_t) snapshots, and
compute a single terminal soft reward R(X_T) = exp(-gamma * S_2(X_T)). Every
snapshot from that trajectory is paired with the *same* terminal target,
since R(X_T) is a Monte-Carlo target for E[R(X_T) | X_t] and each trajectory
contributes one unbiased sample of that conditional expectation at every t
along its own path.

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

from config import Config
from ctmc import simulate_trajectory, table_at_time
from table_space import sample_uniform_tables, squared_margin_error, soft_reward


@dataclass
class HDatasetSample:
    """A single training example for h_theta."""

    current_table: Tensor  # (m, n), normalized by N
    time: float  # normalized by T, in [0, 1]
    terminal_reward: float  # R(X_T) in (0, 1]


class HDataset(Dataset):
    """In-memory dataset of (normalized table, normalized time, terminal reward)."""

    def __init__(self, samples: List[HDatasetSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return s.current_table, torch.tensor(s.time, dtype=torch.float32), torch.tensor(
            s.terminal_reward, dtype=torch.float32
        )

    def save(self, path: str) -> None:
        """Save the dataset to disk as a torch checkpoint."""
        tables = torch.stack([s.current_table for s in self.samples], dim=0)
        times = torch.tensor([s.time for s in self.samples], dtype=torch.float32)
        rewards = torch.tensor(
            [s.terminal_reward for s in self.samples], dtype=torch.float32
        )
        torch.save({"tables": tables, "times": times, "rewards": rewards}, path)

    @staticmethod
    def load(path: str) -> "HDataset":
        """Load a dataset previously saved with .save()."""
        blob = torch.load(path)
        tables, times, rewards = blob["tables"], blob["times"], blob["rewards"]
        samples = [
            HDatasetSample(
                current_table=tables[i], time=float(times[i]), terminal_reward=float(rewards[i])
            )
            for i in range(tables.shape[0])
        ]
        return HDataset(samples)


def generate_h_dataset(
    cfg: Config,
    num_trajectories: Optional[int] = None,
    num_time_samples_per_trajectory: Optional[int] = None,
    seed: Optional[int] = None,
) -> HDataset:
    """Generate an offline dataset of (t, X_t, R(X_T)) triples.

    Each trajectory starts from an independent X_0 ~ Uniform(E_N), drawn via
    exact stars-and-bars sampling (table_space.sample_uniform_tables) -- NOT
    from a single fixed deterministic table. This matches the agreed
    procedure X_0 ~ Uniform(E_N) so h_theta is trained on trajectories that
    start spread across the whole table space rather than all funneling
    through one corner of it.

    Args:
        cfg: resolved Config.
        num_trajectories: overrides cfg.num_trajectories if given.
        num_time_samples_per_trajectory: overrides cfg if given.
        seed: overrides cfg.seed if given.

    Returns:
        An HDataset with N = num_trajectories * num_time_samples_per_trajectory
        samples (each trajectory contributes exactly that many snapshots, all
        sharing the same terminal reward target).
    """
    num_trajectories = num_trajectories or cfg.num_trajectories
    num_time_samples = (
        num_time_samples_per_trajectory or cfg.num_time_samples_per_trajectory
    )
    seed = cfg.seed if seed is None else seed

    generator = torch.Generator()
    generator.manual_seed(seed)

    target_rows = torch.tensor(cfg.target_rows, dtype=torch.float32)
    target_cols = torch.tensor(cfg.target_cols, dtype=torch.float32)

    samples: List[HDatasetSample] = []

    for _ in range(num_trajectories):
        start_table = sample_uniform_tables(
            1, cfg.m, cfg.n, cfg.total_count, generator=generator
        ).squeeze(0)

        # Simulate once, storing the entire jump path (see SimulationResult
        # in ctmc.py). Intermediate observation times are then drawn and
        # looked up from that single stored path -- no re-simulation.
        result = simulate_trajectory(
            start_table,
            cfg.terminal_time,
            cfg.ctmc_rate,
            generator=generator,
        )

        terminal_table = result.terminal_table
        s2_terminal = squared_margin_error(
            terminal_table.unsqueeze(0), target_rows, target_cols
        ).squeeze(0)
        r_terminal = float(soft_reward(s2_terminal, cfg.reward_gamma).item())

        # Sample intermediate times uniformly in (0, T), then sort; always
        # include T itself so the snapshot at t=T is available for the
        # optional boundary loss.
        interior_times = torch.rand(
            max(num_time_samples - 1, 0), generator=generator
        ) * cfg.terminal_time
        times = torch.cat(
            [interior_times, torch.tensor([cfg.terminal_time])]
        )
        times, _ = torch.sort(times)

        for t in times.tolist():
            table_t = table_at_time(result.path, t, cfg.terminal_time)
            normalized_table = table_t / cfg.total_count
            normalized_time = t / cfg.terminal_time
            samples.append(
                HDatasetSample(
                    current_table=normalized_table,
                    time=normalized_time,
                    terminal_reward=r_terminal,
                )
            )

    return HDataset(samples)
