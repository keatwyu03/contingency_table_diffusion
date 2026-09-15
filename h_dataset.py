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
    current_table: Tensor
    time: float
    original_reward: float
    original_table: Tensor
    original_sample_id: int

class HDataset(Dataset):

    def __init__(self, samples: List[HDatasetSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (s.current_table, torch.tensor(s.time, dtype=torch.float32), torch.tensor(s.original_reward, dtype=torch.float32))

    def save(self, path: str) -> None:
        tables = torch.stack([s.current_table for s in self.samples], dim=0)
        times = torch.tensor([s.time for s in self.samples], dtype=torch.float32)
        rewards = torch.tensor([s.original_reward for s in self.samples], dtype=torch.float32)
        original_tables = torch.stack([s.original_table for s in self.samples], dim=0)
        sample_ids = torch.tensor([s.original_sample_id for s in self.samples], dtype=torch.int64)
        torch.save({'tables': tables, 'times': times, 'rewards': rewards, 'original_tables': original_tables, 'sample_ids': sample_ids}, path)

    @staticmethod
    def load(path: str) -> 'HDataset':
        blob = torch.load(path)
        tables, times, rewards = (blob['tables'], blob['times'], blob['rewards'])
        original_tables = blob['original_tables']
        sample_ids = blob['sample_ids']
        samples = [HDatasetSample(current_table=tables[i], time=float(times[i]), original_reward=float(rewards[i]), original_table=original_tables[i], original_sample_id=int(sample_ids[i])) for i in range(tables.shape[0])]
        return HDataset(samples)

def generate_h_dataset(cfg: Config, num_original_samples: Optional[int]=None, seed: Optional[int]=None) -> HDataset:
    num_original_samples = num_original_samples or cfg.num_original_samples
    seed = cfg.seed if seed is None else seed
    generator = torch.Generator()
    generator.manual_seed(seed)
    target_rows = torch.tensor(cfg.target_rows, dtype=torch.float32)
    target_cols = torch.tensor(cfg.target_cols, dtype=torch.float32)
    samples: List[HDatasetSample] = []
    for original_sample_id in tqdm(range(num_original_samples), desc='generate_h_dataset[original samples]'):
        x0 = sample_uniform_tables(1, cfg.m, cfg.n, cfg.total_count, generator=generator).squeeze(0)
        s2_x0 = squared_margin_error(x0.unsqueeze(0), target_rows, target_cols).squeeze(0)
        r_x0 = float(soft_reward(s2_x0, cfg.reward_gamma).item())
        x0_normalized = x0 / cfg.total_count
        tau = float(torch.rand((), generator=generator).item()) * cfg.terminal_time
        if tau <= 0.0:
            x_tau = x0
        else:
            result = simulate_trajectory(x0, tau, cfg.ctmc_rate, generator=generator)
            x_tau = result.terminal_table
        samples.append(HDatasetSample(current_table=x_tau / cfg.total_count, time=tau / cfg.terminal_time, original_reward=r_x0, original_table=x0_normalized, original_sample_id=original_sample_id))
    return HDataset(samples)
