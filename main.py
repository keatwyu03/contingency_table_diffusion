from __future__ import annotations
import argparse
import os
import torch
from config import Config
from ctmc import calibrate_mixing, format_calibration_report
from h_dataset import HDataset, generate_h_dataset
from h_model import HModel
from sample_guided import check_h_constant_at_T, simulate_guided_batch, summarize_guided_samples
from table_space import make_start_table_even, make_start_table_first_cell
from train_h import load_h_model, set_seed, train_h_model

def build_config() -> Config:
    cfg = Config()
    print(cfg.summary())
    cfg.ensure_dirs()
    return cfg

def cmd_calibrate(cfg: Config) -> None:
    set_seed(cfg.seed)
    start_tables = {'first_cell': make_start_table_first_cell(cfg.m, cfg.n, cfg.total_count), 'even': make_start_table_even(cfg.m, cfg.n, cfg.total_count)}
    report = calibrate_mixing(m=cfg.m, n=cfg.n, total_count=cfg.total_count, terminal_time=cfg.terminal_time, ctmc_rate=cfg.ctmc_rate, start_tables=start_tables, num_samples_per_start=100, num_uniform_reference=200, seed=cfg.seed)
    text = format_calibration_report(report)
    print(text)
    out_path = os.path.join(cfg.results_dir, 'calibration_report.txt')
    with open(out_path, 'w') as f:
        f.write(text)
    print(f'\nSaved calibration report to {out_path}')

def cmd_make_dataset(cfg: Config) -> None:
    set_seed(cfg.seed)
    dataset = generate_h_dataset(cfg)
    dataset.save(cfg.dataset_path)
    print(f'Generated {len(dataset)} samples from {cfg.num_original_samples} original tables.')
    print(f'Saved dataset to {cfg.dataset_path}')

def cmd_train_h(cfg: Config) -> None:
    set_seed(cfg.seed)
    if not os.path.exists(cfg.dataset_path):
        print(f'Dataset not found at {cfg.dataset_path}, generating it now.')
        cmd_make_dataset(cfg)
    dataset = HDataset.load(cfg.dataset_path)
    model, history = train_h_model(cfg, dataset)
    print(f'Training complete. best_val_loss={history.best_val_loss:.6f} at epoch {history.best_epoch + 1}')

def cmd_sample_guided(cfg: Config) -> None:
    set_seed(cfg.seed)
    model = load_h_model(cfg)
    model.eval()
    generator = torch.Generator().manual_seed(cfg.seed)
    mean_h_T, std_h_T = check_h_constant_at_T(model, cfg, num_probe_samples=200, generator=generator)
    print(f'[diagnostic] h_theta(T, x) over 200 uniform tables: mean={mean_h_T:.6f} std={std_h_T:.6f}')
    results = simulate_guided_batch(model, cfg, num_samples=cfg.num_guided_samples)
    text = summarize_guided_samples(results, cfg)
    print(text)
    out_path = os.path.join(cfg.results_dir, 'guided_sampling_report.txt')
    with open(out_path, 'w') as f:
        f.write(text)
    print(f'\nSaved guided sampling report to {out_path}')

def cmd_pipeline(cfg: Config) -> None:
    print('\n=== Step 1/3: make-dataset ===')
    cmd_make_dataset(cfg)
    print('\n=== Step 2/3: train-h ===')
    cmd_train_h(cfg)
    print('\n=== Step 3/3: sample-guided ===')
    cmd_sample_guided(cfg)

def main() -> None:
    parser = argparse.ArgumentParser(description='Guided CTMC sampler for conditional contingency tables.')
    parser.add_argument('command', nargs='?', default='pipeline', choices=['calibrate', 'make-dataset', 'train-h', 'sample-guided', 'pipeline'])
    args = parser.parse_args()
    cfg = build_config()
    dispatch = {'calibrate': cmd_calibrate, 'make-dataset': cmd_make_dataset, 'train-h': cmd_train_h, 'sample-guided': cmd_sample_guided, 'pipeline': cmd_pipeline}
    dispatch[args.command](cfg)
if __name__ == '__main__':
    main()
