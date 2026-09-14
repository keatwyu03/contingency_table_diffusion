"""CLI entry point for the guided-CTMC contingency table sampler.

Commands:
    python main.py calibrate
    python main.py make-dataset
    python main.py train-h
    python main.py sample-guided
    python main.py pipeline
"""

from __future__ import annotations

import argparse
import os

import torch

from config import Config
from ctmc import calibrate_mixing, format_calibration_report
from h_dataset import HDataset, generate_h_dataset
from h_model import HModel
from sample_guided import simulate_guided_batch, summarize_guided_samples
from table_space import make_start_table_even, make_start_table_first_cell
from train_h import load_h_model, set_seed, train_h_model


def build_config() -> Config:
    cfg = Config()
    print(cfg.summary())
    cfg.ensure_dirs()
    return cfg


def cmd_calibrate(cfg: Config) -> None:
    """Run mixing calibration comparing CTMC terminal samples to uniform E_N samples."""
    set_seed(cfg.seed)
    start_tables = {
        "first_cell": make_start_table_first_cell(cfg.m, cfg.n, cfg.total_count),
        "even": make_start_table_even(cfg.m, cfg.n, cfg.total_count),
    }
    report = calibrate_mixing(
        m=cfg.m,
        n=cfg.n,
        total_count=cfg.total_count,
        terminal_time=cfg.terminal_time,
        ctmc_rate=cfg.ctmc_rate,
        start_tables=start_tables,
        num_samples_per_start=100,
        num_uniform_reference=200,
        seed=cfg.seed,
    )
    text = format_calibration_report(report)
    print(text)
    out_path = os.path.join(cfg.output_dir, "calibration_report.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\nSaved calibration report to {out_path}")


def cmd_make_dataset(cfg: Config) -> None:
    """Generate and save the offline h-training dataset.

    Each trajectory starts from an independent X_0 ~ Uniform(E_N) (see
    generate_h_dataset), not from a single fixed deterministic table.
    """
    set_seed(cfg.seed)
    dataset = generate_h_dataset(cfg)
    dataset.save(cfg.dataset_path)
    print(f"Generated {len(dataset)} samples from {cfg.num_trajectories} trajectories.")
    print(f"Saved dataset to {cfg.dataset_path}")


def cmd_train_h(cfg: Config) -> None:
    """Train h_theta on the saved dataset (generating it first if missing)."""
    set_seed(cfg.seed)
    if not os.path.exists(cfg.dataset_path):
        print(f"Dataset not found at {cfg.dataset_path}, generating it now.")
        cmd_make_dataset(cfg)
    dataset = HDataset.load(cfg.dataset_path)
    model, history = train_h_model(cfg, dataset)
    print(
        f"Training complete. best_val_loss={history.best_val_loss:.6f} "
        f"at epoch {history.best_epoch + 1}"
    )


def cmd_sample_guided(cfg: Config) -> None:
    """Run the guided CTMC sampler using the trained h checkpoint.

    Each guided sample starts from an independent X_0 ~ Uniform(E_N) (see
    simulate_guided_batch), not from a single fixed deterministic table.
    """
    set_seed(cfg.seed)
    model = load_h_model(cfg)
    model.eval()
    results = simulate_guided_batch(
        model, cfg, num_samples=cfg.num_guided_samples
    )
    text = summarize_guided_samples(results, cfg)
    print(text)
    out_path = os.path.join(cfg.output_dir, "guided_sampling_report.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\nSaved guided sampling report to {out_path}")


def cmd_pipeline(cfg: Config) -> None:
    """Run calibration, dataset generation, training, and guided sampling in order."""
    print("\n=== Step 1/4: calibrate ===")
    cmd_calibrate(cfg)
    print("\n=== Step 2/4: make-dataset ===")
    cmd_make_dataset(cfg)
    print("\n=== Step 3/4: train-h ===")
    cmd_train_h(cfg)
    print("\n=== Step 4/4: sample-guided ===")
    cmd_sample_guided(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Guided CTMC sampler for conditional contingency tables."
    )
    parser.add_argument(
        "command",
        choices=["calibrate", "make-dataset", "train-h", "sample-guided", "pipeline"],
    )
    args = parser.parse_args()

    cfg = build_config()

    dispatch = {
        "calibrate": cmd_calibrate,
        "make-dataset": cmd_make_dataset,
        "train-h": cmd_train_h,
        "sample-guided": cmd_sample_guided,
        "pipeline": cmd_pipeline,
    }
    dispatch[args.command](cfg)


if __name__ == "__main__":
    main()
