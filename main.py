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
from tqdm import tqdm

from config import Config
from ctmc import calibrate_mixing, format_calibration_report
from h_dataset import HDataset, generate_h_dataset
from h_model import HModel
from pretrain_score import pretrain_score
from sample_guided import (
    check_h_constant_at_T,
    format_best_samples,
    save_guided_samples,
    simulate_guided_batch,
    summarize_guided_samples,
)
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
    out_path = os.path.join(cfg.results_dir, "calibration_report.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\nSaved calibration report to {out_path}")


def cmd_make_dataset(cfg: Config) -> None:
    """Generate and save the offline h-training dataset.

    Each independent original sample draws X_0 ~ Uniform(E_N), computes
    R(X_0) once, then forward-noises to exactly one tau ~ Uniform(0, T) to
    obtain a single (X_tau, tau) pair with that R(X_0) target -- see
    generate_h_dataset for the full forward-noising / reverse-guidance
    convention.
    """
    set_seed(cfg.seed)
    dataset = generate_h_dataset(cfg)
    dataset.save(cfg.dataset_path)
    print(f"Generated {len(dataset)} samples from {cfg.num_original_samples} original tables.")
    print(f"Saved dataset to {cfg.dataset_path}")


def cmd_train_h(cfg: Config) -> None:
    """Train h_theta on the saved dataset (generating it first if missing).

    If cfg.use_pretraining is True, E_phi is first pretrained on CTMC
    trajectory scoring (see pretrain_score.py) using the SAME trajectories
    already in ``dataset`` (no separate trajectories are simulated for
    pretraining), transferred into a fresh HModel, and then E_phi + H_omega
    are jointly fine-tuned on the soft terminal reward regression objective
    (E_phi at a reduced learning rate via cfg.pretrain_encoder_lr_scale). If
    False, behavior is unchanged from before pretraining was added: a
    freshly-initialized HModel is trained directly on the reward regression
    objective.
    """
    set_seed(cfg.seed)
    if not os.path.exists(cfg.dataset_path):
        print(f"Dataset not found at {cfg.dataset_path}, generating it now.")
        cmd_make_dataset(cfg)
    dataset = HDataset.load(cfg.dataset_path)

    if cfg.use_pretraining:
        print("[train-h] use_pretraining=True: running CTMC pretraining for E_phi.")
        pretrained_encoder = pretrain_score(cfg, dataset)
        h_model = HModel(cfg.m, cfg.n, cfg.total_count)
        h_model.load_encoder_state_dict(pretrained_encoder.encoder_state_dict())
        model, history = train_h_model(
            cfg, dataset, model=h_model, encoder_lr_scale=cfg.pretrain_encoder_lr_scale
        )
    else:
        model, history = train_h_model(cfg, dataset)

    print(
        f"Training complete. best_val_loss={history.best_val_loss:.6f} "
        f"at epoch {history.best_epoch + 1}"
    )


def cmd_sample_guided(cfg: Config) -> None:
    """Run the guided CTMC sampler backward (t=T -> t=0) using the trained checkpoint.

    Each guided sample starts from an independent X_T drawn from
    p_T^R(x) \\propto h_theta(T,x) (via rejection sampling by default; see
    sample_guided.sample_x_start_reverse) and is simulated backward to
    produce a generated (approximate) X_0.

    cfg.num_guided_samples samples are generated in sequential sub-batches
    of at most cfg.guided_batch_size each (rather than one single batch of
    every sample at once), since simulate_guided_batch builds tensors sized
    by the whole batch (all live tables and, for the exhaustive method,
    every neighbor of every live table) -- for large num_guided_samples
    (e.g. several thousand) that would risk a memory spike. Each sub-batch
    uses a distinct seed derived from cfg.seed so sub-batches are
    independent draws, not repeats.
    """
    set_seed(cfg.seed)
    model = load_h_model(cfg)
    model.eval()

    generator = torch.Generator().manual_seed(cfg.seed)
    mean_h_T, std_h_T = check_h_constant_at_T(
        model, cfg, num_probe_samples=cfg.init_check_num_probe_samples, generator=generator
    )
    coefficient_of_variation = std_h_T / mean_h_T if mean_h_T > 0 else float("inf")
    print(
        f"[diagnostic] h_theta(T, x) over {cfg.init_check_num_probe_samples} uniform tables: "
        f"mean={mean_h_T:.6f} std={std_h_T:.6f} cv={coefficient_of_variation:.6f}"
    )
    if coefficient_of_variation <= cfg.init_uniform_fallback_cv_threshold:
        init_mode = "uniform_fallback"
        print(
            f"[sample-guided] h_theta(T,.) is approximately constant "
            f"(cv={coefficient_of_variation:.4f} <= {cfg.init_uniform_fallback_cv_threshold}): "
            f"using init_mode='uniform_fallback'."
        )
    else:
        init_mode = "rejection"
        print(
            f"[sample-guided] h_theta(T,.) varies meaningfully "
            f"(cv={coefficient_of_variation:.4f} > {cfg.init_uniform_fallback_cv_threshold}): "
            f"using the exact init_mode='rejection' (p_T^R(x) \\propto h_theta(T,x))."
        )

    results = []
    remaining = cfg.num_guided_samples
    batch_idx = 0
    pbar = tqdm(total=cfg.num_guided_samples, desc="sample-guided")
    while remaining > 0:
        batch_n = min(cfg.guided_batch_size, remaining)
        batch_results = simulate_guided_batch(
            model, cfg, num_samples=batch_n, seed=cfg.seed + batch_idx,
            pbar=pbar, verbose=False, init_mode=init_mode,
        )
        results.extend(batch_results)
        remaining -= batch_n
        batch_idx += 1
    pbar.close()

    samples_path = os.path.join(cfg.results_dir, "guided_samples.pt")
    save_guided_samples(results, samples_path)
    print(f"\nSaved {len(results)} generated tables to {samples_path}")

    text = summarize_guided_samples(results, cfg)
    print(text)
    report_path = os.path.join(cfg.results_dir, "guided_sampling_report.txt")
    with open(report_path, "w") as f:
        f.write(text)
    print(f"\nSaved guided sampling report to {report_path}")

    best_text = format_best_samples(results, cfg, top_k=20)
    best_path = os.path.join(cfg.results_dir, "best_samples.txt")
    with open(best_path, "w") as f:
        f.write(best_text)
    print(f"Saved best-20 samples to {best_path}")


def cmd_pipeline(cfg: Config) -> None:
    """Run dataset generation, training, and guided sampling in order.

    Calibration is skipped here (run `python main.py calibrate` directly if
    needed) -- it's a standalone diagnostic on the base CTMC's mixing, not a
    dependency of dataset generation, training, or guided sampling.
    """
    print("\n=== Step 1/3: make-dataset ===")
    cmd_make_dataset(cfg)
    print("\n=== Step 2/3: train-h ===")
    cmd_train_h(cfg)
    print("\n=== Step 3/3: sample-guided ===")
    cmd_sample_guided(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Guided CTMC sampler for conditional contingency tables."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="pipeline",
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
