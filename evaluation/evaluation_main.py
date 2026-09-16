"""Evaluation entry point: loads guided samples produced by `python main.py
sample-guided` and reports diagnostics comparing them against the oracle and
unguided reference distributions (see diagnostics_eval.py). Does not
generate samples itself -- main.py owns running the baseline pipeline and
writing results/; this script only reads results/guided_samples.pt.

Run from the project directory:
    cd /sailhome/kadenwu/contingency_table_diffusion
    python evaluation/evaluation_main.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import Config
from evaluation.diagnostics_eval import (
    diversity,
    mean_reward_unguided_vs_guided,
    oracle_vs_guided_mean_s2,
    per_margin_deviation_breakdown,
    standardized_margin_deviation,
)

SAMPLES_PATH = os.path.join("results", "guided_samples.pt")
REPORT_PATH = os.path.join("results", "diagnostics_eval_report.txt")


def main() -> None:
    cfg = Config()

    if not os.path.exists(SAMPLES_PATH):
        raise FileNotFoundError(
            f"{SAMPLES_PATH} not found. Run `python main.py sample-guided` first."
        )
    blob = torch.load(SAMPLES_PATH)
    guided_tables = blob["tables"]
    n_samples = guided_tables.shape[0]

    generator = torch.Generator().manual_seed(cfg.seed)

    d_avg, d_max = standardized_margin_deviation(guided_tables, cfg, generator)
    breakdown = per_margin_deviation_breakdown(guided_tables, cfg, generator)
    oracle_s2, guided_s2 = oracle_vs_guided_mean_s2(guided_tables, cfg, generator)
    unguided_reward, guided_reward = mean_reward_unguided_vs_guided(guided_tables, cfg, generator)
    distinct, total = diversity(guided_tables)

    lines = ["Diagnostics evaluation report", "=" * 40]
    lines.append(f"num_samples = {n_samples}")

    lines.append("\nAverage standardized margin deviation")
    lines.append(f"  D_avg = {d_avg:.3f} std")

    lines.append("\nMaximum standardized margin deviation")
    lines.append(f"  D_max = {d_max:.3f} std")

    lines.append("\nPer-margin standardized deviation breakdown")
    lines.append(f"  row_mean_deviation = {[round(v, 3) for v in breakdown['row_mean_deviation'].tolist()]}")
    lines.append(f"  col_mean_deviation = {[round(v, 3) for v in breakdown['col_mean_deviation'].tolist()]}")

    lines.append("\nOracle mean S_2 vs. guided empirical mean S_2")
    lines.append(f"  oracle_mean_S2 = {oracle_s2:.2f}")
    lines.append(f"  guided_mean_S2 = {guided_s2:.2f}")

    lines.append("\nMean soft reward (unguided -> guided)")
    lines.append(f"  {unguided_reward:.3f} -> {guided_reward:.3f}")

    lines.append("\nDiversity")
    lines.append(f"  {distinct}/{total} = {distinct / total:.1%} distinct")

    text = "\n".join(lines)
    print(text)

    os.makedirs("results", exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved diagnostics evaluation report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
