"""Render the gamma-sweep oracle S_2 table (see gamma.py) as a saved image.

Loads gamma_sweep_results.pt (produced by gamma.py; generated on demand here
if missing) and draws the same Gamma / Oracle mean S_2 / Mean reward / ESS
fraction table gamma.py prints to stdout, saved as a PNG in tuning/.

Run from the project directory:
    cd /sailhome/kadenwu/contingency_table_diffusion
    python tuning/oracle.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch

from tuning.gamma import main as run_gamma_sweep

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "gamma_sweep_results.pt")
OUT_PATH = os.path.join(os.path.dirname(__file__), "oracle_table.png")


def main() -> None:
    if not os.path.exists(RESULTS_PATH):
        print(f"{RESULTS_PATH} not found, running gamma sweep to generate it...")
        run_gamma_sweep()

    data = torch.load(RESULTS_PATH)
    gammas = data["gammas"].tolist()
    oracle_mean_s2 = data["oracle_mean_s2"].tolist()
    mean_reward = data["mean_reward"].tolist()
    ess_fraction = data["ess_fraction"].tolist()
    pool_size = data["pool_size"]

    col_labels = ["Gamma", "Oracle mean S_2", "Mean reward", "ESS fraction"]
    cell_text = [
        [f"{g:.4f}", f"{os2:.2f}", f"{mr:.5f}", f"{ess:.5f}"]
        for g, os2, mr, ess in zip(gammas, oracle_mean_s2, mean_reward, ess_fraction)
    ]

    fig, ax = plt.subplots(figsize=(7, 0.6 + 0.4 * len(cell_text)))
    ax.axis("off")
    ax.set_title(f"Oracle S_2 vs. gamma (pool_size={pool_size})", fontsize=12, pad=12)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    table.auto_set_column_width(col=list(range(len(col_labels))))

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved oracle table image to {OUT_PATH}")


if __name__ == "__main__":
    main()
