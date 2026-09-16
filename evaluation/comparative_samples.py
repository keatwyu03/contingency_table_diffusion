"""S_2 summary statistics for guided samples: mean, median, min, max, and the
exact-margin rate (fraction of samples with S_2 = 0). Separate from
diagnostics_eval.py's oracle/unguided comparisons -- this is a simpler,
purely descriptive view of the guided sample set's S_2 distribution.

Run from the project directory:
    cd /sailhome/kadenwu/contingency_table_diffusion
    python evaluation/comparative_samples.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import Config
from table_space import exact_margins_satisfied, squared_margin_error

SAMPLES_PATH = os.path.join("results", "guided_samples.pt")
REPORT_PATH = os.path.join("results", "comparative_samples_report.txt")


def main() -> None:
    cfg = Config()

    if not os.path.exists(SAMPLES_PATH):
        raise FileNotFoundError(
            f"{SAMPLES_PATH} not found. Run `python main.py sample-guided` first."
        )
    blob = torch.load(SAMPLES_PATH)
    guided_tables = blob["tables"]

    target_rows = torch.tensor(cfg.target_rows, dtype=torch.float32)
    target_cols = torch.tensor(cfg.target_cols, dtype=torch.float32)

    s2 = squared_margin_error(guided_tables, target_rows, target_cols)
    exact = exact_margins_satisfied(guided_tables, target_rows, target_cols)

    lines = ["Comparative sample statistics (S_2)", "=" * 40]
    lines.append(f"num_samples = {guided_tables.shape[0]}")
    lines.append(f"Mean S_2 = {s2.mean().item():.4f}")
    lines.append(f"Median S_2 = {s2.median().item():.4f}")
    lines.append(f"Minimum S_2 = {s2.min().item():.4f}")
    lines.append(f"Maximum S_2 = {s2.max().item():.4f}")
    lines.append(f"Exact-margin rate (S_2 = 0) = {exact.float().mean().item():.4f}")

    text = "\n".join(lines)
    print(text)

    os.makedirs("results", exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved comparative samples report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
