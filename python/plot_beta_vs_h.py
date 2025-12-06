"""
Standalone script to plot β vs h using precomputed fit results.

This avoids rerunning the power-law fits. It expects the CSV file produced by
``fit_imbalance_decay.py`` (default location:
``plots/beta_decay/tables/beta_decay_results.csv``) and generates
``beta_vs_h_unweighted.png``, ``beta_vs_h_weighted.png``, and
``beta_vs_h_bootstrap.png`` with exponential (log-linear) overlays that mark
where β drops to 0.010 and 0.005.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fit_imbalance_decay import (
    PLOT_DIR_DEFAULT,
    plot_beta_vs_h,
)

import matplotlib as mpl

mpl.rcParams.update({
  "text.usetex": True,          # route text through LaTeX
  "font.family": "serif",       # LaTeX default
  "font.serif": ["Computer Modern Roman"],
  "text.latex.preamble": r"\usepackage{amsmath}",  # optional extras
})

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot β vs h curves from existing fit results."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PLOT_DIR_DEFAULT / "tables" / "beta_decay_results.csv",
        help="Path to the CSV file with β fit results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOT_DIR_DEFAULT,
        help="Base directory where plots will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_csv = args.results

    if not results_csv.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_csv}. "
            "Run fit_imbalance_decay.py first or adjust --results."
        )

    df = pd.read_csv(results_csv)

    # plot_beta_vs_h(
    #     df,
    #     value_column="beta_unweighted",
    #     error_column="beta_unweighted_stderr",
    #     output_path=args.output_dir / "beta_vs_h_unweighted.png",
    #     title=r"$\beta$ vs $h$ (unweighted fits)",
    #     show_fit=True,
    #     fit_thresholds=(0.01, 0.005),
    # )
    # plot_beta_vs_h(
    #     df,
    #     value_column="beta_weighted",
    #     error_column="beta_weighted_stderr",
    #     output_path=args.output_dir / "beta_vs_h_weighted.png",
    #     title=r"$\beta$ vs $h$ (weighted fits)",
    #     show_fit=True,
    #     fit_thresholds=(0.01, 0.005),
    # )
    plot_beta_vs_h(
        df,
        value_column="beta_bootstrap_mean",
        error_column="beta_bootstrap_std",
        output_path=args.output_dir / "beta_vs_h_bootstrap.pdf",
        # title=r"$\beta$ vs $h$ (bootstrap mean +/- sigma)",
        show_fit=True,
        fit_thresholds=(0.01, 0.005),
    )
    print(f"Saved plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
