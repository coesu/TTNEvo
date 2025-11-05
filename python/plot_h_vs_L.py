"""
Plot threshold fields h(β=β*) versus system size L using precomputed fits.

This reads the CSV produced by ``fit_beta_vs_h.py`` (default location:
``plots/beta_decay/tables/beta_hc_estimates.csv``) and creates an error-bar plot
showing how the field values where β reaches 0.010 and 0.005 depend on L.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_THRESHOLDS: Tuple[float, ...] = (0.01, 0.005)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot h values at which β reaches selected thresholds versus L."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("plots") / "beta_decay" / "tables" / "beta_hc_estimates.csv",
        help="CSV file containing threshold crossings from fit_beta_vs_h.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots") / "beta_decay" / "h_vs_L.png",
        help="Output image file (PNG).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Threshold β values to plot (must match columns in the CSV).",
    )
    return parser.parse_args()


def plot_thresholds_vs_L(
    df: pd.DataFrame,
    thresholds: Sequence[float],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    markers = ["o", "s", "D", "^", "v"]
    palette = plt.cm.viridis

    for idx, threshold in enumerate(thresholds):
        suffix = f"{threshold:.3f}".replace(".", "p")
        value_col = f"h_beta_{suffix}"
        err_col = f"h_beta_{suffix}_stderr"

        if value_col not in df.columns:
            raise ValueError(
                f"Column '{value_col}' not found in results. "
                "Did you run fit_beta_vs_h.py with matching thresholds?"
            )

        color = palette(idx / max(1, len(thresholds) - 1))
        ax.errorbar(
            df["L"],
            df[value_col],
            yerr=df.get(err_col),
            marker=markers[idx % len(markers)],
            linestyle="-",
            linewidth=1.2,
            capsize=4,
            color=color,
            label=f"β = {threshold:.3f}",
        )

    ax.set_xlabel("System size L")
    ax.set_ylabel("Field h where β(h) meets threshold")
    ax.set_title("Threshold fields vs system size")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Threshold β")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.results)
    plot_thresholds_vs_L(df, thresholds=args.thresholds, output_path=args.output)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
