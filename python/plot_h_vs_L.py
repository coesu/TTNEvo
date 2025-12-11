"""
Plot threshold fields h(β=β*) versus system size L using precomputed fits.

This reads the CSV produced by ``fit_beta_vs_h.py`` (default location:
``plots/beta_decay/tables/beta_hc_estimates.csv``) and creates an error-bar plot
showing how the field values where β reaches 0.010 and 0.005 depend on L. It can
optionally emit an additional figure displaying the mean h over all requested
thresholds with propagated uncertainties.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

DEFAULT_THRESHOLDS: Tuple[float, ...] = (0.01, 0.005)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot h values at which β reaches selected thresholds versus L."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("plots") / "beta_decay" / "tables" / "beta_hc_bootstrap.csv",
        help="CSV file containing threshold crossings from fit_beta_vs_h.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots") / "beta_decay" / "h_vs_L.png",
        help="Output image file (PNG).",
    )
    parser.add_argument(
        "--mean-output",
        type=Path,
        default=Path("plots") / "beta_decay" / "h_vs_L_bootstrap_mean.pdf",
        help="Optional PNG for the mean-over-thresholds plot (omit by passing 'None').",
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
            linestyle="none",
            linewidth=1.2,
            capsize=4,
            color=color,
            label=rf"\beta = {threshold:.3f}",
        )

    ax.set_xlabel("L")
    ax.set_ylabel(r"$h_c$")
    ax.set_xticks([4, 6, 8, 10, 12])
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_ylim(top=35)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=False,
        right=False,
        bottom=True,
        left=True,
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    # ax.legend(title=r"Threshold $\beta$")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_threshold_mean_vs_L(
    df: pd.DataFrame,
    thresholds: Sequence[float],
    output_path: Optional[Path] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    """Plot the mean h field over all requested thresholds with propagated error."""
    if not thresholds:
        raise ValueError("At least one threshold is required to compute a mean plot.")

    cols = []
    for threshold in thresholds:
        suffix = f"{threshold:.3f}".replace(".", "p")
        value_col = f"h_beta_{suffix}"
        err_col = f"h_beta_{suffix}_stderr"
        if value_col not in df.columns:
            raise ValueError(
                f"Column '{value_col}' missing; run fit_beta_vs_h.py for β={threshold:.3f}."
            )
        cols.append((value_col, err_col))

    mean_vals = []
    mean_errs = []
    for _, row in df.iterrows():
        row_vals = []
        for value_col, err_col in cols:
            val = row.get(value_col)
            if pd.notna(val):
                row_vals.append(val)

        n_vals = len(row_vals)
        if n_vals:
            mean_vals.append(float(sum(row_vals)) / n_vals)
            min_val = min(row_vals)
            max_val = max(row_vals)
            lower_err = float(mean_vals[-1] - min_val)
            upper_err = float(max_val - mean_vals[-1])
            mean_errs.append([lower_err, upper_err])
        else:
            mean_vals.append(float("nan"))
            mean_errs.append(float("nan"))

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    # Convert asymmetric errors to numpy array with shape (2, N)
    yerr = []
    for err in mean_errs:
        if isinstance(err, list):
            yerr.append(err)
        else:
            yerr.append([np.nan, np.nan])
    yerr_arr = np.array(yerr).T if yerr else None

    ax.errorbar(
        df["L"],
        mean_vals,
        yerr=yerr_arr,
        marker="o",
        linestyle="none",
        color="black",
        linewidth=1.4,
        capsize=4,
    )
    ax.set_xlabel("$L$")
    ax.set_ylabel(r"$h_c$")
    ax.set_xticks([4, 6, 8, 10, 12])
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_ylim(top=35)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=False,
        right=False,
        bottom=True,
        left=True,
    )
    ax.grid(True, linestyle="--", alpha=0.3)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if fig:
            fig.tight_layout()
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
        else:
            ax.get_figure().savefig(output_path, dpi=200)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.results)
    # plot_thresholds_vs_L(df, thresholds=args.thresholds, output_path=args.output)
    mean_output = args.mean_output
    if mean_output and str(mean_output).lower() != "none":
        plot_threshold_mean_vs_L(
            df, thresholds=args.thresholds, output_path=mean_output
        )
    # print(f"Saved plot to: {args.output}")
    if mean_output and str(mean_output).lower() != "none":
        print(f"Saved mean plot to: {mean_output}")


if __name__ == "__main__":
    main()
