"""
Module for generating combined plots for the beta decay analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from fit_imbalance_decay import plot_beta_vs_h
from plot_h_vs_L import plot_threshold_mean_vs_L, DEFAULT_THRESHOLDS


def plot_combined_beta_h_and_thresholds(
    beta_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    output_path: Path,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> None:
    """
    Creates a combined figure with two subplots vertically stacked:
    1. Beta vs h (bootstrap mean with fit)
    2. Mean threshold h vs L
    """
    # Create figure with 2 subplots vertically stacked
    fig, axes = plt.subplots(2, 1, figsize=(4, 6))

    # Subplot 1: Beta vs h
    # We use beta_df (results from fits) here.
    # We want "beta_bootstrap_mean" and "beta_bootstrap_std"
    plot_beta_vs_h(
        beta_df,
        value_column="beta_bootstrap_mean",
        error_column="beta_bootstrap_std",
        output_path=None,  # Don't save individual plot
        ax=axes[0],
        show_fit=True,
        fit_thresholds=thresholds,
        # title=r"$\beta$ vs $h$"
    )
    # Add (a) label
    axes[0].text(
        0.01,
        1.1,
        "(a)",
        transform=axes[0].transAxes,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # Subplot 2: h vs L
    # We use threshold_df (bootstrap summary) here.
    plot_threshold_mean_vs_L(
        threshold_df,
        thresholds=thresholds,
        output_path=None,  # Don't save individual plot
        ax=axes[1],
    )
    # Add (b) label
    axes[1].text(
        0.01,
        1.1,
        "(b)",
        transform=axes[1].transAxes,
        fontweight="bold",
        va="top",
        ha="left",
    )

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
