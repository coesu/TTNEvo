"""
End-to-end orchestration script for the imbalance analysis workflow.

By default this script recomputes:
  1. Disorder-averaged data from `subset_data.pkl`.
  2. Power-law (log-log) fits of the imbalance decay (`fit_imbalance_decay.py`).
  3. β(h) exponential fits for weighted and bootstrap means (`fit_beta_vs_h.py` logic).
  4. Threshold-field plots h(β=0.010) and h(β=0.005) vs L.

Flags allow reusing pre-existing outputs to save time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, Callable

import pandas as pd

from load_subset_data import load_from_pickle
from process_data import average_over_disorder, save_averaged_data
from fit_imbalance_decay import (
    main as run_beta_fits,
    PLOT_DIR_DEFAULT,
)
from fit_beta_vs_h import compute_threshold_summary, THRESHOLD_VALUES
from plot_h_vs_L import (
    plot_thresholds_vs_L,
    plot_threshold_mean_vs_L,
    DEFAULT_THRESHOLDS as H_THRESHOLDS,
)

PYTHON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PYTHON_DIR.parent

SUBSET_PKL = PROJECT_ROOT / "data_beta.pkl"
AVERAGED_PKL = PROJECT_ROOT / "data_beta_averaged.pkl"

TABLE_DIR = PYTHON_DIR / "plots" / "beta_decay" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = TABLE_DIR / "beta_decay_results.csv"
WEIGHTED_THRESHOLDS_CSV = TABLE_DIR / "beta_hc_estimates.csv"
BOOTSTRAP_THRESHOLDS_CSV = TABLE_DIR / "beta_hc_bootstrap.csv"

PLOT_DIR = PYTHON_DIR / "plots" / "beta_decay"
H_VS_L_PNG = PLOT_DIR / "h_vs_L.png"
H_VS_L_BOOT_PNG = PLOT_DIR / "h_vs_L_bootstrap.png"
H_VS_L_MEAN_PNG = PLOT_DIR / "h_vs_L_mean.png"
H_VS_L_BOOT_MEAN_PNG = PLOT_DIR / "h_vs_L_bootstrap_mean.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete imbalance analysis pipeline."
    )
    parser.add_argument(
        "--reuse-averaged",
        action="store_true",
        help="Reuse ../subset_data_averaged.pkl if present.",
    )
    parser.add_argument(
        "--reuse-beta-fits",
        action="store_true",
        help="Reuse cached beta decay fits if plots/tables already exist.",
    )
    parser.add_argument(
        "--reuse-thresholds",
        action="store_true",
        help="Reuse weighted threshold table (beta_hc_estimates.csv).",
    )
    parser.add_argument(
        "--reuse-bootstrap-thresholds",
        action="store_true",
        help="Reuse bootstrap threshold table (beta_hc_bootstrap.csv).",
    )
    parser.add_argument(
        "--reuse-h-plots",
        action="store_true",
        help="Reuse h-vs-L plots if already generated.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output from intermediate steps.",
    )
    return parser.parse_args()


def ensure_averaged_data(reuse: bool, quiet: bool) -> None:
    if reuse and AVERAGED_PKL.exists():
        print(f"[skip] Using existing averaged data: {AVERAGED_PKL}")
        return

    if not SUBSET_PKL.exists():
        raise FileNotFoundError(
            f"Raw subset data not found at {SUBSET_PKL}. "
            "Please ensure the file is present or provide --reuse-averaged."
        )

    print(f"[run] Computing disorder-averaged data -> {AVERAGED_PKL}")
    df_raw = load_from_pickle(SUBSET_PKL, verbose=not quiet)
    df_avg = average_over_disorder(
        df_raw,
        compute_std=True,
        compute_sem=True,
        verbose=not quiet,
    )
    save_averaged_data(df_avg, filepath=str(AVERAGED_PKL), verbose=not quiet)


def ensure_beta_fits(reuse: bool, quiet: bool) -> pd.DataFrame:
    if reuse and RESULTS_CSV.exists():
        print(f"[skip] Using cached beta fits: {RESULTS_CSV}")
        return pd.read_csv(RESULTS_CSV)

    print("[run] Fitting imbalance decay (log-log regression)")
    results_df = run_beta_fits(
        averaged_data_path=AVERAGED_PKL,
    )
    if quiet:
        print(f"[done] Saved beta fit outputs under {PLOT_DIR}")
    return results_df


def ensure_threshold_table(
    beta_df: pd.DataFrame,
    value_column: str,
    error_column: str,
    output_path: Path,
    reuse: bool,
    quiet: bool,
) -> pd.DataFrame:
    if reuse and output_path.exists():
        print(f"[skip] Using existing threshold table: {output_path}")
        return pd.read_csv(output_path)

    print(f"[run] Computing threshold summary ({value_column} vs h) -> {output_path}")
    summary_df = compute_threshold_summary(
        beta_df,
        value_column,
        error_column,
        thresholds=THRESHOLD_VALUES,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    if quiet:
        print(f"[done] Wrote {output_path}")
    return summary_df


def ensure_h_vs_L_plot(
    summary_df: pd.DataFrame,
    output_path: Path,
    reuse: bool,
    quiet: bool,
    thresholds: Sequence[float],
    *,
    plotter: Callable[[pd.DataFrame, Sequence[float], Path], None],
    description: str,
) -> None:
    if reuse and output_path.exists():
        print(f"[skip] Using existing {description} figure: {output_path}")
        return

    print(f"[run] Rendering {description} plot -> {output_path}")
    plotter(summary_df, thresholds=thresholds, output_path=output_path)
    if quiet:
        print(f"[done] Saved {output_path}")


def main() -> None:
    args = parse_args()

    ensure_averaged_data(args.reuse_averaged, args.quiet)

    beta_df = ensure_beta_fits(args.reuse_beta_fits, args.quiet)
    if beta_df is None or beta_df.empty:
        if RESULTS_CSV.exists():
            beta_df = pd.read_csv(RESULTS_CSV)
        else:
            raise RuntimeError("No beta fit data available for downstream steps.")

    weighted_summary = ensure_threshold_table(
        beta_df,
        value_column="beta_weighted",
        error_column="beta_weighted_stderr",
        output_path=WEIGHTED_THRESHOLDS_CSV,
        reuse=args.reuse_thresholds,
        quiet=args.quiet,
    )

    bootstrap_summary = ensure_threshold_table(
        beta_df,
        value_column="beta_bootstrap_mean",
        error_column="beta_bootstrap_std",
        output_path=BOOTSTRAP_THRESHOLDS_CSV,
        reuse=args.reuse_bootstrap_thresholds,
        quiet=args.quiet,
    )

    # ensure_h_vs_L_plot(
    #     weighted_summary,
    #     output_path=H_VS_L_PNG,
    #     reuse=args.reuse_h_plots,
    #     quiet=args.quiet,
    #     thresholds=H_THRESHOLDS,
    #     plotter=plot_thresholds_vs_L,
    #     description="threshold h(L)",
    # )
    # ensure_h_vs_L_plot(
    #     weighted_summary,
    #     output_path=H_VS_L_MEAN_PNG,
    #     reuse=args.reuse_h_plots,
    #     quiet=args.quiet,
    #     thresholds=H_THRESHOLDS,
    #     plotter=plot_threshold_mean_vs_L,
    #     description="mean threshold h(L)",
    # )
    # ensure_h_vs_L_plot(
    #     bootstrap_summary,
    #     output_path=H_VS_L_BOOT_PNG,
    #     reuse=args.reuse_h_plots,
    #     quiet=args.quiet,
    #     thresholds=H_THRESHOLDS,
    #     plotter=plot_thresholds_vs_L,
    #     description="bootstrap threshold h(L)",
    # )
    ensure_h_vs_L_plot(
        bootstrap_summary,
        output_path=H_VS_L_BOOT_MEAN_PNG,
        reuse=args.reuse_h_plots,
        quiet=args.quiet,
        thresholds=H_THRESHOLDS,
        plotter=plot_threshold_mean_vs_L,
        description="bootstrap mean threshold h(L)",
    )

    print("[done] Full analysis pipeline completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        raise
