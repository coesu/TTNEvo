"""
Estimate the field values where β(h) reaches selected thresholds.

The script reads the β fit summary table produced by ``fit_imbalance_decay.py``
and performs a Levenberg-Marquardt polynomial fit of β vs h for each system
size L. By default it reports the h positions where β ≈ 0 (extrapolated),
β = 0.010, and β = 0.005, including 1σ uncertainties obtained from parameter
covariances. Results are written to
``plots/beta_decay/tables/beta_hc_estimates.csv``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from fit_imbalance_decay import PLOT_DIR_DEFAULT, fit_beta_curve_lm

THRESHOLD_VALUES = (0.01, 0.005)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit β(h) and estimate h where β crosses zero."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PLOT_DIR_DEFAULT / "tables" / "beta_decay_results.csv",
        help="CSV file containing β fit results.",
    )
    parser.add_argument(
        "--value-column",
        default="beta_weighted",
        choices=["beta_unweighted", "beta_weighted", "beta_bootstrap_mean"],
        help="Column to fit as a function of h.",
    )
    parser.add_argument(
        "--error-column",
        default="beta_weighted_stderr",
        help="Column with 1σ uncertainties for weighting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PLOT_DIR_DEFAULT / "tables" / "beta_hc_estimates.csv",
        help="Path to save the fit summary table.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=4,
        help="Minimum number of data points required for a fit.",
    )
    parser.add_argument(
        "--hmin",
        type=float,
        default=None,
        help="Optional lower bound on h values used in the fit.",
    )
    parser.add_argument(
        "--hmax",
        type=float,
        default=None,
        help="Optional upper bound on h values used in the fit.",
    )
    return parser.parse_args()


def compute_threshold_summary(
    df: pd.DataFrame,
    value_column: str,
    error_column: str,
    *,
    thresholds=THRESHOLD_VALUES,
    min_points: int = 4,
    hmin: Optional[float] = None,
    hmax: Optional[float] = None,
) -> pd.DataFrame:
    records = []
    for L_value in sorted(df["L"].unique()):
        subset = df[df["L"] == L_value].copy()
        if hmin is not None:
            subset = subset[subset["h"] >= hmin]
        if hmax is not None:
            subset = subset[subset["h"] <= hmax]

        values = subset[value_column].to_numpy()
        errors = (
            subset[error_column].to_numpy()
            if error_column in subset.columns
            else None
        )

        mask = np.isfinite(values)
        if errors is not None:
            mask &= np.isfinite(errors) & (errors > 0)
        mask &= np.isfinite(subset["h"])

        subset = subset[mask]
        if len(subset) < min_points:
            record = {
                "L": L_value,
                "n_points": len(subset),
                "order": np.nan,
                "r_squared": np.nan,
                "slope": np.nan,
                "slope_stderr": np.nan,
                "intercept": np.nan,
                "intercept_stderr": np.nan,
            }
            for target in thresholds:
                suffix = f"{target:.3f}".replace(".", "p")
                record[f"h_beta_{suffix}"] = np.nan
                record[f"h_beta_{suffix}_stderr"] = np.nan
            records.append(record)
            continue

        fit_res = fit_beta_curve_lm(
            subset["h"].to_numpy(dtype=float),
            subset[value_column].to_numpy(dtype=float),
            sigma_beta=subset[error_column].to_numpy(dtype=float)
            if errors is not None
            else None,
            thresholds=thresholds,
        )

        record = {
            "L": L_value,
            "n_points": fit_res.n_points,
            "order": fit_res.order,
            "r_squared": fit_res.r_squared,
            "slope": fit_res.slope,
            "slope_stderr": fit_res.slope_stderr,
            "intercept": fit_res.intercept,
            "intercept_stderr": fit_res.intercept_stderr,
        }
        for target in thresholds:
            crossing, std = fit_res.thresholds.get(target, (float("nan"), float("nan")))
            suffix = f"{target:.3f}".replace(".", "p")
            record[f"h_beta_{suffix}"] = crossing
            record[f"h_beta_{suffix}_stderr"] = std
        records.append(record)

    result_df = pd.DataFrame(records)
    base_cols = ["L", "n_points", "order", "r_squared", "slope", "slope_stderr", "intercept", "intercept_stderr"]
    param_cols: list[str] = []
    thresh_cols = []
    for target in thresholds:
        suffix = f"{target:.3f}".replace(".", "p")
        h_col = f"h_beta_{suffix}"
        err_col = f"h_beta_{suffix}_stderr"
        if h_col in result_df.columns:
            thresh_cols.extend([h_col, err_col])
    other_cols = [
        col
        for col in result_df.columns
        if col not in set(base_cols + param_cols + thresh_cols)
    ]
    ordered_cols = base_cols + param_cols + thresh_cols + other_cols
    result_df = result_df[ordered_cols]
    return result_df


def main() -> None:
    args = parse_args()

    if not args.results.exists():
        raise FileNotFoundError(
            f"Results file not found: {args.results}. "
            "Run fit_imbalance_decay.py first or adjust --results."
        )

    df = pd.read_csv(args.results)

    result_df = compute_threshold_summary(
        df,
        args.value_column,
        args.error_column,
        thresholds=THRESHOLD_VALUES,
        min_points=args.min_points,
        hmin=args.hmin,
        hmax=args.hmax,
    )

    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)

    print("β(h) fit results (Levenberg-Marquardt):")
    print(result_df.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    print(f"\nSaved summary to: {args.output}")


if __name__ == "__main__":
    main()
