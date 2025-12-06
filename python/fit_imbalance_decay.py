"""
Power-law fitting utilities for imbalance decay data.

This module loads the disorder-averaged imbalance dataset and performs
log-log linear regression fits of the form

    imbalance(t) = C * t**(-beta)

over a specified time window (default: t in [50, 100]).
It computes beta for every (L, h) parameter combination using both
unweighted and uncertainty-weighted regressions and estimates beta
uncertainties via two complementary approaches:

1. Covariance of the linear regression fit (analytic standard error).
2. Parametric bootstrap using the reported standard errors of the imbalance.

Outputs include tabulated fit results and diagnostic plots.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from process_data import load_averaged_data

import matplotlib as mpl

mpl.rcParams.update({
  "text.usetex": True,          # route text through LaTeX
  "font.family": "serif",       # LaTeX default
  "font.serif": ["Computer Modern Roman"],
  "text.latex.preamble": r"\usepackage{amsmath}",  # optional extras
})

FIT_WINDOW_DEFAULT: Tuple[float, float] = (30.0, 100.0)
MIN_POINTS_DEFAULT: int = 10
BOOTSTRAP_SAMPLES_DEFAULT: int = 10
PLOT_DIR_DEFAULT = Path("plots") / "beta_decay"
LM_BOOTSTRAP_SAMPLES_DEFAULT: int = 10

# Allowed disorder strengths for each system size included in downstream β(h) analysis.
ALLOWED_H_BY_L: Dict[int, Tuple[float, ...]] = {
    4: (5, 10, 15, 20),
    6: (5, 10, 15, 20, 25),
    8: (5, 10, 15, 20, 25, 35),
    10: (5, 10, 15,20, 25, 35),
    12: (10, 15, 20, 25, 30, 35, 40),
}

PALETTE = ['#3A9AB2', '#A5C2A3', '#DCCB4E', '#E79805', '#F11B00']

_H_MATCH_ATOL = 1e-6


class FitError(RuntimeError):
    """Raised when a fit cannot be performed."""


@dataclass
class RegressionResult:
    """Container for log-log regression results."""

    beta: float
    beta_stderr: float
    intercept: float
    intercept_stderr: float
    r_squared: float
    n_points: int
    method: str

    @property
    def prefactor(self) -> float:
        """Return exp(intercept)."""
        return float(np.exp(self.intercept))


def _is_allowed_h(L_value: float, h_value: float) -> bool:
    """Return True when h is within the allowed set for this L (if any)."""
    allowed = ALLOWED_H_BY_L.get(int(L_value))
    if allowed is None:
        return True

    return any(math.isclose(h_value, allowed_h, abs_tol=_H_MATCH_ATOL) for allowed_h in allowed)


def _select_fit_slice(
    times: np.ndarray,
    imbalance: np.ndarray,
    imbalance_sem: Optional[np.ndarray],
    fit_window: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Restrict arrays to the requested time window and valid entries."""
    t_min, t_max = fit_window
    if t_min <= 0.0:
        raise ValueError("fit_window must start at positive time to avoid log(0).")

    mask = (times >= t_min) & (times <= t_max) & np.isfinite(imbalance) & (imbalance > 0)

    if imbalance_sem is not None:
        mask &= np.isfinite(imbalance_sem)

    times_sel = times[mask]
    imb_sel = imbalance[mask]
    imb_sem_sel = None if imbalance_sem is None else imbalance_sem[mask]

    return times_sel, imb_sel, imb_sem_sel


def _log_arrays(
    times: np.ndarray,
    imbalance: np.ndarray,
    imbalance_sem: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return log-transformed arrays and propagated SEM on log scale."""
    log_t = np.log(times)
    log_imb = np.log(imbalance)

    if imbalance_sem is None:
        return log_t, log_imb, None

    # Propagate standard error: σ_log = σ / value
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_log = np.where(
            imbalance > 0,
            imbalance_sem / imbalance,
            np.inf,
        )

    # Guard against zero or negative uncertainties
    sigma_log = np.where((sigma_log == 0) | ~np.isfinite(sigma_log), np.nan, sigma_log)

    return log_t, log_imb, sigma_log


def _linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    sigma_y: Optional[np.ndarray],
    method_label: str,
    min_points: int = MIN_POINTS_DEFAULT,
) -> RegressionResult:
    """Perform linear regression (possibly weighted) in log-log space."""
    if x.size < min_points:
        raise FitError(
            f"{method_label}: insufficient points ({x.size} < {min_points}) for regression."
        )

    if sigma_y is not None:
        # Drop entries with invalid uncertainties
        valid = np.isfinite(sigma_y) & (sigma_y > 0)
        x = x[valid]
        y = y[valid]
        sigma_y = sigma_y[valid]

        if x.size < min_points:
            raise FitError(
                f"{method_label}: insufficient valid points after filtering ({x.size})."
            )

        weights = 1.0 / sigma_y
    else:
        weights = None

    coeffs, cov = np.polyfit(x, y, deg=1, w=weights, cov=True)
    slope, intercept = coeffs
    slope_stderr = math.sqrt(cov[0, 0])
    intercept_stderr = math.sqrt(cov[1, 1])

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return RegressionResult(
        beta=-slope,
        beta_stderr=slope_stderr,
        intercept=intercept,
        intercept_stderr=intercept_stderr,
        r_squared=r_squared,
        n_points=x.size,
        method=method_label,
    )


def bootstrap_beta(
    x: np.ndarray,
    y: np.ndarray,
    sigma_y: np.ndarray,
    *,
    n_samples: int = BOOTSTRAP_SAMPLES_DEFAULT,
    random_state: Optional[int] = 17,
) -> Tuple[float, float]:
    """
    Estimate beta uncertainty via parametric bootstrap on log data.

    Returns mean and standard deviation of the bootstrap distribution.
    """
    if sigma_y is None:
        raise FitError("Bootstrap requires measurement uncertainties (sigma_y).")

    valid = np.isfinite(sigma_y) & (sigma_y > 0)
    x = x[valid]
    y = y[valid]
    sigma_y = sigma_y[valid]

    if x.size < MIN_POINTS_DEFAULT:
        raise FitError("Bootstrap: insufficient valid points for sampling.")

    rng = np.random.default_rng(random_state)
    weights = 1.0 / sigma_y
    slopes = []

    for _ in range(n_samples):
        sampled_y = rng.normal(loc=y, scale=sigma_y)
        coeffs = np.polyfit(x, sampled_y, deg=1, w=weights)
        slopes.append(coeffs[0])

    slopes = np.array(slopes)
    beta_samples = -slopes
    return float(np.mean(beta_samples)), float(np.std(beta_samples, ddof=1) *2.)


def fit_single_row(
    row: pd.Series,
    *,
    fit_window: Tuple[float, float] = FIT_WINDOW_DEFAULT,
    min_points: int = MIN_POINTS_DEFAULT,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES_DEFAULT,
    random_state: Optional[int] = 17,
) -> dict:
    """Compute fit results for one (L, h) parameter row."""
    times = np.asarray(row["t"], dtype=float)
    imb_mean = np.asarray(row["imb_mean"], dtype=float)
    imb_sem_raw = row.get("imb_sem") if "imb_sem" in row else None
    if imb_sem_raw is None:
        imb_sem = None
    else:
        imb_sem = np.asarray(imb_sem_raw, dtype=float)

    times_sel, imb_sel, imb_sem_sel = _select_fit_slice(
        times, imb_mean, imb_sem, fit_window
    )

    if times_sel.size < min_points:
        raise FitError(
            f"L={row['L']}, h={row['h']}: only {times_sel.size} points in fit window."
        )

    log_t, log_imb, sigma_log = _log_arrays(times_sel, imb_sel, imb_sem_sel)

    unweighted_res = _linear_regression(
        log_t, log_imb, sigma_y=None, method_label="unweighted", min_points=min_points
    )

    weighted_res = None
    if sigma_log is not None and np.any(np.isfinite(sigma_log)):
        weighted_res = _linear_regression(
            log_t,
            log_imb,
            sigma_y=sigma_log,
            method_label="weighted",
            min_points=min_points,
        )

    bootstrap_mean = bootstrap_std = math.nan
    if sigma_log is not None and np.any(np.isfinite(sigma_log)):
        bootstrap_mean, bootstrap_std = bootstrap_beta(
            log_t,
            log_imb,
            sigma_log,
            n_samples=bootstrap_samples,
            random_state=random_state,
        )

    result = {
        "L": row["L"],
        "h": row["h"],
        "maxdim": row.get("maxdim"),
        "num_realizations": row.get("num_realizations"),
        "n_points": unweighted_res.n_points,
        "fit_t_min": float(fit_window[0]),
        "fit_t_max": float(fit_window[1]),
        "beta_unweighted": unweighted_res.beta,
        "beta_unweighted_stderr": unweighted_res.beta_stderr,
        "c_unweighted": unweighted_res.prefactor,
        "r2_unweighted": unweighted_res.r_squared,
        "beta_weighted": weighted_res.beta if weighted_res else math.nan,
        "beta_weighted_stderr": (
            weighted_res.beta_stderr if weighted_res else math.nan
        ),
        "c_weighted": weighted_res.prefactor if weighted_res else math.nan,
        "r2_weighted": weighted_res.r_squared if weighted_res else math.nan,
        "beta_bootstrap_mean": bootstrap_mean,
        "beta_bootstrap_std": bootstrap_std,
    }

    return result


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_fit(
    times: np.ndarray,
    imbalance: np.ndarray,
    imbalance_sem: Optional[np.ndarray],
    fit_window: Tuple[float, float],
    fit_params: dict,
    output_path: Path,
) -> None:
    """Create a diagnostic log-log plot of the data and fitted curves."""
    _ensure_output_dir(output_path.parent)

    t_min, t_max = fit_window
    mask = (times >= t_min) & (times <= t_max) & (imbalance > 0)
    times_window = times[mask]
    imb_window = imbalance[mask]

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.loglog(times, imbalance, color="tab:blue", alpha=0.35, label="mean imbalance")
    ax.loglog(
        times_window,
        imb_window,
        marker="o",
        linestyle="",
        color="tab:blue",
        markersize=3,
        label="fit window",
    )

    if imbalance_sem is not None:
        imb_upper = np.clip(imbalance + imbalance_sem, a_min=1e-12, a_max=None)
        imb_lower = np.clip(imbalance - imbalance_sem, a_min=1e-12, a_max=None)
        ax.fill_between(
            times,
            imb_lower,
            imb_upper,
            color="tab:blue",
            alpha=0.15,
            linewidth=0,
            label="mean +/- SEM",
        )

    # Plot fitted curves
    for label, beta_key, c_key in [
        ("unweighted", "beta_unweighted", "c_unweighted"),
        ("weighted", "beta_weighted", "c_weighted"),
    ]:
        beta_val = fit_params.get(beta_key)
        c_val = fit_params.get(c_key)
        if not np.isfinite(beta_val) or not np.isfinite(c_val):
            continue
        fitted = c_val * times_window ** (-beta_val)
        ax.loglog(
            times_window,
            fitted,
            label=rf"{label} fit (\beta={beta_val:.3f})",
        )

    ax.set_xlabel("Time t")
    ax.set_ylabel("Disorder-averaged imbalance")
    ax.set_title(
        f"L={fit_params['L']:.0f}, h={fit_params['h']:.0f}; "
        f"fit window [{t_min:.0f}, {t_max:.0f}]"
    )
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_beta_heatmap(
    df: pd.DataFrame,
    value_column: str,
    *,
    output_path: Path,
    title: str,
) -> None:
    """Generate a heatmap of beta vs (L, h)."""
    _ensure_output_dir(output_path.parent)

    pivot = df.pivot(index="L", columns="h", values=value_column)
    pivot = pivot.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[
            pivot.columns.min(),
            pivot.columns.max(),
            pivot.index.min(),
            pivot.index.max(),
        ],
        cmap="viridis",
    )

    ax.set_xlabel("h")
    ax.set_ylabel("L")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\beta$")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_beta_vs_h(
    df: pd.DataFrame,
    *,
    value_column: str,
    error_column: Optional[str] = None,
    output_path: Path,
    title: Optional[str]=None,
    show_fit: bool = False,
    fit_thresholds: Sequence[float] = (0.01, 0.005),
) -> None:
    """Plot beta vs h for each system size L."""
    _ensure_output_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(4, 3))
    threshold_targets = [t for t in fit_thresholds if t > 0]
    threshold_windows: Dict[int, Dict[str, float]] = {}
    target_L_values = [4, 6, 12]
    available_L_values = set(df["L"].unique())
    unique_L = [L for L in target_L_values if L in available_L_values]
    if len(unique_L) == 3:
        selected_colors = [PALETTE[0], PALETTE[len(PALETTE) // 2], PALETTE[-1]]
    elif len(unique_L) <= len(PALETTE):
        selected_colors = PALETTE[: len(unique_L)]
    else:
        color_cycle = cycle(PALETTE)
        selected_colors = [next(color_cycle) for _ in unique_L]

    color_by_L = {
        L_val: color for L_val, color in zip(unique_L, selected_colors, strict=False)
    }

    for L in unique_L:
        subset_all = df[df["L"] == L].sort_values("h")
        allowed_h = ALLOWED_H_BY_L.get(int(L))
        if allowed_h is not None:
            subset_all = subset_all[subset_all["h"].isin(allowed_h)]

        if subset_all.empty:
            continue
        values_all = subset_all[value_column].to_numpy()
        errs_all = (
            subset_all[error_column].to_numpy()
            if error_column and error_column in subset_all
            else None
        )

        label_prefix = f"L={int(L)}"
        fit_res = None
        color = color_by_L[L]

        if show_fit:
            fit_mask = np.isfinite(values_all) & np.isfinite(subset_all["h"].to_numpy())
            if errs_all is not None:
                fit_mask &= np.isfinite(errs_all) & (errs_all > 0)

            if np.count_nonzero(fit_mask) >= 2:
                fit_res = fit_beta_curve_lm(
                    subset_all["h"].to_numpy(dtype=float)[fit_mask],
                    values_all.astype(float)[fit_mask],
                    sigma_beta=errs_all.astype(float)[fit_mask]
                    if errs_all is not None
                    else None,
                    thresholds=fit_thresholds,
                )

                if fit_res:
                    for threshold in threshold_targets:
                        crossing, crossing_err = fit_res.thresholds.get(
                            threshold, (float("nan"), float("nan"))
                        )
                        if np.isfinite(crossing) and crossing > 0:
                            err_val = (
                                float(crossing_err)
                                if np.isfinite(crossing_err) and crossing_err > 0
                                else 0.0
                            )
                            lower = max(float(crossing) - err_val, 0.0)
                            upper = float(crossing) + err_val

                            window = threshold_windows.setdefault(
                                L,
                                {
                                    "color": color,
                                    "lower": float("inf"),
                                    "upper": float("-inf"),
                                },
                            )
                            window["lower"] = min(window["lower"], lower)
                            window["upper"] = max(window["upper"], upper)

        # Log scale requires strictly positive values; filter accordingly.
        mask_plot = np.isfinite(values_all) & (values_all > 0)
        if errs_all is not None:
            mask_plot &= np.isfinite(errs_all)

        if not np.any(mask_plot):
            continue

        x_vals = subset_all["h"].to_numpy()[mask_plot]
        y_vals = values_all[mask_plot]
        y_errs = errs_all[mask_plot] if errs_all is not None else None

        ax.errorbar(
            x_vals,
            y_vals,
            yerr=y_errs,
            marker="o",
            linestyle="none",
            label=label_prefix,
            capsize=3,
            color=color,
        )

        if show_fit and fit_res:
            h_min = np.min(subset_all["h"])
            h_max = np.max(subset_all["h"])
            h_line = np.linspace(h_min, h_max, 200)
            beta_line = np.exp(fit_res.slope * h_line + fit_res.intercept)
            positive_mask = beta_line > 0
            if np.any(positive_mask):
                ax.plot(
                    h_line[positive_mask],
                    beta_line[positive_mask],
                    color=color,
                    linestyle="-",
                    linewidth=1.2,
                    label=f"{label_prefix} fit",
                )

                cov = fit_res.covariance
                if cov is not None and cov.shape == (2, 2) and np.all(np.isfinite(cov)):
                    # Var[log(beta)] = J Cov J^T with J = [h, 1]
                    log_var = (
                        cov[0, 0] * h_line**2
                        + 2 * cov[0, 1] * h_line
                        + cov[1, 1]
                    )
                    log_var = np.maximum(log_var, 0)
                    sigma_beta = beta_line * np.sqrt(log_var)
                    lower = np.clip(beta_line - sigma_beta, a_min=1e-12, a_max=None)
                    upper = beta_line + sigma_beta
                    ax.fill_between(
                        h_line,
                        lower,
                        upper,
                        where=positive_mask,
                        color=color,
                        alpha=0.18,
                        linewidth=0,
                    )

    ax.set_xlabel("$h$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-3)
    ax.tick_params(direction="in", which="both", top=False, right=False)
    window_label_added = False
    for L in unique_L:
        window = threshold_windows.get(L)
        if not window:
            continue
        lower = window.get("lower")
        upper = window.get("upper")
        if lower is None or upper is None:
            continue
        if not (np.isfinite(lower) and np.isfinite(upper)):
            continue

        color = window["color"]
        label = r"$h_c$ range estimate" if not window_label_added else None
        window_label_added = True

        if upper > lower:
            ax.axvline(
                lower,
                color=color,
                linewidth=1.2,
                linestyle="--",
                alpha=0.8,
                label=label,
            )
            ax.axvline(
                upper,
                color=color,
                linewidth=1.2,
                linestyle="--",
                alpha=0.8,
            )
        else:
            ax.axvline(
                lower,
                color=color,
                linewidth=1.2,
                linestyle="--",
                alpha=0.8,
                label=label,
            )
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(
        fontsize=9,
        title_fontsize=10,
        frameon=False,
        loc="best",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main(
    *,
    averaged_data_path: Path = Path("..") / "subset_data_averaged.pkl",
    fit_window: Tuple[float, float] = FIT_WINDOW_DEFAULT,
    min_points: int = MIN_POINTS_DEFAULT,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES_DEFAULT,
    output_dir: Path = PLOT_DIR_DEFAULT,
) -> pd.DataFrame:
    """Entry point: run fits for all available (L, h) combinations."""
    df = load_averaged_data(str(averaged_data_path), verbose=False)
    results = []
    failures = []

    for _, row in df.iterrows():
        if not _is_allowed_h(row["L"], row["h"]):
            continue
        try:
            res = fit_single_row(
                row,
                fit_window=fit_window,
                min_points=min_points,
                bootstrap_samples=bootstrap_samples,
            )
            results.append(res)

            plot_path = output_dir / "fits" / f"L{int(row['L'])}_h{int(row['h'])}.png"
            plot_fit(
                np.asarray(row["t"], dtype=float),
                np.asarray(row["imb_mean"], dtype=float),
                (np.asarray(row["imb_sem"], dtype=float) if "imb_sem" in row else None),
                fit_window,
                res,
                plot_path,
            )
        except Exception as exc:  # noqa: BLE001 - report any failure
            failures.append(
                {"L": row["L"], "h": row["h"], "maxdim": row.get("maxdim"), "error": str(exc)}
            )

    if failures:
        failure_df = pd.DataFrame(failures)
        failure_path = output_dir / "fit_failures.csv"
        _ensure_output_dir(failure_path.parent)
        failure_df.to_csv(failure_path, index=False)

    if not results:
        raise RuntimeError("No successful fits were computed.")

    results_df = pd.DataFrame(results).sort_values(["L", "h"]).reset_index(drop=True)

    # Save summary tables
    summary_dir = output_dir / "tables"
    _ensure_output_dir(summary_dir)
    results_df.to_csv(summary_dir / "beta_decay_results.csv", index=False)
    try:
        results_df.to_parquet(summary_dir / "beta_decay_results.parquet", index=False)
    except ImportError:
        # Parquet support is optional; CSV is always written above.
        pass

    # Generate heatmaps for the two beta estimates
    if results_df["beta_unweighted"].notna().any():
        plot_beta_heatmap(
            results_df,
            "beta_unweighted",
            output_path=output_dir / "beta_unweighted_heatmap.png",
            title=r"$\beta$ (unweighted log-log fit)",
        )

    if results_df["beta_weighted"].notna().any():
        # plot_beta_heatmap(
        #     results_df,
        #     "beta_weighted",
        #     output_path=output_dir / "beta_weighted_heatmap.png",
        #     title=r"$\beta$ (weighted log-log fit)",
        # )
        # plot_beta_vs_h(
        #     results_df,
        #     value_column="beta_weighted",
        #     error_column="beta_weighted_stderr",
        #     output_path=output_dir / "beta_vs_h_weighted.png",
        #     title=r"$\beta$ vs $h$ (weighted fits)",
        #     show_fit=True,
        #     fit_thresholds=(0.01, 0.005),
        # )
        plot_beta_vs_h(
            results_df,
            value_column="beta_bootstrap_mean",
            error_column="beta_bootstrap_std",
            output_path=output_dir / "beta_vs_h_bootstrap.png",
            show_fit=True,
            fit_thresholds=(0.01, 0.005),
        )

    if results_df["beta_unweighted"].notna().any():
        # plot_beta_vs_h(
        #     results_df,
        #     value_column="beta_unweighted",
        #     error_column="beta_unweighted_stderr",
        #     output_path=output_dir / "beta_vs_h_unweighted.png",
        #     show_fit=True,
        #     fit_thresholds=(0.01, 0.005),
        # )
        pass

    return results_df


@dataclass
class LinearFitResult:
    """Store linear regression results used for β(h) fits."""

    slope: float
    slope_stderr: float
    intercept: float
    intercept_stderr: float
    r_squared: float
    n_points: int
    h_zero: float
    h_zero_stderr: float


@dataclass
class NonlinearFitResult:
    """Store non-linear fit results for β(h)."""

    params: np.ndarray
    covariance: np.ndarray
    slope_stderr: float
    intercept_stderr: float
    order: int
    r_squared: float
    n_points: int
    thresholds: Dict[float, Tuple[float, float]]

    @property
    def slope(self) -> float:
        return float(self.params[0])

    @property
    def intercept(self) -> float:
        return float(self.params[1])


def linear_fit_beta_vs_h(
    x: np.ndarray,
    y: np.ndarray,
    sigma_y: Optional[np.ndarray] = None,
) -> LinearFitResult:
    """Perform (weighted) linear regression y = m x + b."""
    if sigma_y is not None:
        weights = 1.0 / sigma_y
        coeffs, cov = np.polyfit(x, y, deg=1, w=weights, cov=True)
    else:
        coeffs, cov = np.polyfit(x, y, deg=1, cov=True)

    slope, intercept = coeffs
    slope_stderr = float(np.sqrt(cov[0, 0]))
    intercept_stderr = float(np.sqrt(cov[1, 1]))

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    if np.isclose(slope, 0.0):
        h_zero = np.nan
        h_zero_stderr = np.nan
    else:
        h_zero = -intercept / slope
        grad = np.array([intercept / (slope**2), -1.0 / slope])
        h_zero_var = float(grad @ cov @ grad.T)
        h_zero_stderr = np.sqrt(h_zero_var) if h_zero_var >= 0 else np.nan

    return LinearFitResult(
        slope=slope,
        slope_stderr=slope_stderr,
        intercept=intercept,
        intercept_stderr=intercept_stderr,
        r_squared=r_squared,
        n_points=x.size,
        h_zero=h_zero,
        h_zero_stderr=h_zero_stderr,
    )


def fit_beta_curve_lm(
    h: np.ndarray,
    beta: np.ndarray,
    sigma_beta: Optional[np.ndarray] = None,
    *,
    order: int = 1,
    random_state: Optional[int] = 123,
    bootstrap_samples: int = LM_BOOTSTRAP_SAMPLES_DEFAULT,
    thresholds: Sequence[float] = (0.01, 0.005),
) -> NonlinearFitResult:
    """
    Fit β(h) with a polynomial (order 1 or 2) using Levenberg-Marquardt.

    Returns fitted parameters, covariance, R², and estimated h values where the
    fitted curve reaches the requested β thresholds.
    """
    # The exponential fit is equivalent to a weighted linear regression in log-space.
    mask = np.isfinite(h) & np.isfinite(beta) & (beta > 0)
    if sigma_beta is not None:
        mask &= np.isfinite(sigma_beta) & (sigma_beta > 0)

    h = np.asarray(h, dtype=float)[mask]
    beta = np.asarray(beta, dtype=float)[mask]
    sigma_beta = None if sigma_beta is None else np.asarray(sigma_beta, dtype=float)[mask]

    if h.size < 2:
        raise FitError("Not enough points for exponential β(h) fit.")

    log_beta = np.log(beta)
    if sigma_beta is not None:
        sigma_log = sigma_beta / beta
        valid = np.isfinite(sigma_log) & (sigma_log > 0)
        h = h[valid]
        log_beta = log_beta[valid]
        sigma_log = sigma_log[valid]
        if h.size < 2:
            raise FitError("Not enough valid points after uncertainty filtering.")
        weights = 1.0 / sigma_log
    else:
        sigma_log = None
        weights = None

    coeffs, cov = np.polyfit(h, log_beta, deg=1, w=weights, cov=True)
    slope, intercept = coeffs
    slope_stderr = math.sqrt(cov[0, 0]) if np.isfinite(cov[0, 0]) else math.nan
    intercept_stderr = math.sqrt(cov[1, 1]) if np.isfinite(cov[1, 1]) else math.nan

    log_pred = slope * h + intercept
    ss_res = np.sum((log_beta - log_pred) ** 2)
    ss_tot = np.sum((log_beta - np.mean(log_beta)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else math.nan

    threshold_values = list(dict.fromkeys(thresholds))
    threshold_map: Dict[float, Tuple[float, float]] = {}
    for target in threshold_values:
        if target <= 0 or np.isclose(slope, 0.0):
            threshold_map[target] = (float("nan"), float("nan"))
            continue
        crossing = (math.log(target) - intercept) / slope
        threshold_map[target] = (float(crossing), float("nan"))

    samples = None
    if np.all(np.isfinite(cov)):
        rng = np.random.default_rng(random_state)
        try:
            samples = rng.multivariate_normal(coeffs, cov, size=bootstrap_samples)
        except np.linalg.LinAlgError:
            samples = None

    if samples is not None and len(samples) > 0:
        sample_results = {target: [] for target in threshold_values}
        for sample in samples:
            sample_slope, sample_intercept = sample
            for target in threshold_values:
                if target <= 0 or np.isclose(sample_slope, 0.0):
                    continue
                candidate = (math.log(target) - sample_intercept) / sample_slope
                if np.isfinite(candidate):
                    sample_results[target].append(candidate)

        for target in threshold_values:
            arr = np.array(sample_results[target], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                mean_val = float(np.mean(arr))
                std_val = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
                threshold_map[target] = (mean_val, std_val)

    return NonlinearFitResult(
        params=coeffs,
        covariance=cov,
        slope_stderr=slope_stderr,
        intercept_stderr=intercept_stderr,
        order=1,
        r_squared=r_squared,
        n_points=h.size,
        thresholds=threshold_map,
    )
if __name__ == "__main__":
    main()
