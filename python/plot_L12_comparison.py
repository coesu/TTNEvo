#!/usr/bin/env python3
"""
Replicate the Julia `plot_L12_comparison` figure using Python/matplotlib.

The script expects the pickle file `df12.pkl` (default location: project root).
It reproduces the two-column panel layout (imbalance vs. time on the left,
error metrics on the right, colour-coded by execution time) for each requested
field value `h`, plus bottom-row legends for bond dimensions and TTN/MPS styles.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator

# Colour palette matching the Zissou1Continuous sample picked in the Julia code.
PALETTE = ['#3A9AB2', '#A5C2A3', '#DCCB4E', '#E79805', '#F11B00']

COLORBAR_MIN = 20
COLORBAR_MAX = 400


@dataclass(frozen=True)
class PlotInputs:
    dataframe: pd.DataFrame
    h_values: Sequence[float]
    grid_number: int
    output_path: Path
    output_dpi: int = 200


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load the pickle file and normalise the column names we rely on."""
    df = pd.read_pickle(path)

    rename_map = {
        "t": "times",
        "times": "times",
        "imb": "imbalance",
        "imbalance": "imbalance",
        "t_ex": "ex_times",
        "ex_times": "ex_times",
        "h": "model_h",
        "model_h": "model_h",
        "L": "graph_L",
        "grid": "graph_gridnum",
        "graph_gridnum": "graph_gridnum",
        "maxdim": "initial_state_initial_maxdim",
        "initial_state_initial_maxdim": "initial_state_initial_maxdim",
        "method": "graph_type",
        "graph": "graph_type",
    }

    resolved: Dict[str, str] = {}
    for original, canonical in rename_map.items():
        if original in df.columns:
            resolved[original] = canonical

    if resolved:
        df = df.rename(columns=resolved)

    if "graph_type" in df.columns:
        type_values = set(df["graph_type"].astype(str).unique())
        alias_map = {
            "TTN": "FreeGraph",
            "MPS": "SnakeGraph",
        }
        if type_values.issubset(set(alias_map) | {"FreeGraph", "SnakeGraph"}):
            df = df.assign(
                graph_type=df["graph_type"].map(alias_map).fillna(df["graph_type"])
            )

    required = {
        "times",
        "imbalance",
        "ex_times",
        "num_size",
        "graph_L",
        "model_h",
        "graph_gridnum",
        "initial_state_initial_maxdim",
        "graph_type",
    }
    print(df.columns)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataframe is missing required column(s): {', '.join(missing)}. "
            "Please ensure df12.pkl was generated with the full TTN/MPS metadata."
        )

    return df


def extract_series(row: pd.Series, column: str) -> np.ndarray:
    """Ensure array-like columns become 1D numpy arrays."""
    values = row[column]
    if isinstance(values, np.ndarray):
        return values.astype(float)
    return np.asarray(values, dtype=float)


def compute_errors(
    ref_row: pd.Series, compare_row: pd.Series, column: str
) -> float:
    """Mean absolute error between the reference and comparison series."""
    ref = extract_series(ref_row, column)
    cmp = extract_series(compare_row, column)
    min_len = min(ref.size, cmp.size)
    if min_len == 0:
        return float("nan")
    return float(np.mean(np.abs(ref[:min_len] - cmp[:min_len])))


def prepare_axes(
    fig: plt.Figure,
    n_rows: int,
    *,
    row_height: float = 0.9,
    width_split: Tuple[float, float, float] = (0.6, 0.38, 0.08),
    hspace: float = 0.18,
    wspace: float = 0.28,
) -> Tuple[List[plt.Axes], List[plt.Axes], List[plt.Axes], GridSpec]:
    """Create the grid of axes with 60/40 column split plus colour bar column."""
    height_ratios = [row_height] * n_rows
    width_ratios = list(width_split)
    gs = GridSpec(
        nrows=n_rows,
        ncols=3,
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=hspace,
        wspace=wspace,
    )

    axes_time: List[plt.Axes] = []
    axes_error: List[plt.Axes] = []
    axes_cbar: List[plt.Axes] = []

    for idx in range(n_rows):
        shared_time = axes_time[0] if axes_time else None
        shared_error = axes_error[0] if axes_error else None
        axes_time.append(
            fig.add_subplot(gs[idx, 0], sharex=shared_time) if shared_time else fig.add_subplot(gs[idx, 0])
        )
        axes_error.append(
            fig.add_subplot(gs[idx, 1], sharex=shared_error) if shared_error else fig.add_subplot(gs[idx, 1])
        )
        axes_cbar.append(fig.add_subplot(gs[idx, 2]))

    return axes_time, axes_error, axes_cbar, gs


def format_time_axis(ax: plt.Axes, show_xlabel: bool) -> None:
    """Apply consistent styling to the time-series axis."""
    ax.set_ylabel(r"$I(t)$")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.35)
    if show_xlabel:
        ax.set_xlabel(r"$t$")
    else:
        ax.tick_params(labelbottom=False)
    ax.tick_params(direction="in", top=True, right=True, length=3.5, width=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def format_error_axis(ax: plt.Axes, show_xlabel: bool) -> None:
    """Apply consistent styling to the error axis."""
    ax.set_yscale("log")
    ax.set_ylabel(r"$\langle |I_{\chi_{\max}} - I_{\chi}| \rangle$", labelpad=12)
    ax.grid(True, axis="both", linestyle=":", linewidth=0.6, alpha=0.3, which="both")
    if show_xlabel:
        ax.set_xlabel(r"$N_{\mathrm{par}}$")
    else:
        ax.tick_params(labelbottom=False)

    ax.yaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10) * 0.1))
    ax.tick_params(direction="in", top=True, right=True, length=3.5, width=0.7, pad=4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def scientific_formatter(power_offset: Tuple[int, int]) -> FuncFormatter:
    """Return a formatter that writes ticks as 10^{k} within provided bounds."""
    min_exp, max_exp = power_offset

    def _format(value: float, _: int) -> str:
        if value <= 0:
            return ""
        exponent = int(np.round(np.log10(value)))
        if exponent < min_exp or exponent > max_exp:
            return ""
        return rf"$10^{{{exponent}}}$"

    return FuncFormatter(_format)


def collect_dim_palette(df: pd.DataFrame) -> Dict[int, str]:
    """Assign colours to each bond dimension based on global ordering."""
    dims = sorted({int(dim) for dim in df["initial_state_initial_maxdim"].unique()})
    palette = (PALETTE * ((len(dims) // len(PALETTE)) + 1))[: len(dims)]
    return {dim: colour for dim, colour in zip(dims, palette)}


def plot_row(
    ax_time: plt.Axes,
    ax_error: plt.Axes,
    cax: plt.Axes,
    df_mps: pd.DataFrame,
    df_ttn: pd.DataFrame,
    h_value: float,
    row_index: int,
    dim_colors: Dict[int, str],
    show_marker_legend: bool = False,
) -> None:
    """Render one row of plots for a specific h-value."""
    ax_time.text(
        0.02,
        0.92,
        f"({chr(ord('a') + 2 * row_index)})",
        transform=ax_time.transAxes,
        fontsize=9,
        ha="left",
        va="top",
    )
    ax_time.text(
        0.98,
        0.92,
        rf"$h = {h_value:g}$",
        transform=ax_time.transAxes,
        fontsize=9,
        ha="right",
        va="top",
    )

    # Ensure rows are aligned by bond dimension.
    merged = pd.merge(
        df_mps,
        df_ttn,
        on=[
            "graph_gridnum",
            "model_h",
            "initial_state_initial_maxdim",
        ],
        suffixes=("_mps", "_ttn"),
    ).sort_values("initial_state_initial_maxdim")

    if merged.empty:
        raise ValueError(f"No overlapping TTN/MPS data for h={h_value}.")

    ref_mps = merged.iloc[-1]
    ref_ttn = merged.iloc[-1]

    errors_mps: List[float] = []
    errors_ttn: List[float] = []
    mean_sizes_mps: List[float] = []
    mean_sizes_ttn: List[float] = []
    runtimes_mps: List[float] = []
    runtimes_ttn: List[float] = []

    for _, row in merged.iterrows():
        dim = int(row["initial_state_initial_maxdim"])
        colour = dim_colors.get(dim, "#555555")

        times_ttn = extract_series(row, "times_ttn")
        imbalance_ttn = extract_series(row, "imbalance_ttn")
        ax_time.plot(times_ttn, imbalance_ttn, color=colour, linewidth=1.2)

        times_mps = extract_series(row, "times_mps")
        imbalance_mps = extract_series(row, "imbalance_mps")
        ax_time.plot(
            times_mps,
            imbalance_mps,
            color=colour,
            linewidth=1.2,
            linestyle="--",
        )

        if dim != int(ref_mps["initial_state_initial_maxdim"]):
            err_mps = compute_errors(ref_mps, row, "imbalance_mps")
            err_ttn = compute_errors(ref_ttn, row, "imbalance_ttn")
            if not np.isnan(err_mps):
                errors_mps.append(err_mps)
                mean_sizes_mps.append(
                    float(np.mean(extract_series(row, "num_size_mps")))
                )
                runtimes_mps.append(
                    float(np.mean(extract_series(row, "ex_times_mps")))
                )
            if not np.isnan(err_ttn):
                errors_ttn.append(err_ttn)
                mean_sizes_ttn.append(
                    float(np.mean(extract_series(row, "num_size_ttn")))
                )
                runtimes_ttn.append(
                    float(np.mean(extract_series(row, "ex_times_ttn")))
                )

    ax_error.text(
        0.02,
        0.92,
        f"({chr(ord('a') + 2 * row_index + 1)})",
        transform=ax_error.transAxes,
        fontsize=9,
        ha="left",
        va="top",
    )

    if errors_mps:
        errors_mps_arr = np.asarray(errors_mps, dtype=float)
        mask_mps = ~np.isnan(errors_mps_arr)
        ax_error.scatter(
            (np.asarray(mean_sizes_mps) / 1e5)[mask_mps],
            errors_mps_arr[mask_mps],
            c=np.round(np.asarray(runtimes_mps)[mask_mps]),
            cmap="viridis",
            vmin=COLORBAR_MIN,
            vmax=COLORBAR_MAX,
            marker="s",
            s=45,
            edgecolor="black",
            linewidth=0.7,
        )
    else:
        errors_mps_arr = np.empty(0, dtype=float)
        mask_mps = np.array([], dtype=bool)

    errors_ttn_arr = np.asarray(errors_ttn, dtype=float)
    mask_ttn = ~np.isnan(errors_ttn_arr)
    scatter_ttn = ax_error.scatter(
        (np.asarray(mean_sizes_ttn) / 1e5)[mask_ttn],
        errors_ttn_arr[mask_ttn],
        c=np.round(np.asarray(runtimes_ttn)[mask_ttn]),
        cmap="viridis",
        vmin=COLORBAR_MIN,
        vmax=COLORBAR_MAX,
        marker="o",
        s=45,
        edgecolor="black",
        linewidth=0.7,
    )

    if errors_mps_arr.size:
        positive_errors = np.concatenate(
            [errors_mps_arr[mask_mps], errors_ttn_arr[mask_ttn]]
        )
    else:
        positive_errors = errors_ttn_arr[mask_ttn]
    positive_errors = positive_errors[positive_errors > 0]
    if positive_errors.size:
        if row_index == 0:
            min_exp, max_exp = -3, -1
        else:
            min_exp, max_exp = -4, -2
        ax_error.set_ylim(10 ** min_exp, 1.5 * 10 ** max_exp)
        ax_error.yaxis.set_major_formatter(scientific_formatter((min_exp, max_exp)))

    cbar = plt.colorbar(
        scatter_ttn,
        cax=cax,
        ticks=[COLORBAR_MIN, (COLORBAR_MIN + COLORBAR_MAX) / 2, COLORBAR_MAX],
    )
    cbar.ax.set_ylabel(r"$t_{\mathrm{ex}}$")

    if show_marker_legend:
        legend_handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor="#666666",
                markeredgecolor="black",
                markersize=6,
                label="TTN",
            ),
            Line2D(
                [],
                [],
                marker="s",
                linestyle="",
                markerfacecolor="#999999",
                markeredgecolor="black",
                markersize=6,
                label="MPS",
            ),
        ]
        ax_error.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=False,
            fontsize=8,
            borderpad=0.2,
            handletextpad=0.4,
            labelspacing=0.3,
        )


def add_legends(
    fig: plt.Figure,
    axes_time_leg: plt.Axes,
    axes_method_leg: plt.Axes,
    dim_colors: Dict[int, str],
) -> None:
    """Place legends directly in/around the plotted axes."""
    _ = axes_method_leg  # intentionally unused; kept for API symmetry
    dim_handles = [
        Line2D([], [], color=color, linewidth=2)
        for _, color in sorted(dim_colors.items())
    ]
    dim_labels = [rf"$\chi = {int(dim)}$" for dim in sorted(dim_colors)]
    if dim_handles:
        fig.legend(
            dim_handles,
            dim_labels,
            loc="upper center",
            bbox_to_anchor=(0.48, 0.995),
            frameon=False,
            ncol=min(4, len(dim_handles)),
            fontsize=8,
            handlelength=2.2,
            handletextpad=0.4,
            borderpad=0.2,
            labelspacing=0.3,
            columnspacing=0.8,
        )

    method_handles = [
        Line2D([], [], color="black", linewidth=2, linestyle="-"),
        Line2D([], [], color="black", linewidth=2, linestyle="--"),
    ]
    method_legend = axes_time_leg.legend(
        method_handles,
        ["TTN", "MPS"],
        loc="upper left",
        bbox_to_anchor=(0.01, 1.02),
        frameon=False,
        fontsize=8,
        handlelength=2.4,
        handletextpad=0.5,
        borderpad=0.2,
        labelspacing=0.3,
    )
    axes_time_leg.add_artist(method_legend)


def build_plot(inputs: PlotInputs) -> Path:
    df = inputs.dataframe
    df = df[df["graph_gridnum"] == inputs.grid_number]
    if df.empty:
        raise ValueError(f"No rows found for grid={inputs.grid_number}.")

    dim_colors = collect_dim_palette(df)
    fig_height = 1.7 * len(inputs.h_values) + 0.8
    fig = plt.figure(figsize=(6.0, fig_height), dpi=inputs.output_dpi)
    axes_time, axes_error, axes_cbar, gridspec_obj = prepare_axes(
        fig,
        len(inputs.h_values),
        row_height=0.95,
        width_split=(0.55, 0.34, 0.11),
        hspace=0.14,
        wspace=0.68,
    )

    for idx, (ax_time, ax_error, cax, h_value) in enumerate(
        zip(axes_time, axes_error, axes_cbar, inputs.h_values)
    ):
        df_h_mps = df[
            (df["graph_type"] == "SnakeGraph")
            & (df["model_h"] == h_value)
        ].sort_values("initial_state_initial_maxdim")
        df_h_ttn = df[
            (df["graph_type"] == "FreeGraph")
            & (df["model_h"] == h_value)
        ].sort_values("initial_state_initial_maxdim")

        if df_h_mps.empty or df_h_ttn.empty:
            raise ValueError(
                f"Missing TTN/MPS data for h={h_value}. "
                "Ensure df12.pkl contains both graph types."
            )

        format_time_axis(ax_time, show_xlabel=(idx == len(inputs.h_values) - 1))
        format_error_axis(ax_error, show_xlabel=(idx == len(inputs.h_values) - 1))

        plot_row(
            ax_time,
            ax_error,
            cax,
            df_h_mps,
            df_h_ttn,
            h_value,
            idx,
            dim_colors,
            show_marker_legend=(idx == 0),
        )

    add_legends(fig, axes_time[0], axes_error[0], dim_colors)

    fig.align_ylabels(axes_time)
    fig.align_labels()
    fig.subplots_adjust(left=0.12, right=0.97, top=0.985, bottom=0.08, wspace=0.6)
    inputs.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(inputs.output_path, bbox_inches="tight")
    plt.close(fig)
    return inputs.output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate the Julia plot_L12_comparison figure using matplotlib.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("../df12.pkl"),
        help="Path to the pickle file containing the TTN/MPS comparison data.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=2,
        help="Grid number (graph_gridnum) to filter on.",
    )
    parser.add_argument(
        "--h-values",
        nargs="+",
        type=float,
        help="Explicit list of disorder strengths h to plot. Defaults to all available.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/paper/L12_plot_python.pdf"),
        help="Output path for the generated figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataframe(args.data)

    if args.h_values:
        h_values = args.h_values
    else:
        h_values = sorted(
            df.loc[df["graph_gridnum"] == args.grid, "model_h"].unique()
        )

    if not h_values:
        raise ValueError(
            f"No h-values available for grid={args.grid}. "
            "Provide them explicitly via --h-values."
        )

    inputs = PlotInputs(
        dataframe=df,
        h_values=h_values,
        grid_number=args.grid,
        output_path=args.output,
        output_dpi=args.dpi,
    )
    output_file = build_plot(inputs)
    print(f"Wrote figure to {output_file}")


if __name__ == "__main__":
    main()
