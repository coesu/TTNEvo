#!/usr/bin/env python3
"""
Generate comprehensive plots for all parameter combinations.

Creates:
1. Individual plots for each (L, h) combination
2. Multi-panel plots grouped by L
3. Multi-panel plots grouped by h
4. Summary heatmaps
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from process_data import load_averaged_data, get_averaged_row


def create_output_dirs():
    """Create directories for organized output."""
    dirs = {
        'individual': Path('plots/individual'),
        'by_L': Path('plots/by_L'),
        'by_h': Path('plots/by_h'),
        'summary': Path('plots/summary')
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def plot_individual(df_avg, L, h, output_dir):
    """
    Create individual plot for a single (L, h) combination.

    Shows imbalance with error band and std evolution.
    """
    row = get_averaged_row(df_avg, L=L, h=h)

    t = row['t']
    imb_mean = row['imb_mean']
    imb_std = row['imb_std']
    imb_sem = row['imb_sem']
    n = row['num_realizations']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Imbalance with error band
    ax1.plot(t, imb_mean, 'b-', linewidth=2, label='Mean')
    ax1.fill_between(t, imb_mean - imb_sem, imb_mean + imb_sem,
                     alpha=0.3, color='b', label='±1 SEM')
    ax1.fill_between(t, imb_mean - imb_std, imb_mean + imb_std,
                     alpha=0.15, color='r', label='±1 std')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Imbalance', fontsize=12)
    ax1.set_title(f'Disorder-averaged imbalance (L={int(L)}, h={int(h)}, n={n})', fontsize=13)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # Right: Standard deviation and SEM evolution
    ax2.plot(t, imb_std, 'r-', linewidth=2, label='Std (disorder variation)')
    ax2.plot(t, imb_sem, 'g-', linewidth=2, label='SEM (uncertainty)')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Variation', fontsize=12)
    ax2.set_title('Evolution of uncertainty', fontsize=13)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = output_dir / f'L{int(L):02d}_h{int(h):02d}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    return filename


def plot_by_L(df_avg, L, output_dir):
    """
    Create multi-panel plot showing all h values for fixed L.
    """
    h_values = sorted(df_avg['h'].unique())
    n_h = len(h_values)

    # Create grid layout
    n_cols = 5
    n_rows = int(np.ceil(n_h / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
    axes = axes.flatten() if n_h > 1 else [axes]

    fig.suptitle(f'Imbalance evolution for L={int(L)} (all h values)', fontsize=16, y=0.995)

    for idx, h in enumerate(h_values):
        ax = axes[idx]
        row = get_averaged_row(df_avg, L=L, h=h)

        t = row['t']
        imb_mean = row['imb_mean']
        imb_sem = row['imb_sem']
        n = row['num_realizations']

        ax.plot(t, imb_mean, 'b-', linewidth=2)
        ax.fill_between(t, imb_mean - imb_sem, imb_mean + imb_sem,
                       alpha=0.3, color='b')

        ax.set_title(f'h={int(h)} (n={n})', fontsize=11)
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Imbalance', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    # Hide unused subplots
    for idx in range(n_h, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    filename = output_dir / f'L{int(L):02d}_all_h.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    return filename


def plot_by_h(df_avg, h, output_dir):
    """
    Create plot showing all L values for fixed h.
    """
    L_values = sorted(df_avg['L'].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_values)))

    for idx, L in enumerate(L_values):
        row = get_averaged_row(df_avg, L=L, h=h)

        t = row['t']
        imb_mean = row['imb_mean']
        imb_sem = row['imb_sem']
        n = row['num_realizations']

        ax.plot(t, imb_mean, linewidth=2.5, color=colors[idx],
               label=f'L={int(L)} (n={n})')
        ax.fill_between(t, imb_mean - imb_sem, imb_mean + imb_sem,
                       alpha=0.3, color=colors[idx])

    ax.set_xlabel('Time', fontsize=13)
    ax.set_ylabel('Imbalance', fontsize=13)
    ax.set_title(f'System size dependence (h={int(h)})', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    filename = output_dir / f'h{int(h):02d}_all_L.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    return filename


def plot_heatmaps(df_avg, output_dir):
    """
    Create heatmaps showing imbalance at different times as function of L and h.
    """
    L_values = sorted(df_avg['L'].unique())
    h_values = sorted(df_avg['h'].unique())

    # Time points to show
    time_points = [0, 10, 25, 50, 75, 100]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    fig.suptitle('Imbalance as function of L and h at different times', fontsize=16)

    for idx, t_target in enumerate(time_points):
        ax = axes[idx]

        # Find closest time index
        t_idx = int(t_target * 10)  # Since dt=0.1

        # Build matrix
        matrix = np.zeros((len(h_values), len(L_values)))

        for i, h in enumerate(h_values):
            for j, L in enumerate(L_values):
                row = get_averaged_row(df_avg, L=L, h=h)
                matrix[i, j] = row['imb_mean'][t_idx]

        # Plot heatmap
        im = ax.imshow(matrix, aspect='auto', origin='lower', cmap='viridis',
                      vmin=0, vmax=1, interpolation='nearest')

        ax.set_xticks(range(len(L_values)))
        ax.set_xticklabels([int(L) for L in L_values])
        ax.set_yticks(range(len(h_values)))
        ax.set_yticklabels([int(h) for h in h_values])

        ax.set_xlabel('L (system size)', fontsize=11)
        ax.set_ylabel('h (field strength)', fontsize=11)
        ax.set_title(f't = {t_target}', fontsize=12)

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Imbalance')

        # Add text annotations
        for i in range(len(h_values)):
            for j in range(len(L_values)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="white", fontsize=8,
                             weight='bold')

    plt.tight_layout()

    filename = output_dir / 'heatmaps_imbalance_vs_L_h.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    return filename


def main():
    """Generate all plots."""
    print("="*70)
    print("GENERATING COMPREHENSIVE PLOTS FOR ALL PARAMETER COMBINATIONS")
    print("="*70)
    print()

    # Create output directories
    print("Creating output directories...")
    dirs = create_output_dirs()
    print(f"  Output will be saved to: {Path('plots').absolute()}")
    print()

    # Load data
    print("Loading averaged data...")
    df_avg = load_averaged_data("../subset_data_averaged.pkl", verbose=False)
    L_values = sorted(df_avg['L'].unique())
    h_values = sorted(df_avg['h'].unique())
    print(f"  L values: {[int(L) for L in L_values]}")
    print(f"  h values: {[int(h) for h in h_values]}")
    print(f"  Total combinations: {len(df_avg)}")
    print()

    # 1. Individual plots for each (L, h)
    print("="*70)
    print("1. GENERATING INDIVIDUAL PLOTS")
    print("="*70)
    total = len(df_avg)
    for idx, (_, row) in enumerate(df_avg.iterrows(), 1):
        L = row['L']
        h = row['h']
        filename = plot_individual(df_avg, L, h, dirs['individual'])
        if idx % 10 == 0 or idx == total:
            print(f"  Progress: {idx}/{total} plots created")
    print(f"✓ Saved {total} individual plots to: {dirs['individual']}")
    print()

    # 2. Multi-panel plots by L
    print("="*70)
    print("2. GENERATING MULTI-PANEL PLOTS BY L")
    print("="*70)
    for L in L_values:
        filename = plot_by_L(df_avg, L, dirs['by_L'])
        print(f"  ✓ Created: {filename.name}")
    print(f"✓ Saved {len(L_values)} plots to: {dirs['by_L']}")
    print()

    # 3. Comparison plots by h
    print("="*70)
    print("3. GENERATING COMPARISON PLOTS BY H")
    print("="*70)
    for h in h_values:
        filename = plot_by_h(df_avg, h, dirs['by_h'])
        print(f"  ✓ Created: {filename.name}")
    print(f"✓ Saved {len(h_values)} plots to: {dirs['by_h']}")
    print()

    # 4. Heatmaps
    print("="*70)
    print("4. GENERATING SUMMARY HEATMAPS")
    print("="*70)
    filename = plot_heatmaps(df_avg, dirs['summary'])
    print(f"  ✓ Created: {filename.name}")
    print(f"✓ Saved heatmap to: {dirs['summary']}")
    print()

    # Summary
    print("="*70)
    print("COMPLETE!")
    print("="*70)
    total_plots = len(df_avg) + len(L_values) + len(h_values) + 1
    print(f"\nGenerated {total_plots} plots:")
    print(f"  - {len(df_avg)} individual plots (plots/individual/)")
    print(f"  - {len(L_values)} multi-panel plots by L (plots/by_L/)")
    print(f"  - {len(h_values)} comparison plots by h (plots/by_h/)")
    print(f"  - 1 summary heatmap (plots/summary/)")
    print()


if __name__ == "__main__":
    main()
