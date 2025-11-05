#!/usr/bin/env python3
"""
Example: Plotting averaged data with error bands.

Shows how to work with disorder-averaged imbalance data.
"""
import numpy as np
import matplotlib.pyplot as plt
from process_data import load_averaged_data, get_averaged_row


# Load averaged data (fast!)
df_avg = load_averaged_data("../subset_data_averaged.pkl", verbose=True)
print()

# Show available parameters
print("Available parameters:")
print(f"  L values: {sorted(df_avg['L'].unique())}")
print(f"  h values: {sorted(df_avg['h'].unique())}")
print()

# Example 1: Plot for different L values at fixed h
print("="*70)
print("EXAMPLE 1: Imbalance vs time for different L (fixed h=15)")
print("="*70)

fig, ax = plt.subplots(figsize=(10, 6))

h_fixed = 15.0
L_values = sorted(df_avg['L'].unique())

for L in L_values:
    row = get_averaged_row(df_avg, L=L, h=h_fixed)

    t = row['t']
    imb_mean = row['imb_mean']
    imb_sem = row['imb_sem']

    # Plot mean with shaded error band (±1 SEM)
    ax.plot(t, imb_mean, label=f'L={int(L)}', linewidth=2)
    ax.fill_between(t, imb_mean - imb_sem, imb_mean + imb_sem, alpha=0.3)

ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Imbalance', fontsize=12)
ax.set_title(f'Disorder-averaged imbalance (h={h_fixed})', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('averaged_imbalance_vs_L.png', dpi=150, bbox_inches='tight')
print(f"Saved: averaged_imbalance_vs_L.png")
print()

# Example 2: Plot for different h values at fixed L
print("="*70)
print("EXAMPLE 2: Imbalance vs time for different h (fixed L=10)")
print("="*70)

fig, ax = plt.subplots(figsize=(10, 6))

L_fixed = 10.0
h_values = sorted(df_avg['h'].unique())

for h in h_values:
    row = get_averaged_row(df_avg, L=L_fixed, h=h)

    t = row['t']
    imb_mean = row['imb_mean']
    imb_sem = row['imb_sem']

    # Plot mean with shaded error band
    ax.plot(t, imb_mean, label=f'h={int(h)}', linewidth=2, alpha=0.8)
    ax.fill_between(t, imb_mean - imb_sem, imb_mean + imb_sem, alpha=0.2)

ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Imbalance', fontsize=12)
ax.set_title(f'Disorder-averaged imbalance (L={int(L_fixed)})', fontsize=14)
ax.legend(ncol=2, fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('averaged_imbalance_vs_h.png', dpi=150, bbox_inches='tight')
print(f"Saved: averaged_imbalance_vs_h.png")
print()

# Example 3: Compare standard deviation at different times
print("="*70)
print("EXAMPLE 3: Standard deviation evolution")
print("="*70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 3a: imbalance and its std over time
row = get_averaged_row(df_avg, L=10.0, h=15.0)
t = row['t']
imb_mean = row['imb_mean']
imb_std = row['imb_std']
imb_sem = row['imb_sem']

ax1.plot(t, imb_mean, 'b-', linewidth=2, label='Mean')
ax1.fill_between(t, imb_mean - imb_std, imb_mean + imb_std,
                 alpha=0.3, color='b', label='±1 std')
ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('Imbalance', fontsize=12)
ax1.set_title(f'Mean ± std (L=10, h=15, n={row["num_realizations"]})', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 3b: Evolution of std itself
ax2.plot(t, imb_std, 'r-', linewidth=2, label='Std')
ax2.plot(t, imb_sem, 'g-', linewidth=2, label='SEM')
ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('Variation', fontsize=12)
ax2.set_title('Standard deviation and SEM over time', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('averaged_imbalance_std_evolution.png', dpi=150, bbox_inches='tight')
print(f"Saved: averaged_imbalance_std_evolution.png")
print()

# Example 4: Summary statistics
print("="*70)
print("EXAMPLE 4: Summary statistics at t=50")
print("="*70)

print(f"\n{'L':>4}  {'h':>4}  {'imb_mean':>10}  {'imb_std':>10}  {'imb_sem':>10}  {'n_real':>6}")
print("-" * 70)

for L in sorted(df_avg['L'].unique())[:3]:  # Show first 3 L values
    for h in sorted(df_avg['h'].unique())[:5]:  # Show first 5 h values
        row = get_averaged_row(df_avg, L=L, h=h)
        idx_50 = 500  # t=50 is at index 500

        imb_mean = row['imb_mean'][idx_50]
        imb_std = row['imb_std'][idx_50]
        imb_sem = row['imb_sem'][idx_50]
        n_real = row['num_realizations']

        print(f"{L:>4.0f}  {h:>4.0f}  {imb_mean:>10.6f}  {imb_std:>10.6f}  {imb_sem:>10.6f}  {n_real:>6}")

print("\nNote: imb_sem = imb_std / sqrt(n_real)")
