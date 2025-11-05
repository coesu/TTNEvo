"""
Example showing how to use the load_subset_data module.
"""
import matplotlib.pyplot as plt
from load_subset_data import load_subset_data, filter_by_params, get_unique_params

# Load the data
print("Loading data...")
df = load_subset_data(
    "/home/lars/Syncthing/master-thesis/TTNEvo/subset_data.csv",
    convert_arrays_to_numpy=True,  # Convert to numpy arrays for easier manipulation
    verbose=False  # Set to True to see progress
)

print(f"Loaded {len(df)} rows\n")

# Show what parameter values are available
print("Available parameters:")
params = get_unique_params(df)
for name, values in params.items():
    print(f"  {name}: {len(values)} unique values, range [{min(values):.1f}, {max(values):.1f}]")

print("\n" + "="*70)

# Example 1: Get a specific parameter combination
print("\nExample 1: Get data for L=10, h=20")
print("-" * 70)
subset = filter_by_params(df, L=10.0, h=20.0)
print(f"Found {len(subset)} rows")
print(f"Grid values in this subset: {sorted(subset['grid'].unique())}")

# Example 2: Access array data from a specific row
print("\n" + "="*70)
print("\nExample 2: Access time series data")
print("-" * 70)
row = subset.iloc[0]  # Get first row
print(f"Parameters: L={row['L']}, h={row['h']}, grid={row['grid']}, maxdim={row['maxdim']}")
print(f"Array shapes: t={row['t'].shape}, t_ex={row['t_ex'].shape}, imb={row['imb'].shape}")
print(f"Time range: [{row['t'].min():.1f}, {row['t'].max():.1f}]")
print(f"t_ex range: [{row['t_ex'].min():.2f}, {row['t_ex'].max():.2f}]")
print(f"imb range: [{row['imb'].min():.3f}, {row['imb'].max():.3f}]")

# Example 3: Plot data (optional - comment out if you don't want plots)
print("\n" + "="*70)
print("\nExample 3: Plotting data")
print("-" * 70)

# Get a few rows with different grid values
rows_to_plot = filter_by_params(df, L=10.0, h=15.0)
# Take a subset for visualization
rows_to_plot = rows_to_plot.iloc[:5]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, row in rows_to_plot.iterrows():
    label = f"grid={int(row['grid'])}"
    axes[0].plot(row['t'], row['t_ex'], label=label, alpha=0.7)
    axes[1].plot(row['t'], row['imb'], label=label, alpha=0.7)

axes[0].set_xlabel('Time (t)')
axes[0].set_ylabel('t_ex')
axes[0].set_title(f'Evolution of t_ex (L={rows_to_plot.iloc[0]["L"]}, h={rows_to_plot.iloc[0]["h"]})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Time (t)')
axes[1].set_ylabel('Imbalance')
axes[1].set_title(f'Evolution of imbalance (L={rows_to_plot.iloc[0]["L"]}, h={rows_to_plot.iloc[0]["h"]})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('subset_data_example.png', dpi=150, bbox_inches='tight')
print("Saved plot to: subset_data_example.png")

# Example 4: Iterate over all L and h combinations
print("\n" + "="*70)
print("\nExample 4: Count data points per parameter combination")
print("-" * 70)
L_values = sorted(df['L'].unique())[:3]  # Just show first 3
h_values = sorted(df['h'].unique())[:3]

print(f"\n{'L':>4}  {'h':>4}  {'Count':>6}")
print("-" * 20)
for L in L_values:
    for h in h_values:
        count = len(filter_by_params(df, L=L, h=h))
        print(f"{L:>4.0f}  {h:>4.0f}  {count:>6}")
