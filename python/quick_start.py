#!/usr/bin/env python3
"""
Quick start guide for working with subset_data.

This shows the recommended workflow:
1. Convert CSV to pickle once (slow, but only done once)
2. Use fast pickle loading for all subsequent work
"""

# ============================================================================
# STEP 1: ONE-TIME CONVERSION (only run this once!)
# ============================================================================
# Uncomment and run this block ONCE to convert the CSV to pickle format:
"""
from load_subset_data import convert_csv_to_fast_format

df = convert_csv_to_fast_format(
    csv_path="../subset_data.csv",
    output_pickle="../subset_data.pkl"
)
"""

# ============================================================================
# STEP 2: FAST LOADING (use this in your daily work!)
# ============================================================================
from load_subset_data import load_from_pickle, filter_by_params

# Load data in ~0.07 seconds instead of ~32 seconds!
df = load_from_pickle("../subset_data.pkl")

print(f"Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

# Example: Get specific parameter combinations
subset = filter_by_params(df, L=10.0, h=15.0)
print(f"\nFiltered to L=10, h=15: {len(subset)} rows")

# Example: Access time series data
row = subset.iloc[0]
print(f"\nFirst row:")
print(f"  Parameters: L={row['L']}, h={row['h']}, grid={row['grid']}")
print(f"  Time array shape: {row['t'].shape}")
print(f"  Time range: [{row['t'].min():.1f}, {row['t'].max():.1f}]")
print(f"  First 5 time points: {row['t'][:5]}")

# Example: Iterate and analyze
print(f"\nExample analysis:")
for idx, row in subset.iloc[:3].iterrows():
    max_imb = row['imb'].max()
    min_imb = row['imb'].min()
    print(f"  grid={row['grid']:3.0f}: imbalance range [{min_imb:.3f}, {max_imb:.3f}]")
