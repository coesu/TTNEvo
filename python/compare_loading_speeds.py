#!/usr/bin/env python3
"""
Compare loading speeds between CSV and pickle formats.
"""
import time
from load_subset_data import load_subset_data, load_from_pickle

CSV_FILE = "../subset_data.csv"
PICKLE_FILE = "../subset_data.pkl"

print("="*70)
print("LOADING SPEED COMPARISON")
print("="*70)

# Test CSV loading (slow, original method)
print("\nMethod 1: Loading from CSV (original method)")
print("-"*70)
start = time.time()
df_csv = load_subset_data(CSV_FILE, convert_arrays_to_numpy=True, verbose=False)
csv_time = time.time() - start
print(f"Time: {csv_time:.2f} seconds")
print(f"Shape: {df_csv.shape}")

# Test pickle loading (fast, new method)
print("\nMethod 2: Loading from Pickle (fast method)")
print("-"*70)
start = time.time()
df_pkl = load_from_pickle(PICKLE_FILE, verbose=False)
pkl_time = time.time() - start
print(f"Time: {pkl_time:.3f} seconds")
print(f"Shape: {df_pkl.shape}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"CSV loading:    {csv_time:.2f} seconds")
print(f"Pickle loading: {pkl_time:.3f} seconds")
print(f"\nSpeedup: {csv_time/pkl_time:.0f}x faster! 🚀")
print(f"\nYou save {csv_time - pkl_time:.1f} seconds every time you load the data!")
