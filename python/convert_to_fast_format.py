#!/usr/bin/env python3
"""
One-time conversion script: Convert subset_data.csv to fast-loading formats.

Run this once, then use the fast pickle/parquet files for all subsequent loads.
"""
import time
from load_subset_data import convert_csv_to_fast_format, load_from_pickle, load_from_parquet

# Configuration
CSV_FILE = "../subset_data.csv"
PICKLE_FILE = "../subset_data.pkl"
PARQUET_FILE = "../subset_data.parquet"

CSV_FILE = "../df12.csv"
PICKLE_FILE = "../df12.pkl"
PARQUET_FILE = "../df12.parquet"

if __name__ == "__main__":
    print("Converting subset_data.csv to fast-loading formats...")
    print(f"Input:  {CSV_FILE}")
    print(f"Output: {PICKLE_FILE} (recommended, fastest)")

    # Check if pyarrow is available for parquet
    try:
        import pyarrow
        use_parquet = True
        print(f"Output: {PARQUET_FILE} (cross-platform compatible)")
    except ImportError:
        use_parquet = False
        print("Note: pyarrow not installed, skipping parquet format")

    print()

    # Time the conversion
    start = time.time()

    df = convert_csv_to_fast_format(
        csv_path=CSV_FILE,
        output_pickle=PICKLE_FILE,
        output_parquet=PARQUET_FILE if use_parquet else None,
        verbose=True
    )
    print(df.columns)

    total_time = time.time() - start
    print(f"\nTotal conversion time: {total_time:.1f} seconds")

    # Test loading speeds
    print("\n" + "="*70)
    print("TESTING LOAD SPEEDS")
    print("="*70)

    # Test pickle loading
    print("\nPickle format:")
    start = time.time()
    df_pkl = load_from_pickle(PICKLE_FILE, verbose=False)
    pkl_time = time.time() - start
    print(f"  Load time: {pkl_time:.3f} seconds")
    print(f"  Shape: {df_pkl.shape}")

    # Test parquet loading if available
    if use_parquet:
        print("\nParquet format:")
        start = time.time()
        df_pq = load_from_parquet(PARQUET_FILE, verbose=False)
        pq_time = time.time() - start
        print(f"  Load time: {pq_time:.3f} seconds")
        print(f"  Shape: {df_pq.shape}")

        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)
        print(f"\nFor fastest loading, use pickle format:")
        print(f"  from load_subset_data import load_from_pickle")
        print(f"  df = load_from_pickle('{PICKLE_FILE}')")
        print(f"\nPickle is {pq_time/pkl_time:.1f}x faster than parquet!")
    else:
        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)
        print(f"\nUse pickle format for fast loading:")
        print(f"  from load_subset_data import load_from_pickle")
        print(f"  df = load_from_pickle('{PICKLE_FILE}')")
