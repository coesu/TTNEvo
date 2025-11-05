"""
Data processing functions for subset_data.

Main functionality: Average imbalances over different disorder configurations (grid numbers).
"""
import numpy as np
import pandas as pd
from typing import Optional, List
from load_subset_data import load_from_pickle


def average_over_disorder(df: pd.DataFrame,
                          compute_std: bool = True,
                          compute_sem: bool = True,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Average imbalance and t_ex over different disorder configurations (grid numbers).

    For each unique combination of (L, h, maxdim), this function averages the time series
    arrays (imb, t_ex) over all grid realizations (disorder configurations).

    Args:
        df: Input DataFrame from load_from_pickle()
        compute_std: If True, also compute standard deviation
        compute_sem: If True, also compute standard error of the mean (SEM)
        verbose: If True, print progress information

    Returns:
        DataFrame with averaged data. Columns:
            - t: Time array (same for all, not averaged)
            - t_ex_mean: Mean of t_ex over disorder configurations
            - imb_mean: Mean of imbalance over disorder configurations
            - t_ex_std: Standard deviation of t_ex (if compute_std=True)
            - imb_std: Standard deviation of imbalance (if compute_std=True)
            - t_ex_sem: Standard error of t_ex (if compute_sem=True)
            - imb_sem: Standard error of imbalance (if compute_sem=True)
            - L, h, maxdim: Parameter values
            - num_realizations: Number of grid values averaged over
    """
    if verbose:
        print("Averaging over disorder configurations...")
        print(f"Input: {len(df)} rows")

    # Group by all parameters except grid
    grouped = df.groupby(['L', 'h', 'maxdim'])

    results = []

    for (L, h, maxdim), group in grouped:
        if verbose and len(results) % 10 == 0:
            print(f"  Processing L={L}, h={h}... ({len(results)+1}/{len(grouped)})")

        # Stack arrays for vectorized operations
        t_arrays = np.stack(group['t'].values)
        t_ex_arrays = np.stack(group['t_ex'].values)
        imb_arrays = np.stack(group['imb'].values)

        num_realizations = len(group)

        # Verify t arrays are all the same
        if not np.allclose(t_arrays, t_arrays[0]):
            print(f"Warning: t arrays not identical for L={L}, h={h}")

        # Compute means
        t_mean = t_arrays[0]  # They should all be the same
        t_ex_mean = np.mean(t_ex_arrays, axis=0)
        imb_mean = np.mean(imb_arrays, axis=0)

        # Create result dictionary
        result = {
            't': t_mean,
            't_ex_mean': t_ex_mean,
            'imb_mean': imb_mean,
            'L': L,
            'h': h,
            'maxdim': maxdim,
            'num_realizations': num_realizations
        }

        # Compute standard deviation if requested
        if compute_std:
            t_ex_std = np.std(t_ex_arrays, axis=0, ddof=1)
            imb_std = np.std(imb_arrays, axis=0, ddof=1)
            result['t_ex_std'] = t_ex_std
            result['imb_std'] = imb_std

        # Compute standard error of mean if requested
        if compute_sem:
            t_ex_sem = np.std(t_ex_arrays, axis=0, ddof=1) / np.sqrt(num_realizations)
            imb_sem = np.std(imb_arrays, axis=0, ddof=1) / np.sqrt(num_realizations)
            result['t_ex_sem'] = t_ex_sem
            result['imb_sem'] = imb_sem

        results.append(result)

    df_averaged = pd.DataFrame(results)

    if verbose:
        print(f"\nOutput: {len(df_averaged)} averaged rows")
        print(f"Realizations per row: {df_averaged['num_realizations'].min()}-{df_averaged['num_realizations'].max()}")

    return df_averaged


def save_averaged_data(df_averaged: pd.DataFrame,
                       filepath: str = "../subset_data_averaged.pkl",
                       verbose: bool = True) -> None:
    """
    Save averaged data to pickle file for fast loading.

    Args:
        df_averaged: Averaged DataFrame from average_over_disorder()
        filepath: Path where to save the pickle file
        verbose: If True, print save information
    """
    df_averaged.to_pickle(filepath)

    if verbose:
        import os
        size_mb = os.path.getsize(filepath) / 1024**2
        print(f"Saved averaged data to: {filepath} ({size_mb:.1f} MB)")


def load_averaged_data(filepath: str = "../subset_data_averaged.pkl",
                      verbose: bool = True) -> pd.DataFrame:
    """
    Load pre-averaged data from pickle file.

    Args:
        filepath: Path to the averaged pickle file
        verbose: If True, print loading information

    Returns:
        DataFrame with averaged data
    """
    if verbose:
        print(f"Loading averaged data from: {filepath}")

    df = pd.read_pickle(filepath)

    if verbose:
        print(f"Loaded {len(df)} averaged rows")

    return df


def get_averaged_row(df_averaged: pd.DataFrame, L: float, h: float) -> pd.Series:
    """
    Get a specific averaged row by L and h values.

    Args:
        df_averaged: Averaged DataFrame
        L: Lattice size
        h: Field strength

    Returns:
        Single row as pandas Series
    """
    mask = (df_averaged['L'] == L) & (df_averaged['h'] == h)
    rows = df_averaged[mask]

    if len(rows) == 0:
        raise ValueError(f"No data found for L={L}, h={h}")
    if len(rows) > 1:
        print(f"Warning: Multiple rows found for L={L}, h={h}, returning first")

    return rows.iloc[0]


if __name__ == "__main__":
    print("="*70)
    print("AVERAGING DATA OVER DISORDER CONFIGURATIONS")
    print("="*70)
    print()

    # Load original data
    print("Step 1: Loading original data...")
    df = load_from_pickle("../subset_data.pkl", verbose=False)
    print(f"Loaded {len(df)} rows with {df['grid'].nunique()} unique grid values")
    print()

    # Average over disorder
    print("Step 2: Averaging over disorder configurations...")
    df_averaged = average_over_disorder(df, compute_std=True, compute_sem=True, verbose=True)
    print()

    # Save averaged data
    print("Step 3: Saving averaged data...")
    save_averaged_data(df_averaged, verbose=True)
    print()

    # Show example
    print("="*70)
    print("EXAMPLE: Accessing averaged data")
    print("="*70)
    row = get_averaged_row(df_averaged, L=10.0, h=15.0)
    print(f"\nAveraged data for L=10, h=15:")
    print(f"  Number of realizations: {row['num_realizations']}")
    print(f"  Time array shape: {row['t'].shape}")
    print(f"  Mean imbalance shape: {row['imb_mean'].shape}")
    print(f"  Std imbalance shape: {row['imb_std'].shape}")
    print(f"  SEM imbalance shape: {row['imb_sem'].shape}")
    print(f"\n  At t=0:")
    print(f"    imb_mean = {row['imb_mean'][0]:.6f}")
    print(f"    imb_std  = {row['imb_std'][0]:.6f}")
    print(f"    imb_sem  = {row['imb_sem'][0]:.6f}")
    print(f"\n  At t=50 (index 500):")
    print(f"    imb_mean = {row['imb_mean'][500]:.6f}")
    print(f"    imb_std  = {row['imb_std'][500]:.6f}")
    print(f"    imb_sem  = {row['imb_sem'][500]:.6f}")
