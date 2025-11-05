"""
Module for loading and working with the subset_data.csv file.

The file contains rows with array columns (t, t_ex, imb) and scalar columns (L, h, grid, maxdim).
"""
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union


def parse_row(line: str) -> Dict[str, Any]:
    """
    Parse a single data row that has the format:
    [array1],[array2],[array3],scalar1,scalar2,scalar3,scalar4

    Returns:
        Dictionary with 'arrays' (list of lists) and 'scalars' (list of values)
    """
    arrays = []

    # Extract all arrays by finding bracket pairs
    pos = 0
    while True:
        bracket_start = line.find('[', pos)
        if bracket_start == -1:
            break

        # Find the matching closing bracket
        bracket_count = 0
        idx = bracket_start
        while idx < len(line):
            if line[idx] == '[':
                bracket_count += 1
            elif line[idx] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    array_end = idx + 1
                    break
            idx += 1

        # Extract and parse the array
        array_str = line[bracket_start:array_end]
        arrays.append(ast.literal_eval(array_str))

        # Move position forward
        pos = array_end

    # Get the remaining scalar values (after the last array)
    remaining_values = line[pos:].strip(',').split(',')
    scalars = []
    for v in remaining_values:
        v = v.strip()
        if v:
            try:
                scalars.append(float(v))
            except ValueError:
                scalars.append(v)

    return {
        'arrays': arrays,
        'scalars': scalars
    }


def load_subset_data(filepath: str, convert_arrays_to_numpy: bool = False,
                     verbose: bool = True) -> pd.DataFrame:
    """
    Load the subset_data.csv file into a pandas DataFrame.

    Args:
        filepath: Path to the subset_data.csv file
        convert_arrays_to_numpy: If True, convert list arrays to numpy arrays
        verbose: If True, print progress messages

    Returns:
        pandas DataFrame with columns: t, t_ex, imb, L, h, grid, maxdim
        The first three columns contain arrays (as lists or numpy arrays)
    """
    with open(filepath, 'r') as f:
        # Read the header
        header = f.readline().strip().split(',')
        if verbose:
            print(f"Loading file: {filepath}")
            print(f"Columns: {header}")

        # Parse data rows
        rows = []
        line_count = 0

        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                parsed = parse_row(line)
                arrays = parsed['arrays']
                scalars = parsed['scalars']

                # Create row data - first columns are arrays
                row = {}
                for i, array in enumerate(arrays):
                    if i < len(header):
                        # Optionally convert to numpy array
                        row[header[i]] = np.array(array) if convert_arrays_to_numpy else array

                # Add scalar columns (after the array columns)
                num_array_cols = len(arrays)
                for i, scalar_val in enumerate(scalars):
                    if num_array_cols + i < len(header):
                        row[header[num_array_cols + i]] = scalar_val

                rows.append(row)
                line_count += 1

                if verbose and line_count % 1000 == 0:
                    print(f"  Processed {line_count} rows...")

            except Exception as e:
                print(f"Error parsing line {line_count + 1}: {e}")
                print(f"Line preview: {line[:200]}")
                raise

        if verbose:
            print(f"Successfully loaded {line_count} rows")

    return pd.DataFrame(rows)


def filter_by_params(df: pd.DataFrame, L: Optional[float] = None,
                     h: Optional[float] = None, grid: Optional[float] = None,
                     maxdim: Optional[float] = None) -> pd.DataFrame:
    """
    Filter the dataframe by parameter values.

    Args:
        df: DataFrame returned by load_subset_data
        L, h, grid, maxdim: Parameter values to filter by (None means no filter)

    Returns:
        Filtered DataFrame
    """
    result = df.copy()

    if L is not None:
        result = result[result['L'] == L]
    if h is not None:
        result = result[result['h'] == h]
    if grid is not None:
        result = result[result['grid'] == grid]
    if maxdim is not None:
        result = result[result['maxdim'] == maxdim]

    return result


def get_unique_params(df: pd.DataFrame) -> Dict[str, List]:
    """
    Get all unique values for each parameter column.

    Returns:
        Dictionary mapping parameter name to sorted list of unique values
    """
    param_cols = ['L', 'h', 'grid', 'maxdim']
    unique_vals = {}

    for col in param_cols:
        if col in df.columns:
            unique_vals[col] = sorted(df[col].unique())

    return unique_vals


def save_to_pickle(df: pd.DataFrame, filepath: Union[str, Path]) -> None:
    """
    Save DataFrame to pickle format for fast loading.

    This is the fastest option and preserves numpy arrays perfectly.
    Recommended for Python-only workflows.

    Args:
        df: DataFrame to save
        filepath: Path where to save the pickle file
    """
    filepath = Path(filepath)
    df.to_pickle(filepath)
    print(f"Saved to pickle: {filepath} ({filepath.stat().st_size / 1024**2:.1f} MB)")


def save_to_parquet(df: pd.DataFrame, filepath: Union[str, Path]) -> None:
    """
    Save DataFrame to parquet format (requires pyarrow or fastparquet).

    Note: Arrays are converted to lists for parquet compatibility.
    Good for cross-platform use and reasonable speed.

    Args:
        df: DataFrame to save
        filepath: Path where to save the parquet file
    """
    filepath = Path(filepath)

    # Convert numpy arrays to lists for parquet compatibility
    df_copy = df.copy()
    for col in ['t', 't_ex', 'imb']:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    df_copy.to_parquet(filepath, engine='pyarrow', compression='snappy')
    print(f"Saved to parquet: {filepath} ({filepath.stat().st_size / 1024**2:.1f} MB)")


def load_from_pickle(filepath: Union[str, Path], verbose: bool = True) -> pd.DataFrame:
    """
    Load DataFrame from pickle file (fastest method).

    Args:
        filepath: Path to the pickle file
        verbose: If True, print loading info

    Returns:
        Loaded DataFrame with numpy arrays preserved
    """
    filepath = Path(filepath)
    if verbose:
        print(f"Loading from pickle: {filepath}")

    df = pd.read_pickle(filepath)

    if verbose:
        print(f"Loaded {len(df)} rows in fast mode")

    return df


def load_from_parquet(filepath: Union[str, Path], convert_to_numpy: bool = True,
                     verbose: bool = True) -> pd.DataFrame:
    """
    Load DataFrame from parquet file.

    Args:
        filepath: Path to the parquet file
        convert_to_numpy: If True, convert list arrays back to numpy arrays
        verbose: If True, print loading info

    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)
    if verbose:
        print(f"Loading from parquet: {filepath}")

    df = pd.read_parquet(filepath, engine='pyarrow')

    # Convert lists back to numpy arrays if requested
    if convert_to_numpy:
        for col in ['t', 't_ex', 'imb']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)

    if verbose:
        print(f"Loaded {len(df)} rows")

    return df


def convert_csv_to_fast_format(csv_path: Union[str, Path],
                               output_pickle: Optional[Union[str, Path]] = None,
                               output_parquet: Optional[Union[str, Path]] = None,
                               verbose: bool = True) -> pd.DataFrame:
    """
    One-time conversion: Parse the slow CSV and save to fast-loading format(s).

    Args:
        csv_path: Path to the original subset_data.csv file
        output_pickle: If provided, save as pickle file (recommended, fastest)
        output_parquet: If provided, save as parquet file (cross-platform)
        verbose: If True, print progress

    Returns:
        The loaded DataFrame

    Example:
        # Do this once:
        df = convert_csv_to_fast_format(
            "subset_data.csv",
            output_pickle="subset_data.pkl",
            output_parquet="subset_data.parquet"
        )

        # Then use fast loading:
        df = load_from_pickle("subset_data.pkl")
    """
    print("=" * 70)
    print("CONVERTING CSV TO FAST FORMAT")
    print("=" * 70)
    print("This will take a while, but only needs to be done once!\n")

    # Load from CSV (slow)
    df = load_subset_data(csv_path, convert_arrays_to_numpy=True, verbose=verbose)

    # Save to requested formats
    if output_pickle:
        print()
        save_to_pickle(df, output_pickle)

    if output_parquet:
        print()
        save_to_parquet(df, output_parquet)

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE!")
    print("=" * 70)
    if output_pickle:
        print(f"\nFast loading: df = load_from_pickle('{output_pickle}')")
    if output_parquet:
        print(f"Fast loading: df = load_from_parquet('{output_parquet}')")

    return df


# Example usage
if __name__ == "__main__":
    # Load the data
    df = load_subset_data(
        "/home/lars/Syncthing/master-thesis/TTNEvo/subset_data.csv",
        convert_arrays_to_numpy=True,
        verbose=True
    )

    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nArray column lengths (first row):")
    for col in ['t', 't_ex', 'imb']:
        print(f"  {col}: {len(df[col].iloc[0])}")

    # Show unique parameter values
    print(f"\nUnique parameter values:")
    unique_params = get_unique_params(df)
    for param, values in unique_params.items():
        print(f"  {param}: {values}")

    # Example filtering
    print(f"\n{'='*60}")
    print("EXAMPLE: Filter for L=10, h=15")
    print(f"{'='*60}")
    filtered = filter_by_params(df, L=10.0, h=15.0)
    print(f"Number of rows: {len(filtered)}")
    print(f"\nFiltered data (showing scalars):")
    print(filtered[['L', 'h', 'grid', 'maxdim']])

    # Example: accessing array data
    print(f"\n{'='*60}")
    print("EXAMPLE: Accessing array data from first row")
    print(f"{'='*60}")
    row = filtered.iloc[0]
    print(f"Time array (first 10 values): {row['t'][:10]}")
    print(f"t_ex array (first 10 values): {row['t_ex'][:10]}")
    print(f"imb array (first 10 values): {row['imb'][:10]}")
    print(f"\nArray types: {type(row['t'])}")
