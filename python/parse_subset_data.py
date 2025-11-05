"""
Parse the subset_data.csv file which contains arrays in the first two columns.
"""
import ast
import pandas as pd
from typing import List, Dict, Any


def parse_row(line: str) -> Dict[str, Any]:
    """
    Parse a single data row that has the format:
    [array1],[array2],[array3],scalar1,scalar2,scalar3,scalar4
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


def parse_subset_data(filepath: str) -> pd.DataFrame:
    """
    Parse the subset_data.csv file and return a pandas DataFrame.

    Note: Since each row contains arrays of different lengths potentially,
    we'll store the arrays as list objects in the DataFrame columns.
    """
    with open(filepath, 'r') as f:
        # Read the header
        header = f.readline().strip().split(',')
        print(f"Header: {header}")
        print(f"Number of columns: {len(header)}")

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
                        row[header[i]] = array

                # Add scalar columns (after the array columns)
                num_array_cols = len(arrays)
                for i, scalar_val in enumerate(scalars):
                    if num_array_cols + i < len(header):
                        row[header[num_array_cols + i]] = scalar_val

                rows.append(row)
                line_count += 1

                if line_count % 1000 == 0:
                    print(f"Processed {line_count} rows...")

            except Exception as e:
                print(f"Error parsing line {line_count + 1}: {e}")
                print(f"Line preview: {line[:200]}")
                raise

        print(f"Successfully parsed {line_count} rows")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Parse the file
    filepath = "/home/lars/Syncthing/master-thesis/TTNEvo/subset_data.csv"
    df = parse_subset_data(filepath)

    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nColumn names: {df.columns.tolist()}")
    print(f"\nFirst few rows (showing scalars only):")
    scalar_cols = ['L', 'h', 'grid', 'maxdim']
    available_cols = [col for col in scalar_cols if col in df.columns]
    if available_cols:
        print(df[available_cols].head())

    print(f"\nData types:")
    print(df.dtypes)

    print(f"\nArray lengths in first row:")
    for col in ['t', 't_ex', 'imb']:
        if col in df.columns:
            print(f"  {col} array length: {len(df[col].iloc[0])}")

    # Example: Access the arrays
    print(f"\nFirst 5 values of each array in first row:")
    for col in ['t', 't_ex', 'imb']:
        if col in df.columns:
            print(f"  {col}: {df[col].iloc[0][:5]}")
