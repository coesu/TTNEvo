# Parsing subset_data.csv

This directory contains Python utilities to parse and work with the `subset_data.csv` file, which has a non-standard CSV format where some columns contain arrays with commas inside them.

## Problem

The `subset_data.csv` file contains entries with embedded commas, making standard CSV parsing fail. The file structure is:

```
t,t_ex,imb,L,h,grid,maxdim
[0.0, 0.1, ...],[0.0, 103.8, ...],[1.0, 0.991, ...],10,15,100,32
```

Where the first three columns (`t`, `t_ex`, `imb`) are arrays containing 1001 values each, and the remaining columns are scalar parameters.

## Solution: Fast Loading with Pickle

**TL;DR**: Convert the CSV once to pickle format, then load in **0.07 seconds** instead of **32 seconds**!

### Quick Start (Recommended Workflow)

1. **One-time conversion** (run this once):
   ```bash
   python convert_to_fast_format.py
   ```
   This creates `subset_data.pkl` (takes ~34 seconds, but only once!)

2. **Daily use** (load in your scripts):
   ```python
   from load_subset_data import load_from_pickle

   df = load_from_pickle("../subset_data.pkl")  # 0.07 seconds!
   ```

**Speed comparison**: Pickle loading is **452x faster** than CSV parsing!

### Files

1. **`load_subset_data.py`** - Main module with loading, filtering, and conversion functions
2. **`convert_to_fast_format.py`** - One-time conversion script (run this first!)
3. **`quick_start.py`** - Simple example showing recommended workflow
4. **`compare_loading_speeds.py`** - Benchmark comparing CSV vs pickle loading
5. **`example_usage.py`** - Detailed examples with plotting
6. **`parse_subset_data.py`** - Original development/testing script

### Detailed Usage

#### Fast Loading (Recommended)

```python
from load_subset_data import load_from_pickle

# Load from pickle (0.07 seconds!)
df = load_from_pickle("../subset_data.pkl")
```

#### Direct CSV Loading (Slow, for reference)

```python
from load_subset_data import load_subset_data

# Load data with arrays as numpy arrays (takes ~32 seconds)
df = load_subset_data("../subset_data.csv", convert_arrays_to_numpy=True)
```

#### One-time Conversion

```python
from load_subset_data import convert_csv_to_fast_format

# Convert CSV to pickle format (run once)
df = convert_csv_to_fast_format(
    csv_path="../subset_data.csv",
    output_pickle="../subset_data.pkl"
)
```

#### Filtering Data

```python
from load_subset_data import filter_by_params

# Get all rows with specific parameters
subset = filter_by_params(df, L=10.0, h=20.0)

# Filter by multiple parameters
subset = filter_by_params(df, L=10.0, h=20.0, grid=50.0)
```

#### Accessing Array Data

```python
# Get a specific row
row = df.iloc[0]

# Access the time series arrays (if converted to numpy)
t = row['t']        # Time array (1001 values)
t_ex = row['t_ex']  # t_ex array (1001 values)
imb = row['imb']    # Imbalance array (1001 values)

# Access scalar parameters
L = row['L']
h = row['h']
grid = row['grid']
maxdim = row['maxdim']
```

#### Getting Parameter Info

```python
from load_subset_data import get_unique_params

# Get all unique values for each parameter
params = get_unique_params(df)
print(params)
# {'L': [4.0, 6.0, 8.0, 10.0, 12.0],
#  'h': [5.0, 10.0, 15.0, ..., 50.0],
#  'grid': [1.0, 2.0, 3.0, ..., 100.0],
#  'maxdim': [32.0]}
```

## Dataset Information

- **Total rows**: 4626
- **Columns**: 7
  - `t`: Time array (1001 values)
  - `t_ex`: Evolution values (1001 values)
  - `imb`: Imbalance values (1001 values)
  - `L`: Lattice size (5 unique values: 4, 6, 8, 10, 12)
  - `h`: Field strength (10 unique values: 5, 10, ..., 50)
  - `grid`: Grid parameter (100 unique values: 1, 2, ..., 100)
  - `maxdim`: Maximum dimension (always 32)

## Implementation Details

The parser works by:

1. Reading the header line to get column names
2. For each data row:
   - Finding all array columns by matching `[...]` bracket pairs
   - Parsing arrays using `ast.literal_eval()`
   - Extracting remaining scalar values after the arrays
3. Building a pandas DataFrame with:
   - Array columns stored as list or numpy array objects
   - Scalar columns stored as float values

## Examples

See `example_usage.py` for complete examples including:
- Loading and filtering data
- Accessing time series arrays
- Plotting data with matplotlib
- Iterating over parameter combinations

Run it with:
```bash
python example_usage.py
```

This will:
- Load the full dataset
- Show parameter ranges
- Filter data for specific L and h values
- Create example plots saved to `subset_data_example.png`
- Print statistics about data distribution

## Performance

### Loading Times (measured on typical machine)

| Method | Time | Speedup | File Size |
|--------|------|---------|-----------|
| CSV parsing (original) | 32.3 seconds | 1x (baseline) | 173 MB |
| Pickle loading | 0.07 seconds | **452x faster** | 106 MB |

### Recommendations

1. **Use pickle format** for fast loading (recommended for most use cases)
   - Loading time: ~0.07 seconds
   - File size: 106 MB (compressed)
   - Perfect preservation of numpy arrays
   - Python-only (not cross-language compatible)

2. **Use parquet format** if you need cross-platform compatibility
   - Requires `pyarrow`: `pip install pyarrow`
   - Loading time: ~0.2-0.3 seconds (still much faster than CSV)
   - Cross-language compatible (R, Julia, etc.)
   - Slightly larger file size

3. **Don't load from CSV repeatedly** - it's 452x slower!
