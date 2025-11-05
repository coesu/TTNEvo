# Quick Reference - subset_data Loading

## 🚀 Fast Loading (What You Need)

```python
from load_subset_data import load_from_pickle

# Load data in 0.07 seconds (instead of 32 seconds!)
df = load_from_pickle("../subset_data.pkl")
```

## 📊 Common Operations

```python
from load_subset_data import load_from_pickle, filter_by_params

# Load data
df = load_from_pickle("../subset_data.pkl")

# Filter by parameters
subset = filter_by_params(df, L=10.0, h=15.0)

# Access a row
row = df.iloc[0]

# Get time series arrays (numpy arrays)
t = row['t']        # Time points (1001 values)
t_ex = row['t_ex']  # Evolution values (1001 values)
imb = row['imb']    # Imbalance values (1001 values)

# Get parameters
L = row['L']
h = row['h']
grid = row['grid']
maxdim = row['maxdim']
```

## 📦 Dataset Info

- **4626 rows** of simulation data
- **Array columns**: `t`, `t_ex`, `imb` (each 1001 values)
- **Parameters**: `L`, `h`, `grid`, `maxdim`
- **Parameter ranges**:
  - L: 4, 6, 8, 10, 12
  - h: 5, 10, 15, ..., 50
  - grid: 1, 2, 3, ..., 100
  - maxdim: 32 (constant)

## ⚡ Performance

| Method | Time | Speedup |
|--------|------|---------|
| CSV (original) | 32.3 sec | 1x |
| Pickle (new) | 0.07 sec | **452x faster** |

## 🔧 First-Time Setup

If `subset_data.pkl` doesn't exist yet, run once:

```bash
python convert_to_fast_format.py
```

This converts the CSV to pickle format (takes ~34 seconds, but only once!).

## 📚 More Information

- See `README_subset_data.md` for detailed documentation
- See `quick_start.py` for complete example
- See `example_usage.py` for plotting examples
