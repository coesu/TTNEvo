# Data Processing: Averaging Over Disorder Configurations

This guide explains how to work with disorder-averaged data from your simulations.

## Overview

The original `subset_data.csv` contains multiple disorder realizations (different `grid` values) for each parameter combination (L, h). This module averages the imbalance time series over all disorder configurations to get ensemble averages and statistical errors.

**Key transformation:**
- **Input**: 4626 rows (different disorder realizations)
- **Output**: 50 rows (one per unique L, h combination)
- **Reduction**: From 106 MB → 2.7 MB

## Quick Start

### 1. Create Averaged Data (One-time)

```python
from load_subset_data import load_from_pickle
from process_data import average_over_disorder, save_averaged_data

# Load original data
df = load_from_pickle("../subset_data.pkl")

# Average over disorder configurations
df_avg = average_over_disorder(df, compute_std=True, compute_sem=True)

# Save for fast loading
save_averaged_data(df_avg, "../subset_data_averaged.pkl")
```

Or simply run:
```bash
python process_data.py
```

### 2. Use Averaged Data

```python
from process_data import load_averaged_data, get_averaged_row

# Load averaged data (fast!)
df_avg = load_averaged_data("../subset_data_averaged.pkl")

# Get specific parameter combination
row = get_averaged_row(df_avg, L=10.0, h=15.0)

# Access averaged quantities
t = row['t']              # Time array
imb_mean = row['imb_mean']  # Mean imbalance
imb_std = row['imb_std']    # Standard deviation
imb_sem = row['imb_sem']    # Standard error of mean
n = row['num_realizations'] # Number of disorder realizations averaged
```

## Data Structure

### Original Data (per row)
- `t`: Time array (1001 points)
- `t_ex`: Evolution values (1001 points)
- `imb`: Imbalance values (1001 points)
- `L`, `h`, `grid`, `maxdim`: Parameters

### Averaged Data (per row)
- `t`: Time array (same as original)
- `t_ex_mean`: Mean of t_ex over disorder
- `imb_mean`: Mean of imbalance over disorder
- `t_ex_std`: Standard deviation of t_ex
- `imb_std`: Standard deviation of imbalance
- `t_ex_sem`: Standard error of mean for t_ex
- `imb_sem`: Standard error of mean for imbalance
- `L`, `h`, `maxdim`: Parameters
- `num_realizations`: Number of disorder configs averaged (typically 67-100)

## Statistics Computed

For each (L, h) combination and each time point:

1. **Mean**: `mean = (1/N) Σ imb_i`
   - Average over N disorder realizations

2. **Standard Deviation**: `std = sqrt[(1/(N-1)) Σ (imb_i - mean)²]`
   - Measures spread across disorder realizations
   - Uses Bessel's correction (N-1)

3. **Standard Error of Mean**: `sem = std / sqrt(N)`
   - Uncertainty in the mean estimate
   - Decreases as 1/√N with more realizations

## Example: Plotting with Error Bands

```python
import matplotlib.pyplot as plt
from process_data import load_averaged_data, get_averaged_row

# Load data
df_avg = load_averaged_data("../subset_data_averaged.pkl")
row = get_averaged_row(df_avg, L=10.0, h=15.0)

# Extract data
t = row['t']
imb_mean = row['imb_mean']
imb_sem = row['imb_sem']

# Plot with error band
fig, ax = plt.subplots()
ax.plot(t, imb_mean, 'b-', linewidth=2, label='Mean')
ax.fill_between(t, imb_mean - imb_sem, imb_mean + imb_sem,
                alpha=0.3, color='b', label='±1 SEM')
ax.set_xlabel('Time')
ax.set_ylabel('Imbalance')
ax.legend()
plt.show()
```

## Available Parameters

After averaging, you have 50 unique combinations:

- **L values**: 4, 6, 8, 10, 12 (lattice sizes)
- **h values**: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 (field strengths)
- **Realizations per combination**: 67-100 disorder configurations

## Example Results

At **t=50** for **L=10, h=15**:
- `imb_mean = 0.744170`
- `imb_std = 0.058253` (≈8% variation across disorder)
- `imb_sem = 0.006245` (≈0.8% uncertainty in mean)
- `num_realizations = 87`

This means:
- The mean imbalance has decayed to ~74% of initial value
- Different disorder realizations vary by ±5.8% (std)
- The mean is known to ±0.6% precision (sem)

## Files

1. **`process_data.py`** - Main processing module with averaging functions
2. **`plot_averaged_data.py`** - Example plots with error bands
3. **`subset_data_averaged.pkl`** - Averaged data (2.7 MB, created by process_data.py)

## Generated Plots

Running `plot_averaged_data.py` creates:

1. **`averaged_imbalance_vs_L.png`**
   - Imbalance vs time for different L at fixed h=15
   - Shows system size dependence

2. **`averaged_imbalance_vs_h.png`**
   - Imbalance vs time for different h at fixed L=10
   - Shows field strength dependence

3. **`averaged_imbalance_std_evolution.png`**
   - Evolution of mean and standard deviation
   - Shows how disorder effects change over time

## Performance

| Operation | Time |
|-----------|------|
| Load original data | 0.07 s |
| Average over disorder | ~5 s |
| Load averaged data | 0.01 s |

**Tip**: After creating `subset_data_averaged.pkl`, always use `load_averaged_data()` for instant loading!

## Interpretation

### Standard Deviation (std)
- Quantifies disorder-to-disorder variation
- Large std = system behavior strongly depends on specific disorder realization
- Small std = behavior is robust across disorder realizations

### Standard Error of Mean (sem)
- Quantifies uncertainty in the ensemble average
- Used for error bars in plots
- Decreases with √N, where N = number of realizations

### When to Use Which?

- **For comparing typical behavior**: Use `imb_mean ± imb_sem`
- **For understanding disorder effects**: Use `imb_mean ± imb_std`
- **For statistical significance**: Use `imb_sem` to determine if differences are significant
