# Complete Analysis Pipeline - Final Summary

This directory contains a complete pipeline for loading, processing, and visualizing your simulation data.

## 🎯 What's Ready

### 1. Data Loading (Fast!)
- ✅ `subset_data.pkl` - Fast-loading format (106 MB)
- ✅ Loading time: **0.07 seconds** (was 32 seconds from CSV)
- ✅ 452x speedup achieved

### 2. Data Processing (Disorder Averaging)
- ✅ `subset_data_averaged.pkl` - Averaged over disorder configs (2.7 MB)
- ✅ 50 unique (L, h) combinations with statistics
- ✅ Mean, std, and SEM computed for all time points

### 3. Comprehensive Visualization
- ✅ **66 plots generated** covering all parameter combinations
- ✅ Organized in 4 categories (individual, by_L, by_h, summary)
- ✅ Publication-ready quality (150 DPI PNG)

## 📊 Generated Plots

| Category | Count | Description | Location |
|----------|-------|-------------|----------|
| Individual | 50 | Detailed analysis for each (L, h) | `plots/individual/` |
| Multi-panel by L | 5 | All h values for each L | `plots/by_L/` |
| Comparison by h | 10 | System size comparison for each h | `plots/by_h/` |
| Summary heatmap | 1 | Parameter space overview | `plots/summary/` |
| **TOTAL** | **66** | | |

## 🚀 Quick Start Guide

### Load Data (Fastest Method)
```python
from load_subset_data import load_from_pickle
df = load_from_pickle("../subset_data.pkl")
```

### Load Averaged Data
```python
from process_data import load_averaged_data, get_averaged_row

# Load averaged data
df_avg = load_averaged_data("../subset_data_averaged.pkl")

# Get specific parameter combination
row = get_averaged_row(df_avg, L=10.0, h=15.0)

# Access data
t = row['t']              # Time array
imb_mean = row['imb_mean']  # Mean imbalance
imb_sem = row['imb_sem']    # Standard error
```

### Regenerate All Plots
```bash
python plot_all_data.py
```

### Run Entire Pipeline
```bash
python main.py
```
- `python main.py` recomputes disorder averages, imbalance fits, β(h) thresholds, and all derived plots.
- Add flags like `--reuse-averaged`, `--reuse-beta-fits`, or `--reuse-h-plots` to reuse cached outputs.

## 📉 Imbalance Decay Exponent Fits

- Run `python fit_imbalance_decay.py` from this directory to fit every (L, h) combination to `C · t^{-β}` on `t ∈ [50, 100]`.
- Outputs save under `plots/beta_decay/`:
  - `tables/beta_decay_results.csv` summarising β, analytic standard errors (unweighted & weighted regressions), and bootstrap uncertainties.
  - `fits/L{L}_h{h}.png` log-log diagnostics per parameter pair (mean curve, SEM band, fitted power laws).
  - `beta_unweighted_heatmap.png` and `beta_weighted_heatmap.png` visualising β across the parameter grid.
  - `beta_vs_h_unweighted.png`, `beta_vs_h_weighted.png`, and `beta_vs_h_bootstrap.png` plotting β vs h for each L (log-scale y-axis with error bars), overlaid with exponential fits (linear in log-space) and markers where β=0.010 and β=0.005.
  - `h_vs_L.png` (weighted fits) and `h_vs_L_bootstrap.png` (bootstrap fits) summarising the extracted threshold fields h(β=0.010) and h(β=0.005) as a function of L with error bars.
- Tweak fit settings (window, minimum points, bootstrap draws) via the constants at the top of `fit_imbalance_decay.py`.
- To regenerate just the β vs h plots without rerunning fits, execute `python plot_beta_vs_h.py`.
- To tabulate the h values where β reaches 0.010 and 0.005, run `python fit_beta_vs_h.py`; the summary is stored at `plots/beta_decay/tables/beta_hc_estimates.csv`.
- To visualise those threshold fields versus L, run `python plot_h_vs_L.py` (reads the CSV above and emits `plots/beta_decay/h_vs_L.png`). Use `--results plots/beta_decay/tables/beta_hc_bootstrap.csv --output plots/beta_decay/h_vs_L_bootstrap.png` for the bootstrap-derived curve.

## 📁 File Organization

```
python/
├── load_subset_data.py          # Fast data loading module
├── process_data.py              # Disorder averaging module
├── plot_all_data.py             # Comprehensive plotting script
│
├── subset_data.pkl              # Fast-loading original data (106 MB)
├── subset_data_averaged.pkl     # Averaged data (2.7 MB)
│
├── plots/                       # Generated plots
│   ├── individual/              # 50 individual plots
│   ├── by_L/                   # 5 multi-panel plots
│   ├── by_h/                   # 10 comparison plots
│   └── summary/                # 1 heatmap
│
└── README files:
    ├── README_subset_data.md    # Data loading documentation
    ├── README_PROCESSING.md     # Processing documentation
    ├── PLOTS_GUIDE.md          # Plot organization guide
    ├── QUICK_REFERENCE.md      # Quick reference
    └── README_FINAL.md         # This file
```

## 📈 Dataset Summary

- **Original data**: 4626 rows (different disorder realizations)
- **Averaged data**: 50 rows (unique L, h combinations)
- **Parameters**:
  - L: 4, 6, 8, 10, 12
  - h: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
- **Time points**: 1001 (t = 0 to 100, dt = 0.1)
- **Disorder realizations per (L, h)**: 67-100

## 🎓 Key Scripts

### Data Loading & Conversion
- `convert_to_fast_format.py` - One-time CSV → pickle conversion
- `compare_loading_speeds.py` - Benchmark loading speeds
- `quick_start.py` - Simple usage example

### Data Processing
- `process_data.py` - Average over disorder, compute statistics
- Main function: `average_over_disorder()`

### Visualization
- `plot_all_data.py` - Generate all 66 plots
- `plot_averaged_data.py` - Example plotting script (demo)
- `show_sample_plots.py` - Display plot summary

## 💡 Common Tasks

### Task 1: Load and filter data
```python
from load_subset_data import load_from_pickle, filter_by_params

df = load_from_pickle("../subset_data.pkl")
subset = filter_by_params(df, L=10.0, h=15.0)
```

### Task 2: Get averaged data for specific parameters
```python
from process_data import load_averaged_data, get_averaged_row

df_avg = load_averaged_data("../subset_data_averaged.pkl")
row = get_averaged_row(df_avg, L=10.0, h=15.0)

print(f"Number of realizations: {row['num_realizations']}")
print(f"Mean imbalance at t=50: {row['imb_mean'][500]:.3f}")
print(f"Standard error at t=50: {row['imb_sem'][500]:.3f}")
```

### Task 3: Plot with error bands
```python
import matplotlib.pyplot as plt
from process_data import load_averaged_data, get_averaged_row

df_avg = load_averaged_data("../subset_data_averaged.pkl")
row = get_averaged_row(df_avg, L=10.0, h=15.0)

plt.figure(figsize=(10, 6))
plt.plot(row['t'], row['imb_mean'], 'b-', linewidth=2)
plt.fill_between(row['t'],
                 row['imb_mean'] - row['imb_sem'],
                 row['imb_mean'] + row['imb_sem'],
                 alpha=0.3)
plt.xlabel('Time')
plt.ylabel('Imbalance')
plt.title(f"L={int(row['L'])}, h={int(row['h'])}")
plt.show()
```

### Task 4: Analyze all parameter combinations
```python
from process_data import load_averaged_data

df_avg = load_averaged_data("../subset_data_averaged.pkl")

# Get imbalance at t=50 for all combinations
t_idx = 500  # t=50
results = df_avg[['L', 'h']].copy()
results['imb_t50'] = df_avg['imb_mean'].apply(lambda x: x[t_idx])
results['sem_t50'] = df_avg['imb_sem'].apply(lambda x: x[t_idx])

print(results)
```

## ⚡ Performance Summary

| Operation | Time | Notes |
|-----------|------|-------|
| Load original CSV | 32.3 s | Slow, only needed once |
| Load pickle | 0.07 s | **452x faster!** |
| Average over disorder | ~5 s | One-time processing |
| Load averaged data | 0.01 s | Instant access |
| Generate all plots | ~60 s | Creates 66 plots |

## 📚 Documentation

For detailed information, see:
- **QUICK_REFERENCE.md** - Cheat sheet for daily use
- **README_subset_data.md** - Data loading details
- **README_PROCESSING.md** - Processing and statistics explanation
- **PLOTS_GUIDE.md** - Complete guide to all 66 plots

## ✨ What You Gained

1. **Speed**: 452x faster data loading
2. **Organization**: Well-structured, documented code
3. **Reproducibility**: All processing steps automated
4. **Visualization**: Comprehensive plots for all parameters
5. **Statistics**: Proper error analysis with SEM and std
6. **Flexibility**: Easy to extend and modify

## 🔍 Next Steps

Depending on your needs:
1. Explore individual plots in `plots/`
2. Modify `plot_all_data.py` for custom visualizations
3. Use averaged data for further analysis (fitting, scaling, etc.)
4. Extract specific statistics from `subset_data_averaged.pkl`

## 📝 Citation

If using this pipeline, please acknowledge:
- Disorder averaging over grid realizations
- Error bars represent ±1 SEM (standard error of mean)
- N realizations averaged per (L, h) combination (typically 67-100)

---

**Pipeline created**: November 2025
**Total plots generated**: 66
**Data formats**: CSV → Pickle (optimized)
**Processing**: Disorder averaging with full statistics
