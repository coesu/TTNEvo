# Comprehensive Plots Guide

All plots have been generated for the disorder-averaged data. This document explains the organization and content of the 66 generated plots.

## Directory Structure

```
python/plots/
├── individual/     (50 plots) - Detailed plots for each (L, h) combination
├── by_L/          (5 plots)  - Multi-panel comparison for each L value
├── by_h/          (10 plots) - System size comparison for each h value
└── summary/       (1 plot)   - Heatmap overview of all parameters
```

## 1. Individual Plots (50 plots)

**Location**: `plots/individual/`

**Naming**: `L{LL}_h{HH}.png` (e.g., `L04_h05.png`, `L10_h15.png`)

**Content**: Each plot shows detailed analysis for a single (L, h) combination:

### Left Panel: Disorder-averaged Imbalance
- **Blue line**: Mean imbalance over time
- **Light blue band**: ±1 SEM (standard error of mean) - uncertainty in the mean
- **Light red band**: ±1 std (standard deviation) - disorder variation
- **Title**: Shows L, h, and number of disorder realizations (n)

### Right Panel: Uncertainty Evolution
- **Red line**: Standard deviation (variation across disorder realizations)
- **Green line**: Standard error of mean (uncertainty in the mean estimate)

**Use Cases**:
- Detailed examination of specific parameter combinations
- Understanding disorder effects for individual systems
- Publication-quality figures for specific cases

**Example files**:
- `L04_h05.png` - Small system, weak field
- `L10_h15.png` - Medium system, medium field
- `L12_h50.png` - Large system, strong field

---

## 2. Multi-Panel Plots by L (5 plots)

**Location**: `plots/by_L/`

**Naming**: `L{LL}_all_h.png` (e.g., `L04_all_h.png`, `L10_all_h.png`)

**Content**: Each plot shows all 10 h values for a fixed L in a 2×5 grid

**Layout**: 10 subplots arranged in 2 rows × 5 columns
- Each subplot shows imbalance vs time with error bands
- Subplots are ordered by increasing h value
- All use the same y-axis scale (0 to 1.05) for easy comparison

**Use Cases**:
- Compare field strength dependence at fixed system size
- Observe trends in relaxation as h increases
- Identify optimal field strengths for each L

**Files**:
- `L04_all_h.png` - L=4 (smallest system)
- `L06_all_h.png` - L=6
- `L08_all_h.png` - L=8
- `L10_all_h.png` - L=10
- `L12_all_h.png` - L=12 (largest system)

---

## 3. Comparison Plots by h (10 plots)

**Location**: `plots/by_h/`

**Naming**: `h{HH}_all_L.png` (e.g., `h05_all_L.png`, `h15_all_L.png`)

**Content**: Each plot overlays all 5 L values for a fixed h

**Features**:
- **Color coding**: Viridis colormap (purple → yellow) for L=4 → L=12
- **Error bands**: ±1 SEM shown for each L value
- **Legend**: Shows L value and number of realizations (n)
- **Title**: "System size dependence (h={value})"

**Use Cases**:
- Study system size scaling at fixed field strength
- Identify finite-size effects
- Compare relaxation timescales across system sizes
- Observe convergence with increasing L

**Files**:
- `h05_all_L.png` through `h50_all_L.png` (10 files total)

---

## 4. Summary Heatmap (1 plot)

**Location**: `plots/summary/`

**File**: `heatmaps_imbalance_vs_L_h.png`

**Content**: 6 heatmaps showing imbalance as function of (L, h) at different times

**Time points shown**: t = 0, 10, 25, 50, 75, 100

**Features**:
- **Axes**: L (x-axis) vs h (y-axis)
- **Color**: Viridis colormap (0 = purple, 1 = yellow)
- **Annotations**: Numerical values displayed in each cell
- **Colorbar**: Shows imbalance scale (0 to 1)

**Use Cases**:
- Quick overview of entire parameter space
- Identify regions of interest (fast/slow relaxation)
- Compare imbalance at different time snapshots
- Spot patterns and trends in parameter dependence

**Observations from heatmap**:
- At t=0: All systems start at imbalance ≈ 1 (initial state)
- As time increases: Imbalance generally decreases
- Strong h: Faster relaxation (lower imbalance at later times)
- Larger L: Tends to show slower relaxation (higher imbalance)

---

## Quick Reference

| Need | Go To |
|------|-------|
| Specific (L, h) in detail | `python/plots/individual/L{LL}_h{HH}.png` |
| All h at one L | `python/plots/by_L/L{LL}_all_h.png` |
| System size comparison at one h | `python/plots/by_h/h{HH}_all_L.png` |
| Overall parameter space | `python/plots/summary/heatmaps_imbalance_vs_L_h.png` |

## Parameters Available

- **L values**: 4, 6, 8, 10, 12 (5 values)
- **h values**: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 (10 values)
- **Total combinations**: 50
- **Disorder realizations per combination**: 67-100

## Data Quality

All plots show:
- **Disorder-averaged** quantities (mean over grid realizations)
- **Error estimates** (SEM for statistical uncertainty)
- **Number of realizations** (n) for transparency

Error bands are shown as ±1 SEM, which represents:
- ~68% confidence interval for the mean
- Typical statistical uncertainty in the ensemble average

## Regenerating Plots

To regenerate all plots:

```bash
python plot_all_data.py
```

This will:
1. Load averaged data from `subset_data_averaged.pkl`
2. Create output directories if needed
3. Generate all 66 plots (~1-2 minutes)

## File Size

Total plot data: ~8-10 MB
- Individual plots: ~140-150 KB each
- Multi-panel plots: ~130-180 KB each
- Heatmap: ~240 KB

All plots saved as PNG with 150 DPI (print quality).
