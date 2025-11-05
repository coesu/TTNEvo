#!/usr/bin/env python3
"""
Display information about the generated plots and show some examples.
"""
from pathlib import Path

def main():
    plots_dir = Path("plots")

    print("="*70)
    print("PLOT GENERATION SUMMARY")
    print("="*70)
    print()

    # Count plots in each directory
    individual_plots = list((plots_dir / "individual").glob("*.png"))
    by_L_plots = list((plots_dir / "by_L").glob("*.png"))
    by_h_plots = list((plots_dir / "by_h").glob("*.png"))
    summary_plots = list((plots_dir / "summary").glob("*.png"))

    print(f"📁 Directory: {plots_dir.absolute()}")
    print()
    print(f"📊 Plot Statistics:")
    print(f"  Individual plots: {len(individual_plots)} files")
    print(f"  Multi-panel by L: {len(by_L_plots)} files")
    print(f"  Comparison by h:  {len(by_h_plots)} files")
    print(f"  Summary heatmaps: {len(summary_plots)} files")
    print(f"  ─────────────────────────────")
    print(f"  TOTAL:           {len(individual_plots) + len(by_L_plots) + len(by_h_plots) + len(summary_plots)} plots")
    print()

    # Show some example files
    print("="*70)
    print("EXAMPLE FILES")
    print("="*70)
    print()

    print("📈 Individual Plots (examples):")
    examples = ["L04_h05.png", "L08_h15.png", "L10_h20.png", "L12_h50.png"]
    for ex in examples:
        filepath = plots_dir / "individual" / ex
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  ✓ {ex:<20} ({size_kb:.0f} KB)")
    print()

    print("📊 Multi-Panel Plots by L:")
    for plot in sorted(by_L_plots):
        size_kb = plot.stat().st_size / 1024
        print(f"  ✓ {plot.name:<20} ({size_kb:.0f} KB)")
    print()

    print("📉 Comparison Plots by h (first 5):")
    for plot in sorted(by_h_plots)[:5]:
        size_kb = plot.stat().st_size / 1024
        print(f"  ✓ {plot.name:<20} ({size_kb:.0f} KB)")
    print(f"  ... and {len(by_h_plots) - 5} more")
    print()

    print("🗺️  Summary Heatmap:")
    for plot in summary_plots:
        size_kb = plot.stat().st_size / 1024
        print(f"  ✓ {plot.name:<45} ({size_kb:.0f} KB)")
    print()

    # Parameter coverage
    print("="*70)
    print("PARAMETER COVERAGE")
    print("="*70)
    print()
    print("All combinations of:")
    print("  L ∈ {4, 6, 8, 10, 12}        (5 values)")
    print("  h ∈ {5, 10, 15, ..., 50}    (10 values)")
    print("  ─────────────────────────────")
    print("  Total: 5 × 10 = 50 combinations")
    print()
    print("Each with 67-100 disorder realizations averaged")
    print()

    # Access instructions
    print("="*70)
    print("HOW TO ACCESS")
    print("="*70)
    print()
    print("To view plots, navigate to:")
    print(f"  {plots_dir.absolute()}")
    print()
    print("Or use Python:")
    print("  from pathlib import Path")
    print("  from PIL import Image")
    print("  import matplotlib.pyplot as plt")
    print()
    print("  img = Image.open('plots/individual/L10_h15.png')")
    print("  plt.imshow(img)")
    print("  plt.axis('off')")
    print("  plt.show()")
    print()

    print("For more details, see: PLOTS_GUIDE.md")


if __name__ == "__main__":
    main()
