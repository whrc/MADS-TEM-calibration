#!/usr/bin/env python3
"""
Analyze error distribution by soil component across site CSV files.

For each site's SITENAME.csv, extracts Soil Comparison rows (panel, metric, item, value).
Aggregates errors by soil component (SHLWC, DEEPC, MINEC, AVLN, etc.) across all sites
to determine:
  - Error distribution per component (mean, MAE, std, count)
  - Which component has smallest error
  - Which component has greatest error

Error metric: mean absolute error (MAE) across all sites.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze soil error distribution across site CSV files."
    )
    parser.add_argument(
        "--sites-root",
        default="sites_runs",
        help="Directory containing site folders (default: sites_runs).",
    )
    parser.add_argument(
        "--outdir",
        default="analysis_outputs",
        help="Directory to save outputs (default: analysis_outputs).",
    )
    return parser.parse_args()


def load_soil_errors(sites_root: Path) -> pd.DataFrame:
    """Load all Soil Comparison rows from site CSV files."""
    rows = []
    for site_dir in sorted(p for p in sites_root.iterdir() if p.is_dir()):
        csv_path = site_dir / f"{site_dir.name}.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Skipping {csv_path}: {e}")
            continue
        soil_rows = df[df["panel"] == "Soil Comparison"].copy()
        if soil_rows.empty:
            continue
        soil_rows["site"] = site_dir.name
        soil_rows["component"] = soil_rows["item"]
        rows.append(soil_rows[["site", "metric", "component", "value"]])
    if not rows:
        raise RuntimeError(f"No Soil Comparison data found in {sites_root}")
    return pd.concat(rows, ignore_index=True)


def _plot_mae_bar(ax: plt.Axes, agg: pd.DataFrame) -> None:
    """Horizontal bar chart of MAE by soil component."""
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(agg)))
    ax.barh(agg["component"], agg["mae"], color=colors)
    ax.set_xlabel("Mean Absolute Error (MAE)")
    ax.set_ylabel("Soil Component")
    ax.set_title("Soil Error Distribution (MAE across all sites)")
    ax.invert_yaxis()


def _plot_violin(ax: plt.Axes, df: pd.DataFrame, order: list[str]) -> None:
    """Violin plot of residuals by soil component."""
    sns.violinplot(
        data=df,
        y="component",
        x="value",
        order=order,
        hue="component",
        legend=False,
        palette="muted",
        orient="h",
        ax=ax,
    )
    ax.axvline(0, color="k", ls="--", lw=1, alpha=0.7)
    ax.set_ylabel("Soil Component")
    ax.set_xlabel("Residual (model - observation)")
    ax.set_title("Soil Residual Distribution (all components)")


def plot_soil_errors(agg: pd.DataFrame, df: pd.DataFrame, outdir: Path) -> None:
    """Generate soil error plots."""
    sns.set_theme(style="whitegrid")

    order = agg.sort_values("mae")["component"].tolist()

    # Figure 6: combined MAE bar (left) and violin (right)
    fig6, (ax_bar, ax_violin) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    _plot_mae_bar(ax_bar, agg)
    _plot_violin(ax_violin, df, order)
    ax_violin.set_ylabel("")
    plt.tight_layout()
    plt.savefig(outdir / "figure6.png", dpi=300)
    plt.close()

    # 1. Bar chart: MAE by soil component (sorted ascending)
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_mae_bar(ax, agg)
    plt.tight_layout()
    plt.savefig(outdir / "soil_error_mae_bar.png", dpi=300)
    plt.close()

    # 2. Bar chart with error bars (std)
    fig, ax = plt.subplots(figsize=(10, 6))
    agg_std = agg.copy()
    agg_std["std_error"] = agg_std["std_error"].fillna(0)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(agg)))
    ax.barh(agg_std["component"], agg_std["mae"], xerr=agg_std["std_error"], color=colors, capsize=2)
    ax.set_xlabel("Mean Absolute Error (MAE) ± std")
    ax.set_ylabel("Soil Component")
    ax.set_title("Soil Error Distribution with Uncertainty")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(outdir / "soil_error_mae_with_std.png", dpi=300)
    plt.close()

    # 3. Boxplot: residual distribution by soil component
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df, x="component", y="value", order=order, hue="component", legend=False, palette="muted"
    )
    ax.axhline(0, color="k", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Soil Component")
    ax.set_ylabel("Residual (model - observation)")
    ax.set_title("Soil Residual Distribution by Component")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "soil_error_boxplot.png", dpi=300)
    plt.close()

    # 4. Violin plot for all soil components (component on y-axis, residual on x-axis)
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_violin(ax, df, order)
    plt.tight_layout()
    plt.savefig(outdir / "soil_error_violin.png", dpi=300)
    plt.close()


def main() -> int:
    args = parse_args()
    sites_root = Path(args.sites_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_soil_errors(sites_root)
    df["abs_value"] = df["value"].abs()

    # Aggregate by soil component across all sites
    agg = (
        df.groupby("component")
        .agg(
            count=("value", "count"),
            mean_error=("value", "mean"),
            mae=("abs_value", "mean"),
            std_error=("value", "std"),
            median_abs_error=("abs_value", "median"),
        )
        .round(6)
    )
    agg = agg.sort_values("mae", ascending=True).reset_index()

    # Save summary
    agg.to_csv(outdir / "soil_error_summary.csv", index=False)

    # Report
    print("\n=== Soil Error Distribution (across all sites) ===\n")
    print(agg.to_string(index=False))
    print()

    smallest = agg.iloc[0]
    largest = agg.iloc[-1]
    print(f"Smallest error (lowest MAE): {smallest['component']} (MAE = {smallest['mae']:.6f})")
    print(f"Greatest error (highest MAE): {largest['component']} (MAE = {largest['mae']:.6f})")
    print(f"\nSaved: {outdir / 'soil_error_summary.csv'}")

    plot_soil_errors(agg, df, outdir)
    print(f"Saved: {outdir / 'figure6.png'}")
    print(f"Plots saved to: {outdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
