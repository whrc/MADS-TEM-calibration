#!/usr/bin/env python3
"""
Analyze error distribution by Plant Functional Type (PFT) across site CSV files.

For each site's SITENAME.csv, extracts PFT Comparison rows (panel, metric, item, value).
Aggregates errors by PFT across all sites and metrics to determine:
  - Error distribution per PFT (mean, MAE, std, count)
  - Which PFT has smallest error
  - Which PFT has greatest error

Error metric: mean absolute error (MAE) across all site×metric combinations.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import site_pfts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze PFT error distribution across site CSV files."
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


def resolve_pft_name(site_id: str, item: str) -> str:
    """Map pft0, pft1, ... to actual PFT names for a site."""
    m = re.match(r"^pft([0-9]+)$", str(item).strip())
    if not m:
        return item
    pfts = site_pfts.get_site_pfts(site_id)
    if not pfts:
        return item
    idx = int(m.group(1))
    if idx < len(pfts):
        return pfts[idx]
    return item


def load_pft_errors(sites_root: Path) -> pd.DataFrame:
    """Load all PFT Comparison rows from site CSV files."""
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
        pft_rows = df[df["panel"] == "PFT Comparison"].copy()
        if pft_rows.empty:
            continue
        pft_rows["site"] = site_dir.name
        pft_rows["pft"] = pft_rows["item"].apply(
            lambda x: resolve_pft_name(site_dir.name, x)
        )
        rows.append(pft_rows[["site", "metric", "pft", "value"]])
    if not rows:
        raise RuntimeError(f"No PFT Comparison data found in {sites_root}")
    return pd.concat(rows, ignore_index=True)


def _plot_mae_bar(ax: plt.Axes, agg_plot: pd.DataFrame) -> None:
    """Horizontal bar chart of MAE by PFT."""
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(agg_plot)))
    ax.barh(agg_plot["pft"], agg_plot["mae"], color=colors)
    ax.set_xlabel("Mean Absolute Error (MAE)")
    ax.set_ylabel("Plant Functional Type")
    ax.set_title("PFT Error Distribution (MAE across all sites and metrics)")
    ax.invert_yaxis()


def _plot_violin(ax: plt.Axes, df_all: pd.DataFrame, order_all: list[str]) -> None:
    """Violin plot of residuals by PFT."""
    sns.violinplot(
        data=df_all,
        y="pft",
        x="value",
        order=order_all,
        hue="pft",
        legend=False,
        palette="muted",
        ax=ax,
    )
    ax.axvline(0, color="k", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Residual (model - observation)")
    ax.set_ylabel("Plant Functional Type")
    ax.set_title("PFT Residual Distribution (all PFTs)")


def plot_pft_errors(agg: pd.DataFrame, df: pd.DataFrame, outdir: Path) -> None:
    """Generate PFT error plots."""
    sns.set_theme(style="whitegrid")

    # Filter out unresolved pft0-pft9 with very few samples for cleaner plots
    agg_plot = agg[~agg["pft"].str.match(r"^pft\d+$", na=False)].copy()
    order_all = agg_plot.sort_values("mae")["pft"].tolist()
    df_all = df[df["pft"].isin(order_all)].copy()
    n_pfts = len(order_all)

    # Figure 5: combined MAE bar (left) and violin (right)
    fig5, (ax_bar, ax_violin) = plt.subplots(
        1, 2, figsize=(16, max(8, 0.35 * n_pfts)), sharey=True
    )
    _plot_mae_bar(ax_bar, agg_plot)
    _plot_violin(ax_violin, df_all, order_all)
    ax_violin.set_ylabel("")
    plt.tight_layout()
    plt.savefig(outdir / "figure5.png", dpi=300)
    plt.close()

    # 1. Bar chart: MAE by PFT (sorted ascending)
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_mae_bar(ax, agg_plot)
    plt.tight_layout()
    plt.savefig(outdir / "pft_error_mae_bar.png", dpi=300)
    plt.close()

    # 2. Bar chart with error bars (std)
    fig, ax = plt.subplots(figsize=(12, 8))
    agg_plot["std_error"] = agg_plot["std_error"].fillna(0)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(agg_plot)))
    ax.barh(agg_plot["pft"], agg_plot["mae"], xerr=agg_plot["std_error"], color=colors, capsize=2)
    ax.set_xlabel("Mean Absolute Error (MAE) ± std")
    ax.set_ylabel("Plant Functional Type")
    ax.set_title("PFT Error Distribution with Uncertainty")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(outdir / "pft_error_mae_with_std.png", dpi=300)
    plt.close()

    # 3. Boxplot: residual distribution by PFT (top PFTs by sample count)
    pft_counts = df.groupby("pft").size()
    top_pfts = pft_counts.nlargest(20).index.tolist()
    df_top = df[df["pft"].isin(top_pfts)].copy()
    order = df_top.groupby("pft")["value"].apply(lambda x: x.abs().mean()).sort_values().index
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=df_top, y="pft", x="value", order=order, hue="pft", legend=False, palette="muted")
    ax.axvline(0, color="k", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Residual (model - observation)")
    ax.set_ylabel("Plant Functional Type")
    ax.set_title("PFT Residual Distribution (top 20 PFTs by sample count)")
    plt.tight_layout()
    plt.savefig(outdir / "pft_error_boxplot.png", dpi=300)
    plt.close()

    # 4. Violin plot for all PFTs
    fig, ax = plt.subplots(figsize=(10, max(8, 0.35 * n_pfts)))
    _plot_violin(ax, df_all, order_all)
    plt.tight_layout()
    plt.savefig(outdir / "pft_error_violin.png", dpi=300)
    plt.close()

    # 5. Heatmap: site × PFT (mean error per site×PFT across metrics)
    matrix = df.groupby(["site", "pft"])["value"].mean().unstack(fill_value=np.nan)
    # Exclude unresolved pft0-pft9
    pft_cols = [c for c in matrix.columns if not (isinstance(c, str) and re.match(r"^pft\d+$", c))]
    matrix = matrix[pft_cols]
    matrix = matrix.sort_index()
    vmax = np.nanmax(np.abs(matrix.values))
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(matrix.columns)), max(6, 0.35 * len(matrix.index))))
    sns.heatmap(
        matrix,
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "PFT error value"},
    )
    plt.title("PFT Error by Site and Plant Functional Type")
    plt.xlabel("Plant Functional Type")
    plt.ylabel("Site")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "pft_error_site_component_heatmap.png", dpi=300)
    plt.close()
    matrix.to_csv(outdir / "pft_error_site_component_matrix.csv")


def main() -> int:
    args = parse_args()
    sites_root = Path(args.sites_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_pft_errors(sites_root)
    df["abs_value"] = df["value"].abs()

    # Aggregate by PFT across all sites and metrics
    agg = (
        df.groupby("pft")
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
    agg.to_csv(outdir / "pft_error_summary.csv", index=False)

    # Report
    print("\n=== PFT Error Distribution (across all sites and metrics) ===\n")
    print(agg.to_string(index=False))
    print()

    smallest = agg.iloc[0]
    largest = agg.iloc[-1]
    print(f"Smallest error (lowest MAE): {smallest['pft']} (MAE = {smallest['mae']:.6f})")
    print(f"Greatest error (highest MAE): {largest['pft']} (MAE = {largest['mae']:.6f})")
    print(f"\nSaved: {outdir / 'pft_error_summary.csv'}")

    plot_pft_errors(agg, df, outdir)
    print(f"Saved: {outdir / 'figure5.png'}")
    print(f"Plots saved to: {outdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
