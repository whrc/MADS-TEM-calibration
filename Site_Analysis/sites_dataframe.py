#!/usr/bin/env python3
"""
Build a combined dataframe from all site CSV files in sites_runs/.
Each site's SITE.csv (e.g. CA_OBS.csv) is read and flattened into one row.
Columns: site_id, GPP_EverTree, GPP_DecidTree, NPP_EverTree, ..., SHLWC, DEEPC, MINEC, AVLN.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap


def load_site_row(site_dir: Path) -> Optional[pd.Series]:
    """
    Load a site's SITE.csv and return one row as a Series.
    Keys: site_id, metric_item (e.g. GPP_EverTree), or soil item (SHLWC, etc.)
    """
    site_id = site_dir.name
    csv_path = site_dir / f"{site_id}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    row = {"site_id": site_id}

    for _, r in df.iterrows():
        panel = r["panel"]
        metric = r["metric"]
        item = r["item"]
        value = r["value"]

        if panel == "PFT Comparison":
            col = f"{metric}_{item}"
        else:
            col = item
        row[col] = value

    return pd.Series(row)


def build_sites_dataframe(sites_root: str | Path = "sites_runs") -> pd.DataFrame:
    """
    Build a dataframe with one row per site. Columns: site_id, then all
    metric_item and soil columns from all sites.
    """
    sites_root = Path(sites_root)
    if not sites_root.is_dir():
        raise FileNotFoundError(f"Sites root not found: {sites_root}")

    rows = []
    for site_dir in sorted(p for p in sites_root.iterdir() if p.is_dir()):
        row = load_site_row(site_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["site_id"])

    df = pd.concat(rows, axis=1).T
    # Ensure site_id is first column
    cols = ["site_id"] + [c for c in df.columns if c != "site_id"]
    return df.reindex(columns=cols)


def _aggregate_by_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metric_PFT columns to single metric (mean across PFTs)."""
    metric_prefixes = ["GPP", "NPP", "$C_{leaf}$", "$C_{stem}$", "$C_{root}$", "$N_{leaf}$", "$N_{stem}$", "$N_{root}$"]
    soil_cols = ["SHLWC", "DEEPC", "MINEC", "AVLN"]
    out = df[["site_id"]].copy()
    for prefix in metric_prefixes:
        cols = [c for c in df.columns if c.startswith(prefix + "_") and not c.startswith(prefix + "_pft")]
        if cols:
            out[prefix] = df[cols].mean(axis=1)
    for c in soil_cols:
        if c in df.columns:
            out[c] = df[c]
    return out


def plot_dataframe(df: pd.DataFrame, outpath: Path) -> None:
    """Plot site x metric heatmap (aggregated by metric)."""
    plot_df = _aggregate_by_metric(df)
    plot_df = plot_df.set_index("site_id")
    plot_df = plot_df.astype(float)

    color_levels = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
    colors = ["darkblue", "lightblue", "#f2f2f2", "orange", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_discrete", colors, N=len(color_levels) - 1)
    norm = BoundaryNorm(color_levels, cmap.N, clip=True)

    fig, ax = plt.subplots(figsize=(12, max(8, len(plot_df) * 0.35)))
    sns.heatmap(plot_df.T, cmap=cmap, norm=norm, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Site × Metric (Normalized Mean - Observed)")
    ax.set_xlabel("Site")
    ax.set_ylabel("Metric")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {outpath}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a combined dataframe from all site CSVs in sites_runs/."
    )
    parser.add_argument(
        "--sites-root",
        default="sites_runs",
        help="Directory containing site folders (default: sites_runs).",
    )
    parser.add_argument(
        "-o", "--output",
        default="analysis_outputs/all_sites_dataframe.csv",
        help="Output CSV path (default: analysis_outputs/all_sites_dataframe.csv).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the dataframe as a heatmap (site × metric).",
    )
    parser.add_argument(
        "--csv",
        help="Path to existing CSV to plot (skips building from sites_runs).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        df = build_sites_dataframe(args.sites_root)
        outpath = Path(args.output)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outpath, index=False)
        print(f"Saved {len(df)} sites to {outpath}")
        print(f"Columns: {list(df.columns)}")

    if args.plot:
        base = Path(args.csv if args.csv else args.output)
        plot_path = base.parent / (base.stem + "_plot.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_dataframe(df, plot_path)
