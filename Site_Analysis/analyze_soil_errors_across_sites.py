#!/usr/bin/env python3
"""
Analyze soil-error values across site folders from per-site plot CSV files.

Expected per-site file:
  sites_runs/<SITE>/<SITE>.csv

This script reads the 4 soil components from each site CSV:
  SHLWC, DEEPC, MINEC, AVLN

Outputs:
1) soil_error_site_component_matrix.csv
2) soil_error_component_summary.csv
3) soil_error_site_ranking.csv
4) soil_error_site_component_heatmap.png
5) soil_error_component_boxplot.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SOIL_COMPONENTS: List[str] = ["SHLWC", "DEEPC", "MINEC", "AVLN"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read soil error values from sites_runs/<site>/<site>.csv and "
            "analyze them across sites."
        )
    )
    parser.add_argument(
        "--sites-root",
        default="sites_runs",
        help="Root folder containing site subfolders (default: sites_runs).",
    )
    parser.add_argument(
        "--outdir",
        default="analysis_outputs",
        help="Output folder for analysis products (default: analysis_outputs).",
    )
    return parser.parse_args()


def _extract_soil_values(site_csv: Path) -> pd.Series:
    df = pd.read_csv(site_csv)
    required_columns = {"panel", "item", "value"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{site_csv} must contain columns: {sorted(required_columns)}; "
            f"found: {list(df.columns)}"
        )

    soil_df = df[
        (df["panel"] == "Soil Comparison")
        & (df["item"].isin(SOIL_COMPONENTS))
    ][["item", "value"]].copy()

    if soil_df.empty:
        raise ValueError(f"No Soil Comparison rows found in {site_csv}")

    soil_df["value"] = pd.to_numeric(soil_df["value"], errors="coerce")
    # Keep first occurrence if duplicates exist.
    soil_series = soil_df.drop_duplicates(subset=["item"], keep="first").set_index("item")[
        "value"
    ]

    # Reindex to fixed order so every site has the same column order.
    return soil_series.reindex(SOIL_COMPONENTS)


def build_soil_matrix(sites_root: Path) -> pd.DataFrame:
    rows: Dict[str, pd.Series] = {}

    for site_dir in sorted(p for p in sites_root.iterdir() if p.is_dir()):
        site_csv = site_dir / f"{site_dir.name}.csv"
        if not site_csv.exists():
            continue

        try:
            rows[site_dir.name] = _extract_soil_values(site_csv)
        except Exception as exc:
            print(f"Skipping {site_dir.name}: {exc}")
            continue

    if not rows:
        raise RuntimeError(
            f"No valid site CSV files found under {sites_root}. "
            "Expected files like <site>/<site>.csv."
        )

    return pd.DataFrame(rows).T


def component_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "count_non_nan_sites": matrix.notna().sum(axis=0),
            "mean": matrix.mean(axis=0, skipna=True),
            "median": matrix.median(axis=0, skipna=True),
            "std": matrix.std(axis=0, skipna=True),
            "min": matrix.min(axis=0, skipna=True),
            "max": matrix.max(axis=0, skipna=True),
            "mean_abs": matrix.abs().mean(axis=0, skipna=True),
            "positive_pct": 100.0 * (matrix > 0).mean(axis=0, skipna=True),
        }
    )
    return summary.sort_values("mean_abs", ascending=False)


def site_ranking(matrix: pd.DataFrame) -> pd.DataFrame:
    ranking = pd.DataFrame(index=matrix.index)
    ranking["mean_signed_error"] = matrix.mean(axis=1, skipna=True)
    ranking["mean_abs_error"] = matrix.abs().mean(axis=1, skipna=True)
    ranking["max_abs_error"] = matrix.abs().max(axis=1, skipna=True)
    ranking["n_available_components"] = matrix.notna().sum(axis=1)
    ranking = ranking.sort_values("mean_abs_error", ascending=False).reset_index()
    ranking = ranking.rename(columns={"index": "site"})
    return ranking


def save_heatmap(matrix: pd.DataFrame, outpath: Path) -> None:
    sns.set_theme(style="whitegrid")
    plot_df = matrix.copy().sort_index()
    vmax = np.nanmax(np.abs(plot_df.values))
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0

    plt.figure(figsize=(8, max(5, 0.4 * len(plot_df.index))))
    sns.heatmap(
        plot_df,
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Soil error value"},
    )
    plt.title("Soil Error by Site and Component")
    plt.xlabel("Soil component")
    plt.ylabel("Site")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def save_component_boxplot(matrix: pd.DataFrame, outpath: Path) -> None:
    sns.set_theme(style="whitegrid")
    long_df = matrix.reset_index().melt(
        id_vars="index", var_name="component", value_name="value"
    )
    long_df = long_df.rename(columns={"index": "site"}).dropna(subset=["value"])

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=long_df, x="component", y="value", order=SOIL_COMPONENTS)
    sns.stripplot(
        data=long_df,
        x="component",
        y="value",
        order=SOIL_COMPONENTS,
        color="black",
        alpha=0.5,
        size=4,
    )
    plt.axhline(0.0, color="k", lw=1, ls="--", alpha=0.7)
    plt.title("Distribution of Soil Errors Across Sites")
    plt.xlabel("Soil component")
    plt.ylabel("Soil error value")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def main() -> int:
    args = parse_args()
    sites_root = Path(args.sites_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    soil_matrix = build_soil_matrix(sites_root)
    soil_matrix = soil_matrix.reindex(columns=SOIL_COMPONENTS)

    component_stats = component_summary(soil_matrix)
    ranking = site_ranking(soil_matrix)

    matrix_path = outdir / "soil_error_site_component_matrix.csv"
    summary_path = outdir / "soil_error_component_summary.csv"
    ranking_path = outdir / "soil_error_site_ranking.csv"
    heatmap_path = outdir / "soil_error_site_component_heatmap.png"
    boxplot_path = outdir / "soil_error_component_boxplot.png"

    soil_matrix.to_csv(matrix_path)
    component_stats.to_csv(summary_path)
    ranking.to_csv(ranking_path, index=False)
    save_heatmap(soil_matrix, heatmap_path)
    save_component_boxplot(soil_matrix, boxplot_path)

    print(f"Saved: {matrix_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {ranking_path}")
    print(f"Saved: {heatmap_path}")
    print(f"Saved: {boxplot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
