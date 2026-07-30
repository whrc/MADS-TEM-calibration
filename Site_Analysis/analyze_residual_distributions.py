#!/usr/bin/env python3
"""
Cross-site residual distribution analysis.

Products generated:
1) Distribution plots across sites
   - residuals by target
   - residuals by PFT x target
2) Grand mean residual heatmap (target x PFT)
3) Variance heatmap (target x PFT)
4) Sign-consistency heatmap (% sites with positive residual)

Residual definition:
  residual = mean(top-N modeled runs) - observation
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils as ut


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze across-site residual distributions and summary maps."
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
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top-N model runs averaged per site (default: 10).",
    )
    parser.add_argument(
        "--rmetric",
        default="r2rmse",
        help="Ranking metric from utils.get_best_match output (default: r2rmse).",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Use descending sort for ranking metric (default: ascending).",
    )
    return parser.parse_args()


def _valid_site_dir(site_dir: Path) -> bool:
    required = ["sample_matrix.csv", "results.csv", "targets.csv"]
    return all((site_dir / fname).exists() for fname in required)


def _feature_metadata(feature: str) -> Tuple[str, str, bool]:
    """
    Returns: (base_target, pft_label, is_pft_specific)
      - VEGC_pft2_Leaf  -> ("VEGC_Leaf", "pft2", True)
      - NPP_pft1        -> ("NPP", "pft1", True)
      - MINEC           -> ("MINEC", "non_pft", False)
    """
    m = re.match(r"^([A-Za-z0-9]+)_pft([0-9]+)(?:_(.+))?$", feature)
    if not m:
        return feature, "non_pft", False

    var, pft_num, comp = m.group(1), m.group(2), m.group(3)
    base_target = f"{var}_{comp}" if comp else var
    return base_target, f"pft{pft_num}", True


def _site_residuals(site_dir: Path, top_n: int, rmetric: str, ascending: bool) -> pd.Series:
    df_param = pd.read_csv(site_dir / "sample_matrix.csv")
    df_model = pd.read_csv(site_dir / "results.csv")
    target_df = pd.read_csv(site_dir / "targets.csv", skiprows=[0])
    df_model = pd.concat([df_model, target_df], ignore_index=True)

    _, ymodel = ut.get_best_match(df_param, df_model)
    y_sorted = ymodel.sort_values(by=[rmetric], ascending=ascending).iloc[:top_n, :-6].copy()
    obs = df_model.iloc[-1][y_sorted.columns]
    modeled_mean = y_sorted.mean(axis=0)
    residual = (modeled_mean - obs).replace([np.inf, -np.inf], np.nan)
    return residual


def build_residual_table(
    sites_root: Path, top_n: int, rmetric: str, ascending: bool
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for site_dir in sorted(p for p in sites_root.iterdir() if p.is_dir()):
        if not _valid_site_dir(site_dir):
            continue
        try:
            residual = _site_residuals(site_dir, top_n, rmetric, ascending)
        except Exception as exc:
            print(f"Skipping {site_dir.name}: {exc}")
            continue

        for feature, value in residual.items():
            base_target, pft_label, is_pft = _feature_metadata(feature)
            rows.append(
                {
                    "site": site_dir.name,
                    "feature": feature,
                    "base_target": base_target,
                    "pft": pft_label,
                    "is_pft_specific": is_pft,
                    "residual": float(value) if pd.notna(value) else np.nan,
                }
            )

    if not rows:
        raise RuntimeError(f"No residuals could be computed from {sites_root}")

    return pd.DataFrame(rows)


def make_distributions(long_df: pd.DataFrame, outdir: Path) -> None:
    sns.set_theme(style="whitegrid")

    pft_df = long_df[long_df["is_pft_specific"]].copy()
    non_pft_df = long_df[~long_df["is_pft_specific"]].copy()

    # Target-level: aggregate across PFTs within each site for pft-specific variables.
    pft_target_agg = (
        pft_df.groupby(["site", "base_target"], as_index=False)["residual"].mean()
        if not pft_df.empty
        else pd.DataFrame(columns=["site", "base_target", "residual"])
    )
    target_df = pd.concat(
        [pft_target_agg, non_pft_df[["site", "base_target", "residual"]]],
        ignore_index=True,
    )

    order_targets = (
        target_df.groupby("base_target")["residual"].median().sort_values().index.tolist()
    )

    plt.figure(figsize=(max(12, 0.6 * len(order_targets)), 6))
    sns.violinplot(
        data=target_df,
        x="base_target",
        y="residual",
        order=order_targets,
        inner="quartile",
        cut=0,
    )
    plt.axhline(0.0, color="k", lw=1, ls="--", alpha=0.7)
    plt.title("Residual Distribution Across Sites by Target")
    plt.xlabel("Target")
    plt.ylabel("Residual (model - observation)")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "residual_violin_by_target.png", dpi=300)
    plt.close()

    plt.figure(figsize=(max(12, 0.6 * len(order_targets)), 6))
    sns.boxplot(data=target_df, x="base_target", y="residual", order=order_targets)
    plt.axhline(0.0, color="k", lw=1, ls="--", alpha=0.7)
    plt.title("Residual Boxplot Across Sites by Target")
    plt.xlabel("Target")
    plt.ylabel("Residual (model - observation)")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "residual_box_by_target.png", dpi=300)
    plt.close()

    # PFT x target-level distributions.
    pft_df["target_pft"] = pft_df["base_target"] + " x " + pft_df["pft"]
    order_tp = pft_df.groupby("target_pft")["residual"].median().sort_values().index.tolist()

    plt.figure(figsize=(10, max(10, 0.33 * len(order_tp))))
    sns.violinplot(
        data=pft_df,
        y="target_pft",
        x="residual",
        order=order_tp,
        inner="quartile",
        cut=0,
        orient="h",
    )
    plt.axvline(0.0, color="k", lw=1, ls="--", alpha=0.7)
    plt.title("Residual Distribution Across Sites by PFT x Target")
    plt.xlabel("Residual (model - observation)")
    plt.ylabel("PFT x Target")
    plt.tight_layout()
    plt.savefig(outdir / "residual_violin_by_pft_target.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, max(10, 0.33 * len(order_tp))))
    sns.boxplot(data=pft_df, y="target_pft", x="residual", order=order_tp, orient="h")
    plt.axvline(0.0, color="k", lw=1, ls="--", alpha=0.7)
    plt.title("Residual Boxplot Across Sites by PFT x Target")
    plt.xlabel("Residual (model - observation)")
    plt.ylabel("PFT x Target")
    plt.tight_layout()
    plt.savefig(outdir / "residual_box_by_pft_target.png", dpi=300)
    plt.close()

    target_df.to_csv(outdir / "residuals_by_target_long.csv", index=False)
    pft_df.to_csv(outdir / "residuals_by_pft_target_long.csv", index=False)


def make_heatmaps(long_df: pd.DataFrame, outdir: Path) -> None:
    pft_df = long_df[long_df["is_pft_specific"]].copy()
    if pft_df.empty:
        raise RuntimeError("No PFT-specific residual columns found; cannot build target x PFT maps.")

    mean_map = pft_df.pivot_table(
        index="base_target", columns="pft", values="residual", aggfunc="mean"
    )
    var_map = pft_df.pivot_table(
        index="base_target", columns="pft", values="residual", aggfunc="var"
    )
    pos_pct_map = pft_df.pivot_table(
        index="base_target",
        columns="pft",
        values="residual",
        aggfunc=lambda x: 100.0 * np.mean(np.array(x) > 0),
    )

    mean_map = mean_map.sort_index()
    var_map = var_map.sort_index()
    pos_pct_map = pos_pct_map.sort_index()

    vmax = np.nanmax(np.abs(mean_map.values))
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0

    plt.figure(figsize=(8, max(6, 0.38 * len(mean_map.index))))
    sns.heatmap(
        mean_map,
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".2g",
        cbar_kws={"label": "Mean residual (model - observation)"},
    )
    plt.title("Grand Mean Residual Heatmap (Target x PFT)")
    plt.xlabel("PFT")
    plt.ylabel("Target")
    plt.tight_layout()
    plt.savefig(outdir / "heatmap_mean_residual_target_x_pft.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, max(6, 0.38 * len(var_map.index))))
    sns.heatmap(
        var_map,
        cmap="magma",
        annot=True,
        fmt=".2g",
        cbar_kws={"label": "Residual variance across sites"},
    )
    plt.title("Residual Variance Heatmap (Target x PFT)")
    plt.xlabel("PFT")
    plt.ylabel("Target")
    plt.tight_layout()
    plt.savefig(outdir / "heatmap_variance_residual_target_x_pft.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, max(6, 0.38 * len(pos_pct_map.index))))
    sns.heatmap(
        pos_pct_map,
        cmap="RdYlBu_r",
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".0f",
        cbar_kws={"label": "% of sites with positive residual"},
    )
    plt.title("Sign-Consistency Heatmap (Target x PFT)")
    plt.xlabel("PFT")
    plt.ylabel("Target")
    plt.tight_layout()
    plt.savefig(outdir / "heatmap_sign_consistency_positive_pct_target_x_pft.png", dpi=300)
    plt.close()

    mean_map.to_csv(outdir / "mean_residual_target_x_pft.csv")
    var_map.to_csv(outdir / "variance_residual_target_x_pft.csv")
    pos_pct_map.to_csv(outdir / "sign_consistency_positive_pct_target_x_pft.csv")


def main() -> int:
    args = parse_args()
    sites_root = Path(args.sites_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    long_df = build_residual_table(
        sites_root=sites_root,
        top_n=args.top_n,
        rmetric=args.rmetric,
        ascending=not args.descending,
    )
    long_df.to_csv(outdir / "residuals_all_features_long.csv", index=False)

    make_distributions(long_df, outdir)
    make_heatmaps(long_df, outdir)

    print(f"Saved analysis products in: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
