#!/usr/bin/env python3
"""
Plot all site summaries on one Taylor diagram.

For each site in sites_runs/*:
1) Load sample_matrix.csv, results.csv, targets.csv
2) Select top-N model rows with utils.get_best_match(...)
3) Compare top-N mean model vector vs observed target vector
4) Plot each site as a point in Taylor space

Example:
  python3 plot_taylor_all_sites.py --sites-root sites_runs --outdir analysis_outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils as ut


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Taylor diagram with one point per site."
    )
    parser.add_argument(
        "--sites-root",
        default="sites_runs",
        help="Directory containing site folders (default: sites_runs).",
    )
    parser.add_argument(
        "--outdir",
        default="analysis_outputs",
        help="Directory to write outputs (default: analysis_outputs).",
    )
    parser.add_argument(
        "--output-name",
        default="all_sites_taylor_diagram.png",
        help="Taylor diagram filename (default: all_sites_taylor_diagram.png).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of best runs to average per site (default: 10).",
    )
    parser.add_argument(
        "--rmetric",
        default="r2rmse",
        help="Ranking metric used in get_best_match output (default: r2rmse).",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort ranking metric descending (default is ascending).",
    )
    return parser.parse_args()


def _valid_site_dir(site_dir: Path) -> bool:
    required = ["sample_matrix.csv", "results.csv", "targets.csv"]
    return all((site_dir / fname).exists() for fname in required)


def _site_stats(site_dir: Path, top_n: int, rmetric: str, ascending: bool) -> Tuple[float, float, float]:
    df_param = pd.read_csv(site_dir / "sample_matrix.csv")
    df_model = pd.read_csv(site_dir / "results.csv")
    target_df = pd.read_csv(site_dir / "targets.csv", skiprows=[0])
    df_model = pd.concat([df_model, target_df], ignore_index=True)

    _, ymodel = ut.get_best_match(df_param, df_model)
    y_sort = ymodel.sort_values(by=[rmetric], ascending=ascending).iloc[:top_n, :-6].copy()
    obs = df_model.iloc[-1][y_sort.columns].copy()
    mod = y_sort.mean(axis=0)

    pair = pd.concat([mod.rename("mod"), obs.rename("obs")], axis=1).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(pair) < 2:
        raise ValueError("Not enough valid paired values to compute Taylor stats.")

    mod_vals = pair["mod"].values.astype(float)
    obs_vals = pair["obs"].values.astype(float)

    std_mod = float(np.std(mod_vals, ddof=1))
    std_obs = float(np.std(obs_vals, ddof=1))
    corr = float(np.corrcoef(mod_vals, obs_vals)[0, 1])
    corr = float(np.clip(corr, -1.0, 1.0))

    if std_obs <= 0:
        raise ValueError("Observed standard deviation is zero; cannot normalize.")

    std_ratio = std_mod / std_obs
    crmse = float(
        np.sqrt(std_mod**2 + std_obs**2 - 2.0 * std_mod * std_obs * corr)
    ) / std_obs
    return std_ratio, corr, crmse


def collect_site_points(
    sites_root: Path, top_n: int, rmetric: str, ascending: bool
) -> pd.DataFrame:
    rows: List[dict] = []
    for site_dir in sorted(p for p in sites_root.iterdir() if p.is_dir()):
        if not _valid_site_dir(site_dir):
            continue
        try:
            std_ratio, corr, crmse = _site_stats(site_dir, top_n, rmetric, ascending)
        except Exception as exc:  # Keep pipeline robust across inconsistent site files.
            print(f"Skipping {site_dir.name}: {exc}")
            continue
        rows.append(
            {
                "site": site_dir.name,
                "std_ratio": std_ratio,
                "corrcoef": corr,
                "crmse_norm": crmse,
            }
        )

    if not rows:
        raise RuntimeError(f"No valid site stats could be computed under {sites_root}")

    return pd.DataFrame(rows).sort_values("site").reset_index(drop=True)


def plot_taylor(df: pd.DataFrame, out_png: Path) -> None:
    max_r = max(2.0, float(np.ceil(df["std_ratio"].max() * 10.0) / 10.0))
    theta = np.arccos(np.clip(df["corrcoef"].values, -1.0, 1.0))
    radius = df["std_ratio"].values

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_rlim(0, max_r)

    corr_ticks = np.array([1.0, 0.99, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -0.9, -0.95, -0.99, -1.0])
    corr_ticks = corr_ticks[(corr_ticks >= -1.0) & (corr_ticks <= 1.0)]
    ax.set_thetagrids(np.degrees(np.arccos(corr_ticks)), labels=[f"{c:.2g}" for c in corr_ticks])

    # Reference observation point at std_ratio=1 and corr=1.
    ax.plot([0], [1.0], "k*", ms=12, label="Observation")
    ax.plot(np.linspace(0, np.pi, 300), np.ones(300), "k--", lw=1.0, alpha=0.6)

    # Approximate normalized centered-RMSE contours around reference point.
    t = np.linspace(0, np.pi, 400)
    for e in [0.25, 0.5, 0.75, 1.0, 1.5]:
        rr = np.cos(t) + np.sqrt(np.maximum(0.0, e**2 - np.sin(t) ** 2))
        rr[np.isnan(rr)] = np.nan
        rr[(e**2 - np.sin(t) ** 2) < 0] = np.nan
        rr[(rr < 0) | (rr > max_r)] = np.nan
        ax.plot(t, rr, color="gray", lw=0.8, alpha=0.4)

    sc = ax.scatter(theta, radius, c=df["crmse_norm"], cmap="viridis", s=45, alpha=0.9)
    for th, r, site in zip(theta, radius, df["site"]):
        ax.text(th, r, f" {site}", fontsize=7, ha="left", va="center")

    cbar = plt.colorbar(sc, ax=ax, pad=0.10)
    cbar.set_label("Normalized centered RMSE")

    ax.set_title("Taylor Diagram Across Sites")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Normalized Standard Deviation")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    sites_root = Path(args.sites_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = collect_site_points(
        sites_root=sites_root,
        top_n=args.top_n,
        rmetric=args.rmetric,
        ascending=not args.descending,
    )
    csv_path = outdir / "all_sites_taylor_metrics.csv"
    out_png = outdir / args.output_name
    df.to_csv(csv_path, index=False)
    plot_taylor(df, out_png)

    print(f"Saved: {csv_path}")
    print(f"Saved: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
