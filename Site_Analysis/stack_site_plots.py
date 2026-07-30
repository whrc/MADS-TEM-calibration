#!/usr/bin/env python3
"""
Stack all site PFT comparison plots from sites_runs into two figures.
Figure 3: 5 cols × 3 rows = 15 sites
Figure 4: 5 cols × 4 rows = remaining sites + colorbar in last slot

PFT heatmaps are rendered from each site's CSV (panel == "PFT Comparison"), so soil
subplots are excluded entirely.

Also generates a single soil heatmap as figure4.1.png: rows=site_ids, columns=SHLWC,DEEPC,MINEC,AVLN.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable

from plot_site import create_custom_colorbar

SOIL_COMPONENTS = ["SHLWC", "DEEPC", "MINEC", "AVLN"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stack site PFT comparison plots into two figures (5×3 and 5×4)."
    )
    parser.add_argument(
        "--sites-root",
        default="sites_runs",
        help="Directory containing site folders (default: sites_runs).",
    )
    parser.add_argument(
        "--outdir",
        default="analysis_outputs",
        help="Output directory for the stacked plots.",
    )
    return parser.parse_args()


def _load_soil_matrix(sites_root: Path) -> pd.DataFrame:
    """Load soil error values from each site CSV. Rows=sites, cols=SHLWC,DEEPC,MINEC,AVLN."""
    rows = {}
    for site_dir in sorted(p for p in sites_root.iterdir() if p.is_dir()):
        csv_path = site_dir / f"{site_dir.name}.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        soil = df[(df["panel"] == "Soil Comparison") & (df["item"].isin(SOIL_COMPONENTS))]
        if soil.empty:
            continue
        vals = soil.set_index("item")["value"].reindex(SOIL_COMPONENTS)
        rows[site_dir.name] = vals
    return pd.DataFrame(rows).T


def _load_pft_matrix(csv_path: Path) -> pd.DataFrame | None:
    """Load PFT error matrix from site CSV. Rows=metrics, cols=PFT names."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    pft = df[df["panel"] == "PFT Comparison"]
    if pft.empty:
        return None
    matrix = pft.pivot(index="metric", columns="item", values="value")
    return matrix.apply(pd.to_numeric, errors="coerce")


def _plot_pft_on_ax(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    site_name: str,
    *,
    show_y_labels: bool = False,
) -> None:
    """Render a single-site PFT heatmap on the given axis."""
    cmap, norm, _ = create_custom_colorbar("diff")
    sns.heatmap(
        matrix,
        cmap=cmap,
        norm=norm,
        annot=True,
        fmt=".3f",
        cbar=False,
        annot_kws={"fontsize": 6},
        ax=ax,
    )
    ax.set_title(site_name, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    if show_y_labels:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontsize=12)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)


def _add_colorbar(fig, cax: plt.Axes) -> None:
    """Add colorbar matching plot_site.py colormap."""
    cmap, norm, cbar_label = create_custom_colorbar("diff")
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation="vertical", label=cbar_label)
    plt.tight_layout()
    pos = cax.get_position()
    # Use right 12.5% of slot (half of previous 25% width)
    cax.set_position([pos.x0 + 0.875 * pos.width, pos.y0, 0.125 * pos.width, pos.height])


def main() -> int:
    args = parse_args()
    sites_root = Path(args.sites_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Collect sites with PFT CSV data, sorted by site name
    site_entries: list[tuple[str, pd.DataFrame]] = []
    for site_dir in sorted(p for p in sites_root.iterdir() if p.is_dir()):
        csv_path = site_dir / f"{site_dir.name}.csv"
        matrix = _load_pft_matrix(csv_path)
        if matrix is not None:
            site_entries.append((site_dir.name, matrix))

    n_first = 15
    first_batch = site_entries[:n_first]
    second_batch = site_entries[n_first:]

    n_cols = 5

    # Figure 3: 5 cols × 3 rows, 15 sites (PFT only)
    fig1, axes1 = plt.subplots(3, n_cols, figsize=(4 * n_cols, 4 * 3))
    axes1 = axes1.flatten()
    for i, ax in enumerate(axes1):
        if i < len(first_batch):
            site_name, matrix = first_batch[i]
            _plot_pft_on_ax(ax, matrix, site_name, show_y_labels=(i % n_cols == 0))
        else:
            ax.set_axis_off()
    plt.suptitle("Site Analysis Plots (1/2)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / "figure3.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 4: 5 cols × 4 rows, remaining sites + colorbar in last slot
    n_rows2 = 4
    fig2, axes2 = plt.subplots(n_rows2, n_cols, figsize=(4 * n_cols, 4 * n_rows2))
    axes2 = axes2.flatten()
    for i, ax in enumerate(axes2):
        if i < len(second_batch):
            site_name, matrix = second_batch[i]
            _plot_pft_on_ax(ax, matrix, site_name, show_y_labels=(i % n_cols == 0))
        else:
            ax.set_visible(False)

    plt.suptitle("Site Analysis Plots (2/2)", fontsize=14, y=1.02)
    # Colorbar one subplot left of last empty slot
    cax = axes2[-2]
    cax.set_visible(True)
    cax.set_axis_on()
    _add_colorbar(fig2, cax)
    plt.savefig(outdir / "figure4.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Single soil heatmap: rows=site_ids, columns=SHLWC,DEEPC,MINEC,AVLN
    soil_matrix = _load_soil_matrix(sites_root)
    if not soil_matrix.empty:
        cmap, norm, _ = create_custom_colorbar("diff")
        mat = soil_matrix.reindex(columns=SOIL_COMPONENTS)
        fig_s, ax_s = plt.subplots(figsize=(6, max(4, 0.25 * len(mat))))
        sns.heatmap(mat, cmap=cmap, norm=norm, annot=True, fmt=".3f")
        ax_s.set_title("Soil Error by Site")
        ax_s.set_ylabel("Site")
        ax_s.set_xlabel("Soil Component")
        plt.tight_layout()
        plt.savefig(outdir / "figure4.1.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Figure 3: {len(first_batch)} sites (PFT only) -> {outdir / 'figure3.png'}")
    print(f"Figure 4: {len(second_batch)} sites + colorbar (PFT only) -> {outdir / 'figure4.png'}")
    if not soil_matrix.empty:
        print(f"Figure 4.1: {len(soil_matrix)} sites -> {outdir / 'figure4.1.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
