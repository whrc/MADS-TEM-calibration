#!/usr/bin/env python3
"""
Analyze cross-site similarities/differences from site run outputs.

Outputs:
1) Clustering dendrogram image
2) PCA scatter image
3) Variable importance/ranking CSV

Example:
  python3 analyze_site_patterns.py --sites-root sites_runs --outdir analysis_outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import utils as ut


def get_by_index(zscore: pd.DataFrame, index_list: Iterable[str]) -> pd.Series:
    return pd.Series([zscore[0][iname] for iname in index_list], dtype=float)


def relative_error(x: pd.DataFrame, x_true: pd.Series) -> pd.DataFrame:
    mae = np.subtract(x.mean(), x_true)
    df_z = pd.DataFrame(abs(100 * mae / x_true))
    df_z.index = x.columns
    return df_z


def rmse(x: pd.DataFrame, x_true: pd.Series) -> pd.DataFrame:
    mse = np.square(np.subtract(x.mean(), x_true))
    df_z = pd.DataFrame(np.sqrt(mse))
    df_z.index = x.columns
    return df_z


def compute_error_matrices(
    path: Path,
    params: str = "sample_matrix.csv",
    model: str = "results.csv",
    target: str = "targets.csv",
    error: str = "diff",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_param = pd.read_csv(path / params)
    df_model = pd.read_csv(path / model)
    target_df = pd.read_csv(path / target, skiprows=[0])
    df_model = pd.concat([df_model, target_df], ignore_index=True)

    _, ymodel_md1 = ut.get_best_match(df_param, df_model)
    rmetric = "r2rmse"
    nelem = 10
    order = True
    y_sort = ymodel_md1.sort_values(by=[rmetric], ascending=order).iloc[:nelem, :-6].copy()
    df2 = pd.concat([y_sort, df_model.iloc[-1:]], ignore_index=True)

    gpp_cols = [col for col in df_model.columns if "GPP" in col]
    npp_cols = [col for col in df_model.columns if "NPP" in col]
    c_leaf_cols = [col for col in df_model.columns if "VEGC" in col and "Leaf" in col]
    c_stem_cols = [col for col in df_model.columns if "VEGC" in col and "Stem" in col]
    c_root_cols = [col for col in df_model.columns if "VEGC" in col and "Root" in col]
    n_leaf_cols = [col for col in df_model.columns if "VEGN" in col and "Leaf" in col]
    n_stem_cols = [col for col in df_model.columns if "VEGN" in col and "Stem" in col]
    n_root_cols = [col for col in df_model.columns if "VEGN" in col and "Root" in col]
    n_soil = [col for col in df_model.columns if col in ["SHLWC", "DEEPC", "MINEC", "AVLN"]]

    n_size = len(df2.iloc[:, 0]) - 1

    df_model_normalized_gpp = df2[gpp_cols] / max(df2[gpp_cols].max())
    df_model_normalized_npp = df2[npp_cols] / max(df2[npp_cols].max())

    c_combined = c_leaf_cols + c_stem_cols + c_root_cols
    df_model_normalized_c_combined = df2[c_combined] / max(df2[c_combined].max())

    n_combined = n_leaf_cols + n_stem_cols + n_root_cols
    df_model_normalized_n_combined = df2[n_combined] / max(df2[n_combined].max())

    df_model_normalized_soil = df2[n_soil] / max(df2[n_soil].max())

    if error == "re":
        rmse_gpp = relative_error(
            df_model_normalized_gpp.iloc[:n_size], df_model_normalized_gpp.iloc[n_size]
        )
        rmse_npp = relative_error(
            df_model_normalized_npp.iloc[:n_size], df_model_normalized_npp.iloc[n_size]
        )
        rmse_c = relative_error(
            df_model_normalized_c_combined.iloc[:n_size],
            df_model_normalized_c_combined.iloc[n_size],
        )
        rmse_n = relative_error(
            df_model_normalized_n_combined.iloc[:n_size],
            df_model_normalized_n_combined.iloc[n_size],
        )
        rmse_soil = relative_error(
            df_model_normalized_soil.iloc[:n_size], df_model_normalized_soil.iloc[n_size]
        )
    elif error == "diff":
        rmse_gpp = (
            df_model_normalized_gpp.iloc[:n_size].mean() - df_model_normalized_gpp.iloc[n_size]
        ).to_frame()
        rmse_npp = (
            df_model_normalized_npp.iloc[:n_size].mean() - df_model_normalized_npp.iloc[n_size]
        ).to_frame()
        rmse_c = (
            df_model_normalized_c_combined.iloc[:n_size].mean()
            - df_model_normalized_c_combined.iloc[n_size]
        ).to_frame()
        rmse_n = (
            df_model_normalized_n_combined.iloc[:n_size].mean()
            - df_model_normalized_n_combined.iloc[n_size]
        ).to_frame()
        rmse_soil = (
            df_model_normalized_soil.iloc[:n_size].mean() - df_model_normalized_soil.iloc[n_size]
        ).to_frame()
    else:
        rmse_gpp = rmse(df_model_normalized_gpp.iloc[:n_size], df_model_normalized_gpp.iloc[n_size])
        rmse_npp = rmse(df_model_normalized_npp.iloc[:n_size], df_model_normalized_npp.iloc[n_size])
        rmse_c = rmse(
            df_model_normalized_c_combined.iloc[:n_size], df_model_normalized_c_combined.iloc[n_size]
        )
        rmse_n = rmse(
            df_model_normalized_n_combined.iloc[:n_size], df_model_normalized_n_combined.iloc[n_size]
        )
        rmse_soil = rmse(
            df_model_normalized_soil.iloc[:n_size], df_model_normalized_soil.iloc[n_size]
        )

    err_matrix_above = pd.DataFrame(
        {
            "GPP": get_by_index(rmse_gpp, gpp_cols),
            "NPP": get_by_index(rmse_npp, npp_cols),
            "$C_{leaf}$": get_by_index(rmse_c, c_leaf_cols),
            "$C_{stem}$": get_by_index(rmse_c, c_stem_cols),
            "$C_{root}$": get_by_index(rmse_c, c_root_cols),
            "$N_{leaf}$": get_by_index(rmse_n, n_leaf_cols),
            "$N_{stem}$": get_by_index(rmse_n, n_stem_cols),
            "$N_{root}$": get_by_index(rmse_n, n_root_cols),
        }
    )

    pft_list = sorted(
        {
            token
            for col in df_model.columns
            for token in [next((part for part in col.split("_") if part.startswith("pft")), "")]
            if token
        }
    )
    if len(pft_list) == len(err_matrix_above.index):
        err_matrix_above.index = pft_list

    err_matrix_below = pd.DataFrame({"Soils": get_by_index(rmse_soil, n_soil)})
    err_matrix_below.index = n_soil

    return err_matrix_below, err_matrix_above


def site_feature_vector(site_path: Path) -> pd.Series:
    soil_df, pft_df = compute_error_matrices(site_path)
    pft_features = pft_df.stack()
    pft_features.index = [f"{idx}::{col}" for idx, col in pft_features.index]

    soil_features = soil_df["Soils"].copy()
    soil_features.index = [f"{idx}::Soils" for idx in soil_features.index]

    return pd.concat([pft_features, soil_features]).astype(float)


def build_matrix(sites_root: Path) -> pd.DataFrame:
    site_dirs = sorted([p for p in sites_root.iterdir() if p.is_dir()])
    rows: Dict[str, pd.Series] = {}
    for site_dir in site_dirs:
        required = ["sample_matrix.csv", "results.csv", "targets.csv"]
        if all((site_dir / fname).exists() for fname in required):
            rows[site_dir.name] = site_feature_vector(site_dir)

    if not rows:
        raise RuntimeError(f"No valid site folders found under {sites_root}")

    matrix = pd.DataFrame(rows).T
    return matrix


def impute_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_means = df.mean(axis=0, skipna=True)
    df_imputed = df.copy()
    for col in df_imputed.columns:
        fill_value = col_means[col] if pd.notna(col_means[col]) else 0.0
        df_imputed[col] = df_imputed[col].fillna(fill_value)
    return df_imputed


def save_dendrogram(matrix_scaled: np.ndarray, site_names: List[str], outpath: Path) -> None:
    z = linkage(matrix_scaled, method="ward")
    plt.figure(figsize=(12, 6))
    dendrogram(z, labels=site_names, leaf_rotation=45, leaf_font_size=9)
    plt.title("Site Similarity Dendrogram")
    plt.ylabel("Ward Distance")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def save_pca_scatter(matrix_scaled: np.ndarray, site_names: List[str], outpath: Path) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2)
    coords = pca.fit_transform(matrix_scaled)

    plt.figure(figsize=(9, 7))
    plt.scatter(coords[:, 0], coords[:, 1], s=50)
    for i, site in enumerate(site_names):
        plt.text(coords[i, 0], coords[i, 1], site, fontsize=8, ha="left", va="bottom")
    plt.title("PCA of Site Error Patterns")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    return pca.components_, pca.explained_variance_ratio_


def variable_ranking(
    matrix_raw: pd.DataFrame,
    components: np.ndarray,
    explained_ratio: np.ndarray,
) -> pd.DataFrame:
    weights = np.abs(components).T @ explained_ratio
    ranking = pd.DataFrame(
        {
            "variable": matrix_raw.columns,
            "importance_pca_weighted_abs_loading": weights,
            "mean_abs_error_across_sites": matrix_raw.abs().mean(axis=0).values,
            "std_error_across_sites": matrix_raw.std(axis=0).values,
        }
    ).sort_values("importance_pca_weighted_abs_loading", ascending=False)
    return ranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read all sites_runs/* outputs, build cross-site matrix, and create "
            "dendrogram + PCA + variable ranking table."
        )
    )
    parser.add_argument(
        "--sites-root",
        default="sites_runs",
        help="Root directory containing one folder per site (default: sites_runs).",
    )
    parser.add_argument(
        "--outdir",
        default="analysis_outputs",
        help="Directory to write analysis outputs (default: analysis_outputs).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sites_root = Path(args.sites_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    matrix_raw = build_matrix(sites_root)
    matrix_raw.to_csv(outdir / "cross_site_matrix.csv")

    matrix_imputed = impute_columns(matrix_raw)
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix_imputed.values)
    site_names = matrix_imputed.index.tolist()

    save_dendrogram(matrix_scaled, site_names, outdir / "site_clustering_dendrogram.png")
    components, explained = save_pca_scatter(matrix_scaled, site_names, outdir / "site_pca_scatter.png")

    ranking = variable_ranking(matrix_raw, components, explained)
    ranking.to_csv(outdir / "variable_importance_ranking.csv", index=False)

    print(f"Saved: {outdir / 'cross_site_matrix.csv'}")
    print(f"Saved: {outdir / 'site_clustering_dendrogram.png'}")
    print(f"Saved: {outdir / 'site_pca_scatter.png'}")
    print(f"Saved: {outdir / 'variable_importance_ranking.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
