from __future__ import annotations

import matplotlib.image as mpimg
from matplotlib.cm import ScalarMappable

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import seaborn as sns
import os
import re
import utils as ut
import site_pfts as sp


def get_by_index(zscore,index_list):
    return pd.Series([zscore[0][iname] for iname in index_list])

def rmse(x,x_true):
    MSE = np.square(np.subtract(x.mean(),x_true)) 
    df_z = pd.DataFrame(np.sqrt(MSE))
    df_z.index = x.columns
    return df_z

def relative_error(x,x_true):
    MAE = np.subtract(x.mean(),x_true)
    df_z = pd.DataFrame(abs(100*MAE/x_true))
    df_z.index = x.columns
    return df_z

def create_custom_colorbar(error_type='rmse'):
    color_levels = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
    colors = ['darkblue', 'lightblue', '#f2f2f2', 'orange', 'red']
    cmap = LinearSegmentedColormap.from_list('custom_discrete', colors, N=len(color_levels) - 1)
    norm = BoundaryNorm(color_levels, cmap.N, clip=True)

    if error_type == 'rmse':
        cbar_label = 'RMSE score'
    elif error_type == 'diff':
        cbar_label = 'Normalized (Mean - Observed)'
    else:
        cbar_label = 'RE score'

    return cmap, norm, cbar_label


def _collect_pft_columns(site: str, df_model: pd.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    """Collect PFT column groups limited to the number of PFTs defined for the site."""
    pfts = set()
    for col in df_model.columns:
        match = re.search(r'pft[0-9]+', col)
        if match:
            pfts.add(match.group())
    pft_list = sp.get_limited_pft_list(site, sorted(pfts))

    def pft_cols(pattern: str, extra: str = "") -> list[str]:
        cols = [
            col for col in df_model.columns
            if pattern in col and (not extra or extra in col)
        ]
        return sp.filter_columns_for_site(site, cols, pft_list)

    col_groups = {
        "gpp": pft_cols("GPP"),
        "npp": pft_cols("NPP"),
        "c_leaf": pft_cols("VEGC", "Leaf"),
        "c_stem": pft_cols("VEGC", "Stem"),
        "c_root": pft_cols("VEGC", "Root"),
        "n_leaf": pft_cols("VEGN", "Leaf"),
        "n_stem": pft_cols("VEGN", "Stem"),
        "n_root": pft_cols("VEGN", "Root"),
    }
    return pft_list, col_groups


def compute_pft_error_matrix(path, params="sample_matrix.csv", model="results.csv", target="targets.csv"):
    """
    Compute the PFT error matrix for a single site. Returns (err_matrix_above, site_name).
    """
    df_param = pd.read_csv(os.path.join(path, params))
    df_model = pd.read_csv(os.path.join(path, model))
    target_df = pd.read_csv(os.path.join(path, target), skiprows=[0])
    df_model = pd.concat([df_model, target_df], ignore_index=True)

    xparams_MD1, ymodel_MD1 = ut.get_best_match(df_param, df_model)
    rmetric = 'r2rmse'
    nelem = 10
    order = True
    y_sort = ymodel_MD1.sort_values(by=[rmetric], ascending=order).iloc[:nelem, :-6].copy()
    df2 = pd.concat([y_sort, df_model.iloc[-1:]], ignore_index=True)

    error = 'diff'
    site = os.path.basename(os.path.normpath(path))
    pft_list, cols = _collect_pft_columns(site, df_model)
    gpp_cols = cols["gpp"]
    npp_cols = cols["npp"]
    c_leaf_cols = cols["c_leaf"]
    c_stem_cols = cols["c_stem"]
    c_root_cols = cols["c_root"]
    n_leaf_cols = cols["n_leaf"]
    n_stem_cols = cols["n_stem"]
    n_root_cols = cols["n_root"]

    n_size = len(df2.iloc[:, 0]) - 1
    df_model_normalized_gpp = df2[gpp_cols] / max(df2[gpp_cols].max())
    df_model_normalized_npp = df2[npp_cols] / max(df2[npp_cols].max())
    c_combined = c_leaf_cols + c_stem_cols + c_root_cols
    df_model_normalized_c_combined = df2[c_combined] / max(df2[c_combined].max())
    n_combined = n_leaf_cols + n_stem_cols + n_root_cols
    df_model_normalized_n_combined = df2[n_combined] / max(df2[n_combined].max())

    rmse_gpp = (df_model_normalized_gpp.iloc[:n_size].mean() - df_model_normalized_gpp.iloc[n_size]).to_frame()
    rmse_npp = (df_model_normalized_npp.iloc[:n_size].mean() - df_model_normalized_npp.iloc[n_size]).to_frame()
    rmse_c = (df_model_normalized_c_combined.iloc[:n_size].mean() - df_model_normalized_c_combined.iloc[n_size]).to_frame()
    rmse_n = (df_model_normalized_n_combined.iloc[:n_size].mean() - df_model_normalized_n_combined.iloc[n_size]).to_frame()

    err_matrix_above = pd.DataFrame({
        'GPP': get_by_index(rmse_gpp, gpp_cols),
        'NPP': get_by_index(rmse_npp, npp_cols),
        '$C_{leaf}$': get_by_index(rmse_c, c_leaf_cols),
        '$C_{stem}$': get_by_index(rmse_c, c_stem_cols),
        '$C_{root}$': get_by_index(rmse_c, c_root_cols),
        '$N_{leaf}$': get_by_index(rmse_n, n_leaf_cols),
        '$N_{stem}$': get_by_index(rmse_n, n_stem_cols),
        '$N_{root}$': get_by_index(rmse_n, n_root_cols)
    })

    pft_display_names = sp.get_pft_names_for_site(site, pft_list)
    err_matrix_above.index = pft_display_names
    err_matrix_above = err_matrix_above.apply(pd.to_numeric, errors='coerce')
    return err_matrix_above.T, site


def plot_two_sites_pft(path1, path2, outpath=None):
    """
    Stack two sites' PFT comparison heatmaps into a single figure.
    """
    err1, name1 = compute_pft_error_matrix(path1)
    err2, name2 = compute_pft_error_matrix(path2)

    cmap, norm, cbar_label = create_custom_colorbar('diff')
    sns.set(font_scale=1.5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    sns.heatmap(err1, cmap=cmap, norm=norm, annot=True, fmt=".3f",
                cbar=False,
                annot_kws={"fontsize": 12},
                ax=axes[0])
    axes[0].set_title(f"PFT Comparison — {name1}")
    axes[0].set_ylabel("")

    sns.heatmap(err2, cmap=cmap, norm=norm, annot=True, fmt=".3f",
                cbar=False,
                annot_kws={"fontsize": 12},
                ax=axes[1])
    axes[1].set_title(f"PFT Comparison — {name2}")
    axes[1].set_ylabel("")

    fig.suptitle(f"{name1} vs {name2}", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cbar_ax, orientation="vertical", label=cbar_label)

    if outpath is None:
        outdir = os.path.join(os.path.dirname(path1), "..", "analysis_outputs")
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, f"{name1}_{name2}_pft.png")
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close("all")
    print(f"Saved: {outpath}")
    return outpath


def plot_srt(path, params="sample_matrix.csv", model="results.csv", target="targets.csv"):
    df_param = pd.read_csv(os.path.join(path, params))
    df_model = pd.read_csv(os.path.join(path, model))
    target_df = pd.read_csv(os.path.join(path, target), skiprows=[0])
    df_model  = pd.concat([df_model, target_df], ignore_index=True)

    xparams_MD1, ymodel_MD1 =  ut.get_best_match(df_param,df_model)
    rmetric='r2rmse'
    nelem=10
    order=True
    y_sort=ymodel_MD1.sort_values(by=[rmetric],ascending=order).iloc[:nelem,:-6].copy()
    df2 = pd.concat([y_sort, df_model.iloc[-1:]], ignore_index=True)

    error='diff'
    site = os.path.basename(os.path.normpath(path))
    pft_list, cols = _collect_pft_columns(site, df_model)
    gpp_cols = cols["gpp"]
    npp_cols = cols["npp"]
    c_leaf_cols = cols["c_leaf"]
    c_stem_cols = cols["c_stem"]
    c_root_cols = cols["c_root"]
    n_leaf_cols = cols["n_leaf"]
    n_stem_cols = cols["n_stem"]
    n_root_cols = cols["n_root"]
    n_soil = [col for col in df_model.columns if col in ['SHLWC', 'DEEPC', 'MINEC', 'AVLN']]

    n_size=len(df2.iloc[:,0])-1
    df_model_normalized = df2.copy() 

    df_model_normalized_gpp = df2[gpp_cols]/ max(df2[gpp_cols].max())

    df_model_normalized_npp = df2[npp_cols]/ max(df2[npp_cols].max())

    c_combined=c_leaf_cols+c_stem_cols+c_root_cols
    df_model_normalized_c_combined = df2[c_combined]/ max(df2[c_combined].max())

    n_combined=n_leaf_cols+n_stem_cols+n_root_cols
    df_model_normalized_n_combined = df2[n_combined]/ max(df2[n_combined].max())

    df_model_normalized_soil = df2[n_soil]/ max(df2[n_soil].max())

    if error=='re':
        rmse_gpp = relative_error(df_model_normalized_gpp.iloc[:n_size],df_model_normalized_gpp.iloc[n_size]) 
        rmse_npp = relative_error(df_model_normalized_npp.iloc[:n_size],df_model_normalized_npp.iloc[n_size]) 
        rmse_c = relative_error(df_model_normalized_c_combined.iloc[:n_size],df_model_normalized_c_combined.iloc[n_size]) 
        rmse_n = relative_error(df_model_normalized_n_combined.iloc[:n_size],df_model_normalized_n_combined.iloc[n_size]) 
        rmse_soil = relative_error(df_model_normalized_soil.iloc[:n_size],df_model_normalized_soil.iloc[n_size]) 
    elif error=='diff':
        rmse_gpp = (df_model_normalized_gpp.iloc[:n_size].mean()-df_model_normalized_gpp.iloc[n_size]).to_frame() 
        rmse_npp = (df_model_normalized_npp.iloc[:n_size].mean()-df_model_normalized_npp.iloc[n_size]).to_frame() 
        rmse_c = (df_model_normalized_c_combined.iloc[:n_size].mean()-df_model_normalized_c_combined.iloc[n_size]).to_frame() 
        rmse_n = (df_model_normalized_n_combined.iloc[:n_size].mean()-df_model_normalized_n_combined.iloc[n_size]).to_frame() 
        rmse_soil = (df_model_normalized_soil.iloc[:n_size].mean()-df_model_normalized_soil.iloc[n_size]).to_frame() 
    else:
        rmse_gpp = rmse(df_model_normalized_gpp.iloc[:n_size],df_model_normalized_gpp.iloc[n_size]) 
        rmse_npp = rmse(df_model_normalized_npp.iloc[:n_size],df_model_normalized_npp.iloc[n_size]) 
        rmse_c = rmse(df_model_normalized_c_combined.iloc[:n_size],df_model_normalized_c_combined.iloc[n_size]) 
        rmse_n = rmse(df_model_normalized_n_combined.iloc[:n_size],df_model_normalized_n_combined.iloc[n_size]) 
        rmse_soil = rmse(df_model_normalized_soil.iloc[:n_size],df_model_normalized_soil.iloc[n_size])

    #build the error matrix
    err_matrix_above = pd.DataFrame({
        'GPP': get_by_index(rmse_gpp, gpp_cols),
        'NPP': get_by_index(rmse_npp, npp_cols),
        '$C_{leaf}$': get_by_index(rmse_c, c_leaf_cols), 
        '$C_{stem}$': get_by_index(rmse_c, c_stem_cols), 
        '$C_{root}$': get_by_index(rmse_c, c_root_cols),
        '$N_{leaf}$': get_by_index(rmse_n, n_leaf_cols), 
        '$N_{stem}$': get_by_index(rmse_n, n_stem_cols), 
        '$N_{root}$': get_by_index(rmse_n, n_root_cols)
            })

    pft_display_names = sp.get_pft_names_for_site(site, pft_list)

    err_matrix_above.index = pft_display_names

    err_matrix_below = pd.DataFrame({
        'Soils': get_by_index(rmse_soil, n_soil)
            })

    err_matrix_below.index=n_soil

    cmap, norm, cbar_label = create_custom_colorbar(error)

    sns.set(font_scale=1.5) 

    fig_combined = plt.figure(figsize=(7, 8))
    gs = fig_combined.add_gridspec(2, 1, height_ratios=[9, 1])
    ax_combined1 = fig_combined.add_subplot(gs[0, 0])
    ax_combined2 = fig_combined.add_subplot(gs[1, 0])
    err_matrix_above = err_matrix_above.apply(pd.to_numeric, errors='coerce')

    err_matrix_above_plot = err_matrix_above.T
    err_matrix_below_plot = err_matrix_below.T

    sns.heatmap(err_matrix_above_plot, cmap=cmap, norm=norm, annot=True, fmt=".3f",
                cbar_kws={'label': cbar_label, "orientation": 'horizontal', 'pad': 0.15},
                cbar=False,
                annot_kws={"fontsize":12},
                ax=ax_combined1)
    ax_combined1.set_title("PFT Comparison")

    sns.heatmap(err_matrix_below_plot, cmap=cmap, norm=norm, annot=True, fmt=".3f",
                cbar_kws={'label': cbar_label, "orientation": 'horizontal', 'pad': 0.35},
                cbar=False,
                annot_kws={"fontsize":12},
                ax=ax_combined2)
    ax_combined2.set_title("Soil Comparison")

    fig_combined.suptitle(path, fontsize=20, y=0.98, fontweight="bold")
    fig_combined.tight_layout()

    combined_path = os.path.join(path, f"{site}.png")
    fig_combined.savefig(combined_path, dpi=300, bbox_inches='tight')

    # Save the exact values shown in both heatmaps to a CSV matching the plot name.
    above_long = err_matrix_above_plot.stack().reset_index()
    above_long.columns = ["metric", "item", "value"]
    above_long["panel"] = "PFT Comparison"

    below_long = err_matrix_below_plot.stack().reset_index()
    below_long.columns = ["metric", "item", "value"]
    below_long["panel"] = "Soil Comparison"

    plotted_values_csv = os.path.join(path, f"{site}.csv")
    pd.concat([above_long, below_long], ignore_index=True)[
        ["panel", "metric", "item", "value"]
    ].to_csv(plotted_values_csv, index=False)
    
    plt.close("all")
    
    return err_matrix_below, err_matrix_above


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot site metrics from a run directory."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Path(s) to site run folder(s). One path: single-site plot. Two paths: stacked PFT-only comparison (e.g. sites_runs/CA_OBS sites_runs/CMT02).",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for two-site stacked plot (default: analysis_outputs/SITE1_SITE2_pft.png).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if len(args.paths) == 1:
        plot_srt(args.paths[0])
    elif len(args.paths) == 2:
        plot_two_sites_pft(args.paths[0], args.paths[1], outpath=args.output)
    else:
        raise SystemExit("Provide 1 path for single-site plot, or 2 paths for stacked PFT comparison.")