# Site Analysis

Scripts for fetching site calibration runs, plotting per-site PFT/soil comparisons, and building manuscript figures.

## Manuscript Figures 3 and 4

To generate **Figure 3** and **Figure 4** for the manuscript, run `stack_site_plots.py` from this directory:

```bash
cd Site_Analysis
python stack_site_plots.py
```

This writes:

- `analysis_outputs/figure3.png` — first 15 sites (5×3 grid, PFT comparison only)
- `analysis_outputs/figure4.png` — remaining sites plus colorbar (5×4 grid, PFT comparison only)

Optional arguments:

```bash
python stack_site_plots.py --sites-root sites_runs --outdir analysis_outputs
```

### Prerequisites

Each site folder under `sites_runs/<SITE>/` must contain a CSV with PFT comparison data (e.g. `sites_runs/CMT20/CMT20.csv`). These are produced by:

```bash
python plot_site.py sites_runs/<SITE>
```

Or for all sites:

```bash
python run_sites_pipeline.py
```

## Manuscript Figure 5

To generate **Figure 5** (combined PFT error MAE bar chart and violin plot), run `analyze_pft_errors.py` from this directory:

```bash
cd Site_Analysis
python analyze_pft_errors.py
```

This writes:

- `analysis_outputs/figure5.png` — 1×2 subplots: MAE by PFT (left) and residual distribution violin plot (right)

Optional arguments:

```bash
python analyze_pft_errors.py --sites-root sites_runs --outdir analysis_outputs
```

The same command also saves `pft_error_summary.csv` and additional diagnostic plots in `analysis_outputs/`. It uses the same per-site CSV files under `sites_runs/` as Figures 3 and 4 (see prerequisites above).

## Manuscript Figure 6

To generate **Figure 6** (combined soil error MAE bar chart and violin plot), run `analyze_soil_errors.py` from this directory:

```bash
cd Site_Analysis
python analyze_soil_errors.py
```

This writes:

- `analysis_outputs/figure6.png` — 1×2 subplots: MAE by soil component (left) and residual distribution violin plot (right)

Optional arguments:

```bash
python analyze_soil_errors.py --sites-root sites_runs --outdir analysis_outputs
```

The same command also saves `soil_error_summary.csv` and additional diagnostic plots in `analysis_outputs/`. It uses the same per-site CSV files under `sites_runs/` as Figures 3 and 4 (see prerequisites above).
