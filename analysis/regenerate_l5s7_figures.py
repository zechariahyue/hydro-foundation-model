#!/usr/bin/env python3
"""
Regenerate figures 4, 5, 6, 7 on the consistent last-5yr / stride-7 (_l5s7) protocol.

Adapted from experiments/analysis/generate_figures.py. Differences:
  * reads the *_l5s7.csv zero-shot results (the consistent-protocol run);
  * does NOT touch figure_3 (the few-shot curve, regenerated separately on _l5y);
  * figure_7 uses the actual evaluated basin counts (Table 4 / Data section),
    total 6,569, not the old 5,454 set.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.patches as mpatches

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
# Fonts enlarged from the original 9-10 pt: reviewers found the figure text
# unreadable at print size. Nothing else about these figures changed.
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14.5
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
FIGURES_DIR = Path(os.environ.get("FIGURES_DIR", "figures"))
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

COLORS = {
    'chronos': '#2E86AB',
    'timesfm': '#A23B72',
    'patchtst': '#F18F01',
    'persistence': '#C73E1D',
    'lstm': '#6A994E'
}

DATASETS = ['CAMELS-US', 'CAMELS-BR', 'CAMELS-CL', 'CAMELS-AUS', 'CAMELS-IND', 'LamaH-CE']
MODELS = ['chronos', 'timesfm', 'patchtst', 'persistence']

# Evaluated basin counts on the consistent five-year window (match Table 4 / Data section).
EVAL_COUNTS = {
    'CAMELS-US': 671, 'CAMELS-BR': 3836, 'CAMELS-CL': 394,
    'CAMELS-AUS': 560, 'CAMELS-IND': 228, 'LamaH-CE': 880,
}
TOTAL_BASINS = sum(EVAL_COUNTS.values())  # 6569


def load_zero_shot_data():
    all_data = []
    for dataset in DATASETS:
        for model in MODELS:
            file = RESULTS_DIR / f"{model}_zero_shot_{dataset}_l5s7.csv"
            if file.exists():
                all_data.append(pd.read_csv(file))
            else:
                print(f"  WARNING missing {file.name}")
    return pd.concat(all_data, ignore_index=True)


NSE_MAX_FOR_SIGMA = 0.90   # sigma = RMSE/sqrt(1-NSE) is ill-conditioned as NSE -> 1


def add_dimensionless_tail(df):
    """Express tail RMSE in standard deviations of observed flow.

    The processed discharge column carries each archive's NATIVE units, so raw tail_RMSE
    magnitudes cannot be compared across datasets. sigma_obs is recovered exactly from
    columns we already have, since NSE = 1 - RMSE^2/sigma_obs^2 implies

        sigma_obs = RMSE / sqrt(1 - NSE)

    and sigma_obs is a property of the OBSERVATIONS, so it is shared by every model scored
    on the same target points. We therefore estimate it per basin as the median over the
    numerically stable model rows and divide the tail errors by it.
    """
    d = df.copy()
    one_minus = 1.0 - d["NSE"]
    est = np.where(np.isfinite(d["NSE"]) & np.isfinite(d["RMSE"]) & (one_minus > 0),
                   d["RMSE"] / np.sqrt(np.clip(one_minus, 1e-12, None)), np.nan)
    d["_sigma_est"] = np.where(d["NSE"] <= NSE_MAX_FOR_SIGMA, est, np.nan)
    sigma = d.groupby(["dataset", "basin_id"])["_sigma_est"].median().rename("sigma_obs")
    d = d.merge(sigma, on=["dataset", "basin_id"], how="left")
    for p in ("95", "99"):
        d[f"rel_tail_{p}"] = np.where(d["sigma_obs"] > 0, d[f"tail_RMSE_{p}"] / d["sigma_obs"], np.nan)
    return d.drop(columns=["_sigma_est"])


print("Loading l5s7 zero-shot data...")
zero_shot_df = load_zero_shot_data()
zero_shot_df = add_dimensionless_tail(zero_shot_df)
print(f"Loaded {len(zero_shot_df)} zero-shot records; total evaluated basins (reported) = {TOTAL_BASINS}")
print(f"  dimensionless tail error available for {int(np.isfinite(zero_shot_df['rel_tail_95']).sum())} rows")


def plot_physical_diagnostics():
    pivot_data = zero_shot_df.groupby(['model', 'dataset'])['FDC_KGE'].median().reset_index()
    pivot_table = pivot_data.pivot(index='model', columns='dataset', values='FDC_KGE')
    pivot_table = pivot_table.reindex([m for m in MODELS if m in pivot_table.index])
    pivot_table = pivot_table[[d for d in DATASETS if d in pivot_table.columns]]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1, cbar_kws={'label': 'FDC KGE'}, ax=ax)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Model')
    ax.set_title('Flow Duration Curve Fidelity (FDC KGE)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_4_physical_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure_4_physical_diagnostics.pdf', bbox_inches='tight')
    print("[OK] figure_4_physical_diagnostics")
    plt.close()


def plot_model_comparison():
    chronos_data = zero_shot_df[zero_shot_df['model'] == 'chronos'][['basin_id', 'dataset', 'NSE']].rename(columns={'NSE': 'chronos_nse'})
    timesfm_data = zero_shot_df[zero_shot_df['model'] == 'timesfm'][['basin_id', 'dataset', 'NSE']].rename(columns={'NSE': 'timesfm_nse'})
    merged = pd.merge(chronos_data, timesfm_data, on=['basin_id', 'dataset'])

    fig, ax = plt.subplots(figsize=(7, 7))
    for dataset in [d for d in DATASETS if d in merged['dataset'].unique()]:
        subset = merged[merged['dataset'] == dataset]
        ax.scatter(subset['chronos_nse'], subset['timesfm_nse'], alpha=0.5, s=20, label=dataset)
    ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Chronos NSE')
    ax.set_ylabel('TimesFM NSE')
    ax.set_title('Model Performance Comparison')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.5, 1])
    ax.set_ylim([-0.5, 1])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_5_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure_5_model_comparison.pdf', bbox_inches='tight')
    print("[OK] figure_5_model_comparison")
    plt.close()


def plot_extreme_events():
    tail_data = []
    for model in MODELS:
        for dataset in DATASETS:
            subset = zero_shot_df[(zero_shot_df['model'] == model) & (zero_shot_df['dataset'] == dataset)]
            if len(subset) > 0:
                tail_data.append({'model': model, 'dataset': dataset,
                                  'tail_95': subset['rel_tail_95'].median(),
                                  'tail_99': subset['rel_tail_99'].median()})
    tail_df = pd.DataFrame(tail_data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for idx, percentile in enumerate(['tail_95', 'tail_99']):
        ax = axes[idx]
        pivot = tail_df.pivot(index='dataset', columns='model', values=percentile)
        pivot = pivot.reindex([d for d in DATASETS if d in pivot.index])
        pivot = pivot[[m for m in MODELS if m in pivot.columns]]
        pivot.plot(kind='bar', ax=ax, color=[COLORS.get(m, 'gray') for m in pivot.columns], legend=False)
        ax.set_ylabel(r'Tail RMSE / $\sigma_{obs}$')
        ax.set_xlabel('Dataset')
        ax.set_title(f'{"95th" if percentile == "tail_95" else "99th"} Percentile Flow Error')
        if idx == 0:                      # single shared legend on the first panel only
            ax.legend(title='Model', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_6_extreme_events.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure_6_extreme_events.pdf', bbox_inches='tight')
    print("[OK] figure_6_extreme_events")
    plt.close()


def plot_geographic_distribution():
    dataset_info = {
        'CAMELS-US': {'lat': 39, 'lon': -98, 'color': '#2E86AB'},
        'CAMELS-BR': {'lat': -15, 'lon': -55, 'color': '#A23B72'},
        'CAMELS-CL': {'lat': -35, 'lon': -71, 'color': '#F18F01'},
        'CAMELS-AUS': {'lat': -25, 'lon': 135, 'color': '#C73E1D'},
        'CAMELS-IND': {'lat': 20, 'lon': 78, 'color': '#6A994E'},
        'LamaH-CE': {'lat': 48, 'lon': 13, 'color': '#9B59B6'},
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Geographic Distribution of CAMELS Datasets ({TOTAL_BASINS:,} basins)')
    ax.grid(True, alpha=0.3)
    for name, info in dataset_info.items():
        basins = EVAL_COUNTS[name]
        size = np.sqrt(basins) * 3
        ax.scatter(info['lon'], info['lat'], s=size**2, alpha=0.6,
                   color=info['color'], edgecolors='black', linewidth=1.5)
        ax.text(info['lon'], info['lat'] - 8, f"{name}\n({basins:,} basins)",
                ha='center', fontsize=8, weight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_7_geographic_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure_7_geographic_distribution.pdf', bbox_inches='tight')
    print("[OK] figure_7_geographic_distribution")
    plt.close()


def plot_spatial_skill_map():
    """P1: centroid map coloured by best-foundation-model median NSE per dataset."""
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    centroids = {
        'CAMELS-US': (39, -98), 'CAMELS-BR': (-15, -55), 'CAMELS-CL': (-35, -71),
        'CAMELS-AUS': (-25, 135), 'CAMELS-IND': (20, 78), 'LamaH-CE': (48, 13),
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('Foundation Model Skill Across Global Basins (best FM, last-5yr protocol)')
    ax.grid(True, alpha=0.3)
    cmap = cm.get_cmap('RdYlBu'); norm = Normalize(vmin=0.3, vmax=0.9)
    for ds, (lat, lon) in centroids.items():
        sub = zero_shot_df[zero_shot_df['dataset'] == ds]
        chro = sub[sub['model'] == 'chronos']['NSE'].median()
        tfm = sub[sub['model'] == 'timesfm']['NSE'].median()
        best = max(chro, tfm)
        n = EVAL_COUNTS[ds]
        size = np.sqrt(n) * 3
        ax.scatter(lon, lat, s=size**2, alpha=0.75,
                   color=cmap(norm(max(0.3, min(0.9, best)))), edgecolors='black', linewidth=1.5)
        ax.text(lon, lat, f"{ds}\nNSE={best:.3f}\nn={n:,}", ha='center', va='center', fontsize=8, weight='bold')
    sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Best foundation-model median NSE')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'P1_spatial_skill_map.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'P1_spatial_skill_map.pdf', bbox_inches='tight')
    print("[OK] P1_spatial_skill_map")
    plt.close()


def plot_tail_error():
    """P2: tail RMSE (P95). Left: distribution by model; right: FM tail RMSE by dataset."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # Left: boxplot of per-basin tail_RMSE_95 by model (log scale, FMs+persistence)
    box_models = ['chronos', 'timesfm', 'patchtst', 'persistence']
    data = [zero_shot_df[zero_shot_df['model'] == m]['rel_tail_95'].dropna().clip(upper=zero_shot_df['rel_tail_95'].quantile(0.95)) for m in box_models]
    bp = ax1.boxplot(data, labels=box_models, patch_artist=True, showfliers=False)
    for patch, m in zip(bp['boxes'], box_models):
        patch.set_facecolor(COLORS.get(m, 'gray')); patch.set_alpha(0.7)
    ax1.set_ylabel(r'Tail RMSE @ P95 / $\sigma_{obs}$'); ax1.set_xlabel('Model')
    ax1.set_title('Extreme-flow error distribution by model')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
    # Right: median FM tail RMSE by dataset
    fm = ['chronos', 'timesfm']
    med = zero_shot_df[zero_shot_df['model'].isin(fm)].groupby(['dataset', 'model'])['rel_tail_95'].median().unstack()
    med = med.reindex([d for d in DATASETS if d in med.index])
    med.plot(kind='bar', ax=ax2, color=[COLORS['chronos'], COLORS['timesfm']])
    ax2.set_ylabel(r'Median tail RMSE @ P95 / $\sigma_{obs}$'); ax2.set_xlabel('Dataset')
    ax2.set_title('Foundation-model extreme-flow error by region')
    ax2.legend(title='Model', fontsize=8); ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'P2_tail_error_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'P2_tail_error_plot.pdf', bbox_inches='tight')
    print("[OK] P2_tail_error_plot")
    plt.close()


def plot_incremental_ablation():
    """P3: pooled median NSE, persistence -> patchtst -> chronos -> timesfm."""
    order = ['persistence', 'patchtst', 'chronos', 'timesfm']
    medians = [zero_shot_df[zero_shot_df['model'] == m]['NSE'].median() for m in order]
    colors = ['#8d6e63', COLORS['patchtst'], COLORS['chronos'], COLORS['timesfm']]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(order)), medians, color=colors, alpha=0.85, edgecolor='black')
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=20, ha='right')
    ax.set_ylabel('Pooled median NSE (all 6,569 basins)')
    ax.set_title('Pooled Median NSE by Model')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01 if val > 0 else val - 0.03,
                f'{val:.3f}', ha='center', fontweight='bold')
    ax.axhline(0, color='gray', lw=1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'P3_incremental_ablation.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'P3_incremental_ablation.pdf', bbox_inches='tight')
    print(f"[OK] P3_incremental_ablation (pooled medians: {dict(zip(order, [round(m,3) for m in medians]))})")
    plt.close()


if __name__ == "__main__":
    plot_physical_diagnostics()
    plot_model_comparison()
    plot_extreme_events()
    plot_geographic_distribution()
    plot_spatial_skill_map()
    plot_tail_error()
    plot_incremental_ablation()
    # quick provenance print
    fdc = zero_shot_df.groupby(['model', 'dataset'])['FDC_KGE'].median().unstack()
    print("\nFDC_KGE medians (sanity):")
    print(fdc.reindex(MODELS)[[d for d in DATASETS]].round(3))
    print("\nDone (figures 4,5,6,7 regenerated on l5s7).")
