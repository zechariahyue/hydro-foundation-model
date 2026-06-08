#!/usr/bin/env python3
"""
Generate publication-quality figures from actual experimental data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Paths
RESULTS_DIR = Path("../results")
FIGURES_DIR = Path("../../paper_material/figures")
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# Color scheme
COLORS = {
    'chronos': '#2E86AB',
    'timesfm': '#A23B72',
    'patchtst': '#F18F01',
    'persistence': '#C73E1D',
    'lstm': '#6A994E'
}

def load_zero_shot_data():
    """Load all zero-shot results"""
    datasets = ['CAMELS-US', 'CAMELS-BR', 'CAMELS-CL', 'CAMELS-AUS', 'CAMELS-IND', 'LamaH-CE']
    models = ['chronos', 'timesfm', 'patchtst', 'persistence']

    all_data = []
    for dataset in datasets:
        for model in models:
            file = RESULTS_DIR / f"{model}_zero_shot_{dataset}.csv"
            if file.exists():
                df = pd.read_csv(file)
                all_data.append(df)

    return pd.concat(all_data, ignore_index=True)

def load_few_shot_data():
    """Load few-shot results"""
    fractions = [0.01, 0.05, 0.1, 0.25]
    datasets_fs = ['CAMELS-AUS', 'CAMELS-BR', 'CAMELS-IND']

    all_data = []
    for dataset in datasets_fs:
        for frac in fractions:
            file = RESULTS_DIR / f"chronos_few_shot_{frac}_{dataset}.csv"
            if file.exists():
                df = pd.read_csv(file)
                df['fraction'] = frac
                all_data.append(df)

    return pd.concat(all_data, ignore_index=True) if all_data else None

print("Loading data...")
zero_shot_df = load_zero_shot_data()
few_shot_df = load_few_shot_data()

print(f"Loaded {len(zero_shot_df)} zero-shot records")
if few_shot_df is not None:
    print(f"Loaded {len(few_shot_df)} few-shot records")

# ============================================================================
# FIGURE 3: Few-shot learning curves
# ============================================================================
def plot_few_shot_curves():
    """Few-shot learning curves for CAMELS-AUS, CAMELS-BR, and CAMELS-IND"""
    if few_shot_df is None:
        print("No few-shot data available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for idx, dataset in enumerate(['CAMELS-AUS', 'CAMELS-BR', 'CAMELS-IND']):
        ax = axes[idx]

        # Get zero-shot baseline
        zero_shot = zero_shot_df[(zero_shot_df['dataset'] == dataset) &
                                  (zero_shot_df['model'] == 'chronos')]
        baseline_nse = zero_shot['NSE'].median()

        # Get few-shot results
        fs_data = few_shot_df[few_shot_df['dataset'] == dataset]

        # Calculate median NSE for each fraction
        fractions = sorted(fs_data['fraction'].unique())
        medians = [fs_data[fs_data['fraction'] == f]['NSE'].median() for f in fractions]

        # Plot
        ax.plot([0] + fractions, [baseline_nse] + medians,
                marker='o', linewidth=2, markersize=8, color=COLORS['chronos'])
        ax.axhline(baseline_nse, linestyle='--', color='gray', alpha=0.5, label='Zero-shot')

        ax.set_xlabel('Training Fraction')
        ax.set_ylabel('Median NSE')
        ax.set_title(f'{dataset}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    axes[0].legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_3_few_shot_curves.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved figure_3_few_shot_curves.png")
    plt.close()

# ============================================================================
# FIGURE 4: Physical diagnostics heatmap
# ============================================================================
def plot_physical_diagnostics():
    """Heatmap of FDC_KGE across models and datasets"""
    # Calculate median FDC_KGE for each model-dataset combination
    pivot_data = zero_shot_df.groupby(['model', 'dataset'])['FDC_KGE'].median().reset_index()
    pivot_table = pivot_data.pivot(index='model', columns='dataset', values='FDC_KGE')

    # Reorder models
    model_order = ['chronos', 'timesfm', 'patchtst', 'persistence']
    pivot_table = pivot_table.reindex([m for m in model_order if m in pivot_table.index])

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1, cbar_kws={'label': 'FDC KGE'}, ax=ax)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Model')
    ax.set_title('Flow Duration Curve Fidelity (FDC KGE)')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_4_physical_diagnostics.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved figure_4_physical_diagnostics.png")
    plt.close()

plot_few_shot_curves()
plot_physical_diagnostics()

# ============================================================================
# FIGURE 5: Model comparison scatter plot
# ============================================================================
def plot_model_comparison():
    """Scatter plot comparing TimesFM vs Chronos NSE"""
    # Get paired data
    chronos_data = zero_shot_df[zero_shot_df['model'] == 'chronos'][['basin_id', 'dataset', 'NSE']].rename(columns={'NSE': 'chronos_nse'})
    timesfm_data = zero_shot_df[zero_shot_df['model'] == 'timesfm'][['basin_id', 'dataset', 'NSE']].rename(columns={'NSE': 'timesfm_nse'})

    merged = pd.merge(chronos_data, timesfm_data, on=['basin_id', 'dataset'])

    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot by dataset with different colors
    datasets = merged['dataset'].unique()
    for dataset in datasets:
        subset = merged[merged['dataset'] == dataset]
        ax.scatter(subset['chronos_nse'], subset['timesfm_nse'],
                  alpha=0.5, s=20, label=dataset)

    # Add diagonal line
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
    print("[OK] Saved figure_5_model_comparison.png")
    plt.close()

# ============================================================================
# FIGURE 6: Extreme event performance
# ============================================================================
def plot_extreme_events():
    """Tail RMSE comparison for extreme events"""
    models = ['chronos', 'timesfm', 'patchtst', 'persistence']
    datasets = zero_shot_df['dataset'].unique()

    # Calculate median tail RMSE for each model-dataset
    tail_data = []
    for model in models:
        for dataset in datasets:
            subset = zero_shot_df[(zero_shot_df['model'] == model) &
                                 (zero_shot_df['dataset'] == dataset)]
            if len(subset) > 0:
                tail_95 = subset['tail_RMSE_95'].median()
                tail_99 = subset['tail_RMSE_99'].median()
                tail_data.append({'model': model, 'dataset': dataset,
                                'tail_95': tail_95, 'tail_99': tail_99})

    tail_df = pd.DataFrame(tail_data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for idx, percentile in enumerate(['tail_95', 'tail_99']):
        ax = axes[idx]
        pivot = tail_df.pivot(index='dataset', columns='model', values=percentile)
        pivot = pivot[[m for m in models if m in pivot.columns]]

        pivot.plot(kind='bar', ax=ax, color=[COLORS.get(m, 'gray') for m in pivot.columns])
        ax.set_ylabel('Tail RMSE (m³/s)')
        ax.set_xlabel('Dataset')
        ax.set_title(f'{"95th" if percentile == "tail_95" else "99th"} Percentile Flow Error')
        ax.legend(title='Model', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_6_extreme_events.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved figure_6_extreme_events.png")
    plt.close()

plot_model_comparison()
plot_extreme_events()

# ============================================================================
# FIGURE 7: Geographic distribution of datasets
# ============================================================================
def plot_geographic_distribution():
    """Map showing geographic distribution of CAMELS datasets"""
    import matplotlib.patches as mpatches

    # Dataset locations (approximate centroids)
    dataset_info = {
        'CAMELS-US': {'lat': 39, 'lon': -98, 'basins': 673, 'color': '#2E86AB'},
        'CAMELS-BR': {'lat': -15, 'lon': -55, 'basins': 2670, 'color': '#A23B72'},
        'CAMELS-CL': {'lat': -35, 'lon': -71, 'basins': 430, 'color': '#F18F01'},
        'CAMELS-AUS': {'lat': -25, 'lon': 135, 'basins': 560, 'color': '#C73E1D'},
        'CAMELS-IND': {'lat': 20, 'lon': 78, 'basins': 241, 'color': '#6A994E'},
        'LamaH-CE': {'lat': 48, 'lon': 13, 'basins': 880, 'color': '#9B59B6'}
    }

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': 'robinson'} if False else {})

    # Simple world map outline
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 75)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Geographic Distribution of CAMELS Datasets (5,454 basins)')
    ax.grid(True, alpha=0.3)

    # Plot dataset locations
    for name, info in dataset_info.items():
        size = np.sqrt(info['basins']) * 3
        ax.scatter(info['lon'], info['lat'], s=size**2, alpha=0.6,
                  color=info['color'], edgecolors='black', linewidth=1.5)
        ax.text(info['lon'], info['lat']-8, f"{name}\n({info['basins']} basins)",
               ha='center', fontsize=8, weight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_7_geographic_distribution.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved figure_7_geographic_distribution.png")
    plt.close()

plot_geographic_distribution()

print("\n" + "="*60)
print("All figures generated successfully!")
print("="*60)
