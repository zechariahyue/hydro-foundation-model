#!/usr/bin/env python3
"""
Step 8: Generate publication figures from real experimental results.
Reads CSV results from experiments/results/ and produces figures in figures/.

Usage:
    python generate_figures.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from glob import glob

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
FIG_DIR = Path(os.environ.get("FIGURES_DIR", "figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

MODEL_COLORS = {
    "timesfm": "#1a6faf",
    "chronos": "#e05c2a",
    "lstm": "#2ca02c",
    "convlstm": "#9467bd",
    "persistence": "#8c564b",
    "patchtst": "#d62728",
    "envgpt": "#e377c2",
    "lstm_transfer": "#17becf",
}

DATASET_COLORS = {
    "CAMELS-US": "#1a6faf",
    "CAMELS-BR": "#e05c2a",
    "CAMELS-CL": "#2ca02c",
    "CAMELS-AUS": "#9467bd",
    "LamaH-CE": "#8c564b",
}


def load_all_results():
    """Load all result CSVs into one DataFrame."""
    csvs = list(RESULTS_DIR.glob("*.csv"))
    if not csvs:
        print("No result CSVs found in", RESULTS_DIR)
        return pd.DataFrame()

    dfs = []
    for f in csvs:
        if "training_history" in f.name or "shap" in f.name or "causal" in f.name:
            continue
        try:
            df = pd.read_csv(f)
            if "model" in df.columns and "NSE" in df.columns:
                dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} basin-model results from {len(dfs)} files.")
    return combined


def fig_model_performance_comparison(results):
    """Figure 1: Model performance comparison across all datasets."""
    if results.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, metric in enumerate(["NSE", "KGE", "RMSE"]):
        if metric not in results.columns:
            continue
        ax = axes[i]

        # Aggregate by model
        model_stats = results.groupby("model")[metric].agg(["median", "mean", "std", "count"])
        model_stats = model_stats.sort_values("median", ascending=(metric == "RMSE"))

        colors = [MODEL_COLORS.get(m, "#999999") for m in model_stats.index]
        bars = ax.barh(range(len(model_stats)), model_stats["median"],
                       xerr=model_stats["std"], color=colors, alpha=0.85,
                       edgecolor="black", linewidth=0.5, capsize=3)

        ax.set_yticks(range(len(model_stats)))
        ax.set_yticklabels(model_stats.index)
        ax.set_xlabel(metric)
        ax.set_title(f"Median {metric} by Model")
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for j, (_, row) in enumerate(model_stats.iterrows()):
            ax.text(row["median"] + 0.01, j, f'{row["median"]:.3f}',
                    va="center", fontsize=9)

    plt.suptitle("Model Performance Comparison Across All Basins", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_1_model_performance_comparison.png")
    plt.close()
    print("  Saved figure_1_model_performance_comparison.png")


def fig_spatial_skill_map(results):
    """Figure P1: Spatial skill map — NSE by dataset/region."""
    if results.empty or "dataset" not in results.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    # Region coordinates (approximate centroids)
    region_coords = {
        "CAMELS-US": (-100, 40),
        "CAMELS-BR": (-50, -15),
        "CAMELS-CL": (-70, -35),
        "CAMELS-AUS": (135, -25),
        "LamaH-CE": (15, 48),
        "CAMELS-IND": (80, 22),
    }

    ax.set_facecolor("#d0e8f5")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)

    # Continent patches
    for (x, y, w, h) in [(-130, 15, 80, 60), (-90, -55, 60, 70), (-20, 35, 50, 35),
                          (-20, -35, 75, 70), (60, 10, 90, 50), (110, -45, 55, 45)]:
        ax.add_patch(plt.Rectangle((x, y), w, h, color="#c8d8a0", zorder=1, alpha=0.7))

    cmap = plt.cm.RdYlBu
    norm = plt.Normalize(0.3, 0.9)

    # Best foundation model per dataset
    fm_models = ["timesfm", "chronos"]
    fm_results = results[results["model"].isin(fm_models)]

    for ds_name, (lon, lat) in region_coords.items():
        ds_data = fm_results[fm_results["dataset"] == ds_name]
        if ds_data.empty:
            continue

        median_nse = ds_data["NSE"].median()
        n_basins = ds_data["basin_id"].nunique()

        color = cmap(norm(max(0.3, min(0.9, median_nse))))
        is_source = ds_name == "CAMELS-US"
        lw = 3 if is_source else 1.5
        ls = "-" if is_source else "--"

        circle = plt.Circle((lon, lat), 12, color=color, zorder=3,
                             linewidth=lw, linestyle=ls,
                             edgecolor="black" if is_source else "#555")
        ax.add_patch(circle)
        ax.text(lon, lat, f"{ds_name}\nNSE={median_nse:.3f}\nn={n_basins}",
                ha="center", va="center", fontsize=8, fontweight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Median NSE", fontsize=11)

    src_patch = mpatches.Patch(edgecolor="black", facecolor="grey", linewidth=2,
                               label="Source region")
    tgt_patch = mpatches.Patch(edgecolor="#555", facecolor="grey", linewidth=1.5,
                               linestyle="--", label="Target region")
    ax.legend(handles=[src_patch, tgt_patch], loc="lower left", fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Foundation Model Skill Across Global Basins", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "P1_spatial_skill_map.png")
    plt.close()
    print("  Saved P1_spatial_skill_map.png")


def fig_tail_error(results):
    """Figure P2: Tail RMSE comparison (extreme events)."""
    if results.empty or "tail_RMSE_95" not in results.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Tail RMSE by model
    ax = axes[0]
    models = results["model"].unique()
    data_for_plot = []
    for m in models:
        vals = results[results["model"] == m]["tail_RMSE_95"].dropna()
        for v in vals:
            data_for_plot.append({"model": m, "tail_RMSE_95": v})

    if data_for_plot:
        plot_df = pd.DataFrame(data_for_plot)
        palette = {m: MODEL_COLORS.get(m, "#999") for m in models}
        sns.boxplot(data=plot_df, x="model", y="tail_RMSE_95", palette=palette, ax=ax)
        ax.set_xlabel("Model")
        ax.set_ylabel("Tail RMSE (95th percentile)")
        ax.set_title("A) Extreme Event Error by Model")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

    # Panel B: Tail RMSE by dataset
    ax = axes[1]
    if "dataset" in results.columns:
        data_for_plot2 = []
        fm_results = results[results["model"].isin(["timesfm", "chronos"])]
        for ds in fm_results["dataset"].unique():
            vals = fm_results[fm_results["dataset"] == ds]["tail_RMSE_95"].dropna()
            for v in vals:
                data_for_plot2.append({"dataset": ds, "tail_RMSE_95": v})

        if data_for_plot2:
            plot_df2 = pd.DataFrame(data_for_plot2)
            palette2 = {d: DATASET_COLORS.get(d, "#999") for d in plot_df2["dataset"].unique()}
            sns.boxplot(data=plot_df2, x="dataset", y="tail_RMSE_95", palette=palette2, ax=ax)
            ax.set_xlabel("Dataset")
            ax.set_ylabel("Tail RMSE (95th percentile)")
            ax.set_title("B) Foundation Model Tail Error by Region")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Extreme Event Performance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "P2_tail_error_plot.png")
    plt.close()
    print("  Saved P2_tail_error_plot.png")


def fig_ablation(results):
    """Figure P3: Incremental ablation study."""
    if results.empty:
        return

    # Compare: persistence -> LSTM -> ConvLSTM -> PatchTST -> Chronos -> TimesFM
    model_order = ["persistence", "patchtst", "chronos", "timesfm"]
    available = [m for m in model_order if m in results["model"].unique()]

    if len(available) < 2:
        print("  Not enough models for ablation figure.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    medians = []
    for m in available:
        med = results[results["model"] == m]["NSE"].median()
        medians.append(med)

    colors = [MODEL_COLORS.get(m, "#999") for m in available]
    bars = ax.bar(range(len(available)), medians, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.5)

    # Add improvement arrows
    for i in range(1, len(medians)):
        if not np.isnan(medians[i]) and not np.isnan(medians[i - 1]):
            diff = medians[i] - medians[i - 1]
            ax.annotate(f"+{diff:.3f}" if diff > 0 else f"{diff:.3f}",
                        xy=(i, medians[i]), xytext=(i - 0.5, medians[i] + 0.02),
                        fontsize=9, color="darkgreen" if diff > 0 else "red",
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(available, rotation=30, ha="right")
    ax.set_ylabel("Median NSE")
    ax.set_title("Incremental Model Ablation", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Value labels
    for bar, val in zip(bars, medians):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "P3_incremental_ablation.png")
    plt.close()
    print("  Saved P3_incremental_ablation.png")


def fig_calibration(results):
    """Figure P4: Uncertainty calibration plot (if CRPS available)."""
    if results.empty or "CRPS" not in results.columns:
        print("  No CRPS data for calibration plot.")
        return

    crps_data = results[results["CRPS"].notna()]
    if crps_data.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: CRPS by model
    ax = axes[0]
    models_with_crps = crps_data["model"].unique()
    crps_by_model = []
    for m in models_with_crps:
        vals = crps_data[crps_data["model"] == m]["CRPS"].dropna()
        crps_by_model.append({"model": m, "median_CRPS": vals.median(), "mean_CRPS": vals.mean()})

    if crps_by_model:
        cdf = pd.DataFrame(crps_by_model).sort_values("median_CRPS")
        colors = [MODEL_COLORS.get(m, "#999") for m in cdf["model"]]
        ax.barh(range(len(cdf)), cdf["median_CRPS"], color=colors, alpha=0.85,
                edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(cdf)))
        ax.set_yticklabels(cdf["model"])
        ax.set_xlabel("Median CRPS (lower is better)")
        ax.set_title("A) Probabilistic Forecast Quality")
        ax.grid(True, alpha=0.3, axis="x")

    # Panel B: CRPS vs NSE scatter
    ax = axes[1]
    for m in models_with_crps:
        mdata = crps_data[crps_data["model"] == m]
        ax.scatter(mdata["NSE"], mdata["CRPS"], alpha=0.3, s=20,
                   color=MODEL_COLORS.get(m, "#999"), label=m)
    ax.set_xlabel("NSE")
    ax.set_ylabel("CRPS")
    ax.set_title("B) Deterministic vs Probabilistic Skill")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Uncertainty Calibration", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "P4_calibration_plot.png")
    plt.close()
    print("  Saved P4_calibration_plot.png")


def fig_shap_summary():
    """Figure P5: SHAP feature importance summary."""
    shap_file = RESULTS_DIR / "shap_importance_lstm.csv"
    if not shap_file.exists():
        print("  No SHAP results found.")
        return

    imp = pd.read_csv(shap_file)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp)))
    bars = ax.barh(range(len(imp)), imp["mean_abs_shap"], color=colors,
                   edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp["feature"])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance for Streamflow Prediction", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "P5_shap_summary.png")
    plt.close()
    print("  Saved P5_shap_summary.png")


def fig_causal_graph():
    """Figure: Causal discovery results."""
    causal_file = RESULTS_DIR / "causal_summary.csv"
    if not causal_file.exists():
        print("  No causal discovery results found.")
        return

    causal = pd.read_csv(causal_file)
    lag1 = causal[causal["lag"] == 1].sort_values("mean_abs_corr", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1a6faf" if c > 0 else "#e05c2a" for c in lag1["mean_corr"]]
    ax.barh(range(len(lag1)), lag1["mean_abs_corr"], color=colors, alpha=0.85,
            edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(lag1)))
    ax.set_yticklabels([f"{r['cause']} (r={r['mean_corr']:.2f})" for _, r in lag1.iterrows()])
    ax.set_xlabel("Mean |Correlation| with Streamflow (lag-1)")
    ax.set_title("Causal Drivers of Streamflow (Lag-1 Analysis)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    blue_patch = mpatches.Patch(color="#1a6faf", label="Positive correlation")
    red_patch = mpatches.Patch(color="#e05c2a", label="Negative correlation")
    ax.legend(handles=[blue_patch, red_patch], loc="lower right")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "causal_graph.png")
    plt.close()
    print("  Saved causal_graph.png")


def fig_few_shot_recovery():
    """Figure: Few-shot skill recovery curve."""
    few_shot_files = list(RESULTS_DIR.glob("*few_shot*.csv"))
    if not few_shot_files:
        print("  No few-shot results found.")
        return

    dfs = []
    for f in few_shot_files:
        try:
            df = pd.read_csv(f)
            if "fraction" in df.columns and "NSE" in df.columns:
                dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return

    combined = pd.concat(dfs, ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for model in combined["model"].unique():
        mdata = combined[combined["model"] == model]
        curve = mdata.groupby("fraction")["NSE"].median().sort_index()
        ax.plot(curve.index * 100, curve.values, "o-",
                color=MODEL_COLORS.get(model, "#999"), label=model, linewidth=2, markersize=8)

    ax.set_xlabel("Few-Shot Fraction (%)")
    ax.set_ylabel("Median NSE")
    ax.set_title("Few-Shot Skill Recovery", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "skill_recovery_curve.png")
    plt.close()
    print("  Saved skill_recovery_curve.png")


def fig_extreme_event_performance(results):
    """Figure: Extreme event performance (tail RMSE at multiple percentiles)."""
    if results.empty:
        return

    metrics_95 = "tail_RMSE_95" in results.columns
    metrics_99 = "tail_RMSE_99" in results.columns
    if not metrics_95:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    models = results["model"].unique()
    x = np.arange(len(models))
    width = 0.35

    if metrics_95:
        vals_95 = [results[results["model"] == m]["tail_RMSE_95"].median() for m in models]
        ax.bar(x - width / 2, vals_95, width, label="95th percentile",
               color="#1a6faf", alpha=0.85, edgecolor="black", linewidth=0.5)

    if metrics_99:
        vals_99 = [results[results["model"] == m]["tail_RMSE_99"].median() for m in models]
        ax.bar(x + width / 2, vals_99, width, label="99th percentile",
               color="#e05c2a", alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Median Tail RMSE")
    ax.set_title("Extreme Event Prediction Error", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "extreme_event_performance.png")
    plt.close()
    print("  Saved extreme_event_performance.png")


def generate_summary_table(results):
    """Generate summary statistics table as CSV."""
    if results.empty:
        return

    metrics = ["NSE", "KGE", "RMSE", "tail_RMSE_95", "CRPS"]
    available_metrics = [m for m in metrics if m in results.columns]

    summary_rows = []
    for model in sorted(results["model"].unique()):
        mdata = results[results["model"] == model]
        row = {"model": model, "n_basins": mdata["basin_id"].nunique()}
        for metric in available_metrics:
            vals = mdata[metric].dropna()
            row[f"{metric}_median"] = vals.median()
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_std"] = vals.std()
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    print(f"\n  Summary table saved ({len(summary)} models)")
    print(summary.to_string(index=False))


def main():
    print("=" * 60)
    print("Generating Publication Figures from Real Results")
    print("=" * 60)

    results = load_all_results()

    if results.empty:
        print("\nNo results found. Run experiments first:")
        print("  python run_all.py --step baselines")
        print("  python run_all.py --step foundation")
        return

    print(f"\nModels: {sorted(results['model'].unique())}")
    if "dataset" in results.columns:
        print(f"Datasets: {sorted(results['dataset'].unique())}")
    print(f"Total basin-model evaluations: {len(results)}")

    # Generate all figures
    fig_model_performance_comparison(results)
    fig_spatial_skill_map(results)
    fig_tail_error(results)
    fig_ablation(results)
    fig_calibration(results)
    fig_shap_summary()
    fig_causal_graph()
    fig_few_shot_recovery()
    fig_extreme_event_performance(results)
    generate_summary_table(results)

    print(f"\nAll figures saved to: {FIG_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
