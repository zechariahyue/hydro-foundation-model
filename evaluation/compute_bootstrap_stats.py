#!/usr/bin/env python3
"""Compute bootstrap CIs and statistical tests for Table 2."""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))

def bootstrap_ci(data, n_boot=10000, ci=0.95):
    """Compute bootstrap confidence interval for median."""
    boot_medians = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_medians.append(np.median(sample))
    
    lower = np.percentile(boot_medians, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_medians, (1 + ci) / 2 * 100)
    return lower, upper

def compute_statistics():
    """Compute bootstrap CIs and Wilcoxon tests for all models."""
    results = []
    
    datasets = ["CAMELS-US", "CAMELS-BR", "CAMELS-CL", "CAMELS-AUS", "LamaH-CE", "CAMELS-IND"]
    models = ["chronos", "timesfm", "patchtst", "persistence"]
    
    for dataset in datasets:
        print(f"\nProcessing {dataset}...")
        
        # Load results for each model
        model_data = {}
        for model in models:
            result_file = RESULTS_DIR / f"{model}_zero_shot_{dataset}.csv"
            if result_file.exists():
                df = pd.read_csv(result_file)
                model_data[model] = df["NSE"].dropna().values
                print(f"  {model}: {len(model_data[model])} basins, median NSE: {np.median(model_data[model]):.3f}")
        
        # Compute statistics for each model
        for model in models:
            if model not in model_data:
                continue
            
            nse_values = model_data[model]
            median_nse = np.median(nse_values)
            ci_lower, ci_upper = bootstrap_ci(nse_values)
            
            # Wilcoxon test vs persistence
            p_value = np.nan
            if model != "persistence" and "persistence" in model_data:
                pers_data = model_data["persistence"]
                min_len = min(len(nse_values), len(pers_data))
                if min_len > 0:
                    try:
                        stat, p_value = stats.wilcoxon(nse_values[:min_len], pers_data[:min_len])
                    except:
                        p_value = np.nan
            
            results.append({
                "dataset": dataset,
                "model": model,
                "median_nse": median_nse,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": p_value,
                "n_basins": len(nse_values),
            })
    
    # Save results
    df_results = pd.DataFrame(results)
    
    # Apply Benjamini-Hochberg correction
    p_values_mask = df_results["p_value"].notna()
    if p_values_mask.sum() > 0:
        from statsmodels.stats.multitest import multipletests
        p_values = df_results.loc[p_values_mask, "p_value"].values
        reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
        
        df_results.loc[p_values_mask, "p_adj"] = p_adj
        
        def sig_marker(p):
            if pd.isna(p):
                return ""
            elif p < 0.001:
                return "***"
            elif p < 0.01:
                return "**"
            elif p < 0.05:
                return "*"
            else:
                return "ns"
        
        df_results["significance"] = df_results["p_adj"].apply(sig_marker)
    
    df_results.to_csv(RESULTS_DIR / "bootstrap_statistics.csv", index=False)
    print(f"\n{'='*70}")
    print("BOOTSTRAP STATISTICS SUMMARY")
    print('='*70)
    print(df_results.to_string(index=False))
    print(f"\nSaved to: {RESULTS_DIR / 'bootstrap_statistics.csv'}")

if __name__ == "__main__":
    compute_statistics()
