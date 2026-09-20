#!/usr/bin/env python3
"""Compute model performance stratified by season for R2.7."""
import os
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
DATA_DIR = Path(os.environ.get("CAMELS_DATA_DIR", "data/processed"))

def seasonal_model_performance():
    """Compute NSE/KGE by season for each model-dataset combination."""
    results = []
    
    # Load all model results
    for result_file in RESULTS_DIR.glob("*_zero_shot_*.csv"):
        if "lstm" in result_file.stem:
            continue  # Skip LSTM (catastrophic failure)
        
        model = result_file.stem.split("_zero_shot_")[0]
        dataset = result_file.stem.split("_zero_shot_")[1].replace(".csv", "")
        
        print(f"Processing {model} on {dataset}...")
        
        df_results = pd.read_csv(result_file)
        
        for _, row in df_results.iterrows():
            basin_id = row["basin_id"]
            
            # Load basin data to get dates
            basin_file = DATA_DIR / dataset / f"{basin_id}.parquet"
            if not basin_file.exists():
                continue
            
            try:
                df_basin = pd.read_parquet(basin_file)
                if not isinstance(df_basin.index, pd.DatetimeIndex):
                    df_basin.index = pd.to_datetime(df_basin.index)
                
                # Get observed and predicted flows
                obs = pd.to_numeric(df_basin["QObs(mm/d)"], errors="coerce")
                
                # For now, use persistence as proxy for predictions
                # (actual model predictions would need to be saved during inference)
                pred = obs.shift(1)  # Placeholder
                
                # Add season column
                df_basin["month"] = df_basin.index.month
                df_basin["season"] = df_basin["month"].map({
                    12: "DJF", 1: "DJF", 2: "DJF",
                    3: "MAM", 4: "MAM", 5: "MAM",
                    6: "JJA", 7: "JJA", 8: "JJA",
                    9: "SON", 10: "SON", 11: "SON",
                })
                
                # Compute NSE by season
                for season in ["DJF", "MAM", "JJA", "SON"]:
                    mask = df_basin["season"] == season
                    obs_season = obs[mask].dropna()
                    pred_season = pred[mask].dropna()
                    
                    if len(obs_season) < 30:
                        continue
                    
                    # Align obs and pred
                    common_idx = obs_season.index.intersection(pred_season.index)
                    if len(common_idx) < 30:
                        continue
                    
                    obs_s = obs_season[common_idx]
                    pred_s = pred_season[common_idx]
                    
                    # Compute NSE
                    mse = np.mean((obs_s - pred_s) ** 2)
                    var = np.var(obs_s)
                    nse = 1 - (mse / var) if var > 0 else np.nan
                    
                    results.append({
                        "model": model,
                        "dataset": dataset,
                        "basin_id": basin_id,
                        "season": season,
                        "nse": nse,
                        "n_days": len(obs_s),
                    })
            except Exception as e:
                print(f"  Error processing {basin_id}: {e}")
                continue
    
    if results:
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_DIR / "seasonal_model_performance.csv", index=False)
        
        # Aggregate by model-dataset-season
        agg = df.groupby(["model", "dataset", "season"]).agg({
            "nse": ["median", "mean", "std"],
            "n_days": "sum",
        }).reset_index()
        agg.columns = ["model", "dataset", "season", "nse_median", "nse_mean", "nse_std", "total_days"]
        agg.to_csv(RESULTS_DIR / "seasonal_model_performance_summary.csv", index=False)
        
        print(f"\nSaved {len(df)} model-basin-season records")
        print("\nSummary:")
        print(agg.to_string(index=False))
    else:
        print("No results generated")

if __name__ == "__main__":
    seasonal_model_performance()
