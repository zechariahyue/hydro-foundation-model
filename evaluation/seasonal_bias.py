#!/usr/bin/env python3
"""Seasonal bias analysis for reviewer response."""
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
DATA_DIR = Path(os.environ.get("CAMELS_DATA_DIR", "data/processed"))

def seasonal_bias():
    seasonal_stats = []
    for dataset in ["CAMELS-US", "CAMELS-BR", "CAMELS-CL", "CAMELS-AUS", "LamaH-CE"]:
        ds_dir = DATA_DIR / dataset
        if not ds_dir.exists():
            continue
        print(f"Processing {dataset}...")
        for pf in list(ds_dir.glob("*.parquet"))[:50]:
            try:
                df = pd.read_parquet(pf)
                if "QObs(mm/d)" not in df.columns:
                    continue
                df["month"] = df.index.month
                df["season"] = df["month"].map({
                    12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                    6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON",
                })
                q = pd.to_numeric(df["QObs(mm/d)"], errors="coerce")
                for season in ["DJF", "MAM", "JJA", "SON"]:
                    season_q = q[df["season"] == season].dropna()
                    if len(season_q) > 30:
                        seasonal_stats.append({
                            "dataset": dataset, "basin_id": pf.stem, "season": season,
                            "mean_flow": season_q.mean(), "median_flow": season_q.median(),
                            "std_flow": season_q.std(), "n_days": len(season_q),
                        })
            except Exception:
                continue
    if seasonal_stats:
        df = pd.DataFrame(seasonal_stats)
        df.to_csv(RESULTS_DIR / "seasonal_flow_statistics.csv", index=False)
        agg = df.groupby(["dataset", "season"]).agg({
            "mean_flow": ["mean", "median"], "n_days": "sum",
        }).reset_index()
        agg.columns = ["dataset", "season", "mean_flow_avg", "median_flow_avg", "total_days"]
        agg.to_csv(RESULTS_DIR / "seasonal_summary.csv", index=False)
        print(f"\nSaved: {len(df)} basin-seasons")
        print(agg.to_string(index=False))

if __name__ == "__main__":
    seasonal_bias()
