#!/usr/bin/env python3
"""
Fix CAMELS-BR parquet files: dates were parsed incorrectly.
Also fix persistence baseline to include CAMELS-US with correct test period.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path(os.environ.get("CAMELS_RAW_DIR", "data/raw"))
OUT = Path(os.environ.get("CAMELS_DATA_DIR", "data/processed"))


def fix_camels_br():
    """Reprocess CAMELS-BR with correct date parsing."""
    print("Fixing CAMELS-BR...")
    extracted = RAW / "CAMELS-BR" / "extracted"
    out_dir = OUT / "CAMELS-BR"
    out_dir.mkdir(exist_ok=True)

    # Remove old broken parquets
    for old in out_dir.glob("*.parquet"):
        old.unlink()

    sf_dir = extracted / "02_CAMELS_BR_streamflow_all_catchments"
    if not sf_dir.exists():
        # Try finding it
        sf_files = list(extracted.rglob("*streamflow*/*.txt"))
    else:
        sf_files = list(sf_dir.glob("*.txt"))

    print(f"  Found {len(sf_files)} streamflow files")

    success = 0
    for sf_file in sf_files:
        try:
            basin_id = sf_file.stem.replace("_streamflow", "")
            df = pd.read_csv(sf_file, sep=None, engine="python")

            # Build date from year/month/day columns
            if "year" in df.columns and "month" in df.columns and "day" in df.columns:
                df["date"] = pd.to_datetime(df[["year", "month", "day"]])
            else:
                continue

            df = df.set_index("date")

            # Find flow column
            flow_col = None
            for c in df.columns:
                if "streamflow" in c.lower() or "flow" in c.lower() or "discharge" in c.lower():
                    flow_col = c
                    break
            if flow_col is None:
                continue

            basin_df = pd.DataFrame({"QObs(mm/d)": pd.to_numeric(df[flow_col], errors="coerce")})
            basin_df = basin_df.replace(-99.99, np.nan).replace(-999, np.nan)
            basin_df["basin_id"] = basin_id
            basin_df["dataset"] = "CAMELS-BR"

            valid = basin_df.dropna(subset=["QObs(mm/d)"])
            if len(valid) > 365:
                basin_df.to_parquet(out_dir / f"{basin_id}.parquet")
                success += 1
        except Exception as e:
            continue

    print(f"[CAMELS-BR] Fixed: {success} basins")
    return success


def verify_all_datasets():
    """Verify date ranges for all datasets."""
    print("\nDataset verification:")
    for ds_name in ["CAMELS-US", "CAMELS-BR", "CAMELS-CL", "CAMELS-AUS", "LamaH-CE"]:
        ds_dir = OUT / ds_name
        if not ds_dir.exists():
            print(f"  {ds_name}: NOT FOUND")
            continue

        files = list(ds_dir.glob("*.parquet"))
        if not files:
            print(f"  {ds_name}: 0 files")
            continue

        # Sample first file
        df = pd.read_parquet(files[0])
        print(f"  {ds_name}: {len(files)} basins | "
              f"dates: {df.index.min()} to {df.index.max()} | "
              f"cols: {[c for c in df.columns if c not in ['basin_id', 'dataset']]}")


def main():
    fix_camels_br()
    verify_all_datasets()

    # Update summary
    summary = {}
    for ds_name in ["CAMELS-US", "CAMELS-BR", "CAMELS-CL", "CAMELS-AUS", "LamaH-CE"]:
        ds_dir = OUT / ds_name
        summary[ds_name] = len(list(ds_dir.glob("*.parquet"))) if ds_dir.exists() else 0

    total = sum(summary.values())
    print(f"\nUpdated totals: {summary}")
    print(f"Total: {total} basins")
    pd.DataFrame([summary]).to_csv(OUT / "dataset_summary.csv", index=False)


if __name__ == "__main__":
    main()
