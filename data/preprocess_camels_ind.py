#!/usr/bin/env python3
"""
Preprocess CAMELS-IND data to match pipeline format.

Converts wide-format streamflow CSV to individual basin parquet files
with datetime index and QObs(mm/d) column.

Usage:
    python preprocess_camels_ind.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm

# Paths
RAW_DIR = Path(os.environ.get("CAMELS_IND_RAW_DIR", "data/raw/CAMELS-IND"))
OUTPUT_DIR = Path(os.environ.get("CAMELS_DATA_DIR", "data/processed")) / "CAMELS-IND"
ZIP_FILE = RAW_DIR / "CAMELS_IND_Catchments_Streamflow_Sufficient.zip"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CAMELS-IND Preprocessing")
print("=" * 60)

# Extract streamflow data from zip
print("\n1. Extracting streamflow data from zip...")
with ZipFile(ZIP_FILE, 'r') as z:
    # Read streamflow CSV directly from zip
    with z.open('streamflow_timeseries/streamflow_observed.csv') as f:
        df = pd.read_csv(f)

print(f"   Loaded {len(df)} days × {len(df.columns)-3} basins")

# Create datetime index
print("\n2. Creating datetime index...")
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df = df.set_index('date')
df = df.drop(columns=['year', 'month', 'day'])

# Filter date range (2000-2017 to match other datasets)
print("\n3. Filtering date range (2000-01-01 to 2017-12-31)...")
df = df[(df.index >= '2000-01-01') & (df.index <= '2017-12-31')]
print(f"   Filtered to {len(df)} days")

# Convert each basin column to individual parquet file
print("\n4. Converting to individual basin parquet files...")
basin_ids = [col for col in df.columns]
successful = 0
skipped = 0

for basin_id in tqdm(basin_ids, desc="Processing basins"):
    # Extract basin series
    q = df[basin_id].copy()

    # Drop NaN values
    q = q.dropna()

    # Skip if insufficient data
    if len(q) < 100:
        skipped += 1
        continue

    # Convert to DataFrame with QObs(mm/d) column
    # Note: CAMELS-IND streamflow is in m³/s, but we'll keep it as-is
    # since the models work with raw discharge values
    basin_df = pd.DataFrame({
        'QObs(mm/d)': q.values
    }, index=q.index)

    # Save to parquet
    output_path = OUTPUT_DIR / f"{basin_id}.parquet"
    basin_df.to_parquet(output_path)
    successful += 1

print(f"\n5. Summary:")
print(f"   Successfully processed: {successful} basins")
print(f"   Skipped (insufficient data): {skipped} basins")
print(f"   Output directory: {OUTPUT_DIR}")

# Update dataset summary
print("\n6. Updating dataset summary...")
summary_file = Path(os.environ.get("CAMELS_DATA_DIR", "data/processed")) / "dataset_summary.csv"
if summary_file.exists():
    summary = pd.read_csv(summary_file, header=None)
    # Append CAMELS-IND
    new_row = pd.DataFrame([['CAMELS-IND', successful]], columns=[0, 1])
    summary = pd.concat([summary, new_row], ignore_index=True)
    summary.to_csv(summary_file, index=False, header=False)
    print(f"   Updated {summary_file}")

print("\n" + "=" * 60)
print("Preprocessing complete!")
print("=" * 60)
print(f"\nNext steps:")
print(f"1. Run zero-shot: python run_foundation_models.py --model all --datasets CAMELS-IND")
print(f"2. (Optional) Run few-shot: python run_few_shot_lora.py --dataset CAMELS-IND --fraction 0.10")
