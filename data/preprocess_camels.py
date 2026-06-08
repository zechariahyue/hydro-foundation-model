#!/usr/bin/env python3
"""
Step 1: Extract and preprocess all CAMELS datasets into a unified format.

Produces per-basin parquet files with daily forcing + observed streamflow,
and a single basin_attributes.csv with static attributes for all basins.

Usage:
    python preprocess_camels.py
"""

import os
import sys
import zipfile
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Paths ────────────────────────────────────────────────────────────────────
RAW = Path(os.environ.get("CAMELS_RAW_DIR", "data/raw"))
OUT = Path(os.environ.get("CAMELS_DATA_DIR", "data/processed"))
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CAMELS-US
# ============================================================
def extract_camels_us():
    """Extract CAMELS-US timeseries zip and return extraction path."""
    src = RAW / "CAMELS"
    dst = src / "extracted"
    marker = dst / ".extracted_ok"
    if marker.exists():
        print("[CAMELS-US] Already extracted.")
        return dst

    zf = src / "basin_timeseries_v1p2_metForcing_obsFlow.zip"
    print(f"[CAMELS-US] Extracting {zf.name} (~3.2 GB) ...")
    with zipfile.ZipFile(zf, "r") as z:
        z.extractall(dst)
    marker.touch()
    print("[CAMELS-US] Extraction complete.")
    return dst


def load_camels_us_attributes():
    """Load all CAMELS-US static attributes into one DataFrame."""
    src = RAW / "CAMELS"
    attr_files = {
        "clim": "camels_clim.txt",
        "geol": "camels_geol.txt",
        "hydro": "camels_hydro.txt",
        "soil": "camels_soil.txt",
        "topo": "camels_topo.txt",
        "vege": "camels_vege.txt",
    }
    dfs = []
    for key, fname in attr_files.items():
        df = pd.read_csv(src / fname, sep=";", dtype={"gauge_id": str})
        df["gauge_id"] = df["gauge_id"].str.strip().str.zfill(8)
        df = df.set_index("gauge_id")
        dfs.append(df)

    attrs = pd.concat(dfs, axis=1)
    # Remove duplicate columns
    attrs = attrs.loc[:, ~attrs.columns.duplicated()]
    attrs["dataset"] = "CAMELS-US"
    print(f"[CAMELS-US] Loaded attributes for {len(attrs)} basins.")
    return attrs


def load_camels_us_basin(basin_id: str, extracted_root: Path, forcing: str = "daymet"):
    """Load daily forcing + streamflow for one CAMELS-US basin."""
    basin_id = str(basin_id).zfill(8)
    huc = basin_id[:2]

    # Streamflow
    sf_pattern = extracted_root / "basin_dataset_public_v1p2" / "usgs_streamflow" / huc / f"{basin_id}_streamflow_qc.txt"
    if not sf_pattern.exists():
        return None

    sf = pd.read_csv(
        sf_pattern, sep=r"\s+", header=None,
        names=["basin", "year", "month", "day", "QObs(mm/d)", "qc_flag"],
        dtype={"basin": str},
    )
    sf["date"] = pd.to_datetime(sf[["year", "month", "day"]])
    sf = sf.set_index("date")[["QObs(mm/d)"]]

    # Forcing
    forcing_dir = extracted_root / "basin_dataset_public_v1p2" / f"basin_mean_forcing" / forcing
    forcing_file = None
    for f in (forcing_dir / huc).glob(f"{basin_id}_*_forcing_leap.txt"):
        forcing_file = f
        break
    if forcing_file is None:
        return None

    # Skip first 3 header lines
    frc = pd.read_csv(forcing_file, sep=r"\s+", skiprows=3)
    frc["date"] = pd.to_datetime(frc[["Year", "Mnth", "Day"]].rename(
        columns={"Year": "year", "Mnth": "month", "Day": "day"}
    ))
    frc = frc.set_index("date")
    # Rename to standard names
    rename_map = {
        "dayl(s)": "dayl", "prcp(mm/day)": "prcp", "srad(W/m2)": "srad",
        "swe(mm)": "swe", "tmax(C)": "tmax", "tmin(C)": "tmin", "vp(Pa)": "vp",
    }
    frc = frc.rename(columns=rename_map)
    keep_cols = [c for c in ["prcp", "srad", "tmax", "tmin", "vp", "swe", "dayl"] if c in frc.columns]
    frc = frc[keep_cols]

    # Merge
    df = frc.join(sf, how="inner")
    df["basin_id"] = basin_id
    df["dataset"] = "CAMELS-US"
    return df


def process_camels_us():
    """Process all CAMELS-US basins."""
    print("\n" + "=" * 60)
    print("Processing CAMELS-US")
    print("=" * 60)

    extracted = extract_camels_us()
    attrs = load_camels_us_attributes()

    # Get basin list
    basin_ids = attrs.index.tolist()
    out_dir = OUT / "CAMELS-US"
    out_dir.mkdir(exist_ok=True)

    success = 0
    for bid in basin_ids:
        df = load_camels_us_basin(bid, extracted)
        if df is not None and len(df) > 365:
            df.to_parquet(out_dir / f"{bid}.parquet")
            success += 1

    print(f"[CAMELS-US] Processed {success}/{len(basin_ids)} basins.")
    return attrs


# ============================================================
# CAMELS-BR
# ============================================================
def extract_camels_br():
    """Extract CAMELS-BR streamflow and forcing."""
    src = RAW / "CAMELS-BR"
    dst = src / "extracted"
    marker = dst / ".extracted_ok"
    if marker.exists():
        print("[CAMELS-BR] Already extracted.")
        return dst

    dst.mkdir(parents=True, exist_ok=True)
    zips_needed = [
        "02_CAMELS_BR_streamflow_all_catchments.zip",
        "05_CAMELS_BR_precipitation.zip",
        "09_CAMELS_BR_temperature.zip",
        "07_CAMELS_BR_potential_evapotransp.zip",
    ]
    for zname in zips_needed:
        zf = src / zname
        if zf.exists():
            print(f"[CAMELS-BR] Extracting {zname} ...")
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(dst)

    marker.touch()
    return dst


def load_camels_br_attributes():
    """Load CAMELS-BR attributes."""
    src = RAW / "CAMELS-BR"
    dst = src / "extracted"

    # Extract attributes if needed
    attr_zip = src / "01_CAMELS_BR_attributes.zip"
    if attr_zip.exists() and not (dst / ".attrs_ok").exists():
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(attr_zip, "r") as z:
            z.extractall(dst)
        (dst / ".attrs_ok").touch()

    # Find attribute files
    attr_files = list(dst.rglob("*_attributes_*.txt")) + list(dst.rglob("*_attributes_*.csv"))
    if not attr_files:
        print("[CAMELS-BR] No attribute files found, creating minimal attributes.")
        return pd.DataFrame()

    dfs = []
    for f in attr_files:
        try:
            df = pd.read_csv(f, sep=None, engine="python", dtype={0: str})
            id_col = [c for c in df.columns if "gauge" in c.lower() or "id" in c.lower() or "code" in c.lower()]
            if id_col:
                df = df.rename(columns={id_col[0]: "gauge_id"})
                df["gauge_id"] = df["gauge_id"].astype(str).str.strip()
                df = df.set_index("gauge_id")
                dfs.append(df)
        except Exception:
            continue

    if dfs:
        attrs = pd.concat(dfs, axis=1)
        attrs = attrs.loc[:, ~attrs.columns.duplicated()]
    else:
        attrs = pd.DataFrame()

    attrs["dataset"] = "CAMELS-BR"
    print(f"[CAMELS-BR] Loaded attributes for {len(attrs)} basins.")
    return attrs


def process_camels_br():
    """Process all CAMELS-BR basins."""
    print("\n" + "=" * 60)
    print("Processing CAMELS-BR")
    print("=" * 60)

    extracted = extract_camels_br()
    attrs = load_camels_br_attributes()

    out_dir = OUT / "CAMELS-BR"
    out_dir.mkdir(exist_ok=True)

    # Find streamflow files
    sf_files = list(extracted.rglob("*streamflow*/*.txt")) + list(extracted.rglob("*streamflow*/*.csv"))
    if not sf_files:
        sf_files = list(extracted.rglob("*_streamflow_*.txt"))

    # Find precip and temp files
    precip_files = {f.stem.split("_")[0]: f for f in extracted.rglob("*precipitation*/*.txt")}
    temp_files = {f.stem.split("_")[0]: f for f in extracted.rglob("*temperature*/*.txt")}

    success = 0
    for sf_file in sf_files:
        try:
            basin_id = sf_file.stem.split("_")[0]
            sf = pd.read_csv(sf_file, sep=None, engine="python")

            # Find date column
            date_col = [c for c in sf.columns if "date" in c.lower() or "time" in c.lower()]
            if not date_col:
                # Try first column
                date_col = [sf.columns[0]]

            sf["date"] = pd.to_datetime(sf[date_col[0]], errors="coerce")
            sf = sf.dropna(subset=["date"]).set_index("date")

            # Find flow column
            flow_col = [c for c in sf.columns if "flow" in c.lower() or "discharge" in c.lower()
                        or "streamflow" in c.lower() or "vazao" in c.lower()]
            if not flow_col:
                flow_col = [sf.select_dtypes(include=[np.number]).columns[0]]

            df = sf[[flow_col[0]]].rename(columns={flow_col[0]: "QObs(mm/d)"})
            df["basin_id"] = basin_id
            df["dataset"] = "CAMELS-BR"

            if len(df) > 365:
                df.to_parquet(out_dir / f"{basin_id}.parquet")
                success += 1
        except Exception as e:
            continue

    print(f"[CAMELS-BR] Processed {success} basins.")
    return attrs


# ============================================================
# CAMELS-CL
# ============================================================
def extract_camels_cl():
    """Extract CAMELS-CL streamflow and forcing."""
    src = RAW / "CAMELS-CL"
    dst = src / "extracted"
    marker = dst / ".extracted_ok"
    if marker.exists():
        print("[CAMELS-CL] Already extracted.")
        return dst

    dst.mkdir(parents=True, exist_ok=True)
    zips_needed = [
        "2_CAMELScl_streamflow_m3s.zip",
        "3_CAMELScl_streamflow_mm.zip",
        "4_CAMELScl_precip_cr2met.zip",
        "8_CAMELScl_tmin_cr2met.zip",
        "9_CAMELScl_tmax_cr2met.zip",
        "1_CAMELScl_attributes.zip",
    ]
    for zname in zips_needed:
        zf = src / zname
        if zf.exists():
            print(f"[CAMELS-CL] Extracting {zname} ...")
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(dst)

    marker.touch()
    return dst


def process_camels_cl():
    """Process all CAMELS-CL basins."""
    print("\n" + "=" * 60)
    print("Processing CAMELS-CL")
    print("=" * 60)

    extracted = extract_camels_cl()
    out_dir = OUT / "CAMELS-CL"
    out_dir.mkdir(exist_ok=True)

    # Find streamflow_mm files (preferred) or m3s
    sf_files = list(extracted.rglob("*streamflow_mm*/*.txt")) + list(extracted.rglob("*streamflow_mm*.txt"))
    if not sf_files:
        sf_files = list(extracted.rglob("*streamflow*/*.txt"))

    # CAMELS-CL uses wide format: columns = basin IDs, rows = dates
    success = 0
    for sf_file in sf_files:
        try:
            sf = pd.read_csv(sf_file, sep=None, engine="python", index_col=0, parse_dates=True)
            for basin_id in sf.columns:
                series = sf[basin_id].dropna()
                if len(series) > 365:
                    df = pd.DataFrame({
                        "QObs(mm/d)": series.values,
                    }, index=series.index)
                    df.index.name = "date"
                    df["basin_id"] = str(basin_id).strip()
                    df["dataset"] = "CAMELS-CL"
                    df.to_parquet(out_dir / f"{str(basin_id).strip()}.parquet")
                    success += 1
        except Exception:
            continue

    print(f"[CAMELS-CL] Processed {success} basins.")

    # Attributes
    attrs = pd.DataFrame()
    attr_files = list(extracted.rglob("*attributes*/*.txt")) + list(extracted.rglob("*attributes*.txt"))
    for f in attr_files:
        try:
            df = pd.read_csv(f, sep=None, engine="python", index_col=0)
            if len(df.columns) > 2:
                attrs = pd.concat([attrs, df.T], axis=1) if len(attrs) == 0 else attrs
        except Exception:
            continue
    attrs["dataset"] = "CAMELS-CL"
    return attrs


# ============================================================
# CAMELS-AUS
# ============================================================
def extract_camels_aus():
    """Extract CAMELS-AUS."""
    src = RAW / "CAMELS-AUS"
    dst = src / "extracted"
    marker = dst / ".extracted_ok"
    if marker.exists():
        print("[CAMELS-AUS] Already extracted.")
        return dst

    dst.mkdir(parents=True, exist_ok=True)
    zips = [
        "03_streamflow.zip",
        "05_hydrometeorology.zip",
        "04_attributes.zip",
    ]
    for zname in zips:
        zf = src / zname
        if zf.exists():
            print(f"[CAMELS-AUS] Extracting {zname} ...")
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(dst)

    marker.touch()
    return dst


def process_camels_aus():
    """Process all CAMELS-AUS basins."""
    print("\n" + "=" * 60)
    print("Processing CAMELS-AUS")
    print("=" * 60)

    extracted = extract_camels_aus()
    out_dir = OUT / "CAMELS-AUS"
    out_dir.mkdir(exist_ok=True)

    # Also load the master attributes CSV directly
    master_csv = RAW / "CAMELS-AUS" / "CAMELS_AUS_Attributes&Indices_MasterTable.csv"
    attrs = pd.DataFrame()
    if master_csv.exists():
        attrs = pd.read_csv(master_csv, dtype={0: str})
        id_col = [c for c in attrs.columns if "station" in c.lower() or "id" in c.lower()][0]
        attrs = attrs.rename(columns={id_col: "gauge_id"})
        attrs["gauge_id"] = attrs["gauge_id"].astype(str).str.strip()
        attrs = attrs.set_index("gauge_id")
        print(f"[CAMELS-AUS] Loaded attributes for {len(attrs)} basins.")

    # Find streamflow files
    sf_files = list(extracted.rglob("*streamflow*/*.csv")) + list(extracted.rglob("*streamflow*/*.txt"))
    success = 0
    for sf_file in sf_files:
        try:
            basin_id = sf_file.stem.split("_")[0]
            sf = pd.read_csv(sf_file, sep=None, engine="python")
            date_col = [c for c in sf.columns if "date" in c.lower() or "time" in c.lower()]
            if not date_col:
                date_col = [sf.columns[0]]
            sf["date"] = pd.to_datetime(sf[date_col[0]], errors="coerce")
            sf = sf.dropna(subset=["date"]).set_index("date")

            flow_col = [c for c in sf.columns if "flow" in c.lower() or "discharge" in c.lower()
                        or "streamflow" in c.lower()]
            if not flow_col:
                num_cols = sf.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    flow_col = [num_cols[0]]
                else:
                    continue

            df = sf[[flow_col[0]]].rename(columns={flow_col[0]: "QObs(mm/d)"})
            df["basin_id"] = basin_id
            df["dataset"] = "CAMELS-AUS"

            if len(df) > 365:
                df.to_parquet(out_dir / f"{basin_id}.parquet")
                success += 1
        except Exception:
            continue

    print(f"[CAMELS-AUS] Processed {success} basins.")
    attrs["dataset"] = "CAMELS-AUS"
    return attrs


# ============================================================
# LamaH-CE
# ============================================================
def extract_lamah():
    """Extract LamaH-CE daily data."""
    src = RAW / "LamaH-CE"
    dst = src / "extracted"
    marker = dst / ".extracted_ok"
    if marker.exists():
        print("[LamaH-CE] Already extracted.")
        return dst

    dst.mkdir(parents=True, exist_ok=True)
    # Try daily archive first (smaller)
    tar = src / "2_LamaH-CE_daily.tar.gz"
    if tar.exists():
        print(f"[LamaH-CE] Extracting {tar.name} ...")
        import tarfile
        with tarfile.open(tar, "r:gz") as t:
            t.extractall(dst)
    marker.touch()
    return dst


def process_lamah():
    """Process LamaH-CE basins."""
    print("\n" + "=" * 60)
    print("Processing LamaH-CE")
    print("=" * 60)

    extracted = extract_lamah()
    out_dir = OUT / "LamaH-CE"
    out_dir.mkdir(exist_ok=True)

    # Find streamflow/discharge files
    sf_files = (
        list(extracted.rglob("**/Q_gauges/*.csv"))
        + list(extracted.rglob("**/q_gauges/*.csv"))
        + list(extracted.rglob("**/*discharge*.csv"))
        + list(extracted.rglob("**/*runoff*.csv"))
    )

    # Also look for forcing
    forcing_files = list(extracted.rglob("**/basin_forcing/*.csv"))

    success = 0
    for sf_file in sf_files:
        try:
            basin_id = sf_file.stem.split("_")[0]
            sf = pd.read_csv(sf_file, sep=None, engine="python")
            date_col = [c for c in sf.columns if "date" in c.lower() or "time" in c.lower()]
            if not date_col:
                date_col = [sf.columns[0]]
            sf["date"] = pd.to_datetime(sf[date_col[0]], errors="coerce")
            sf = sf.dropna(subset=["date"]).set_index("date")

            flow_col = [c for c in sf.columns if any(k in c.lower() for k in
                        ["flow", "discharge", "runoff", "qobs", "q_mm"])]
            if not flow_col:
                num_cols = sf.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    flow_col = [num_cols[0]]
                else:
                    continue

            df = sf[[flow_col[0]]].rename(columns={flow_col[0]: "QObs(mm/d)"})
            df["basin_id"] = basin_id
            df["dataset"] = "LamaH-CE"

            if len(df) > 365:
                df.to_parquet(out_dir / f"{basin_id}.parquet")
                success += 1
        except Exception:
            continue

    print(f"[LamaH-CE] Processed {success} basins.")
    return pd.DataFrame({"dataset": ["LamaH-CE"]})


# ============================================================
# Main
# ============================================================
def create_dataset_summary(all_attrs):
    """Create a summary of all processed datasets."""
    summary = {}
    for ds_name in ["CAMELS-US", "CAMELS-BR", "CAMELS-CL", "CAMELS-AUS", "LamaH-CE"]:
        ds_dir = OUT / ds_name
        if ds_dir.exists():
            n_files = len(list(ds_dir.glob("*.parquet")))
            summary[ds_name] = n_files
        else:
            summary[ds_name] = 0

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    total = 0
    for ds, n in summary.items():
        print(f"  {ds:15s}: {n:5d} basins")
        total += n
    print(f"  {'TOTAL':15s}: {total:5d} basins")
    print("=" * 60)

    # Save summary
    pd.DataFrame([summary]).to_csv(OUT / "dataset_summary.csv", index=False)
    return summary


def main():
    print("Foundation Models for Global Streamflow Forecasting")
    print("Step 1: Data Extraction & Preprocessing")
    print("=" * 60)

    all_attrs = []

    # Process each dataset
    attrs_us = process_camels_us()
    all_attrs.append(attrs_us)

    attrs_br = process_camels_br()
    all_attrs.append(attrs_br)

    attrs_cl = process_camels_cl()
    all_attrs.append(attrs_cl)

    attrs_aus = process_camels_aus()
    all_attrs.append(attrs_aus)

    attrs_lamah = process_lamah()
    all_attrs.append(attrs_lamah)

    # Combine all attributes
    combined_attrs = pd.concat(all_attrs, axis=0, ignore_index=False)
    combined_attrs.to_csv(OUT / "all_basin_attributes.csv")
    print(f"\nSaved combined attributes: {len(combined_attrs)} basins")

    # Summary
    create_dataset_summary(combined_attrs)
    print("\nDone! Processed data saved to:", OUT)


if __name__ == "__main__":
    main()
