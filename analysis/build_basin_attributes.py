#!/usr/bin/env python3
"""
Build a per-basin attribute table for the hydrological-controls analysis.

Two tiers:

  TIER 1 -- flow signatures computed HERE from the discharge series, on the SAME
  last-5-years window the models were scored on (replicating the truncation logic
  of run_foundation_models.load_basin_series, including the CAMELS-BR degenerate-
  index position fallback). Available for EVERY evaluated basin, computed
  identically across all six archives.

  Every Tier-1 signature is SCALE-INVARIANT (a ratio, a correlation, or a
  log-space slope). This matters: the processed column 'QObs(mm/d)' is known to
  carry each archive's NATIVE discharge units, not mm/d. Scale-invariant
  signatures are unaffected by that mislabel, so the controls analysis is immune
  to it. Absolute-magnitude attributes (q_mean etc.) are deliberately NOT used.

  TIER 2 -- catchment attributes harmonized from the raw archives on D:
  (aridity, snow fraction, precipitation seasonality, area, elevation, slope,
  regulation degree, forest fraction). Coverage is reported honestly per field;
  LamaH-CE ships no hydrological signatures, which is precisely why Tier 1 exists.

Output: experiments/results/basin_attributes.csv
"""
import os
import io
import re
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CAMELS_DATA_DIR", "data/processed"))
RES = Path(os.environ.get("RESULTS_DIR", "results"))
QCOL = "QObs(mm/d)"          # native units despite the label -- see module docstring
LAST_YEARS = 5

DATASETS = ["CAMELS-US", "CAMELS-BR", "CAMELS-CL", "CAMELS-AUS", "LamaH-CE", "CAMELS-IND"]


def norm(x):
    """Normalize a basin id for cross-table joins (matches build_real_basin_map.norm)."""
    s = str(x).strip().upper()
    s = re.sub(r"^ID[_-]?", "", s)     # LamaH 'ID_1' -> '1'
    s = s.lstrip("0")
    return s or "0"


# ---------------------------------------------------------------------------
# TIER 1: flow signatures from the discharge series (scale-invariant)
# ---------------------------------------------------------------------------
def load_window(pf: Path) -> pd.Series | None:
    """Load one basin's discharge, truncated to the last LAST_YEARS of its record.

    Mirrors run_foundation_models.load_basin_series: date-based truncation when the
    index is a genuine daily series, position-based when it is degenerate (CAMELS-BR).
    """
    try:
        df = pd.read_parquet(pf)
    except Exception:
        return None
    if QCOL not in df.columns:
        return None

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return None

    n_keep = int(LAST_YEARS * 365.25)
    try:
        degenerate = df.index.normalize().nunique() < 0.5 * len(df)
    except Exception:
        degenerate = True

    if degenerate:
        df = df.iloc[-n_keep:]
    else:
        cutoff = df.index.max() - pd.Timedelta(days=n_keep)
        df = df[df.index > cutoff]

    q = pd.to_numeric(df[QCOL], errors="coerce")
    q = q[~q.isna()]
    q[q < 0] = np.nan                      # negative discharge is a rating artefact
    q = q.dropna()
    return q if len(q) >= 365 else None


def bfi_lyne_hollick(q: np.ndarray, alpha: float = 0.925, passes: int = 3) -> float:
    """Baseflow index via the Lyne-Hollick recursive digital filter (ratio -> scale-free)."""
    if len(q) < 10 or np.all(q == 0):
        return np.nan
    b = q.astype(float).copy()
    for p in range(passes):
        series = b if p % 2 == 0 else b[::-1]
        f = np.zeros_like(series)
        f[0] = series[0]
        for i in range(1, len(series)):
            f[i] = alpha * f[i - 1] + 0.5 * (1 + alpha) * (series[i] - series[i - 1])
        quick = np.maximum(f, 0.0)
        base = series - quick
        base = np.clip(base, 0.0, series)
        b = base if p % 2 == 0 else base[::-1]
    tot = np.sum(q)
    return float(np.sum(b) / tot) if tot > 0 else np.nan


def rb_flashiness(q: np.ndarray) -> float:
    """Richards-Baker flashiness index: sum|dQ| / sumQ (ratio -> scale-free)."""
    tot = np.sum(q)
    if tot <= 0 or len(q) < 2:
        return np.nan
    return float(np.sum(np.abs(np.diff(q))) / tot)


def fdc_slope(q: np.ndarray) -> float:
    """Slope of the flow-duration curve between the 33rd and 66th exceedance percentiles,
    in log space (a log-ratio -> scale-free). Standard Addor/Sawicz signature."""
    pos = q[q > 0]
    if len(pos) < 30:
        return np.nan
    q33, q66 = np.percentile(pos, [66, 33])   # exceedance: 33% exceedance = 66th pctile
    if q33 <= 0 or q66 <= 0:
        return np.nan
    return float((np.log(q33) - np.log(q66)) / (0.66 - 0.33))


def flow_signatures(q: pd.Series) -> dict:
    v = q.to_numpy(dtype=float)
    mu = float(np.mean(v))
    out = {
        "n_days": len(v),
        "ac1": np.nan, "ac7": np.nan, "ac30": np.nan,
        "bfi": bfi_lyne_hollick(v),
        "rbi": rb_flashiness(v),
        "q_cv": float(np.std(v) / mu) if mu > 0 else np.nan,
        "q_skew": float(pd.Series(v).skew()),
        "fdc_slope": fdc_slope(v),
        "zero_q_frac": float(np.mean(v <= 0)),
    }
    # autocorrelation at 1, 7, 30 days -- the memory of the hydrograph.
    s = pd.Series(v)
    for lag, key in ((1, "ac1"), (7, "ac7"), (30, "ac30")):
        if len(v) > lag + 10:
            a = s.autocorr(lag=lag)
            out[key] = float(a) if pd.notna(a) else np.nan
    # Q95/Q50 -- high-flow ratio (scale-free)
    pos = v[v > 0]
    if len(pos) >= 30:
        q50, q95 = np.percentile(pos, [50, 95])
        out["q95_q50"] = float(q95 / q50) if q50 > 0 else np.nan
    else:
        out["q95_q50"] = np.nan
    return out


def build_tier1() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        d = DATA_DIR / ds
        if not d.exists():
            print(f"  [WARN] {d} missing")
            continue
        files = sorted(d.glob("*.parquet"))
        ok = 0
        for pf in files:
            q = load_window(pf)
            if q is None:
                continue
            sig = flow_signatures(q)
            sig["dataset"] = ds
            sig["basin_id"] = pf.stem
            sig["bid"] = norm(pf.stem)
            rows.append(sig)
            ok += 1
        print(f"  {ds:12s} signatures for {ok:5d} / {len(files):5d} basins")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TIER 2: harmonized archive catchment attributes
# ---------------------------------------------------------------------------
# harmonized -> source column, per dataset
def attrs_us() -> pd.DataFrame:
    df = pd.read_csv(Path(os.environ.get("CAMELS_DATA_DIR", "data/processed")) / "attributes" / "attributes_camels_harmonized.csv")
    m = {"gauge_id": "gauge_id", "aridity": "aridity", "p_mean": "precipitation",
         "p_seasonality": "p_seasonality", "frac_snow": "snow_fraction",
         "area": "area", "elev_mean": "elevation", "slope_mean": "slope",
         "forest_frac": "forest_fraction"}
    out = pd.DataFrame({k: df[v] for k, v in m.items() if v in df.columns})
    out["regulation"] = np.nan          # CAMELS-US ships no regulation index
    return out


def attrs_br() -> pd.DataFrame:
    zf = zipfile.ZipFile(str(Path(os.environ.get("CAMELS_RAW_DIR", "data/raw")) / "CAMELS-BR/01_CAMELS_BR_attributes.zip"))

    def rd(name):
        return pd.read_csv(io.BytesIO(zf.read(f"01_CAMELS_BR_attributes/camels_br_{name}.txt")),
                           sep=r"\s+")
    cl, hy, tp, hu = rd("climate"), rd("hydrology"), rd("topography"), rd("human_intervention")
    df = cl.merge(tp, on="gauge_id", how="outer").merge(hu, on="gauge_id", how="outer")
    return pd.DataFrame({
        "gauge_id": df["gauge_id"], "aridity": df.get("aridity"), "p_mean": df.get("p_mean"),
        "p_seasonality": df.get("p_seasonality"), "frac_snow": df.get("frac_snow"),
        "area": df.get("area"), "elev_mean": df.get("elev_mean"),
        "slope_mean": df.get("slope_mean"), "forest_frac": np.nan,
        "regulation": df.get("regulation_degree"),
    })


def attrs_cl() -> pd.DataFrame:
    df = pd.read_csv(str(Path(os.environ.get("CAMELS_RAW_DIR", "data/raw")) / "CAMELS-CL/extracted/1_CAMELScl_attributes.txt"),
                     sep="\t", index_col=0).T
    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    return pd.DataFrame({
        "gauge_id": df.index, "aridity": num("aridity_cr2met"), "p_mean": num("p_mean_cr2met"),
        "p_seasonality": num("p_seasonality_cr2met"), "frac_snow": num("frac_snow_cr2met"),
        "area": num("area"), "elev_mean": num("elev_mean"), "slope_mean": num("slope_mean"),
        "forest_frac": num("forest_frac"), "regulation": num("interv_degree"),
    }).reset_index(drop=True)


def attrs_aus() -> pd.DataFrame:
    df = pd.read_csv(str(Path(os.environ.get("CAMELS_RAW_DIR", "data/raw")) / "CAMELS-AUS/CAMELS_AUS_Attributes&Indices_MasterTable.csv"))
    return pd.DataFrame({
        "gauge_id": df["station_id"], "aridity": df.get("aridity"), "p_mean": df.get("p_mean"),
        "p_seasonality": df.get("p_seasonality"), "frac_snow": df.get("frac_snow"),
        "area": df.get("catchment_area"), "elev_mean": df.get("elev_mean"),
        "slope_mean": df.get("mean_slope_pct"), "forest_frac": df.get("prop_forested"),
        "regulation": df.get("river_di"),      # river disturbance index
    })


def attrs_lamah() -> pd.DataFrame:
    df = pd.read_csv(
        str(Path(os.environ.get("CAMELS_RAW_DIR", "data/raw")) / "LamaH-CE/extracted/B_basins_intermediate_all/1_attributes/Catchment_attributes.csv"),
        sep=";")
    return pd.DataFrame({
        "gauge_id": df["ID"], "aridity": df.get("arid_1"), "p_mean": df.get("p_mean"),
        "p_seasonality": df.get("p_season"), "frac_snow": df.get("frac_snow"),
        "area": df.get("area_calc"), "elev_mean": df.get("elev_mean"),
        "slope_mean": df.get("slope_mean"), "forest_frac": df.get("forest_fra"),
        "regulation": np.nan,                  # LamaH ships no regulation index here
    })


def attrs_ind() -> pd.DataFrame:
    zf = zipfile.ZipFile(str(Path(os.environ.get("CAMELS_RAW_DIR", "data/raw")) / "Camels-IND/CAMELS_IND_All_Catchments.zip"))

    def rd(n):
        return pd.read_csv(io.BytesIO(zf.read(f"attributes_csv/camels_ind_{n}.csv")))
    cl, tp, an = rd("clim"), rd("topo"), rd("anth")
    df = cl.merge(tp, on="gauge_id", how="outer").merge(an, on="gauge_id", how="outer")
    return pd.DataFrame({
        "gauge_id": df["gauge_id"], "aridity": df.get("aridity_p_pet"), "p_mean": df.get("p_mean"),
        "p_seasonality": np.nan,               # IND has no directly comparable seasonality index
        "frac_snow": np.nan,                   # IND ships no snow fraction
        "area": df.get("cwc_area"), "elev_mean": df.get("elev_mean"),
        "slope_mean": df.get("slope_mean"), "forest_frac": np.nan,
        "regulation": df.get("reservoir_index"),
    })


ATTR_LOADERS = {
    "CAMELS-US": attrs_us, "CAMELS-BR": attrs_br, "CAMELS-CL": attrs_cl,
    "CAMELS-AUS": attrs_aus, "LamaH-CE": attrs_lamah, "CAMELS-IND": attrs_ind,
}
TIER2 = ["aridity", "p_mean", "p_seasonality", "frac_snow", "area",
         "elev_mean", "slope_mean", "forest_frac", "regulation"]


def build_tier2() -> pd.DataFrame:
    frames = []
    for ds, fn in ATTR_LOADERS.items():
        try:
            a = fn()
        except Exception as e:
            print(f"  [WARN] {ds} attributes failed: {e}")
            continue
        a = a.copy()
        a["dataset"] = ds
        a["bid"] = a["gauge_id"].map(norm)
        for c in TIER2:
            if c not in a.columns:
                a[c] = np.nan
            a[c] = pd.to_numeric(a[c], errors="coerce")
        a = a[["dataset", "bid"] + TIER2].drop_duplicates(subset=["dataset", "bid"])
        cov = {c: int(a[c].notna().sum()) for c in TIER2}
        print(f"  {ds:12s} n={len(a):5d}  coverage: " +
              " ".join(f"{c}={cov[c]}" for c in TIER2))
        frames.append(a)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# per-basin model skill
# ---------------------------------------------------------------------------
MODELS = {"timesfm": "timesfm_zero_shot", "chronos": "chronos_zero_shot",
          "patchtst": "patchtst_zero_shot", "persistence": "persistence_zero_shot",
          "lstm": "lstm_global"}


def build_skill() -> pd.DataFrame:
    frames = []
    for ds in DATASETS:
        base = None
        for m, pref in MODELS.items():
            f = RES / f"{pref}_{ds}_l5s7.csv"
            if not f.exists():
                print(f"  [WARN] missing {f.name}")
                continue
            d = pd.read_csv(f)[["basin_id", "NSE", "KGE", "PBIAS"]]
            d["bid"] = d["basin_id"].map(norm)
            d = d.rename(columns={"NSE": f"nse_{m}", "KGE": f"kge_{m}", "PBIAS": f"pbias_{m}"})
            d = d.drop(columns=["basin_id"]).drop_duplicates(subset=["bid"])
            base = d if base is None else base.merge(d, on="bid", how="outer")
        if base is None:
            continue
        base["dataset"] = ds
        frames.append(base)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    print("TIER 1 -- flow signatures from discharge (last 5 yr, scale-invariant)")
    t1 = build_tier1()
    print(f"  total: {len(t1)} basins\n")

    print("TIER 2 -- harmonized archive catchment attributes")
    t2 = build_tier2()
    print(f"  total: {len(t2)} rows\n")

    print("SKILL -- per-basin NSE/KGE/PBIAS by model (_l5s7)")
    sk = build_skill()
    print(f"  total: {len(sk)} rows\n")

    df = (sk.merge(t1.drop(columns=["basin_id"]), on=["dataset", "bid"], how="left")
            .merge(t2, on=["dataset", "bid"], how="left"))

    # The headline derived quantities: when does the FM add value over the trivial
    # baseline, and when is local training still worth it?
    df["d_tfm_pers"] = df["nse_timesfm"] - df["nse_persistence"]
    df["d_lstm_tfm"] = df["nse_lstm"] - df["nse_timesfm"]

    out = RES / "basin_attributes.csv"
    df.to_csv(out, index=False)

    print(f"[OK] wrote {out}  ({len(df)} basins x {len(df.columns)} cols)")
    print("\nCoverage of joined table:")
    for c in ["nse_timesfm", "nse_persistence", "nse_lstm", "ac1", "bfi", "rbi",
              "fdc_slope", "aridity", "frac_snow", "p_seasonality", "regulation"]:
        if c in df.columns:
            print(f"  {c:16s} {df[c].notna().sum():5d} / {len(df)}")
