"""
Regenerate the geographic / spatial-skill figures with REAL per-basin locations
(was: approximate dataset centroids). Assembles per-basin lat/lon from the raw
CAMELS-family archives on D:, joins to per-basin TimesFM zero-shot NSE (_l5s7),
and renders two maps:
  * figure_7_geographic_distribution.png  -- points coloured by dataset
  * P1_spatial_skill_map.png              -- points coloured by TimesFM NSE
cartopy/geopandas are not installed, so this draws a clean lon/lat scatter
(global extent, graticule) -- still a genuine per-basin distribution.
"""
import io, re, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import os

RES = Path(os.environ.get("RESULTS_DIR", "results"))
FIG = Path(os.environ.get("FIGURES_DIR", "figures"))
RAW = os.environ.get("CAMELS_RAW_DIR", "data/raw")


def norm(x):
    s = str(x).strip().upper()
    s = re.sub(r"^ID[_-]?", "", s)   # LamaH 'ID_1' -> '1'
    s = s.lstrip("0")
    return s or "0"


# ---- per-dataset coordinate loaders -> dict{norm_id: (lat, lon)} ----
def coords_us():
    df = pd.read_csv(f"{RAW}/CAMELS/camels_topo.txt", sep=";")
    return {norm(r.gauge_id): (r.gauge_lat, r.gauge_lon) for r in df.itertuples()}


def coords_br():
    zf = zipfile.ZipFile(f"{RAW}/CAMELS-BR/01_CAMELS_BR_attributes.zip")
    df = pd.read_csv(io.BytesIO(zf.read("01_CAMELS_BR_attributes/camels_br_location.txt")),
                     sep=r"\s+")
    return {norm(r.gauge_id): (r.gauge_lat, r.gauge_lon) for r in df.itertuples()}


def coords_cl():
    # transposed: rows = attributes, columns = gauge_id
    df = pd.read_csv(f"{RAW}/CAMELS-CL/extracted/1_CAMELScl_attributes.txt",
                     sep="\t", index_col=0)
    lat = df.loc["gauge_lat"].astype(float)
    lon = df.loc["gauge_lon"].astype(float)
    return {norm(c): (lat[c], lon[c]) for c in df.columns}


def coords_aus():
    df = pd.read_csv(f"{RAW}/CAMELS-AUS/CAMELS_AUS_Attributes&Indices_MasterTable.csv")
    return {norm(r.station_id): (r.lat_outlet, r.long_outlet) for r in df.itertuples()}


def coords_lamah():
    df = pd.read_csv(f"{RAW}/LamaH-CE/extracted/D_gauges/1_attributes/Gauge_attributes.csv",
                     sep=";")
    return {norm(r.ID): (r.lat, r.lon) for r in df.itertuples()}


def coords_ind():
    zf = zipfile.ZipFile(f"{RAW}/Camels-IND/CAMELS_IND_All_Catchments.zip")
    df = pd.read_csv(io.BytesIO(zf.read("attributes_csv/camels_ind_topo.csv")))
    return {norm(r.gauge_id): (r.cwc_lat, r.cwc_lon) for r in df.itertuples()}


LOADERS = {
    "CAMELS-US": coords_us, "CAMELS-BR": coords_br, "CAMELS-CL": coords_cl,
    "CAMELS-AUS": coords_aus, "LamaH-CE": coords_lamah, "CAMELS-IND": coords_ind,
}
COLORS = {
    "CAMELS-US": "#1f77b4", "CAMELS-BR": "#2ca02c", "CAMELS-CL": "#ff7f0e",
    "CAMELS-AUS": "#d62728", "LamaH-CE": "#9467bd", "CAMELS-IND": "#8c564b",
}


def assemble():
    rows = []
    for ds, loader in LOADERS.items():
        cm = loader()
        res = pd.read_csv(RES / f"timesfm_zero_shot_{ds}_l5s7.csv").dropna(subset=["NSE"])
        hit = 0
        for r in res.itertuples():
            c = cm.get(norm(r.basin_id))
            if c is None or pd.isna(c[0]) or pd.isna(c[1]):
                continue
            rows.append({"dataset": ds, "lat": float(c[0]), "lon": float(c[1]),
                         "nse": float(r.NSE)})
            hit += 1
        print(f"  {ds:12s} matched {hit:5d} / {len(res):5d} evaluated basins")
    return pd.DataFrame(rows)


def base_ax(ax, title):
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 78)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-60, 79, 30))
    ax.grid(True, ls=":", lw=0.5, color="0.7", zorder=0)
    ax.set_xlabel("Longitude ($^\\circ$E)")
    ax.set_ylabel("Latitude ($^\\circ$N)")
    ax.set_title(title)
    ax.set_facecolor("#f7fbff")


def plot_distribution(df):
    fig, ax = plt.subplots(figsize=(12, 5.6))
    base_ax(ax, f"Geographic distribution of evaluated basins ({len(df):,} basins)")
    for ds, g in df.groupby("dataset"):
        ax.scatter(g.lon, g.lat, s=5, c=COLORS[ds], label=f"{ds} ({len(g):,})",
                   alpha=0.6, edgecolors="none", zorder=3)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9, markerscale=2, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"figure_7_geographic_distribution.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] figure_7_geographic_distribution")


def plot_skill(df):
    fig, ax = plt.subplots(figsize=(12, 5.6))
    base_ax(ax, "Zero-shot skill by basin (TimesFM median NSE)")
    d = df.copy()
    d["nse_c"] = d["nse"].clip(-0.2, 1.0)
    d = d.sort_values("nse_c")  # plot low first so high-skill points sit on top
    sc = ax.scatter(d.lon, d.lat, s=6, c=d.nse_c, cmap="RdYlBu",
                    vmin=-0.2, vmax=1.0, alpha=0.75, edgecolors="none", zorder=3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("TimesFM zero-shot NSE (clipped at $-0.2$)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"P1_spatial_skill_map.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] P1_spatial_skill_map")


if __name__ == "__main__":
    df = assemble()
    print(f"TOTAL mapped basins: {len(df)}")
    plot_distribution(df)
    plot_skill(df)
