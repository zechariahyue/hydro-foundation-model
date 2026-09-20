#!/usr/bin/env python3
"""
CAMELS-BR data-quality sensitivity: median NSE of every model on all evaluated BR stations vs
on the 897 quality-selected catchments (03_CAMELS_BR_streamflow_selected_catchments.zip).
No GPU: filters the existing _l5s7 result CSVs. Backs SI Sect. "CAMELS-BR data quality".
Originally an ad-hoc computation (2026-07); scripted 2026-09-06 so it can be re-run after
the Chronos 20-sample re-run.  Output: experiments/results/BR_QUALITY_SUBSET.txt
"""
import os
import zipfile
from pathlib import Path
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
ZIP = Path(str(Path(os.environ.get("CAMELS_RAW_DIR", "data/raw")) / "CAMELS-BR/03_CAMELS_BR_streamflow_selected_catchments.zip"))
ids = {Path(n).name.split("_")[0] for n in zipfile.ZipFile(ZIP).namelist() if n.endswith("_streamflow.txt")}
lines = [f"quality-selected ids in zip: {len(ids)}"]
for model, f in [("TimesFM", "timesfm_zero_shot_CAMELS-BR_l5s7.csv"), ("Chronos", "chronos_zero_shot_CAMELS-BR_l5s7.csv"),
                 ("persistence", "persistence_zero_shot_CAMELS-BR_l5s7.csv"), ("LSTM (global, fair)", "lstm_global_CAMELS-BR_l5s7.csv"),
                 ("PatchTST", "patchtst_zero_shot_CAMELS-BR_l5s7.csv")]:
    d = pd.read_csv(RES / f, dtype={"basin_id": str})
    d = d[d.NSE.notna() & (d.NSE.abs() < float("inf"))]
    sub = d[d.basin_id.isin(ids)]
    lines.append(f"{model:20s} all n={len(d):4d} median NSE {d.NSE.median():.3f} | quality-selected n={len(sub):3d} median NSE {sub.NSE.median():.3f} | shift {sub.NSE.median()-d.NSE.median():+.3f}")
out = "\n".join(lines); print(out)
(RES / "BR_QUALITY_SUBSET.txt").write_text(out + "\n", encoding="utf-8")
