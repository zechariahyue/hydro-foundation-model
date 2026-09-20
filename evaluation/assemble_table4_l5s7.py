#!/usr/bin/env python3
"""Assemble the consistent-protocol (last-5yr, stride-7) zero-shot Table 4 from the
_l5s7 result CSVs: per-dataset median NSE + 95% bootstrap CI + basin counts, and
Wilcoxon signed-rank vs persistence (Benjamini-Hochberg corrected across the model x
dataset grid). Prints a LaTeX-ready summary and the overall NSE ranges for the abstract.
Safe to run mid-run: datasets without CSVs are skipped."""
import os
import glob, os
import numpy as np
import pandas as pd
from scipy import stats

RES = "experiments/results"
SUFFIX = "_l5s7"
DATASETS = ["CAMELS-US", "CAMELS-BR", "CAMELS-CL", "CAMELS-AUS", "LamaH-CE", "CAMELS-IND"]
REGION = {"CAMELS-US": "N. America", "CAMELS-BR": "S. America", "CAMELS-CL": "S. America",
          "CAMELS-AUS": "Australia", "LamaH-CE": "Europe", "CAMELS-IND": "Asia"}
FMS = ["chronos", "timesfm", "patchtst"]


def load(model, ds):
    p = f"{RES}/{model}_zero_shot_{ds}{SUFFIX}.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)[["basin_id", "NSE"]].dropna()
    d = d[np.isfinite(d["NSE"])]
    return d.set_index("basin_id")["NSE"]


def boot_ci(x, n=10000, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    meds = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return np.percentile(meds, (1 - ci) / 2 * 100), np.percentile(meds, (1 + ci) / 2 * 100)


rows, pvals, pkeys, all_med = [], [], [], {}
for ds in DATASETS:
    pers = load("persistence", ds)
    rec = {"dataset": ds, "region": REGION[ds]}
    for m in FMS:
        s = load(m, ds)
        if s is None:
            rec[m] = None
            continue
        med = float(np.median(s))
        lo, hi = boot_ci(s.values)
        rec[m] = (med, lo, hi, len(s))
        all_med.setdefault(m, []).append(med)
        if pers is not None:
            common = s.index.intersection(pers.index)
            if len(common) >= 10:
                try:
                    _, p = stats.wilcoxon(s.loc[common].values - pers.loc[common].values)
                except Exception:
                    p = np.nan
                pvals.append(p); pkeys.append((ds, m))
    if pers is not None:
        med = float(np.median(pers)); lo, hi = boot_ci(pers.values)
        rec["persistence"] = (med, lo, hi, len(pers))
        all_med.setdefault("persistence", []).append(med)
    else:
        rec["persistence"] = None
    rows.append(rec)

# Benjamini-Hochberg
bh = {}
valid = [(k, p) for k, p in zip(pkeys, pvals) if np.isfinite(p)]
if valid:
    order = sorted(range(len(valid)), key=lambda i: valid[i][1])
    M = len(valid)
    for rank, i in enumerate(order, 1):
        bh[valid[i][0]] = min(valid[i][1] * M / rank, 1.0)


def stars(p):
    if p is None or not np.isfinite(p): return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else " (ns)"


def fmt(c):
    if c is None: return "   --- (pending)"
    med, lo, hi, n = c
    return f"{med:.3f} [{lo:.3f},{hi:.3f}] n={n}"

print(f"\n{'Dataset':12s} {'Region':11s} {'Chronos':28s} {'TimesFM':28s} {'PatchTST':22s} {'Persistence':22s}")
for r in rows:
    line = f"{r['dataset']:12s} {r['region']:11s} "
    for m in ["chronos", "timesfm", "patchtst", "persistence"]:
        c = r[m]
        cell = fmt(c)
        if m in FMS and c is not None:
            cell += stars(bh.get((r["dataset"], m)))
        line += f"{cell:30s} "
    print(line)

print("\n--- Overall median-NSE ranges (completed datasets only) ---")
for m in ["chronos", "timesfm", "patchtst", "persistence"]:
    if m in all_med and all_med[m]:
        print(f"  {m:12s}: {min(all_med[m]):.3f} - {max(all_med[m]):.3f}  ({len(all_med[m])} datasets)")
fm_all = [v for m in ["chronos", "timesfm"] for v in all_med.get(m, [])]
if fm_all:
    print(f"  FM (Chronos+TimesFM) overall NSE range: {min(fm_all):.3f} - {max(fm_all):.3f}")
tot = sum(r['timesfm'][3] for r in rows if r['timesfm']) if any(r['timesfm'] for r in rows) else 0
print(f"\n  Total basins (TimesFM, completed datasets): {tot}")
