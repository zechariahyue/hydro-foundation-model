#!/usr/bin/env python3
"""
Does TimesFM's low-memory advantage survive a FITTED autoregressive baseline?

Joins ar{1,2,5}_zero_shot_<DS>_l5s7.csv (from models/run_ar_baseline.py; AR(p)+intercept
fitted by OLS on each 512-day context window, same target points as every other model)
with basin_attributes.csv, and reports by archive and by rho_1 quartile:
  * median NSE of persistence / AR(p) / TimesFM
  * % basins where AR(p) beats persistence, and where TimesFM beats AR(p)
  * PSS of TimesFM relative to AR(2) (ceiling-free, identical targets)
Writes experiments/results/AR_BASELINE_REPORT.txt and ar_baseline_quartiles.csv.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

RES = Path(__file__).resolve().parents[1] / "results"
DATASETS = ["CAMELS-US", "CAMELS-BR", "CAMELS-CL", "CAMELS-AUS", "LamaH-CE", "CAMELS-IND"]
ORDERS = [1, 2, 5]
lines = []


def say(s=""):
    print(s); lines.append(s)


def load_ar():
    frames = []
    for ds in DATASETS:
        for p in ORDERS:
            f = RES / f"ar{p}_zero_shot_{ds}_l5s7.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f, dtype={"basin_id": str})[["basin_id", "dataset", "NSE"]]
            d = d.rename(columns={"basin_id": "bid"}); d["order"] = f"nse_ar{p}"
            frames.append(d)
    if not frames:
        raise SystemExit("no AR result files found")
    long = pd.concat(frames, ignore_index=True)
    return long.pivot_table(index=["dataset", "bid"], columns="order", values="NSE").reset_index()


ar = load_ar()
attr = pd.read_csv(RES / "basin_attributes.csv", dtype={"bid": str})
# CAMELS-US ids are 8-digit with leading zeros; basin_attributes.csv stores them as ints
for d_ in (attr, ar):
    us = d_.dataset == "CAMELS-US"
    d_.loc[us, "bid"] = d_.loc[us, "bid"].astype(str).str.zfill(8)
    # LamaH-CE ids are "ID_<n>" in the result files but bare "<n>" in basin_attributes.csv
    la = d_.dataset == "LamaH-CE"
    d_.loc[la, "bid"] = d_.loc[la, "bid"].astype(str).str.replace("ID_", "", regex=False)
df = attr.merge(ar, on=["dataset", "bid"], how="inner")
df = df.replace([np.inf, -np.inf], np.nan)
have = [p for p in ORDERS if f"nse_ar{p}" in df]
df = df.dropna(subset=["nse_timesfm", "nse_persistence", "ac1"] + [f"nse_ar{p}" for p in have])
say(f"Joined {len(df)} basins with AR orders {have}  (archives: {sorted(df.dataset.unique())})")


def pss(m, ref):
    den = 1 - ref
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1 - (1 - m) / den
    return np.where(np.abs(den) < 1e-6, np.nan, out)


say("\n(1) MEDIAN NSE BY ARCHIVE")
say(f"{'archive':12s} {'n':>5s} {'pers':>7s} " + " ".join(f"{'AR'+str(p):>7s}" for p in have) + f" {'TFM':>7s} | %AR2>pers  %TFM>AR2  %TFM>bestAR")
rows = []
for ds, g in list(df.groupby("dataset")) + [("ALL", df)]:
    best_ar = g[[f"nse_ar{p}" for p in have]].max(axis=1)
    r = {"archive": ds, "n": len(g), "pers": g.nse_persistence.median(), "tfm": g.nse_timesfm.median()}
    for p in have:
        r[f"ar{p}"] = g[f"nse_ar{p}"].median()
    r["pct_ar2_gt_pers"] = 100 * (g.nse_ar2 > g.nse_persistence).mean() if "nse_ar2" in g else np.nan
    r["pct_tfm_gt_ar2"] = 100 * (g.nse_timesfm > g.nse_ar2).mean() if "nse_ar2" in g else np.nan
    r["pct_tfm_gt_bestar"] = 100 * (g.nse_timesfm > best_ar).mean()
    rows.append(r)
    say(f"{ds:12s} {r['n']:5d} {r['pers']:7.3f} " + " ".join(f"{r['ar'+str(p)]:7.3f}" for p in have)
        + f" {r['tfm']:7.3f} | {r['pct_ar2_gt_pers']:8.1f}  {r['pct_tfm_gt_ar2']:8.1f}  {r['pct_tfm_gt_bestar']:9.1f}")
pd.DataFrame(rows).to_csv(RES / "ar_baseline_by_archive.csv", index=False)

say("\n(2) BY rho_1 QUARTILE (pooled; quartiles on the joined sample)")
df["q"] = pd.qcut(df.ac1, 4, labels=["Q1 flashiest", "Q2", "Q3", "Q4 damped"])
hdr = f"{'bin':13s} {'n':>5s} {'rho1':>5s} {'pers':>6s} {'AR2':>6s} {'AR5':>6s} {'TFM':>6s} | %AR2>pers %AR5>pers %TFM>pers %TFM>AR2 %TFM>AR5 | PSS_TFM|pers PSS_TFM|AR2 PSS_AR2|pers"
say(hdr)
qrows = []
for q, g in df.groupby("q", observed=True):
    r = dict(bin=str(q), n=len(g), rho1=g.ac1.median(), pers=g.nse_persistence.median(),
             ar2=g.nse_ar2.median(), ar5=g.nse_ar5.median() if "nse_ar5" in g else np.nan, tfm=g.nse_timesfm.median(),
             pct_ar2_pers=100 * (g.nse_ar2 > g.nse_persistence).mean(),
             pct_ar5_pers=100 * (g.nse_ar5 > g.nse_persistence).mean() if "nse_ar5" in g else np.nan,
             pct_tfm_pers=100 * (g.nse_timesfm > g.nse_persistence).mean(),
             pct_tfm_ar2=100 * (g.nse_timesfm > g.nse_ar2).mean(),
             pct_tfm_ar5=100 * (g.nse_timesfm > g.nse_ar5).mean() if "nse_ar5" in g else np.nan,
             pss_tfm_pers=np.nanmedian(pss(g.nse_timesfm.values, g.nse_persistence.values)),
             pss_tfm_ar2=np.nanmedian(pss(g.nse_timesfm.values, g.nse_ar2.values)),
             pss_ar2_pers=np.nanmedian(pss(g.nse_ar2.values, g.nse_persistence.values)))
    # paired Wilcoxon TimesFM vs AR2
    r["wilcoxon_p_tfm_ar2"] = stats.wilcoxon(g.nse_timesfm, g.nse_ar2).pvalue
    qrows.append(r)
    say(f"{r['bin']:13s} {r['n']:5d} {r['rho1']:5.2f} {r['pers']:6.3f} {r['ar2']:6.3f} {r['ar5']:6.3f} {r['tfm']:6.3f} | "
        f"{r['pct_ar2_pers']:8.1f} {r['pct_ar5_pers']:8.1f} {r['pct_tfm_pers']:8.1f} {r['pct_tfm_ar2']:7.1f} {r['pct_tfm_ar5']:7.1f} | "
        f"{r['pss_tfm_pers']:+11.3f} {r['pss_tfm_ar2']:+11.3f} {r['pss_ar2_pers']:+11.3f}   (p_TFMvsAR2={r['wilcoxon_p_tfm_ar2']:.1e})")
pd.DataFrame(qrows).to_csv(RES / "ar_baseline_quartiles.csv", index=False)

# LaTeX table for the SI (input by supplement.tex)
tex = [r"\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
       r"Memory bin & $n$ & NSE$_{\text{pers}}$ & NSE$_{\text{AR2}}$ & NSE$_{\text{TFM}}$ & AR2$>$pers & TFM$>$pers & TFM$>$AR2 & PSS$_{\text{TFM}\mid\text{AR2}}$ \\",
       r"\midrule"]
for r in qrows:
    lab = r["bin"].replace("Q1 flashiest", "Q1 (flashiest)").replace("Q4 damped", "Q4 (most damped)")
    tex.append(f"{lab} & {r['n']:,} & {r['pers']:.3f} & {r['ar2']:.3f} & {r['tfm']:.3f} & "
               f"{r['pct_ar2_pers']:.1f}\% & {r['pct_tfm_pers']:.1f}\% & {r['pct_tfm_ar2']:.1f}\% & ${r['pss_tfm_ar2']:+.3f}$ \\\\")
tex += [r"\bottomrule", r"\end{tabular}"]
(RES / "table_ar_baseline.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

say("\n(3) Spearman(rho_1, NSE) for each reference, pooled")
for c in ["nse_persistence"] + [f"nse_ar{p}" for p in have] + ["nse_timesfm"]:
    rho, _ = stats.spearmanr(df.ac1, df[c])
    say(f"  {c:16s} rho = {rho:+.3f}")

say("\nREAD: if %TFM>AR2 stays above 50% in Q1 and TimesFM's PSS relative to AR2 is positive there,")
say("      the low-memory advantage is not something a fitted linear model on the same context delivers.")
(RES / "AR_BASELINE_REPORT.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"[OK] wrote {RES / 'AR_BASELINE_REPORT.txt'}")
