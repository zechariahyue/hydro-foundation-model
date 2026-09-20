#!/usr/bin/env python3
"""
Adversarial checks on the "FM adds value in low-memory basins" finding.

Three things could make that result an artefact rather than hydrology:

  (1) CIRCULARITY. For a stationary series the lag-1 persistence forecast satisfies
      NSE_pers = 2*rho1 - 1 analytically. So persistence NSE *is* lag-1
      autocorrelation, rescaled -- correlating (NSE_fm - NSE_pers) against AC1 is
      partly correlating a quantity with itself.

  (2) CEILING EFFECT. Where AC1 is high, persistence already scores ~0.97, so no
      model can win by much. dNSE is compressed toward zero by construction, not
      by hydrology.

  (3) REGRESSION TO THE MEAN. In basins where persistence happens to score badly,
      *any* alternative looks better. A skill-less model would show the same trend.

CONTROLS:
  * Verify the NSE_pers = 2*AC1 - 1 identity empirically.
  * Re-run the stratification on the PERSISTENCE SKILL SCORE,
        PSS = 1 - MSE_fm/MSE_pers = 1 - (1-NSE_fm)/(1-NSE_pers),
    which is ceiling-free (both models are scored on identical targets, so the
    observed variance cancels exactly). PSS > 0 means the model removes error that
    persistence could not, regardless of how good persistence already was.
  * NULL CONTROL: run the identical stratification for PatchTST (a model with
    ~zero skill). If PatchTST shows the same "wins more in flashy basins" trend,
    the trend is regression to the mean, not information.
  * Paired Wilcoxon within each stratum.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

RES = Path(os.environ.get("RESULTS_DIR", "results"))
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def pss(nse_model, nse_ref):
    """Persistence skill score. Both models scored on identical targets => Var cancels.
    Undefined when persistence is already perfect (NSE_ref -> 1)."""
    denom = 1.0 - nse_ref
    out = 1.0 - (1.0 - nse_model) / denom
    out[np.abs(denom) < 1e-6] = np.nan
    return out


df = pd.read_csv(RES / "basin_attributes.csv")
df = df[df["nse_timesfm"].notna() & df["nse_persistence"].notna()].copy()

say("=" * 78)
say("(1) CIRCULARITY CHECK:  is  NSE_persistence == 2*AC1 - 1 ?")
say("=" * 78)
d = df.dropna(subset=["ac1", "nse_persistence"])
pred = 2 * d["ac1"] - 1
r = stats.pearsonr(pred, d["nse_persistence"])
say(f"  n = {len(d)}")
say(f"  Pearson r( 2*AC1-1 , NSE_pers ) = {r[0]:.4f}")
say(f"  median |2*AC1-1  -  NSE_pers|   = {np.median(np.abs(pred - d['nse_persistence'])):.4f}")
say(f"  Spearman rho(AC1, NSE_pers)     = {stats.spearmanr(d['ac1'], d['nse_persistence'])[0]:.4f}")
say("")
say("  => If r ~ 1, AC1 and persistence-NSE are the SAME quantity. Any 'FM beats")
say("     persistence where AC1 is low' result is then partly definitional, and")
say("     must be re-tested on a ceiling-free skill score (below).")
say("")

# ---------------------------------------------------------------------------
say("=" * 78)
say("(2)+(3) CEILING-FREE RE-TEST + NULL CONTROL")
say("=" * 78)
say("PSS = 1 - (1-NSE_model)/(1-NSE_pers) = fraction of persistence's squared error")
say("      that the model removes.  PSS>0 = genuine information beyond lag-1.")
say("      PatchTST is the NULL CONTROL: it has ~zero skill, so if it also shows a")
say("      rising trend toward flashy basins, the trend is an artefact.")
say("")

for m in ["timesfm", "chronos", "patchtst"]:
    if f"nse_{m}" in df.columns:
        df[f"pss_{m}"] = pss(df[f"nse_{m}"], df["nse_persistence"])

d = df.dropna(subset=["ac1"]).copy()
qs = d["ac1"].quantile([0.25, 0.5, 0.75]).values
d["mem_bin"] = pd.cut(d["ac1"], [-np.inf, qs[0], qs[1], qs[2], np.inf],
                      labels=["Q1 flashy", "Q2", "Q3", "Q4 damped"])

rows = []
for b, g in d.groupby("mem_bin", observed=True):
    row = {"bin": b, "n": len(g), "median_ac1": g["ac1"].median(),
           "nse_pers": g["nse_persistence"].median(),
           "nse_tfm": g["nse_timesfm"].median()}
    for m in ["timesfm", "chronos", "patchtst"]:
        col = f"pss_{m}"
        if col not in g:
            continue
        s = g[col].dropna()
        # clip only for a robust median summary; sign test uses raw values
        row[f"medPSS_{m}"] = s.median()
        row[f"pctPos_{m}"] = 100.0 * (s > 0).mean()
    # paired Wilcoxon TimesFM vs persistence on raw NSE within the stratum
    p = g.dropna(subset=["nse_timesfm", "nse_persistence"])
    if len(p) > 20:
        try:
            row["wilcoxon_p"] = stats.wilcoxon(p["nse_timesfm"], p["nse_persistence"])[1]
        except Exception:
            row["wilcoxon_p"] = np.nan
    rows.append(row)

t = pd.DataFrame(rows)
say(f"{'bin':12s} {'n':>5s} {'AC1':>6s} {'NSEpers':>8s} {'NSEtfm':>7s} "
    f"{'PSS_tfm':>8s} {'%>0':>6s} | {'PSS_chr':>8s} {'%>0':>6s} | {'PSS_pTST':>9s} {'%>0':>6s} | {'wilcox_p':>9s}")
for r in t.itertuples():
    say(f"{str(r.bin):12s} {r.n:5d} {r.median_ac1:6.3f} {r.nse_pers:8.3f} {r.nse_tfm:7.3f} "
        f"{r.medPSS_timesfm:8.3f} {r.pctPos_timesfm:5.1f}% | "
        f"{r.medPSS_chronos:8.3f} {r.pctPos_chronos:5.1f}% | "
        f"{r.medPSS_patchtst:9.3f} {r.pctPos_patchtst:5.1f}% | {r.wilcoxon_p:9.2e}")

say("")
say("READ THIS TABLE AS FOLLOWS:")
say("  * If PSS_tfm is ~flat across bins -> the dNSE trend was a CEILING ARTEFACT.")
say("  * If PSS_tfm rises toward flashy basins AND PSS_patchtst does not")
say("    -> TimesFM genuinely extracts information beyond lag-1 in flashy basins.")
say("  * If PSS_patchtst ALSO rises -> regression to the mean. Finding is dead.")
say("")

# Spearman of PSS vs the signatures, ceiling-free
say("=" * 78)
say("(4) SPEARMAN of PSS (ceiling-free) vs catchment signatures  [correct BH]")
say("=" * 78)
PRED = ["ac1", "ac7", "bfi", "rbi", "q_cv", "fdc_slope", "zero_q_frac",
        "aridity", "frac_snow", "p_seasonality", "area", "elev_mean", "regulation"]
rows = []
for m in ["timesfm", "patchtst"]:
    for p in PRED:
        dd = df[[f"pss_{m}", p]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(dd) < 100:
            continue
        rho, pv = stats.spearmanr(dd[p], dd[f"pss_{m}"])
        rows.append({"model": m, "predictor": p, "n": len(dd), "rho": rho, "p": pv})
sp = pd.DataFrame(rows)
# CORRECT Benjamini-Hochberg: sort ascending, p*m/rank, then REVERSE cumulative min
sp = sp.sort_values("p").reset_index(drop=True)
mtot = len(sp)
adj = sp["p"].values * mtot / (np.arange(mtot) + 1)
sp["p_bh"] = np.minimum.accumulate(adj[::-1])[::-1].clip(max=1.0)

for m in ["timesfm", "patchtst"]:
    s = sp[sp.model == m].sort_values("rho")
    say(f"\n--- PSS ({m}) vs signatures ---")
    say(f"{'predictor':16s} {'rho':>7s} {'p_BH':>10s} {'n':>6s}  sig")
    for r in s.itertuples():
        star = "***" if r.p_bh < 1e-3 else "**" if r.p_bh < 0.01 else "*" if r.p_bh < 0.05 else "ns"
        say(f"{r.predictor:16s} {r.rho:+7.3f} {r.p_bh:10.2e} {r.n:6d}  {star}")


# (5) PARTIAL correlation controlling for baseline strength (2026-08-17 review, MAJOR #1)
say("")
say("=" * 78)
say("(5) PARTIAL SPEARMAN: does the memory effect survive conditioning on NSE_pers?")
say("    partial rho(Y, ac1 | NSE_pers) = Pearson of rank-residuals after regressing")
say("    rank(Y) and rank(ac1) on rank(NSE_pers). Also a within-decile check.")
say("=" * 78)


def partial_spearman(y, x, z):
    r = np.column_stack([stats.rankdata(v) for v in (y, x, z)]).astype(float)
    Z = np.column_stack([np.ones(len(r)), r[:, 2]])
    res = [v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0] for v in (r[:, 0], r[:, 1])]
    rho, pv = stats.pearsonr(res[0], res[1])
    return rho, pv


for m in ["timesfm", "chronos", "patchtst"]:
    dd = df[[f"nse_{m}", "nse_persistence", "ac1"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    dd["dnse"] = dd[f"nse_{m}"] - dd["nse_persistence"]
    dd["pss"] = pss(dd[f"nse_{m}"].values, dd["nse_persistence"].values)
    dd["win"] = (dd["dnse"] > 0).astype(float)
    say(f"\n--- {m}  (n={len(dd)}) ---")
    for yname in ["dnse", "pss", "win"]:
        d2 = dd.dropna(subset=[yname])
        raw, _ = stats.spearmanr(d2["ac1"], d2[yname])
        prho, pp = partial_spearman(d2[yname].values, d2["ac1"].values, d2["nse_persistence"].values)
        say(f"  {yname:5s} vs ac1 : raw rho={raw:+.3f}   partial rho | NSE_pers = {prho:+.3f}  (p={pp:.2e}, n={len(d2)})")
    # within-decile of NSE_pers: Spearman(dnse, ac1) and win rate spread
    dd["dec"] = pd.qcut(dd["nse_persistence"].rank(method="first"), 10, labels=False)
    say("  within NSE_pers deciles (dnse vs ac1):  decile  rho    win%")
    for k, g in dd.groupby("dec"):
        if len(g) < 30:
            continue
        r_, _ = stats.spearmanr(g["ac1"], g["dnse"])
        say(f"      {k:2d}  {r_:+.3f}  {100*g['win'].mean():5.1f}   (NSE_pers {g['nse_persistence'].min():+.2f}..{g['nse_persistence'].max():+.2f})")
say("")
say("READ: if partial rho keeps the sign of raw rho and stays clearly non-zero,")
say("      the memory effect is not baseline-conditioning in disguise.")

sp.to_csv(RES / "hydro_controls_pss_spearman.csv", index=False)
t.to_csv(RES / "hydro_controls_ceiling_check.csv", index=False)
(RES / "CONTROLS_DIAGNOSTICS.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\n[OK] wrote {RES/'CONTROLS_DIAGNOSTICS.txt'}")
