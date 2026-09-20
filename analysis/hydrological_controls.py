#!/usr/bin/env python3
"""
Hydrological controls on foundation-model skill.

The question a hydrologist actually wants answered is not "which model wins" but
"under what hydrological conditions does a pre-trained foundation model add value
over the trivial baseline, and when is local training still required?"

So the response variables are:
  nse_timesfm            -- absolute zero-shot skill
  d_tfm_pers             -- TimesFM minus lag-1 persistence  (does the FM add value?)
  d_lstm_tfm             -- trained LSTM minus TimesFM       (is local training worth it?)

and the predictors are catchment signatures / climate attributes.

Outputs:
  experiments/results/hydro_controls_spearman.csv
  experiments/results/hydro_controls_importance.csv
  experiments/results/hydro_controls_regimes.csv
  experiments/results/HYDRO_CONTROLS_REPORT.txt
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

RES = Path(os.environ.get("RESULTS_DIR", "results"))

PREDICTORS = [
    # Tier 1 -- flow signatures, computed identically for all basins, scale-invariant
    ("ac1",           "Lag-1 flow autocorrelation"),
    ("ac7",           "Lag-7 flow autocorrelation"),
    ("ac30",          "Lag-30 flow autocorrelation"),
    ("bfi",           "Baseflow index (Lyne-Hollick)"),
    ("rbi",           "Richards-Baker flashiness"),
    ("q_cv",          "Flow coefficient of variation"),
    ("q_skew",        "Flow skewness"),
    ("fdc_slope",     "Flow-duration-curve slope"),
    ("q95_q50",       "Q95/Q50 ratio"),
    ("zero_q_frac",   "Zero-flow day fraction"),
    # Tier 2 -- archive catchment attributes
    ("aridity",       "Aridity (PET/P)"),
    ("p_mean",        "Mean precipitation"),
    ("p_seasonality", "Precipitation seasonality"),
    ("frac_snow",     "Snow fraction"),
    ("area",          "Catchment area"),
    ("elev_mean",     "Mean elevation"),
    ("slope_mean",    "Mean slope"),
    ("forest_frac",   "Forest fraction"),
    ("regulation",    "Regulation / disturbance index"),
]

TARGETS = [
    ("nse_timesfm",  "TimesFM zero-shot NSE"),
    ("d_tfm_pers",   "TimesFM - persistence (FM value-add)"),
    ("d_lstm_tfm",   "LSTM - TimesFM (value of local training)"),
]

lines = []


def say(s=""):
    print(s)
    lines.append(s)


def spearman_table(df):
    rows = []
    for tgt, tlab in TARGETS:
        for p, plab in PREDICTORS:
            d = df[[tgt, p]].dropna()
            if len(d) < 100:
                continue
            rho, pv = stats.spearmanr(d[p], d[tgt])
            rows.append({"target": tgt, "target_label": tlab, "predictor": p,
                         "predictor_label": plab, "n": len(d),
                         "spearman_rho": rho, "p_value": pv})
    out = pd.DataFrame(rows)
    # Benjamini-Hochberg across the whole family of tests.
    # Step-up: sort ascending, scale by m/rank, then take the cumulative minimum
    # from the LARGEST p downward so the adjusted values stay monotone.
    if len(out):
        m = len(out)
        out = out.sort_values("p_value").reset_index(drop=True)
        adj = out["p_value"].to_numpy() * m / (np.arange(m) + 1)
        out["p_bh"] = np.minimum.accumulate(adj[::-1])[::-1].clip(max=1.0)
    return out


def rf_importance(df, target):
    """Permutation importance from a gradient-boosted tree -- ranks controls allowing
    for nonlinearity and correlated predictors. Reported as a ranking, not a causal claim.

    NSE has an unbounded negative tail (the global LSTM reaches -302,970 in CAMELS-AUS),
    which makes a squared-error fit meaningless: an unclipped model scores a NEGATIVE
    held-out R2, i.e. worse than predicting the mean, and its importances are pure
    outlier-chasing. We therefore fit on the RANK-transformed target, which preserves the
    ordering of basins by skill while removing the tail's leverage.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split
    except ImportError:
        say("  [skip] scikit-learn not available")
        return pd.DataFrame()

    cols = [p for p, _ in PREDICTORS]
    d = df[cols + [target]].replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    keep = [c for c in cols if d[c].notna().sum() > 0.5 * len(d)]   # HistGBR handles NaN
    if len(d) < 300 or not keep:
        return pd.DataFrame()

    X = d[keep]
    y = d[target].rank(pct=True)          # rank-transform: outlier-proof, order-preserving
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    m = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xtr, ytr)
    r2 = m.score(Xte, yte)
    pi = permutation_importance(m, Xte, yte, n_repeats=10, random_state=0)
    out = (pd.DataFrame({"predictor": keep, "importance": pi.importances_mean,
                         "std": pi.importances_std})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    out["target"] = target
    out["model_r2"] = r2
    out["n"] = len(d)
    return out


def regimes(df):
    """Stratify by hydrological regime and report where the FM beats persistence."""
    d = df.dropna(subset=["d_tfm_pers", "ac1"]).copy()
    rows = []

    def block(name, mask, label):
        s = d.loc[mask]
        if len(s) < 30:
            return
        rows.append({
            "stratifier": name, "bin": label, "n": len(s),
            "median_nse_timesfm": s["nse_timesfm"].median(),
            "median_nse_persistence": s["nse_persistence"].median(),
            "median_d_tfm_pers": s["d_tfm_pers"].median(),
            "pct_basins_fm_wins": 100.0 * (s["d_tfm_pers"] > 0).mean(),
        })

    # memory (lag-1 autocorrelation) quartiles
    qs = d["ac1"].quantile([0.25, 0.5, 0.75]).values
    block("flow_memory_ac1", d["ac1"] <= qs[0], "Q1 lowest memory (flashy)")
    block("flow_memory_ac1", (d["ac1"] > qs[0]) & (d["ac1"] <= qs[1]), "Q2")
    block("flow_memory_ac1", (d["ac1"] > qs[1]) & (d["ac1"] <= qs[2]), "Q3")
    block("flow_memory_ac1", d["ac1"] > qs[2], "Q4 highest memory (damped)")

    # flashiness quartiles
    if d["rbi"].notna().sum() > 200:
        qs = d["rbi"].quantile([0.25, 0.5, 0.75]).values
        block("flashiness_rbi", d["rbi"] <= qs[0], "Q1 least flashy")
        block("flashiness_rbi", (d["rbi"] > qs[0]) & (d["rbi"] <= qs[1]), "Q2")
        block("flashiness_rbi", (d["rbi"] > qs[1]) & (d["rbi"] <= qs[2]), "Q3")
        block("flashiness_rbi", d["rbi"] > qs[2], "Q4 most flashy")

    # aridity classes (UNEP convention)
    a = d.dropna(subset=["aridity"])
    if len(a) > 200:
        for lab, mask in [("humid (PET/P<0.75)", a["aridity"] < 0.75),
                          ("sub-humid (0.75-1.5)", (a["aridity"] >= 0.75) & (a["aridity"] < 1.5)),
                          ("semi-arid (1.5-3)", (a["aridity"] >= 1.5) & (a["aridity"] < 3)),
                          ("arid (>3)", a["aridity"] >= 3)]:
            s = a.loc[mask]
            if len(s) >= 30:
                rows.append({"stratifier": "aridity_class", "bin": lab, "n": len(s),
                             "median_nse_timesfm": s["nse_timesfm"].median(),
                             "median_nse_persistence": s["nse_persistence"].median(),
                             "median_d_tfm_pers": s["d_tfm_pers"].median(),
                             "pct_basins_fm_wins": 100.0 * (s["d_tfm_pers"] > 0).mean()})

    # snow influence
    sn = d.dropna(subset=["frac_snow"])
    if len(sn) > 200:
        for lab, mask in [("snow-free (<0.05)", sn["frac_snow"] < 0.05),
                          ("mixed (0.05-0.25)", (sn["frac_snow"] >= 0.05) & (sn["frac_snow"] < 0.25)),
                          ("snow-influenced (>=0.25)", sn["frac_snow"] >= 0.25)]:
            s = sn.loc[mask]
            if len(s) >= 30:
                rows.append({"stratifier": "snow_influence", "bin": lab, "n": len(s),
                             "median_nse_timesfm": s["nse_timesfm"].median(),
                             "median_nse_persistence": s["nse_persistence"].median(),
                             "median_d_tfm_pers": s["d_tfm_pers"].median(),
                             "pct_basins_fm_wins": 100.0 * (s["d_tfm_pers"] > 0).mean()})

    # intermittency
    for lab, mask in [("perennial (no zero-flow)", d["zero_q_frac"] <= 0.001),
                      ("intermittent (>1% zero-flow)", d["zero_q_frac"] > 0.01)]:
        block("intermittency", mask, lab)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(RES / "basin_attributes.csv")
    df = df[df["nse_timesfm"].notna()]
    say(f"Basins with TimesFM skill: {len(df)}")
    say("")

    say("=" * 78)
    say("SPEARMAN RANK CORRELATIONS (BH-corrected across the full family)")
    say("=" * 78)
    sp = spearman_table(df)
    sp.to_csv(RES / "hydro_controls_spearman.csv", index=False)
    for tgt, tlab in TARGETS:
        s = sp[sp.target == tgt].sort_values("spearman_rho")
        if not len(s):
            continue
        say(f"\n--- {tlab} ---")
        say(f"{'predictor':16s} {'rho':>7s} {'p_BH':>10s} {'n':>6s}   {'':s}")
        for r in s.itertuples():
            star = "***" if r.p_bh < 1e-3 else "**" if r.p_bh < 0.01 else "*" if r.p_bh < 0.05 else "ns"
            say(f"{r.predictor:16s} {r.spearman_rho:+7.3f} {r.p_bh:10.2e} {r.n:6d}   {star}")

    say("")
    say("=" * 78)
    say("PERMUTATION IMPORTANCE (gradient-boosted trees)")
    say("=" * 78)
    imps = []
    for tgt, tlab in TARGETS:
        say(f"\n--- {tlab} ---")
        imp = rf_importance(df, tgt)
        if not len(imp):
            say("  (insufficient data)")
            continue
        say(f"  held-out R2 = {imp['model_r2'].iloc[0]:.3f}  (n={imp['n'].iloc[0]})")
        for r in imp.head(8).itertuples():
            say(f"  {r.predictor:16s} {r.importance:+.4f}")
        imps.append(imp)
    if imps:
        pd.concat(imps).to_csv(RES / "hydro_controls_importance.csv", index=False)

    say("")
    say("=" * 78)
    say("REGIME STRATIFICATION -- where does the FM beat persistence?")
    say("=" * 78)
    rg = regimes(df)
    rg.to_csv(RES / "hydro_controls_regimes.csv", index=False)
    for strat, g in rg.groupby("stratifier", sort=False):
        say(f"\n--- {strat} ---")
        say(f"{'bin':30s} {'n':>5s} {'TFM':>7s} {'pers':>7s} {'dNSE':>7s} {'%FM wins':>9s}")
        for r in g.itertuples():
            say(f"{r.bin:30s} {r.n:5d} {r.median_nse_timesfm:7.3f} "
                f"{r.median_nse_persistence:7.3f} {r.median_d_tfm_pers:+7.3f} "
                f"{r.pct_basins_fm_wins:8.1f}%")

    (RES / "HYDRO_CONTROLS_REPORT.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] wrote {RES / 'HYDRO_CONTROLS_REPORT.txt'}")
