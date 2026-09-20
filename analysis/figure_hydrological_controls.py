#!/usr/bin/env python3
"""
Figure: hydrological controls on zero-shot foundation-model skill.

(a) Zero-shot skill is governed by the memory of the hydrograph.
(b) SIGN TEST -- fraction of basins in which each model beats lag-1 persistence,
    by flow-memory quartile. This is the ceiling-free headline: a win is a win
    regardless of how much headroom persistence left.
(c) Persistence skill score by flashiness quartile (how much of persistence's
    squared error the FM actually removes).
(d) Flow dynamics dominate; static catchment descriptors barely matter.

Fonts are deliberately large -- reviewers flagged unreadable figure text.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

RES = Path(os.environ.get("RESULTS_DIR", "results"))
FIG = Path(os.environ.get("FIGURES_DIR", "figures"))

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14.5,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11.5,
    "axes.spines.top": False, "axes.spines.right": False,
})

C_TFM, C_CHR, C_PTS, C_PERS = "#0072B2", "#E69F00", "#999999", "#D55E00"

df = pd.read_csv(RES / "basin_attributes.csv")
df = df[df["nse_timesfm"].notna() & df["nse_persistence"].notna()].copy()


def pss(m, ref):
    den = 1.0 - ref
    out = 1.0 - (1.0 - m) / den
    out[np.abs(den) < 1e-6] = np.nan
    return out


for m in ["timesfm", "chronos", "patchtst"]:
    df[f"pss_{m}"] = pss(df[f"nse_{m}"], df["nse_persistence"])
    df[f"win_{m}"] = (df[f"nse_{m}"] > df["nse_persistence"]).astype(float)

qs = df["ac1"].quantile([0.25, 0.5, 0.75]).values
df["mem_q"] = pd.cut(df["ac1"], [-np.inf, *qs, np.inf], labels=["Q1", "Q2", "Q3", "Q4"])
rq = df["rbi"].quantile([0.25, 0.5, 0.75]).values
df["rbi_q"] = pd.cut(df["rbi"], [-np.inf, *rq, np.inf], labels=["Q1", "Q2", "Q3", "Q4"])

fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))

# ---- (a) skill vs memory -------------------------------------------------
ax = axes[0, 0]
d = df.dropna(subset=["ac1", "nse_timesfm"])
hb = ax.hexbin(d["ac1"], d["nse_timesfm"].clip(-0.5, 1.0), gridsize=45,
               cmap="Blues", bins="log", mincnt=1, linewidths=0)
cb = fig.colorbar(hb, ax=ax, pad=0.02)
cb.set_label("basins (log)", fontsize=11)
# median NSE in AC1 bins
bins = np.linspace(d["ac1"].min(), 1.0, 21)
ctr, med = [], []
for i in range(len(bins) - 1):
    s = d[(d["ac1"] >= bins[i]) & (d["ac1"] < bins[i + 1])]
    if len(s) >= 25:
        ctr.append(0.5 * (bins[i] + bins[i + 1]))
        med.append(s["nse_timesfm"].median())
ax.plot(ctr, med, color=C_TFM, lw=3, label="TimesFM median")
xx = np.linspace(0.4, 1.0, 50)
ax.plot(xx, 2 * xx - 1, ls="--", lw=2.5, color=C_PERS, label="persistence ($2\\rho_1-1$)")
rho = stats.spearmanr(d["ac1"], d["nse_timesfm"])[0]
ax.text(0.04, 0.95, f"Spearman $\\rho$ = {rho:+.3f}\n$n$ = {len(d):,} basins",
        transform=ax.transAxes, va="top", fontsize=12.5,
        bbox=dict(fc="white", ec="0.7", alpha=0.9))
ax.set_xlabel("Lag-1 flow autocorrelation $\\rho_1$  (hydrograph memory)")
ax.set_ylabel("TimesFM zero-shot NSE")
ax.set_title("(a) Zero-shot skill tracks hydrograph memory", loc="left", fontweight="bold")
ax.set_ylim(-0.55, 1.02)
ax.legend(loc="lower right", framealpha=0.9)

# ---- (b) SIGN TEST: % basins beating persistence -------------------------
ax = axes[0, 1]
labs = ["Q1\nflashy", "Q2", "Q3", "Q4\ndamped"]
x = np.arange(4)
w = 0.26
for i, (m, c, nm) in enumerate([("timesfm", C_TFM, "TimesFM"),
                                ("chronos", C_CHR, "Chronos"),
                                ("patchtst", C_PTS, "PatchTST")]):
    v = [100 * df.loc[df.mem_q == q, f"win_{m}"].mean() for q in ["Q1", "Q2", "Q3", "Q4"]]
    b = ax.bar(x + (i - 1) * w, v, w, color=c, label=nm, edgecolor="white", lw=0.5)
    if m == "timesfm":
        ax.bar_label(b, fmt="%.0f%%", fontsize=11, fontweight="bold", padding=2)
ax.axhline(50, color="k", ls="--", lw=1.8)
ax.text(3.42, 52, "50%\n(coin flip)", fontsize=10.5, ha="center", va="bottom")
ax.set_xticks(x)
ax.set_xticklabels(labs)
ax.set_xlabel("Flow-memory quartile ($\\rho_1$)")
ax.set_ylabel("% of basins beating lag-1 persistence")
ax.set_title("(b) TimesFM adds value only where memory is weak", loc="left", fontweight="bold")
ax.set_ylim(0, 85)
ax.legend(loc="upper right", framealpha=0.9)

# ---- (c) persistence skill score by flashiness ---------------------------
ax = axes[1, 0]
med = [df.loc[df.rbi_q == q, "pss_timesfm"].median() for q in ["Q1", "Q2", "Q3", "Q4"]]
lo = [df.loc[df.rbi_q == q, "pss_timesfm"].quantile(0.25) for q in ["Q1", "Q2", "Q3", "Q4"]]
hi = [df.loc[df.rbi_q == q, "pss_timesfm"].quantile(0.75) for q in ["Q1", "Q2", "Q3", "Q4"]]
cols = [C_TFM if v > 0 else C_PERS for v in med]
ax.bar(range(4), med, 0.62, color=cols, edgecolor="white",
       yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
       error_kw=dict(lw=1.4, capsize=5, ecolor="0.35"))
ax.axhline(0, color="k", lw=1.5)
ax.set_xticks(range(4))
ax.set_xticklabels(["Q1\nleast flashy", "Q2", "Q3", "Q4\nmost flashy"])
ax.set_xlabel("Richards--Baker flashiness quartile")
ax.set_ylabel("TimesFM persistence skill score\n(median, IQR)")
ax.set_title("(c) Error removed beyond lag-1, ceiling-free", loc="left", fontweight="bold")
ax.text(0.03, 0.95, "PSS > 0: FM removes error\nthat persistence could not",
        transform=ax.transAxes, va="top", fontsize=11.5,
        bbox=dict(fc="white", ec="0.7", alpha=0.9))

# ---- (d) dynamics vs descriptors ----------------------------------------
ax = axes[1, 1]
DYN = [("ac1", "Lag-1 autocorr."), ("rbi", "Flashiness"), ("q_skew", "Flow skewness"),
       ("bfi", "Baseflow index"), ("q_cv", "Flow CV"), ("zero_q_frac", "Zero-flow frac.")]
STAT = [("area", "Catchment area"), ("elev_mean", "Mean elevation"),
        ("slope_mean", "Mean slope"), ("frac_snow", "Snow fraction"),
        ("aridity", "Aridity"), ("regulation", "Regulation")]
names, vals, cols = [], [], []
for k, lab in DYN + STAT:
    d = df[[k, "nse_timesfm"]].dropna()
    if len(d) < 100:
        continue
    names.append(lab)
    vals.append(abs(stats.spearmanr(d[k], d["nse_timesfm"])[0]))
    cols.append(C_TFM if (k, lab) in DYN else "#BBBBBB")
order = np.argsort(vals)
ax.barh([names[i] for i in order], [vals[i] for i in order],
        color=[cols[i] for i in order], edgecolor="white")
ax.set_xlabel("|Spearman $\\rho$| with TimesFM zero-shot NSE")
ax.set_title("(d) Flow dynamics dominate; catchment\n      descriptors barely matter",
             loc="left", fontweight="bold")
h = [plt.Rectangle((0, 0), 1, 1, color=C_TFM), plt.Rectangle((0, 0), 1, 1, color="#BBBBBB")]
ax.legend(h, ["Flow signature (dynamic)", "Catchment attribute (static)"],
          loc="lower right", framealpha=0.95)
ax.set_xlim(0, 0.92)

fig.tight_layout(pad=1.6)
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"figure_hydrological_controls.{ext}", dpi=300, bbox_inches="tight")
print(f"[OK] wrote {FIG/'figure_hydrological_controls.png'}")

# ---- LaTeX table ---------------------------------------------------------
rows = []
for q, lab in zip(["Q1", "Q2", "Q3", "Q4"],
                  ["Q1 (flashiest)", "Q2", "Q3", "Q4 (most damped)"]):
    s = df[df.mem_q == q]
    rows.append((lab, len(s), s["ac1"].median(), s["nse_persistence"].median(),
                 s["nse_timesfm"].median(), s["pss_timesfm"].median(),
                 100 * s["win_timesfm"].mean(), 100 * s["win_chronos"].mean(),
                 100 * s["win_patchtst"].mean()))
tex = [
    r"\begin{table}[t]", r"\centering",
    r"\caption{Hydrological control on zero-shot foundation-model skill. Basins "
    r"($n=6{,}569$) are binned by the lag-1 autocorrelation of their observed hydrograph. "
    r"PSS is the persistence skill score, $1-(1-\mathrm{NSE_{model}})/(1-\mathrm{NSE_{pers}})$; "
    r"because all models are scored on identical target points the observed variance cancels, "
    r"so PSS is free of the ceiling that compresses $\Delta$NSE in high-memory basins. The "
    r"final three columns are a sign test---the percentage of basins in which each model "
    r"beats lag-1 persistence---which is insensitive to both the ceiling and NSE's "
    r"unbounded negative tail.}",
    r"\label{tab:hydro_controls}",
    r"\begin{tabular}{lrrrrrrrr}", r"\hline",
    r"Memory bin & $n$ & $\rho_1$ & NSE$_{\text{pers}}$ & NSE$_{\text{TFM}}$ & "
    r"PSS$_{\text{TFM}}$ & \multicolumn{3}{c}{\% basins beating persistence} \\",
    r"\cline{7-9}", r" & & & & & & TimesFM & Chronos & PatchTST \\", r"\hline",
]
for r in rows:
    tex.append(f"{r[0]} & {r[1]:,} & {r[2]:.2f} & {r[3]:.3f} & {r[4]:.3f} & "
               f"{r[5]:+.3f} & {r[6]:.1f}\\% & {r[7]:.1f}\\% & {r[8]:.1f}\\% \\\\")
tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
(RES / "table_hydro_controls.tex").write_text("\n".join(tex), encoding="utf-8")
print(f"[OK] wrote {RES/'table_hydro_controls.tex'}")
