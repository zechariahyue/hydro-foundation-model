#!/usr/bin/env python3
"""
Figure + final statistics for the rainfall-ceiling test.

Uses the stride-1 per-basin table (rainfall_ceiling_v2_per_basin.csv), which is better
powered than the stride-7 version. The LINEAR oracle is reported as primary: the
gradient-boosted oracle generalises WORSE on held-out data (10.1% vs 15.7% in Q1), i.e. it
overfits, so the simpler instrument is the more trustworthy one. Reporting the weaker-
generalising model as if it were the stronger test would overstate the result.
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
C_WET, C_DRY, C_TFM, C_NULL = "#0072B2", "#E69F00", "#0072B2", "#BBBBBB"

d = pd.read_csv(RES / "rainfall_ceiling_v2_per_basin.csv").dropna(subset=["ac1", "err_wet"])
d["wet_dry"] = d.err_wet / d.err_dry
qs = d.ac1.quantile([0.25, 0.5, 0.75]).values
d["q"] = pd.cut(d.ac1, [-np.inf, *qs, np.inf], labels=["Q1", "Q2", "Q3", "Q4"])
QL = ["Q1\nflashy", "Q2", "Q3", "Q4\ndamped"]

out = []


def say(s=""):
    print(s)
    out.append(s)


say("=" * 78)
say("RAINFALL-CEILING TEST -- FINAL STATISTICS (stride 1)")
say("=" * 78)
say(f"n basins = {len(d)}  ({', '.join(f'{k}={v}' for k, v in d.dataset.value_counts().items())})")
say()
w = stats.wilcoxon(d.err_wet, d.err_dry)
say("H-a: TimesFM's error is concentrated on rain days")
say(f"  median |err| on rain days (P>1mm) = {d.err_wet.median():.3f} (per unit mean flow)")
say(f"  median |err| on dry  days (P<=0.1) = {d.err_dry.median():.3f}")
say(f"  median ratio = {d.wet_dry.median():.2f}x ; basins with wet>dry = "
    f"{100*(d.err_wet>d.err_dry).mean():.1f}% ; Wilcoxon p = {w[1]:.2e}")
say()
r = stats.spearmanr(d.ac1, d.wet_dry)
say("H-c: the concentration intensifies as hydrograph memory falls")
say(f"  Spearman(rho_1, wet/dry ratio) = {r[0]:+.3f}  p = {r[1]:.2e}")
for ds, g in d.groupby("dataset"):
    rr = stats.spearmanr(g.ac1, g.wet_dry)
    say(f"    {ds:12s} n={len(g):4d} median ratio={g.wet_dry.median():5.2f}x  "
        f"rho={rr[0]:+.3f} p={rr[1]:.2e}")
say()
say("H-b: how much of the residual can OBSERVED rainfall remove? (held-out)")
say(f"  {'bin':6s} {'linear':>8s} {'shuffled null':>14s}")
for q in ["Q1", "Q2", "Q3", "Q4"]:
    g = d[d.q == q]
    say(f"  {q:6s} {g.gain_lin.median():7.1%} {g.gain_lin_shuf.median():13.1%}")
rl = stats.spearmanr(d.ac1, d.gain_lin, nan_policy="omit")
say(f"  Spearman(rho_1, linear gain) = {rl[0]:+.3f} p={rl[1]:.2e}")
for ds, g in d.groupby("dataset"):
    rr = stats.spearmanr(g.ac1, g.gain_lin, nan_policy="omit")
    say(f"    {ds:12s} rho={rr[0]:+.3f} p={rr[1]:.2e}  median gain={g.gain_lin.median():.1%}")
say()
say("NOTE: the gradient-boosted oracle generalises WORSE than the linear one")
say(f"  (median held-out gain: linear {d.gain_lin.median():.1%} vs GBT {d.gain_gbt.median():.1%})")
say("  => it overfits; the linear oracle is the more trustworthy instrument.")
(RES / "RAINFALL_CEILING_FINAL.txt").write_text("\n".join(out), encoding="utf-8")

# ----------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

# (a) wet vs dry error by memory quartile
ax = axes[0]
x = np.arange(4)
wet = [d.loc[d.q == q, "err_wet"].median() for q in ["Q1", "Q2", "Q3", "Q4"]]
dry = [d.loc[d.q == q, "err_dry"].median() for q in ["Q1", "Q2", "Q3", "Q4"]]
ax.bar(x - 0.2, wet, 0.38, color=C_WET, label="rain days ($P>1$ mm)", edgecolor="white")
ax.bar(x + 0.2, dry, 0.38, color=C_DRY, label="dry days ($P\\leq0.1$ mm)", edgecolor="white")
# label with the median of the per-basin ratio (matches the table), not the ratio of medians
for i, q in enumerate(["Q1", "Q2", "Q3", "Q4"]):
    ax.text(i, wet[i] + 0.03, f"{d.loc[d.q == q, 'wet_dry'].median():.1f}$\\times$",
            ha="center", fontsize=12, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(QL)
ax.set_xlabel("Flow-memory quartile ($\\rho_1$)")
ax.set_ylabel("Median $|$error$|$ (per unit mean flow)")
ax.set_title("(a) TimesFM's error is a rain-day error", loc="left", fontweight="bold")
ax.legend(framealpha=0.9)

# (b) wet/dry ratio vs rho1
ax = axes[1]
for ds, c, mk in [("CAMELS-US", "#0072B2", "o"), ("CAMELS-AUS", "#D55E00", "^")]:
    g = d[d.dataset == ds]
    ax.scatter(g.ac1, g.wet_dry.clip(0, 40), s=13, c=c, alpha=0.45,
               marker=mk, edgecolors="none", label=f"{ds} ($n$={len(g)})")
bins = np.linspace(d.ac1.min(), 1.0, 16)
ctr, med = [], []
for i in range(len(bins) - 1):
    s = d[(d.ac1 >= bins[i]) & (d.ac1 < bins[i + 1])]
    if len(s) >= 15:
        ctr.append(0.5 * (bins[i] + bins[i + 1])); med.append(s.wet_dry.median())
ax.plot(ctr, med, color="black", lw=3, label="median")
ax.axhline(1, color="0.4", ls=":", lw=1.5)
ax.text(0.03, 0.28, f"Spearman $\\rho$ = {r[0]:+.3f}\n$p$ = {r[1]:.0e}",
        transform=ax.transAxes, va="top", fontsize=12.5,
        bbox=dict(fc="white", ec="0.7", alpha=0.9))
ax.set_xlabel("Lag-1 flow autocorrelation $\\rho_1$")
ax.set_ylabel("Rain-day / dry-day error ratio")
ax.set_ylim(0, 40)
ax.set_title("(b) Worse where memory is weak", loc="left", fontweight="bold")
ax.legend(loc="upper right", framealpha=0.95, markerscale=1.6)

# (c) oracle gain
ax = axes[2]
gl = [d.loc[d.q == q, "gain_lin"].median() * 100 for q in ["Q1", "Q2", "Q3", "Q4"]]
gs = [d.loc[d.q == q, "gain_lin_shuf"].median() * 100 for q in ["Q1", "Q2", "Q3", "Q4"]]
ax.bar(x - 0.2, gl, 0.38, color=C_TFM, label="observed rainfall", edgecolor="white")
ax.bar(x + 0.2, gs, 0.38, color=C_NULL, label="shuffled rainfall (null)", edgecolor="white")
ax.axhline(0, color="k", lw=1.2)
ax.set_xticks(x); ax.set_xticklabels(QL)
ax.set_xlabel("Flow-memory quartile ($\\rho_1$)")
ax.set_ylabel("% of held-out residual variance removed")
ax.set_title("(c) But rainfall alone does not close it", loc="left", fontweight="bold")
ax.legend(framealpha=0.9)

fig.tight_layout(pad=1.5)
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"figure_rainfall_ceiling.{ext}", dpi=300, bbox_inches="tight")
print(f"\n[OK] wrote {FIG/'figure_rainfall_ceiling.png'}")

# ----------------------------------------------------------------- LaTeX table
rows = []
for q, lab in zip(["Q1", "Q2", "Q3", "Q4"],
                  ["Q1 (flashiest)", "Q2", "Q3", "Q4 (most damped)"]):
    g = d[d.q == q]
    rows.append((lab, len(g), g.ac1.median(), g.err_wet.median(), g.err_dry.median(),
                 g.wet_dry.median(), g.gain_lin.median() * 100,
                 g.gain_lin_shuf.median() * 100))
tex = [r"\begin{table}[t]", r"\centering",
       r"\caption{Direct test of the rainfall ceiling on the "
       r"$1{,}230$ basins with daily precipitation (CAMELS-US, CAMELS-AUS). Errors are "
       r"TimesFM's one-day-ahead residuals, normalised by each basin's mean flow so that they "
       r"are comparable across archives. Rain days are $P>1$\,mm, dry days $P\leq0.1$\,mm. The "
       r"final two columns give the percentage of each basin's \emph{held-out} residual "
       r"variance removed by a rainfall-only linear correction fitted on the first half of the "
       r"record; the correction is given the \emph{observed} precipitation on the forecast day, "
       r"so it is an upper bound on what perfect rainfall knowledge could supply, not a "
       r"deployable model. The negative control is the \emph{same linear estimator} fitted to shuffled precipitation; pairing this oracle with a higher-capacity shuffled control would overstate the separation, so the null is matched to the estimator.}",
       r"\label{tab:rainfall_ceiling}",
       r"\begin{tabular}{lrrrrrrr}", r"\hline",
       r"Memory bin & $n$ & $\rho_1$ & $|e|$ rain & $|e|$ dry & ratio & "
       r"\multicolumn{2}{c}{\% residual removed} \\", r"\cline{7-8}",
       r" & & & & & & rainfall & shuffled \\", r"\hline"]
for r_ in rows:
    tex.append(f"{r_[0]} & {r_[1]:,} & {r_[2]:.2f} & {r_[3]:.3f} & {r_[4]:.3f} & "
               f"{r_[5]:.1f}$\\times$ & {r_[6]:.1f}\\% & {r_[7]:.1f}\\% \\\\")
tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
(RES / "table_rainfall_ceiling.tex").write_text("\n".join(tex), encoding="utf-8")
print(f"[OK] wrote {RES/'table_rainfall_ceiling.tex'}")
