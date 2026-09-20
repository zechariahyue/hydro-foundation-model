#!/usr/bin/env python3
"""
Replacement for the old 'system architecture' diagram.

The old figure showed an ETL pipeline fusing ERA5-Land, CHIRPS and HydroATLAS into the
models. That is not what the benchmark does -- every model is univariate and sees only
discharge -- and reviewers correctly flagged the figure as inconsistent with the text.
It also omitted the LSTM baseline entirely.

This draws the benchmark as it actually is: one input variable, five models, one protocol.
Large fonts throughout (reviewers flagged unreadable figure text).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

FIG = Path(os.environ.get("FIGURES_DIR", "figures"))

C_DATA, C_ZS, C_TRAIN, C_EVAL, C_PERS = "#0072B2", "#009E73", "#E69F00", "#555555", "#D55E00"

fig, ax = plt.subplots(figsize=(14, 7.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 56)
ax.axis("off")


def box(x, y, w, h, text, fc, fs=12.5, tc="white", bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                fc=fc, ec="white", lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.45)


def arrow(x1, y1, x2, y2, c="0.35"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=22,
                                 lw=2.0, color=c, shrinkA=2, shrinkB=2))


# ---- column 1: data ----
ax.text(11, 53.4, "DATA", ha="center", fontsize=13.5, fontweight="bold", color=C_DATA)
box(1, 33, 20, 15,
    "Six CAMELS-family archives\n\nCAMELS-US / BR / CL / AUS\nLamaH-CE / CAMELS-IND\n\n"
    "6,569 basins, 5 continents", C_DATA, fs=11.5)
box(1, 20.5, 20, 8.5,
    "THE ONLY MODEL INPUT\n\nDaily discharge $Q_{t-511..t}$", "#003f5c", fs=12)
ax.text(11, 15.6,
        "No climate forcing and no static\nattribute enters any forecast.\n"
        "ERA5-Land / CHIRPS / HydroATLAS\nare used ONLY in the covariate\n"
        "ablation and the controls analysis.",
        ha="center", va="top", fontsize=10.5, color="0.25", linespacing=1.5,
        bbox=dict(fc="#f2f2f2", ec="0.75", boxstyle="round,pad=0.5"))

arrow(21.5, 25, 30, 25)

# ---- column 2: models ----
ax.text(45, 53.4, "MODELS  (all univariate)", ha="center", fontsize=13.5,
        fontweight="bold", color="0.25")

ax.text(45, 49.2, "Zero-shot: no target-basin discharge updates any weight",
        ha="center", fontsize=11, color=C_ZS, fontweight="bold")
box(30, 41.5, 30, 6.6, "TimesFM        Chronos        PatchTST", C_ZS, fs=12)

ax.text(45, 37.6, "Few-shot: LoRA adapters on Chronos  ($f$ = 1 to 25% of record)",
        ha="center", fontsize=11, color="#7a5195", fontweight="bold")
box(30, 30.2, 30, 6.4, "Chronos + LoRA", "#7a5195", fs=12)

ax.text(45, 26.3, "Trained baselines", ha="center", fontsize=11,
        color=C_TRAIN, fontweight="bold")
box(30, 18.8, 30, 6.6, "Global LSTM  (input size = 1)", C_TRAIN, fs=12)
box(30, 10.4, 30, 6.6, "Lag-1 persistence  $\\hat{Q}_{t+1}=Q_t$", C_PERS, fs=12)

arrow(60.5, 25, 69, 25)

# ---- column 3: protocol / evaluation ----
ax.text(84.5, 53.4, "PROTOCOL & METRICS", ha="center", fontsize=13.5,
        fontweight="bold", color=C_EVAL)
box(69, 36, 30, 12,
    "ONE CONSISTENT PROTOCOL\n\nLast 5 years of each basin's\nrecord · stride 7\n"
    "ALL models scored on\nIDENTICAL target points", C_EVAL, fs=11.5)
box(69, 24, 30, 9.5,
    "Skill:  NSE · KGE · RMSE\nBias / shape:  PBIAS · FDC-KGE\nTails:  RMSE$_{95}$, RMSE$_{99}$",
    "#7f7f7f", fs=11.5)
box(69, 10.4, 30, 11.2,
    "HYDROLOGICAL CONTROLS\n\nPer-basin skill regressed on\nflow signatures ($\\rho_1$, BFI,\n"
    "flashiness) + catchment attributes", "#117733", fs=11.5)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"figure_benchmark_design.{ext}", dpi=300, bbox_inches="tight")
print(f"[OK] wrote {FIG/'figure_benchmark_design.png'}")
