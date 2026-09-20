#!/usr/bin/env python3
"""
Regenerate the two SI overview figures from the live _l5s7 result CSVs:
  figure_1_performance_comparison_ACTUAL.png  -- per-dataset NSE box plots, four models
  figure_2_dataset_summary_ACTUAL.png         -- median NSE by dataset (left) and by model (right)
Both were previously ad-hoc (May 2026) and still showed the single-draw Chronos run; scripted
2026-09-06 after the ERL round-2 review caught them. Common finite-NSE basin set per dataset,
as in Table 1.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parents[1] / "results"
FIG = Path(os.environ.get("FIGURES_DIR", "figures"))
DATASETS = ["CAMELS-US", "CAMELS-BR", "CAMELS-CL", "CAMELS-AUS", "LamaH-CE", "CAMELS-IND"]
SHORT = {"CAMELS-US": "US", "CAMELS-BR": "BR", "CAMELS-CL": "CL", "CAMELS-AUS": "AUS", "LamaH-CE": "LamaH", "CAMELS-IND": "IND"}
MODELS = [("chronos", "Chronos", "#4c8fb5"), ("timesfm", "TimesFM", "#b5679c"), ("patchtst", "PatchTST", "#f2a93b"), ("persistence", "Persistence", "#d9694d")]
plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "axes.titlesize": 15, "figure.dpi": 300})

data = {}
for ds in DATASETS:
    fr = {m: pd.read_csv(RES / f"{m}_zero_shot_{ds}_l5s7.csv", dtype={"basin_id": str}).set_index("basin_id").NSE for m, _, _ in MODELS}
    idx = None
    for m, s in fr.items():
        ok = s.index[np.isfinite(s)]
        idx = ok if idx is None else idx.intersection(ok)
    data[ds] = {m: fr[m].loc[idx].values for m in fr}

# ---- figure 1: box plots
fig, ax = plt.subplots(figsize=(20, 8.5))
w = 0.19
for j, (m, lab, col) in enumerate(MODELS):
    pos = [i + (j - 1.5) * (w + 0.02) for i in range(len(DATASETS))]
    bp = ax.boxplot([np.clip(data[ds][m], -1, 1) for ds in DATASETS], positions=pos, widths=w, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", lw=1.5))
    for b in bp["boxes"]:
        b.set_facecolor(col); b.set_alpha(0.85)
    ax.plot([], [], color=col, lw=10, label=lab)
ax.set_xticks(range(len(DATASETS)))
ax.set_xticklabels([f"{SHORT[ds]}\n({len(data[ds]['timesfm'])})" for ds in DATASETS])
ax.set_ylim(-0.5, 1.0); ax.axhline(0, color="grey", ls="--", lw=1)
ax.set_ylabel("Nash-Sutcliffe Efficiency (NSE)", fontweight="bold"); ax.set_xlabel("Dataset (Number of Basins)", fontweight="bold")
ax.set_title("Zero-Shot Performance Comparison Across Datasets", fontweight="bold")
ax.grid(axis="y", alpha=0.3, ls="--"); ax.legend(loc="lower right")
pooled = {lab: np.median(np.concatenate([data[ds][m] for ds in DATASETS])) for m, lab, _ in MODELS}
ax.text(0.01, 0.98, "Median NSE by Model (pooled):\n" + "\n".join(f"{k}: {v:.3f}" for k, v in pooled.items()),
        transform=ax.transAxes, va="top", fontsize=12, bbox=dict(boxstyle="round", fc="#fff5dc", alpha=0.8))
fig.tight_layout(); fig.savefig(FIG / "figure_1_performance_comparison_ACTUAL.png"); plt.close(fig)

# ---- figure 2: median bars
med = pd.DataFrame({lab: [np.median(data[ds][m]) for ds in DATASETS] for m, lab, _ in MODELS}, index=DATASETS)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 7.4))
x = np.arange(len(DATASETS)); w = 0.2
for j, (m, lab, col) in enumerate(MODELS):
    a1.bar(x + (j - 1.5) * w, med[lab].values, w, color=col, edgecolor="black", lw=0.6, label=lab)
a1.set_xticks(x); a1.set_xticklabels(DATASETS, rotation=45, ha="right"); a1.set_ylim(-0.1, 1.0)
a1.axhline(0, color="grey", ls="--", lw=1); a1.set_ylabel("Median NSE", fontweight="bold"); a1.set_xlabel("Dataset", fontweight="bold")
a1.set_title("Zero-Shot Performance by Dataset", fontweight="bold"); a1.grid(axis="y", alpha=0.3, ls="--"); a1.legend(loc="lower right")
dcols = ["#3b74b6", "#4faf4f", "#a97fd0", "#f39ac0", "#c9c840", "#39c8d0"]
x2 = np.arange(len(MODELS)); w2 = 0.13
for i, ds in enumerate(DATASETS):
    a2.bar(x2 + (i - 2.5) * w2, [med.loc[ds, lab] for _, lab, _ in MODELS], w2, color=dcols[i], edgecolor="black", lw=0.6, label=ds)
a2.set_xticks(x2); a2.set_xticklabels([lab for _, lab, _ in MODELS]); a2.set_ylim(-0.1, 1.0)
a2.axhline(0, color="grey", ls="--", lw=1); a2.set_ylabel("Median NSE", fontweight="bold"); a2.set_xlabel("Model", fontweight="bold")
a2.set_title("Zero-Shot Performance by Model", fontweight="bold"); a2.grid(axis="y", alpha=0.3, ls="--"); a2.legend(loc="lower right", ncol=2, fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "figure_2_dataset_summary_ACTUAL.png"); plt.close(fig)
print(med.round(3)); print("[OK] wrote figure_1_performance_comparison_ACTUAL.png, figure_2_dataset_summary_ACTUAL.png")
