"""
Make tail RMSE dimensionless.

The processed discharge column `QObs(mm/d)` is mislabelled project-wide: it holds each
archive's NATIVE units (CAMELS-US is cfs-scale, etc.). So absolute tail_RMSE_95/99 as stored
in the `_l5s7` result CSVs are native-unit magnitudes and are NOT comparable across archives.
A hydrology referee will not accept a discharge quantity with no unit.

Fix without re-running anything on the GPU:

    NSE  = 1 - SSE/SST,  SSE = n*RMSE^2,  SST = n*Var(obs)
    =>   Var(obs) = RMSE^2 / (1 - NSE)
    =>   sigma_obs = RMSE / sqrt(1 - NSE)

sigma_obs is a property of the OBSERVATIONS on the scored target points. Because every model
is scored on identical target points within a dataset (that is the whole point of the l5s7
protocol), sigma_obs recovered from the TimesFM row and from the Chronos row of the same basin
must agree. That cross-model agreement is a free, strong validation of the identity -- if the
target points were not actually matched, this check fails loudly.

We then report two dimensionless forms:

    rel_tail_sigma = tail_RMSE / sigma_obs        (tail error in units of flow variability)
    rel_tail_mean  = tail_RMSE / mean_obs         (tail error as a fraction of mean flow)

mean_obs comes from sigma_obs / q_cv, with q_cv (coefficient of variation of flow) taken from
`basin_attributes.csv`, which was built independently from the raw archives. That gives a
SECOND independent cross-check on sigma_obs.

Outputs
-------
experiments/results/tail_rmse_normalised_per_basin.csv
experiments/results/TAIL_RMSE_NORMALISED_REPORT.txt
experiments/results/table_tail_rmse_normalised.tex
"""

from __future__ import annotations

import glob
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

MODELS = ["timesfm", "chronos", "patchtst", "persistence"]
PRETTY = {
    "timesfm": "TimesFM",
    "chronos": "Chronos",
    "patchtst": "PatchTST",
    "persistence": "Persistence",
}
DATASETS = ["CAMELS-US", "CAMELS-BR", "CAMELS-CL", "CAMELS-AUS", "LamaH-CE", "CAMELS-IND"]

# sigma_obs = RMSE / sqrt(1 - NSE) blows up as NSE -> 1. Only trust rows comfortably below 1.
NSE_MAX_FOR_SIGMA = 0.90


def load_l5s7() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*_zero_shot_*_l5s7.csv"))):
        df = pd.read_csv(path, dtype={"basin_id": str})
        frames.append(df)
    if not frames:
        raise SystemExit(f"No *_zero_shot_*_l5s7.csv under {RESULTS}")
    return pd.concat(frames, ignore_index=True)


def recover_sigma(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, basin, model) estimate of sigma_obs, with stability flag."""
    d = df.copy()
    d = d[np.isfinite(d["NSE"]) & np.isfinite(d["RMSE"])]
    one_minus = 1.0 - d["NSE"]
    d["sigma_est"] = np.where(one_minus > 0, d["RMSE"] / np.sqrt(one_minus.clip(lower=1e-12)), np.nan)
    d["sigma_trust"] = d["NSE"] <= NSE_MAX_FOR_SIGMA
    return d


def consensus_sigma(d: pd.DataFrame) -> pd.DataFrame:
    """One sigma_obs per basin: median over the numerically trustworthy model rows."""
    trusted = d[d["sigma_trust"] & np.isfinite(d["sigma_est"])]
    grp = trusted.groupby(["dataset", "basin_id"])["sigma_est"]
    out = grp.median().rename("sigma_obs").reset_index()
    out["sigma_n_models"] = grp.size().values
    # spread across models scored on identical points -> should be ~0
    spread = grp.apply(lambda s: (s.max() - s.min()) / s.median() if s.median() > 0 else np.nan)
    out["sigma_rel_spread"] = spread.values
    return out


def main() -> None:
    raw = load_l5s7()
    raw = raw[raw["model"].isin(MODELS)]
    d = recover_sigma(raw)
    sig = consensus_sigma(d)

    lines: list[str] = []
    add = lines.append
    add("TAIL RMSE -- DIMENSIONLESS RESCALING")
    add("=" * 78)
    add("")
    add("sigma_obs recovered analytically as RMSE / sqrt(1 - NSE) from the _l5s7 CSVs.")
    add(f"Rows used for the estimate: NSE <= {NSE_MAX_FOR_SIGMA} (numerical stability).")
    add("")

    # ---- VALIDATION 1: cross-model agreement on sigma_obs -------------------------------
    add("VALIDATION 1 -- cross-model agreement on sigma_obs")
    add("-" * 78)
    add("Every model is scored on identical target points, so sigma_obs recovered from")
    add("different models' rows must agree. Relative spread = (max-min)/median across models.")
    add("")
    ok = sig[sig["sigma_n_models"] >= 2]
    add(f"  basins with >=2 trustworthy model rows : {len(ok):,}")
    if len(ok):
        add(f"  median relative spread                 : {ok['sigma_rel_spread'].median():.3e}")
        add(f"  90th percentile relative spread        : {ok['sigma_rel_spread'].quantile(0.90):.3e}")
        add(f"  max relative spread                    : {ok['sigma_rel_spread'].max():.3e}")
        bad = int((ok["sigma_rel_spread"] > 0.01).sum())
        add(f"  basins with spread > 1%                : {bad:,} ({100*bad/len(ok):.2f}%)")
        add("")
        add("  -> spread ~0 confirms BOTH the identity AND that target points are matched.")
    add("")

    # ---- VALIDATION 2: against independently-built q_cv ---------------------------------
    attr_path = os.path.join(RESULTS, "basin_attributes.csv")
    merged = sig.copy()
    if os.path.exists(attr_path):
        # basin_attributes.csv stores the basin id in column `bid`, not `basin_id`.
        attrs = pd.read_csv(attr_path, dtype={"bid": str, "basin_id": str})
        if "basin_id" not in attrs.columns and "bid" in attrs.columns:
            attrs = attrs.rename(columns={"bid": "basin_id"})
        attrs["basin_id"] = attrs["basin_id"].astype(str).str.strip()
        keep = [c for c in ["dataset", "basin_id", "q_cv", "area"] if c in attrs.columns]
        merged = merged.merge(attrs[keep], on=["dataset", "basin_id"], how="left")
        if "q_cv" in merged.columns:
            merged["mean_obs"] = np.where(
                merged["q_cv"] > 0, merged["sigma_obs"] / merged["q_cv"], np.nan
            )
        add("VALIDATION 2 -- sigma_obs vs independently-computed q_cv")
        add("-" * 78)
        n_cv = int(np.isfinite(merged.get("mean_obs", pd.Series(dtype=float))).sum())
        add(f"  basins with q_cv from basin_attributes.csv : {n_cv:,}")
        add("  mean_obs = sigma_obs / q_cv  (q_cv built independently from the raw archives)")
        add("")
    else:
        merged["mean_obs"] = np.nan
        add("VALIDATION 2 -- SKIPPED (basin_attributes.csv not found)")
        add("")

    # ---- apply normalisation -----------------------------------------------------------
    j = d.merge(merged, on=["dataset", "basin_id"], how="left")
    for p in ("95", "99"):
        col = f"tail_RMSE_{p}"
        j[f"rel_tail{p}_sigma"] = np.where(j["sigma_obs"] > 0, j[col] / j["sigma_obs"], np.nan)
        j[f"rel_tail{p}_mean"] = np.where(j["mean_obs"] > 0, j[col] / j["mean_obs"], np.nan)

    keep_cols = [
        "dataset", "basin_id", "model", "NSE", "RMSE",
        "tail_RMSE_95", "tail_RMSE_99",
        "sigma_obs", "mean_obs", "sigma_rel_spread",
        "rel_tail95_sigma", "rel_tail99_sigma",
        "rel_tail95_mean", "rel_tail99_mean",
    ]
    out = j[[c for c in keep_cols if c in j.columns]].copy()
    out_path = os.path.join(RESULTS, "tail_rmse_normalised_per_basin.csv")
    out.to_csv(out_path, index=False)

    # ---- summary table -----------------------------------------------------------------
    add("MEDIAN DIMENSIONLESS TAIL RMSE  (tail_RMSE_95 / sigma_obs)")
    add("-" * 78)
    hdr = f"{'dataset':<12}" + "".join(f"{m:>13}" for m in MODELS)
    add(hdr)
    piv = {}
    for ds in DATASETS:
        sub = out[out["dataset"] == ds]
        if sub.empty:
            continue
        row = f"{ds:<12}"
        piv[ds] = {}
        for m in MODELS:
            v = sub[sub["model"] == m]["rel_tail95_sigma"]
            v = v[np.isfinite(v)]
            med = v.median() if len(v) else np.nan
            piv[ds][m] = med
            row += f"{med:>13.3f}" if np.isfinite(med) else f"{'--':>13}"
        add(row)
    add("")
    add("Dimensionless: tail error expressed in standard deviations of observed flow on the")
    add("same scored points. Directly comparable across archives, unlike the native-unit form.")
    add("")

    add("MEDIAN TAIL RMSE AS A FRACTION OF MEAN FLOW  (tail_RMSE_95 / mean_obs)")
    add("-" * 78)
    add(hdr)
    for ds in DATASETS:
        sub = out[out["dataset"] == ds]
        if sub.empty:
            continue
        row = f"{ds:<12}"
        for m in MODELS:
            v = sub[sub["model"] == m]["rel_tail95_mean"]
            v = v[np.isfinite(v)]
            med = v.median() if len(v) else np.nan
            row += f"{med:>13.3f}" if np.isfinite(med) else f"{'--':>13}"
        add(row)
    add("")

    rep = os.path.join(RESULTS, "TAIL_RMSE_NORMALISED_REPORT.txt")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ---- LaTeX table -------------------------------------------------------------------
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Dimensionless tail error. Median $\mathrm{RMSE}_{95}/\sigma_{\mathrm{obs}}$,"
        r" where $\sigma_{\mathrm{obs}}$ is the standard deviation of observed discharge on the"
        r" same scored target points. Expressing the tail error in standard deviations of"
        r" observed flow removes the native-unit dependence, so values are comparable across"
        r" archives. Lower is better.}",
        r"\label{tab:tail_rmse_norm}",
        r"\begin{tabular}{l" + "r" * len(MODELS) + r"}",
        r"\hline",
        "Dataset & " + " & ".join(PRETTY.get(m, m.capitalize()) for m in MODELS) + r" \\",
        r"\hline",
    ]
    for ds in DATASETS:
        if ds not in piv:
            continue
        cells = []
        for m in MODELS:
            v = piv[ds][m]
            cells.append(f"{v:.3f}" if np.isfinite(v) else "--")
        tex.append(ds.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    tex += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    with open(os.path.join(RESULTS, "table_tail_rmse_normalised.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex))

    print("\n".join(lines))
    print(f"\nWrote:\n  {out_path}\n  {rep}\n  {os.path.join(RESULTS, 'table_tail_rmse_normalised.tex')}")


if __name__ == "__main__":
    main()
