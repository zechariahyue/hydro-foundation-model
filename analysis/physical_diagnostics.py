"""
Physical diagnostics for the zero-shot evaluation:
  - Flow Duration Curve KGE (FDC_KGE)  — already in result CSVs
  - PBIAS (volume bias, %)
  - High-flow tail RMSE at 95th percentile (tail_RMSE_95)
  - Richards-Baker flashiness ratio (RB_flashiness_sim / RB_flashiness_obs)

Outputs:
  experiments/results/physical_diagnostics.csv  — full per-basin table
  A LaTeX Table 3 snippet printed to stdout

Usage: python experiments/analysis/physical_diagnostics.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).parents[1] / "results"
DATASETS = ["CAMELS-US", "CAMELS-BR", "CAMELS-CL", "CAMELS-AUS", "LamaH-CE", "CAMELS-IND"]
MODELS = ["chronos", "timesfm", "patchtst", "persistence"]


def load_df(model: str, dataset: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"{model}_zero_shot_{dataset}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def median_stat(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    return float(np.median(vals)) if len(vals) > 0 else None


def flashiness_ratio(df: pd.DataFrame) -> float | None:
    if "RB_flashiness_obs" not in df.columns or "RB_flashiness_sim" not in df.columns:
        return None
    obs = df["RB_flashiness_obs"].replace([np.inf, -np.inf], np.nan)
    sim = df["RB_flashiness_sim"].replace([np.inf, -np.inf], np.nan)
    valid = (obs > 0) & obs.notna() & sim.notna()
    if valid.sum() == 0:
        return None
    ratios = sim[valid].values / obs[valid].values
    return float(np.median(ratios))


def main():
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            df = load_df(model, dataset)
            if df is None:
                continue
            rows.append(dict(
                dataset=dataset,
                model=model,
                n=len(df),
                median_NSE=median_stat(df, "NSE"),
                median_KGE=median_stat(df, "KGE"),
                median_PBIAS=median_stat(df, "PBIAS"),
                median_FDC_KGE=median_stat(df, "FDC_KGE"),
                median_tail_RMSE_95=median_stat(df, "tail_RMSE_95"),
                flashiness_ratio=flashiness_ratio(df),
            ))

    diag_df = pd.DataFrame(rows)
    diag_df.to_csv(RESULTS_DIR / "physical_diagnostics.csv", index=False)

    # ── Print summary ───────────────────────────────────────────────────────
    print("\n=== PHYSICAL DIAGNOSTICS SUMMARY (medians) ===\n")
    print(f"{'Dataset':<14} {'Model':<12} {'FDC_KGE':>8} {'PBIAS%':>8} {'tail95':>9} {'Flash.R':>8}")
    print("-" * 65)
    for _, r in diag_df.iterrows():
        fdc = f"{r['median_FDC_KGE']:.3f}" if r["median_FDC_KGE"] is not None else "  n/a"
        pb  = f"{r['median_PBIAS']:.1f}"   if r["median_PBIAS"]   is not None else "  n/a"
        t95 = f"{r['median_tail_RMSE_95']:.1f}" if r["median_tail_RMSE_95"] is not None else "  n/a"
        fr  = f"{r['flashiness_ratio']:.3f}" if r["flashiness_ratio"] is not None else "  n/a"
        print(f"{r['dataset']:<14} {r['model']:<12} {fdc:>8} {pb:>8} {t95:>9} {fr:>8}")

    # ── LaTeX Table 3 snippet ────────────────────────────────────────────────
    print("\n=== LATEX TABLE 3 ROWS (FDC_KGE | PBIAS | flashiness ratio) ===\n")
    DATASET_LABELS = {
        "CAMELS-US": "CAMELS-US", "CAMELS-BR": "CAMELS-BR", "CAMELS-CL": "CAMELS-CL",
        "CAMELS-AUS": "CAMELS-AUS", "LamaH-CE": "LamaH-CE", "CAMELS-IND": "CAMELS-IND",
    }
    col_models = ["chronos", "timesfm", "patchtst", "persistence"]
    print("% FDC KGE")
    for dataset in DATASETS:
        sub = diag_df[diag_df["dataset"] == dataset]
        cells = [DATASET_LABELS[dataset]]
        for model in col_models:
            m = sub[sub["model"] == model]
            if m.empty or m["median_FDC_KGE"].isna().all():
                cells.append("---")
            else:
                cells.append(f"{m['median_FDC_KGE'].values[0]:.3f}")
        print(" & ".join(cells) + " \\\\")

    print("\n% PBIAS (%)")
    for dataset in DATASETS:
        sub = diag_df[diag_df["dataset"] == dataset]
        cells = [DATASET_LABELS[dataset]]
        for model in col_models:
            m = sub[sub["model"] == model]
            if m.empty or m["median_PBIAS"].isna().all():
                cells.append("---")
            else:
                cells.append(f"{m['median_PBIAS'].values[0]:.1f}")
        print(" & ".join(cells) + " \\\\")

    print("\nDone. Saved to experiments/results/physical_diagnostics.csv")


if __name__ == "__main__":
    main()
