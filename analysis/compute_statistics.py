"""
Compute bootstrap confidence intervals and Wilcoxon signed-rank tests
for the zero-shot evaluation results. Outputs LaTeX-ready table rows.

Usage: python experiments/analysis/compute_statistics.py
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from pathlib import Path

RESULTS_DIR = Path(__file__).parents[1] / "results"
DATASETS = ["CAMELS-US", "CAMELS-BR", "CAMELS-CL", "CAMELS-AUS", "LamaH-CE", "CAMELS-IND"]
MODELS = ["chronos", "timesfm", "patchtst", "persistence"]
N_BOOTSTRAP = 10_000
RNG = np.random.default_rng(42)


# ── helpers ────────────────────────────────────────────────────────────────

def load_df(model: str, dataset: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"{model}_zero_shot_{dataset}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)[["basin_id", "NSE"]].rename(columns={"NSE": model})
    return df.set_index("basin_id")


def bootstrap_median_ci(values: np.ndarray, n: int = N_BOOTSTRAP, alpha: float = 0.05):
    """Percentile bootstrap CI for the median."""
    idx = RNG.integers(0, len(values), size=(n, len(values)))
    medians = np.median(values[idx], axis=1)
    return float(np.percentile(medians, 100 * alpha / 2)), float(np.percentile(medians, 100 * (1 - alpha / 2)))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """BH FDR correction. Returns adjusted p-values."""
    n = len(p_values)
    order = np.argsort(p_values)
    p_sorted = np.array(p_values)[order]
    adjusted = np.minimum(1.0, p_sorted * n / (np.arange(1, n + 1)))
    # enforce monotonicity (cumulative min from the right)
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    result = np.empty(n)
    result[order] = adjusted
    return result.tolist()


# ── main ────────────────────────────────────────────────────────────────────

def main():
    # Load all data, aligning on basin_id
    frames: dict[str, dict[str, np.ndarray]] = {}  # dataset -> model -> array
    merged: dict[str, pd.DataFrame] = {}

    for dataset in DATASETS:
        dfs = {}
        for model in MODELS:
            df = load_df(model, dataset)
            if df is not None:
                dfs[model] = df
        if not dfs:
            continue
        base = list(dfs.values())[0]
        for df in list(dfs.values())[1:]:
            base = base.join(df, how="inner")
        # Drop rows with any NaN to ensure paired tests work
        base_clean = base.dropna()
        merged[dataset] = base_clean
        for model in dfs:
            if model in base_clean.columns:
                frames.setdefault(dataset, {})[model] = base_clean[model].values

    # ── Bootstrap CIs ───────────────────────────────────────────────────────
    print("\n=== BOOTSTRAP 95% CI FOR MEDIAN NSE ===\n")
    print(f"{'Dataset':<14} {'Model':<12} {'Median':>7} {'CI_lo':>7} {'CI_hi':>7} {'N':>5}")
    print("-" * 56)

    stat_rows = []
    for dataset in DATASETS:
        if dataset not in frames:
            continue
        for model in MODELS:
            if model not in frames[dataset]:
                continue
            vals = frames[dataset][model]
            median = float(np.median(vals))
            lo, hi = bootstrap_median_ci(vals)
            n = len(vals)
            print(f"{dataset:<14} {model:<12} {median:7.3f} {lo:7.3f} {hi:7.3f} {n:5d}")
            stat_rows.append(dict(dataset=dataset, model=model, median=median,
                                  ci_lo=lo, ci_hi=hi, n=n))

    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(RESULTS_DIR / "bootstrap_ci_results.csv", index=False)

    # ── Wilcoxon signed-rank tests ───────────────────────────────────────────
    print("\n=== WILCOXON SIGNED-RANK: model vs persistence ===\n")
    print(f"{'Dataset':<14} {'Model':<12} {'W':>10} {'p_raw':>10} {'p_BH':>10} {'sig':>5}")
    print("-" * 60)

    wilcoxon_rows = []
    for dataset in DATASETS:
        if dataset not in frames or "persistence" not in frames[dataset]:
            continue
        p_vals = frames[dataset]["persistence"]
        for model in ["chronos", "timesfm", "patchtst"]:
            if model not in frames[dataset]:
                continue
            m_vals = frames[dataset][model]
            # Wilcoxon on differences (model NSE - persistence NSE)
            diff = m_vals - p_vals
            if np.all(diff == 0):
                continue
            stat, p_raw = wilcoxon(diff, alternative="two-sided")
            wilcoxon_rows.append(dict(dataset=dataset, model=model,
                                      W=stat, p_raw=float(p_raw)))

    # Apply BH correction across all tests
    p_raw_all = [r["p_raw"] for r in wilcoxon_rows]
    p_bh_all = benjamini_hochberg(p_raw_all)
    for row, p_bh in zip(wilcoxon_rows, p_bh_all):
        row["p_BH"] = p_bh
        sig = "***" if p_bh < 0.001 else ("**" if p_bh < 0.01 else ("*" if p_bh < 0.05 else "ns"))
        row["sig"] = sig
        print(f"{row['dataset']:<14} {row['model']:<12} {row['W']:10.1f} "
              f"{row['p_raw']:10.4f} {row['p_BH']:10.4f} {sig:>5}")

    wilcoxon_df = pd.DataFrame(wilcoxon_rows)
    wilcoxon_df.to_csv(RESULTS_DIR / "wilcoxon_results.csv", index=False)

    # ── LaTeX table snippet ─────────────────────────────────────────────────
    print("\n=== LATEX TABLE 2 ROWS (median [CI_lo, CI_hi]) ===\n")
    DATASET_LABELS = {
        "CAMELS-US": "CAMELS-US$^\\ddagger$",
        "CAMELS-BR": "CAMELS-BR",
        "CAMELS-CL": "CAMELS-CL",
        "CAMELS-AUS": "CAMELS-AUS",
        "LamaH-CE": "LamaH-CE",
        "CAMELS-IND": "CAMELS-IND",
    }
    REGIONS = {
        "CAMELS-US": "N. America", "CAMELS-BR": "S. America", "CAMELS-CL": "S. America",
        "CAMELS-AUS": "Australia", "LamaH-CE": "Europe", "CAMELS-IND": "Asia",
    }
    BASINS = {
        "CAMELS-US": 673, "CAMELS-BR": 2670, "CAMELS-CL": 430,
        "CAMELS-AUS": 560, "LamaH-CE": 880, "CAMELS-IND": 241,
    }
    MODEL_ORDER = ["chronos", "timesfm", "patchtst", "persistence"]

    # Build p_bh lookup
    bh_lookup = {(r["dataset"], r["model"]): r for r in wilcoxon_rows}

    for dataset in DATASETS:
        sub = stats_df[stats_df["dataset"] == dataset]
        row_cells = [DATASET_LABELS[dataset], REGIONS[dataset], str(BASINS[dataset])]
        best_median = sub[sub["model"].isin(["chronos", "timesfm", "patchtst", "persistence"])]["median"].max()
        for model in MODEL_ORDER:
            m_row = sub[sub["model"] == model]
            if m_row.empty:
                row_cells.append("---")
                continue
            med = m_row["median"].values[0]
            lo = m_row["ci_lo"].values[0]
            hi = m_row["ci_hi"].values[0]
            cell = f"{med:.3f} [{lo:.3f}, {hi:.3f}]"
            if abs(med - best_median) < 1e-6:
                cell = f"\\textbf{{{med:.3f}}} [{lo:.3f}, {hi:.3f}]"
            # add significance marker vs persistence
            wrow = bh_lookup.get((dataset, model))
            if wrow and model != "persistence":
                cell += f"$^{{{wrow['sig']}}}$"
            row_cells.append(cell)
        # LSTM column
        lstm_path = RESULTS_DIR / f"lstm_zero_shot_{dataset}.csv"
        if lstm_path.exists():
            lstm_df = pd.read_csv(lstm_path)
            lstm_med = float(np.median(lstm_df["NSE"].values))
            row_cells.append(f"{lstm_med:.1f}")
        else:
            row_cells.append("---")
        print(" & ".join(row_cells) + " \\\\")

    print("\nDone. Results saved to experiments/results/bootstrap_ci_results.csv")
    print("                         experiments/results/wilcoxon_results.csv")


if __name__ == "__main__":
    main()
