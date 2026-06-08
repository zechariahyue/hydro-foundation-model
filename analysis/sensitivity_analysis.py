#!/usr/bin/env python3
"""
Data quality sensitivity analysis: missing data handling.

Artificially introduces gaps of 1-7 days into each basin's streamflow record,
applies three interpolation strategies, and measures NSE degradation vs the
complete record using persistence as the prediction baseline.

Usage:
    python experiments/analysis/sensitivity_analysis.py

Output:
    experiments/results/sensitivity_analysis.csv
    Prints a summary table to stdout.
"""

import sys
import functools
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from statsmodels.tsa.seasonal import STL

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import nse

DATA_DIR = Path(os.environ.get("CAMELS_DATA_DIR", "data/processed"))
RESULTS_DIR = Path(__file__).parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["CAMELS-US", "CAMELS-BR", "CAMELS-CL", "CAMELS-AUS", "LamaH-CE", "CAMELS-IND"]
GAP_LENGTHS = [1, 2, 3, 5, 7]
N_GAPS_PER_BASIN = 20   # number of random gaps to introduce per basin
RNG = np.random.default_rng(42)
MAX_BASINS_PER_DATASET = 50  # cap for speed; remove to run on all


# ── interpolation strategies ──────────────────────────────────────────────────

def interp_linear(series: pd.Series, gap_idx: np.ndarray) -> pd.Series:
    """Linear interpolation over masked gap."""
    s = series.copy().astype(float)
    s.iloc[gap_idx] = np.nan
    return s.interpolate(method="linear", limit_direction="both")


def interp_stl(series: pd.Series, gap_idx: np.ndarray,
               trend: np.ndarray = None, seasonal: np.ndarray = None) -> pd.Series:
    """
    STL-based interpolation.
    If precomputed trend/seasonal arrays are provided (from a basin-level
    STL fit), fills the gap with trend[i] + seasonal[i].
    Falls back to linear interpolation if STL is unavailable.
    """
    s = series.copy().astype(float)
    s.iloc[gap_idx] = np.nan

    if trend is not None and seasonal is not None:
        for i in gap_idx:
            s.iloc[i] = max(float(trend[i] + seasonal[i]), 0.0)
    else:
        # On-the-fly fallback (slow; only reached if precomputation failed)
        s_linear = s.interpolate(method="linear", limit_direction="both")
        try:
            res = STL(s_linear, period=365, robust=True).fit()
            for i in gap_idx:
                s.iloc[i] = max(float(res.trend[i] + res.seasonal[i]), 0.0)
        except Exception:
            return s_linear

    return s.interpolate(method="linear", limit_direction="both")


# ── core analysis ─────────────────────────────────────────────────────────────

def load_basins(dataset: str, max_basins: int) -> dict:
    ds_dir = DATA_DIR / dataset
    basins = {}
    if not ds_dir.exists():
        return basins
    paths = sorted(ds_dir.glob("*.parquet"))[:max_basins]
    for p in paths:
        df = pd.read_parquet(p)
        col = next((c for c in ["QObs(mm/d)", "streamflow", "q_obs"] if c in df.columns), None)
        if col is None:
            continue
        # coerce handles space-strings and other non-numeric placeholders
        q = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(q) > 365:
            basins[p.stem] = q
    return basins


def baseline_nse(q: pd.Series) -> float:
    """Persistence baseline NSE on the full series (lag-1)."""
    obs = q.values[1:]
    pred = q.values[:-1]
    return nse(obs, pred)


def evaluate_gap(q: pd.Series, gap_start: int, gap_len: int,
                 strategies: list) -> dict:
    """
    For a single gap of gap_len days starting at gap_start:
    - record original values
    - apply each interpolation strategy
    - compute NSE of interpolated vs original for the gap region only
    Returns dict of {strategy_name: nse_in_gap}.
    """
    gap_idx = np.arange(gap_start, min(gap_start + gap_len, len(q)))
    original = q.values[gap_idx]

    results = {}
    for name, fn in strategies:
        filled = fn(q, gap_idx)
        predicted = filled.values[gap_idx]
        results[name] = nse(original, predicted)

    return results


def run_dataset(dataset: str) -> list[dict]:
    basins = load_basins(dataset, MAX_BASINS_PER_DATASET)
    if not basins:
        return []

    rows = []
    for bid, q in tqdm(basins.items(), desc=dataset, leave=False):
        n = len(q)

        # Precompute STL decomposition once per basin (period = 365 days)
        stl_trend = stl_seasonal = None
        try:
            res = STL(q.values.astype(float), period=365, robust=True).fit()
            stl_trend, stl_seasonal = res.trend, res.seasonal
        except Exception:
            pass  # interp_stl will fall back to on-the-fly fit

        stl_fn = functools.partial(interp_stl, trend=stl_trend, seasonal=stl_seasonal)
        strategies = [("linear", interp_linear), ("stl", stl_fn)]

        for gap_len in GAP_LENGTHS:
            max_start = n - gap_len - 1
            if max_start <= gap_len:
                continue
            starts = RNG.integers(gap_len, max_start, size=N_GAPS_PER_BASIN)
            for start in starts:
                gap_result = evaluate_gap(q, int(start), gap_len, strategies)
                for strategy, gap_nse in gap_result.items():
                    rows.append(dict(
                        dataset=dataset,
                        basin_id=bid,
                        gap_length=gap_len,
                        strategy=strategy,
                        nse_in_gap=gap_nse,
                    ))
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    all_rows = []
    for dataset in DATASETS:
        rows = run_dataset(dataset)
        all_rows.extend(rows)
        print(f"  {dataset}: {len(rows)} gap evaluations")

    if not all_rows:
        print("No data found. Check DATA_DIR path.")
        return

    df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / "sensitivity_analysis.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}\n")

    # ── summary table ─────────────────────────────────────────────────────────
    print("=== NSE IN GAP BY STRATEGY AND GAP LENGTH ===\n")
    summary = (
        df.groupby(["gap_length", "strategy"])["nse_in_gap"]
        .agg(["median", "mean", "count"])
        .round(3)
    )
    print(summary.to_string())

    # ── short vs long gap comparison ─────────────────────────────────────────
    print("\n=== SHORT GAPS (1-3 days) vs LONG GAPS (5-7 days) ===\n")
    df["gap_category"] = df["gap_length"].apply(
        lambda x: "short (1-3d)" if x <= 3 else "long (5-7d)"
    )
    cat_summary = (
        df.groupby(["gap_category", "strategy"])["nse_in_gap"]
        .median()
        .round(3)
        .unstack("strategy")
    )
    print(cat_summary.to_string())

    # ── best strategy per gap length ──────────────────────────────────────────
    print("\n=== BEST STRATEGY PER GAP LENGTH (by median NSE in gap) ===\n")
    best = (
        df.groupby(["gap_length", "strategy"])["nse_in_gap"]
        .median()
        .reset_index()
    )
    for gap_len in GAP_LENGTHS:
        sub = best[best["gap_length"] == gap_len].sort_values("nse_in_gap", ascending=False)
        best_row = sub.iloc[0]
        print(f"  Gap {gap_len}d: best = {best_row['strategy']} "
              f"(median NSE in gap = {best_row['nse_in_gap']:.3f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
