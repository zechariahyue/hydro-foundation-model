#!/usr/bin/env python3
"""
Fitted autoregressive AR(p) baseline on the SAME target points as the l5s7 benchmark.

For every forecast origin (start of a 512-day context window, stride 7, last 5 years
of record) an AR(p) with intercept is fitted by OLS on that context window alone
(exactly what the foundation models see) and used to forecast the next day.
Lag-1 persistence is AR(1) with phi fixed at 1 and no intercept; this asks whether
a *fitted* linear model on the same information closes TimesFM's low-memory advantage.

Added 2026-09-06 in response to the 2026-08-17 peer review ("no fitted autoregressive
baseline"). Output: experiments/results/ar{p}_zero_shot_<DATASET>_l5s7.csv, same
column layout as persistence_zero_shot_*_l5s7.csv.

Usage:
    python run_ar_baseline.py --datasets CAMELS-US ... --orders 1 2 5
"""
import os
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_foundation_models as rfm  # reuse loader, window protocol, metrics, save

RIDGE = 1e-6  # ponytail: tiny ridge so lstsq never fails on constant contexts


def ar_forecast(ctx: np.ndarray, p: int) -> float:
    """Fit AR(p)+intercept by OLS on ctx, return one-step-ahead forecast."""
    n = len(ctx)
    y = ctx[p:]
    X = np.column_stack([np.ones(n - p)] + [ctx[p - k: n - k] for k in range(1, p + 1)])
    # closed-form ridge-OLS: (X'X + lam I)^-1 X'y
    XtX = X.T @ X + RIDGE * np.eye(p + 1)
    beta = np.linalg.solve(XtX, X.T @ y)
    x_next = np.concatenate([[1.0], ctx[::-1][:p]])
    return float(beta @ x_next)


def run_ar(basins: dict, p: int, context_length: int = 512, horizon: int = 1):
    results = {}
    for bid, q_series in basins.items():
        q = q_series.values.astype(np.float64)
        if len(q) < context_length + horizon:
            continue
        obs, sim = [], []
        for start in range(0, len(q) - context_length - horizon + 1, rfm.STRIDE):
            t = start + context_length
            obs.append(q[t])
            sim.append(ar_forecast(q[start:t], p))
        if not obs:
            continue
        obs = np.asarray(obs); sim = np.asarray(sim)
        results[bid] = rfm.compute_all_metrics(obs, sim)
        results[bid]["n_predictions"] = len(obs)
    return results


def _selfcheck():
    rng = np.random.default_rng(0)
    # exact AR(2) process -> AR(2) fit must recover the next value almost exactly
    x = np.zeros(600); x[:2] = rng.normal(size=2)
    for t in range(2, 600):
        x[t] = 0.5 + 0.6 * x[t-1] + 0.3 * x[t-2]
    pred = ar_forecast(x[:-1], 2)
    assert abs(pred - x[-1]) < 1e-6, (pred, x[-1])
    # constant context must not crash and must return the constant
    assert abs(ar_forecast(np.full(512, 3.0), 2) - 3.0) < 1e-3
    print("selfcheck OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["CAMELS-US", "CAMELS-BR", "CAMELS-CL", "CAMELS-AUS", "LamaH-CE", "CAMELS-IND"])
    ap.add_argument("--orders", nargs="+", type=int, default=[1, 2, 5])
    ap.add_argument("--last_years", type=int, default=5)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--out_suffix", default="_l5s7")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck(); sys.exit(0)
    rfm.STRIDE = a.stride
    for ds in a.datasets:
        basins = rfm.load_basin_series(ds, last_years=a.last_years)
        print(f"{ds}: {len(basins)} basins", flush=True)
        for p in a.orders:
            res = run_ar(basins, p)
            rfm.save_results(res, f"ar{p}", "zero_shot", ds, a.out_suffix)
