#!/usr/bin/env python3
"""
Chronos reliability diagrams: nominal vs empirical coverage, stratified by flow regime.

Runs Chronos with num_samples=100 on 50 basins per dataset (test period 2015-2019).
Produces figure_reliability_diagrams.png for inclusion in the paper.

Usage:
    python evaluation/reliability_diagrams.py
    python evaluation/reliability_diagrams.py --n_basins 30 --num_samples 50
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR   = Path(os.environ.get("CAMELS_DATA_DIR", "data/processed"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
FIGURES_DIR = Path(os.environ.get("FIGURES_DIR", "figures"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_LENGTH = 512
HORIZON = 1
TEST_START = "2010-01-01"
TEST_END   = "2017-12-31"
RANDOM_SEED = 42

# CAMELS-BR excluded: date index corrupted in processed_camels (years stored as nanoseconds)
DATASETS = ["CAMELS-US", "CAMELS-BR", "CAMELS-CL", "CAMELS-AUS", "LamaH-CE", "CAMELS-IND"]

# Nominal coverage levels for the reliability diagram
NOMINAL_LEVELS = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])

REGIME_NAMES  = ["Low flow (<P25)", "Normal (P25–P75)", "Flood (>P95)"]
REGIME_COLORS = ["#2196F3", "#4CAF50", "#F44336"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_basin_subset(dataset_name: str, n_basins: int) -> dict:
    ds_dir = DATA_DIR / dataset_name
    if not ds_dir.exists():
        print(f"  Warning: {ds_dir} not found — skipping")
        return {}

    all_files = sorted(ds_dir.glob("*.parquet"))
    if not all_files:
        print(f"  Warning: no parquet files in {ds_dir} — skipping")
        return {}

    rng = random.Random(RANDOM_SEED)
    selected = rng.sample(all_files, min(n_basins, len(all_files)))

    basins = {}
    for pf in selected:
        try:
            df = pd.read_parquet(pf)
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    pass
            # Consistent protocol: most recent 5 years; position-based fallback for the
            # corrupted-date CAMELS-BR index (so CAMELS-BR can now be included).
            n_keep = int(5 * 365.25)
            try:
                ndates = df.index.normalize().nunique()
            except Exception:
                ndates = 0
            if ndates >= 0.5 * len(df):
                cutoff = df.index.max() - pd.DateOffset(years=5)
                df = df[df.index > cutoff]
            else:
                df = df.iloc[-n_keep:]
            if "QObs(mm/d)" not in df.columns:
                continue
            q = pd.to_numeric(df["QObs(mm/d)"], errors="coerce").dropna()
            if len(q) >= CONTEXT_LENGTH + HORIZON + 10:
                basins[pf.stem] = q
        except Exception as e:
            print(f"  Could not load {pf.name}: {e}")

    return basins


# ── Chronos inference ─────────────────────────────────────────────────────────

def run_chronos_probabilistic(basins: dict, num_samples: int) -> dict:
    """
    Run Chronos with num_samples draws per prediction step.

    Returns per-basin dict:
        {"obs": np.ndarray (N,), "samples": np.ndarray (N, num_samples)}
    """
    print(f"  Loading Chronos (device={DEVICE}) ...")
    from chronos import ChronosPipeline
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-large",
        device_map=DEVICE,
        torch_dtype=torch.float32,
    )

    BATCH = 4  # small batch to fit GPU memory with num_samples=100
    results = {}

    for bid, q_series in tqdm(basins.items(), desc="  Chronos inference"):
        q = q_series.values.astype(np.float32)

        contexts, targets = [], []
        step = 30  # monthly stride — enough windows for coverage estimates, avoids redundancy
        for start in range(0, len(q) - CONTEXT_LENGTH - HORIZON + 1, step):
            contexts.append(torch.tensor(q[start: start + CONTEXT_LENGTH]))
            targets.append(q[start + CONTEXT_LENGTH: start + CONTEXT_LENGTH + HORIZON])

        if not contexts:
            continue

        all_samples = []
        for i in range(0, len(contexts), BATCH):
            batch = contexts[i: i + BATCH]
            # forecast shape: (batch, num_samples, prediction_length)
            forecast = pipeline.predict(
                batch,
                prediction_length=HORIZON,
                num_samples=num_samples,
            )
            all_samples.append(forecast.numpy())

        torch.cuda.empty_cache()

        # samples: (N_steps, num_samples, horizon) → squeeze horizon dim
        samples_np = np.concatenate(all_samples, axis=0)  # (N, num_samples, H)
        obs = np.array([t[0] for t in targets], dtype=np.float32)  # (N,)
        pred = samples_np[:, :, 0]  # (N, num_samples)

        # Clip to same length
        n = min(len(obs), pred.shape[0])
        results[bid] = {"obs": obs[:n], "samples": pred[:n]}

    return results


# ── Coverage computation ──────────────────────────────────────────────────────

def classify_regimes(obs: np.ndarray) -> np.ndarray:
    """
    Classify each timestep into flow regime using basin-specific percentiles.
    Returns int array: 0=low, 1=normal, 2=flood.
    """
    p25 = np.percentile(obs, 25)
    p75 = np.percentile(obs, 75)
    p95 = np.percentile(obs, 95)

    regime = np.full(len(obs), -1, dtype=int)
    regime[obs < p25]  = 0   # low flow
    regime[(obs >= p25) & (obs <= p75)] = 1  # normal
    regime[obs > p95]  = 2   # flood
    return regime


def empirical_coverage(obs: np.ndarray, samples: np.ndarray, alpha: float) -> float:
    """
    Coverage at nominal level (1-alpha).
    Interval: [q(alpha/2), q(1-alpha/2)] over the sample dimension.
    obs: (N,), samples: (N, S)
    """
    lo = np.percentile(samples, 100 * alpha / 2,       axis=1)
    hi = np.percentile(samples, 100 * (1 - alpha / 2), axis=1)
    return float(np.mean((obs >= lo) & (obs <= hi)))


def compute_coverage_by_regime(basin_results: dict) -> dict:
    """
    Aggregate coverage data across all basins, stratified by regime.

    Returns:
        {regime_name: np.ndarray of empirical coverages at NOMINAL_LEVELS}
    """
    regime_obs     = {0: [], 1: [], 2: []}
    regime_samples = {0: [], 1: [], 2: []}

    for bid, data in basin_results.items():
        obs     = data["obs"]
        samples = data["samples"]
        regimes = classify_regimes(obs)
        for r in [0, 1, 2]:
            mask = regimes == r
            if np.sum(mask) >= 3:
                regime_obs[r].append(obs[mask])
                regime_samples[r].append(samples[mask])

    coverage_by_regime = {}
    for r, name in enumerate(REGIME_NAMES):
        if not regime_obs[r]:
            continue
        obs_all     = np.concatenate(regime_obs[r])
        samples_all = np.concatenate(regime_samples[r])
        coverages = []
        for level in NOMINAL_LEVELS:
            alpha = 1.0 - level
            cov = empirical_coverage(obs_all, samples_all, alpha)
            coverages.append(cov)
        coverage_by_regime[name] = np.array(coverages)
        n = len(obs_all)
        print(f"    {name}: n={n:,}  coverages={[f'{c:.2f}' for c in coverages]}")

    return coverage_by_regime


# ── Figure generation ─────────────────────────────────────────────────────────

def plot_reliability_diagrams(all_coverage: dict) -> Path:
    """
    all_coverage: {dataset_name: {regime_name: np.ndarray}}
    """
    valid_ds = [(ds, cov) for ds, cov in all_coverage.items() if cov]
    n = len(valid_ds)
    if n == 0:
        print("No valid coverage data to plot.")
        return None
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for idx, (ds_name, cov_data) in enumerate(valid_ds):
        ax = axes[idx]

        # Perfect calibration reference
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Perfect calibration")

        # Shaded reliability band (acceptable range ±0.05)
        ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05],
                        color="gray", alpha=0.08)

        for name, color in zip(REGIME_NAMES, REGIME_COLORS):
            if name not in cov_data:
                continue
            coverages = cov_data[name]
            ax.plot(NOMINAL_LEVELS, coverages, "o-",
                    color=color, lw=2, ms=5, label=name)

        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Nominal coverage", fontsize=10)
        ax.set_ylabel("Empirical coverage", fontsize=10)
        ax.set_title(ds_name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(True, alpha=0.25)

    # Hide unused panels
    for idx in range(len(valid_ds), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Chronos reliability diagrams by flow regime (20 basins per dataset, test period 2010-2017)\n"
        "Points below the diagonal indicate overconfident intervals (empirical < nominal); "
        "points above indicate underconfident intervals.",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()

    out = FIGURES_DIR / "figure_reliability_diagrams.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_basins",   type=int, default=50,
                        help="Basins per dataset (default 50)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Chronos Monte-Carlo samples (default 100)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--force_rerun", action="store_true",
                        help="Ignore cached results and re-run Chronos")
    args = parser.parse_args()

    print(f"\nReliability diagram computation")
    print(f"  Device        : {DEVICE}")
    print(f"  Basins/dataset: {args.n_basins}")
    print(f"  Chronos samples: {args.num_samples}")
    print(f"  Datasets      : {args.datasets}")

    cache_path = RESULTS_DIR / "reliability_coverage_cache.npz"
    all_coverage = {}

    if cache_path.exists() and not args.force_rerun:
        print(f"\nLoading cached coverage data from {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        all_coverage = cache["data"].item()
    else:
        for ds_name in args.datasets:
            print(f"\n{'-'*50}")
            print(f"Dataset: {ds_name}")
            basins = load_basin_subset(ds_name, args.n_basins)
            print(f"  {len(basins)} basins loaded")
            if not basins:
                continue

            basin_results = run_chronos_probabilistic(basins, args.num_samples)
            print(f"  Computing coverage by regime ...")
            cov = compute_coverage_by_regime(basin_results)
            all_coverage[ds_name] = cov

        np.savez(cache_path, data=all_coverage)
        print(f"\nSaved coverage cache -> {cache_path}")

    plot_reliability_diagrams(all_coverage)
    print("\nDone. Add figure_reliability_diagrams.png to the manuscript.")


if __name__ == "__main__":
    main()
