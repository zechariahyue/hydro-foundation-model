#!/usr/bin/env python3
"""
Evaluate trained LSTM baseline on all target datasets (zero-shot transfer).

Loads the saved lstm_best.pt checkpoint (trained on CAMELS-US) and evaluates
per-basin NSE/KGE/RMSE on each target dataset using CAMELS-US normalization stats.

Usage:
    python eval_baselines_on_targets.py
    python eval_baselines_on_targets.py --test_start 2000-01-01 --test_end 2017-12-31
"""

import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.baselines import StreamflowLSTM
from evaluation.metrics import nse, kge, rmse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(os.environ.get("CAMELS_DATA_DIR", "data/processed"))
CKPT_DIR = Path(os.environ.get("CKPT_DIR", "checkpoints"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DATASETS = ["CAMELS-BR", "CAMELS-CL", "CAMELS-AUS", "LamaH-CE", "CAMELS-IND"]
FORCING_SIZE = 7   # Must match dataloader hardcoded value
STATIC_SIZE = 50   # Must match dataloader hardcoded value
TARGET_COL = "QObs(mm/d)"


def load_basin_parquet(path: Path, date_start: str, date_end: str):
    """Load a single basin parquet, filter date range, return (forcing, target) arrays."""
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return None, None, None

    if date_start:
        df = df[df.index >= date_start]
    if date_end:
        df = df[df.index <= date_end]

    if TARGET_COL not in df.columns:
        return None, None, None

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL])

    if len(df) < 366:
        return None, None, None

    # Forcing: all columns except target/metadata
    fcols = [c for c in df.columns if c not in [TARGET_COL, "basin_id", "dataset"]]
    raw_forcing = df[fcols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)

    # Pad/truncate to FORCING_SIZE to match training
    if raw_forcing.shape[1] >= FORCING_SIZE:
        forcing = raw_forcing[:, :FORCING_SIZE]
    else:
        pad = np.zeros((len(raw_forcing), FORCING_SIZE - raw_forcing.shape[1]), dtype=np.float32)
        forcing = np.concatenate([raw_forcing, pad], axis=1)

    target = df[TARGET_COL].values.astype(np.float32)
    return forcing, target, df.index


def eval_basin(model, forcing: np.ndarray, target: np.ndarray,
               norm_stats: dict, seq_length: int = 365) -> dict:
    """Run LSTM on one basin, return per-basin metrics."""
    # Apply CAMELS-US normalization
    fm = norm_stats["forcing_mean"]
    fs = norm_stats["forcing_std"]
    tm = norm_stats["target_mean"]
    ts = norm_stats["target_std"]

    n_cols = min(len(fm), forcing.shape[1]) if fm is not None else forcing.shape[1]
    if fm is not None:
        f_norm = forcing.copy()
        f_norm[:, :n_cols] = (forcing[:, :n_cols] - fm[:n_cols]) / fs[:n_cols]
    else:
        f_norm = forcing

    # Sliding window inference
    n_windows = len(target) - seq_length
    if n_windows <= 0:
        return None

    all_pred = []
    all_obs = []

    model.eval()
    with torch.no_grad():
        # Batch all windows for efficiency
        batch_size = 512
        for start in range(0, n_windows, batch_size):
            end = min(start + batch_size, n_windows)
            x_batch = np.stack([f_norm[i: i + seq_length] for i in range(start, end)])
            x_t = torch.from_numpy(x_batch).to(DEVICE)
            # No static attrs for cross-dataset zero-shot (use zeros)
            x_s = torch.zeros(len(x_batch), STATIC_SIZE, device=DEVICE)
            pred = model(x_t, x_s).cpu().numpy().flatten()
            # Denormalize
            if tm is not None:
                pred = pred * ts + tm
            obs = target[start + seq_length: end + seq_length]
            all_pred.append(pred)
            all_obs.append(obs)

    obs_all = np.concatenate(all_obs)
    pred_all = np.concatenate(all_pred)

    # Clip predictions to non-negative (streamflow cannot be negative)
    pred_all = np.clip(pred_all, 0, None)

    return {
        "NSE": nse(obs_all, pred_all),
        "KGE": kge(obs_all, pred_all),
        "RMSE": rmse(obs_all, pred_all),
        "n_predictions": len(obs_all),
    }


def evaluate_dataset(model, norm_stats: dict, dataset_name: str,
                     date_start: str, date_end: str, seq_length: int = 365):
    """Evaluate LSTM on all basins in a target dataset."""
    ds_dir = DATA_DIR / dataset_name
    if not ds_dir.exists():
        print(f"  [SKIP] {dataset_name}: processed data not found at {ds_dir}")
        return None

    parquet_files = sorted(ds_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"  [SKIP] {dataset_name}: no parquet files found")
        return None

    print(f"\n  Evaluating {dataset_name} ({len(parquet_files)} basins)...")
    records = []
    skipped = 0

    for pf in parquet_files:
        basin_id = pf.stem
        forcing, target, _ = load_basin_parquet(pf, date_start, date_end)
        if forcing is None:
            skipped += 1
            continue

        metrics = eval_basin(model, forcing, target, norm_stats, seq_length)
        if metrics is None:
            skipped += 1
            continue

        records.append({
            "basin_id": basin_id,
            "model": "lstm",
            "mode": "zero_shot",
            "dataset": dataset_name,
            **metrics,
        })

    print(f"  Done: {len(records)} basins evaluated, {skipped} skipped.")
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_start", default="2000-01-01",
                        help="Start of test period (default: 2000-01-01)")
    parser.add_argument("--test_end", default="2017-12-31",
                        help="End of test period (default: 2017-12-31)")
    parser.add_argument("--seq_length", type=int, default=365,
                        help="Input sequence length (must match training)")
    parser.add_argument("--datasets", nargs="+", default=TARGET_DATASETS,
                        help="Target datasets to evaluate")
    args = parser.parse_args()

    # Load checkpoint
    ckpt_path = CKPT_DIR / "lstm_best.pt"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        print("  Run: python models/train_baselines.py --model lstm")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    norm_stats = ckpt["norm_stats"]

    print(f"  Model config: {cfg}")
    print(f"  Best epoch: {ckpt.get('epoch', 'unknown')} | Device: {DEVICE}")
    print(f"  Test period: {args.test_start} to {args.test_end}")

    # Reconstruct model
    model = StreamflowLSTM(
        n_forcing=cfg["n_forcing"],
        n_static=cfg["n_static"],
        hidden_size=256,
        num_layers=2,
        dropout=0.0,  # No dropout at eval time
        horizon=cfg["horizon"],
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Evaluate on each target dataset
    all_results = []
    for ds_name in args.datasets:
        df = evaluate_dataset(
            model, norm_stats, ds_name,
            args.test_start, args.test_end, args.seq_length
        )
        if df is not None and len(df) > 0:
            out_path = RESULTS_DIR / f"lstm_zero_shot_{ds_name}.csv"
            df.to_csv(out_path, index=False)
            print(f"  Saved: {out_path.name}")
            median_nse = df["NSE"].median()
            print(f"  Median NSE: {median_nse:.3f}")
            all_results.append(df)

    if all_results:
        print(f"\n{'=' * 60}")
        print("LSTM Zero-Shot Transfer Summary")
        print(f"{'=' * 60}")
        combined = pd.concat(all_results, ignore_index=True)
        summary = combined.groupby("dataset")["NSE"].agg(["median", "mean", "std", "count"])
        print(summary.to_string())
        print(f"\nOverall median NSE: {combined['NSE'].median():.3f}")
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()
