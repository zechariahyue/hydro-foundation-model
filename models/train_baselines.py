#!/usr/bin/env python3
"""
Step 3: Train LSTM and ConvLSTM baselines on source basins (CAMELS-US).

Usage:
    python train_baselines.py --model lstm
    python train_baselines.py --model convlstm
"""

import sys
import os
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# GPU Optimization: Enable TF32 for faster computation on Ampere+ GPUs (RTX 30/40/50 series)
# TF32 provides ~1.3x speedup with negligible accuracy loss for most workloads
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.baselines import StreamflowLSTM, StreamflowConvLSTM
from data.dataloader import MultiBasinDataset
from evaluation.metrics import nse, kge, rmse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(os.environ.get("CAMELS_DATA_DIR", "data/processed"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
CKPT_DIR = Path(os.environ.get("CKPT_DIR", "checkpoints"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def train_epoch(model, loader, optimizer, criterion, norm_stats, scaler=None):
    """
    Train for one epoch with optional mixed precision.

    Args:
        scaler: torch.amp.GradScaler for mixed precision training (provides ~1.5-2x speedup)
    """
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        if len(batch) == 3:
            x_forcing, x_static, y = batch
            x_forcing = x_forcing.to(DEVICE)
            x_static = x_static.to(DEVICE)
            y = y.to(DEVICE)

            # Mixed precision training: wrap forward pass in autocast
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    pred = model(x_forcing, x_static)
                    loss = criterion(pred, y)
            else:
                pred = model(x_forcing, x_static)
                loss = criterion(pred, y)
        else:
            x_forcing, y = batch
            x_forcing = x_forcing.to(DEVICE)
            y = y.to(DEVICE)

            # Mixed precision training: wrap forward pass in autocast
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    pred = model(x_forcing)
                    loss = criterion(pred, y)
            else:
                pred = model(x_forcing)
                loss = criterion(pred, y)

        optimizer.zero_grad()

        # Mixed precision backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, norm_stats):
    """Evaluate model, return per-basin metrics."""
    model.eval()
    all_obs = []
    all_sim = []

    target_mean = norm_stats.get("target_mean", 0)
    target_std = norm_stats.get("target_std", 1)

    for batch in loader:
        if len(batch) == 3:
            x_forcing, x_static, y = batch
            x_forcing = x_forcing.to(DEVICE)
            x_static = x_static.to(DEVICE)
            pred = model(x_forcing, x_static)
        else:
            x_forcing, y = batch
            x_forcing = x_forcing.to(DEVICE)
            pred = model(x_forcing)

        # Denormalize
        pred_np = pred.cpu().numpy() * target_std + target_mean
        y_np = y.numpy() * target_std + target_mean

        all_obs.append(y_np.flatten())
        all_sim.append(pred_np.flatten())

    obs = np.concatenate(all_obs)
    sim = np.concatenate(all_sim)

    return {
        "NSE": nse(obs, sim),
        "KGE": kge(obs, sim),
        "RMSE": rmse(obs, sim),
        "n_samples": len(obs),
    }


def train_model(model_type: str = "lstm"):
    print(f"\n{'=' * 60}")
    print(f"Training {model_type.upper()} baseline")
    print(f"Device: {DEVICE}")
    print(f"{'=' * 60}\n")

    # Load attributes
    attrs_path = DATA_DIR / "all_basin_attributes.csv"
    if not attrs_path.exists():
        print("ERROR: Run preprocess_camels.py first!")
        return
    attrs_df = pd.read_csv(attrs_path, index_col=0, dtype={0: str})

    # Config
    seq_length = 365
    horizon = 1
    batch_size = 2048  # Increased from 512 to 2048 for better GPU utilization (4x speedup)
    epochs = 30
    lr = 0.001

    # Create datasets
    # CAMELS-US v1.2 data runs 1980-2014
    print("Loading training data (CAMELS-US, 1980-2007)...")
    train_ds = MultiBasinDataset(
        data_dir=str(DATA_DIR),
        dataset_names=["CAMELS-US"],
        attrs_df=attrs_df,
        seq_length=seq_length,
        forecast_horizon=horizon,
        date_start="1980-01-01",
        date_end="2007-12-31",
    )

    print("Loading validation data (CAMELS-US, 2008-2014)...")
    val_ds = MultiBasinDataset(
        data_dir=str(DATA_DIR),
        dataset_names=["CAMELS-US"],
        attrs_df=attrs_df,
        seq_length=seq_length,
        forecast_horizon=horizon,
        date_start="2008-01-01",
        date_end="2014-12-31",
        normalize=False,
    )
    # Copy normalization stats from train
    norm_stats = train_ds.get_norm_stats()
    val_ds.forcing_mean = norm_stats["forcing_mean"]
    val_ds.forcing_std = norm_stats["forcing_std"]
    val_ds.target_mean = norm_stats["target_mean"]
    val_ds.target_std = norm_stats["target_std"]
    val_ds.static_mean = norm_stats["static_mean"]
    val_ds.static_std = norm_stats["static_std"]
    val_ds.normalize = True

    if len(train_ds) == 0:
        print("ERROR: No training samples found!")
        return

    # num_workers=0 required on Windows (multiprocessing spawn causes silent crashes)
    nw = 0 if sys.platform == "win32" else 4
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True,
    )

    # Determine input dimensions from first batch
    sample = train_ds[0]
    if len(sample) == 3:
        n_forcing = sample[0].shape[1]
        n_static = sample[1].shape[0]
    else:
        n_forcing = sample[0].shape[1]
        n_static = 0

    print(f"  n_forcing={n_forcing}, n_static={n_static}")
    print(f"  train samples={len(train_ds)}, val samples={len(val_ds)}")

    # Create model
    if model_type == "lstm":
        model = StreamflowLSTM(
            n_forcing=n_forcing, n_static=n_static,
            hidden_size=256, num_layers=2, dropout=0.2, horizon=horizon,
        )
    elif model_type == "convlstm":
        model = StreamflowConvLSTM(
            n_forcing=n_forcing, n_static=n_static,
            hidden_size=128, num_layers=2, kernel_size=3,
            dropout=0.2, horizon=horizon,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )
    criterion = nn.MSELoss()

    # Mixed precision training: GradScaler for automatic loss scaling (~1.5-2x speedup)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    # Training loop
    best_val_nse = -999
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, norm_stats, scaler)
        val_metrics = evaluate(model, val_loader, norm_stats)
        elapsed = time.time() - t0

        scheduler.step(train_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_NSE": val_metrics["NSE"],
            "val_KGE": val_metrics["KGE"],
            "val_RMSE": val_metrics["RMSE"],
            "time_s": elapsed,
        })

        print(f"  Epoch {epoch:2d}/{epochs} | loss={train_loss:.4f} | "
              f"val_NSE={val_metrics['NSE']:.4f} | val_KGE={val_metrics['KGE']:.4f} | "
              f"{elapsed:.1f}s")

        # Save best
        if val_metrics["NSE"] > best_val_nse:
            best_val_nse = val_metrics["NSE"]
            ckpt_path = CKPT_DIR / f"{model_type}_best.pt"
            torch.save({
                "model_state": model.state_dict(),
                "norm_stats": norm_stats,
                "config": {
                    "model_type": model_type,
                    "n_forcing": n_forcing,
                    "n_static": n_static,
                    "seq_length": seq_length,
                    "horizon": horizon,
                },
                "epoch": epoch,
                "val_NSE": best_val_nse,
            }, ckpt_path)
            print(f"    -> Saved best model (NSE={best_val_nse:.4f})")

    # Save training history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(RESULTS_DIR / f"{model_type}_training_history.csv", index=False)

    print(f"\nTraining complete. Best val NSE: {best_val_nse:.4f}")
    print(f"Checkpoint: {CKPT_DIR / f'{model_type}_best.pt'}")
    return best_val_nse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "convlstm", "both"], default="both")
    args = parser.parse_args()

    if args.model in ("lstm", "both"):
        train_model("lstm")
    if args.model in ("convlstm", "both"):
        train_model("convlstm")


if __name__ == "__main__":
    main()
