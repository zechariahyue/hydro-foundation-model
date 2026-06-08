#!/usr/bin/env python3
"""
Unified PyTorch data loader for multi-basin streamflow forecasting.
Loads preprocessed parquet files and serves (forcing_seq, static_attrs) -> streamflow pairs.
"""

import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Windows multiprocessing spawn causes silent DataLoader crashes
_NUM_WORKERS = 0 if sys.platform == "win32" else 4
from pathlib import Path
from typing import List, Optional, Tuple


class BasinDataset(Dataset):
    """
    Dataset for a single basin: sliding windows of (forcing, static) -> discharge.
    """

    def __init__(
        self,
        timeseries: pd.DataFrame,
        static_attrs: Optional[np.ndarray],
        seq_length: int = 365,
        forecast_horizon: int = 1,
        forcing_cols: Optional[List[str]] = None,
        target_col: str = "QObs(mm/d)",
    ):
        self.seq_length = seq_length
        self.horizon = forecast_horizon
        self.static = static_attrs  # (n_static,) or None

        if forcing_cols is None:
            forcing_cols = [c for c in timeseries.columns
                           if c not in [target_col, "basin_id", "dataset"]]

        # Extract arrays
        self.forcing = timeseries[forcing_cols].values.astype(np.float32)
        self.target = timeseries[target_col].values.astype(np.float32)

        # Valid indices: need seq_length input + horizon output
        self.n_samples = len(self.target) - self.seq_length - self.horizon + 1

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        # Input: forcing over [idx, idx+seq_length)
        x_forcing = self.forcing[idx: idx + self.seq_length]
        # Target: discharge at [idx+seq_length, idx+seq_length+horizon)
        y = self.target[idx + self.seq_length: idx + self.seq_length + self.horizon]

        x_forcing = torch.from_numpy(x_forcing)
        y = torch.from_numpy(y)

        if self.static is not None:
            x_static = torch.from_numpy(self.static.astype(np.float32))
            return x_forcing, x_static, y
        return x_forcing, y


class MultiBasinDataset(Dataset):
    """
    Combines multiple basins into one dataset for training.
    """

    def __init__(
        self,
        data_dir: str,
        dataset_names: List[str],
        attrs_df: pd.DataFrame,
        seq_length: int = 365,
        forecast_horizon: int = 1,
        date_start: str = None,
        date_end: str = None,
        forcing_cols: Optional[List[str]] = None,
        target_col: str = "QObs(mm/d)",
        normalize: bool = True,
    ):
        self.seq_length = seq_length
        self.horizon = forecast_horizon
        self.target_col = target_col
        self.normalize = normalize

        # Collect all samples across basins
        self.samples = []  # list of (forcing_window, static, target)
        self.basin_ids = []
        self.basin_sample_counts = {}

        # Compute normalization stats
        self.forcing_mean = None
        self.forcing_std = None
        self.target_mean = None
        self.target_std = None
        self.static_mean = None
        self.static_std = None

        all_forcing = []
        all_target = []
        all_static = []

        data_path = Path(data_dir)

        for ds_name in dataset_names:
            ds_dir = data_path / ds_name
            if not ds_dir.exists():
                print(f"  Warning: {ds_dir} not found, skipping.")
                continue

            parquet_files = sorted(ds_dir.glob("*.parquet"))
            for pf in parquet_files:
                basin_id = pf.stem
                try:
                    df = pd.read_parquet(pf)
                except Exception:
                    continue

                # Filter date range
                if date_start:
                    df = df[df.index >= date_start]
                if date_end:
                    df = df[df.index <= date_end]

                if len(df) < seq_length + forecast_horizon:
                    continue

                # Get forcing columns
                if forcing_cols is None:
                    fcols = [c for c in df.columns
                             if c not in [target_col, "basin_id", "dataset"]]
                else:
                    fcols = [c for c in forcing_cols if c in df.columns]

                if target_col not in df.columns:
                    continue

                # Coerce target to numeric (handles string columns from CAMELS-CL)
                df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

                # Drop rows with NaN target
                df = df.dropna(subset=[target_col])
                if len(df) < seq_length + forecast_horizon:
                    continue

                # Pad forcing to fixed width so all basins collate cleanly
                FORCING_SIZE = 7
                raw_forcing = df[fcols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
                if raw_forcing.shape[1] >= FORCING_SIZE:
                    forcing = raw_forcing[:, :FORCING_SIZE]
                else:
                    pad = np.zeros((len(raw_forcing), FORCING_SIZE - raw_forcing.shape[1]), dtype=np.float32)
                    forcing = np.concatenate([raw_forcing, pad], axis=1)
                target = df[target_col].values.astype(np.float32)

                # Static attributes (fixed size for stacking)
                STATIC_SIZE = 50
                static = None
                full_id = basin_id
                if full_id in attrs_df.index:
                    static_row = attrs_df.loc[full_id]
                    if isinstance(static_row, pd.DataFrame):
                        static_row = static_row.iloc[0]
                    numeric_vals = pd.to_numeric(static_row, errors="coerce").values.astype(np.float32)
                    numeric_vals = np.nan_to_num(numeric_vals, nan=0.0)
                    # Pad or truncate to fixed size
                    if len(numeric_vals) >= STATIC_SIZE:
                        static = numeric_vals[:STATIC_SIZE]
                    else:
                        static = np.zeros(STATIC_SIZE, dtype=np.float32)
                        static[:len(numeric_vals)] = numeric_vals

                all_forcing.append(forcing)
                all_target.append(target)
                if static is not None:
                    all_static.append(static)

                # Create sliding windows
                n_windows = len(target) - seq_length - forecast_horizon + 1
                for i in range(n_windows):
                    self.samples.append((forcing, target, static, i, basin_id))
                    self.basin_ids.append(basin_id)

                self.basin_sample_counts[basin_id] = n_windows

        # Compute normalization statistics
        if all_forcing and self.normalize:
            cat_forcing = np.concatenate(all_forcing, axis=0)
            self.forcing_mean = np.nanmean(cat_forcing, axis=0)
            self.forcing_std = np.nanstd(cat_forcing, axis=0) + 1e-8

            cat_target = np.concatenate(all_target)
            self.target_mean = np.nanmean(cat_target)
            self.target_std = np.nanstd(cat_target) + 1e-8

            if all_static:
                cat_static = np.stack(all_static)
                self.static_mean = np.nanmean(cat_static, axis=0)
                self.static_std = np.nanstd(cat_static, axis=0) + 1e-8

        print(f"  Loaded {len(self.samples)} samples from "
              f"{len(self.basin_sample_counts)} basins.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        forcing, target, static, window_idx, basin_id = self.samples[idx]

        # Extract window
        x = forcing[window_idx: window_idx + self.seq_length].copy()
        y = target[window_idx + self.seq_length:
                   window_idx + self.seq_length + self.horizon].copy()

        # Normalize
        if self.normalize and self.forcing_mean is not None:
            # Only normalize columns that exist in this basin's forcing
            n_cols = min(len(self.forcing_mean), x.shape[1])
            x[:, :n_cols] = (x[:, :n_cols] - self.forcing_mean[:n_cols]) / self.forcing_std[:n_cols]
            y = (y - self.target_mean) / self.target_std

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        # Always return 3 tensors so DataLoader can collate uniformly
        STATIC_SIZE = 50
        if static is not None:
            s = static.copy()
            if self.normalize and self.static_mean is not None:
                n_s = min(len(self.static_mean), len(s))
                s[:n_s] = (s[:n_s] - self.static_mean[:n_s]) / self.static_std[:n_s]
        else:
            s = np.zeros(STATIC_SIZE, dtype=np.float32)

        return x, torch.from_numpy(s), y

    def get_norm_stats(self) -> dict:
        """Return normalization statistics for denormalization at eval time."""
        return {
            "forcing_mean": self.forcing_mean,
            "forcing_std": self.forcing_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "static_mean": self.static_mean,
            "static_std": self.static_std,
        }


class TimeSeriesDataset(Dataset):
    """
    Simple dataset for foundation model inference.
    Returns raw discharge time series per basin (no windowing — the model handles context).
    """

    def __init__(
        self,
        data_dir: str,
        dataset_names: List[str],
        date_start: str = None,
        date_end: str = None,
        context_length: int = 512,
        target_col: str = "QObs(mm/d)",
    ):
        self.context_length = context_length
        self.basins = []  # list of (basin_id, dataset_name, discharge_array)

        data_path = Path(data_dir)
        for ds_name in dataset_names:
            ds_dir = data_path / ds_name
            if not ds_dir.exists():
                continue
            for pf in sorted(ds_dir.glob("*.parquet")):
                basin_id = pf.stem
                try:
                    df = pd.read_parquet(pf)
                except Exception:
                    continue

                if date_start:
                    df = df[df.index >= date_start]
                if date_end:
                    df = df[df.index <= date_end]

                if target_col not in df.columns:
                    continue

                q = df[target_col].dropna().values.astype(np.float32)
                if len(q) >= context_length + 1:
                    self.basins.append((basin_id, ds_name, q))

        print(f"  TimeSeriesDataset: {len(self.basins)} basins loaded.")

    def __len__(self):
        return len(self.basins)

    def __getitem__(self, idx):
        basin_id, ds_name, q = self.basins[idx]
        return basin_id, ds_name, torch.from_numpy(q)


def create_dataloaders(
    data_dir: str,
    attrs_path: str,
    source_datasets: List[str],
    target_datasets: List[str],
    config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.
    Train: source basins, train period
    Val: source basins, val period
    Test: target basins, test period (zero-shot)
    """
    attrs_df = pd.read_csv(attrs_path, index_col=0, dtype={0: str})

    print("Creating training dataset (source basins, train period)...")
    train_ds = MultiBasinDataset(
        data_dir=data_dir,
        dataset_names=source_datasets,
        attrs_df=attrs_df,
        seq_length=config["seq_length"],
        forecast_horizon=config["forecast_horizon"],
        date_start=config["train_start"],
        date_end=config["train_end"],
    )

    print("Creating validation dataset (source basins, val period)...")
    val_ds = MultiBasinDataset(
        data_dir=data_dir,
        dataset_names=source_datasets,
        attrs_df=attrs_df,
        seq_length=config["seq_length"],
        forecast_horizon=config["forecast_horizon"],
        date_start=config["val_start"],
        date_end=config["val_end"],
        normalize=False,
    )
    # Use train stats for val normalization
    val_ds.forcing_mean = train_ds.forcing_mean
    val_ds.forcing_std = train_ds.forcing_std
    val_ds.target_mean = train_ds.target_mean
    val_ds.target_std = train_ds.target_std
    val_ds.static_mean = train_ds.static_mean
    val_ds.static_std = train_ds.static_std
    val_ds.normalize = True

    train_loader = DataLoader(
        train_ds, batch_size=config.get("batch_size", 256),
        shuffle=True, num_workers=config.get("num_workers", _NUM_WORKERS),
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.get("batch_size", 256),
        shuffle=False, num_workers=config.get("num_workers", _NUM_WORKERS),
        pin_memory=True,
    )

    return train_loader, val_loader, train_ds.get_norm_stats()
