#!/usr/bin/env python3
"""
Fair supervised baseline: a per-dataset GLOBAL LSTM with per-basin standardisation,
evaluated on the SAME consistent protocol as the foundation models (last 5 years of each
basin's record, stride-7, target day t = start + CTX), using the SAME compute_all_metrics.

Why this exists
---------------
The Table-4 "LSTM (zero-shot)" baseline transfers a CAMELS-US-trained LSTM to other regions
using SOURCE normalisation -> it collapses (median NSE down to -302,970) purely from a
normalisation mismatch. That is a straw-man. This script instead trains one LSTM per dataset,
pooled across that dataset's basins, with PER-BASIN z-scoring from each basin's TRAINING
period (everything before the last-5yr test window). This is the standard large-sample /
Caravan-style "global LSTM" and is the fair supervised comparison for the zero-shot FMs:
  - univariate (past discharge only), matching the FM inputs;
  - trained on the target basins' own pre-test history (the "gauged, data-available" setting);
  - never sees any test-window value during training; standardisation uses train-period stats only.

The FM advantage being tested: a frozen, zero-shot FM prior vs. a model actually TRAINED on
the region's data. If the FM ties or beats this, that is a strong result; if the trained LSTM
wins where data are plentiful, that is the honest, expected nuance.

Outputs: experiments/results/lstm_global_<DATASET>_l5s7.csv  (same columns as the other models).
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "evaluation"))
from metrics import compute_all_metrics  # noqa: E402

DATA_DIR = Path(os.environ.get("CAMELS_DATA_DIR", "data/processed"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CTX = 512      # defines the target points (identical to the FM benchmark)
STRIDE = 7     # uniform evaluation stride (identical to the FM benchmark)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _norm_id(x):
    """Canonical basin id (robust to leading-zero loss when CSVs are read as int, e.g. USGS
    gauge ids like 01013500 -> 1013500)."""
    return str(x).strip().lstrip("0") or "0"


def load_fm_basin_set(dataset_name: str):
    """The exact set of basins the foundation models evaluated on this dataset (so the LSTM is
    reported on a strictly matched basin set). Read from the TimesFM zero-shot l5s7 CSV."""
    f = RESULTS_DIR / f"timesfm_zero_shot_{dataset_name}_l5s7.csv"
    if not f.exists():
        return None
    ids = pd.read_csv(f, dtype={"basin_id": str})["basin_id"]
    return {_norm_id(x) for x in ids}


def load_basins_split(dataset_name: str, last_years: int = 5):
    """Per basin: train array (pre-test), test array, train mean/std.

    The TEST series is constructed with the SAME logic as the foundation-model loader
    (run_foundation_models.load_basin_series): truncate the RAW dataframe first
    (date-based for genuine daily indices; position-based for degenerate ones such as
    CAMELS-BR), THEN coerce/dropna QObs. This guarantees the LSTM is evaluated on the
    identical test window -> identical one-day-ahead target points (stride 7) as the FMs.
    The TRAIN series is the pre-test remainder. Eligibility (len(test) > 100) mirrors the FM
    loader; additionally the LSTM requires enough pre-test history to form lookback windows."""
    ds_dir = DATA_DIR / dataset_name
    n_keep = int(last_years * 365.25)
    out = {}
    for pf in sorted(ds_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(pf)
        except Exception:
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                continue
        if "QObs(mm/d)" not in df.columns:
            continue
        try:
            ndates = df.index.normalize().nunique()
        except Exception:
            ndates = 0
        if ndates >= 0.5 * len(df):                      # genuine daily dates -> date split
            cutoff = df.index.max() - pd.DateOffset(years=last_years)
            test_df = df[df.index > cutoff]
            train_df = df[df.index <= cutoff]
        else:                                            # degenerate index (CAMELS-BR) -> position split
            test_df = df.iloc[-n_keep:]
            train_df = df.iloc[:-n_keep]
        test = pd.to_numeric(test_df["QObs(mm/d)"], errors="coerce").dropna().values.astype(np.float32)
        train = pd.to_numeric(train_df["QObs(mm/d)"], errors="coerce").dropna().values.astype(np.float32)
        if len(test) <= 100:                             # FM eligibility (load_basin_series: len(q) > 100)
            continue
        if len(test) < (CTX + 1) or len(train) < (LOOKBACK + 60):
            continue                                     # need a usable test window + enough train history
        mu = float(np.mean(train))
        sd = float(np.std(train))
        if not np.isfinite(sd) or sd <= 0:
            continue
        out[pf.stem] = {"train": train, "test": test, "mu": mu, "sd": sd}
    return out


class GlobalLSTM(nn.Module):
    def __init__(self, hidden=128, layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):              # x: (B, L, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)   # predict next-day (z-scored)


@torch.no_grad()
def infer(model, X, bs=2048):
    """Chunked inference (LSTM activations over many long sequences blow up memory if
    run in one shot). X: torch tensor (N, L, 1) on CPU. Returns np array (N,)."""
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs].to(DEVICE)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def build_training_windows(basins: dict, max_windows: int, rng: np.random.RandomState):
    """Pool z-scored (lookback -> next-day) windows across basins from the TRAIN period only."""
    X_parts, y_parts = [], []
    # budget windows per basin so large datasets (BR) don't dominate / explode memory
    n_basins = len(basins)
    per_basin = max(200, max_windows // max(1, n_basins))
    for b in basins.values():
        z = (b["train"] - b["mu"]) / b["sd"]
        n = len(z) - LOOKBACK - 1
        if n <= 0:
            continue
        starts = np.arange(0, n + 1)
        if len(starts) > per_basin:
            starts = rng.choice(starts, size=per_basin, replace=False)
        # vectorised window gather
        idx = starts[:, None] + np.arange(LOOKBACK)[None, :]
        X_parts.append(z[idx].astype(np.float32))           # (k, L)
        y_parts.append(z[starts + LOOKBACK].astype(np.float32))  # (k,)
    X = np.concatenate(X_parts, axis=0)[..., None]          # (N, L, 1)
    y = np.concatenate(y_parts, axis=0)                     # (N,)
    return X, y


def train_one(dataset: str, basins: dict, epochs: int, hidden: int, batch: int,
              max_windows: int, seed: int):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    X, y = build_training_windows(basins, max_windows, rng)
    # train/val split
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    nval = max(1, int(0.1 * len(X)))
    Xv, yv = X[:nval], y[:nval]
    Xt, yt = X[nval:], y[nval:]
    print(f"  [{dataset}] training windows: {len(Xt):,} (val {len(Xv):,}) from {len(basins)} basins")

    Xt_t = torch.from_numpy(Xt); yt_t = torch.from_numpy(yt)
    Xv_t = torch.from_numpy(Xv)  # kept on CPU; inference is chunked to GPU

    model = GlobalLSTM(hidden=hidden).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    best_val = float("inf"); best_state = None; patience = 5; bad = 0
    nt = len(Xt_t)
    for ep in range(epochs):
        model.train()
        order = torch.randperm(nt)
        tot = 0.0
        for i in range(0, nt, batch):
            bi = order[i:i + batch]
            xb = Xt_t[bi].to(DEVICE); yb = yt_t[bi].to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = lossf(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(bi)
        vpred = infer(model, Xv_t)
        vloss = float(np.mean((vpred - yv) ** 2))
        if vloss < best_val - 1e-5:
            best_val = vloss; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
        print(f"    ep{ep+1:02d} train_mse={tot/nt:.4f} val_mse={vloss:.4f}{'  *' if bad==0 else ''}")
        if bad >= patience:
            print("    early stop"); break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate(dataset: str, basins: dict, model, fm_set=None):
    """Eval on the test window at target points t=start+CTX (stride STRIDE), restricted to the
    foundation-model basin set (fm_set) so the comparison is strictly matched; the test series is
    FM-identical (load_basins_split), so target points coincide. LSTM input = z(q_test[t-LOOKBACK:t])
    using TRAIN stats; sim = inverse z-score."""
    model.eval()
    rows = []
    skipped = 0
    for bid, b in basins.items():
        if fm_set is not None and _norm_id(bid) not in fm_set:
            skipped += 1
            continue
        q = b["test"]; mu, sd = b["mu"], b["sd"]
        if len(q) < CTX + 1:
            continue
        starts = list(range(0, len(q) - CTX - 1 + 1, STRIDE))
        Xb, obs = [], []
        for start in starts:
            t = start + CTX
            Xb.append(((q[t - LOOKBACK:t] - mu) / sd).astype(np.float32))
            obs.append(q[t])
        if not obs:
            continue
        Xb = torch.from_numpy(np.asarray(Xb)[..., None])
        pred_z = infer(model, Xb)
        sim = pred_z * sd + mu                 # back to physical units
        obs = np.asarray(obs, dtype=np.float32)
        m = compute_all_metrics(obs, sim)
        m["n_predictions"] = len(obs)
        rows.append({"basin_id": bid, "model": "lstm_global", "mode": "supervised",
                     "dataset": dataset, **m})
    return rows


def main():
    global LOOKBACK
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["CAMELS-US", "CAMELS-BR", "CAMELS-CL", "CAMELS-AUS", "LamaH-CE", "CAMELS-IND"])
    ap.add_argument("--last_years", type=int, default=5)
    ap.add_argument("--lookback", type=int, default=365)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max_windows", type=int, default=200000)
    ap.add_argument("--out_suffix", default="_l5s7")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    LOOKBACK = args.lookback

    print(f"Device: {DEVICE} | CTX={CTX} stride={STRIDE} lookback={LOOKBACK}")
    summary = []
    for ds in args.datasets:
        print(f"\n=== {ds} ===")
        basins = load_basins_split(ds, args.last_years)
        if not basins:
            print("  no usable basins"); continue
        fm_set = load_fm_basin_set(ds)
        n_match = len([b for b in basins if fm_set is None or _norm_id(b) in fm_set])
        print(f"  usable basins: {len(basins)} (trained on all; evaluated on {n_match} matched to FM set)")
        model = train_one(ds, basins, args.epochs, args.hidden, args.batch, args.max_windows, args.seed)
        rows = evaluate(ds, basins, model, fm_set=fm_set)
        if not rows:
            print("  no eval rows"); continue
        df = pd.DataFrame(rows)
        fname = f"lstm_global_{ds}{args.out_suffix}.csv"
        df.to_csv(RESULTS_DIR / fname, index=False)
        med = df["NSE"].median()
        print(f"  -> {fname}: {len(df)} basins, median NSE = {med:.3f}")
        summary.append((ds, len(df), med))

    print("\n==== GLOBAL LSTM (fair supervised) median NSE ====")
    for ds, n, med in summary:
        print(f"  {ds:12s} n={n:5d}  median NSE = {med:.3f}")


if __name__ == "__main__":
    main()
