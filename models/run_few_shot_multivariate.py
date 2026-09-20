#!/usr/bin/env python3
"""Multivariate few-shot experiment (answers: can adding climate forcings + a little local
data beat univariate few-shot, persistence, and the zero-shot foundation model?).

Only datasets with forcings in the processed data can be used:
  CAMELS-US  -> prcp, srad, tmax, tmin, vp, swe, dayl  (richest test)
  CAMELS-AUS -> prcp                                   (precip-only test)
(CAMELS-IND / CAMELS-BR have NO forcings in the processed files -> cannot be done.)

Protocol (matches the univariate few-shot in spirit): per basin, first `fraction` of the
most-recent-5yr record is training, the rest is the test split. Training windows are POOLED
across basins; one model per (dataset, fraction). Two variants trained identically except for
inputs: UNIVARIATE (discharge only) vs MULTIVARIATE (discharge + forcings). Persistence is
computed on the IDENTICAL test windows as the matched naive baseline (the "fair baseline").
Per-basin NSE; report medians. Short context (CTX) so higher fractions have trainable windows.

Usage: python run_few_shot_multivariate.py --dataset CAMELS-US --fraction 0.10
"""
import os
import argparse, time, os, sys
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.metrics import nse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(os.environ.get("CAMELS_DATA_DIR", "data/processed"))
OUT = Path(os.environ.get("RESULTS_DIR", "results"))
CTX, HORIZON, STRIDE = 30, 1, 7
FORC = {"CAMELS-US": ["prcp", "srad", "tmax", "tmin", "vp", "swe", "dayl"],
        "CAMELS-AUS": ["prcp"]}


def load(ds, last_years=5):
    out = {}
    for pf in sorted((DATA_DIR / ds).glob("*.parquet")):
        df = pd.read_parquet(pf)
        if not isinstance(df.index, pd.DatetimeIndex):
            try: df.index = pd.to_datetime(df.index)
            except Exception: pass
        nk = int(last_years * 365.25)
        try: nd = df.index.normalize().nunique()
        except Exception: nd = 0
        df = df[df.index > df.index.max() - pd.DateOffset(years=last_years)] if nd >= 0.5*len(df) else df.iloc[-nk:]
        if "QObs(mm/d)" not in df.columns: continue
        cols = ["QObs(mm/d)"] + FORC[ds]
        d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(d) > 100:
            out[pf.stem] = d
    return out


class LSTM(nn.Module):
    def __init__(self, n_in, hid=64):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hid, batch_first=True)
        self.fc = nn.Linear(hid, 1)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1, :]).squeeze(-1)


def windows(arr, q, feat_idx):
    """arr: (T, F) normalized features; q: (T,) normalized discharge target source.
    Returns X (n,CTX,len(feat_idx)), y (n,)."""
    X, Y = [], []
    for s in range(0, len(arr) - CTX - HORIZON + 1, STRIDE):
        X.append(arr[s:s+CTX, feat_idx]); Y.append(q[s+CTX])
    if not X: return None, None
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def train_eval(splits, feat_idx, norm, epochs=15):
    mu, sd, qmu, qsd = norm
    Xtr, Ytr = [], []
    for bid, sp in splits.items():
        a = (sp["train_feats"] - mu) / sd
        qn = (sp["train_q"] - qmu) / qsd
        X, Y = windows(a, qn, feat_idx)
        if X is not None: Xtr.append(X); Ytr.append(Y)
    if not Xtr: return None
    Xtr = np.concatenate(Xtr); Ytr = np.concatenate(Ytr)
    model = LSTM(len(feat_idx)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    Xt = torch.tensor(Xtr).to(DEVICE); Yt = torch.tensor(Ytr).to(DEVICE)
    bs = 256
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]
            opt.zero_grad(); loss = lossf(model(Xt[idx]), Yt[idx]); loss.backward(); opt.step()
    # eval per basin on test split
    model.eval(); res = {}
    with torch.no_grad():
        for bid, sp in splits.items():
            a = (sp["test_feats"] - mu) / sd
            qn = (sp["test_q"] - qmu) / qsd
            X, Y = windows(a, qn, feat_idx)
            if X is None or len(X) < 5: continue
            pred = model(torch.tensor(X).to(DEVICE)).cpu().numpy()
            pred = pred * qsd + qmu; obs = Y * qsd + qmu
            v = nse(obs, pred)
            if np.isfinite(v): res[bid] = v
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CAMELS-US")
    ap.add_argument("--fraction", type=float, default=0.10)
    args = ap.parse_args()
    ds = args.dataset
    print(f"=== Multivariate few-shot: {ds} f={args.fraction} (device {DEVICE}) ===")
    basins = load(ds)
    print(f"Loaded {len(basins)} basins with forcings {FORC[ds]}")

    # per-basin split + persistence on test windows
    splits, pers = {}, {}
    for bid, d in basins.items():
        n = len(d); ntr = max(CTX + HORIZON + 1, int(n * args.fraction))
        if n - ntr < CTX + HORIZON + 5: continue
        feats = d.values.astype(np.float32)            # col0 = discharge, rest forcings
        q = feats[:, 0]
        splits[bid] = {"train_feats": feats[:ntr], "train_q": q[:ntr],
                       "test_feats": feats[ntr:], "test_q": q[ntr:]}
        # persistence on identical test windows
        po, ps = [], []
        for s in range(0, len(q[ntr:]) - CTX - HORIZON + 1, STRIDE):
            t = ntr + s + CTX; po.append(q[t]); ps.append(q[t-1])
        if len(po) >= 5:
            v = nse(np.array(po), np.array(ps))
            if np.isfinite(v): pers[bid] = v
    print(f"Usable basins: {len(splits)}")
    if not splits: return

    # normalization from pooled training
    allf = np.concatenate([s["train_feats"] for s in splits.values()])
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    qmu, qsd = float(allf[:, 0].mean()), float(allf[:, 0].std() + 1e-6)
    norm = (mu, sd, qmu, qsd)

    t0 = time.time()
    uni = train_eval(splits, [0], norm)                       # discharge only
    multi = train_eval(splits, list(range(allf.shape[1])), norm)  # discharge + forcings
    def med(d): return float(np.median(list(d.values()))) if d else float("nan")
    rows = []
    for bid in splits:
        rows.append({"basin_id": bid, "dataset": ds, "fraction": args.fraction,
                     "nse_univariate_lstm": uni.get(bid, np.nan) if uni else np.nan,
                     "nse_multivariate_lstm": multi.get(bid, np.nan) if multi else np.nan,
                     "nse_persistence": pers.get(bid, np.nan)})
    pd.DataFrame(rows).to_csv(OUT / f"mv_fewshot_{args.fraction}_{ds}.csv", index=False)
    print(f"[{time.time()-t0:.0f}s] medians  uni-LSTM={med(uni):.3f}  "
          f"multi-LSTM={med(multi):.3f}  persistence={med(pers):.3f}  (n={len(splits)})")
    print(f"Saved mv_fewshot_{args.fraction}_{ds}.csv")


if __name__ == "__main__":
    main()
