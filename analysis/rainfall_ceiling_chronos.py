#!/usr/bin/env python3
"""
RAINFALL-CEILING TEST for CHRONOS (and TimesFM like-for-like), 2026-09-13.

The v2 test (rainfall_ceiling_test_v2.py) was TimesFM-only, which SI S13 had to state as a
limitation after the 20-sample Chronos re-run made the two models near-equivalent. This runs the
identical protocol -- same 1,230 precip basins, same features, same half-split oracle, same
shuffled null -- for Chronos (twenty sampled trajectories, point forecast = their mean) and, on the
same target points, for TimesFM, so the two can be compared basin-by-basin.

Stride 3 rather than v2's stride 1: Chronos at 20 samples costs ~30 ms per context, so stride 1
(~1,300 contexts/basin) would be ~14 h; stride 3 (~430 contexts/basin, ~215 training rows for a
six-feature linear oracle) is ~5 h. v2's TimesFM stride-1 numbers remain the primary rainfall
result; this file is the model-comparison supplement to it.

Outputs: experiments/results/rainfall_ceiling_s3_per_basin.csv   (column `model`)
         experiments/results/RAINFALL_CEILING_CHRONOS.txt
Usage:   python rainfall_ceiling_chronos.py [--limit N] [--stride 3] [--num_samples 20]
"""
import os
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rainfall_ceiling_test_v2 as v2  # noqa: E402  (main-guarded; safe to import)

RES = v2.RES
lines = []


def say(s=""):
    print(s, flush=True); lines.append(s)


def make_chronos_forecaster(num_samples, batch):
    from chronos import ChronosPipeline
    from tqdm import tqdm
    pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-large", device_map="cuda", torch_dtype=torch.float32)

    def forecast(basins):
        out = {}
        for bid, (q, p) in tqdm(basins.items(), desc=f"Chronos s{v2.STRIDE} x{num_samples}"):
            starts = list(range(0, len(q) - v2.CTX - v2.HORIZON + 1, v2.STRIDE))
            if len(starts) < 100:
                continue
            ctxs = [torch.tensor(q[s:s + v2.CTX]) for s in starts]
            idx = np.array([s + v2.CTX for s in starts])
            preds = []
            for i in range(0, len(ctxs), batch):
                f = pipe.predict(ctxs[i:i + batch], prediction_length=v2.HORIZON, num_samples=num_samples)
                preds.extend(f.numpy().mean(axis=1)[:, 0].tolist())
            torch.cuda.empty_cache()
            out[bid] = (idx, q[idx], np.array(preds, np.float32), q[idx - 1])
        return out
    return forecast


def run(model, forecaster, limit):
    v2.forecast = forecaster
    frames = []
    for ds in ["CAMELS-US", "CAMELS-AUS"]:
        if limit:
            full = v2.load_with_precip
            v2.load_with_precip = lambda d, _f=full: dict(list(_f(d).items())[:limit])
        df = v2.analyse(ds)
        if limit:
            v2.load_with_precip = full
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["model"] = model
    df["wet_dry"] = df.err_wet / df.err_dry
    return df.rename(columns={"nse_tfm": "nse_model"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N basins per dataset")
    a = ap.parse_args()
    v2.STRIDE = a.stride
    tag = "_smoke" if a.limit else ""

    tfm_forecast = v2.forecast  # keep the original TimesFM forecaster before run() overwrites it
    d_c = run("chronos", make_chronos_forecaster(a.num_samples, a.batch), a.limit)
    torch.cuda.empty_cache()
    d_t = run("timesfm", tfm_forecast, a.limit)
    df = pd.concat([d_c, d_t], ignore_index=True)
    df.to_csv(RES / f"rainfall_ceiling_s3_per_basin{tag}.csv", index=False)

    say("=" * 86)
    say(f"RAINFALL CEILING -- Chronos ({a.num_samples} samples) vs TimesFM, stride {a.stride}, same basins/points")
    say("=" * 86)
    both = df.pivot_table(index=["dataset", "basin_id"], columns="model",
                          values=["wet_dry", "gain_lin", "gain_lin_shuf", "nse_model", "ac1", "err_wet", "err_dry", "rho_absres_P"])
    both = both.dropna(subset=[("wet_dry", "chronos"), ("wet_dry", "timesfm")])
    say(f"basins with both models: {len(both)}")
    for m in ["chronos", "timesfm"]:
        wd = both[("wet_dry", m)]
        say(f"\n--- {m} ---")
        say(f"  median per-basin wet/dry |residual| ratio = {wd.median():.2f}; wet>dry in {100*(wd>1).mean():.1f}% of basins")
        w, p_ = both[("err_wet", m)], both[("err_dry", m)]
        say(f"  Wilcoxon wet vs dry: p = {stats.wilcoxon(w, p_).pvalue:.1e}")
        r = stats.spearmanr(both[("ac1", m)], wd)
        say(f"  Spearman(rho_1, wet/dry) = {r[0]:+.3f}  p={r[1]:.1e}")
        r2 = stats.spearmanr(both[("ac1", m)], both[("gain_lin", m)], nan_policy="omit")
        say(f"  Spearman(rho_1, linear oracle gain) = {r2[0]:+.3f}  p={r2[1]:.1e}")
    ac1 = both[("ac1", "timesfm")]
    qs = ac1.quantile([0.25, 0.5, 0.75]).values
    qb = pd.cut(ac1, [-np.inf, *qs, np.inf], labels=["Q1 flashy", "Q2", "Q3", "Q4 damped"])
    say(f"\n{'bin':12s} {'n':>4s} {'ac1':>6s} | {'wet/dry CHR':>11s} {'wet/dry TFM':>11s} | {'lin.gain CHR':>12s} {'shuf CHR':>9s} {'lin.gain TFM':>12s} {'shuf TFM':>9s}")
    for b in ["Q1 flashy", "Q2", "Q3", "Q4 damped"]:
        g = both[qb == b]
        say(f"{b:12s} {len(g):4d} {g[('ac1','timesfm')].median():6.3f} | {g[('wet_dry','chronos')].median():11.2f} {g[('wet_dry','timesfm')].median():11.2f} | "
            f"{g[('gain_lin','chronos')].median():11.1%} {g[('gain_lin_shuf','chronos')].median():8.1%} "
            f"{g[('gain_lin','timesfm')].median():11.1%} {g[('gain_lin_shuf','timesfm')].median():8.1%}")
    dwd = both[("wet_dry", "chronos")] - both[("wet_dry", "timesfm")]
    say(f"\nPaired wet/dry ratio, Chronos minus TimesFM: median {dwd.median():+.2f}, Wilcoxon p = {stats.wilcoxon(dwd).pvalue:.1e}")
    say(f"Median NSE on these points: Chronos {both[('nse_model','chronos')].median():.3f}, TimesFM {both[('nse_model','timesfm')].median():.3f}")
    (RES / f"RAINFALL_CEILING_CHRONOS{tag}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote RAINFALL_CEILING_CHRONOS{tag}.txt")
