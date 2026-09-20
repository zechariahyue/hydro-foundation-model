#!/usr/bin/env python3
"""
DIRECT TEST OF THE RAINFALL-CEILING CLAIM.

The paper argues that the ceiling on a univariate foundation model is set by rainfall it
cannot observe, not by architecture. Until now that was an INTERPRETATION of an absence
(flashy basins forecast poorly; flashy basins are rainfall-driven). This script tests it.

Logic
-----
If TimesFM's residual error is caused by unobserved rainfall, then:
  (H-a) the residual should be systematically larger on rain days than on dry days, and
  (H-b) the residual should be PREDICTABLE FROM RAINFALL -- i.e. an oracle that is handed the
        observed precipitation should be able to remove a substantial part of it, and
  (H-c) both effects should strengthen as hydrograph memory (rho_1) falls.
Conversely, in high-memory basins the residual should be nearly rainfall-free, because
yesterday's flow already encodes the catchment state.

The oracle test is deliberately generous to the hypothesis: the correction is given the
*observed* precipitation on the very day being forecast, which a real forecaster would not
have (they would have a rainfall forecast). It therefore measures an UPPER BOUND on what
perfect rainfall knowledge could buy -- exactly the right quantity for a claim about an
information ceiling, and it must be reported as an upper bound, not as a proposed model.

Guards against fooling ourselves
--------------------------------
* The correction is fit on the first half of each basin's prediction points and scored ONLY
  on the held-out second half, so "rainfall helps" cannot be an artefact of overfitting.
* A NEGATIVE CONTROL uses SHUFFLED rainfall (same values, wrong days). If shuffled rain
  "helps" as much as real rain, the gain is fitting noise and the result is dead.
* The identical correction is applied to the PERSISTENCE residual. If rainfall is genuinely
  the missing information, it should help there too.

Protocol matches run_foundation_models.py exactly: last 5 years, context 512, horizon 1,
stride 7, target at index start+512.

Outputs: experiments/results/RAINFALL_CEILING_TEST.txt
         experiments/results/rainfall_ceiling_per_basin.csv
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CAMELS_DATA_DIR", "data/processed"))
RES = Path(os.environ.get("RESULTS_DIR", "results"))
QCOL = "QObs(mm/d)"
CTX, HORIZON, STRIDE, LAST_YEARS = 512, 1, 7, 5

lines = []


def say(s=""):
    print(s)
    lines.append(s)


def load_with_precip(ds):
    """Load basins truncated to the last 5 yr, keeping discharge AND precipitation aligned.
    Mirrors run_foundation_models.load_basin_series (incl. the degenerate-index fallback)."""
    out = {}
    for pf in sorted((DATA / ds).glob("*.parquet")):
        try:
            df = pd.read_parquet(pf)
        except Exception:
            continue
        if QCOL not in df.columns or "prcp" not in df.columns:
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                continue
        n_keep = int(LAST_YEARS * 365.25)
        try:
            degenerate = df.index.normalize().nunique() < 0.5 * len(df)
        except Exception:
            degenerate = True
        if degenerate:
            df = df.iloc[-n_keep:]
        else:
            df = df[df.index > df.index.max() - pd.Timedelta(days=n_keep)]

        q = pd.to_numeric(df[QCOL], errors="coerce")
        p = pd.to_numeric(df["prcp"], errors="coerce")
        m = q.notna() & p.notna() & (q >= 0)
        q, p = q[m].to_numpy(np.float32), p[m].to_numpy(np.float32)
        if len(q) >= CTX + HORIZON + 60:
            out[pf.stem] = (q, p)
    return out


def forecast_timesfm(basins):
    """Per-day TimesFM forecasts on the paper's protocol. Returns {bid: (idx, obs, sim, pers)}."""
    import timesfm
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(backend="gpu", per_core_batch_size=128,
                                       horizon_len=HORIZON, input_patch_len=32,
                                       output_patch_len=128),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
    )
    from tqdm import tqdm
    out = {}
    for bid, (q, p) in tqdm(basins.items(), desc="TimesFM"):
        starts = list(range(0, len(q) - CTX - HORIZON + 1, STRIDE))
        if len(starts) < 60:
            continue
        ctxs = [q[s:s + CTX] for s in starts]
        tgt_idx = np.array([s + CTX for s in starts])
        preds = []
        for i in range(0, len(ctxs), 128):
            b = ctxs[i:i + 128]
            f, _ = tfm.forecast(b, freq=[0] * len(b))
            preds.extend([x[0] for x in f])
        out[bid] = (tgt_idx, q[tgt_idx], np.array(preds, np.float32), q[tgt_idx - 1])
    return out


def nse(obs, sim):
    d = ((obs - obs.mean()) ** 2).sum()
    return 1 - ((obs - sim) ** 2).sum() / d if d > 0 else np.nan


def oracle_gain(resid, feats, seed=0):
    """Fit resid ~ rainfall on the FIRST half, score on the HELD-OUT second half.
    Returns fraction of held-out residual variance removed (<=0 means rainfall does not help)."""
    n = len(resid)
    h = n // 2
    Xtr, ytr = feats[:h], resid[:h]
    Xte, yte = feats[h:], resid[h:]
    if len(yte) < 20 or np.allclose(Xtr, 0):
        return np.nan
    X1 = np.column_stack([np.ones(len(Xtr)), Xtr])
    try:
        beta, *_ = np.linalg.lstsq(X1, ytr, rcond=None)
    except Exception:
        return np.nan
    pred = np.column_stack([np.ones(len(Xte)), Xte]) @ beta
    sse0 = (yte ** 2).sum()
    sse1 = ((yte - pred) ** 2).sum()
    return 1 - sse1 / sse0 if sse0 > 0 else np.nan


def analyse(ds):
    say("=" * 78)
    say(f"DATASET: {ds}")
    say("=" * 78)
    basins = load_with_precip(ds)
    say(f"  basins with aligned discharge+precip: {len(basins)}")
    if not basins:
        return pd.DataFrame()
    fc = forecast_timesfm(basins)
    say(f"  basins forecast: {len(fc)}")

    rows = []
    rng = np.random.default_rng(0)
    for bid, (idx, obs, sim, pers) in fc.items():
        q, p = basins[bid]
        P0 = p[idx]                       # rain on the forecast day itself
        P1 = p[np.maximum(idx - 1, 0)]    # yesterday
        P2 = p[np.maximum(idx - 2, 0)]
        r_tfm = obs - sim
        r_per = obs - pers
        feats = np.column_stack([P0, P1, P2])
        shuf = feats.copy()
        rng.shuffle(shuf)                 # negative control: same rain, wrong days

        wet = P0 > 1.0
        dry = P0 <= 0.1
        if wet.sum() < 15 or dry.sum() < 15:
            continue
        # scale-free: |residual| normalised by the basin's mean flow
        mu = obs.mean()
        if mu <= 0:
            continue
        rows.append({
            "dataset": ds, "basin_id": bid, "n": len(obs),
            "ac1": float(pd.Series(q).autocorr(1)),
            "nse_tfm": nse(obs, sim), "nse_pers": nse(obs, pers),
            "err_wet": float(np.abs(r_tfm[wet]).mean() / mu),
            "err_dry": float(np.abs(r_tfm[dry]).mean() / mu),
            "rho_absres_P": stats.spearmanr(np.abs(r_tfm), P0)[0],
            "oracle_gain_tfm": oracle_gain(r_tfm, feats),
            "oracle_gain_pers": oracle_gain(r_per, feats),
            "oracle_gain_shuffled": oracle_gain(r_tfm, shuf),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.concat([analyse("CAMELS-US"), analyse("CAMELS-AUS")], ignore_index=True)
    df["wet_dry_ratio"] = df.err_wet / df.err_dry
    df.to_csv(RES / "rainfall_ceiling_per_basin.csv", index=False)

    d = df.dropna(subset=["ac1", "oracle_gain_tfm"]).copy()
    qs = d.ac1.quantile([0.25, 0.5, 0.75]).values
    d["q"] = pd.cut(d.ac1, [-np.inf, *qs, np.inf],
                    labels=["Q1 flashy", "Q2", "Q3", "Q4 damped"])

    say()
    say("=" * 78)
    say("RESULT 1 (H-a): is TimesFM's error concentrated on rain days?")
    say("=" * 78)
    say(f"{'bin':12s} {'n':>4s} {'ac1':>6s} {'|err| wet':>10s} {'|err| dry':>10s} "
        f"{'wet/dry':>8s} {'rho(|r|,P)':>11s}")
    for b, g in d.groupby("q", observed=True):
        say(f"{str(b):12s} {len(g):4d} {g.ac1.median():6.3f} {g.err_wet.median():10.3f} "
            f"{g.err_dry.median():10.3f} {g.wet_dry_ratio.median():8.2f} "
            f"{g.rho_absres_P.median():11.3f}")

    say()
    say("=" * 78)
    say("RESULT 2 (H-b, H-c): ORACLE TEST -- how much residual can OBSERVED rain remove?")
    say("           (fit on first half of each basin, scored on held-out second half)")
    say("=" * 78)
    say(f"{'bin':12s} {'n':>4s} {'TimesFM resid':>14s} {'persistence resid':>18s} "
        f"{'SHUFFLED (null)':>16s}")
    for b, g in d.groupby("q", observed=True):
        say(f"{str(b):12s} {len(g):4d} {g.oracle_gain_tfm.median():13.1%} "
            f"{g.oracle_gain_pers.median():17.1%} {g.oracle_gain_shuffled.median():15.1%}")

    say()
    say("READ:")
    say("  * oracle gain > 0  -> the residual IS rainfall-shaped: rainfall is missing information.")
    say("  * gain rising as memory falls (Q4 -> Q1) -> the ceiling is rainfall-driven exactly")
    say("    where the paper claims it is.")
    say("  * the SHUFFLED column is the null. If it matches the real column, the gain is noise")
    say("    and the rainfall-ceiling claim is NOT supported.")

    say()
    r = stats.spearmanr(d.ac1, d.oracle_gain_tfm)
    say(f"Spearman(rho_1, oracle gain on TimesFM residual) = {r[0]:+.3f}  (p={r[1]:.2e}, n={len(d)})")
    say("  Negative => rainfall explains MORE of the residual in LOW-memory basins.")

    say()
    for ds, g in d.groupby("dataset"):
        rr = stats.spearmanr(g.ac1, g.oracle_gain_tfm)
        say(f"  {ds:12s} n={len(g):4d}  rho={rr[0]:+.3f}  p={rr[1]:.2e}  "
            f"median gain={g.oracle_gain_tfm.median():.1%}")

    (RES / "RAINFALL_CEILING_TEST.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] wrote {RES/'RAINFALL_CEILING_TEST.txt'}")
