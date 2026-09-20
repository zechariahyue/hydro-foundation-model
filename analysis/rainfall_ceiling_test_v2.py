#!/usr/bin/env python3
"""
RAINFALL-CEILING TEST, v2 -- giving the rainfall hypothesis its strongest fair shot.

v1 used a LINEAR oracle (residual ~ P_t + P_{t-1} + P_{t-2}) at stride 7 and found the
oracle gain peaked in MID-memory basins rather than in the flashiest ones. That is evidence
against the paper's claim ONLY if the oracle is a fair instrument. It is not:

  * rainfall-runoff response in flashy catchments is threshold-driven and strongly
    NONLINEAR, so a linear fit is weakest exactly where the hypothesis needs it to be
    strongest -- the test was rigged against the hypothesis by construction;
  * stride 7 leaves ~188 points per basin, so a half-split gives ~94 training rows, which
    is too few to fit any richer model.

v2 removes both handicaps:
  * STRIDE 1 -> ~1,300 points per basin (a superset of the stride-7 scored points; identical
    model, context, window and protocol otherwise), so ~650 training rows.
  * A NONLINEAR oracle (gradient-boosted trees) with antecedent-wetness features, alongside
    the linear one for comparison.

If the gain STILL does not rise toward flashy basins under a fair oracle, the paper's
rainfall-ceiling claim is not supported by its own data and must be rewritten.

Guards retained: fit on the first half, score ONLY on the held-out second half; a SHUFFLED
rainfall negative control; the same correction applied to the persistence residual.

Outputs: experiments/results/RAINFALL_CEILING_TEST_V2.txt
         experiments/results/rainfall_ceiling_v2_per_basin.csv
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CAMELS_DATA_DIR", "data/processed"))
RES = Path(os.environ.get("RESULTS_DIR", "results"))
QCOL = "QObs(mm/d)"
CTX, HORIZON, STRIDE, LAST_YEARS = 512, 1, 1, 5   # STRIDE 1 -- see docstring

lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def load_with_precip(ds):
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
        df = df.iloc[-n_keep:] if degenerate else df[df.index > df.index.max() - pd.Timedelta(days=n_keep)]
        q = pd.to_numeric(df[QCOL], errors="coerce")
        p = pd.to_numeric(df["prcp"], errors="coerce")
        m = q.notna() & p.notna() & (q >= 0)
        q, p = q[m].to_numpy(np.float32), p[m].to_numpy(np.float32)
        if len(q) >= CTX + 200:
            out[pf.stem] = (q, p)
    return out


def forecast(basins):
    import timesfm
    from tqdm import tqdm
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(backend="gpu", per_core_batch_size=128,
                                       horizon_len=HORIZON, input_patch_len=32,
                                       output_patch_len=128),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
    )
    out = {}
    for bid, (q, p) in tqdm(basins.items(), desc="TimesFM s1"):
        starts = list(range(0, len(q) - CTX - HORIZON + 1, STRIDE))
        if len(starts) < 300:
            continue
        ctxs = [q[s:s + CTX] for s in starts]
        idx = np.array([s + CTX for s in starts])
        preds = []
        for i in range(0, len(ctxs), 256):
            b = ctxs[i:i + 256]
            f, _ = tfm.forecast(b, freq=[0] * len(b))
            preds.extend([x[0] for x in f])
        out[bid] = (idx, q[idx], np.array(preds, np.float32), q[idx - 1])
    return out


def rain_features(p, idx):
    """Rainfall-only predictors: same-day, recent lags, and antecedent wetness."""
    def at(k):
        return p[np.maximum(idx - k, 0)]

    def win(a, b):
        return np.array([p[max(0, i - b):max(1, i - a + 1)].sum() for i in idx], np.float32)

    return np.column_stack([
        at(0), at(1), at(2), at(3),
        np.sqrt(at(0)), np.sqrt(at(1)),
        win(0, 6),      # 7-day antecedent
        win(7, 29),     # 8-30 day antecedent wetness
    ])


def gain(resid, X, nonlinear):
    """Fraction of HELD-OUT residual variance removed by a rainfall-only correction."""
    n = len(resid)
    h = n // 2
    Xtr, ytr, Xte, yte = X[:h], resid[:h], X[h:], resid[h:]
    if len(yte) < 50 or np.allclose(Xtr, 0):
        return np.nan
    try:
        if nonlinear:
            m = HistGradientBoostingRegressor(max_iter=200, max_depth=4,
                                              learning_rate=0.06, random_state=0)
            m.fit(Xtr, ytr)
            pred = m.predict(Xte)
        else:
            A = np.column_stack([np.ones(len(Xtr)), Xtr])
            beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
            pred = np.column_stack([np.ones(len(Xte)), Xte]) @ beta
    except Exception:
        return np.nan
    sse0 = (yte ** 2).sum()
    return 1 - ((yte - pred) ** 2).sum() / sse0 if sse0 > 0 else np.nan


def nse(o, s):
    d = ((o - o.mean()) ** 2).sum()
    return 1 - ((o - s) ** 2).sum() / d if d > 0 else np.nan


def analyse(ds):
    say(f"--- {ds} ---")
    basins = load_with_precip(ds)
    say(f"  basins: {len(basins)}")
    fc = forecast(basins)
    say(f"  forecast: {len(fc)}")
    rng = np.random.default_rng(0)
    rows = []
    for bid, (idx, obs, sim, pers) in fc.items():
        q, p = basins[bid]
        X = rain_features(p, idx)
        Xs = X.copy()
        rng.shuffle(Xs)
        r_t, r_p = obs - sim, obs - pers
        mu = obs.mean()
        if mu <= 0:
            continue
        P0 = X[:, 0]
        wet, dry = P0 > 1.0, P0 <= 0.1
        if wet.sum() < 40 or dry.sum() < 40:
            continue
        rows.append({
            "dataset": ds, "basin_id": bid, "n": len(obs),
            "ac1": float(pd.Series(q).autocorr(1)),
            "nse_tfm": nse(obs, sim), "nse_pers": nse(obs, pers),
            "err_wet": float(np.abs(r_t[wet]).mean() / mu),
            "err_dry": float(np.abs(r_t[dry]).mean() / mu),
            "gain_lin": gain(r_t, X, False),
            "gain_gbt": gain(r_t, X, True),
            "gain_gbt_pers": gain(r_p, X, True),
            # Negative controls. There must be ONE PER ESTIMATOR: comparing the linear
            # oracle's gain against a gradient-boosted shuffled null is not like-for-like,
            # because the two estimators have different capacities to overfit noise.
            "gain_lin_shuf": gain(r_t, Xs, False),
            "gain_gbt_shuf": gain(r_t, Xs, True),
            # Rank association between |residual| and same-day precipitation, on the same
            # v2 sample. v1 reported this on its own (stride-7, n=1,215) file; recomputing
            # it here keeps every number in the rainfall paragraph on one sample.
            "rho_absres_P": float(stats.spearmanr(np.abs(r_t), P0).statistic),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.concat([analyse("CAMELS-US"), analyse("CAMELS-AUS")], ignore_index=True)
    df["wet_dry"] = df.err_wet / df.err_dry
    df.to_csv(RES / "rainfall_ceiling_v2_per_basin.csv", index=False)

    d = df.dropna(subset=["ac1", "gain_gbt"]).copy()
    qs = d.ac1.quantile([0.25, 0.5, 0.75]).values
    d["q"] = pd.cut(d.ac1, [-np.inf, *qs, np.inf],
                    labels=["Q1 flashy", "Q2", "Q3", "Q4 damped"])

    say()
    say("=" * 86)
    say("ORACLE TEST v2 (stride 1, nonlinear oracle) -- median fraction of TimesFM's")
    say("held-out residual variance removed by a RAINFALL-ONLY correction")
    say("=" * 86)
    say("Each oracle is shown beside ITS OWN shuffled-rainfall null (like-for-like).")
    say(f"{'bin':12s} {'n':>4s} {'ac1':>6s} {'wet/dry':>8s} {'LINEAR':>8s} {'lin.SHUF':>9s} "
        f"{'NONLINEAR':>10s} {'gbt.SHUF':>9s} {'on pers.':>9s}")
    for b, g in d.groupby("q", observed=True):
        say(f"{str(b):12s} {len(g):4d} {g.ac1.median():6.3f} {g.wet_dry.median():8.2f} "
            f"{g.gain_lin.median():7.1%} {g.gain_lin_shuf.median():8.1%} "
            f"{g.gain_gbt.median():9.1%} {g.gain_gbt_shuf.median():8.1%} "
            f"{g.gain_gbt_pers.median():8.1%}")

    say()
    say(f"Rank association |residual| vs same-day precipitation (v2 sample, n={len(df)}):")
    say(f"  positive in {100*(df.rho_absres_P > 0).mean():.1f}% of basins; "
        f"median rho = {df.rho_absres_P.median():+.3f}")

    say()
    for name, col in [("LINEAR", "gain_lin"), ("NONLINEAR", "gain_gbt")]:
        r = stats.spearmanr(d.ac1, d[col], nan_policy="omit")
        say(f"Spearman(rho_1, {name:9s} oracle gain) = {r[0]:+.3f}  p={r[1]:.2e}")
    say("  NEGATIVE => rainfall explains more of the residual in LOW-memory basins")
    say("              (this is what the paper's rainfall-ceiling claim REQUIRES).")

    say()
    for ds, g in d.groupby("dataset"):
        r = stats.spearmanr(g.ac1, g.gain_gbt, nan_policy="omit")
        say(f"  {ds:12s} n={len(g):4d} rho={r[0]:+.3f} p={r[1]:.2e} "
            f"median nonlinear gain={g.gain_gbt.median():.1%}")

    (RES / "RAINFALL_CEILING_TEST_V2.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] wrote {RES/'RAINFALL_CEILING_TEST_V2.txt'}")
