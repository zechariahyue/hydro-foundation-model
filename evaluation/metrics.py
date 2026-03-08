#!/usr/bin/env python3
"""
Evaluation metrics for streamflow forecasting.
All metrics operate on numpy arrays of observed (y) and predicted (y_hat).
"""

import numpy as np
from scipy import stats


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency. Range (-inf, 1], 1 = perfect."""
    obs, sim = _clean(obs, sim)
    if len(obs) == 0:
        return np.nan
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((obs - sim) ** 2) / denom


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    """Kling-Gupta Efficiency. Range (-inf, 1], 1 = perfect."""
    obs, sim = _clean(obs, sim)
    if len(obs) < 2:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs) if np.std(obs) > 0 else np.nan
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) > 0 else np.nan
    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        return np.nan
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def kge_components(obs: np.ndarray, sim: np.ndarray) -> dict:
    """Return KGE and its three components."""
    obs, sim = _clean(obs, sim)
    if len(obs) < 2:
        return {"kge": np.nan, "r": np.nan, "alpha": np.nan, "beta": np.nan}
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs) if np.std(obs) > 0 else np.nan
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) > 0 else np.nan
    kge_val = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return {"kge": kge_val, "r": r, "alpha": alpha, "beta": beta}


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Root Mean Square Error."""
    obs, sim = _clean(obs, sim)
    if len(obs) == 0:
        return np.nan
    return np.sqrt(np.mean((obs - sim) ** 2))


def mae(obs: np.ndarray, sim: np.ndarray) -> float:
    """Mean Absolute Error."""
    obs, sim = _clean(obs, sim)
    if len(obs) == 0:
        return np.nan
    return np.mean(np.abs(obs - sim))


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    """Percent Bias (%). Positive = overestimation."""
    obs, sim = _clean(obs, sim)
    if len(obs) == 0 or np.sum(obs) == 0:
        return np.nan
    return 100.0 * np.sum(sim - obs) / np.sum(obs)


def crps_ensemble(obs: np.ndarray, ensemble: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score from ensemble predictions.
    obs: (T,) observed values
    ensemble: (T, N) ensemble members
    Lower is better.
    """
    T = len(obs)
    if T == 0:
        return np.nan
    scores = []
    for t in range(T):
        ens = np.sort(ensemble[t])
        N = len(ens)
        # CRPS = E|X-y| - 0.5*E|X-X'|
        term1 = np.mean(np.abs(ens - obs[t]))
        term2 = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                term2 += np.abs(ens[i] - ens[j])
        term2 = term2 / (N * (N - 1) / 2) if N > 1 else 0.0
        scores.append(term1 - 0.5 * term2)
    return np.mean(scores)


def crps_gaussian(obs: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """CRPS for Gaussian predictive distribution. Lower is better."""
    obs, mu = _clean(obs, mu)
    sigma = sigma[np.isfinite(obs) & np.isfinite(mu)]
    if len(obs) == 0:
        return np.nan
    z = (obs - mu) / (sigma + 1e-8)
    crps_vals = sigma * (z * (2 * stats.norm.cdf(z) - 1) +
                         2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return np.mean(crps_vals)


def ece(obs: np.ndarray, pred_lower: np.ndarray, pred_upper: np.ndarray,
        nominal_levels: np.ndarray = None) -> float:
    """
    Expected Calibration Error for prediction intervals.
    obs: (T,) observed
    pred_lower: (T, L) lower bounds at L confidence levels
    pred_upper: (T, L) upper bounds at L confidence levels
    nominal_levels: (L,) e.g. [0.1, 0.2, ..., 0.9]
    """
    if nominal_levels is None:
        nominal_levels = np.arange(0.1, 1.0, 0.1)

    T = len(obs)
    if T == 0:
        return np.nan

    errors = []
    for i, level in enumerate(nominal_levels):
        if pred_lower.ndim == 2:
            covered = np.mean((obs >= pred_lower[:, i]) & (obs <= pred_upper[:, i]))
        else:
            covered = np.mean((obs >= pred_lower) & (obs <= pred_upper))
        errors.append(np.abs(covered - level))

    return np.mean(errors)


def fdc_slope_error(obs: np.ndarray, sim: np.ndarray,
                    low_pct: float = 33, high_pct: float = 66) -> float:
    """
    Flow Duration Curve slope error.
    Compares the slope of the FDC between low_pct and high_pct exceedance.
    """
    obs, sim = _clean(obs, sim)
    if len(obs) < 10:
        return np.nan

    def fdc_slope(q):
        sorted_q = np.sort(q)[::-1]
        n = len(sorted_q)
        idx_low = int(n * low_pct / 100)
        idx_high = int(n * high_pct / 100)
        if idx_low == idx_high or sorted_q[idx_low] <= 0 or sorted_q[idx_high] <= 0:
            return np.nan
        return (np.log(sorted_q[idx_low]) - np.log(sorted_q[idx_high])) / (high_pct - low_pct)

    s_obs = fdc_slope(obs)
    s_sim = fdc_slope(sim)
    if np.isnan(s_obs) or np.isnan(s_sim):
        return np.nan
    return s_sim - s_obs


def rb_flashiness(q: np.ndarray) -> float:
    """Richards-Baker Flashiness Index."""
    q = q[np.isfinite(q)]
    if len(q) < 2 or np.sum(q) == 0:
        return np.nan
    return np.sum(np.abs(np.diff(q))) / np.sum(q)


def fdc_kge(obs: np.ndarray, sim: np.ndarray) -> float:
    """KGE computed on sorted flow duration curves."""
    obs, sim = _clean(obs, sim)
    if len(obs) < 10:
        return np.nan
    obs_sorted = np.sort(obs)[::-1]
    sim_sorted = np.sort(sim)[::-1]
    return kge(obs_sorted, sim_sorted)


def tail_rmse(obs: np.ndarray, sim: np.ndarray, percentile: float = 95) -> float:
    """RMSE for flows above the given percentile (extreme events)."""
    obs, sim = _clean(obs, sim)
    if len(obs) == 0:
        return np.nan
    threshold = np.percentile(obs, percentile)
    mask = obs >= threshold
    if np.sum(mask) == 0:
        return np.nan
    return rmse(obs[mask], sim[mask])


def water_balance_error(obs: np.ndarray, precip: np.ndarray, et: np.ndarray) -> float:
    """
    Annual water balance closure error.
    |sum(Q_pred) - (sum(P) - sum(ET))| / (sum(P) - sum(ET))
    """
    obs = obs[np.isfinite(obs)]
    precip = precip[np.isfinite(precip)]
    et = et[np.isfinite(et)]
    n = min(len(obs), len(precip), len(et))
    if n == 0:
        return np.nan
    p_minus_et = np.sum(precip[:n]) - np.sum(et[:n])
    if p_minus_et <= 0:
        return np.nan
    return np.abs(np.sum(obs[:n]) - p_minus_et) / p_minus_et


def compute_all_metrics(obs: np.ndarray, sim: np.ndarray) -> dict:
    """Compute all deterministic metrics."""
    return {
        "NSE": nse(obs, sim),
        "KGE": kge(obs, sim),
        "RMSE": rmse(obs, sim),
        "MAE": mae(obs, sim),
        "PBIAS": pbias(obs, sim),
        "FDC_slope_error": fdc_slope_error(obs, sim),
        "FDC_KGE": fdc_kge(obs, sim),
        "tail_RMSE_95": tail_rmse(obs, sim, 95),
        "tail_RMSE_99": tail_rmse(obs, sim, 99),
        "RB_flashiness_obs": rb_flashiness(obs),
        "RB_flashiness_sim": rb_flashiness(sim),
    }


def bootstrap_ci(obs, sim, metric_fn, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(seed)
    n = len(obs)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        scores.append(metric_fn(obs[idx], sim[idx]))
    scores = np.array(scores)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return np.nan, np.nan, np.nan
    alpha = (1 - ci) / 2
    return np.mean(scores), np.percentile(scores, 100 * alpha), np.percentile(scores, 100 * (1 - alpha))


def _clean(obs, sim):
    """Remove NaN/Inf from paired arrays."""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    return obs[mask], sim[mask]
