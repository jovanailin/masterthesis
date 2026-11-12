"""Classical benchmark forecasts for intermittent demand (thesis Section 6.4).

Every method forecasts the 28 days after the forecast origin from the history up to the origin, for all series
at once. The history is that of the panel (`features.Panel.y`): it starts at the series' first sale and
Christmas Day is missing; the recursive methods skip missing days.

- naive: the last observed value;
- snaive: the last observed week, repeated;
- mean: the mean over the active period;
- croston: SES of the non-zero demand sizes divided by SES of the intervals between them (Croston, 1972);
- sba: Croston multiplied by 1 - alpha / 2 to remove its bias (Syntetos & Boylan, 2005);
- tsb: SES of the demand size times SES of the probability of a demand, updated every day, so that the
  forecast decays during long runs of zeros (Teunter, Syntetos & Babai, 2011);
- adida: SES on the sums over non-overlapping buckets of ADI days, spread evenly over the days of a bucket
  (Nikolopoulos et al., 2011).

All smoothing constants are fixed at 0.1, within the 0.05-0.2 range recommended for intermittent demand.
"""
import numpy as np

ALPHA = 0.1  # smoothing of demand sizes and intervals (Croston, SBA, TSB, ADIDA)
BETA = 0.1  # smoothing of the demand probability (TSB)
HORIZON = 28
METHODS = ["naive", "snaive", "mean", "croston", "sba", "tsb", "adida"]


def _flat(level: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(level[:, None], horizon, axis=1).astype(np.float32)


def naive(y: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    last = y.shape[1] - 1 - np.argmax(~np.isnan(y[:, ::-1]), axis=1)
    return _flat(y[np.arange(len(y)), last], horizon)


def snaive(y: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    week = y[:, -7:]
    week = np.where(np.isnan(week), naive(y, 7), week)
    return np.tile(week, (1, -(-horizon // 7)))[:, :horizon].astype(np.float32)


def mean(y: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    return _flat(np.nanmean(y, axis=1), horizon)


def _croston_states(y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed demand size z and smoothed interval p (observed days between demands) of each series."""
    n, T = y.shape
    z = np.full(n, np.nan)
    p = np.full(n, np.nan)
    q = np.zeros(n)  # observed days since the last demand
    seen = np.zeros(n, dtype=bool)
    for t in range(T):
        v = y[:, t]
        observed = ~np.isnan(v)
        demand = observed & (v > 0)
        q[observed & seen] += 1
        first = demand & ~seen
        z[first] = v[first]
        later = demand & seen
        second = later & np.isnan(p)
        p[second] = q[second]
        rest = later & ~second
        p[rest] += alpha * (q[rest] - p[rest])
        z[later] += alpha * (v[later] - z[later])
        q[demand] = 0
        seen |= first
    single = np.isnan(p)  # one demand only: its interval is at least the days observed since
    p[single] = q[single] + 1
    return z, p


def croston(y: np.ndarray, horizon: int = HORIZON, alpha: float = ALPHA) -> np.ndarray:
    z, p = _croston_states(y, alpha)
    return _flat(z / p, horizon)


def sba(y: np.ndarray, horizon: int = HORIZON, alpha: float = ALPHA) -> np.ndarray:
    z, p = _croston_states(y, alpha)
    return _flat((1 - alpha / 2) * z / p, horizon)


def tsb(y: np.ndarray, horizon: int = HORIZON, alpha: float = ALPHA, beta: float = BETA) -> np.ndarray:
    n, T = y.shape
    z = np.full(n, np.nan)
    prob = np.full(n, np.nan)
    for t in range(T):
        v = y[:, t]
        observed = ~np.isnan(v)
        demand = observed & (v > 0)
        first = demand & np.isnan(z)
        z[first], prob[first] = v[first], 1.0
        update = observed & ~np.isnan(z) & ~first
        prob[update] += beta * (demand[update] - prob[update])
        size = update & demand
        z[size] += alpha * (v[size] - z[size])
    return _flat(prob * z, horizon)


def adida(y: np.ndarray, horizon: int = HORIZON, alpha: float = ALPHA) -> np.ndarray:
    """Buckets of k = round(ADI) days ending at the origin; SES on the bucket sums; forecast = level / k."""
    n, T = y.shape
    observed = ~np.isnan(y)
    adi = observed.sum(axis=1) / (np.nan_to_num(y) > 0).sum(axis=1)
    k = np.maximum(1, np.rint(adi)).astype(int)
    first = np.argmax(observed, axis=1)
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(np.nan_to_num(y), axis=1)], axis=1)
    level = np.full(n, np.nan)
    for kk in np.unique(k):
        rows = np.flatnonzero(k == kk)
        n_buckets = (T - first[rows]) // kk
        n_max = n_buckets.max()
        ends = T - kk * np.arange(n_max)[::-1]  # chronological bucket ends (exclusive cumsum index)
        sums = cs[rows][:, ends] - cs[rows][:, np.maximum(ends - kk, 0)]
        start = n_max - n_buckets  # first complete bucket of each row
        ok = n_buckets > 0
        lv = np.where(ok, sums[np.arange(rows.size), np.minimum(start, n_max - 1)], np.nan)
        for j in range(n_max):
            upd = ok & (j > start)
            lv[upd] += alpha * (sums[upd, j] - lv[upd])
        level[rows] = lv / kk
    fallback = np.isnan(level)
    level[fallback] = np.nanmean(y[fallback], axis=1)
    return _flat(level, horizon)


def forecast_all(y: np.ndarray, horizon: int = HORIZON) -> dict[str, np.ndarray]:
    """Forecasts of every method from the history y (n_series, n_days) up to the forecast origin."""
    funcs = {"naive": naive, "snaive": snaive, "mean": mean, "croston": croston, "sba": sba, "tsb": tsb, "adida": adida}
    return {m: funcs[m](y, horizon) for m in METHODS}
