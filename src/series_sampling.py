"""Selection of the intermittent series sample used in the replication study.

Four-stage filter of Tsantilis et al. (2025), applied to each series' active period
(first sale to the end of the selection window):

1. Augmented Dickey-Fuller test: keep stationary series (unit root rejected).
2. Ljung-Box white-noise screen at lags 13, 26 and 52: drop series for which no
   lag shows significant autocorrelation (nothing to learn beyond the mean).
3. Syntetos-Boylan-Croston classification (ADI, CV2).
4. Keep the intermittent quadrant only: ADI > 1.32 and CV2 <= 0.49.

The sample is then drawn stratified over regime cells with a largest-remainder
(Hamilton) allocation, so that it has exactly the requested size.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

LB_LAGS = (13, 26, 52)


def _screen_one(y: np.ndarray) -> tuple[float, ...]:
    adf_p = adfuller(y, regression="c", autolag="AIC")[1]
    lb_p = acorr_ljungbox(y, lags=list(LB_LAGS))["lb_pvalue"].to_numpy()
    return (adf_p, *lb_p)


def screen_series(y: np.ndarray, first_sale: np.ndarray, n_jobs: int = -1) -> pd.DataFrame:
    """ADF and Ljung-Box p-values of each row of y over its active period.

    `first_sale` is the 1-based day of the first non-zero value (NaN if none).
    """
    rows = Parallel(n_jobs=n_jobs, batch_size=64)(
        delayed(_screen_one)(row[int(fs) - 1 :].astype(float)) for row, fs in zip(y, first_sale)
    )
    return pd.DataFrame(rows, columns=["adf_p", *[f"lb_p{lag}" for lag in LB_LAGS]])


def hamilton_allocation(counts: pd.Series, n: int) -> pd.Series:
    """Largest-remainder apportionment of n seats proportional to `counts`."""
    quota = counts / counts.sum() * n
    seats = np.floor(quota).astype(int)
    remainder = (quota - seats).sort_values(ascending=False, kind="stable")
    seats[remainder.index[: n - seats.sum()]] += 1
    return seats


def stratified_sample(cells: pd.Series, n: int, seed: int) -> pd.Index:
    """Index labels of a stratified sample of size n, allocated over `cells` by Hamilton's rule."""
    counts = cells.value_counts().sort_index()
    allocation = hamilton_allocation(counts, n)
    rng = np.random.default_rng(seed)
    picked = [
        rng.choice(cells.index[cells == cell].to_numpy(), size=k, replace=False)
        for cell, k in allocation.items()
        if k > 0
    ]
    return pd.Index(np.sort(np.concatenate(picked)))
