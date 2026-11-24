"""Classical forecasting methods on weekly sales (additional experiment, compared with the weekly global XGBoost).

Every method is local: it is fitted to each series separately, on that series' weekly sales only (no prices,
events or other series). The protocol is that of weekly_xgb.py: each week of a window is forecast one week
ahead from the Sunday before it. The parameters of ses, ets and arima are estimated on the history up to the
first origin of the window (24.04.2016 for validation, 22.05.2016 for the test) and kept for the 3 later
origins, where the model only takes in the new weeks (the forward method of statsforecast), as the weekly
XGBoost is trained once per window. Refitting at every origin changes these forecasts only in the third
decimal and costs four times as much (AutoARIMA takes about half a second per series). The history of a
series starts at its first whole week of sales (weekly.build_panel), so the classical methods and the weekly
XGBoost see the same weekly sales.

- naive: the sales of the last week;
- mean: the mean over the whole history;
- ma_k: the mean of the last k weeks, k in MA_WINDOWS; the window is chosen on the validation window;
- ses: simple exponential smoothing, the smoothing constant and the initial level estimated per series by
  maximum likelihood (statsforecast AutoETS restricted to the model ANN);
- ets: the exponential smoothing model (additive or multiplicative error, none, additive or damped trend)
  chosen per series by AICc (statsforecast AutoETS), without seasonality: a season of 52 weeks is too long
  for ETS;
- arima: ARIMA chosen per series by the stepwise search of Hyndman & Khandakar (2008) (statsforecast
  AutoARIMA), without seasonality.

A series with fewer than MIN_WEEKS weeks of history at the first origin of a window gets the mean of its weeks
so far from ses, ets and arima as well.

Run:  python src/weekly_classical.py            (CPU, all series; writes data/processed/weekly_classical/)
      python src/weekly_classical.py --backtest (three earlier 4-week windows, forecasts_backtest.npz)
      python src/weekly_classical.py --step 100 (quick check on every 100th series)
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

import evaluation as E
import m5_data as m5
import weekly as W

OUT = m5.PROCESSED_DIR / "weekly_classical"
MA_WINDOWS = (4, 8, 13, 26, 52)
MIN_WEEKS = 8
SIMPLE = ["naive", "mean", *(f"ma_{k}" for k in MA_WINDOWS)]
STATISTICAL = ["ses", "ets", "arima"]


def simple_forecasts(y: np.ndarray, origin: int) -> dict[str, np.ndarray]:
    """One-week-ahead forecasts of the simple methods from the weekly history y[:, :origin + 1] (NaN before the
    first week of a series); 0 for a series without history."""
    hist = y[:, : origin + 1]
    observed = ~np.isnan(hist)
    last = origin - np.argmax(observed[:, ::-1], axis=1)
    out = {"naive": hist[np.arange(len(hist)), last]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # mean of an empty slice: series without history
        out["mean"] = np.nanmean(hist, axis=1)
        for k in MA_WINDOWS:
            out[f"ma_{k}"] = np.nanmean(hist[:, -k:], axis=1)
    return {m: np.nan_to_num(f).astype(np.float32) for m, f in out.items()}


def statistical_forecasts(y: np.ndarray, first_week: np.ndarray, T: int, n_jobs: int) -> dict[str, np.ndarray]:
    """One-week-ahead forecasts (n_series, 4) of ses, ets and arima from the origins T, ..., T + 3, fitted at T
    (statsforecast cross-validation with h = 1, 4 windows, refit=False)."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS

    n = len(y)
    long_enough = np.flatnonzero(T - first_week.astype(np.int64) + 1 >= MIN_WEEKS)
    frames = [pd.DataFrame({"unique_id": i, "ds": np.arange(first_week[i], T + W.EVAL_WEEKS + 1),
                            "y": y[i, first_week[i] : T + W.EVAL_WEEKS + 1].astype(np.float64)}) for i in long_enough]
    df = pd.concat(frames, ignore_index=True)
    assert not df["y"].isna().any(), "the weekly sales are complete after the first week of a series"
    sf = StatsForecast(models=[AutoETS(model="ANN", season_length=1, alias="ses"), AutoETS(season_length=1, alias="ets"),
                               AutoARIMA(season_length=1, alias="arima")], freq=1, n_jobs=n_jobs)
    cv = sf.cross_validation(df=df, h=1, n_windows=W.EVAL_WEEKS, step_size=1, refit=False)
    out = {}
    for m in STATISTICAL:
        f = np.full((n, W.EVAL_WEEKS), np.nan, dtype=np.float32)
        f[cv["unique_id"].to_numpy(np.int64), cv["cutoff"].to_numpy(np.int64) - T] = cv[m].to_numpy(np.float32)
        out[m] = f
    short = np.setdiff1d(np.arange(n), long_enough)
    for k in range(W.EVAL_WEEKS):  # short histories: mean of the weeks so far
        fallback = simple_forecasts(y[short], T + k)["mean"]
        for m in STATISTICAL:
            out[m][short, k] = fallback
    for m in STATISTICAL:
        assert not np.isnan(out[m]).any(), f"{m}: missing forecasts"
        out[m] = np.maximum(out[m], 0)
    return out


def forecast_stage(y: np.ndarray, first_week: np.ndarray, stage: str, n_jobs: int) -> dict[str, np.ndarray]:
    T = W.STAGES[stage]
    parts = [simple_forecasts(y, T + k) for k in range(W.EVAL_WEEKS)]
    out = {m: np.stack([p[m] for p in parts], axis=1) for m in SIMPLE}
    start = time.perf_counter()
    out |= statistical_forecasts(y, first_week, T, n_jobs)
    print(json.dumps({"stage": stage, "series": len(y), "minutes": round((time.perf_counter() - start) / 60, 1)}),
          flush=True)
    return out


BACKTEST_WEEKS = (269, 265, 261)  # origins 27.03., 28.02. and 31.01.2016, the windows of weekly_xgb.run_extra


def main(step: int | None = None, n_jobs: int = os.cpu_count(), backtest: bool = False) -> None:
    panel = W.build_panel(E.test_panel())  # weekly sales up to 19.06.2016; each stage uses those up to its window
    y, first_week = panel.y, panel.first_week
    rows = np.arange(0, len(y), step) if step else None
    if rows is not None:
        y, first_week = y[rows], first_week[rows]
    del panel
    stages = [W.add_stage(w) for w in BACKTEST_WEEKS] if backtest else ["valid", "test"]
    out = OUT if step is None else OUT / f"step_{step}"
    out.mkdir(parents=True, exist_ok=True)
    file = out / ("forecasts_backtest.npz" if backtest else "forecasts.npz")
    result = {}
    for stage in stages:
        result |= {f"{stage}_{m}": f for m, f in forecast_stage(y, first_week, stage, n_jobs).items()}
        np.savez_compressed(file, rows=np.arange(len(y)) if rows is None else rows, **result)
    print("written", file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=None, help="every step-th series only (quick check)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--backtest", action="store_true", help="the three earlier windows instead of validation and test")
    args = parser.parse_args()
    main(args.step, args.jobs, args.backtest)
