"""Global XGBoost on weekly sales (additional experiment; the inputs are in weekly.py).

The procedure is that of global_xgb.py with weeks instead of days. Hyperopt (TPE) tunes each variant (input
set, loss) on the validation window 25.04.-22.05.2016 (weeks 274-277): each trial trains on the weeks up to
24.04.2016 with early stopping on the RMSE of the validation forecasts and is scored by their WRMSSE on
weekly sums (7 levels, weekly.evaluator), which Hyperopt minimises. The best trial is refitted on the weeks
up to 22.05.2016; this model forecasts the test window 23.05.-19.06.2016 in the final evaluation
(test_forecast), after the test sales are opened.

Main design (horizon 1): each week is forecast one week ahead from the Sunday before it, the forecast origins
of the daily models, so that the forecasts can be compared with the weekly sums of the daily forecasts.
With horizon 4 the 4 weeks of a window are forecast from its first Sunday only.

Kaggle GPU kernels (kaggle_jobs/weekly_xgb/) write per variant trials.json, best.json, valid_forecast.npy
(series x 4 weeks), model_valid.ubj and model_test.ubj.
"""
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import xgboost as xgb

import global_xgb as G
import weekly as W


def training_matrix(panel: W.Panel, table: W.OriginTable, groups, stage: str, seed: int = 0, *, horizon: int = 1,
                    per_origin: int | None = None) -> xgb.QuantileDMatrix:
    rows, h = W.horizon_rows(len(table), horizon, per_origin, seed)
    X, y, _ = W.design(panel, table, h, groups, stage=stage, rows=rows)
    return xgb.QuantileDMatrix(X, y, max_bin=G.MAX_BIN, enable_categorical=True)


class ForecastRows:
    """Rows of the forecasts of a stage's 4 weeks for every series: 4 origins at h = 1 (horizon 1), or the
    first origin at h = 1..4 (horizon 4)."""

    def __init__(self, panel: W.Panel, stage: str, horizon: int = 1):
        assert horizon in (1, W.EVAL_WEEKS), "the 4 weeks are forecast from 4 origins or from the first one"
        s, t = W.forecast_origins(panel, stage, horizon)
        table = W.origin_table(panel, s, t)
        rows, h = W.horizon_rows(len(table), horizon)
        self.n, self.horizon = len(panel.ids), horizon
        self.series = s[rows]
        self.X, _, _ = W.design(panel, table, h, tuple(W.GROUPS), target=False, rows=rows)
        week = t[rows].astype(np.int64) + h
        self.y = panel.y[s[rows], week] if week.max() < panel.y.shape[1] else None

    def inputs(self, groups):
        return self.X[W.columns(groups)]

    def matrix(self, pred: np.ndarray) -> np.ndarray:
        """Forecasts as (n_series, 4 weeks), negative values set to 0. Horizon 1: rows ordered by origin and
        series; horizon 4: by series and horizon."""
        f = pred.reshape(W.EVAL_WEEKS, self.n).T if self.horizon == 1 else pred.reshape(self.n, W.EVAL_WEEKS)
        return np.maximum(f, 0).astype(np.float32)


def run(job: dict, work: Path) -> None:
    """Tune and refit the variants of a job: {"variants": [[inputs, loss], ...], "budget_hours": ...,
    "horizon": 1 or 4, "per_origin": horizons drawn per origin (default: all)}."""
    start = time.perf_counter()
    device = job.get("device", "cuda")
    horizon = job.get("horizon", 1)
    make_train = partial(training_matrix, horizon=horizon, per_origin=job.get("per_origin"))
    variants = job["variants"]
    deadline = start + job["budget_hours"] * 3600 - G.REFIT_RESERVE * len(variants)

    panel, rows = W.build_panel(), None
    if job.get("series_step"):  # quick check of the code on a subset of the series
        rows = np.arange(0, len(panel.ids), job["series_step"])
        panel = panel.subset(rows)
    ev = W.evaluator("valid", rows=rows)
    val = ForecastRows(panel, "valid", horizon)
    G._log(step="data", series=len(panel.ids), validation_rows=len(val.y), horizon=horizon,
           seconds=round(time.perf_counter() - start, 1))

    table = W.origin_table(panel, *W.training_origins(panel, "valid"))
    summary = []
    for k, (inputs, loss) in enumerate(variants):
        out = work / G.variant_name(inputs, loss)
        best = G.tune(panel, table, val, ev, inputs, loss, out, max_evals=job.get("max_evals", G.MAX_EVALS),
                      timeout=(deadline - time.perf_counter()) / (len(variants) - k), device=device,
                      max_rounds=job.get("max_rounds", G.MAX_ROUNDS), patience=job.get("patience", G.PATIENCE),
                      make_train=make_train)
        best |= {"level": "weekly", "horizon": horizon, "per_origin": job.get("per_origin")}
        G._write(out / "best.json", best)
        summary.append({k: best.get(k) for k in ["variant", "trials", "wrmsse", "mean_rmsse", "valid_rmse", "rounds"]})
        G._write(work / "summary.json", summary)
    del table

    table = W.origin_table(panel, *W.training_origins(panel, "test"))
    for inputs, loss in variants:
        G.refit(panel, table, work / G.variant_name(inputs, loss), device, make_train=make_train)
    G._log(step="done", hours=round((time.perf_counter() - start) / 3600, 2))


def _train_forecast(panel, table, val: ForecastRows, best: dict, stage: str, device: str, seed: int) -> np.ndarray:
    """Train a tuned variant on the targets of a stage with its hyperparameters and number of rounds, and
    forecast the stage's 4 weeks."""
    groups = G.INPUT_SETS[best["inputs"]]
    dtrain = training_matrix(panel, table, groups, stage, seed, horizon=best.get("horizon", 1),
                             per_origin=best.get("per_origin"))
    bst = xgb.train(best["params"] | {"device": device, "seed": seed}, dtrain, best["rounds"])
    dval = xgb.QuantileDMatrix(val.inputs(groups), ref=dtrain, enable_categorical=True)
    return val.matrix(bst.predict(dval))


def run_extra(job: dict, work: Path) -> None:
    """Extra runs of tuned variants, nothing tuned again: {"model_dirs": {variant: folder of its best.json},
    "seeds": [...], "origins": [weeks]}.

    Seeds: each variant is trained with every seed on the validation stage (valid_forecast_seed<s>.npy, scores
    in seeds.json) and on the test stage (model_test_seed<s>.ubj). Origins: each variant is trained on the
    weeks up to each origin week and forecasts the 4 weeks after it (w<week>/<variant>.npy), as a check that
    the results do not depend on one test window; the hyperparameters were chosen with later data than these
    windows, the forecasts use only data up to their origin."""
    start = time.perf_counter()
    device = job.get("device", "cuda")
    panel, rows = W.build_panel(), None
    if job.get("series_step"):
        rows = np.arange(0, len(panel.ids), job["series_step"])
        panel = panel.subset(rows)
    bests = {}
    for name, folder in job["model_dirs"].items():
        bests[name] = json.loads((Path(folder) / "best.json").read_text())
        assert bests[name].get("horizon", 1) == 1, "the extra runs forecast one week ahead"
        (work / name).mkdir(parents=True, exist_ok=True)
        G._write(work / name / "best.json", bests[name])

    seeds = job.get("seeds", [])
    if seeds:
        ev, val = W.evaluator("valid", rows=rows), ForecastRows(panel, "valid")
        table = W.origin_table(panel, *W.training_origins(panel, "valid"))
        for name, best in bests.items():
            records = []
            for seed in seeds:
                t = time.perf_counter()
                forecast = _train_forecast(panel, table, val, best, "valid", device, seed)
                np.save(work / name / f"valid_forecast_seed{seed}.npy", forecast)
                score = ev.wrmsse(forecast)
                records.append({"variant": name, "seed": seed, "wrmsse": score["all"],
                                "mean_rmsse": float(ev.bottom(forecast)["rmsse"].mean()),
                                "wrmsse_by_level": {str(k): v for k, v in score.items()},
                                "seconds": round(time.perf_counter() - t, 1)})
                G._write(work / name / "seeds.json", records)
                G._log(step="seed", **{k: v for k, v in records[-1].items() if k != "wrmsse_by_level"})
        del table, ev, val
        table = W.origin_table(panel, *W.training_origins(panel, "test"))
        for name, best in bests.items():
            make_train = partial(training_matrix, horizon=1, per_origin=best.get("per_origin"))
            for seed in seeds:
                G.refit(panel, table, work / name, device, seed, make_train=make_train)
        del table

    for week in job.get("origins", []):
        stage = W.add_stage(week)
        (work / stage).mkdir(parents=True, exist_ok=True)
        val = ForecastRows(panel, stage)
        table = W.origin_table(panel, *W.training_origins(panel, stage))
        for name, best in bests.items():
            t = time.perf_counter()
            np.save(work / stage / f"{name}.npy", _train_forecast(panel, table, val, best, stage, device, best["seed"]))
            G._log(step="backtest", origin=week, model=name, seconds=round(time.perf_counter() - t, 1))
        del table, val
    G._log(step="done", hours=round((time.perf_counter() - start) / 3600, 2))


def test_forecast(model_dir: Path, panel: W.Panel, seed: int | None = None) -> np.ndarray:
    """(n_series, 4 weeks) test forecasts of a refitted variant; `panel` must hold the test sales
    (weekly.build_panel(evaluation.test_panel())) for the later origins. `seed`: the model of an extra seed."""
    best = json.loads((Path(model_dir) / "best.json").read_text())
    rows = ForecastRows(panel, "test", best.get("horizon", 1))
    bst = xgb.Booster()
    bst.load_model(Path(model_dir) / ("model_test.ubj" if seed is None else f"model_test_seed{seed}.ubj"))
    bst.set_param({"device": "cpu"})
    return rows.matrix(bst.predict(xgb.DMatrix(rows.inputs(G.INPUT_SETS[best["inputs"]]), enable_categorical=True)))
