"""Two checks of the weekly global XGBoost against the classical methods (weekly_classical.py), on the CPU.

subsets:  the global model when only a few series are available. For each size in SIZES, SAMPLES random
          samples of the series are drawn; on each sample the model (L+E+S+ID, MSE) is tuned on the validation
          window (TRIALS Hyperopt trials, as in weekly_xgb.py), retrained on the weeks up to 22.05.2016 and
          forecasts the test window. It sees the sampled series only, and is scored on them; the classical
          methods are local, so their forecasts of these series are those of the full run.
defaults: the global model without tuning: the default hyperparameters of XGBoost (learning rate 0.3, depth 6,
          100 trees) on all series, for validation, test and the three earlier windows; the training and
          forecasting times are recorded, together with those of the tuned hyperparameters on the test stage.
classical_times: time per series of the classical methods on one core, at the test origin: fitting with a
          one-week forecast, and the update with one new week (forward); 400 random series, scaled to 30,490.

Run:  python src/weekly_checks.py subsets | defaults | classical_times     (writes data/processed/weekly_checks/)
"""
import json
import os
import sys
import time
from functools import partial

import numpy as np
import xgboost as xgb

import evaluation as E
import global_xgb as G
import m5_data as m5
import weekly as W
import weekly_classical as C
import weekly_xgb as WX

OUT = m5.PROCESSED_DIR / "weekly_checks"
SIZES = (100, 300, 1000, 3000)
SAMPLES = 5
TRIALS = 15
DEFAULT = {"learning_rate": 0.3, "max_depth": 6, "min_child_weight": 1.0, "subsample": 1.0, "colsample_bytree": 1.0,
           "reg_lambda": 1.0}
DEFAULT_ROUNDS = 100
TUNED = m5.PROCESSED_DIR / "weekly_xgb" / "main_mse_cpu" / "LESID_mse"
CORES = os.cpu_count()  # XGBoost uses all of them; the classical methods are timed on one


def _test_actual() -> np.ndarray:
    return m5.load_test_sales()[m5.day_cols(m5.VALID_END + 1, m5.TEST_END)].to_numpy(np.float64)


def subsets() -> None:
    full = W.build_panel(E.test_panel())
    actual = _test_actual()
    make_train = partial(WX.training_matrix, horizon=1)
    OUT.mkdir(parents=True, exist_ok=True)
    records, forecasts = [], {}
    for size in SIZES:
        for sample in range(SAMPLES):
            t0 = time.perf_counter()
            rows = np.sort(np.random.default_rng(1000 * size + sample).choice(len(full.ids), size, replace=False))
            panel = full.subset(rows)
            out = OUT / "subsets" / f"n{size}_s{sample}"
            ev, val = W.evaluator("valid", rows=rows), WX.ForecastRows(panel, "valid")
            table = W.origin_table(panel, *W.training_origins(panel, "valid"))
            best = G.tune(panel, table, val, ev, "LESID", "mse", out, max_evals=TRIALS, timeout=3600, device="cpu",
                          make_train=make_train)
            table = W.origin_table(panel, *W.training_origins(panel, "test"))
            f_test = WX._train_forecast(panel, table, WX.ForecastRows(panel, "test"), best, "test", "cpu", 0)
            (out / "model_valid.ubj").unlink(missing_ok=True)
            ev_t = W.evaluator("test", actual=actual, rows=rows)
            rec = {"size": size, "sample": sample, "valid_wrmsse": best["wrmsse"], "valid_mean_rmsse": best["mean_rmsse"],
                   "test_wrmsse": ev_t.wrmsse(f_test)["all"], "test_mean_rmsse": float(np.nanmean(ev_t.bottom(f_test)["rmsse"])),
                   "rounds": best["rounds"], "max_depth": best["params"]["max_depth"],
                   "minutes": round((time.perf_counter() - t0) / 60, 2)}
            records.append(rec)
            forecasts[f"n{size}_s{sample}_rows"], forecasts[f"n{size}_s{sample}_test"] = rows, f_test
            forecasts[f"n{size}_s{sample}_valid"] = np.load(out / "valid_forecast.npy")
            G._write(OUT / "subsets.json", records)
            np.savez_compressed(OUT / "subsets.npz", **forecasts)
            G._log(step="subset", **rec)


def defaults() -> None:
    import psutil

    proc = psutil.Process()
    panel = W.build_panel(E.test_panel())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "default").mkdir(exist_ok=True)
    best_default = {"inputs": "LESID", "params": G.xgb_params("mse", DEFAULT, "cpu", 0), "rounds": DEFAULT_ROUNDS}
    best_tuned = json.loads((TUNED / "best.json").read_text())
    times = []
    stages = ["valid", "test"] + [W.add_stage(w) for w in C.BACKTEST_WEEKS]
    for stage in stages:
        t = time.perf_counter()
        val = WX.ForecastRows(panel, stage)
        table = W.origin_table(panel, *W.training_origins(panel, stage))
        t_inputs = time.perf_counter() - t
        for name, best in [("default", best_default)] + ([("tuned", best_tuned)] if stage == "test" else []):
            groups = G.INPUT_SETS[best["inputs"]]
            t = time.perf_counter()
            dtrain = WX.training_matrix(panel, table, groups, stage, 0)
            t_matrix = time.perf_counter() - t
            t = time.perf_counter()
            bst = xgb.train(best["params"] | {"device": "cpu", "seed": 0}, dtrain, best["rounds"])
            t_train = time.perf_counter() - t
            t = time.perf_counter()
            f = val.matrix(bst.predict(xgb.QuantileDMatrix(val.inputs(groups), ref=dtrain, enable_categorical=True)))
            t_predict = time.perf_counter() - t
            if name == "default":
                np.save(OUT / "default" / f"{stage}.npy", f)
            rec = {"stage": stage, "model": name, "rows": dtrain.num_row(), "rounds": best["rounds"],
                   "inputs_seconds": round(t_inputs, 1), "matrix_seconds": round(t_matrix, 1),
                   "train_seconds": round(t_train, 1), "predict_seconds": round(t_predict, 2),
                   "memory_gb": round(proc.memory_info().rss / 2**30, 2), "threads": xgb.config.get_config().get("nthread")}
            times.append(rec)
            G._write(OUT / "default_times.json", times)
            G._log(step="defaults", **rec)
            del dtrain, bst


def classical_times(n: int = 400) -> None:
    import warnings

    from statsforecast.models import AutoARIMA, AutoETS

    warnings.simplefilter("ignore")
    panel = W.build_panel(E.test_panel())
    y, first, T = panel.y, panel.first_week.astype(int), W.STAGES["test"]
    t = time.perf_counter()
    C.simple_forecasts(y, T)
    out = {"simple_all_series_seconds": time.perf_counter() - t}
    idx = [i for i in np.random.default_rng(0).choice(len(y), n, replace=False) if T - first[i] + 1 >= C.MIN_WEEKS]
    series = [y[i, first[i] : T + 2].astype(np.float64) for i in idx]
    out["n"] = len(series)
    makers = {"ses": lambda: AutoETS(model="ANN", season_length=1), "ets": lambda: AutoETS(season_length=1),
              "arima": lambda: AutoARIMA(season_length=1)}
    for name, make in makers.items():
        make().fit(series[0][:-1]).predict(1)  # compile
        t = time.perf_counter()
        models = [make().fit(s[:-1]) for s in series]
        for m in models:
            m.predict(1)
        fit = (time.perf_counter() - t) / len(series)
        t = time.perf_counter()
        for m, s in zip(models, series):
            m.forward(s, h=1)
        forward = (time.perf_counter() - t) / len(series)
        out[name] = {"fit_ms": 1000 * fit, "forward_ms": 1000 * forward, "fit_all_core_minutes": fit * len(y) / 60,
                     "forward_all_core_minutes": forward * len(y) / 60}
        G._log(step="classical_times", method=name, **out[name])
    OUT.mkdir(parents=True, exist_ok=True)
    G._write(OUT / "classical_times.json", out)


if __name__ == "__main__":
    {"subsets": subsets, "defaults": defaults, "classical_times": classical_times}[sys.argv[1]]()
