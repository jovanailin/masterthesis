"""Global XGBoost: variants, tuning on the validation window and refit for the test window (Section 6.6).

A variant is an input set (L, L+E, L+E+S, L+E+S+ID) with a loss (MSE, Poisson, Tweedie). Each variant is
one model for all series and all 7 horizons, trained on weekly origins with K horizons drawn at random
per origin.

Tuning. Hyperopt (TPE) proposes the hyperparameters. Each trial trains on targets up to d_1913 with early
stopping on the RMSE of the rolling validation forecasts (d_1914 - d_1941, 4 weekly origins) and is scored
by their WRMSSE, which Hyperopt minimises. Negative forecasts (possible with the MSE loss) are set to 0.

Refit. The best hyperparameters and number of rounds are refitted on targets up to d_1941. This model
forecasts the test window in the final evaluation; the test sales are not used here.

Seeds. The variant with the lowest validation WRMSSE is trained again with other seeds (which change the
row and column subsampling and the horizons drawn per origin), with the hyperparameters and the number of
rounds of its best trial: once on targets up to d_1913, to measure the spread of the validation scores,
and once on targets up to d_1941, for the final evaluation.

The code runs as Kaggle GPU kernels (kaggle_jobs/global_xgb/) and writes, per variant: trials.json,
best.json, valid_forecast.npy, model_valid.ubj and model_test.ubj; for the seeds also seeds.json,
valid_forecast_seed<k>.npy and model_test_seed<k>.ubj.

Partially global models and the recursive strategy (run_extra) are described in the last part of the file.
"""
import dataclasses
import json
import shutil
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe

import features as F
import intermittency
import m5_data as m5
import metrics as M

INPUT_SETS = {"L": ("L",), "LE": ("L", "E"), "LES": ("L", "E", "S"), "LESID": ("L", "E", "S", "ID")}
LOSSES = {
    "mse": {"objective": "reg:squarederror"},
    "poisson": {"objective": "count:poisson"},
    "tweedie": {"objective": "reg:tweedie"},
}
PER_ORIGIN = 2  # horizons drawn per weekly origin (about 13M training rows)
MAX_BIN = 256
BASE = {"tree_method": "hist", "max_bin": MAX_BIN, "eval_metric": "rmse"}
MAX_ROUNDS, PATIENCE = 1500, 30
MAX_EVALS = 30
REFIT_RESERVE = 12 * 60  # seconds of the time budget kept for the refit of each variant


def variant_name(inputs: str, loss: str) -> str:
    return f"{inputs}_{loss}"


def search_space(loss: str) -> dict:
    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.03), np.log(0.15)),
        "max_depth": hp.quniform("max_depth", 4, 12, 1),
        "min_child_weight": hp.loguniform("min_child_weight", np.log(1), np.log(300)),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(0.1), np.log(100)),
    }
    if loss == "tweedie":
        space["tweedie_variance_power"] = hp.uniform("tweedie_variance_power", 1.05, 1.6)
    return space


def xgb_params(loss: str, hyper: dict, device: str, seed: int) -> dict:
    hyper = {k: (int(v) if k == "max_depth" else float(v)) for k, v in hyper.items()}
    return BASE | LOSSES[loss] | hyper | {"device": device, "seed": seed}


def training_matrix(panel: F.Panel, table: F.OriginTable, groups, stage: str, seed: int = 0) -> xgb.QuantileDMatrix:
    rows, h = F.horizon_rows(len(table), PER_ORIGIN, seed)
    X, y, _ = F.design(panel, table, h, groups, stage=stage, rows=rows)
    return xgb.QuantileDMatrix(X, y, max_bin=MAX_BIN, enable_categorical=True)


class ForecastRows:
    """Rows of the rolling forecasts of a stage's window: every series, EVAL_WEEKS origins, 7 horizons."""

    def __init__(self, panel: F.Panel, stage: str):
        s, t = F.forecast_origins(panel, stage)
        table = F.origin_table(panel, s, t)
        rows, h = F.horizon_rows(len(table))
        self.n = len(panel.ids)
        self.series = s[rows]
        self.X, _, _ = F.design(panel, table, h, target=False, rows=rows)
        day = t[rows].astype(np.int64) + h
        self.y = panel.y[s[rows], day] if day.max() < panel.y.shape[1] else None
        # rows are ordered by origin, series, horizon; the forecast matrix is (series, day)
        assert (s[rows].reshape(m5.EVAL_WEEKS, self.n, F.HORIZON)[0, :, 0] == np.arange(self.n)).all()

    def inputs(self, groups):
        return self.X[F.columns(groups)]

    def matrix(self, pred: np.ndarray) -> np.ndarray:
        """Forecasts as (n_series, 28 days), negative values set to 0."""
        f = pred.reshape(m5.EVAL_WEEKS, self.n, F.HORIZON).transpose(1, 0, 2).reshape(self.n, -1)
        return np.maximum(f, 0).astype(np.float32)


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=float))


def _log(**info) -> None:
    print(json.dumps(info, default=float), flush=True)


def tune(panel, table, val: ForecastRows, ev: M.Evaluator, inputs: str, loss: str, out: Path, *, max_evals: int,
         timeout: float, device: str, max_rounds: int = MAX_ROUNDS, patience: int = PATIENCE, seed: int = 0) -> dict:
    """Hyperopt search of one variant; returns (and saves) the best trial."""
    out.mkdir(parents=True, exist_ok=True)
    groups = INPUT_SETS[inputs]
    name = variant_name(inputs, loss)
    t = time.perf_counter()
    dtrain = training_matrix(panel, table, groups, "valid", seed)
    dval = xgb.QuantileDMatrix(val.inputs(groups), val.y, ref=dtrain, enable_categorical=True)
    _log(variant=name, step="matrices", rows=dtrain.num_row(), columns=dtrain.num_col(),
         seconds=round(time.perf_counter() - t, 1))

    log, best = [], {"wrmsse": np.inf}

    def objective(hyper):
        nonlocal best
        params = xgb_params(loss, hyper, device, seed)
        rec = {"trial": len(log), **{k: params[k] for k in hyper}}
        t = time.perf_counter()
        try:
            bst = xgb.train(params, dtrain, max_rounds, evals=[(dval, "valid")], early_stopping_rounds=patience,
                            verbose_eval=False)
            rounds = bst.best_iteration + 1
            forecast = val.matrix(bst.predict(dval, iteration_range=(0, rounds)))
            score = ev.wrmsse(forecast)
            rec |= {"rounds": rounds, "valid_rmse": bst.best_score, "wrmsse": score["all"],
                    "mean_rmsse": float(ev.bottom(forecast)["rmsse"].mean()), "status": "ok"}
        except Exception as e:  # noqa: BLE001 - e.g. out of GPU memory: record the trial and go on
            rec |= {"status": "fail", "error": repr(e)[:300]}
        rec["seconds"] = round(time.perf_counter() - t, 1)
        log.append(rec)
        _write(out / "trials.json", log)
        _log(variant=name, **rec)
        if rec["status"] != "ok":
            return {"status": STATUS_FAIL}
        if rec["wrmsse"] < best["wrmsse"]:
            best = rec | {"variant": name, "inputs": inputs, "loss": loss, "seed": seed, "params": params,
                          "wrmsse_by_level": {str(k): v for k, v in score.items()}}
            bst[:rounds].save_model(out / "model_valid.ubj")
            np.save(out / "valid_forecast.npy", forecast)
            _write(out / "best.json", best)
        return {"loss": rec["wrmsse"], "status": STATUS_OK}

    fmin(objective, search_space(loss), algo=tpe.suggest, max_evals=max_evals, trials=Trials(),
         timeout=max(int(timeout), 60), rstate=np.random.default_rng(seed), show_progressbar=False)
    best["trials"] = len(log)
    _write(out / "best.json", best)
    return best


def refit(panel, table, out: Path, device: str, seed: int | None = None) -> None:
    """Refit the best trial of a variant on targets up to d_1941; with `seed`, as an extra run of that seed."""
    best = json.loads((out / "best.json").read_text())
    params = best["params"] | {"device": device}
    file = "model_test.ubj"
    if seed is not None:
        params["seed"], file = seed, f"model_test_seed{seed}.ubj"
    t = time.perf_counter()
    dtrain = training_matrix(panel, table, INPUT_SETS[best["inputs"]], "test", params["seed"])
    bst = xgb.train(params, dtrain, best["rounds"])
    bst.save_model(out / file)
    _log(variant=best["variant"], step="refit", file=file, rows=dtrain.num_row(), rounds=best["rounds"],
         seconds=round(time.perf_counter() - t, 1))


def seed_valid(panel, table, val: ForecastRows, ev: M.Evaluator, out: Path, device: str, seed: int) -> None:
    """Train the best trial of a variant with another seed on targets up to d_1913 and score its forecasts."""
    best = json.loads((out / "best.json").read_text())
    groups = INPUT_SETS[best["inputs"]]
    t = time.perf_counter()
    dtrain = training_matrix(panel, table, groups, "valid", seed)
    bst = xgb.train(best["params"] | {"device": device, "seed": seed}, dtrain, best["rounds"])
    dval = xgb.QuantileDMatrix(val.inputs(groups), ref=dtrain, enable_categorical=True)
    forecast = val.matrix(bst.predict(dval))
    np.save(out / f"valid_forecast_seed{seed}.npy", forecast)
    score = ev.wrmsse(forecast)
    rec = {"variant": best["variant"], "seed": seed, "wrmsse": score["all"],
           "mean_rmsse": float(ev.bottom(forecast)["rmsse"].mean()),
           "wrmsse_by_level": {str(k): v for k, v in score.items()}, "seconds": round(time.perf_counter() - t, 1)}
    file = out / "seeds.json"
    _write(file, (json.loads(file.read_text()) if file.exists() else []) + [rec])
    _log(step="seed", **{k: v for k, v in rec.items() if k != "wrmsse_by_level"})


def run(job: dict, work: Path) -> None:
    """Tune and refit the variants of a job: {"variants": [[inputs, loss], ...], "budget_hours": ..., ...}.

    The time budget is shared equally among the variants that are still to be tuned; a variant whose
    share runs out is left with fewer than `max_evals` trials (recorded in best.json).
    """
    start = time.perf_counter()
    device = job.get("device", "cuda")
    variants = job["variants"]
    deadline = start + job["budget_hours"] * 3600 - REFIT_RESERVE * len(variants)

    panel, rows = F.build_panel(), None
    if job.get("series_step"):  # quick check of the code on a subset of the series
        rows = np.arange(0, len(panel.ids), job["series_step"])
        panel = panel.subset(rows)
    ev = M.Evaluator.for_origin(m5.TRAIN_END, (m5.TRAIN_END + 1, m5.VALID_END), rows=rows)
    val = ForecastRows(panel, "valid")
    _log(step="data", series=len(panel.ids), validation_rows=len(val.y), seconds=round(time.perf_counter() - start, 1))

    table = F.origin_table(panel, *F.training_origins(panel, "valid"))
    summary = []
    for k, (inputs, loss) in enumerate(variants):
        out = work / variant_name(inputs, loss)
        best = tune(panel, table, val, ev, inputs, loss, out, max_evals=job.get("max_evals", MAX_EVALS),
                    timeout=(deadline - time.perf_counter()) / (len(variants) - k), device=device,
                    max_rounds=job.get("max_rounds", MAX_ROUNDS), patience=job.get("patience", PATIENCE))
        summary.append({k: best.get(k) for k in ["variant", "trials", "wrmsse", "mean_rmsse", "valid_rmse", "rounds"]})
        _write(work / "summary.json", summary)

    seeds, top = job.get("seeds", []), None
    if seeds:  # the best variant among those of this job and those tuned by earlier jobs
        dirs = [work / variant_name(*v) for v in variants] + [Path(d) for d in job.get("seed_candidate_dirs", [])]
        scores = {d: json.loads((d / "best.json").read_text())["wrmsse"] for d in dirs}
        source = min(scores, key=scores.get)
        top = work / source.name
        top.mkdir(parents=True, exist_ok=True)
        if source != top:
            shutil.copy(source / "best.json", top / "best.json")
        _log(step="best variant", variant=source.name, candidates={d.name: s for d, s in scores.items()})
        for seed in seeds:
            seed_valid(panel, table, val, ev, top, device, seed)
    del table

    table = F.origin_table(panel, *F.training_origins(panel, "test"))
    for inputs, loss in variants:
        refit(panel, table, work / variant_name(inputs, loss), device)
    for seed in seeds:
        refit(panel, table, top, device, seed)
    _log(step="done", hours=round((time.perf_counter() - start) / 3600, 2))


# --------------------------------------------------------------------------- partially global models and the recursive strategy
#
# Both use the hyperparameters of the best trial of a tuned global variant (they are not tuned again); only
# the number of rounds is chosen again, by early stopping on the validation window.


def series_groups(panel: F.Panel, scheme: str, rows: np.ndarray | None = None) -> np.ndarray:
    """Group label of each series: its store, or its demand class (Syntetos-Boylan, training window)."""
    if scheme == "store":
        return panel.hier["store_id"].astype(str).to_numpy()
    sales = m5.load_sales(last_day=m5.TRAIN_END)
    if rows is not None:
        sales = sales.iloc[rows]
    dates = m5.load_calendar().set_index("d_num")["date"].loc[1 : m5.TRAIN_END]
    open_days = ~((dates.dt.month == 12) & (dates.dt.day == 25)).to_numpy()  # as in the EDA: without Christmas
    stats = intermittency.demand_stats(sales[m5.day_cols(1, m5.TRAIN_END)].to_numpy()[:, open_days])
    return np.asarray(intermittency.classify_sbc(stats["adi"], stats["cv2"]).astype(str))


def partial_valid(panel, table, val: ForecastRows, ev, best: dict, scheme: str, labels: np.ndarray, out: Path,
                  device: str, max_rounds: int, patience: int) -> None:
    """One model per group of series, trained on the group's rows of the global training table."""
    out.mkdir(parents=True, exist_ok=True)
    groups = INPUT_SETS[best["inputs"]]
    params = best["params"] | {"device": device}
    rows, h = F.horizon_rows(len(table), PER_ORIGIN, best["seed"])
    pred, info = np.full(len(val.y), np.nan, dtype=np.float32), []
    for g in sorted(set(labels)):
        t = time.perf_counter()
        member = labels == g
        sel = member[table.series[rows]]
        X, y, _ = F.design(panel, table, h[sel], groups, stage="valid", rows=rows[sel])
        dtrain = xgb.QuantileDMatrix(X, y, max_bin=MAX_BIN, enable_categorical=True)
        vsel = member[val.series]
        dval = xgb.QuantileDMatrix(val.inputs(groups)[vsel], val.y[vsel], ref=dtrain, enable_categorical=True)
        bst = xgb.train(params, dtrain, max_rounds, evals=[(dval, "valid")], early_stopping_rounds=patience,
                        verbose_eval=False)
        rounds = bst.best_iteration + 1
        pred[vsel] = bst.predict(dval, iteration_range=(0, rounds))
        info.append({"group": g, "series": int(member.sum()), "rows": int(len(y)), "rounds": rounds,
                     "valid_rmse": bst.best_score, "seconds": round(time.perf_counter() - t, 1)})
        _log(step="partial", scheme=scheme, **info[-1])
    forecast = val.matrix(pred)
    np.save(out / "valid_forecast.npy", forecast)
    score = ev.wrmsse(forecast)
    _write(out / "best.json", {"variant": f"{best['variant']}_per_{scheme}", "scheme": scheme, "inputs": best["inputs"],
                               "loss": best["loss"], "seed": best["seed"], "params": params,
                               "hyperparameters_of": best["variant"], "groups": info, "wrmsse": score["all"],
                               "mean_rmsse": float(ev.bottom(forecast)["rmsse"].mean()),
                               "wrmsse_by_level": {str(k): v for k, v in score.items()}})
    _log(step="partial", scheme=scheme, wrmsse=score["all"])


def partial_refit(panel, table, labels: np.ndarray, out: Path, device: str) -> None:
    best = json.loads((out / "best.json").read_text())
    rows, h = F.horizon_rows(len(table), PER_ORIGIN, best["seed"])
    for g in best["groups"]:
        sel = (labels == g["group"])[table.series[rows]]
        X, y, _ = F.design(panel, table, h[sel], INPUT_SETS[best["inputs"]], stage="test", rows=rows[sel])
        dtrain = xgb.QuantileDMatrix(X, y, max_bin=MAX_BIN, enable_categorical=True)
        xgb.train(best["params"] | {"device": device}, dtrain, g["rounds"]).save_model(out / f"model_test_{g['group']}.ubj")
    _log(step="partial refit", scheme=best["scheme"], models=len(best["groups"]))


def one_step_table(panel: F.Panel, stage: str, seed: int) -> F.OriginTable:
    """Daily origins of the one-step model, thinned at random to the size of the direct model's table."""
    s, t = F.training_origins(panel, stage, stride=1)
    keep = np.random.default_rng(seed).random(s.size) < PER_ORIGIN / F.HORIZON
    return F.origin_table(panel, s[keep], t[keep])


def recursive_forecast(panel: F.Panel, bst: xgb.Booster, groups, stage: str) -> np.ndarray:
    """Rolling 7-day forecasts of a one-step model: each forecast is fed back as the sales of its day.

    Returns (n_series, 28). Within a week the model sees the actual sales up to the origin and its own
    forecasts after it; the inputs derived from the sales history are computed from this mixed history.
    """
    n = len(panel.ids)
    series = np.arange(n, dtype=np.int32)
    out = np.empty((n, m5.EVAL_WEEKS * F.HORIZON), dtype=np.float32)
    for w in range(m5.EVAL_WEEKS):
        origin = F.STAGES[stage] + F.HORIZON * w
        y = panel.y.copy()
        for k in range(1, F.HORIZON + 1):
            p = dataclasses.replace(panel, y=y)
            table = F.origin_table(p, series, np.full(n, origin + k - 1, dtype=np.int16))
            X, _, _ = F.design(p, table, 1, groups, target=False)
            pred = np.maximum(bst.predict(xgb.DMatrix(X, enable_categorical=True)), 0)
            out[:, F.HORIZON * w + k - 1] = pred
            y[:, origin + k] = pred
    return out


def recursive_valid(panel, ev, best: dict, out: Path, device: str, max_rounds: int, patience: int) -> None:
    """One-step model (h = 1, daily origins) with early stopping on the one-step validation forecasts."""
    out.mkdir(parents=True, exist_ok=True)
    groups = INPUT_SETS[best["inputs"]]
    params = best["params"] | {"device": device}
    t = time.perf_counter()
    table = one_step_table(panel, "valid", best["seed"])
    X, y, _ = F.design(panel, table, 1, groups, stage="valid")
    dtrain = xgb.QuantileDMatrix(X, y, max_bin=MAX_BIN, enable_categorical=True)
    del X, table
    n = len(panel.ids)  # one-step validation rows: every day of the window from the day before it
    days = np.arange(m5.TRAIN_END, m5.VALID_END, dtype=np.int16)
    vt = F.origin_table(panel, np.tile(np.arange(n, dtype=np.int32), days.size), np.repeat(days, n))
    Xv, _, _ = F.design(panel, vt, 1, groups, target=False)
    dval = xgb.QuantileDMatrix(Xv, panel.y[vt.series, vt.t.astype(np.int64) + 1], ref=dtrain, enable_categorical=True)
    bst = xgb.train(params, dtrain, max_rounds, evals=[(dval, "valid")], early_stopping_rounds=patience, verbose_eval=False)
    rounds, rmse = bst.best_iteration + 1, bst.best_score
    bst = bst[:rounds]
    bst.save_model(out / "model_valid.ubj")
    one_step = np.maximum(bst.predict(dval), 0).reshape(days.size, n).T.astype(np.float32)
    forecast = recursive_forecast(panel, bst, groups, "valid")
    np.save(out / "valid_forecast.npy", forecast)
    np.save(out / "valid_forecast_one_step.npy", one_step)
    score = ev.wrmsse(forecast)
    _write(out / "best.json", {"variant": f"{best['variant']}_recursive", "inputs": best["inputs"], "loss": best["loss"],
                               "seed": best["seed"], "params": params, "hyperparameters_of": best["variant"],
                               "rows": dtrain.num_row(), "rounds": rounds, "valid_rmse_one_step": rmse,
                               "wrmsse": score["all"], "mean_rmsse": float(ev.bottom(forecast)["rmsse"].mean()),
                               "wrmsse_by_level": {str(k): v for k, v in score.items()},
                               "wrmsse_one_step": ev.wrmsse(one_step)["all"],
                               "mean_rmsse_one_step": float(ev.bottom(one_step)["rmsse"].mean())})
    _log(step="recursive", rows=dtrain.num_row(), rounds=rounds, wrmsse=score["all"],
         seconds=round(time.perf_counter() - t, 1))


def recursive_refit(panel, out: Path, device: str) -> None:
    best = json.loads((out / "best.json").read_text())
    table = one_step_table(panel, "test", best["seed"])
    X, y, _ = F.design(panel, table, 1, INPUT_SETS[best["inputs"]], stage="test")
    dtrain = xgb.QuantileDMatrix(X, y, max_bin=MAX_BIN, enable_categorical=True)
    xgb.train(best["params"] | {"device": device}, dtrain, best["rounds"]).save_model(out / "model_test.ubj")
    _log(step="recursive refit", rows=dtrain.num_row(), rounds=best["rounds"])


def run_extra(job: dict, work: Path) -> None:
    """Models per store and per demand class, and the recursive strategy, with the hyperparameters of the
    tuned variant in job["source_dir"]."""
    start = time.perf_counter()
    device = job.get("device", "cuda")
    stop = {"max_rounds": job.get("max_rounds", MAX_ROUNDS), "patience": job.get("patience", PATIENCE)}
    best = json.loads((Path(job["source_dir"]) / "best.json").read_text())
    panel, rows = F.build_panel(), None
    if job.get("series_step"):
        rows = np.arange(0, len(panel.ids), job["series_step"])
        panel = panel.subset(rows)
    ev = M.Evaluator.for_origin(m5.TRAIN_END, (m5.TRAIN_END + 1, m5.VALID_END), rows=rows)
    val = ForecastRows(panel, "valid")
    labels = {scheme: series_groups(panel, scheme, rows) for scheme in job.get("schemes", [])}

    table = F.origin_table(panel, *F.training_origins(panel, "valid"))
    for scheme, lab in labels.items():
        partial_valid(panel, table, val, ev, best, scheme, lab, work / f"{best['variant']}_per_{scheme}", device, **stop)
    del table
    table = F.origin_table(panel, *F.training_origins(panel, "test"))
    for scheme, lab in labels.items():
        partial_refit(panel, table, lab, work / f"{best['variant']}_per_{scheme}", device)
    del table

    if job.get("recursive"):
        out = work / f"{best['variant']}_recursive"
        recursive_valid(panel, ev, best, out, device, **stop)
        recursive_refit(panel, out, device)
    _log(step="done", hours=round((time.perf_counter() - start) / 3600, 2))
