"""Global XGBoost: variants, tuning on the validation window and refit for the test window (Section 6.5).

A variant is an input set (L, L+E, L+E+S, L+E+S+ID) with a loss (MSE, Poisson, Tweedie). Each variant is
one model for all series and all 7 horizons, trained on weekly origins with K horizons drawn at random
per origin.

Tuning. Hyperopt (TPE) proposes the hyperparameters. Each trial trains on targets up to d_1913 with early
stopping on the RMSE of the rolling validation forecasts (d_1914 - d_1941, 4 weekly origins) and is scored
by their WRMSSE, which Hyperopt minimises. Negative forecasts (possible with the MSE loss) are set to 0.

Refit. The best hyperparameters and number of rounds are refitted on targets up to d_1941. This model
forecasts the test window in the final evaluation; the test sales are not used here.

The code runs as Kaggle GPU kernels (kaggle_jobs/global_xgb/) and writes, per variant: trials.json,
best.json, valid_forecast.npy, model_valid.ubj and model_test.ubj.
"""
import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe

import features as F
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
    del table

    table = F.origin_table(panel, *F.training_origins(panel, "test"))
    for inputs, loss in variants:
        refit(panel, table, work / variant_name(inputs, loss), device)
    _log(step="done", hours=round((time.perf_counter() - start) / 3600, 2))
