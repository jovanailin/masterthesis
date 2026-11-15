"""Models per series and a global model on the same sample of series (Sections 6.6, 7.2).

Sample. 1,000 series, 250 from each Syntetos-Boylan class (training window), drawn at random among the series
with at least MIN_HISTORY days since their first sale.

Models, all with the Tweedie loss (the loss chosen for the global model):
  per series   one model per series and input set (L, L+E, L+E+S): 3,000 models;
  sample       one model for the 1,000 series per input set (L, L+E, L+E+S, L+E+S+ID): 4 models.

Both are trained on the weekly origins up to d_1913 with all 7 horizons of each origin (the sample is small,
so the horizons are not thinned as in the global model on all series) and tuned as the global model:
Hyperopt (TPE), 30 trials, early stopping on the validation RMSE. A model per series minimises the RMSE of
its series on the validation window (which is its RMSSE up to a constant); the model of the sample
minimises the mean RMSSE of the 1,000 series. The best trial is refitted on targets up to d_1941 for the
final evaluation.

The search space is that of the global model, except for the models per series, which have about 1,900
rows each: their trees are shallower (max_depth 2-8 instead of 4-12) and min_child_weight, a sum over the
rows of a leaf, is searched in 0.1-30 instead of 1-300.

Selection bias. A model per series chooses its trial and its number of rounds on the 28 validation days of
its own series, so its validation score is optimistic by more than that of a model shared by many series.
With --holdout the models per series are chosen on the first 14 validation days only and their forecasts of
the last 14 days are saved, as a check that does not touch the test window.

Run:  python src/local_xgb.py [--quick] [--holdout]      (CPU; writes data/processed/local_xgb/ and results/local_xgb/)
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from joblib import Parallel, delayed

import features as F
import global_xgb as G
import m5_data as m5
import metrics as M

LOSS = "tweedie"
PER_CLASS = 250
MIN_HISTORY = 365
SAMPLE_SEED = 0
SERIES_INPUTS = ["L", "LE", "LES"]
SAMPLE_INPUTS = ["L", "LE", "LES", "LESID"]
OUT = m5.PROCESSED_DIR / "local_xgb"
RESULTS = m5.ROOT / "results" / "local_xgb"


def sample_series() -> pd.DataFrame:
    """The sample: row in the M5 sales file, id and demand class of each series."""
    stats = pd.read_parquet(m5.PROCESSED_DIR / "series_stats_train.parquet")
    rng = np.random.default_rng(SAMPLE_SEED)
    eligible = stats[stats["n_active"] >= MIN_HISTORY]
    rows = np.sort(np.concatenate([rng.choice(eligible.index[eligible["sbc_class"] == c].to_numpy(), PER_CLASS, replace=False)
                                   for c in sorted(eligible["sbc_class"].astype(str).unique())]))
    return stats.loc[rows, ["id", "sbc_class", "n_active", "zero_share"]].rename_axis("row").reset_index()


def series_space() -> dict:
    space = G.search_space(LOSS)
    space["max_depth"] = hp.quniform("max_depth", 2, 8, 1)
    space["min_child_weight"] = hp.loguniform("min_child_weight", np.log(0.1), np.log(30))
    return space


def tune(Xtr, ytr, Xv, yv, score, space, *, max_evals, seed, nthread, max_rounds=G.MAX_ROUNDS, patience=G.PATIENCE):
    """Hyperopt search of one model; `score(forecast)` is minimised. Returns the best trial and all trials."""
    dtrain = xgb.DMatrix(Xtr, ytr, enable_categorical=True, nthread=nthread)
    dval = xgb.DMatrix(Xv, yv, enable_categorical=True, nthread=nthread)
    log, best = [], {"score": np.inf}

    def objective(hyper):
        nonlocal best
        params = G.xgb_params(LOSS, hyper, "cpu", seed) | {"nthread": nthread}
        bst = xgb.train(params, dtrain, max_rounds, evals=[(dval, "valid")], early_stopping_rounds=patience,
                        verbose_eval=False)
        rounds = bst.best_iteration + 1
        forecast = np.maximum(bst.predict(dval, iteration_range=(0, rounds)), 0)
        rec = {"trial": len(log), **{k: params[k] for k in hyper}, "rounds": rounds, "valid_rmse": bst.best_score,
               "score": float(score(forecast))}
        log.append(rec)
        if rec["score"] < best["score"]:
            best = rec | {"params": params, "forecast": forecast, "booster": bst}
        return {"loss": rec["score"], "status": STATUS_OK}

    fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=Trials(), rstate=np.random.default_rng(seed),
         show_progressbar=False)
    return best, log


def refit(best: dict, X, y, nthread: int) -> bytes:
    """The best trial trained on targets up to d_1941, as the bytes of the model."""
    bst = xgb.train(best["params"] | {"nthread": nthread}, xgb.DMatrix(X, y, enable_categorical=True, nthread=nthread),
                    best["rounds"])
    return bytes(bst.save_raw("ubj"))


class Data:
    """Training rows of both stages and the rolling validation rows of the sample."""

    def __init__(self, panel: F.Panel):
        self.n = len(panel.ids)
        self.train = {}
        for stage in F.STAGES:
            table = F.origin_table(panel, *F.training_origins(panel, stage))
            rows, h = F.horizon_rows(len(table))
            X, y, keep = F.design(panel, table, h, stage=stage, rows=rows)
            self.train[stage] = (X, y, table.series[rows[keep]])
        self.val = G.ForecastRows(panel, "valid")

    def series_rows(self, i: int, inputs: str):
        cols = F.columns(G.INPUT_SETS[inputs])
        out = []
        for stage in F.STAGES:
            X, y, s = self.train[stage]
            out.append((X.loc[s == i, cols], y[s == i]))
        v = self.val.series == i
        return out[0], out[1], (self.val.X.loc[v, cols], self.val.y[v])


def one_series(rows: tuple, i: int, inputs: str, max_evals: int, max_rounds: int, select_days: int | None = None) -> dict:
    """Tune the model of series i. With `select_days`, the trial and the number of rounds are chosen on the first
    `select_days` days of the validation window only (check of the selection bias); no refit is made then."""
    (Xtr, ytr), (Xte, yte), (Xv, yv) = rows  # of this series only, so that little is sent to the worker
    t = time.perf_counter()
    k = select_days or len(yv)
    best, log = tune(Xtr, ytr, Xv.iloc[:k], yv[:k], lambda f: np.sqrt(np.mean((f - yv[:k]) ** 2)), series_space(),
                     max_evals=max_evals, seed=i, nthread=1, max_rounds=max_rounds)
    bst = best.pop("booster")
    if select_days:
        full = bst.predict(xgb.DMatrix(Xv, enable_categorical=True, nthread=1), iteration_range=(0, best["rounds"]))
        return {"series": i, "inputs": inputs, "forecast": np.maximum(full, 0).astype(np.float32)}
    # forecast rows of a series are ordered by origin and horizon, i.e. by day
    return {"series": i, "inputs": inputs, "rows": len(ytr), "forecast": best.pop("forecast").astype(np.float32),
            "model": refit(best, Xte, yte, 1), "best": best, "trials": log, "seconds": round(time.perf_counter() - t, 1)}


def run(quick: bool = False, holdout: bool = False) -> None:
    start = time.perf_counter()
    max_evals, max_rounds = (3, 30) if quick else (G.MAX_EVALS, G.MAX_ROUNDS)
    out, results = (OUT / "quick", RESULTS / "quick") if quick else (OUT, RESULTS)
    out.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    sample = sample_series()
    if quick:
        sample = sample.groupby("sbc_class", observed=True).head(2).reset_index(drop=True)
    if not holdout:
        sample.to_csv(results / "sample.csv", index=False)
    rows = sample["row"].to_numpy()
    panel = F.Panel.load(m5.PROCESSED_DIR / "panel") if (m5.PROCESSED_DIR / "panel" / "arrays.npz").exists() else F.build_panel()
    panel = panel.subset(rows)
    ev = M.Evaluator.for_origin(m5.TRAIN_END, (m5.TRAIN_END + 1, m5.VALID_END), rows=rows)
    data = Data(panel)
    print("data", {s: len(v[1]) for s, v in data.train.items()}, round(time.perf_counter() - start, 1), "s", flush=True)

    if holdout:  # models per series chosen on the first two weeks of the validation window only
        done = Parallel(n_jobs=-1, verbose=5)(delayed(one_series)(data.series_rows(i, inputs), i, inputs, max_evals, max_rounds, 2 * F.HORIZON)
                                              for inputs in SERIES_INPUTS for i in range(data.n))
        np.savez(out / "valid_forecasts_first_half.npz", **{
            f"per_series_{inputs}": np.stack([d["forecast"] for d in sorted((d for d in done if d["inputs"] == inputs),
                                                                            key=lambda d: d["series"])])
            for inputs in SERIES_INPUTS})
        return

    # ---- one model per series
    tasks = [(i, inputs) for inputs in SERIES_INPUTS for i in range(data.n)]
    done = Parallel(n_jobs=-1, verbose=5)(delayed(one_series)(data.series_rows(i, inputs), i, inputs, max_evals, max_rounds) for i, inputs in tasks)
    forecasts, models, table, trials = {}, {}, [], []
    for inputs in SERIES_INPUTS:
        part = sorted((d for d in done if d["inputs"] == inputs), key=lambda d: d["series"])
        forecasts[f"per_series_{inputs}"] = np.stack([d["forecast"] for d in part])
        models[f"per_series_{inputs}"] = [d["model"] for d in part]
        for d in part:
            b = d["best"]
            table.append({"id": sample["id"][d["series"]], "inputs": inputs, "rows": d["rows"],
                          **{k: v for k, v in b.items() if k not in ("params", "score")}, "seconds": d["seconds"]})
            trials += [{"id": sample["id"][d["series"]], "inputs": inputs, **t} for t in d["trials"]]
    pd.DataFrame(table).to_csv(results / "per_series_best.csv", index=False)
    pd.DataFrame(trials).to_csv(results / "per_series_trials.csv.gz", index=False, float_format="%.5g")
    print("per series done", round((time.perf_counter() - start) / 60, 1), "min", flush=True)

    # ---- one model for the sample
    Xtr, ytr, _ = data.train["valid"]
    Xte, yte, _ = data.train["test"]
    summary = []
    for inputs in SAMPLE_INPUTS:
        cols = F.columns(G.INPUT_SETS[inputs])
        t = time.perf_counter()
        best, log = tune(Xtr[cols], ytr, data.val.X[cols], data.val.y,
                         lambda f: ev.bottom(data.val.matrix(f))["rmsse"].mean(), G.search_space(LOSS),
                         max_evals=max_evals, seed=0, nthread=-1, max_rounds=max_rounds)
        best.pop("booster")
        forecasts[f"sample_{inputs}"] = data.val.matrix(best.pop("forecast"))
        models[f"sample_{inputs}"] = refit(best, Xte[cols], yte, -1)
        best |= {"inputs": inputs, "rows": len(ytr), "trials": len(log), "seconds": round(time.perf_counter() - t, 1)}
        summary.append(best)
        (results / f"sample_{inputs}_trials.json").write_text(json.dumps(log, indent=1, default=float))
        print("sample", inputs, round(best["score"], 4), best["rounds"], best["seconds"], "s", flush=True)
    (results / "sample_best.json").write_text(json.dumps(summary, indent=1, default=float))

    np.savez(out / "valid_forecasts.npz", **forecasts)
    with open(out / "models_test.pkl", "wb") as f:
        pickle.dump(models, f)
    print("done", round((time.perf_counter() - start) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    run("--quick" in sys.argv, "--holdout" in sys.argv)
