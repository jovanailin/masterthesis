"""Replication of the 3 x 2 factorial study of Tsantilis et al. (2025).

Time partition (1-based M5 day indices):
    tuning   : train d_1 - d_1885, validation d_1886 - d_1913
    final    : refit on d_1 - d_1913, blind test d_1914 - d_1941

Per-series regime: each architecture is tuned and trained separately on every series.
Global regime: one model per architecture on the stacked windows of all series
(no series identifier). In both regimes every series is min-max scaled to [0, 1]
with a scaler fit on its own training data, and forecasts are transformed back
before any error is computed.

Tuning: Hyperopt TPE with a single Trials object and `max_evals` evaluations,
seeded. The three configurations with the lowest validation RMSSE are refit on the
full training window; their forecasts are clipped at zero and averaged.

Results are cached as JSON (one file per series for the per-series cells, one per
global cell), so an interrupted run resumes where it stopped. Usage:

    python src/experiment.py --run full --cells all --max-evals 100 --n-jobs 6
"""
import argparse
import json
import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m5_data as m5  # noqa: E402
from forecasters import SEARCH_SPACES, make_forecaster  # noqa: E402  (also silences hyperopt's import warning)
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, space_eval, tpe  # noqa: E402

TUNE_END = 1885  # last training day during tuning; validation is d_1886 - d_1913
TRAIN_END = 1913  # last training day of the final refit; blind test is d_1914 - d_1941
TEST_END = 1941
LOOKBACK = 56
HORIZON = 28
SEED = 42
TOP_K = 3

# Cell name -> (architecture, regime), in the order of the paper's Table I.
CELLS = {
    "LGBM-MSE-PS": ("lgbm", "ps"),
    "LGBM-MSE-GL": ("lgbm", "gl"),
    "LSTM-PS": ("lstm", "ps"),
    "LSTM-GL": ("lstm", "gl"),
    "RES-PS": ("esn", "ps"),
    "RES-GL": ("esn", "gl"),
}

RESULTS_DIR = m5.ROOT / "results" / "replication"
CHECKPOINT_DIR = m5.PROCESSED_DIR / "replication"
SAMPLE_FILE = RESULTS_DIR / "sample.csv"

logging.getLogger("hyperopt").setLevel(logging.WARNING)
log = logging.getLogger("replication")


# --------------------------------------------------------------------------- data


@dataclass
class SeriesData:
    """Training windows and evaluation targets of one series for one phase."""

    X: np.ndarray  # (n_windows, LOOKBACK), scaled
    Y: np.ndarray  # (n_windows, HORIZON), scaled
    x_last: np.ndarray  # (LOOKBACK,) scaled input of the forecast origin
    lo: float  # min-max scaler
    span: float
    actual: np.ndarray  # (HORIZON,) unscaled target window
    scale: float  # RMSSE denominator


def prepare(y: np.ndarray, train_end: int) -> SeriesData:
    """Windows from d_1 - d_{train_end}; the target is the following HORIZON days."""
    hist = y[:train_end].astype(np.float64)
    lo, hi = hist.min(), hist.max()
    span = hi - lo if hi > lo else 1.0
    z = ((hist - lo) / span).astype(np.float32)
    win = np.lib.stride_tricks.sliding_window_view(z, LOOKBACK + HORIZON)
    start = np.flatnonzero(hist)[0]  # M5 convention: the leading zero prefix is skipped in the scale
    return SeriesData(
        X=np.ascontiguousarray(win[:, :LOOKBACK]),
        Y=np.ascontiguousarray(win[:, LOOKBACK:]),
        x_last=z[-LOOKBACK:].copy(),
        lo=lo,
        span=span,
        actual=y[train_end : train_end + HORIZON].astype(np.float64),
        scale=float(np.mean(np.diff(hist[start:]) ** 2)),
    )


def to_units(z_hat: np.ndarray, d: SeriesData) -> np.ndarray:
    """Inverse min-max transform, clipped at zero."""
    return np.clip(z_hat * d.span + d.lo, 0, None)


def rmsse(actual: np.ndarray, forecast: np.ndarray, scale: float) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2) / scale))


def rmsse_by_horizon(actual: np.ndarray, forecast: np.ndarray, scale: float) -> np.ndarray:
    """RMSSE@h for h = 1..H, using the first h days of the forecast window."""
    sq = (actual - forecast) ** 2
    return np.sqrt(np.cumsum(sq, axis=-1) / np.arange(1, sq.shape[-1] + 1) / scale)


def load_panel(sample_file: Path = SAMPLE_FILE) -> tuple[list[str], np.ndarray]:
    """Ids and daily sales d_1 - d_1941 of the sampled series, in sample order."""
    ids = pd.read_csv(sample_file)["id"].tolist()
    sales = m5.load_sales(last_day=TEST_END).set_index("id")
    return ids, sales.loc[ids, m5.day_cols(1, TEST_END)].to_numpy()


# --------------------------------------------------------------------------- tuning


def _jsonable(params: dict) -> dict:
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()}


def _trial_params(trial: dict, space: dict) -> dict:
    return _jsonable(space_eval(space, {k: v[0] for k, v in trial["misc"]["vals"].items() if v}))


def _trial_log(trials: Trials, space: dict) -> list[dict]:
    return [
        {"loss": t["result"].get("loss"), "seconds": t["result"].get("seconds"), "params": _trial_params(t, space)}
        for t in trials.trials
    ]


def _top_configs(trials: Trials, space: dict, k: int = TOP_K) -> tuple[list[dict], list[float]]:
    done = sorted((t for t in trials.trials if t["result"]["status"] == STATUS_OK), key=lambda t: t["result"]["loss"])
    return [_trial_params(t, space) for t in done[:k]], [t["result"]["loss"] for t in done[:k]]


def _result(loss: float, t0: float) -> dict:
    seconds = time.perf_counter() - t0
    if not np.isfinite(loss):
        return {"status": STATUS_FAIL, "seconds": seconds}
    return {"loss": loss, "status": STATUS_OK, "seconds": seconds}


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def run_series(cell: str, series_id: str, y: np.ndarray, max_evals: int, out_file: Path, seed: int = SEED) -> None:
    """Per-series regime for one series: tune, refit the top-3 configurations, write JSON."""
    logging.getLogger("hyperopt").setLevel(logging.WARNING)
    arch = CELLS[cell][0]
    space = SEARCH_SPACES[arch]
    tune, final = prepare(y, TUNE_END), prepare(y, TRAIN_END)

    def objective(params):
        t0 = time.perf_counter()
        model = make_forecaster(arch, params, seed, n_threads=1).fit(tune.X, tune.Y)
        return _result(rmsse(tune.actual, to_units(model.predict(tune.x_last[None])[0], tune), tune.scale), t0)

    t_start = time.perf_counter()
    trials = Trials()
    fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials,
         rstate=np.random.default_rng(seed), show_progressbar=False)
    top, top_loss = _top_configs(trials, space)
    members = [
        to_units(make_forecaster(arch, p, seed, n_threads=1).fit(final.X, final.Y).predict(final.x_last[None])[0], final)
        for p in top
    ]
    _write_json(out_file, {
        "cell": cell, "series_id": series_id, "max_evals": max_evals, "seed": seed,
        "top_configs": top, "top_val_rmsse": top_loss,
        "member_forecasts": [m.tolist() for m in members],
        "forecast": np.mean(members, axis=0).tolist(),
        "seconds": time.perf_counter() - t_start,
        "trials": _trial_log(trials, space),
    })


def _stack(data: list[SeriesData]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.concatenate([d.X for d in data]), np.concatenate([d.Y for d in data]), np.stack([d.x_last for d in data]))


def run_global(cell: str, ids: list[str], Y: np.ndarray, max_evals: int, out_file: Path, checkpoint: Path,
               n_threads: int, seed: int = SEED) -> None:
    """Global regime: one model on the pooled windows; the objective is the mean validation RMSSE."""
    arch = CELLS[cell][0]
    space = SEARCH_SPACES[arch]
    tune = [prepare(y, TUNE_END) for y in Y]
    final = [prepare(y, TRAIN_END) for y in Y]
    X_tr, Y_tr, x_last = _stack(tune)

    def objective(params):
        t0 = time.perf_counter()
        model = make_forecaster(arch, params, seed, n_threads).fit(X_tr, Y_tr)
        z_hat = model.predict(x_last)
        return _result(np.mean([rmsse(d.actual, to_units(z, d), d.scale) for z, d in zip(z_hat, tune)]), t0)

    t_start = time.perf_counter()
    trials = pickle.loads(checkpoint.read_bytes()) if checkpoint.exists() else Trials()
    while len(trials.trials) < max_evals:
        n_done = len(trials.trials)
        # One evaluation per call so the search can be checkpointed; the seed depends only on
        # the number of completed trials, so a resumed run makes the same suggestions.
        fmin(objective, space, algo=tpe.suggest, max_evals=n_done + 1, trials=trials,
             rstate=np.random.default_rng(seed + n_done), show_progressbar=False)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(pickle.dumps(trials))
        last = trials.trials[-1]["result"]
        best = min((v for v in trials.losses() if v is not None), default=np.nan)
        log.info("%s eval %d/%d  loss=%s  best=%.4f  (%.0f s)", cell, n_done + 1, max_evals,
                 f"{last['loss']:.4f}" if "loss" in last else "failed", best, last["seconds"])

    top, top_loss = _top_configs(trials, space)
    X_fin, Y_fin, x_fin = _stack(final)
    members = []
    for p in top:
        z_hat = make_forecaster(arch, p, seed, n_threads).fit(X_fin, Y_fin).predict(x_fin)
        members.append(np.stack([to_units(z, d) for z, d in zip(z_hat, final)]))
    _write_json(out_file, {
        "cell": cell, "series_ids": ids, "max_evals": max_evals, "seed": seed,
        "top_configs": top, "top_val_rmsse": top_loss,
        "member_forecasts": [m.tolist() for m in members],
        "forecast": np.mean(members, axis=0).tolist(),
        "seconds": time.perf_counter() - t_start,
        "trials": _trial_log(trials, space),
    })


# --------------------------------------------------------------------------- driver


def cell_dir(run: str, cell: str) -> Path:
    return RESULTS_DIR / "runs" / run / cell


def series_file(run: str, cell: str, series_id: str) -> Path:
    return cell_dir(run, cell) / "series" / f"{series_id}.json"


def run_cell(cell: str, ids: list[str], Y: np.ndarray, run: str, max_evals: int, n_jobs: int) -> None:
    """Run one cell, skipping work whose results are already on disk."""
    _, regime = CELLS[cell]
    if regime == "ps":
        todo = [(sid, y) for sid, y in zip(ids, Y) if not series_file(run, cell, sid).exists()]
        log.info("%s: %d of %d series to run", cell, len(todo), len(ids))
        Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(run_series)(cell, sid, y, max_evals, series_file(run, cell, sid)) for sid, y in todo
        )
    else:
        out = cell_dir(run, cell) / "global.json"
        if out.exists():
            log.info("%s: already done", cell)
            return
        run_global(cell, ids, Y, max_evals, out, CHECKPOINT_DIR / run / f"{cell}_trials.pkl", n_threads=n_jobs)


def load_cell(run: str, cell: str, ids: list[str]) -> dict | None:
    """Forecasts (n_series x HORIZON) and run metadata of a finished cell, or None if incomplete."""
    _, regime = CELLS[cell]
    if regime == "ps":
        files = [series_file(run, cell, sid) for sid in ids]
        if not all(f.exists() for f in files):
            return None
        res = [json.loads(f.read_text()) for f in files]
        return {
            "forecast": np.array([r["forecast"] for r in res]),
            "top_val_rmsse": np.array([r["top_val_rmsse"][0] for r in res]),
            "max_evals": res[0]["max_evals"],
            "seconds": sum(r["seconds"] for r in res),
            "top_configs": [r["top_configs"] for r in res],
        }
    f = cell_dir(run, cell) / "global.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    assert r["series_ids"] == ids, "global run was made on a different sample"
    return {
        "forecast": np.array(r["forecast"]),
        "top_val_rmsse": np.full(len(ids), r["top_val_rmsse"][0]),
        "max_evals": r["max_evals"],
        "seconds": r["seconds"],
        "top_configs": r["top_configs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="full", help="name of the run (results/replication/runs/<run>)")
    parser.add_argument("--cells", nargs="+", default=["all"], help=f"cells to run: all or any of {list(CELLS)}")
    parser.add_argument("--max-evals", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=os.cpu_count())
    parser.add_argument("--n-series", type=int, default=None, help="use only the first n series (smoke tests)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ids, Y = load_panel()
    if args.n_series:
        ids, Y = ids[: args.n_series], Y[: args.n_series]
    cells = list(CELLS) if args.cells == ["all"] else args.cells
    for cell in cells:
        t0 = time.perf_counter()
        run_cell(cell, ids, Y, args.run, args.max_evals, args.n_jobs)
        log.info("%s finished in %.1f min", cell, (time.perf_counter() - t0) / 60)


if __name__ == "__main__":
    main()
