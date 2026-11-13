"""Speed test of the global XGBoost on a Kaggle GPU.

Runs as a Kaggle kernel (kernel-metadata.json) with the M5 competition data and the thesis code from the
private dataset m5-thesis-src (kaggle_jobs/upload_src.py). One model forecasts the 7 horizons and has the
horizon as an input; it is trained on weekly origins with K horizons drawn at random per origin. For
K = 2 (about 13M rows) the script measures

- the time and memory needed to build the inputs (h+L+E+S+ID, all 30,490 series),
- the GPU training time of the three losses with early stopping on the validation window,
- the time per boosting round at three tree depths,

and for K = 3 (about 20M rows) whether the inputs still fit in the memory of the GPU. The results go to
speed_test.json.

The first version of this test timed one model per horizon of a 28-day forecast (results in
results/kaggle/speed_test/v1_28_days/) and showed that daily origins (46M rows) do not fit in the GPU memory.

Local check on 305 series and the CPU:  SPEED_TEST_SMOKE=1 python kaggle_jobs/speed_test/speed_test.py
"""
import glob
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ON_KAGGLE = Path("/kaggle/input").exists()
SMOKE = os.environ.get("SPEED_TEST_SMOKE") == "1"
XGB_VERSION = "2.1.1"  # as in requirements.txt

if ON_KAGGLE:
    WORK = Path("/kaggle/working")
    pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"xgboost=={XGB_VERSION}"],
                         capture_output=True, text=True)
    PIP_OK = pip.returncode == 0
    if not PIP_OK:
        print("pip install failed (no internet?); using the preinstalled xgboost\n", pip.stderr[-2000:], flush=True)

    def _dir_of(name):
        hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
        assert hits, f"{name} not found under /kaggle/input"
        return str(Path(hits[0]).parent)

    os.environ["M5_KAGGLE_DIR"] = _dir_of("sales_train_evaluation.csv")
    os.environ["M5_PROCESSED_DIR"] = str(WORK / "processed")
    sys.path.insert(0, _dir_of("features.py"))
else:
    WORK = Path(__file__).resolve().parent / "out"
    PIP_OK = None
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
WORK.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import psutil
import xgboost as xgb

import features as F

DEVICE = "cpu" if SMOKE else "cuda"
GROUPS = ("L", "E", "S", "ID")
LOSSES = {
    "mse": {"objective": "reg:squarederror"},
    "poisson": {"objective": "count:poisson"},
    "tweedie": {"objective": "reg:tweedie", "tweedie_variance_power": 1.2},
}
BASE = dict(tree_method="hist", device=DEVICE, max_bin=256, learning_rate=0.05, max_depth=8, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=1, eval_metric="rmse", seed=0)
MAX_ROUNDS, PATIENCE = (30, 5) if SMOKE else (2000, 50)
FIXED_ROUNDS = 10 if SMOKE else 200  # for the time per round at each depth
DEPTHS = (4, 8, 12)

T0 = time.perf_counter()
LOG = {"steps": []}


def ram_gb():
    return psutil.Process().memory_info().rss / 1e9


def peak_ram_gb():
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB on Linux
    except ImportError:
        return float("nan")


def record(step, **info):
    info = {"step": step, **info, "ram_gb": round(ram_gb(), 2), "peak_ram_gb": round(peak_ram_gb(), 2),
            "elapsed_min": round((time.perf_counter() - T0) / 60, 2)}
    LOG["steps"].append(info)
    print(json.dumps(info, default=float), flush=True)
    (WORK / "speed_test.json").write_text(json.dumps(LOG, indent=1, default=float))


def environment():
    env = {"python": platform.python_version(), "xgboost": xgb.__version__, "numpy": np.__version__,
           "pandas": pd.__version__, "pip_install_ok": PIP_OK, "cpus": os.cpu_count(),
           "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1), "smoke": SMOKE, "device": DEVICE}
    try:
        q = "--query-gpu=name,memory.total,driver_version"
        env["gpus"] = subprocess.run(["nvidia-smi", q, "--format=csv,noheader"], capture_output=True,
                                     text=True).stdout.strip().splitlines()
    except FileNotFoundError:
        env["gpus"] = []
    LOG["environment"] = env
    record("environment", **env)


def subset(p, rows):
    return F.Panel(ids=p.ids.iloc[rows].reset_index(drop=True), hier=p.hier.iloc[rows].reset_index(drop=True),
                   y=p.y[rows], first_sale=p.first_sale[rows], cal=p.cal, snap=p.snap, state=p.state[rows],
                   week=p.week, price={k: v[rows] for k, v in p.price.items()})


def validation(panel):
    """Rows of the forecast origin d_1913 for the 7 horizons, with their targets d_1914 - d_1920."""
    s, t = F.forecast_origins(panel, "valid")
    tab = F.origin_table(panel, s, t)
    rows, h = F.horizon_rows(len(tab))
    X, _, _ = F.design(panel, tab, h, GROUPS, target=False, rows=rows)
    y = panel.y[s[rows], t[rows].astype(np.int64) + h]
    ok = ~np.isnan(y)
    return X[ok].reset_index(drop=True), y[ok]


def train_all(dtrain, dval, label, full=True):
    """Early-stopped training of each loss, then fixed-round timings at each depth (Poisson)."""
    for loss, obj in LOSSES.items() if full else []:
        try:
            t = time.perf_counter()
            bst = xgb.train(BASE | obj, dtrain, MAX_ROUNDS, evals=[(dval, "valid")],
                            early_stopping_rounds=PATIENCE, verbose_eval=False)
            sec = time.perf_counter() - t
            rounds = bst.num_boosted_rounds()
            record(f"{label} train {loss}", seconds=round(sec, 1), rounds=rounds, best_iteration=bst.best_iteration,
                   best_valid_rmse=round(bst.best_score, 4), sec_per_round=round(sec / rounds, 4))
        except Exception as e:  # noqa: BLE001 - log and continue with the other runs
            record(f"{label} train {loss}", error=repr(e)[:500])
    for depth in DEPTHS if full else DEPTHS[1:2]:
        try:
            t = time.perf_counter()
            xgb.train(BASE | LOSSES["poisson"] | {"max_depth": depth}, dtrain, FIXED_ROUNDS, verbose_eval=False)
            sec = time.perf_counter() - t
            record(f"{label} fixed depth {depth}", seconds=round(sec, 1), rounds=FIXED_ROUNDS,
                   sec_per_round=round(sec / FIXED_ROUNDS, 4))
        except Exception as e:  # noqa: BLE001
            record(f"{label} fixed depth {depth}", error=repr(e)[:500])


def main():
    environment()
    t = time.perf_counter()
    panel = F.build_panel()
    if SMOKE:
        panel = subset(panel, np.arange(0, len(panel.ids), 100))
    record("build panel", seconds=round(time.perf_counter() - t, 1), series=len(panel.ids))

    t = time.perf_counter()
    s, o = F.training_origins(panel, "valid", stride=7)
    tab = F.origin_table(panel, s, o)
    Xv, yv = validation(panel)
    record("origin table", seconds=round(time.perf_counter() - t, 1), origins=len(tab), validation_rows=len(yv))

    for k in (2, 3):  # horizons drawn per origin
        label = f"K={k}"
        try:
            t = time.perf_counter()
            rows, h = F.horizon_rows(len(tab), per_origin=k)
            X, y, _ = F.design(panel, tab, h, GROUPS, stage="valid", rows=rows)
            build = time.perf_counter() - t
            t = time.perf_counter()
            dtrain = xgb.QuantileDMatrix(X, y, max_bin=BASE["max_bin"], enable_categorical=True)
            dval = xgb.QuantileDMatrix(Xv, yv, ref=dtrain, enable_categorical=True)
            record(f"{label} inputs", build_seconds=round(build, 1), dmatrix_seconds=round(time.perf_counter() - t, 1),
                   rows=len(y), columns=X.shape[1], table_gb=round(X.memory_usage(deep=True).sum() / 1e9, 2))
            del X
        except Exception as e:  # noqa: BLE001
            record(f"{label} inputs", error=repr(e)[:500])
            continue
        train_all(dtrain, dval, label, full=k == 2)
        del dtrain, dval
    record("done")


if __name__ == "__main__":
    main()
