"""Tuning and refit of the global XGBoost variants on a Kaggle GPU (the logic is in src/global_xgb.py).

One kernel per job, pushed with  python kaggle_jobs/global_xgb/push.py <job>.  The nine variants of the
3 x 3 design are split by loss into three kernels, because a kernel may run for 12 hours at most.

Local check on 305 series and the CPU:  GLOBAL_XGB_JOB=smoke_local python kaggle_jobs/global_xgb/job.py
"""
import glob
import os
import subprocess
import sys
from pathlib import Path

INPUTS = ["L", "LE", "LES"]
JOBS = {
    "mse": {"variants": [[i, "mse"] for i in INPUTS], "budget_hours": 10.5},
    "poisson": {"variants": [[i, "poisson"] for i in INPUTS], "budget_hours": 10.5},
    "tweedie": {"variants": [[i, "tweedie"] for i in INPUTS], "budget_hours": 10.5},
    # quick checks of the code: a few short trials
    "smoke": {"variants": [["L", "mse"], ["LESID", "tweedie"]], "budget_hours": 0.5, "max_evals": 2, "max_rounds": 30,
              "patience": 5},
    "smoke_local": {"variants": [["L", "mse"], ["LESID", "tweedie"]], "budget_hours": 0.5, "max_evals": 3,
                    "max_rounds": 30, "patience": 5, "series_step": 100, "device": "cpu"},
}
JOB = os.environ.get("GLOBAL_XGB_JOB", "@JOB@")  # push.py writes the job name here

if Path("/kaggle/input").exists():
    WORK = Path("/kaggle/working")
    pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xgboost==2.1.1", "hyperopt==0.2.7",
                          "setuptools<81"], capture_output=True, text=True)
    print("pip install:", "ok" if pip.returncode == 0 else pip.stderr[-2000:], flush=True)

    def _dir_of(name):
        hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
        assert hits, f"{name} not found under /kaggle/input"
        return str(Path(hits[0]).parent)

    os.environ["M5_KAGGLE_DIR"] = _dir_of("sales_train_evaluation.csv")
    sys.path.insert(0, _dir_of("global_xgb.py"))
else:
    WORK = Path(__file__).resolve().parent / "out" / JOB
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
WORK.mkdir(parents=True, exist_ok=True)

import global_xgb  # noqa: E402

if __name__ == "__main__":
    print("job", JOB, JOBS[JOB], flush=True)
    global_xgb.run(JOBS[JOB], WORK)
