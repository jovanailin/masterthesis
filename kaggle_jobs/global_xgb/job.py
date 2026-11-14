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
    # L+E+S+ID with the best loss, then the best of the ten variants with four more seeds
    "final": {"variants": [["LESID", "tweedie"]], "budget_hours": 8, "seeds": [1, 2, 3, 4],
              "seed_candidates": ["LES_tweedie"]},
    # models per store and per demand class, and the recursive strategy, with the hyperparameters of L+E+S Tweedie
    "extra": {"source": "LES_tweedie", "schemes": ["store", "class"], "recursive": True},
    "extra_local": {"source": "LES_tweedie", "schemes": ["store", "class"], "recursive": True, "max_rounds": 30,
                    "patience": 5, "series_step": 100, "device": "cpu"},
    # quick checks of the code: a few short trials
    "smoke": {"variants": [["L", "mse"], ["LESID", "tweedie"]], "budget_hours": 0.5, "max_evals": 2, "max_rounds": 30,
              "patience": 5},
    "smoke_local": {"variants": [["L", "mse"], ["LESID", "tweedie"]], "budget_hours": 0.5, "max_evals": 3,
                    "max_rounds": 30, "patience": 5, "series_step": 100, "device": "cpu", "seeds": [1, 2]},
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
    TUNED = "/kaggle/input/**/{}/best.json"  # outputs of the earlier kernels (kernel_sources in push.py)
else:
    ROOT = Path(__file__).resolve().parents[2]
    WORK = Path(__file__).resolve().parent / "out" / JOB
    sys.path.insert(0, str(ROOT / "src"))
    TUNED = str(ROOT / "data" / "processed" / "global_xgb" / "*" / "{}" / "best.json")
WORK.mkdir(parents=True, exist_ok=True)

import global_xgb  # noqa: E402

if __name__ == "__main__":
    job = JOBS[JOB]
    job["seed_candidate_dirs"] = []
    for name in job.get("seed_candidates", []):
        hits = sorted(glob.glob(TUNED.format(name), recursive=True))
        assert hits, f"the tuned variant {name} was not found"
        job["seed_candidate_dirs"].append(str(Path(hits[0]).parent))
    print("job", JOB, job, flush=True)
    if "source" in job:
        hits = sorted(glob.glob(TUNED.format(job["source"]), recursive=True))
        assert hits, f"the tuned variant {job['source']} was not found"
        job["source_dir"] = str(Path(hits[0]).parent)
        global_xgb.run_extra(job, WORK)
    else:
        global_xgb.run(job, WORK)
