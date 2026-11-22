"""Tuning and refit of the global XGBoost variants on a Kaggle GPU (the logic is in src/global_xgb.py).

One kernel per job, pushed with  python kaggle_jobs/global_xgb/push.py <job>.  The nine variants of the
3 x 3 design are split by loss into three kernels, because a kernel may run for 12 hours at most.

Local check on 305 series and the CPU:  GLOBAL_XGB_JOB=smoke_local python kaggle_jobs/global_xgb/job.py
"""
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUTS = ["L", "LE", "LES"]
BACKTEST_MODELS = ["L_tweedie", "LE_tweedie", "LES_tweedie", "LESID_tweedie", "LES_mse", "LES_poisson",
                   "LESID_tweedie_per_store", "LESID_tweedie_per_class", "LESID_tweedie_recursive"]
JOBS = {
    "mse": {"variants": [[i, "mse"] for i in INPUTS], "budget_hours": 10.5},
    "poisson": {"variants": [[i, "poisson"] for i in INPUTS], "budget_hours": 10.5},
    "tweedie": {"variants": [[i, "tweedie"] for i in INPUTS], "budget_hours": 10.5},
    # L+E+S+ID with the best loss, then the best of the ten variants with four more seeds
    "final": {"variants": [["LESID", "tweedie"]], "budget_hours": 8, "seeds": [1, 2, 3, 4],
              "seed_candidates": ["LES_tweedie"]},
    # L+E+S+ID with the sales of higher levels of the hierarchy (group H), tuned as the others, with four more seeds
    # (the kernel ran out of memory on the third test refit, so the test models of the seeds have kernels of their own)
    "hier": {"variants": [["LESIDH", "tweedie"]], "budget_hours": 8, "seeds": [1, 2, 3, 4], "seed_refits": False},
    "hier_seeds12": {"refit_of": "LESIDH_tweedie", "seeds": [1, 2]},
    "hier_seeds34": {"refit_of": "LESIDH_tweedie", "seeds": [3, 4]},
    "hier_seeds_local": {"refit_of": "LESIDH_tweedie", "seeds": [1], "device": "cpu", "series_step": 100},
    "hier_local": {"variants": [["LESIDH", "tweedie"]], "budget_hours": 0.5, "max_evals": 2, "max_rounds": 30,
                   "patience": 5, "series_step": 100, "device": "cpu", "seeds": [1]},
    # models per store and per demand class, and the recursive strategy, with the hyperparameters of L+E+S Tweedie
    "extra": {"source": "LES_tweedie", "schemes": ["store", "class"], "recursive": True},
    "store_cat": {"source": "LES_tweedie", "schemes": ["store_cat"]},  # 30 models, one per store and category
    # the same with the hyperparameters of the best variant, L+E+S+ID Tweedie
    "extra_id": {"source": "LESID_tweedie", "schemes": ["store", "store_cat", "class"], "recursive": True},
    "extra_id_local": {"source": "LESID_tweedie", "schemes": ["store", "store_cat", "class"], "recursive": True,
                       "max_rounds": 30, "patience": 5, "series_step": 100, "device": "cpu"},
    "extra_local": {"source": "LES_tweedie", "schemes": ["store", "store_cat", "class"], "recursive": True, "max_rounds": 30,
                    "patience": 5, "series_step": 100, "device": "cpu"},
    # full runs on the local CPU, for when the Kaggle GPU queue does not move
    "store_cat_cpu": {"source": "LES_tweedie", "schemes": ["store_cat"], "device": "cpu"},
    "extra_id_cpu": {"source": "LESID_tweedie", "schemes": ["store", "store_cat", "class"], "recursive": True,
                     "device": "cpu"},
    # earlier 28-day windows (ending d_1913, d_1885, d_1857), the main models retrained with fixed hyperparameters
    "backtest": {"origins": [1885, 1857, 1829], "models": BACKTEST_MODELS},
    "backtest_hier": {"origins": [1885, 1857, 1829], "models": ["LESIDH_tweedie"]},
    "backtest_local": {"origins": [1885], "models": BACKTEST_MODELS, "series_step": 100, "device": "cpu"},
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
    if "origins" in job:
        job["model_dirs"] = {}
        for name in job["models"]:
            hits = sorted(glob.glob(TUNED.format(name), recursive=True))
            if not hits:  # the kernel that tuned it gave no output: best.json uploaded with the code (upload_src.py)
                copies = sorted(glob.glob(f"/kaggle/input/**/tuned_{name}_best.json", recursive=True))
                assert copies, f"the tuned variant {name} was not found"
                (WORK / "tuned" / name).mkdir(parents=True, exist_ok=True)
                shutil.copy(copies[0], WORK / "tuned" / name / "best.json")
                hits = [str(WORK / "tuned" / name / "best.json")]
            job["model_dirs"][name] = str(Path(hits[-1]).parent)
        global_xgb.run_backtest(job, WORK)
    elif "refit_of" in job:
        hits = sorted(glob.glob(TUNED.format(job["refit_of"]), recursive=True))
        assert hits, f"the tuned variant {job['refit_of']} was not found"
        job["source_dir"] = str(Path(hits[0]).parent)
        global_xgb.run_seed_refits(job, WORK)
    elif "source" in job:
        hits = sorted(glob.glob(TUNED.format(job["source"]), recursive=True))
        assert hits, f"the tuned variant {job['source']} was not found"
        job["source_dir"] = str(Path(hits[0]).parent)
        global_xgb.run_extra(job, WORK)
    else:
        global_xgb.run(job, WORK)
