"""Tuning and refit of the weekly global XGBoost variants on a Kaggle GPU (the logic is in src/weekly_xgb.py).

One kernel per job, pushed with  python kaggle_jobs/weekly_xgb/push.py <job>  and fetched with
python kaggle_jobs/weekly_xgb/fetch.py <job>  (outputs in data/processed/weekly_xgb/, results/weekly_xgb/).

Local check on 305 series and the CPU:  WEEKLY_XGB_JOB=smoke_local python kaggle_jobs/weekly_xgb/job.py
"""
import os
import subprocess
import sys
from pathlib import Path

JOBS = {
    # the main weekly model: L+E+S+ID, with the Tweedie loss of the daily main model and with MSE (weekly sales
    # have far fewer zeros); every week forecast one week ahead from the Sunday before it
    "main": {"variants": [["LESID", "tweedie"], ["LESID", "mse"]], "budget_hours": 10},
    # the same on the CPU (4 cores), one loss per kernel, for weeks with too little GPU quota; the budget leaves
    # room within the 12 hours of a kernel for a trial that runs past it and for the slower refit
    "main_tweedie_cpu": {"variants": [["LESID", "tweedie"]], "budget_hours": 9, "device": "cpu"},
    "main_mse_cpu": {"variants": [["LESID", "mse"]], "budget_hours": 9, "device": "cpu"},
    # the global model that sees what the classical methods see: the past weekly sales only
    "l_mse_cpu": {"variants": [["L", "mse"]], "budget_hours": 9, "device": "cpu"},
    # lags with prices and the calendar: the inputs that ARIMA with regressors (weekly_arimax.py) can use
    "le_mse_cpu": {"variants": [["LE", "mse"]], "budget_hours": 9, "device": "cpu"},
    # the tuned main model with four more seeds, and on three earlier 4-week windows (origins 27.03., 28.02. and
    # 31.01.2016, the windows of the daily backtest)
    "extra_cpu": {"models": ["LESID_mse"], "seeds": [1, 2, 3, 4], "origins": [269, 265, 261], "device": "cpu"},
    "backtest_l_cpu": {"models": ["L_mse"], "origins": [269, 265, 261], "device": "cpu"},
    "extra_local": {"models": ["LESID_mse"], "seeds": [1], "origins": [269], "device": "cpu", "series_step": 100},
    # quick checks of the code: a few short trials on 305 series
    "smoke": {"variants": [["LESID", "tweedie"], ["L", "mse"]], "budget_hours": 0.5, "max_evals": 2, "max_rounds": 30,
              "patience": 5, "series_step": 100, "device": "cpu"},
    "smoke_local": {"variants": [["LESID", "tweedie"], ["L", "mse"]], "budget_hours": 0.5, "max_evals": 2,
                    "max_rounds": 30, "patience": 5, "series_step": 100, "device": "cpu"},
}
JOB = os.environ.get("WEEKLY_XGB_JOB", "@JOB@")  # push.py writes the job name here

if Path("/kaggle/input").exists():
    import glob

    WORK = Path("/kaggle/working")
    pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xgboost==2.1.1", "hyperopt==0.2.7",
                          "setuptools<81"], capture_output=True, text=True)
    print("pip install:", "ok" if pip.returncode == 0 else pip.stderr[-2000:], flush=True)

    def _dir_of(name):
        hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
        assert hits, f"{name} not found under /kaggle/input"
        return str(Path(hits[0]).parent)

    os.environ["M5_KAGGLE_DIR"] = _dir_of("sales_train_evaluation.csv")
    sys.path.insert(0, _dir_of("weekly_xgb.py"))
    TUNED = "/kaggle/input/**/{}/best.json"  # outputs of the earlier kernels (SOURCES in push.py)
else:
    TUNED = str(Path(__file__).resolve().parents[2] / "data" / "processed" / "weekly_xgb" / "main*" / "{}" / "best.json")
    WORK = Path(__file__).resolve().parent / "out" / JOB
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
WORK.mkdir(parents=True, exist_ok=True)

import weekly_xgb  # noqa: E402

if __name__ == "__main__":
    job = JOBS[JOB]
    print("job", JOB, job, flush=True)
    if "models" in job:
        import glob as _glob

        job["model_dirs"] = {}
        for name in job["models"]:
            hits = sorted(_glob.glob(TUNED.format(name), recursive=True))
            assert hits, f"the tuned variant {name} was not found"
            job["model_dirs"][name] = str(Path(hits[0]).parent)
        weekly_xgb.run_extra(job, WORK)
    else:
        weekly_xgb.run(job, WORK)
