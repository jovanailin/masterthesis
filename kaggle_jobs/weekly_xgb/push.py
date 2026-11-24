"""Push one job of job.py as the private Kaggle kernel <user>/m5-weekly-xgb-<job> (see global_xgb/push.py).

    python kaggle_jobs/upload_src.py "message"      # first, if src/ has changed
    python kaggle_jobs/weekly_xgb/push.py main
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("global_push", HERE.parent / "global_xgb" / "push.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

# kernels whose tuned variants a job reads
SOURCES = {"extra_cpu": ["m5-weekly-xgb-main-mse-cpu"], "backtest_l_cpu": ["m5-weekly-xgb-l-mse-cpu"]}

if __name__ == "__main__":
    job = sys.argv[1]
    P.main(job, here=HERE, prefix="m5-weekly-xgb", sources=SOURCES, gpu=not (job.startswith("smoke") or job.endswith("_cpu")))
