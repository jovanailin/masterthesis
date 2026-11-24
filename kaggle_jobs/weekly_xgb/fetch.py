"""Fetch the output of a weekly job into data/processed/weekly_xgb/<job>/ and its small result files into
results/weekly_xgb/ (see global_xgb/fetch.py).

    python kaggle_jobs/weekly_xgb/fetch.py <job> [--no-download]
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("global_fetch", HERE.parent / "global_xgb" / "fetch.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

if __name__ == "__main__":
    P.main(sys.argv[1], "--no-download" not in sys.argv, folder="weekly_xgb", prefix="m5-weekly-xgb")
