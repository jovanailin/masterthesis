"""Fetch the output of the kernel of a job and copy its small result files into the repository.

    python kaggle_jobs/global_xgb/fetch.py <job> [--no-download]

The whole output (forecasts and models included) goes to data/processed/global_xgb/<job>/, which is not
tracked by git; best.json, trials.json and seeds.json of each variant are copied to results/global_xgb/.
The folder is emptied first, because `kaggle kernels output` does not overwrite existing files.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(job: str, download: bool = True) -> None:
    user = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
    out = ROOT / "data" / "processed" / "global_xgb" / job
    if download:
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        kernel = f"{user}/m5-global-xgb-{job.replace('_', '-')}"
        run = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", kernel, "-p", str(out)],
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stderr[-2000:]
    for best in sorted(out.glob("*/best.json")):
        target = ROOT / "results" / "global_xgb" / best.parent.name
        target.mkdir(parents=True, exist_ok=True)
        for name in ["best.json", "trials.json", "seeds.json"]:
            if (best.parent / name).exists():
                shutil.copy(best.parent / name, target / name)
        print(best.parent.name, "->", target.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1], "--no-download" not in sys.argv)
