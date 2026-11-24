"""Push one job of job.py as the private Kaggle GPU kernel <user>/m5-global-xgb-<job>.

    python kaggle_jobs/upload_src.py "message"      # first, if src/ has changed
    python kaggle_jobs/global_xgb/push.py mse

Follow it with   python -m kaggle kernels status <user>/m5-global-xgb-<job>
and fetch it with python -m kaggle kernels output <user>/m5-global-xgb-<job> -p <folder>
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL_SOURCES = {"final": ["m5-global-xgb-tweedie"], "extra": ["m5-global-xgb-tweedie"],
                  "store_cat": ["m5-global-xgb-tweedie"], "extra_id": ["m5-global-xgb-final"],
                  "backtest": ["m5-global-xgb-mse", "m5-global-xgb-poisson", "m5-global-xgb-tweedie", "m5-global-xgb-final",
                               "m5-global-xgb-extra-id"],
                  "backtest_hier": ["m5-global-xgb-hier"], "hier_seeds12": ["m5-global-xgb-hier"],
                  "hier_seeds34": ["m5-global-xgb-hier"]}  # earlier kernels whose outputs the job reads


def main(job: str, here: Path = HERE, prefix: str = "m5-global-xgb", sources: dict = KERNEL_SOURCES,
         gpu: bool = True) -> None:
    """Push job.py of the folder `here` (kaggle_jobs/weekly_xgb/push.py calls this with its own folder)."""
    user = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
    code = (here / "job.py").read_text(encoding="utf-8")
    assert code.count('"@JOB@"') == 1 and f'"{job}":' in code, f"unknown job {job}"
    slug = f"{prefix}-{job.replace('_', '-')}"
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": "job.py", "language": "python",
            "kernel_type": "script", "is_private": "true", "enable_gpu": str(gpu).lower(), "enable_tpu": "false",
            "enable_internet": "true", "machine_shape": "", "dataset_sources": [f"{user}/m5-thesis-src"],
            "competition_sources": ["m5-forecasting-accuracy"],
            "kernel_sources": [f"{user}/{k}" for k in sources.get(job, [])], "model_sources": []}
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "job.py").write_text(code.replace('"@JOB@"', f'"{job}"'), encoding="utf-8", newline="\n")
        (Path(tmp) / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        out = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", tmp, "-t", "43200"],
                             capture_output=True, text=True)
    print(out.stdout, out.stderr, sep="\n")
    sys.exit(out.returncode)


if __name__ == "__main__":
    main(sys.argv[1])
