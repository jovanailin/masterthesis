"""Upload src/*.py to the private Kaggle dataset <user>/m5-thesis-src, used by the kernels in kaggle_jobs/.

Creates the dataset on the first run and adds a new version afterwards. The M5 data themselves are not
uploaded: the kernels read them from the competition (competition_sources in kernel-metadata.json).

    python kaggle_jobs/upload_src.py "message"
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "m5-thesis-src"


def kaggle(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True)


def main(message: str) -> None:
    user = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        for f in sorted((ROOT / "src").glob("*.py")):
            shutil.copy(f, tmp)
        meta = {"title": SLUG, "id": f"{user}/{SLUG}", "licenses": [{"name": "CC0-1.0"}],
                "description": f"Code of the thesis (src/ at git {commit}, plus local changes if any)."}
        (Path(tmp) / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
        exists = kaggle("datasets", "status", f"{user}/{SLUG}").returncode == 0
        args = ["version", "-p", tmp, "-m", f"{message} ({commit})"] if exists else ["create", "-p", tmp]
        out = kaggle("datasets", *args)
    print(out.stdout, out.stderr, sep="\n")
    sys.exit(out.returncode)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "update")
