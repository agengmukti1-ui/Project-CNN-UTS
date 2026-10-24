"""Run dataset download, split, training, and evaluation sequentially."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    [sys.executable, "download_dataset.py"],
    [sys.executable, "src/prepare_dataset.py"],
    [sys.executable, "src/train.py"],
    [sys.executable, "src/evaluate.py"],
]
for step in STEPS:
    print("\n>>>", " ".join(step), flush=True)
    result = subprocess.run(step, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
print("\nPipeline selesai. Periksa folder results/ dan models/.")
