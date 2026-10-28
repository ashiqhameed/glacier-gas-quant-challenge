"""Run the whole pipeline in order (needs the competition files in data/)."""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
STEPS = sorted(p for p in HERE.glob("[0-9][0-9]_*.py"))

for step in STEPS:
    print(f"\n=== {step.name} " + "=" * (60 - len(step.name)), flush=True)
    start = time.time()
    subprocess.run([sys.executable, str(step)], check=True)
    print(f"--- done in {time.time() - start:.0f}s", flush=True)
