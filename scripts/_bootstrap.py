"""Put the project root (parent of src/) on sys.path so `import src.*` works
regardless of the current working directory."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_ROOT = ROOT / "dataset"
CKPT_ROOT = ROOT / "checkpoints"
RESULTS_ROOT = ROOT / "results"
