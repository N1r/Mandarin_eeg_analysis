"""Run the production batch analysis."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from run_batch_analysis import run


if __name__ == "__main__":
    run(default_task="production")
