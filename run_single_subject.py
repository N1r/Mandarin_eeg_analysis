"""Run one subject through the package CLI."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from mandarin_speech_eeg.cli import analyze_subject_cli


if __name__ == "__main__":
    analyze_subject_cli()
