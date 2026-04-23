"""Shared helpers for single-purpose workflow step scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DATA_ROOT, AnalysisConfig, make_config


ANALYSIS_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ANALYSIS_ROOT.parents[1]
BATCH_ROOT = ANALYSIS_ROOT.parent / "results" / "batch_analysis"


def force_single_worker_environment() -> None:
    """Keep every step single-worker unless a caller explicitly changes code."""

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "LOKY_MAX_CPU_COUNT",
    ):
        os.environ[name] = "1"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ.setdefault("MPLBACKEND", "Agg")


def subject_to_session_subject(subject: str) -> str:
    """Convert p06 / sub-6 / 6 into the modern session folder subject name."""

    value = str(subject).strip().lower()
    if value.startswith("sub-"):
        return value
    if value.startswith("p"):
        value = value[1:]
    return f"sub-{int(value)}"


def resolve_session_dir(
    *,
    session_dir: str | Path | None,
    group: str,
    subject: str,
) -> Path:
    """Resolve a modern session directory from explicit path or group/subject."""

    if session_dir is not None:
        path = Path(session_dir).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Session directory not found: {path}")
        return path

    subject_dir = DATA_ROOT / group / subject_to_session_subject(subject)
    candidates = sorted(path for path in subject_dir.glob("ses-*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No ses-* directory found under {subject_dir}")
    return candidates[-1].resolve()


def default_bdf_path(session_dir: Path, subject: str) -> Path:
    bdf_dir = session_dir / "eeg_data"
    candidates = sorted(bdf_dir.glob("*.bdf"))
    if not candidates:
        raise FileNotFoundError(f"No BDF file found under {bdf_dir}")

    subject_token = subject.lower()
    for candidate in candidates:
        if subject_token in candidate.stem.lower():
            return candidate.resolve()
    return candidates[0].resolve()


def build_step_config(*, task: str, group: str) -> AnalysisConfig:
    """Create an AnalysisConfig for single-step scripts, pinned to one worker."""

    force_single_worker_environment()
    config = make_config(task=task)
    config.statistics.enabled = False
    config.statistics.n_jobs = 1

    paths = config.paths.with_roots(
        results_dir=BATCH_ROOT / "results" / group / task,
        figures_dir=BATCH_ROOT / "figures" / group / task,
        cache_dir=BATCH_ROOT / "cache" / group / task,
    )
    config = config.with_paths(paths)
    config.paths.ensure_directories()
    return config


def load_marker_table(config: AnalysisConfig, *, task: str, marker_csv: str | Path | None = None) -> pd.DataFrame:
    marker_path = Path(marker_csv) if marker_csv else config.paths.marker_csv
    marker_df = pd.read_csv(marker_path)
    if task == "perception":
        marker_df = marker_df.copy()
        marker_df["marker"] = marker_df["marker"] + 100
    return marker_df


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2, ensure_ascii=False), encoding="utf-8")


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(_json_safe(data), indent=2, ensure_ascii=False), flush=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value
