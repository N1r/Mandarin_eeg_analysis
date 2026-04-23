"""Step 3: run acoustic onset analysis for one modern production session."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .onset import run_onset_analysis
from .step_common import force_single_worker_environment, print_json, resolve_session_dir


def run_step(
    *,
    session_dir: str | Path | None = None,
    group: str = "Production_Perception",
    subject: str = "p06",
    clear_output: bool = True,
) -> dict:
    force_single_worker_environment()
    session = resolve_session_dir(session_dir=session_dir, group=group, subject=subject)
    result = run_onset_analysis(session, clear_output=clear_output)

    trial_path = result["output_dir"] / "onset_trial_level.csv"
    trials = pd.read_csv(trial_path)
    detected = trials[pd.to_numeric(trials["detected_onset"], errors="coerce").fillna(0).astype(bool)]

    summary = {
        "step": "onset",
        "session_dir": str(session),
        "output_dir": str(result["output_dir"]),
        "trial_csv": str(trial_path),
        "report_path": str(result["report_path"]),
        "n_trials": int(len(trials)),
        "n_detected": int(len(detected)),
        "n_missed": int(len(trials) - len(detected)),
        "mean_latency_ms": None if detected.empty else float(detected["detected_latency_ms"].mean()),
        "median_latency_ms": None if detected.empty else float(detected["detected_latency_ms"].median()),
    }
    print_json(summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run onset analysis only for one session.")
    parser.add_argument("--session-dir", default=None, help="Explicit modern session directory.")
    parser.add_argument("--group", default="Production_Perception")
    parser.add_argument("--subject", default="p06")
    parser.add_argument("--keep-output", action="store_true", help="Keep existing onset output contents.")
    args = parser.parse_args(argv)

    run_step(
        session_dir=args.session_dir,
        group=args.group,
        subject=args.subject,
        clear_output=not args.keep_output,
    )


if __name__ == "__main__":
    main()
