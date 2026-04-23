"""Step 1: run Qwen ASR for one modern production session."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .asr import run_asr_analysis
from .step_common import force_single_worker_environment, print_json, resolve_session_dir


def run_step(
    *,
    session_dir: str | Path | None = None,
    group: str = "Production_Perception",
    subject: str = "p06",
    blocks: list[int] | None = None,
) -> dict:
    force_single_worker_environment()
    session = resolve_session_dir(session_dir=session_dir, group=group, subject=subject)

    result = run_asr_analysis(session, blocks=blocks, ensure_onset=False)
    trial_path = result["output_dir"] / "asr_trial_level.csv"
    trials = pd.read_csv(trial_path)

    summary = {
        "step": "asr",
        "session_dir": str(session),
        "output_dir": str(result["output_dir"]),
        "trial_csv": str(trial_path),
        "report_path": str(result["report_path"]),
        "n_trials": int(len(trials)),
        "n_aligned": int((trials["alignment_status"] == "aligned").sum()),
        "n_omission": int((trials["alignment_status"] == "omission").sum()),
        "strict_pinyin_match": int(pd.to_numeric(trials["pinyin_match"], errors="coerce").fillna(0).sum()),
        "word_match": int(pd.to_numeric(trials["word_match"], errors="coerce").fillna(0).sum()),
    }
    print_json(summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run ASR only for one session.")
    parser.add_argument("--session-dir", default=None, help="Explicit modern session directory.")
    parser.add_argument("--group", default="Production_Perception")
    parser.add_argument("--subject", default="p06")
    parser.add_argument("--block", type=int, action="append", help="Optional block(s) to process.")
    args = parser.parse_args(argv)

    run_step(
        session_dir=args.session_dir,
        group=args.group,
        subject=args.subject,
        blocks=args.block,
    )


if __name__ == "__main__":
    main()
