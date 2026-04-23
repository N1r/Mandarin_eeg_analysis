"""Step 2: build trial selection manifest from formal trials and ASR/onset outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .step_common import force_single_worker_environment, print_json, resolve_session_dir
from .trial_selection import build_trial_manifest


def run_step(
    *,
    session_dir: str | Path | None = None,
    group: str = "Production_Perception",
    subject: str = "p06",
    run_missing: bool = False,
    force: bool = True,
) -> dict:
    force_single_worker_environment()
    session = resolve_session_dir(session_dir=session_dir, group=group, subject=subject)
    result = build_trial_manifest(
        session,
        asr_policy="pinyin_fuzzy",
        run_missing=run_missing,
        force=force,
    )
    summary = {
        "step": "selection",
        "session_dir": str(session),
        "manifest_path": str(result.manifest_path),
        "summary_path": str(result.summary_path),
        **result.summary,
    }
    print_json(summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build selection manifest only.")
    parser.add_argument("--session-dir", default=None, help="Explicit modern session directory.")
    parser.add_argument("--group", default="Production_Perception")
    parser.add_argument("--subject", default="p06")
    parser.add_argument("--run-missing", action="store_true", help="Allow missing ASR/onset outputs to be generated.")
    parser.add_argument("--no-force", action="store_true", help="Reuse existing manifest if present.")
    args = parser.parse_args(argv)

    run_step(
        session_dir=args.session_dir,
        group=args.group,
        subject=args.subject,
        run_missing=args.run_missing,
        force=not args.no_force,
    )


if __name__ == "__main__":
    main()
