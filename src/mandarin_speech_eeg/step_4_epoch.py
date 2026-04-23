"""Step 4: build manifest-filtered EEG epochs without decoding."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .preprocessing import preprocess_session
from .step_common import (
    BATCH_ROOT,
    DATA_ROOT,
    build_step_config,
    default_bdf_path,
    load_marker_table,
    print_json,
    resolve_session_dir,
    write_json,
)


def run_step(
    *,
    session_dir: str | Path | None = None,
    group: str = "Production_Perception",
    subject: str = "p06",
    task: str = "production",
    bdf_path: str | Path | None = None,
    marker_csv: str | Path | None = None,
    manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    use_cache: bool = True,
    legacy: bool = False,
) -> dict:
    session = None if legacy and session_dir is None else resolve_session_dir(session_dir=session_dir, group=group, subject=subject)
    bdf = Path(bdf_path).resolve() if bdf_path else (
        DATA_ROOT / group / f"{subject}.bdf"
        if legacy else default_bdf_path(session, subject)
    )
    manifest = None
    manifest_df = None
    if not legacy:
        manifest = Path(manifest_path).resolve() if manifest_path else session / "analysis_selection" / "trial_manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"Selection manifest not found: {manifest}")

    config = build_step_config(task=task, group=group)
    config.preprocessing.use_epoch_cache = use_cache

    marker_df = load_marker_table(config, task=task, marker_csv=marker_csv)
    if manifest is not None:
        manifest_df = pd.read_csv(manifest)

    print(f"BDF: {bdf}", flush=True)
    if manifest_df is not None:
        print(f"Manifest rows={len(manifest_df)} keep={int(manifest_df['keep_trial'].sum())}", flush=True)
    else:
        print("Selection skipped: legacy dataset", flush=True)

    result = preprocess_session(
        config,
        bdf,
        marker_df,
        subject,
        trial_manifest_df=manifest_df,
    )
    epochs = result.epochs
    metadata = epochs.metadata.reset_index(drop=True).copy()

    out_dir = Path(output_dir).resolve() if output_dir else (
        BATCH_ROOT / "results" / group / task / subject / "epoch_only"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs_path = out_dir / f"{subject}_selection_filtered-epo.fif"
    metadata_path = out_dir / f"{subject}_selection_filtered_metadata.csv"
    summary_path = out_dir / f"{subject}_epoch_summary.json"

    epochs.save(str(epochs_path), overwrite=True, verbose=False)
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    summary = {
        "step": "epoch",
        "session": subject,
        "task": task,
        "group": group,
        "bdf_path": str(bdf),
        "manifest_path": None if manifest is None else str(manifest),
        "n_manifest_trials": None if manifest_df is None else int(len(manifest_df)),
        "n_manifest_keep": None if manifest_df is None else int(manifest_df["keep_trial"].sum()),
        "n_epochs_after_preprocessing": int(len(epochs)),
        "n_metadata_rows": int(len(metadata)),
        "cache_used": bool(result.cache_used),
        "selection_log": result.log,
        "epochs_path": str(epochs_path),
        "metadata_path": str(metadata_path),
        "summary_path": str(summary_path),
    }
    write_json(summary_path, summary)
    print_json(summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build selection-filtered epochs only.")
    parser.add_argument("--session-dir", default=None, help="Explicit modern session directory.")
    parser.add_argument("--group", default="Production_Perception")
    parser.add_argument("--subject", default="p06")
    parser.add_argument("--task", default="production", choices=["production", "perception"])
    parser.add_argument("--bdf", default=None, help="Explicit BDF path.")
    parser.add_argument("--marker-csv", default=None)
    parser.add_argument("--manifest", default=None, help="Explicit trial_manifest.csv path.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-cache", action="store_true", help="Recompute instead of reading epoch cache.")
    parser.add_argument("--legacy", action="store_true", help="Run BDF-only legacy epoching without a manifest.")
    args = parser.parse_args(argv)

    run_step(
        session_dir=args.session_dir,
        group=args.group,
        subject=args.subject,
        task=args.task,
        bdf_path=args.bdf,
        marker_csv=args.marker_csv,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        use_cache=not args.no_cache,
        legacy=args.legacy,
    )


if __name__ == "__main__":
    main()
