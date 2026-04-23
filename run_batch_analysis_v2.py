"""Batch runner — v2 minimal preprocessing + event-lock & response-lock analysis.

Writes to ``results/batch_analysis_v2/`` and does not touch any output of the
original pipeline. Preprocessing: bandpass 0.5-30 Hz, resample 100 Hz, common-
average reference, no ICA, no AutoReject. For every subject two analyses run in
parallel: event-locked (stimulus onset at t=0) and response-locked (speech onset
at t=0). Response-locking re-slices the event-locked epochs per trial using the
behavioural onset latency — fast but approximate (baseline inherited from the
event-locked window).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_N_JOBS = 1
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_DEFAULT_N_JOBS))
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from mandarin_speech_eeg import (
    AnalysisConfig,
    disable_statistics,
    make_minimal_v2_config,
    make_response_locked_epochs,
    plot_decoding_time_series,
    plot_heterorc_decoding_time_series,
    plot_rsa_time_series,
    preprocess_session,
    run_decoding,
    run_heterorc_decoding,
    run_rsa,
    run_statistics,
    save_statistics,
)
# Lazy-import trial_selection inside _maybe_build_trial_manifest: importing it
# pulls in qwen_asr/torch, which triggers a Windows DLL access violation on
# machines without a working torch. v2 only needs a pre-built manifest CSV.

DATA_ROOT = REPO_ROOT.parent.parent / "Data"
RESULTS_ROOT = REPO_ROOT.parent / "results" / "batch_analysis_v2"
SUBJECT_GROUPS = {
    "Production_only": {"subjects": ["p00", "p01", "p02"], "tasks": ["production"]},
    "Production_Perception": {"subjects": ["p05", "p06", "p07"], "tasks": ["production", "perception"]},
}

RESPONSE_LOCK_TMIN_S = -1.0
RESPONSE_LOCK_TMAX_S = 0.5
RESPONSE_LOCK_BASELINE_S = (-1.0, -0.8)
RESPONSE_SOURCE_TMIN_S = -1.0
RESPONSE_SOURCE_TMAX_S = 2.5


@dataclass(frozen=True)
class BatchTarget:
    group_name: str
    task: str


def _iter_targets(group_name: str | None, task: str | None) -> list[BatchTarget]:
    targets: list[BatchTarget] = []
    for current_group_name, group_info in SUBJECT_GROUPS.items():
        if group_name and group_name != current_group_name:
            continue
        for current_task in group_info["tasks"]:
            if task and task != current_task:
                continue
            targets.append(BatchTarget(group_name=current_group_name, task=current_task))
    return targets


def _build_config(target: BatchTarget, quick: bool, no_stats: bool) -> AnalysisConfig:
    config = make_minimal_v2_config(task=target.task, quick=quick)
    results_root = RESULTS_ROOT / "results"
    figures_root = RESULTS_ROOT / "figures"
    paths = config.paths.with_roots(
        results_dir=results_root / target.group_name / target.task,
        figures_dir=figures_root / target.group_name / target.task,
        cache_dir=RESULTS_ROOT / "cache" / target.group_name / target.task,
    )
    config = config.with_paths(paths)
    config.paths.ensure_directories()
    if no_stats:
        disable_statistics(config)
    return config


def _build_response_source_config(config: AnalysisConfig) -> AnalysisConfig:
    """Clone config with an extended window used only for response-lock slicing."""
    source_config = copy.deepcopy(config)
    source_config.preprocessing.epoch_tmin_s = RESPONSE_SOURCE_TMIN_S
    source_config.preprocessing.epoch_tmax_s = RESPONSE_SOURCE_TMAX_S
    source_config.preprocessing.baseline_window_s = None
    return source_config


def _load_marker_table(config: AnalysisConfig, task: str) -> pd.DataFrame:
    marker_df = pd.read_csv(config.paths.marker_csv)
    if task == "perception":
        marker_df = marker_df.copy()
        marker_df["marker"] = marker_df["marker"] + 100
    return marker_df


def _find_prebuilt_manifest(bdf_path: Path) -> Path | None:
    """Find an existing trial_manifest.csv under Data/ for this subject.

    v2 reuses manifests produced by the original pipeline; it does NOT rebuild
    them, so the runner has no dependency on qwen_asr/torch.
    """
    for manifest in DATA_ROOT.rglob("analysis_selection/trial_manifest.csv"):
        # match by subject number: "p06" -> "sub-6" or "sub-06"
        subject_num = "".join(ch for ch in bdf_path.stem if ch.isdigit())
        if not subject_num:
            continue
        if f"sub-{int(subject_num)}" in manifest.as_posix() or f"sub-{int(subject_num):02d}" in manifest.as_posix():
            return manifest
    return None


def _maybe_build_trial_manifest(config: AnalysisConfig, bdf_path: Path) -> pd.DataFrame | None:
    if config.task != "production":
        return None
    manifest_path = _find_prebuilt_manifest(bdf_path)
    if manifest_path is None:
        print(f"  no prebuilt manifest found for {bdf_path.stem}; running legacy mode")
        return None
    print(f"  using prebuilt manifest: {manifest_path}")
    return pd.read_csv(manifest_path)


def _attach_latency_from_manifest(epochs, manifest_df: pd.DataFrame) -> None:
    """Post-hoc align manifest rows to epoch events by trigger sequence.

    When preprocess_session falls back to legacy mode (e.g. trigger count
    mismatch), epoch metadata lacks ``detected_latency_ms``. We align the
    manifest trigger sequence to the observed epoch triggers by finding the
    best contiguous sub-run, then copy latency + keep_trial into metadata.
    Only kept (keep_trial==1) trials retain a latency; others get NaN.
    """
    md = epochs.metadata
    if md is None or "marker" not in md.columns:
        print("  latency-attach skipped: no marker column in epoch metadata")
        return
    observed = pd.to_numeric(md["marker"], errors="coerce").astype("Int64").to_numpy()
    formal = manifest_df.copy()
    if "global_trial" in formal.columns:
        formal["global_trial"] = pd.to_numeric(formal["global_trial"], errors="coerce")
        formal = formal.sort_values(["global_trial", "block", "trial"], kind="stable")
    formal = formal.reset_index(drop=True)
    trig_col = "trigger" if "trigger" in formal.columns else "marker"
    manifest_trig = pd.to_numeric(formal[trig_col], errors="coerce").astype("Int64").to_numpy()

    n_obs, n_man = len(observed), len(manifest_trig)
    best_offset, best_score = None, -1
    for offset in range(0, max(1, n_man - n_obs + 1)):
        sub = manifest_trig[offset : offset + n_obs]
        if len(sub) != n_obs:
            continue
        score = int(np.sum(sub == observed))
        if score > best_score:
            best_offset, best_score = offset, score
    if best_offset is None or best_score < int(0.95 * n_obs):
        print(f"  latency-attach failed: best match {best_score}/{n_obs} at offset={best_offset}")
        return
    aligned = formal.iloc[best_offset : best_offset + n_obs].reset_index(drop=True)
    new_md = md.reset_index(drop=True).copy()
    if "detected_latency_ms" in aligned.columns:
        new_md["detected_latency_ms"] = pd.to_numeric(
            aligned["detected_latency_ms"], errors="coerce"
        ).to_numpy()
    if "keep_trial" in aligned.columns:
        new_md["keep_trial"] = pd.to_numeric(aligned["keep_trial"], errors="coerce").fillna(0).astype(int).to_numpy()
    epochs.metadata = new_md
    n_lat = int(np.isfinite(new_md.get("detected_latency_ms", pd.Series([])).to_numpy(dtype=float)).sum())
    print(f"  latency-attach ok: offset={best_offset}, triggers matched={best_score}/{n_obs}, latencies={n_lat}")


def _mean_speech_onset_ms(epochs) -> float | None:
    if epochs.metadata is None or "detected_latency_ms" not in epochs.metadata.columns:
        return None
    values = pd.to_numeric(epochs.metadata["detected_latency_ms"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(values.mean())


def _run_lock(
    *,
    config: AnalysisConfig,
    epochs,
    lock_dir: Path,
    lock_figure_dir: Path,
    tag: str,
    analyses: set[str],
) -> None:
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_figure_dir.mkdir(parents=True, exist_ok=True)
    epochs_path = lock_dir / "epochs-epo.fif"
    epochs.save(str(epochs_path), overwrite=True, verbose=False)
    summary = {
        "tag": tag,
        "n_epochs": int(len(epochs)),
        "tmin_s": float(epochs.tmin),
        "tmax_s": float(epochs.tmax),
        "sfreq": float(epochs.info["sfreq"]),
        "n_channels": int(len(epochs.ch_names)),
        "mean_speech_onset_ms": _mean_speech_onset_ms(epochs),
    }
    (lock_dir / "epoch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not analyses:
        print(f"  [{tag}] epochs saved: n_epochs={len(epochs)}, window=({epochs.tmin}, {epochs.tmax})")
        return

    speech_onset_ms = _mean_speech_onset_ms(epochs) if tag == "event-lock" else None
    if "decoding" in analyses:
        _run_decoding_panel(config, epochs, lock_dir, lock_figure_dir, speech_onset_ms)
        print(f"  [{tag}] decoding done: n_epochs={len(epochs)}")

    if "rsa" in analyses:
        _run_rsa_panel(config, epochs, lock_dir, lock_figure_dir, speech_onset_ms)
        print(f"  [{tag}] rsa done: n_epochs={len(epochs)}")

    if "heterorc" in analyses:
        _run_heterorc_panel(config, epochs, lock_dir, lock_figure_dir, speech_onset_ms)
        print(f"  [{tag}] heterorc done: n_epochs={len(epochs)}")


def _run_decoding_panel(
    config: AnalysisConfig,
    epochs,
    result_dir: Path,
    figure_dir: Path,
    speech_onset_ms: float | None,
) -> None:
    for contrast_name, result in run_decoding(config, epochs, save_dir=result_dir).items():
        stats = run_statistics(result, config)
        save_statistics(stats, result_dir / f"{contrast_name}_decoding_stats.json")
        plot_decoding_time_series(
            result,
            config,
            figure_dir,
            contrast_name,
            stats=stats,
            speech_onset_ms=speech_onset_ms,
        )


def _run_rsa_panel(
    config: AnalysisConfig,
    epochs,
    result_dir: Path,
    figure_dir: Path,
    speech_onset_ms: float | None,
) -> None:
    for contrast_name, result in run_rsa(config, epochs, save_dir=result_dir).items():
        stats = run_statistics(result, config)
        save_statistics(stats, result_dir / f"{contrast_name}_rsa_stats.json")
        noise_ceiling = None
        if result.noise_ceiling.size:
            noise_ceiling = (result.noise_ceiling[0], result.noise_ceiling[1])
        plot_rsa_time_series(
            result,
            config,
            figure_dir,
            contrast_name,
            stats=stats,
            noise_ceiling=noise_ceiling,
            speech_onset_ms=speech_onset_ms,
        )


def _run_heterorc_panel(
    config: AnalysisConfig,
    epochs,
    result_dir: Path,
    figure_dir: Path,
    speech_onset_ms: float | None,
) -> None:
    results = run_heterorc_decoding(config, epochs, save_dir=result_dir, quick_mode=True)
    for contrast_name, result in results.items():
        stats = run_statistics(result, config)
        save_statistics(stats, result_dir / f"{contrast_name}_heterorc_decoding_stats.json")
        plot_heterorc_decoding_time_series(
            result,
            config,
            figure_dir,
            contrast_name,
            stats=stats,
            speech_onset_ms=speech_onset_ms,
        )


def run_subject(
    config: AnalysisConfig,
    group_name: str,
    subject: str,
    marker_df: pd.DataFrame,
    *,
    locks: set[str],
    analyses: set[str],
) -> Path | None:
    bdf_path = DATA_ROOT / group_name / f"{subject}.bdf"
    if not bdf_path.exists():
        print(f"Skip {subject}: missing {bdf_path}")
        return None

    subject_dir = config.paths.results_dir / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    subject_figure_dir = config.paths.subject_figure_dir(subject)
    print(f"[START] v2 {group_name} / {config.task} / {subject}")

    print(f"  step: build manifest", flush=True)
    trial_manifest_df = _maybe_build_trial_manifest(config, bdf_path)
    if "event" in locks:
        print(f"  step: preprocess event-lock", flush=True)
        try:
            pre = preprocess_session(config, bdf_path, marker_df, subject, trial_manifest_df=trial_manifest_df)
        except ValueError as exc:
            if trial_manifest_df is None or "formal trial" not in str(exc):
                raise
            print(f"selection fallback: {subject} ({exc})")
            pre = preprocess_session(config, bdf_path, marker_df, subject)

        event_epochs = pre.epochs
        if (
            config.task == "production"
            and trial_manifest_df is not None
            and (event_epochs.metadata is None or "detected_latency_ms" not in event_epochs.metadata.columns)
        ):
            _attach_latency_from_manifest(event_epochs, trial_manifest_df)
        _run_lock(
            config=config,
            epochs=event_epochs,
            lock_dir=subject_dir / "event_locked",
            lock_figure_dir=subject_figure_dir / "event_locked",
            tag="event-lock",
            analyses=analyses,
        )

    if config.task == "production" and "response" in locks:
        try:
            response_source_config = _build_response_source_config(config)
            response_pre = preprocess_session(
                response_source_config,
                bdf_path,
                marker_df,
                f"{subject}_response_source",
                trial_manifest_df=trial_manifest_df,
            )
            response_source_epochs = response_pre.epochs
            if (
                trial_manifest_df is not None
                and (
                    response_source_epochs.metadata is None
                    or "detected_latency_ms" not in response_source_epochs.metadata.columns
                )
            ):
                _attach_latency_from_manifest(response_source_epochs, trial_manifest_df)
            rlock = make_response_locked_epochs(
                response_source_epochs,
                tmin_s=RESPONSE_LOCK_TMIN_S,
                tmax_s=RESPONSE_LOCK_TMAX_S,
                baseline_s=RESPONSE_LOCK_BASELINE_S,
            )
        except (ValueError, RuntimeError) as exc:
            print(f"response-lock skipped ({subject}): {exc}")
        else:
            (subject_dir / "response_locked").mkdir(parents=True, exist_ok=True)
            (subject_dir / "response_locked" / "rlock_summary.json").write_text(
                json.dumps(
                    {
                        "n_input": rlock.n_input,
                        "n_kept": rlock.n_kept,
                        "n_dropped_missing": rlock.n_dropped_missing,
                        "n_dropped_out_of_range": rlock.n_dropped_out_of_range,
                        "tmin_s": RESPONSE_LOCK_TMIN_S,
                        "tmax_s": RESPONSE_LOCK_TMAX_S,
                        "baseline_s": list(RESPONSE_LOCK_BASELINE_S),
                        "rt_stats": rlock.rt_stats,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"  response-lock: kept={rlock.n_kept}/{rlock.n_input} "
                f"(missing={rlock.n_dropped_missing}, out_of_range={rlock.n_dropped_out_of_range})"
            )
            _run_lock(
                config=config,
                epochs=rlock.epochs,
                lock_dir=subject_dir / "response_locked",
                lock_figure_dir=subject_figure_dir / "response_locked",
                tag="response-lock",
                analyses=analyses,
            )
    elif "response" in locks:
        print(f"response-lock skipped: task={config.task}")

    print(f"Done v2 {group_name} / {config.task} / {subject}")
    return subject_dir


def run() -> None:
    parser = argparse.ArgumentParser(description="v2 batch EEG analysis (minimal preproc + response lock)")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-stats", action="store_true")
    parser.add_argument("--group", choices=list(SUBJECT_GROUPS), default=None)
    parser.add_argument("--task", choices=["production", "perception"], default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--stage", choices=["epoch", "decoding", "rsa", "heterorc", "all"], default="all")
    parser.add_argument("--lock", choices=["event", "response", "both"], default="both")
    args = parser.parse_args()

    analyses = {
        "epoch": set(),
        "decoding": {"decoding"},
        "rsa": {"rsa"},
        "heterorc": {"heterorc"},
        "all": {"decoding", "rsa", "heterorc"},
    }[args.stage]
    locks = {"event", "response"} if args.lock == "both" else {args.lock}

    for target in _iter_targets(args.group, args.task):
        config = _build_config(target, quick=args.quick, no_stats=args.no_stats)
        print(f"== v2 Target: {target.group_name} / {target.task} ==")
        marker_df = _load_marker_table(config, target.task)
        for subject in SUBJECT_GROUPS[target.group_name]["subjects"]:
            if args.subject and args.subject != subject:
                continue
            try:
                run_subject(
                    config,
                    target.group_name,
                    subject,
                    marker_df,
                    locks=locks,
                    analyses=analyses,
                )
            except Exception:
                import traceback
                print(f"!!! FAILED {target.group_name}/{target.task}/{subject}")
                traceback.print_exc()
                raise


if __name__ == "__main__":
    run()
