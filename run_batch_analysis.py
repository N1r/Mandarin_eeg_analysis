"""Batch runner for production/perception EEG analysis."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_N_JOBS = 1
# Pin BLAS libraries to a single thread to prevent nested parallelism
# (joblib workers x BLAS threads -> oversubscription -> Windows deadlock).
# For stability on Windows, the whole pipeline currently stays single-worker.
for _thread_env_name in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(_thread_env_name, "1")
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
    load_decoding_result,
    load_heterorc_decoding_result,
    make_config,
    plot_decoding_time_series,
    plot_group_time_series,
    plot_heterorc_decoding_time_series,
    plot_heterorc_group_time_series,
    plot_rsa_time_series,
    preprocess_session,
    run_decoding,
    run_group_statistics,
    run_heterorc_decoding,
    run_heterorc_interpretation,
    run_rsa,
    run_statistics,
    save_statistics,
)
from mandarin_speech_eeg.plotting import (
    plot_condition_comparison,
    plot_contrast_integrated,
    plot_modality_grid,
    plot_multi_contrast_overlay,
)
from mandarin_speech_eeg.weight_projection import (
    compute_weight_projection,
    plot_weight_projection_topomaps,
)
from mandarin_speech_eeg.trial_selection import (
    build_trial_manifest,
    load_trial_manifest,
    try_resolve_session_dir_from_bdf,
)

DATA_ROOT = REPO_ROOT.parent.parent / "Data"
RESULTS_ROOT = REPO_ROOT.parent / "results" / "batch_analysis"
SUBJECT_GROUPS = {
    "Production_only": {"subjects": ["p00", "p01", "p02"], "tasks": ["production"]},
    "Production_Perception": {"subjects": ["p05", "p06", "p07"], "tasks": ["production", "perception"]},
}


def _compute_speech_onset_ms(manifest_df: pd.DataFrame | None) -> float | None:
    """Trimmed mean of detected speech-onset latencies (kept trials only).

    Drops non-finite + outliers (<150 ms anticipations, >2000 ms drifts), then trims
    5% on each tail so a handful of extreme RTs don't shift the mean. Returns None
    when the manifest is missing the column (e.g. perception task).
    """
    if manifest_df is None or "detected_latency_ms" not in manifest_df.columns:
        return None
    frame = manifest_df
    if "keep_trial" in frame.columns:
        frame = frame[frame["keep_trial"].astype(int) == 1]
    latencies = pd.to_numeric(frame["detected_latency_ms"], errors="coerce").dropna()
    latencies = latencies[(latencies >= 150.0) & (latencies <= 2000.0)]
    if latencies.size < 10:
        return None
    low, high = np.quantile(latencies, [0.05, 0.95])
    trimmed = latencies[(latencies >= low) & (latencies <= high)]
    if trimmed.size == 0:
        return None
    return float(trimmed.mean())


def _save_speech_onset(subject_dir: Path, onset_ms: float | None) -> None:
    import json
    (subject_dir / "speech_onset.json").write_text(
        json.dumps({"speech_onset_ms": onset_ms}, indent=2), encoding="utf-8"
    )


def _load_speech_onset(subject_dir: Path) -> float | None:
    import json
    path = subject_dir / "speech_onset.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("speech_onset_ms")
    except (json.JSONDecodeError, OSError):
        return None


def _plot_weight_projections(config, epochs, figure_dir) -> None:
    """Per-subject Haufe activation topography figure per contrast."""
    for contrast_name, column_name in config.dataset.contrasts.items():
        try:
            projection = compute_weight_projection(config, epochs, column_name)
        except Exception as exc:  # decoding patterns can fail on degenerate splits
            print(f"weight projection skipped ({contrast_name}): {exc}")
            continue
        plot_weight_projection_topomaps(projection, epochs.info, config, figure_dir, contrast_name)


def _plot_subject_overlays(
    config,
    figure_dir,
    decoding_results,
    rsa_results,
    decoding_stats,
    rsa_stats,
    speech_onset_ms: float | None = None,
) -> None:
    """Generate single-subject multi-contrast overlays for decoding and RSA."""
    if len(decoding_results) >= 2:
        curves = {}
        for name, res in decoding_results.items():
            fold_sem = None
            if res.fold_scores.size:
                fold_sem = res.fold_scores.std(axis=0) / np.sqrt(len(res.fold_scores))
            curves[name] = (res.scores, fold_sem)
        times_s = next(iter(decoding_results.values())).times_s
        chance = next(iter(decoding_results.values())).chance_level
        plot_multi_contrast_overlay(
            curves=curves,
            times_s=times_s,
            chance_level=chance,
            mode="decoding",
            config=config,
            save_dir=figure_dir,
            title=f"Decoding overlay | {config.task}",
            stats_by_contrast=decoding_stats,
            speech_onset_ms=speech_onset_ms,
        )

    if len(rsa_results) >= 2:
        curves = {name: (res.scores, None) for name, res in rsa_results.items()}
        times_s = next(iter(rsa_results.values())).times_s
        plot_multi_contrast_overlay(
            curves=curves,
            times_s=times_s,
            chance_level=0.0,
            mode="rsa",
            config=config,
            save_dir=figure_dir,
            title=f"RSA overlay | {config.task}",
            stats_by_contrast=rsa_stats,
            speech_onset_ms=speech_onset_ms,
        )


_BINARY_CONTRASTS = ("Animacy", "Initial Type", "Rhyme Type")
_MULTI_CONTRAST = "Tone"


def _plot_integrated_figures(
    config,
    figure_dir: Path,
    decoding_results: dict,
    rsa_results: dict,
    heterorc_results: dict,
    decoding_stats: dict,
    rsa_stats: dict,
    heterorc_stats: dict,
    speech_onset_ms: float | None,
) -> None:
    """Per-contrast integrated figure + 3-binary combined + tone standalone."""
    contrasts = list(config.dataset.contrasts)

    # 1) Per-contrast integrated plot: decoding + RSA + HeteroRC in one figure
    for contrast in contrasts:
        plot_contrast_integrated(
            contrast_name=contrast,
            decoding_result=decoding_results.get(contrast),
            rsa_result=rsa_results.get(contrast),
            heterorc_result=heterorc_results.get(contrast),
            decoding_stats=decoding_stats.get(contrast),
            rsa_stats=rsa_stats.get(contrast),
            heterorc_stats=heterorc_stats.get(contrast),
            config=config,
            save_dir=figure_dir,
            speech_onset_ms=speech_onset_ms,
        )

    # 2) 3-binary combined (decoding/RSA/HeteroRC rows x {Animacy, Initial Type, Rhyme Type})
    binaries = [c for c in _BINARY_CONTRASTS if c in contrasts]
    if binaries:
        plot_modality_grid(
            contrasts=binaries,
            results_by_modality={
                "decoding": {c: decoding_results.get(c) for c in binaries if decoding_results.get(c) is not None},
                "rsa": {c: rsa_results.get(c) for c in binaries if rsa_results.get(c) is not None},
                "heterorc": {c: heterorc_results.get(c) for c in binaries if heterorc_results.get(c) is not None},
            },
            stats_by_modality={
                "decoding": {c: decoding_stats.get(c) for c in binaries},
                "rsa": {c: rsa_stats.get(c) for c in binaries},
                "heterorc": {c: heterorc_stats.get(c) for c in binaries},
            },
            config=config,
            save_dir=figure_dir,
            stem="binary_contrasts_combined",
            title=f"Binary contrasts | {config.task}",
            speech_onset_ms=speech_onset_ms,
        )

    # 3) Tone standalone (single-column modality grid for the multiclass contrast)
    if _MULTI_CONTRAST in contrasts:
        plot_modality_grid(
            contrasts=[_MULTI_CONTRAST],
            results_by_modality={
                "decoding": {_MULTI_CONTRAST: decoding_results.get(_MULTI_CONTRAST)}
                if decoding_results.get(_MULTI_CONTRAST) is not None else {},
                "rsa": {_MULTI_CONTRAST: rsa_results.get(_MULTI_CONTRAST)}
                if rsa_results.get(_MULTI_CONTRAST) is not None else {},
                "heterorc": {_MULTI_CONTRAST: heterorc_results.get(_MULTI_CONTRAST)}
                if heterorc_results.get(_MULTI_CONTRAST) is not None else {},
            },
            stats_by_modality={
                "decoding": {_MULTI_CONTRAST: decoding_stats.get(_MULTI_CONTRAST)},
                "rsa": {_MULTI_CONTRAST: rsa_stats.get(_MULTI_CONTRAST)},
                "heterorc": {_MULTI_CONTRAST: heterorc_stats.get(_MULTI_CONTRAST)},
            },
            config=config,
            save_dir=figure_dir,
            stem="tone_combined",
            title=f"Tone | {config.task}",
            speech_onset_ms=speech_onset_ms,
        )


def _interpret_stats(stats) -> str:
    """One-sentence verbal summary of a StatisticalTestResult."""
    if stats is None:
        return "no statistics computed."
    cps_raw = getattr(stats, "cluster_p_values", None)
    if cps_raw is None:
        cps = []
    else:
        cps = list(np.asarray(cps_raw).ravel())
    sig = [float(p) for p in cps if float(p) < 0.05]
    if sig:
        return f"{len(sig)} / {len(cps)} cluster(s) reached p<.05 (min p={min(sig):.3g})."
    if len(cps) > 0:
        return f"no cluster reached p<.05 (min p={min(float(p) for p in cps):.3g}, {len(cps)} clusters tested)."
    return "no clusters were detected."


def _write_subject_html_report(
    subject: str,
    task: str,
    figure_dir: Path,
    subject_dir: Path,
    contrasts: list[str],
    decoding_stats: dict,
    rsa_stats: dict,
    heterorc_stats: dict,
    speech_onset_ms: float | None,
    per_modality: bool,
) -> None:
    from mandarin_speech_eeg.plotting import figure_stem

    onset_txt = f"{speech_onset_ms:.0f} ms" if speech_onset_ms is not None else "n/a (non-production task)"
    parts: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{subject} {task} summary</title>",
        "<style>body{font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}"
        "h1{font-size:20px;margin-bottom:4px}h2{font-size:16px;border-bottom:1px solid #ddd;padding-bottom:4px;margin-top:28px}"
        "h3{font-size:14px;margin-top:18px}img{max-width:100%;border:1px solid #eee;border-radius:4px;margin:6px 0}"
        "table{border-collapse:collapse;margin:8px 0}td,th{padding:4px 10px;border:1px solid #ddd;font-size:12.5px;text-align:left}"
        ".note{color:#555;font-size:12.5px;line-height:1.5}</style></head><body>",
        f"<h1>{subject} &mdash; {task}</h1>",
        f"<div class='note'>Speech onset (trimmed mean): <b>{onset_txt}</b>.</div>",
    ]

    def _img(path: Path, caption: str) -> str:
        if not path.exists():
            return ""
        rel = os.path.relpath(path, subject_dir).replace("\\", "/")
        return f"<figure><img src='{rel}' alt='{caption}'><figcaption class='note'>{caption}</figcaption></figure>"

    parts.append("<h2>Summary table</h2><table><tr><th>Contrast</th><th>Decoding</th><th>RSA</th><th>HeteroRC</th></tr>")
    for c in contrasts:
        parts.append(
            f"<tr><td>{c}</td><td>{_interpret_stats(decoding_stats.get(c))}</td>"
            f"<td>{_interpret_stats(rsa_stats.get(c))}</td>"
            f"<td>{_interpret_stats(heterorc_stats.get(c))}</td></tr>"
        )
    parts.append("</table>")

    parts.append("<h2>Overviews</h2>")
    parts.append(_img(figure_dir / "binary_contrasts_combined.png", "3 binary contrasts (Animacy / Initial Type / Rhyme Type) across modalities."))
    parts.append(_img(figure_dir / "tone_combined.png", "Tone (4-way) across modalities."))

    parts.append("<h2>Per-contrast integrated views</h2>")
    for c in contrasts:
        stem = figure_stem(c)
        parts.append(f"<h3>{c}</h3>")
        parts.append(_img(figure_dir / f"{stem}_integrated.png",
                          f"{c}: decoding + RSA + HeteroRC with behavioural onset marker."))

    parts.append("<h2>Topographies and optional detailed plots</h2>")
    for c in contrasts:
        stem = figure_stem(c)
        parts.append(f"<h3>{c}</h3>")
        parts.append(_img(figure_dir / f"{stem}_weight_topomap.png", f"{c} &mdash; Haufe weight topography."))
        parts.append(_img(figure_dir / f"{stem}_heterorc_interpretation.png", f"{c} &mdash; HeteroRC interpretation."))
        if per_modality:
            parts.append(_img(figure_dir / f"{stem}_decoding.png", f"{c} &mdash; decoding."))
            parts.append(_img(figure_dir / f"{stem}_rsa.png", f"{c} &mdash; RSA (with noise ceiling)."))
            parts.append(_img(figure_dir / f"{stem}_heterorc_decoding.png", f"{c} &mdash; HeteroRC decoding."))

    parts.append("</body></html>")
    (subject_dir / "report.html").write_text("\n".join(p for p in parts if p), encoding="utf-8")
    print(f"wrote report: {subject_dir / 'report.html'}")


@dataclass(frozen=True)
class BatchTarget:
    group_name: str
    task: str


def run(default_task: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run batch EEG analysis")
    parser.add_argument("--quick", action="store_true", help="Use the quick analysis preset")
    parser.add_argument("--no-stats", action="store_true", help="Disable permutation statistics")
    parser.add_argument("--n-permutations", type=int, default=None, help="Override permutation count for statistics")
    parser.add_argument("--n-jobs", type=int, default=_DEFAULT_N_JOBS, help="Maximum parallel jobs/cores to use; currently forced to 1")
    parser.add_argument("--per-modality", action="store_true", help="Also save per-modality single-contrast figures")
    parser.add_argument("--decoder", choices=["svm", "lda", "gnb", "logreg", "ridge"], default=None)
    parser.add_argument("--window-ms", type=float, default=None)
    parser.add_argument("--window-step-ms", type=float, default=None)
    parser.add_argument("--with-heterorc", action="store_true", help="Run HeteroRC decoding alongside the default pipeline")
    parser.add_argument(
        "--with-heterorc-interpretation",
        action="store_true",
        help="Also generate HeteroRC interpretation figures",
    )
    parser.add_argument("--heterorc-readout", choices=["ridge", "svm", "lda", "logreg"], default=None)
    parser.add_argument("--heterorc-window-ms", type=float, default=None)
    parser.add_argument("--heterorc-window-step-ms", type=float, default=None)
    parser.add_argument("--group", choices=list(SUBJECT_GROUPS), default=None)
    parser.add_argument("--task", choices=["production", "perception"], default=default_task)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--no-group", action="store_true", help="Skip group-level analysis and cross-task comparisons")
    args = parser.parse_args()

    results_root = RESULTS_ROOT / "results"
    figures_root = RESULTS_ROOT / "figures"
    results_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    # Remember per-(group, task) artifacts so we can cross-task compare afterwards.
    completed_runs: dict[tuple[str, str], tuple[AnalysisConfig, list[tuple[str, Path]]]] = {}

    for target in _iter_targets(args.group, args.task):
        config = _build_batch_config(
            target=target,
            quick=args.quick,
            no_stats=args.no_stats,
            results_root=results_root,
            figures_root=figures_root,
            with_heterorc=args.with_heterorc,
            with_heterorc_interpretation=args.with_heterorc_interpretation,
            n_permutations=args.n_permutations,
            n_jobs=args.n_jobs,
            decoder=args.decoder,
            window_ms=args.window_ms,
            window_step_ms=args.window_step_ms,
            heterorc_readout=args.heterorc_readout,
            heterorc_window_ms=args.heterorc_window_ms,
            heterorc_window_step_ms=args.heterorc_window_step_ms,
        )
        print(f"== Target: {target.group_name} / {target.task} ==")
        marker_df = _load_marker_table(config, target.task)
        processed_subjects: list[tuple[str, Path]] = []

        for subject in SUBJECT_GROUPS[target.group_name]["subjects"]:
            if args.subject and args.subject != subject:
                continue
            subject_dir = run_subject(
                config,
                target.group_name,
                subject,
                marker_df,
                quick_mode=args.quick,
                per_modality=args.per_modality,
            )
            if subject_dir is not None:
                processed_subjects.append((subject, subject_dir))

        if not args.no_group:
            run_group_level(config, processed_subjects)
        completed_runs[(target.group_name, target.task)] = (config, processed_subjects)

    if not args.no_group:
        _plot_task_comparisons(completed_runs, figures_root)


def _plot_task_comparisons(
    completed_runs: dict[tuple[str, str], tuple["AnalysisConfig", list[tuple[str, Path]]]],
    figures_root: Path,
) -> None:
    """Per group, overlay production vs perception group-mean curves for each contrast."""
    from mandarin_speech_eeg import load_rsa_result
    from mandarin_speech_eeg.statistics import load_statistics

    groups_with_both: dict[str, dict[str, tuple[AnalysisConfig, list[tuple[str, Path]]]]] = {}
    for (group_name, task), payload in completed_runs.items():
        groups_with_both.setdefault(group_name, {})[task] = payload

    for group_name, tasks in groups_with_both.items():
        if "production" not in tasks or "perception" not in tasks:
            continue
        prod_config, prod_subjects = tasks["production"]
        perc_config, perc_subjects = tasks["perception"]
        save_dir = figures_root / group_name / "comparison"
        onset_ms = _group_speech_onset_ms(prod_subjects)

        for mode, suffix, loader in (
            ("decoding", "decoding", load_decoding_result),
            ("rsa", "rsa", load_rsa_result),
        ):
            for contrast_name in prod_config.dataset.contrasts:
                prod_mean = _load_group_mean(prod_subjects, contrast_name, suffix, loader)
                perc_mean = _load_group_mean(perc_subjects, contrast_name, suffix, loader)
                if prod_mean is None or perc_mean is None:
                    continue

                times_s, chance_level, prod_curve, prod_sem = prod_mean
                _, _, perc_curve, perc_sem = perc_mean
                curves = {
                    "Production": (prod_curve, prod_sem),
                    "Perception": (perc_curve, perc_sem),
                }
                plot_condition_comparison(
                    curves=curves,
                    times_s=times_s,
                    chance_level=chance_level,
                    mode=mode,
                    config=prod_config,  # palette is task-agnostic; production picked for defaults
                    save_dir=save_dir,
                    title=f"{contrast_name} | production vs perception",
                )


def _load_group_mean(
    processed_subjects: list[tuple[str, Path]],
    contrast_name: str,
    file_suffix: str,
    loader_fn,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray] | None:
    curves = []
    times_s = None
    chance_level = 0.0
    for _, subject_dir in processed_subjects:
        path = subject_dir / f"{contrast_name}_{file_suffix}.npz"
        if not path.exists():
            continue
        result = loader_fn(path)
        curves.append(result.scores)
        times_s = result.times_s
        chance_level = result.chance_level
    if len(curves) < 2 or times_s is None:
        return None
    arr = np.array(curves)
    return times_s, chance_level, arr.mean(axis=0), arr.std(axis=0) / np.sqrt(len(arr))


def run_subject(
    config: AnalysisConfig,
    group_name: str,
    subject: str,
    marker_df: pd.DataFrame,
    quick_mode: bool = False,
    per_modality: bool = False,
) -> Path | None:
    bdf_path = DATA_ROOT / group_name / f"{subject}.bdf"
    if not bdf_path.exists():
        print(f"Skip {subject}: missing {bdf_path}")
        return None

    subject_dir = config.paths.results_dir / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = config.paths.subject_figure_dir(subject)
    print(f"[START] {group_name} / {config.task} / {subject}")

    trial_manifest_df = _maybe_build_trial_manifest(config, bdf_path)
    speech_onset_ms = _compute_speech_onset_ms(trial_manifest_df)
    _save_speech_onset(subject_dir, speech_onset_ms)
    if speech_onset_ms is not None:
        print(f"speech onset (trimmed mean): {speech_onset_ms:.0f} ms")
    try:
        preprocessing_result = preprocess_session(
            config,
            bdf_path,
            marker_df,
            subject,
            trial_manifest_df=trial_manifest_df,
        )
    except ValueError as exc:
        if trial_manifest_df is None or "formal trial" not in str(exc):
            raise
        print(
            "selection fallback: "
            f"{subject} manifest could not be aligned to EEG events ({exc}); "
            "retrying legacy preprocessing for this subject."
        )
        preprocessing_result = preprocess_session(config, bdf_path, marker_df, subject)
    decoding_results = run_decoding(config, preprocessing_result.epochs, save_dir=subject_dir)
    rsa_results = run_rsa(config, preprocessing_result.epochs, save_dir=subject_dir)

    decoding_stats: dict[str, object] = {}
    for contrast_name, result in decoding_results.items():
        stats = run_statistics(result, config)
        save_statistics(stats, subject_dir / f"{contrast_name}_decoding_stats.json")
        if per_modality:
            plot_decoding_time_series(
                result, config, figure_dir, contrast_name,
                stats=stats, speech_onset_ms=speech_onset_ms,
            )
        decoding_stats[contrast_name] = stats

    rsa_stats: dict[str, object] = {}
    for contrast_name, result in rsa_results.items():
        stats = run_statistics(result, config)
        save_statistics(stats, subject_dir / f"{contrast_name}_rsa_stats.json")
        noise_ceiling = None
        if result.noise_ceiling.size:
            noise_ceiling = (result.noise_ceiling[0], result.noise_ceiling[1])
        if per_modality:
            plot_rsa_time_series(
                result, config, figure_dir, contrast_name,
                stats=stats, noise_ceiling=noise_ceiling, speech_onset_ms=speech_onset_ms,
            )
        rsa_stats[contrast_name] = stats

    # Haufe-transformed weight topographies (one figure per contrast)
    _plot_weight_projections(config, preprocessing_result.epochs, figure_dir)

    # Single-subject multi-contrast overlays
    _plot_subject_overlays(
        config=config,
        figure_dir=figure_dir,
        decoding_results=decoding_results,
        rsa_results=rsa_results,
        decoding_stats=decoding_stats,
        rsa_stats=rsa_stats,
        speech_onset_ms=speech_onset_ms,
    )

    heterorc_results: dict[str, object] = {}
    heterorc_stats: dict[str, object] = {}
    if config.heterorc.enabled:
        heterorc_results = run_heterorc_decoding(
            config,
            preprocessing_result.epochs,
            save_dir=subject_dir,
            quick_mode=quick_mode,
        )
        for contrast_name, result in heterorc_results.items():
            stats = run_statistics(result, config)
            save_statistics(stats, subject_dir / f"{contrast_name}_heterorc_decoding_stats.json")
            heterorc_stats[contrast_name] = stats
            if per_modality:
                plot_heterorc_decoding_time_series(
                    result, config, figure_dir, contrast_name,
                    stats=stats, speech_onset_ms=speech_onset_ms,
                )

        if config.heterorc.interpretation_enabled:
            run_heterorc_interpretation(
                config,
                preprocessing_result.epochs,
                save_dir=figure_dir,
                quick_mode=quick_mode,
            )

    # Per-contrast integrated (decoding + RSA + HeteroRC) and modality grids.
    _plot_integrated_figures(
        config=config,
        figure_dir=figure_dir,
        decoding_results=decoding_results,
        rsa_results=rsa_results,
        heterorc_results=heterorc_results,
        decoding_stats=decoding_stats,
        rsa_stats=rsa_stats,
        heterorc_stats=heterorc_stats,
        speech_onset_ms=speech_onset_ms,
    )

    # Per-subject HTML summary.
    _write_subject_html_report(
        subject=subject,
        task=config.task,
        figure_dir=figure_dir,
        subject_dir=subject_dir,
        contrasts=list(config.dataset.contrasts),
        decoding_stats=decoding_stats,
        rsa_stats=rsa_stats,
        heterorc_stats=heterorc_stats,
        speech_onset_ms=speech_onset_ms,
        per_modality=per_modality,
    )

    print(f"Done {group_name} / {config.task} / {subject}")
    return subject_dir


def _maybe_build_trial_manifest(config: AnalysisConfig, bdf_path: Path) -> pd.DataFrame | None:
    if config.task != "production":
        print(f"selection skipped: task={config.task}")
        return None

    session_dir = try_resolve_session_dir_from_bdf(bdf_path, search_root=DATA_ROOT)
    if session_dir is None:
        print(f"selection skipped: legacy dataset ({bdf_path})")
        return None

    result = build_trial_manifest(session_dir)
    print(
        "selection modern: "
        f"keep={result.summary.get('n_keep')} "
        f"drop={result.summary.get('n_drop')} "
        f"manifest={result.manifest_path}"
    )
    return load_trial_manifest(result.manifest_path)


def _group_speech_onset_ms(processed_subjects: list[tuple[str, Path]]) -> float | None:
    onsets = [v for _, d in processed_subjects for v in (_load_speech_onset(d),) if v is not None]
    return float(np.mean(onsets)) if onsets else None


def run_group_level(config: AnalysisConfig, processed_subjects: list[tuple[str, Path]]) -> None:
    if len(processed_subjects) < 2:
        return

    group_figure_dir = config.paths.figures_dir / "group"
    group_figure_dir.mkdir(parents=True, exist_ok=True)
    group_onset_ms = _group_speech_onset_ms(processed_subjects)

    for contrast_name in config.dataset.contrasts:
        _run_group_family(
            config=config,
            processed_subjects=processed_subjects,
            contrast_name=contrast_name,
            group_figure_dir=group_figure_dir,
            file_suffix="decoding",
            plot_fn=plot_group_time_series,
            loader_fn=load_decoding_result,
            speech_onset_ms=group_onset_ms,
        )

        if config.heterorc.enabled:
            _run_group_family(
                config=config,
                processed_subjects=processed_subjects,
                contrast_name=contrast_name,
                group_figure_dir=group_figure_dir,
                file_suffix="heterorc_decoding",
                plot_fn=plot_heterorc_group_time_series,
                loader_fn=load_heterorc_decoding_result,
                speech_onset_ms=group_onset_ms,
            )

    _plot_group_overlays(config, processed_subjects, group_figure_dir, speech_onset_ms=group_onset_ms)


def _plot_group_overlays(
    config: AnalysisConfig,
    processed_subjects: list[tuple[str, Path]],
    group_figure_dir: Path,
    speech_onset_ms: float | None = None,
) -> None:
    """Collect group mean curves across all contrasts and produce overlay figures."""
    from mandarin_speech_eeg import load_rsa_result

    for mode, suffix, loader_fn in (
        ("decoding", "decoding", load_decoding_result),
        ("rsa", "rsa", load_rsa_result),
    ):
        curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        times_s = None
        chance_level = 0.0
        for contrast_name in config.dataset.contrasts:
            subject_curves = []
            for _, subject_dir in processed_subjects:
                path = subject_dir / f"{contrast_name}_{suffix}.npz"
                if not path.exists():
                    continue
                r = loader_fn(path)
                subject_curves.append(r.scores)
                times_s = r.times_s
                chance_level = r.chance_level
            if len(subject_curves) < 2:
                continue
            arr = np.array(subject_curves)
            mean_curve = arr.mean(axis=0)
            sem = arr.std(axis=0) / np.sqrt(len(arr))
            curves[contrast_name] = (mean_curve, sem)

        if len(curves) >= 2 and times_s is not None:
            plot_multi_contrast_overlay(
                curves=curves,
                times_s=times_s,
                chance_level=chance_level,
                mode=mode,
                config=config,
                save_dir=group_figure_dir,
                title=f"Group {mode} overlay | {config.task} (n={len(processed_subjects)})",
                speech_onset_ms=speech_onset_ms,
            )


def _run_group_family(
    config: AnalysisConfig,
    processed_subjects: list[tuple[str, Path]],
    contrast_name: str,
    group_figure_dir: Path,
    file_suffix: str,
    plot_fn,
    loader_fn,
    speech_onset_ms: float | None = None,
) -> None:
    subject_curves = []
    times_s = None
    chance_level = None

    for _, subject_dir in processed_subjects:
        result_path = subject_dir / f"{contrast_name}_{file_suffix}.npz"
        if not result_path.exists():
            continue
        result = loader_fn(result_path)
        subject_curves.append(result.scores)
        times_s = result.times_s
        chance_level = result.chance_level

    if len(subject_curves) < 2 or times_s is None or chance_level is None:
        return

    subject_curves_array = np.array(subject_curves)
    stats = run_group_statistics(subject_curves_array, chance_level, config)
    save_statistics(stats, config.paths.results_dir / f"{contrast_name}_{file_suffix}_group_stats.json")

    if file_suffix == "decoding":
        plot_fn(
            subject_scores=subject_curves_array,
            times_s=times_s,
            chance_level=chance_level,
            contrast_name=contrast_name,
            mode="decoding",
            config=config,
            save_dir=group_figure_dir,
            stats=stats,
            speech_onset_ms=speech_onset_ms,
        )
    else:
        plot_fn(
            subject_scores=subject_curves_array,
            times_s=times_s,
            chance_level=chance_level,
            contrast_name=contrast_name,
            config=config,
            save_dir=group_figure_dir,
            stats=stats,
            speech_onset_ms=speech_onset_ms,
        )


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


def _build_batch_config(
    target: BatchTarget,
    quick: bool,
    no_stats: bool,
    results_root: Path,
    figures_root: Path,
    with_heterorc: bool,
    with_heterorc_interpretation: bool,
    n_permutations: int | None,
    n_jobs: int,
    decoder: str | None,
    window_ms: float | None,
    window_step_ms: float | None,
    heterorc_readout: str | None,
    heterorc_window_ms: float | None,
    heterorc_window_step_ms: float | None,
) -> AnalysisConfig:
    config = make_config(task=target.task, quick=quick)
    _limit_parallel_threads(n_jobs)
    paths = config.paths.with_roots(
        results_dir=results_root / target.group_name / target.task,
        figures_dir=figures_root / target.group_name / target.task,
        cache_dir=RESULTS_ROOT / "cache" / target.group_name / target.task,
    )
    config = config.with_paths(paths)
    config.paths.ensure_directories()
    config.heterorc.enabled = with_heterorc
    config.heterorc.interpretation_enabled = with_heterorc_interpretation
    if decoder:
        config.decoding.decoder = decoder
    if window_ms is not None:
        config.decoding.temporal_window_ms = window_ms
    if window_step_ms is not None:
        config.decoding.temporal_step_ms = window_step_ms
    if heterorc_readout:
        config.heterorc.readout_decoder = heterorc_readout
    if heterorc_window_ms is not None:
        config.heterorc.temporal_window_ms = heterorc_window_ms
    if heterorc_window_step_ms is not None:
        config.heterorc.temporal_step_ms = heterorc_window_step_ms
    if no_stats:
        disable_statistics(config)
    elif n_permutations is not None:
        config.statistics.n_permutations = n_permutations
        config.statistics.quick_n_permutations = n_permutations
    config.statistics.n_jobs = _normalize_n_jobs(n_jobs)
    return config


def _normalize_n_jobs(n_jobs: int | None) -> int:
    return 1


def _limit_parallel_threads(n_jobs: int | None) -> None:
    # Keep both BLAS and joblib/loky single-worker to avoid Windows hangs.
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        os.environ[name] = "1"
    os.environ["LOKY_MAX_CPU_COUNT"] = str(_normalize_n_jobs(n_jobs))
    os.environ["MKL_DYNAMIC"] = "FALSE"


def _load_marker_table(config: AnalysisConfig, task: str) -> pd.DataFrame:
    marker_df = pd.read_csv(config.paths.marker_csv)
    if task == "perception":
        marker_df = marker_df.copy()
        marker_df["marker"] = marker_df["marker"] + 100
    return marker_df


def main() -> None:
    run()


if __name__ == "__main__":
    main()
