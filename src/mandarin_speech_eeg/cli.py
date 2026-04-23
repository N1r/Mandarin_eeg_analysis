"""Command-line entry points for the EEG analysis package."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AnalysisConfig, DATA_ROOT, disable_statistics, make_config
from .asr import main as analyze_asr_main
from .onset import main as analyze_onset_main
from .trial_selection import (
    build_trial_manifest,
    load_trial_manifest,
    main as analyze_selection_main,
    try_resolve_session_dir_from_bdf,
)

logger = logging.getLogger(__name__)


def analyze_onset_cli() -> None:
    analyze_onset_main()


def analyze_asr_cli() -> None:
    analyze_asr_main()


def analyze_selection_cli() -> None:
    analyze_selection_main()


def analyze_session_cli() -> None:
    parser = _base_parser("Run single-session EEG analysis")
    parser.add_argument("--bdf", required=True, help="Path to BDF file")
    parser.add_argument("--session-name", required=True, help="Session name")
    parser.add_argument("--marker-csv", default=None, help="Optional marker CSV path")
    parser.add_argument("--save-dir", default=None, help="Optional output directory")
    args = parser.parse_args()

    config = _build_config(args)
    marker_df = _load_marker_table(config, args.marker_csv)
    save_dir = _resolve_save_dir(config, args.save_dir, args.session_name)
    trial_manifest_df = _maybe_build_trial_manifest(config, Path(args.bdf))

    from .decoding import run_decoding
    from .heterorc_analysis import run_heterorc_decoding, run_heterorc_interpretation
    from .preprocessing import preprocess_session
    from .rsa import run_rsa

    preprocessing_result = preprocess_session(
        config,
        args.bdf,
        marker_df,
        args.session_name,
        trial_manifest_df=trial_manifest_df,
    )
    decoding_results = run_decoding(config, preprocessing_result.epochs, save_dir=save_dir, quick_mode=args.quick)
    rsa_results = run_rsa(config, preprocessing_result.epochs, save_dir=save_dir, quick_mode=args.quick)
    heterorc_results: dict[str, Any] = {}
    if args.with_heterorc:
        heterorc_results = run_heterorc_decoding(
            config,
            preprocessing_result.epochs,
            save_dir=save_dir,
            quick_mode=args.quick,
        )
        if args.with_heterorc_interpretation:
            run_heterorc_interpretation(
                config,
                preprocessing_result.epochs,
                save_dir=config.paths.subject_figure_dir(args.session_name),
                quick_mode=args.quick,
            )

    _save_session_statistics_and_figures(config, decoding_results, rsa_results, save_dir)
    _save_heterorc_statistics_and_figures(config, heterorc_results, save_dir, config.paths.subject_figure_dir(args.session_name))
    logger.info("Session analysis complete: %s", save_dir)


def analyze_subject_cli() -> None:
    analyze_session_cli()


def analyze_group_cli() -> None:
    parser = _base_parser("Run group EEG analysis")
    parser.add_argument("--subjects", nargs="+", required=True, help="Subject IDs")
    parser.add_argument("--data-dir", default=None, help="Directory containing BDF files")
    parser.add_argument("--marker-csv", default=None, help="Optional marker CSV path")
    parser.add_argument("--save-dir", default=None, help="Optional output directory")
    args = parser.parse_args()

    config = _build_config(args)
    marker_df = _load_marker_table(config, args.marker_csv)
    data_dir = Path(args.data_dir) if args.data_dir else DATA_ROOT
    save_dir = _resolve_save_dir(config, args.save_dir, "group")

    subject_result_dirs = _run_subject_level_inputs(
        config=config,
        subjects=args.subjects,
        data_dir=data_dir,
        marker_df=marker_df,
        save_dir=save_dir,
        quick_mode=args.quick,
        with_heterorc=args.with_heterorc,
        with_heterorc_interpretation=args.with_heterorc_interpretation,
    )
    _run_group_level_analysis(config, subject_result_dirs, save_dir, with_heterorc=args.with_heterorc)
    logger.info("Group analysis complete: %s", save_dir)


def _base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--task", default="production", choices=["production", "perception"])
    parser.add_argument("--quick", action="store_true", help="Use the quick analysis preset")
    parser.add_argument("--no-stats", action="store_true", help="Disable permutation statistics")
    parser.add_argument("--n-permutations", type=int, default=None, help="Override permutation count for statistics")
    parser.add_argument("--decoder", choices=["svm", "lda", "logreg", "ridge"], default=None, help="Decoder for the standard MVPA branch")
    parser.add_argument("--window-ms", type=float, default=None, help="Temporal averaging window for the standard MVPA branch")
    parser.add_argument("--window-step-ms", type=float, default=None, help="Step size for the standard MVPA temporal window")
    parser.add_argument("--with-heterorc", action="store_true", help="Run HeteroRC decoding in addition to the default pipeline")
    parser.add_argument(
        "--with-heterorc-interpretation",
        action="store_true",
        help="Also generate HeteroRC interpretation figures (requires --with-heterorc)",
    )
    parser.add_argument("--heterorc-readout", choices=["ridge", "svm", "lda", "logreg"], default=None, help="Readout classifier for HeteroRC")
    parser.add_argument("--heterorc-window-ms", type=float, default=None, help="Temporal averaging window for HeteroRC states")
    parser.add_argument("--heterorc-window-step-ms", type=float, default=None, help="Step size for the HeteroRC temporal window")
    return parser


def _build_config(args: argparse.Namespace) -> AnalysisConfig:
    config = make_config(task=args.task, quick=args.quick)
    config.heterorc.enabled = bool(args.with_heterorc)
    config.heterorc.interpretation_enabled = bool(args.with_heterorc_interpretation)
    if args.decoder:
        config.decoding.decoder = args.decoder
    if args.window_ms is not None:
        config.decoding.temporal_window_ms = args.window_ms
    if args.window_step_ms is not None:
        config.decoding.temporal_step_ms = args.window_step_ms
    if args.heterorc_readout:
        config.heterorc.readout_decoder = args.heterorc_readout
    if args.heterorc_window_ms is not None:
        config.heterorc.temporal_window_ms = args.heterorc_window_ms
    if args.heterorc_window_step_ms is not None:
        config.heterorc.temporal_step_ms = args.heterorc_window_step_ms
    if args.no_stats:
        disable_statistics(config)
    elif args.n_permutations is not None:
        config.statistics.n_permutations = args.n_permutations
        config.statistics.quick_n_permutations = args.n_permutations
    config.paths.ensure_directories()
    return config


def _load_marker_table(config: AnalysisConfig, marker_csv: str | None) -> pd.DataFrame:
    marker_path = Path(marker_csv) if marker_csv else config.paths.marker_csv
    return pd.read_csv(marker_path)


def _resolve_save_dir(config: AnalysisConfig, save_dir: str | None, session_name: str) -> Path:
    output_dir = Path(save_dir) if save_dir else config.paths.session_dir(session_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_session_statistics_and_figures(
    config: AnalysisConfig,
    decoding_results: dict[str, Any],
    rsa_results: dict[str, Any],
    save_dir: Path,
) -> None:
    from .plotting import plot_decoding_time_series, plot_rsa_time_series
    from .statistics import run_statistics, save_statistics

    for contrast_name, result in decoding_results.items():
        stats = run_statistics(result, config)
        save_statistics(stats, save_dir / f"{contrast_name}_decoding_stats.json")
        plot_decoding_time_series(result, config, save_dir, contrast_name)

    for contrast_name, result in rsa_results.items():
        stats = run_statistics(result, config)
        save_statistics(stats, save_dir / f"{contrast_name}_rsa_stats.json")
        plot_rsa_time_series(result, config, save_dir, contrast_name)


def _save_heterorc_statistics_and_figures(
    config: AnalysisConfig,
    heterorc_results: dict[str, Any],
    save_dir: Path,
    figure_dir: Path,
) -> None:
    from .heterorc_analysis import plot_heterorc_decoding_time_series
    from .statistics import run_statistics, save_statistics

    for contrast_name, result in heterorc_results.items():
        stats = run_statistics(result, config)
        save_statistics(stats, save_dir / f"{contrast_name}_heterorc_decoding_stats.json")
        plot_heterorc_decoding_time_series(result, config, figure_dir, contrast_name)


def _run_subject_level_inputs(
    config: AnalysisConfig,
    subjects: list[str],
    data_dir: Path,
    marker_df: pd.DataFrame,
    save_dir: Path,
    quick_mode: bool,
    with_heterorc: bool = False,
    with_heterorc_interpretation: bool = False,
) -> list[tuple[str, Path]]:
    result_dirs: list[tuple[str, Path]] = []

    from .decoding import run_decoding
    from .heterorc_analysis import run_heterorc_decoding, run_heterorc_interpretation
    from .preprocessing import preprocess_session

    for subject in subjects:
        bdf_path = _resolve_subject_bdf(data_dir, subject)
        if not bdf_path.exists():
            logger.warning("Missing BDF for subject %s under %s", subject, data_dir)
            continue

        subject_dir = save_dir / subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        trial_manifest_df = _maybe_build_trial_manifest(config, bdf_path)
        preprocessing_result = preprocess_session(
            config,
            bdf_path,
            marker_df,
            subject,
            trial_manifest_df=trial_manifest_df,
        )
        run_decoding(config, preprocessing_result.epochs, save_dir=subject_dir, quick_mode=quick_mode)
        if with_heterorc:
            run_heterorc_decoding(config, preprocessing_result.epochs, save_dir=subject_dir, quick_mode=quick_mode)
            if with_heterorc_interpretation:
                run_heterorc_interpretation(
                    config,
                    preprocessing_result.epochs,
                    save_dir=config.paths.subject_figure_dir(subject),
                    quick_mode=quick_mode,
                )
        result_dirs.append((subject, subject_dir))

    return result_dirs


def _maybe_build_trial_manifest(config: AnalysisConfig, bdf_path: Path) -> pd.DataFrame | None:
    if config.task != "production":
        logger.info("selection skipped: task=%s", config.task)
        return None

    session_dir = try_resolve_session_dir_from_bdf(bdf_path, search_root=DATA_ROOT)
    if session_dir is None:
        logger.info("selection skipped: legacy dataset (%s)", bdf_path)
        return None

    logger.info("selection mode: modern session (%s)", session_dir)
    result = build_trial_manifest(session_dir)
    logger.info(
        "selection summary: keep=%s drop=%s manifest=%s",
        result.summary.get("n_keep"),
        result.summary.get("n_drop"),
        result.manifest_path,
    )
    return load_trial_manifest(result.manifest_path)


def _resolve_subject_bdf(data_dir: Path, subject: str) -> Path:
    legacy_path = data_dir / f"{subject}.bdf"
    if legacy_path.exists():
        return legacy_path

    exact_matches = sorted(data_dir.glob(f"**/{subject}.bdf"))
    if exact_matches:
        return exact_matches[0]

    normalized_subject = subject
    if subject.startswith("sub-") and subject[4:].isdigit():
        normalized_subject = f"p{int(subject[4:]):02d}"
    elif subject.startswith("p") and subject[1:].isdigit():
        normalized_subject = f"sub-{int(subject[1:])}"

    if normalized_subject != subject:
        normalized_matches = sorted(data_dir.glob(f"**/{normalized_subject}.bdf"))
        if normalized_matches:
            return normalized_matches[0]

    return legacy_path


def _run_group_level_analysis(
    config: AnalysisConfig,
    subject_result_dirs: list[tuple[str, Path]],
    save_dir: Path,
    with_heterorc: bool = False,
) -> None:
    from .decoding import load_decoding_result
    from .heterorc_analysis import load_heterorc_decoding_result, plot_heterorc_group_time_series
    from .plotting import plot_group_time_series
    from .statistics import run_group_statistics, save_statistics

    for contrast_name in config.dataset.contrasts:
        subject_curves = []
        times_s = None
        chance_level = None

        for _, subject_dir in subject_result_dirs:
            result_path = subject_dir / f"{contrast_name}_decoding.npz"
            if not result_path.exists():
                continue
            result = load_decoding_result(result_path)
            subject_curves.append(result.scores)
            times_s = result.times_s
            chance_level = result.chance_level

        if len(subject_curves) >= 2 and times_s is not None and chance_level is not None:
            subject_curves_array = np.array(subject_curves)
            stats = run_group_statistics(subject_curves_array, chance_level, config)
            save_statistics(stats, save_dir / f"{contrast_name}_group_decoding_stats.json")
            plot_group_time_series(
                subject_scores=subject_curves_array,
                times_s=times_s,
                chance_level=chance_level,
                contrast_name=contrast_name,
                mode="decoding",
                config=config,
                save_dir=save_dir,
            )

        if not with_heterorc:
            continue

        heterorc_curves = []
        heterorc_times_s = None
        heterorc_chance_level = None
        for _, subject_dir in subject_result_dirs:
            result_path = subject_dir / f"{contrast_name}_heterorc_decoding.npz"
            if not result_path.exists():
                continue
            result = load_heterorc_decoding_result(result_path)
            heterorc_curves.append(result.scores)
            heterorc_times_s = result.times_s
            heterorc_chance_level = result.chance_level

        if len(heterorc_curves) < 2 or heterorc_times_s is None or heterorc_chance_level is None:
            continue

        heterorc_curves_array = np.array(heterorc_curves)
        stats = run_group_statistics(heterorc_curves_array, heterorc_chance_level, config)
        save_statistics(stats, save_dir / f"{contrast_name}_heterorc_group_decoding_stats.json")
        plot_heterorc_group_time_series(
            subject_scores=heterorc_curves_array,
            times_s=heterorc_times_s,
            chance_level=heterorc_chance_level,
            contrast_name=contrast_name,
            config=config,
            save_dir=save_dir,
        )
