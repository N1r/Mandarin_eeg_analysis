"""Step 9: group-level curves and quick statistics from subject outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np

from .config import AnalysisConfig, make_config
from .decoding import load_decoding_result
from .heterorc_analysis import load_heterorc_decoding_result
from .plotting import plot_group_time_series, plot_heterorc_group_time_series
from .rsa import load_rsa_result
from .statistics import run_group_cluster_statistics, save_statistics
from .step_common import BATCH_ROOT, force_single_worker_environment, print_json, write_json


def run_step(
    *,
    group: str = "Production_Perception",
    task: str = "production",
    subjects: list[str] | None = None,
    input_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    figure_dir: str | Path | None = None,
    quick: bool = True,
) -> dict:
    force_single_worker_environment()
    config = make_config(task=task)
    config.statistics.enabled = True
    config.statistics.n_jobs = 1
    if quick:
        config.statistics.n_permutations = config.statistics.quick_n_permutations

    input_base = Path(input_root).resolve() if input_root else (
        BATCH_ROOT / "results" / group / task
    )
    subject_ids = subjects or _discover_subjects(input_base)
    if len(subject_ids) < 2:
        raise ValueError("Group analysis requires at least two subjects.")

    save_dir = Path(output_dir).resolve() if output_dir else input_base / "group"
    fig_dir = Path(figure_dir).resolve() if figure_dir else (
        BATCH_ROOT / "figures" / group / task / "group"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, dict[str, str]] = {}
    for mode, suffix, loader in _MODES:
        outputs[mode] = _run_mode(
            mode=mode,
            suffix=suffix,
            loader=loader,
            config=config,
            input_base=input_base,
            subjects=subject_ids,
            save_dir=save_dir,
            fig_dir=fig_dir,
        )

    summary = {
        "step": "group",
        "group": group,
        "task": task,
        "subjects": subject_ids,
        "n_subjects": len(subject_ids),
        "quick": bool(quick),
        "n_permutations": int(config.statistics.n_permutations),
        "input_root": str(input_base),
        "output_dir": str(save_dir),
        "figure_dir": str(fig_dir),
        "outputs": outputs,
        "note": "Small-n group statistics are for workflow validation and pilot visualization, not final inference.",
    }
    summary_path = save_dir / "group_summary.json"
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    print_json(summary)
    return summary


def _run_mode(
    *,
    mode: str,
    suffix: str,
    loader: Callable[[Path], object],
    config: AnalysisConfig,
    input_base: Path,
    subjects: list[str],
    save_dir: Path,
    fig_dir: Path,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for contrast_name in config.dataset.contrasts:
        loaded = []
        used_subjects = []
        for subject in subjects:
            path = input_base / subject / f"{contrast_name}_{suffix}.npz"
            if path.exists():
                loaded.append(loader(path))
                used_subjects.append(subject)
        if len(loaded) < 2:
            continue

        times_s = loaded[0].times_s
        chance_level = float(loaded[0].chance_level)
        scores = np.stack([result.scores for result in loaded], axis=0)
        if any(not np.array_equal(result.times_s, times_s) for result in loaded):
            raise ValueError(f"Time axis mismatch for {mode} {contrast_name}.")
        if any(float(result.chance_level) != chance_level for result in loaded):
            raise ValueError(f"Chance-level mismatch for {mode} {contrast_name}.")

        observation_scores, observation_level = _group_observations(loaded, scores)
        stats = run_group_cluster_statistics(
            observation_scores,
            chance_level,
            config,
            observation_level=observation_level,
        )
        stem = _stem(contrast_name)
        result_path = save_dir / f"{stem}_{mode}_group.npz"
        stats_path = save_dir / f"{stem}_{mode}_group_stats.json"
        np.savez(
            result_path,
            subject_scores=scores,
            observation_scores=observation_scores,
            mean=scores.mean(axis=0),
            sem=scores.std(axis=0) / np.sqrt(scores.shape[0]),
            observation_mean=observation_scores.mean(axis=0),
            observation_sem=observation_scores.std(axis=0) / np.sqrt(observation_scores.shape[0]),
            times_s=times_s,
            chance_level=chance_level,
            subjects=np.array(used_subjects),
            observation_level=observation_level,
        )
        save_statistics(stats, stats_path)

        if mode == "heterorc":
            plot_heterorc_group_time_series(
                observation_scores,
                times_s,
                chance_level,
                contrast_name,
                config,
                fig_dir,
                stats=stats,
            )
        else:
            plot_group_time_series(
                observation_scores,
                times_s,
                chance_level,
                contrast_name,
                mode,
                config,
                fig_dir,
                stats=stats,
            )

        outputs[f"{contrast_name}:result"] = str(result_path)
        outputs[f"{contrast_name}:stats"] = str(stats_path)
    return outputs


def _group_observations(loaded: list[object], subject_scores: np.ndarray) -> tuple[np.ndarray, str]:
    fold_scores = [
        np.asarray(getattr(result, "fold_scores", np.empty((0, 0))), dtype=float)
        for result in loaded
    ]
    if fold_scores and all(scores.ndim == 2 and scores.size for scores in fold_scores):
        return np.concatenate(fold_scores, axis=0), "cv_fold"
    return subject_scores, "subject"


def _discover_subjects(input_base: Path) -> list[str]:
    return sorted(
        path.name for path in input_base.iterdir()
        if path.is_dir() and path.name.lower() != "group"
    )


def _stem(text: str) -> str:
    return text.replace(" ", "_").replace("|", "_").replace("/", "_").replace("\\", "_").lower()


_MODES = (
    ("decoding", "decoding", load_decoding_result),
    ("rsa", "rsa", load_rsa_result),
    ("heterorc", "heterorc_decoding", load_heterorc_decoding_result),
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run group-level curves and quick stats.")
    parser.add_argument("--group", default="Production_Perception")
    parser.add_argument("--task", default="production", choices=["production", "perception"])
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--figure-dir", default=None)
    parser.add_argument("--full-stats", action="store_true", help="Use full configured permutations instead of quick stats.")
    args = parser.parse_args(argv)

    run_step(
        group=args.group,
        task=args.task,
        subjects=args.subjects,
        input_root=args.input_root,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        quick=not args.full_stats,
    )


if __name__ == "__main__":
    main()
