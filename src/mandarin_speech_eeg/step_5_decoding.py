"""Step 5: run standard decoding and decoding figures from filtered epochs."""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np

from .decoding import run_decoding
from .plotting import plot_decoding_time_series
from .statistics import run_statistics, save_statistics
from .step_common import BATCH_ROOT, build_step_config, print_json, write_json
from .weight_projection import compute_weight_projection, plot_weight_projection_topomaps


def run_step(
    *,
    group: str = "Production_Perception",
    subject: str = "p06",
    task: str = "production",
    epochs_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    figure_dir: str | Path | None = None,
    quick: bool = False,
    with_stats: bool = False,
    with_topomaps: bool = True,
) -> dict:
    config = build_step_config(task=task, group=group)
    config.statistics.enabled = with_stats
    config.statistics.n_jobs = 1

    subject_results_dir = BATCH_ROOT / "results" / group / task / subject
    subject_figure_dir = BATCH_ROOT / "figures" / group / task / subject
    epochs_file = Path(epochs_path).resolve() if epochs_path else (
        subject_results_dir / "epoch_only" / f"{subject}_selection_filtered-epo.fif"
    )
    if not epochs_file.exists():
        raise FileNotFoundError(f"Filtered epochs not found. Run step_4_epoch first: {epochs_file}")

    save_dir = Path(output_dir).resolve() if output_dir else subject_results_dir
    fig_dir = Path(figure_dir).resolve() if figure_dir else subject_figure_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Epochs: {epochs_file}", flush=True)
    epochs = mne.read_epochs(str(epochs_file), preload=True, verbose=False)
    speech_onset_ms = _speech_onset_ms(epochs)

    decoding_results = run_decoding(config, epochs, save_dir=save_dir, quick_mode=quick)
    stats_paths: dict[str, str] = {}
    figure_paths: dict[str, list[str]] = {}

    for contrast_name, result in decoding_results.items():
        stats = run_statistics(result, config)
        stats_path = save_dir / f"{contrast_name}_decoding_stats.json"
        save_statistics(stats, stats_path)
        stats_paths[contrast_name] = str(stats_path)

        before = set(fig_dir.glob("*"))
        plot_decoding_time_series(
            result,
            config,
            fig_dir,
            contrast_name,
            stats=stats,
            speech_onset_ms=speech_onset_ms,
        )
        after = set(fig_dir.glob("*"))
        figure_paths.setdefault(contrast_name, []).extend(str(path) for path in sorted(after - before))

    topomap_errors: dict[str, str] = {}
    if with_topomaps:
        for contrast_name, column_name in config.dataset.contrasts.items():
            try:
                projection = compute_weight_projection(config, epochs, column_name)
                before = set(fig_dir.glob("*"))
                plot_weight_projection_topomaps(projection, epochs.info, config, fig_dir, contrast_name)
                after = set(fig_dir.glob("*"))
                figure_paths.setdefault(contrast_name, []).extend(str(path) for path in sorted(after - before))
            except Exception as exc:
                topomap_errors[contrast_name] = str(exc)

    summary = {
        "step": "decoding",
        "group": group,
        "task": task,
        "subject": subject,
        "epochs_path": str(epochs_file),
        "n_epochs": int(len(epochs)),
        "n_times": int(len(epochs.times)),
        "statistics_enabled": bool(config.statistics.enabled),
        "speech_onset_ms": speech_onset_ms,
        "decoding_result_paths": {
            contrast: str(save_dir / f"{contrast}_decoding.npz")
            for contrast in decoding_results
        },
        "stats_paths": stats_paths,
        "figure_paths": figure_paths,
        "topomap_errors": topomap_errors,
        "output_dir": str(save_dir),
        "figure_dir": str(fig_dir),
    }
    summary_path = save_dir / f"{subject}_decoding_summary.json"
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    print_json(summary)
    return summary


def _speech_onset_ms(epochs: mne.Epochs) -> float | None:
    metadata = epochs.metadata
    if metadata is None or "detected_latency_ms" not in metadata.columns:
        return None
    values = metadata["detected_latency_ms"].dropna().astype(float).to_numpy()
    if values.size == 0:
        return None
    values = np.sort(values)
    trim = int(values.size * 0.1)
    if trim and values.size > 2 * trim:
        values = values[trim:-trim]
    return float(np.mean(values))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run decoding and decoding plots from filtered epochs.")
    parser.add_argument("--group", default="Production_Perception")
    parser.add_argument("--subject", default="p06")
    parser.add_argument("--task", default="production", choices=["production", "perception"])
    parser.add_argument("--epochs", default=None, help="Explicit filtered epochs FIF path.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--figure-dir", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--with-stats", action="store_true", help="Enable permutation statistics.")
    parser.add_argument("--no-topomaps", action="store_true", help="Skip Haufe weight topomaps.")
    args = parser.parse_args(argv)

    run_step(
        group=args.group,
        subject=args.subject,
        task=args.task,
        epochs_path=args.epochs,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        quick=args.quick,
        with_stats=args.with_stats,
        with_topomaps=not args.no_topomaps,
    )


if __name__ == "__main__":
    main()
