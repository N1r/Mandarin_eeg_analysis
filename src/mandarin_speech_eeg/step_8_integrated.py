"""Step 8: build per-contrast integrated figures from existing method outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import make_config
from .decoding import load_decoding_result
from .heterorc_analysis import load_heterorc_decoding_result
from .plotting import plot_contrast_integrated
from .rsa import load_rsa_result
from .statistics import StatisticalTestResult, load_statistics
from .step_common import BATCH_ROOT, force_single_worker_environment, print_json, write_json


def run_step(
    *,
    group: str = "Production_Perception",
    subject: str = "p06",
    task: str = "production",
    input_dir: str | Path | None = None,
    figure_dir: str | Path | None = None,
    show_rsa_noise_ceiling: bool = False,
) -> dict:
    force_single_worker_environment()
    config = make_config(task=task)

    subject_results_dir = Path(input_dir).resolve() if input_dir else (
        BATCH_ROOT / "results" / group / task / subject
    )
    subject_figure_dir = Path(figure_dir).resolve() if figure_dir else (
        BATCH_ROOT / "figures" / group / task / subject
    )
    subject_figure_dir.mkdir(parents=True, exist_ok=True)

    speech_onset_ms = _load_speech_onset(subject_results_dir, subject)
    generated: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}

    for contrast_name in config.dataset.contrasts:
        decoding_result = _load_optional(subject_results_dir / f"{contrast_name}_decoding.npz", load_decoding_result)
        rsa_result = _load_optional(subject_results_dir / f"{contrast_name}_rsa.npz", load_rsa_result)
        heterorc_result = _load_optional(
            subject_results_dir / f"{contrast_name}_heterorc_decoding.npz",
            load_heterorc_decoding_result,
        )

        decoding_stats = _load_optional_stats(subject_results_dir / f"{contrast_name}_decoding_stats.json")
        rsa_stats = _load_optional_stats(subject_results_dir / f"{contrast_name}_rsa_stats.json")
        heterorc_stats = _load_optional_stats(subject_results_dir / f"{contrast_name}_heterorc_decoding_stats.json")

        missing[contrast_name] = []
        if decoding_result is None:
            missing[contrast_name].append("decoding")
        if rsa_result is None:
            missing[contrast_name].append("rsa")
        if heterorc_result is None:
            missing[contrast_name].append("heterorc")

        before = set(subject_figure_dir.glob("*"))
        plot_contrast_integrated(
            contrast_name=contrast_name,
            decoding_result=decoding_result,
            rsa_result=rsa_result,
            heterorc_result=heterorc_result,
            decoding_stats=decoding_stats,
            rsa_stats=rsa_stats,
            heterorc_stats=heterorc_stats,
            config=config,
            save_dir=subject_figure_dir,
            speech_onset_ms=speech_onset_ms,
            show_rsa_noise_ceiling=show_rsa_noise_ceiling,
        )
        after = set(subject_figure_dir.glob("*"))
        generated[contrast_name] = [
            str(path) for path in sorted(after - before)
            if path.stem == f"{_stem(contrast_name)}_integrated"
        ]

    summary = {
        "step": "integrated_figures",
        "group": group,
        "task": task,
        "subject": subject,
        "input_dir": str(subject_results_dir),
        "figure_dir": str(subject_figure_dir),
        "speech_onset_ms": speech_onset_ms,
        "show_rsa_noise_ceiling": bool(show_rsa_noise_ceiling),
        "generated": generated,
        "missing_modalities": missing,
    }
    summary_path = subject_results_dir / f"{subject}_integrated_summary.json"
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    print_json(summary)
    return summary


def _load_optional(path: Path, loader):
    if not path.exists():
        return None
    return loader(path)


def _load_optional_stats(path: Path) -> StatisticalTestResult | None:
    if not path.exists():
        return None
    return load_statistics(path)


def _load_speech_onset(subject_results_dir: Path, subject: str) -> float | None:
    for name in (
        f"{subject}_decoding_summary.json",
        f"{subject}_rsa_summary.json",
        f"{subject}_heterorc_summary.json",
    ):
        path = subject_results_dir / name
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8")).get("speech_onset_ms")
                return None if value is None else float(value)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
    return None


def _stem(text: str) -> str:
    return text.replace(" ", "_").replace("|", "_").replace("/", "_").replace("\\", "_").lower()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build one integrated three-panel figure per contrast.")
    parser.add_argument("--group", default="Production_Perception")
    parser.add_argument("--subject", default="p06")
    parser.add_argument("--task", default="production", choices=["production", "perception"])
    parser.add_argument("--input-dir", default=None, help="Directory containing *_decoding/rsa/heterorc outputs.")
    parser.add_argument("--figure-dir", default=None)
    parser.add_argument(
        "--show-rsa-noise-ceiling",
        action="store_true",
        help="Draw RSA noise ceiling in single-subject integrated figures.",
    )
    args = parser.parse_args(argv)

    run_step(
        group=args.group,
        subject=args.subject,
        task=args.task,
        input_dir=args.input_dir,
        figure_dir=args.figure_dir,
        show_rsa_noise_ceiling=args.show_rsa_noise_ceiling,
    )


if __name__ == "__main__":
    main()
