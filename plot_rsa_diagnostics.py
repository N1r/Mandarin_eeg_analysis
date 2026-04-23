"""RSA 诊断绘图脚本。

这个脚本专门做三件事：
1. 快速查看几条 RSA 曲线形状是否正常。
2. 查看若干关键时间点的 neural RDM。
3. 查看完整的 model RDM。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mandarin_speech_eeg import disable_statistics, load_rsa_result, make_config, preprocess_session
from mandarin_speech_eeg.rsa import (
    _build_item_patterns,
    _build_item_table,
    _build_model_rdm,
    _compute_neural_rdms,
)


DATA_ROOT = REPO_ROOT.parent.parent / "Data"
BATCH_ROOT = REPO_ROOT.parent / "results" / "batch_analysis"
RESULTS_ROOT = BATCH_ROOT / "results"
FIGURE_ROOT = BATCH_ROOT / "diagnostics" / "rsa"

SUBJECT_GROUPS = {
    "Production_only": {
        "subjects": ["p00", "p01", "p02"],
        "tasks": ["production"],
    },
    "Production_Perception": {
        "subjects": ["p05", "p06", "p07"],
        "tasks": ["production", "perception"],
    },
}

TIME_POINTS_S = [0.0, 0.1, 0.3, 0.6]


@dataclass(frozen=True)
class DiagnosticCase:
    group_name: str
    task: str
    subject: str
    primary_contrast: str = "Tone"


CASES = [
    DiagnosticCase(group_name="Production_only", task="production", subject="p00"),
    DiagnosticCase(group_name="Production_Perception", task="production", subject="p05"),
    DiagnosticCase(group_name="Production_Perception", task="perception", subject="p05"),
]


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for case in CASES:
        saved_paths.extend(plot_case(case))

    print("RSA diagnostics generated:")
    for path in saved_paths:
        print(path)


def plot_case(case: DiagnosticCase) -> list[Path]:
    case_root = FIGURE_ROOT / case.group_name / case.task / case.subject
    case_root.mkdir(parents=True, exist_ok=True)

    rsa_results = load_case_rsa_results(case)
    curve_path = case_root / "rsa_curves.png"
    plot_rsa_curves(case, rsa_results, curve_path)

    config = build_case_config(case)
    marker_df = load_marker_table(config.paths.marker_csv, case.task)
    bdf_path = DATA_ROOT / case.group_name / f"{case.subject}.bdf"
    preprocessing_result = preprocess_session(config, bdf_path, marker_df, case.subject)
    epochs = preprocessing_result.epochs

    primary_column = config.dataset.contrasts[case.primary_contrast]
    neural_rdms, times_s, model_rdm = build_rdms(epochs, config, primary_column)

    neural_rdm_path = case_root / f"{case.primary_contrast.lower()}_neural_rdms.png"
    plot_neural_rdms(
        case=case,
        neural_rdms=neural_rdms,
        times_s=times_s,
        rsa_scores=rsa_results[case.primary_contrast].scores,
        save_path=neural_rdm_path,
    )

    model_rdm_path = case_root / "model_rdms.png"
    plot_model_rdms(case, epochs.metadata.reset_index(drop=True), config, model_rdm_path)

    return [curve_path, neural_rdm_path, model_rdm_path]


def build_case_config(case: DiagnosticCase):
    config = make_config(task=case.task, quick=False)
    config.paths = config.paths.with_roots(
        results_dir=RESULTS_ROOT / case.group_name / case.task,
        figures_dir=BATCH_ROOT / "figures" / case.group_name / case.task,
        cache_dir=BATCH_ROOT / "cache" / case.group_name / case.task,
    )
    disable_statistics(config)
    return config


def load_marker_table(marker_csv: Path, task: str) -> pd.DataFrame:
    marker_df = pd.read_csv(marker_csv)
    if task == "perception":
        marker_df = marker_df.copy()
        marker_df["marker"] = marker_df["marker"] + 100
    return marker_df


def load_case_rsa_results(case: DiagnosticCase):
    subject_dir = RESULTS_ROOT / case.group_name / case.task / case.subject
    rsa_results = {}
    for contrast_name in ["Tone", "Animacy", "Initial Type", "Rhyme Type"]:
        rsa_results[contrast_name] = load_rsa_result(subject_dir / f"{contrast_name}_rsa.npz")
    return rsa_results


def build_rdms(epochs, config, contrast_column: str):
    metadata = epochs.metadata.reset_index(drop=True)
    eeg_data = epochs.get_data()
    item_table = _build_item_table(metadata, "character", contrast_column)
    item_patterns = _build_item_patterns(
        eeg_data=eeg_data,
        metadata=metadata,
        items=item_table["character"].to_list(),
        item_column="character",
    )
    model_rdm = _build_model_rdm(item_table[contrast_column].to_numpy())
    neural_rdms = _compute_neural_rdms(config, item_patterns)
    return neural_rdms, epochs.times.copy(), model_rdm


def plot_rsa_curves(case: DiagnosticCase, rsa_results: dict, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    color_map = {
        "Tone": "#D1495B",
        "Animacy": "#2E86AB",
        "Initial Type": "#3B8B5E",
        "Rhyme Type": "#8E6CBB",
    }
    for contrast_name, result in rsa_results.items():
        ax.plot(result.times_s * 1000, result.scores, label=contrast_name, linewidth=2, color=color_map[contrast_name])

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
    ax.axvline(0.0, color="black", linewidth=1, alpha=0.4, linestyle="--")
    ax.set_title(f"{case.group_name} | {case.task} | {case.subject} | RSA curves")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("RSA (Spearman)")
    ax.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        borderaxespad=0.4,
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_neural_rdms(
    case: DiagnosticCase,
    neural_rdms: np.ndarray,
    times_s: np.ndarray,
    rsa_scores: np.ndarray,
    save_path: Path,
) -> None:
    chosen_indices = select_time_indices(times_s, rsa_scores)
    n_panels = len(chosen_indices)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.3 * n_panels, 4.2), constrained_layout=True)

    if n_panels == 1:
        axes = [axes]

    vmax = np.percentile(neural_rdms, 95)
    for axis, time_index in zip(axes, chosen_indices):
        rdm_matrix = squareform(neural_rdms[time_index])
        image = axis.imshow(rdm_matrix, cmap="gray", vmin=0.0, vmax=vmax)
        axis.set_title(f"{times_s[time_index] * 1000:.0f} ms")
        add_tone_guides(axis, rdm_matrix.shape[0])
        axis.set_xlabel("Tone-ordered item")
        axis.set_ylabel("Tone-ordered item")

    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
    fig.suptitle(f"{case.group_name} | {case.task} | {case.subject} | {case.primary_contrast} neural RDMs")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def select_time_indices(times_s: np.ndarray, rsa_scores: np.ndarray) -> list[int]:
    indices = [nearest_time_index(times_s, time_s) for time_s in TIME_POINTS_S]
    peak_index = int(np.argmax(np.abs(rsa_scores)))
    indices.append(peak_index)

    deduplicated: list[int] = []
    for index in indices:
        if index not in deduplicated:
            deduplicated.append(index)
    return deduplicated


def nearest_time_index(times_s: np.ndarray, target_s: float) -> int:
    return int(np.argmin(np.abs(times_s - target_s)))


def add_tone_guides(axis, item_count: int) -> None:
    """为按 tone 排序的 RDM 添加分块线和分组标签。"""
    block_size = item_count // 4
    if block_size == 0 or block_size * 4 != item_count:
        return

    boundaries = [block_size, block_size * 2, block_size * 3]
    for boundary in boundaries:
        axis.axhline(boundary - 0.5, color="#9A9A9A", linewidth=1.2)
        axis.axvline(boundary - 0.5, color="#9A9A9A", linewidth=1.2)

    centers = [block_size * index + (block_size - 1) / 2 for index in range(4)]
    labels = ["T1", "T2", "T3", "T4"]
    axis.set_xticks(centers, labels)
    axis.set_yticks(centers, labels)


def plot_model_rdms(case: DiagnosticCase, metadata: pd.DataFrame, config, save_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    axes = axes.ravel()

    for axis, (contrast_name, column_name) in zip(axes, config.dataset.contrasts.items()):
        item_table = _build_item_table(metadata, "character", column_name)
        model_rdm = _build_model_rdm(item_table[column_name].to_numpy())
        model_matrix = squareform(model_rdm)
        image = axis.imshow(model_matrix, cmap="magma", vmin=0.0, vmax=1.0)
        axis.set_title(contrast_name)
        axis.set_xlabel("Item")
        axis.set_ylabel("Item")

    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
    fig.suptitle(f"{case.group_name} | {case.task} | {case.subject} | model RDMs")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
