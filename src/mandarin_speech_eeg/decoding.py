"""Time-resolved decoding analysis.

Pipeline follows Grootswagers, Wardle & Carlson (2017, JoCN):
- Pseudotrial averaging within exemplar (SNR boost).
- PCA inside the CV loop (variance-retention based).
- Leave-one-exemplar-out CV to avoid exemplar identity confounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import AnalysisConfig


@dataclass
class DecodingResult:
    scores: np.ndarray
    times_s: np.ndarray
    chance_level: float
    null_distribution: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    fold_scores: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))


def run_decoding(
    config: AnalysisConfig,
    epochs: mne.Epochs,
    save_dir: str | Path | None = None,
    quick_mode: bool = False,
) -> dict[str, DecodingResult]:
    eeg_data = epochs.get_data()
    metadata = epochs.metadata.reset_index(drop=True) if epochs.metadata is not None else None
    times_s = epochs.times.copy()
    sfreq = float(epochs.info["sfreq"])

    eeg_data, times_s = _apply_temporal_window(
        eeg_data,
        times_s,
        sfreq,
        config.decoding.temporal_window_ms,
        config.decoding.temporal_step_ms,
    )

    results: dict[str, DecodingResult] = {}
    for contrast_name, column_name in config.dataset.contrasts.items():
        labels = epochs.metadata[column_name].to_numpy()
        groups = _extract_groups(metadata, config.decoding.group_column)
        result = _run_single_contrast(config, eeg_data, labels, groups, times_s, quick_mode)
        results[contrast_name] = result
        if save_dir is not None:
            save_decoding_result(result, Path(save_dir) / f"{contrast_name}_decoding.npz")

    return results


def save_decoding_result(result: DecodingResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        scores=result.scores,
        times_s=result.times_s,
        chance_level=result.chance_level,
        null_distribution=result.null_distribution,
        fold_scores=result.fold_scores,
    )


def load_decoding_result(path: str | Path) -> DecodingResult:
    data = np.load(Path(path), allow_pickle=False)
    return DecodingResult(
        scores=data["scores"],
        times_s=data["times_s"],
        chance_level=float(data["chance_level"]),
        null_distribution=data["null_distribution"],
        fold_scores=data["fold_scores"],
    )


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def _run_single_contrast(
    config: AnalysisConfig,
    eeg_data: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None,
    times_s: np.ndarray,
    quick_mode: bool,
) -> DecodingResult:
    seed = config.reproducibility.random_seed
    rng = np.random.default_rng(seed)

    # 1) Pseudotrial averaging (SNR boost, tutorial Fig. 8)
    eeg_data, labels, groups = _apply_pseudotrials(
        eeg_data, labels, groups, config.decoding.pseudotrial_size, rng
    )

    # 2) Build estimator and CV
    estimator = _build_estimator(config)
    cv, cv_groups = _build_cv(config, labels, groups, seed)

    # 3) Cross-validated scoring. Keep joblib single-worker for stability.
    fold_scores = cross_val_multiscore(
        estimator,
        eeg_data,
        labels,
        cv=cv,
        groups=cv_groups,
        n_jobs=config.statistics.n_jobs,
    )
    observed_scores = fold_scores.mean(axis=0)
    null_distribution = _build_null_distribution(
        config, estimator, eeg_data, labels, cv, cv_groups, quick_mode
    )

    return DecodingResult(
        scores=observed_scores,
        times_s=times_s,
        chance_level=1.0 / len(np.unique(labels)),
        null_distribution=null_distribution,
        fold_scores=fold_scores,
    )


def _build_estimator(config: AnalysisConfig) -> SlidingEstimator:
    steps = [StandardScaler()]
    if config.decoding.pca_variance is not None:
        steps.append(PCA(n_components=config.decoding.pca_variance, svd_solver="full"))
    steps.append(_build_classifier(config.decoding.decoder, config.reproducibility.random_seed))
    pipeline = make_pipeline(*steps)
    return SlidingEstimator(pipeline, scoring=config.decoding.scoring, n_jobs=1)


def _build_classifier(decoder: str, random_seed: int):
    decoder = decoder.lower()
    if decoder == "lda":
        # `lsqr` with shrinkage is markedly faster than the default SVD solver
        # and gives better-regularised fits on high-dim EEG features.
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if decoder == "gnb":
        return GaussianNB()
    if decoder == "svm":
        return SVC(kernel="linear", random_state=random_seed, class_weight="balanced")
    if decoder == "logreg":
        return LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=random_seed,
            class_weight="balanced",
        )
    if decoder == "ridge":
        return RidgeClassifier(class_weight="balanced")
    raise ValueError(f"Unsupported decoder: {decoder}")


def _build_cv(
    config: AnalysisConfig,
    labels: np.ndarray,
    groups: np.ndarray | None,
    seed: int,
) -> tuple[object, np.ndarray | None]:
    strategy = config.decoding.cv_strategy
    can_use_groups = groups is not None and len(np.unique(groups)) >= 2
    n_splits = _safe_n_splits(config.decoding.cv_folds, labels)

    if strategy == "leave_one_exemplar_out" and can_use_groups:
        # Guard against degenerate cases where a single exemplar maps 1:1 to a class.
        if _groups_span_multiple_classes(labels, groups):
            return LeaveOneGroupOut(), groups
        try:
            return (
                StratifiedGroupKFold(n_splits=min(n_splits, len(np.unique(groups))),
                                     shuffle=True, random_state=seed),
                groups,
            )
        except ValueError:
            pass  # fall through to plain KFold

    return (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        None,
    )


def _safe_n_splits(requested: int, labels: np.ndarray) -> int:
    _, counts = np.unique(labels, return_counts=True)
    if counts.size < 2:
        raise ValueError("Decoding requires at least two classes.")
    return max(2, min(int(requested), int(counts.min())))


def _groups_span_multiple_classes(labels: np.ndarray, groups: np.ndarray) -> bool:
    """Return True when at least one group contains trials of multiple classes.

    When every group maps to a single class (e.g. one character → one Tone), leaving
    that group out means the classifier never trained on its class mapping, so use
    a stratified group-kfold instead to keep each fold class-balanced.
    """
    df = pd.DataFrame({"label": labels, "group": groups})
    class_count_per_group = df.groupby("group")["label"].nunique()
    return bool((class_count_per_group > 1).any())


def _extract_groups(metadata: pd.DataFrame | None, column: str) -> np.ndarray | None:
    if metadata is None or column not in metadata.columns:
        return None
    return metadata[column].to_numpy()


def _apply_pseudotrials(
    eeg_data: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None,
    size: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Average every `size` trials sharing the same (label, group) to form pseudotrials.

    When `groups` is None, averaging is done within label only. Pseudotrial labels
    inherit from the source trials; groups (if provided) are preserved so downstream
    CV can still leave an exemplar out.
    """
    if not size or size <= 1:
        return eeg_data, labels, groups

    n_trials = len(labels)
    indices = np.arange(n_trials)
    key = labels if groups is None else np.array([f"{l}|{g}" for l, g in zip(labels, groups)])
    new_trials: list[np.ndarray] = []
    new_labels: list = []
    new_groups: list = [] if groups is not None else None

    for unique_key in pd.unique(key):
        bucket = indices[key == unique_key]
        rng.shuffle(bucket)
        for start in range(0, len(bucket) - size + 1, size):
            chunk = bucket[start : start + size]
            new_trials.append(eeg_data[chunk].mean(axis=0))
            new_labels.append(labels[chunk[0]])
            if new_groups is not None:
                new_groups.append(groups[chunk[0]])

    if not new_trials:
        # Not enough trials to form a pseudotrial; fall back to raw trials.
        return eeg_data, labels, groups

    return (
        np.stack(new_trials, axis=0),
        np.array(new_labels),
        np.array(new_groups) if new_groups is not None else None,
    )


def _build_null_distribution(
    config: AnalysisConfig,
    estimator: SlidingEstimator,
    eeg_data: np.ndarray,
    labels: np.ndarray,
    cv,
    cv_groups: np.ndarray | None,
    quick_mode: bool,
) -> np.ndarray:
    n_permutations = config.permutation_count(quick_mode)
    if n_permutations == 0:
        return np.empty((0, eeg_data.shape[2]))

    rng = np.random.default_rng(config.reproducibility.random_seed)
    # Pre-materialise shuffled labels so workers don't share RNG state.
    shuffled = [rng.permutation(labels) for _ in range(n_permutations)]

    # Keep permutation scoring single-worker; nested parallelism caused hangs.
    def _one(shuffled_labels: np.ndarray) -> np.ndarray:
        scores = cross_val_multiscore(
            estimator, eeg_data, shuffled_labels,
            cv=cv, groups=cv_groups, n_jobs=1,
        )
        return scores.mean(axis=0)

    results = Parallel(n_jobs=config.statistics.n_jobs, backend="loky")(
        delayed(_one)(s) for s in shuffled
    )
    return np.stack(results, axis=0)


def _apply_temporal_window(
    data: np.ndarray,
    times_s: np.ndarray,
    sfreq: float,
    window_ms: float | None,
    step_ms: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not window_ms or window_ms <= (1000.0 / sfreq):
        return data, times_s

    window_samples = max(1, int(round(window_ms * sfreq / 1000.0)))
    step_samples = max(1, int(round((step_ms or window_ms) * sfreq / 1000.0)))
    n_times = data.shape[-1]
    if window_samples >= n_times:
        return data.mean(axis=-1, keepdims=True), np.array([times_s.mean()])

    starts = range(0, n_times - window_samples + 1, step_samples)
    windowed = [data[..., start : start + window_samples].mean(axis=-1) for start in starts]
    window_times = [times_s[start : start + window_samples].mean() for start in starts]
    return np.stack(windowed, axis=-1), np.asarray(window_times)
