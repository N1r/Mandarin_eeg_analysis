"""HeteroRC bridge integrated into the local EEG analysis framework."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mne
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from .config import AnalysisConfig


@dataclass
class HeteroRCDecodingResult:
    scores: np.ndarray
    times_s: np.ndarray
    chance_level: float
    null_distribution: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    fold_scores: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))


def run_heterorc_decoding(
    config: AnalysisConfig,
    epochs: mne.Epochs,
    save_dir: str | Path | None = None,
    quick_mode: bool = False,
) -> dict[str, HeteroRCDecodingResult]:
    heterorc_mod, _ = _import_heterorc_modules(config)
    results: dict[str, HeteroRCDecodingResult] = {}

    X_full = epochs.get_data()
    times_s = epochs.times.copy()
    fs = float(epochs.info["sfreq"])
    rc_params = _rc_params(config, fs, quick_mode)

    for contrast_name, column_name in config.dataset.contrasts.items():
        labels = epochs.metadata[column_name].to_numpy()
        keep_mask = np.array([value == value for value in labels], dtype=bool)
        if keep_mask.sum() < 2:
            continue

        encoder = LabelEncoder()
        y = encoder.fit_transform(labels[keep_mask])
        if len(np.unique(y)) < 2 or _min_class_count(y) < 2:
            continue

        scores, fold_scores, window_times = _cross_validated_heterorc_scores(
            config=config,
            heterorc_mod=heterorc_mod,
            X=X_full[keep_mask],
            y=y,
            times_s=times_s,
            rc_params=rc_params,
        )

        result = HeteroRCDecodingResult(
            scores=scores,
            times_s=window_times,
            chance_level=1.0 / len(encoder.classes_),
            fold_scores=fold_scores,
        )
        results[contrast_name] = result
        if save_dir is not None:
            save_heterorc_decoding_result(result, Path(save_dir) / f"{contrast_name}_heterorc_decoding.npz")

    return results


def save_heterorc_decoding_result(result: HeteroRCDecodingResult, path: str | Path) -> None:
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


def load_heterorc_decoding_result(path: str | Path) -> HeteroRCDecodingResult:
    data = np.load(Path(path), allow_pickle=False)
    return HeteroRCDecodingResult(
        scores=data["scores"],
        times_s=data["times_s"],
        chance_level=float(data["chance_level"]),
        null_distribution=data["null_distribution"],
        fold_scores=data["fold_scores"],
    )


def run_heterorc_interpretation(
    config: AnalysisConfig,
    epochs: mne.Epochs,
    save_dir: str | Path,
    quick_mode: bool = False,
) -> list[Path]:
    heterorc_mod, interpretation_mod = _import_heterorc_modules(config)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    X_full = epochs.get_data()
    times_s = epochs.times.copy()
    fs = float(epochs.info["sfreq"])
    rc_params = _rc_params(config, fs, quick_mode)
    output_paths: list[Path] = []

    for contrast_name, column_name in config.dataset.contrasts.items():
        labels = epochs.metadata[column_name].to_numpy()
        keep_mask = np.array([value == value for value in labels], dtype=bool)
        if keep_mask.sum() < 2:
            continue

        encoder = LabelEncoder()
        y = encoder.fit_transform(labels[keep_mask])
        if len(np.unique(y)) < 2 or _min_class_count(y) < 2:
            continue

        scores, _, window_times = _cross_validated_heterorc_scores(
            config=config,
            heterorc_mod=heterorc_mod,
            X=X_full[keep_mask],
            y=y,
            times_s=times_s,
            rc_params=rc_params,
        )
        peak_idx_window = int(np.argmax(scores))
        peak_time = float(window_times[peak_idx_window])
        peak_idx_raw = int(np.argmin(np.abs(times_s - peak_time)))

        scale_val = np.percentile(np.abs(X_full[keep_mask]), config.heterorc.scale_percentile)
        if not np.isfinite(scale_val) or scale_val == 0:
            scale_val = 1.0
        X_scaled = X_full[keep_mask] / scale_val

        esn = heterorc_mod.HeteroRC(
            n_in=X_scaled.shape[1],
            random_state=config.reproducibility.random_seed,
            **rc_params,
        )
        S_full = esn.transform(X_scaled)
        S_windowed, _ = _apply_temporal_window(
            S_full,
            times_s,
            fs,
            config.heterorc.temporal_window_ms,
            config.heterorc.temporal_step_ms,
        )
        clf = _build_readout_pipeline(config.heterorc.readout_decoder, config.reproducibility.random_seed)
        clf.fit(S_windowed[:, :, peak_idx_window], y)

        results = interpretation_mod.analyze_dynamics(
            esn=esn,
            classifier=clf,
            target_time=peak_time,
            state_snapshot=S_full,
            y_labels=y,
            times=times_s,
            n_clusters=config.heterorc.interpretation_n_clusters,
            top_n=config.heterorc.interpretation_top_n,
            phase_name=f"{contrast_name} | {config.task}",
            plot_style="poster",
            erp_range=(float(times_s[0]), float(times_s[-1])),
            erp_baseline_mode="mean",
            erp_baseline_range=(-0.2, 0.0),
            tfr_baseline_mode="logratio",
            tfr_baseline_range=(-0.2, 0.0),
            tfr_freqs=np.arange(2, 35, 1),
            fooof_params={"max_n_peaks": 4, "peak_width_limits": [1, 8]},
            figsize=(18, 12),
            class_names=[str(value) for value in encoder.classes_],
            inline_topomaps=True,
            info=epochs.info,
            raw_X_snapshot=X_full[keep_mask],
            cov_window_half_width=0.1,
            return_results=True,
        )
        figure_path = save_path / f"{_stem(contrast_name)}_heterorc_interpretation.png"
        results["figure"].savefig(figure_path, dpi=160, bbox_inches="tight")
        plt.close(results["figure"])
        output_paths.append(figure_path)

    return output_paths


def _cross_validated_heterorc_scores(
    config: AnalysisConfig,
    heterorc_mod,
    X: np.ndarray,
    y: np.ndarray,
    times_s: np.ndarray,
    rc_params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_splits = _safe_n_splits(config.heterorc.cv_folds, y)
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.reproducibility.random_seed,
    )
    fs = rc_params["fs"]
    fold_scores: np.ndarray | None = None
    window_times: np.ndarray | None = None

    for fold_index, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scale_val = np.percentile(np.abs(X_train), config.heterorc.scale_percentile)
        if not np.isfinite(scale_val) or scale_val == 0:
            scale_val = 1.0
        X_train_s = X_train / scale_val
        X_test_s = X_test / scale_val

        esn = heterorc_mod.HeteroRC(
            n_in=X.shape[1],
            random_state=config.reproducibility.random_seed,
            **rc_params,
        )
        S_train = esn.transform(X_train_s)
        S_test = esn.transform(X_test_s)
        S_train, window_times = _apply_temporal_window(
            S_train,
            times_s,
            fs,
            config.heterorc.temporal_window_ms,
            config.heterorc.temporal_step_ms,
        )
        S_test, _ = _apply_temporal_window(
            S_test,
            times_s,
            fs,
            config.heterorc.temporal_window_ms,
            config.heterorc.temporal_step_ms,
        )

        if fold_scores is None:
            fold_scores = np.zeros((n_splits, S_train.shape[-1]), dtype=float)

        for time_index in range(S_train.shape[-1]):
            clf = _build_readout_pipeline(config.heterorc.readout_decoder, config.reproducibility.random_seed)
            clf.fit(S_train[:, :, time_index], y_train)
            predictions = clf.predict(S_test[:, :, time_index])
            fold_scores[fold_index, time_index] = _score_predictions(config.heterorc.scoring, y_test, predictions)

    if fold_scores is None or window_times is None:
        raise RuntimeError("Failed to compute HeteroRC scores.")
    return fold_scores.mean(axis=0), fold_scores, window_times


def _safe_n_splits(requested: int, labels: np.ndarray) -> int:
    return max(2, min(int(requested), _min_class_count(labels)))


def _min_class_count(labels: np.ndarray) -> int:
    _, counts = np.unique(labels, return_counts=True)
    if counts.size == 0:
        return 0
    return int(counts.min())


def _build_readout_pipeline(decoder: str, random_seed: int):
    classifier = _build_readout_classifier(decoder, random_seed)
    return make_pipeline(StandardScaler(), classifier)


def _build_readout_classifier(decoder: str, random_seed: int):
    decoder = decoder.lower()
    if decoder == "ridge":
        return RidgeClassifierCV(
            alphas=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
            class_weight="balanced",
        )
    if decoder == "svm":
        return SVC(kernel="linear", random_state=random_seed, class_weight="balanced")
    if decoder == "lda":
        return LinearDiscriminantAnalysis()
    if decoder == "logreg":
        return LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=random_seed,
            class_weight="balanced",
        )
    raise ValueError(f"Unsupported HeteroRC readout decoder: {decoder}")


def _score_predictions(scoring: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if scoring == "accuracy":
        return accuracy_score(y_true, y_pred)
    if scoring == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    raise ValueError(f"Unsupported HeteroRC scoring: {scoring}")


def _import_heterorc_modules(config: AnalysisConfig):
    repo_root = Path(config.heterorc.repo_root)
    if not repo_root.exists():
        raise FileNotFoundError(f"HeteroRC repo root not found: {repo_root}")
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    heterorc_mod = importlib.import_module("heterorc")
    interpretation_mod = importlib.import_module("heterorc_interpretation")
    return heterorc_mod, interpretation_mod


def _rc_params(config: AnalysisConfig, fs: float, quick_mode: bool) -> dict:
    return dict(
        n_res=config.heterorc.n_res_for_mode(quick_mode),
        fs=fs,
        spectral_radius=config.heterorc.spectral_radius,
        input_scaling=config.heterorc.input_scaling,
        bias_scaling=config.heterorc.bias_scaling,
        tau_mode=config.heterorc.tau_mode,
        tau_sigma=config.heterorc.tau_sigma,
        tau_min=config.heterorc.tau_min,
        tau_max=config.heterorc.tau_max,
        bidirectional=config.heterorc.bidirectional,
        merge_mode=config.heterorc.merge_mode,
    )


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


def _stem(text: str) -> str:
    return text.replace(" ", "_").replace("|", "_").lower()
