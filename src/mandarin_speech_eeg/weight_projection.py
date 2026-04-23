"""Haufe-transformed classifier weight projection onto sensor space.

Reference: Haufe et al. (2014) *NeuroImage*; Grootswagers et al. (2017, JoCN) §Weight
Projection. Raw linear-classifier weights cannot be interpreted as "this channel
carries information" because they can be nonzero solely to suppress noise. The
activation pattern `A = cov(X) * w` (per time point) is interpretable.

MNE implements this via `get_coef(..., inverse_transform=True)` on a
SlidingEstimator, which already chains through Scaler/PCA when present.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import LinearModel, SlidingEstimator, get_coef
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import AnalysisConfig


@dataclass
class WeightProjectionResult:
    """Activation patterns per channel × time for one contrast."""

    patterns: np.ndarray  # shape (n_channels, n_times)
    times_s: np.ndarray
    channel_names: list[str]


def compute_weight_projection(
    config: AnalysisConfig,
    epochs: mne.Epochs,
    contrast_column: str,
) -> WeightProjectionResult:
    """Fit a per-timepoint LDA on all trials and return the Haufe-transformed pattern.

    Only binary contrasts are supported — LDA weights are a single vector in that case.
    For multi-class contrasts (e.g. 4-way Tone), we collapse by taking the L2 norm
    across class vectors so the topography summarises "how much this channel
    discriminates among classes" at each time point.
    """
    labels = epochs.metadata[contrast_column].to_numpy()
    data = epochs.get_data()  # (n_trials, n_channels, n_times)

    pipeline = make_pipeline(StandardScaler(), LinearModel(LinearDiscriminantAnalysis()))
    sliding = SlidingEstimator(pipeline, scoring=None, n_jobs=1, verbose=False)
    sliding.fit(data, labels)
    patterns = get_coef(sliding, "patterns_", inverse_transform=True)
    patterns = _reduce_classes(patterns)

    channel_names = [
        name for name, kind in zip(epochs.info["ch_names"], epochs.get_channel_types()) if kind == "eeg"
    ]
    patterns = patterns[: len(channel_names)]

    return WeightProjectionResult(
        patterns=patterns,
        times_s=epochs.times.copy(),
        channel_names=channel_names,
    )


def plot_weight_projection_topomaps(
    result: WeightProjectionResult,
    epochs_info: mne.Info,
    config: AnalysisConfig,
    save_dir: str | Path,
    contrast_name: str,
) -> None:
    """Grid of scalp topographies at the configured time points."""
    requested_ms = config.decoding.weight_projection_time_points_ms
    time_points_s = np.array([t / 1000.0 for t in requested_ms])
    # Clip to the available window and drop duplicates after snapping to sample grid.
    time_points_s = time_points_s[
        (time_points_s >= result.times_s[0]) & (time_points_s <= result.times_s[-1])
    ]
    if time_points_s.size == 0:
        return

    snap_indices = [int(np.argmin(np.abs(result.times_s - t))) for t in time_points_s]
    snap_indices = sorted(set(snap_indices))

    info = mne.pick_info(
        epochs_info, mne.pick_channels(epochs_info["ch_names"], include=result.channel_names)
    )
    v_abs = float(np.nanmax(np.abs(result.patterns)))
    if v_abs == 0.0:
        v_abs = 1.0

    n_panels = len(snap_indices)
    figure, axes = plt.subplots(1, n_panels, figsize=(1.8 * n_panels + 0.6, 2.2))
    axes = np.atleast_1d(axes)

    for axis, time_index in zip(axes, snap_indices):
        mne.viz.plot_topomap(
            result.patterns[:, time_index],
            info,
            axes=axis,
            show=False,
            cmap="RdBu_r",
            vlim=(-v_abs, v_abs),
            contours=4,
            sensors=True,
        )
        axis.set_title(f"{result.times_s[time_index] * 1000:.0f} ms", fontsize=9)

    figure.suptitle(
        f"{contrast_name} | {config.task} | Haufe-transformed activation", fontsize=10
    )
    figure.tight_layout()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    stem = _stem(contrast_name)
    for file_format in config.plotting.save_formats:
        figure.savefig(save_path / f"{stem}_weight_topomap.{file_format}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _reduce_classes(patterns: np.ndarray) -> np.ndarray:
    """LinearModel.patterns_ has shape (n_classes, n_features) for multi-class LDA,
    or (n_features,) for binary. Collapse into (n_features,) by L2 norm across classes.
    After get_coef with inverse_transform, we get (n_channels, n_times) for binary
    or (n_channels, n_classes, n_times). We reduce class dim if present.
    """
    if patterns.ndim == 3:
        return np.linalg.norm(patterns, axis=1)
    return patterns


def _stem(text: str) -> str:
    return text.replace(" ", "_").replace("|", "_").lower()
