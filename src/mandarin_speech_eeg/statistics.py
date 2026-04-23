"""统计检验。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from mne.stats import permutation_cluster_1samp_test

from .config import AnalysisConfig


class HasStatisticsInput(Protocol):
    """供统计函数使用的最小结果接口。"""

    scores: np.ndarray
    null_distribution: np.ndarray


@dataclass
class StatisticalTestResult:
    """统计结果。"""

    pointwise_p_values: np.ndarray = field(default_factory=lambda: np.empty(0))
    cluster_masks: list[np.ndarray] = field(default_factory=list)
    cluster_p_values: np.ndarray = field(default_factory=lambda: np.empty(0))
    summary: dict[str, object] = field(default_factory=dict)


def run_statistics(result: HasStatisticsInput, config: AnalysisConfig) -> StatisticalTestResult:
    """对单个时间序列结果做置换统计。"""
    if not config.statistics.enabled or result.null_distribution.size == 0:
        return StatisticalTestResult(
            pointwise_p_values=np.ones(len(result.scores)),
            summary={"statistics_enabled": False, "n_significant_clusters": 0},
        )

    pointwise_p = _compute_pointwise_p_values(result.scores, result.null_distribution)
    z_scores = _compute_z_scores(result.scores, result.null_distribution)
    clusters, cluster_p_values = _cluster_permutation_test(
        z_scores,
        result.null_distribution,
        config.statistics.cluster_threshold_z,
    )

    significant_clusters = [
        cluster for cluster, p_value in zip(clusters, cluster_p_values) if p_value < config.statistics.alpha
    ]
    return StatisticalTestResult(
        pointwise_p_values=pointwise_p,
        cluster_masks=significant_clusters,
        cluster_p_values=np.array(cluster_p_values),
        summary={
            "statistics_enabled": True,
            "any_significant_cluster": len(significant_clusters) > 0,
            "n_significant_clusters": len(significant_clusters),
        },
    )


def run_group_statistics(
    subject_curves: np.ndarray,
    chance_level: float,
    config: AnalysisConfig,
) -> StatisticalTestResult:
    """对组水平曲线做符号翻转检验。"""
    centered_curves = subject_curves - chance_level
    observed_scores = centered_curves.mean(axis=0)
    null_distribution = _build_group_null_distribution(centered_curves, config)
    group_result = _GroupStatisticsInput(scores=observed_scores, null_distribution=null_distribution)
    return run_statistics(group_result, config)


def run_group_cluster_statistics(
    observation_curves: np.ndarray,
    chance_level: float,
    config: AnalysisConfig,
    *,
    observation_level: str = "subject",
) -> StatisticalTestResult:
    """Run MNE's 1-sample cluster permutation test across time."""
    n_times = observation_curves.shape[-1] if observation_curves.ndim else 0
    if not config.statistics.enabled or observation_curves.size == 0:
        return StatisticalTestResult(
            pointwise_p_values=np.ones(n_times),
            summary={"statistics_enabled": False, "n_significant_clusters": 0},
        )

    centered_curves = np.asarray(observation_curves, dtype=float) - float(chance_level)
    if centered_curves.ndim != 2:
        raise ValueError("Group statistics expects shape (n_observations, n_times).")
    n_observations, n_times = centered_curves.shape
    if n_observations < 2:
        return StatisticalTestResult(
            pointwise_p_values=np.ones(n_times),
            summary={
                "statistics_enabled": False,
                "n_observations": int(n_observations),
                "n_significant_clusters": 0,
                "reason": "fewer than two observations",
            },
        )

    _, clusters, cluster_p_values, h0 = permutation_cluster_1samp_test(
        centered_curves,
        threshold=None,
        n_permutations=config.statistics.n_permutations,
        tail=1,
        seed=config.reproducibility.random_seed,
        n_jobs=1,
        out_type="indices",
        verbose=False,
    )
    cluster_masks = [_cluster_to_indices(cluster, n_times) for cluster in clusters]
    significant_clusters = [
        cluster for cluster, p_value in zip(cluster_masks, cluster_p_values)
        if p_value < config.statistics.alpha
    ]
    return StatisticalTestResult(
        pointwise_p_values=np.ones(n_times),
        cluster_masks=cluster_masks,
        cluster_p_values=np.asarray(cluster_p_values, dtype=float),
        summary={
            "statistics_enabled": True,
            "method": "mne.stats.permutation_cluster_1samp_test",
            "tail": 1,
            "threshold": "auto",
            "observation_level": observation_level,
            "n_observations": int(n_observations),
            "n_permutations": int(config.statistics.n_permutations),
            "any_significant_cluster": len(significant_clusters) > 0,
            "n_significant_clusters": len(significant_clusters),
            "n_clusters": int(len(cluster_masks)),
            "h0_size": int(len(h0)),
        },
    )


def _cluster_to_indices(cluster: object, n_times: int) -> np.ndarray:
    if isinstance(cluster, tuple):
        return np.asarray(cluster[0], dtype=int)
    if isinstance(cluster, slice):
        return np.arange(n_times, dtype=int)[cluster]
    return np.asarray(cluster, dtype=int)


@dataclass
class _GroupStatisticsInput:
    scores: np.ndarray
    null_distribution: np.ndarray


def _compute_pointwise_p_values(scores: np.ndarray, null_distribution: np.ndarray) -> np.ndarray:
    return (1.0 + np.sum(null_distribution >= scores[None, :], axis=0)) / (
        1.0 + len(null_distribution)
    )


def _compute_z_scores(scores: np.ndarray, null_distribution: np.ndarray) -> np.ndarray:
    null_mean = null_distribution.mean(axis=0)
    null_std = null_distribution.std(axis=0) + 1e-12
    return (scores - null_mean) / null_std


def _cluster_permutation_test(
    z_scores: np.ndarray,
    null_distribution: np.ndarray,
    threshold: float,
) -> tuple[list[np.ndarray], list[float]]:
    clusters = _find_clusters(z_scores > threshold)
    if not clusters:
        return [], []

    observed_masses = [z_scores[cluster].sum() for cluster in clusters]
    null_max_masses = _build_null_max_cluster_masses(null_distribution, threshold)
    cluster_p_values = [
        (1.0 + np.sum(null_max_masses >= mass)) / (1.0 + len(null_max_masses))
        for mass in observed_masses
    ]
    return clusters, cluster_p_values


def _find_clusters(active_mask: np.ndarray) -> list[np.ndarray]:
    clusters: list[np.ndarray] = []
    current_indices: list[int] = []

    for index, is_active in enumerate(active_mask):
        if is_active:
            current_indices.append(index)
            continue
        if current_indices:
            clusters.append(np.array(current_indices))
            current_indices = []

    if current_indices:
        clusters.append(np.array(current_indices))

    return clusters


def _build_null_max_cluster_masses(null_distribution: np.ndarray, threshold: float) -> np.ndarray:
    null_mean = null_distribution.mean(axis=0)
    null_std = null_distribution.std(axis=0) + 1e-12
    max_masses = np.zeros(len(null_distribution))

    for index, null_curve in enumerate(null_distribution):
        z_curve = (null_curve - null_mean) / null_std
        masses = [z_curve[cluster].sum() for cluster in _find_clusters(z_curve > threshold)]
        max_masses[index] = max(masses) if masses else 0.0

    return max_masses


def _build_group_null_distribution(
    centered_curves: np.ndarray,
    config: AnalysisConfig,
) -> np.ndarray:
    n_subjects, n_times = centered_curves.shape
    n_permutations = config.statistics.n_permutations
    rng = np.random.default_rng(config.reproducibility.random_seed)
    null_distribution = np.empty((n_permutations, n_times))

    for permutation_index in range(n_permutations):
        flips = rng.choice((-1.0, 1.0), size=(n_subjects, 1))
        null_distribution[permutation_index] = (centered_curves * flips).mean(axis=0)

    return null_distribution


def save_statistics(result: StatisticalTestResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "pointwise_p_values": result.pointwise_p_values.tolist(),
                "cluster_masks": [mask.tolist() for mask in result.cluster_masks],
                "cluster_p_values": result.cluster_p_values.tolist(),
                "summary": result.summary,
            },
            file,
            indent=2,
        )


def load_statistics(path: str | Path) -> StatisticalTestResult:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    return StatisticalTestResult(
        pointwise_p_values=np.array(data["pointwise_p_values"]),
        cluster_masks=[np.array(mask) for mask in data["cluster_masks"]],
        cluster_p_values=np.array(data["cluster_p_values"]),
        summary=data["summary"],
    )
