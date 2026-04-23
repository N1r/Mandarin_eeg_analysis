"""RSA 分析。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from .config import AnalysisConfig


@dataclass
class RSAResult:
    """单个 contrast 的 RSA 结果。

    `noise_ceiling` 是 (lower, upper) 两条曲线（长度 == 时间点数）。
    采用 split-half：把 trial 随机二等分，分别算 RDM，再与模型相关 → 取均值得下界；
    再把两半 RDM 相互相关 → Spearman-Brown 校正得上界。
    """

    scores: np.ndarray
    times_s: np.ndarray
    chance_level: float = 0.0
    null_distribution: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    noise_ceiling: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))


def run_rsa(
    config: AnalysisConfig,
    epochs: mne.Epochs,
    save_dir: str | Path | None = None,
    quick_mode: bool = False,
) -> dict[str, RSAResult]:
    """运行所有 contrast 的 RSA。"""
    if epochs.metadata is None:
        raise ValueError("RSA 需要 epochs.metadata")

    eeg_data = epochs.get_data()
    metadata = epochs.metadata.reset_index(drop=True)
    times_s = epochs.times.copy()
    results: dict[str, RSAResult] = {}

    for contrast_name, column_name in config.dataset.contrasts.items():
        result = _run_single_contrast(
            config=config,
            eeg_data=eeg_data,
            metadata=metadata,
            item_column="character",
            contrast_column=column_name,
            times_s=times_s,
            quick_mode=quick_mode,
        )
        results[contrast_name] = result
        if save_dir is not None:
            save_rsa_result(result, Path(save_dir) / f"{contrast_name}_rsa.npz")

    return results


def save_rsa_result(result: RSAResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        scores=result.scores,
        times_s=result.times_s,
        chance_level=result.chance_level,
        null_distribution=result.null_distribution,
        noise_ceiling=result.noise_ceiling,
    )


def load_rsa_result(path: str | Path) -> RSAResult:
    data = np.load(Path(path), allow_pickle=False)
    noise_ceiling = data["noise_ceiling"] if "noise_ceiling" in data.files else np.empty((0, 0))
    return RSAResult(
        scores=data["scores"],
        times_s=data["times_s"],
        chance_level=float(data["chance_level"]),
        null_distribution=data["null_distribution"],
        noise_ceiling=noise_ceiling,
    )


def _run_single_contrast(
    config: AnalysisConfig,
    eeg_data: np.ndarray,
    metadata: pd.DataFrame,
    item_column: str,
    contrast_column: str,
    times_s: np.ndarray,
    quick_mode: bool,
) -> RSAResult:
    item_table = _build_item_table(metadata, item_column, contrast_column)
    items = item_table[item_column].to_list()
    item_patterns = _build_item_patterns(eeg_data, metadata, items, item_column)
    model_rdm = _build_model_rdm(item_table[contrast_column].to_numpy())
    neural_rdms = _compute_neural_rdms(config, item_patterns)
    observed_scores = _correlate_rdms(neural_rdms, model_rdm)
    null_distribution = _build_null_distribution(config, neural_rdms, item_table[contrast_column].to_numpy(), quick_mode)
    noise_ceiling = _estimate_noise_ceiling(config, eeg_data, metadata, items, item_column, model_rdm)

    return RSAResult(
        scores=observed_scores,
        times_s=times_s,
        chance_level=0.0,
        null_distribution=null_distribution,
        noise_ceiling=noise_ceiling,
    )


def _build_item_table(metadata: pd.DataFrame, item_column: str, contrast_column: str) -> pd.DataFrame:
    selected_columns = [column for column in ("marker", item_column, contrast_column) if column in metadata.columns]
    item_table = metadata[selected_columns].drop_duplicates(subset=[item_column], keep="first").copy()
    if item_table.empty:
        raise ValueError(f"RSA 未找到 {item_column} / {contrast_column} 对应的项目表")
    if item_table[contrast_column].nunique() < 2:
        raise ValueError(f"RSA contrast {contrast_column} 只有一个类别，无法建立模型 RDM")
    if "marker" in item_table.columns:
        item_table = item_table.sort_values("marker", kind="stable")
    return item_table.reset_index(drop=True)


def _build_item_patterns(
    eeg_data: np.ndarray,
    metadata: pd.DataFrame,
    items: list[str],
    item_column: str,
) -> np.ndarray:
    n_items = len(items)
    n_channels = eeg_data.shape[1]
    n_times = eeg_data.shape[2]
    patterns = np.empty((n_items, n_channels, n_times), dtype=float)

    for item_index, item_name in enumerate(items):
        item_mask = metadata[item_column].to_numpy() == item_name
        if not np.any(item_mask):
            raise ValueError(f"RSA item {item_name} 没有对应 trial")
        patterns[item_index] = eeg_data[item_mask].mean(axis=0)

    return patterns


def _build_model_rdm(labels: np.ndarray) -> np.ndarray:
    encoded_labels = pd.Categorical(labels).codes.astype(float)
    model_rdm = pdist(encoded_labels[:, None], metric="hamming")
    if np.allclose(model_rdm, model_rdm[0]):
        raise ValueError("RSA 模型 RDM 没有变异，无法进行相关分析")
    return model_rdm


def _compute_neural_rdms(config: AnalysisConfig, item_patterns: np.ndarray) -> np.ndarray:
    n_times = item_patterns.shape[2]
    rdm_length = len(pdist(item_patterns[:, :, 0], metric=config.rsa.neural_metric))
    neural_rdms = np.empty((n_times, rdm_length), dtype=float)

    for time_index in range(n_times):
        neural_rdms[time_index] = pdist(
            item_patterns[:, :, time_index],
            metric=config.rsa.neural_metric,
        )

    return neural_rdms


def _correlate_rdms(neural_rdms: np.ndarray, model_rdm: np.ndarray) -> np.ndarray:
    correlations = np.empty(neural_rdms.shape[0], dtype=float)
    for time_index, neural_rdm in enumerate(neural_rdms):
        correlations[time_index] = spearmanr(neural_rdm, model_rdm).statistic
    return np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)


def _estimate_noise_ceiling(
    config: AnalysisConfig,
    eeg_data: np.ndarray,
    metadata: pd.DataFrame,
    items: list[str],
    item_column: str,
    model_rdm: np.ndarray,
) -> np.ndarray:
    """Split-half 噪声天花板。

    每次重复：把每个 item 的 trial 随机二等分，分别算 neural RDM。
    - upper = 两半 RDM 的 Spearman 相关（Spearman-Brown 校正，越大说明数据稳定性越好）
    - lower = 两半各自与 model RDM 的相关的均值（保守估计）
    返回形状 (2, n_times)：[lower, upper]。
    """
    n_splits = config.rsa.noise_ceiling_splits
    if n_splits <= 0:
        return np.empty((0, 0))

    rng = np.random.default_rng(config.reproducibility.random_seed)
    n_items = len(items)
    n_channels = eeg_data.shape[1]
    n_times = eeg_data.shape[2]

    # Pre-index trials per item for speed.
    item_to_trials = {
        name: np.nonzero(metadata[item_column].to_numpy() == name)[0] for name in items
    }
    # Require at least 2 trials per item to form halves; else bail out.
    if any(len(idx) < 2 for idx in item_to_trials.values()):
        return np.empty((0, 0))

    split_upper = np.empty((n_splits, n_times))
    split_lower = np.empty((n_splits, n_times))

    for split_index in range(n_splits):
        patterns_a = np.empty((n_items, n_channels, n_times))
        patterns_b = np.empty((n_items, n_channels, n_times))
        for item_index, name in enumerate(items):
            trial_indices = item_to_trials[name].copy()
            rng.shuffle(trial_indices)
            mid = len(trial_indices) // 2
            patterns_a[item_index] = eeg_data[trial_indices[:mid]].mean(axis=0)
            patterns_b[item_index] = eeg_data[trial_indices[mid : 2 * mid]].mean(axis=0)

        rdms_a = _compute_neural_rdms(config, patterns_a)
        rdms_b = _compute_neural_rdms(config, patterns_b)

        raw_reliability = np.array(
            [spearmanr(rdms_a[t], rdms_b[t]).statistic for t in range(n_times)]
        )
        raw_reliability = np.nan_to_num(raw_reliability, nan=0.0)
        # Spearman-Brown: r_full = 2r / (1 + r)
        split_upper[split_index] = (2.0 * raw_reliability) / (1.0 + np.clip(raw_reliability, -0.999, 0.999))

        corr_a = _correlate_rdms(rdms_a, model_rdm)
        corr_b = _correlate_rdms(rdms_b, model_rdm)
        split_lower[split_index] = 0.5 * (corr_a + corr_b)

    lower = split_lower.mean(axis=0)
    upper = split_upper.mean(axis=0)
    return np.stack([lower, upper], axis=0)


def _build_null_distribution(
    config: AnalysisConfig,
    neural_rdms: np.ndarray,
    labels: np.ndarray,
    quick_mode: bool,
) -> np.ndarray:
    n_permutations = config.permutation_count(quick_mode)
    if n_permutations == 0:
        return np.empty((0, neural_rdms.shape[0]))

    rng = np.random.default_rng(config.reproducibility.random_seed)
    null_scores = np.empty((n_permutations, neural_rdms.shape[0]), dtype=float)

    for permutation_index in range(n_permutations):
        shuffled_labels = rng.permutation(labels)
        shuffled_model_rdm = _build_model_rdm(shuffled_labels)
        null_scores[permutation_index] = _correlate_rdms(neural_rdms, shuffled_model_rdm)

    return null_scores
