"""分析包配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
ANALYSIS_ROOT = SRC_ROOT.parent
PROJECT_ROOT = ANALYSIS_ROOT.parent.parent

DATA_ROOT = PROJECT_ROOT / "Data"
RESULTS_ROOT = PROJECT_ROOT / "results"

EEG_CHANNELS = (
    "Fp1",
    "AF3",
    "F7",
    "F3",
    "FC1",
    "FC5",
    "T7",
    "C3",
    "CP1",
    "CP5",
    "P7",
    "P3",
    "Pz",
    "PO3",
    "O1",
    "Oz",
    "O2",
    "PO4",
    "P4",
    "P8",
    "CP6",
    "CP2",
    "C4",
    "T8",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "AF4",
    "Fp2",
    "Fz",
    "Cz",
)
EOG_CHANNELS = ("HEOG1", "HEOG2", "VEOG1", "VEOG2")
EOG_ALIASES = {
    "HEOG1": ("HEOG1", "HEOG", "EXG1"),
    "HEOG2": ("HEOG2", "EXG2"),
    "VEOG1": ("VEOG1", "VEOG", "EXG3"),
    "VEOG2": ("VEOG2", "EXG4"),
}
STATUS_CHANNELS = ("Status", "STATUS", "status", "Trigger", "TRIGGER", "trigger")
DEFAULT_CONTRASTS = {
    "Tone": "tone",
    "Animacy": "animacy",
    "Initial Type": "initial_type",
    "Rhyme Type": "rhyme_type",
}


@dataclass
class PathConfig:
    """所有输入输出路径统一放在这里。"""

    marker_csv: Path = DATA_ROOT / "marker_condition.csv"
    results_dir: Path = RESULTS_ROOT / "results"
    figures_dir: Path = RESULTS_ROOT / "figures"
    cache_dir: Path = RESULTS_ROOT / "cache" / "epochs"

    def session_dir(self, session_name: str) -> Path:
        return self.results_dir / session_name

    def subject_figure_dir(self, session_name: str) -> Path:
        return self.figures_dir / session_name

    def with_roots(
        self,
        *,
        results_dir: Path | None = None,
        figures_dir: Path | None = None,
        cache_dir: Path | None = None,
    ) -> "PathConfig":
        """返回一个替换了输出根目录的新路径配置。"""
        return replace(
            self,
            results_dir=results_dir or self.results_dir,
            figures_dir=figures_dir or self.figures_dir,
            cache_dir=cache_dir or self.cache_dir,
        )

    def ensure_directories(self) -> None:
        for directory in (self.results_dir, self.figures_dir, self.cache_dir):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class PreprocessingConfig:
    highpass_hz: float = 0.5
    lowpass_hz: float = 40.0
    resample_hz: int = 256
    filter_phase: str = "minimum"
    epoch_tmin_s: float = -0.3
    epoch_tmax_s: float = 1.8
    baseline_window_s: tuple[float, float] | None = (-0.2, 0.0)
    ica_n_components: int = 20
    ica_method: str = "fastica"
    ica_eog_threshold: float = 2.0
    ica_eog_reference_channels: tuple[str, str] = ("VEOG2", "HEOG2")
    autoreject_interpolate: tuple[int, ...] = (1, 2, 4)
    autoreject_consensus: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
    eeg_channels: tuple[str, ...] = EEG_CHANNELS
    eog_channels: tuple[str, ...] = EOG_CHANNELS
    eog_aliases: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(EOG_ALIASES)
    )
    status_channels: tuple[str, ...] = STATUS_CHANNELS
    minimum_expected_eeg_channels: int = 16
    use_epoch_cache: bool = True
    # v2-minimal switches. When both are True the pipeline reduces to: filter +
    # resample + common-average reref + baseline (no ICA, no autoreject).
    skip_ica: bool = False
    skip_autoreject: bool = False


@dataclass
class DecodingConfig:
    """时间解析解码设置。默认遵循 Grootswagers et al. (2017, JoCN) 建议。"""

    decoder: str = "lda"
    scoring: str = "balanced_accuracy"
    # Plain stratified 10-fold is cheaper than leave-one-exemplar-out and
    # gives equally clean group-level estimates on pilot sizes.
    cv_folds: int = 5
    cv_strategy: str = "stratified_kfold"  # "stratified_kfold" | "leave_one_exemplar_out" | "kfold"
    group_column: str = "character"  # 仅 leave-one-exemplar-out 时使用
    pseudotrial_size: int | None = 4  # 每个伪试次合并的原始 trial 数；None=禁用
    pca_variance: float | None = 0.99  # PCA 保留方差比例；None=禁用
    weight_projection_time_points_ms: tuple[float, ...] = (100.0, 200.0, 350.0, 600.0, 900.0)
    # Small boxcar smoothing: visually removes time-to-time jitter while
    # preserving event-related structure (40 ms is the common MVPA default).
    temporal_window_ms: float | None = 40.0
    temporal_step_ms: float | None = 10.0


@dataclass
class RSAConfig:
    neural_metric: str = "euclidean"
    compare_method: str = "spearman"
    noise_ceiling_splits: int = 50  # split-half 重复次数；0=禁用


@dataclass
class StatisticsConfig:
    enabled: bool = True
    n_permutations: int = 1000
    quick_n_permutations: int = 50
    cluster_threshold_z: float = 1.67
    alpha: float = 0.05
    n_jobs: int = 1


@dataclass
class PlottingConfig:
    stimulus_window_ms: tuple[float, float] = (0.0, 100.0)
    save_formats: tuple[str, ...] = ("png",)
    show_subject_rsa_noise_ceiling: bool = False
    decoding_color_production: str = "#355C8C"
    decoding_color_perception: str = "#C4682D"
    rsa_color_production: str = "#3D7A57"
    rsa_color_perception: str = "#7A5AA6"
    heterorc_color_production: str = "#6F4EAD"
    heterorc_color_perception: str = "#B84C7A"
    tgm_cmap: str = "RdYlBu_r"


@dataclass
class HeteroRCConfig:
    enabled: bool = False
    interpretation_enabled: bool = False
    repo_root: Path = PROJECT_ROOT / "Full_Analysis" / "heterorc-main" / "heterorc-main"
    n_res: int = 180
    quick_n_res: int = 100
    metric: str = "accuracy"
    scoring: str = "balanced_accuracy"
    readout_decoder: str = "ridge"
    cv_folds: int = 5
    scale_percentile: float = 99.0
    smooth_fwhm_ms: float = 25.0
    temporal_window_ms: float | None = None
    temporal_step_ms: float | None = None
    spectral_radius: float = 0.95
    input_scaling: float = 0.5
    bias_scaling: float = 0.5
    tau_mode: float = 0.01
    tau_sigma: float = 0.8
    tau_min: float = 0.002
    tau_max: float = 0.08
    bidirectional: bool = True
    merge_mode: str = "product"
    interpretation_n_clusters: int = 3
    interpretation_top_n: int = 25

    def n_res_for_mode(self, quick_mode: bool) -> int:
        return self.quick_n_res if quick_mode else self.n_res


@dataclass
class DatasetConfig:
    contrasts: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CONTRASTS))


@dataclass
class ReproducibilityConfig:
    random_seed: int = 97
    preset_name: str = "publication"


@dataclass
class AnalysisConfig:
    """分析主配置。"""

    task: str = "production"
    paths: PathConfig = field(default_factory=PathConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    rsa: RSAConfig = field(default_factory=RSAConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    heterorc: HeteroRCConfig = field(default_factory=HeteroRCConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)

    def __post_init__(self) -> None:
        if self.task not in {"production", "perception"}:
            raise ValueError(f"Unsupported task: {self.task}")
        if self.preprocessing.lowpass_hz <= self.preprocessing.highpass_hz:
            raise ValueError("lowpass_hz 必须大于 highpass_hz")
        if self.decoding.cv_folds < 2:
            raise ValueError("cv_folds 至少为 2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_paths(self, paths: PathConfig) -> "AnalysisConfig":
        """返回一个替换了路径配置的新对象。"""
        return replace(self, paths=paths)

    def permutation_count(self, quick_mode: bool) -> int:
        if not self.statistics.enabled:
            return 0
        if quick_mode:
            return self.statistics.quick_n_permutations
        return self.statistics.n_permutations

    @property
    def decoding_color(self) -> str:
        if self.task == "production":
            return self.plotting.decoding_color_production
        return self.plotting.decoding_color_perception

    @property
    def rsa_color(self) -> str:
        if self.task == "production":
            return self.plotting.rsa_color_production
        return self.plotting.rsa_color_perception

    @property
    def heterorc_color(self) -> str:
        if self.task == "production":
            return self.plotting.heterorc_color_production
        return self.plotting.heterorc_color_perception


def make_config(task: str = "production", quick: bool = False) -> AnalysisConfig:
    """创建配置对象。"""
    config = AnalysisConfig(task=task)
    if quick:
        config.statistics.n_permutations = config.statistics.quick_n_permutations
        config.reproducibility.preset_name = "quick"
    return config


def make_minimal_v2_config(task: str = "production", quick: bool = False) -> AnalysisConfig:
    """Minimal preprocessing preset: 0.5-30 Hz, 100 Hz, common-average, no ICA/AR."""
    config = make_config(task=task, quick=quick)
    pre = config.preprocessing
    pre.highpass_hz = 0.5
    pre.lowpass_hz = 30.0
    pre.resample_hz = 100
    pre.filter_phase = "zero"
    pre.baseline_window_s = None
    pre.skip_ica = True
    pre.skip_autoreject = True
    pre.epoch_tmin_s = -0.3
    pre.epoch_tmax_s = 1.8
    config.reproducibility.preset_name = "minimal_v2"
    return config


def disable_statistics(config: AnalysisConfig) -> AnalysisConfig:
    """关闭统计置换。"""
    config.statistics.enabled = False
    config.statistics.n_permutations = 0
    config.statistics.quick_n_permutations = 0
    return config
