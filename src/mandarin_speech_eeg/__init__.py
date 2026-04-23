"""Mandarin speech EEG analysis package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .config import (
    AnalysisConfig,
    DATA_ROOT,
    RESULTS_ROOT,
    disable_statistics,
    make_config,
    make_minimal_v2_config,
)

_LAZY_EXPORTS = {
    "PreprocessingResult": ("preprocessing", "PreprocessingResult"),
    "preprocess_session": ("preprocessing", "preprocess_session"),
    "load_preprocessed_epochs": ("preprocessing", "load_preprocessed_epochs"),
    "DecodingResult": ("decoding", "DecodingResult"),
    "run_decoding": ("decoding", "run_decoding"),
    "save_decoding_result": ("decoding", "save_decoding_result"),
    "load_decoding_result": ("decoding", "load_decoding_result"),
    "HeteroRCDecodingResult": ("heterorc_analysis", "HeteroRCDecodingResult"),
    "run_heterorc_decoding": ("heterorc_analysis", "run_heterorc_decoding"),
    "save_heterorc_decoding_result": ("heterorc_analysis", "save_heterorc_decoding_result"),
    "load_heterorc_decoding_result": ("heterorc_analysis", "load_heterorc_decoding_result"),
    "plot_heterorc_decoding_time_series": ("plotting", "plot_heterorc_decoding_time_series"),
    "plot_heterorc_group_time_series": ("plotting", "plot_heterorc_group_time_series"),
    "run_heterorc_interpretation": ("heterorc_analysis", "run_heterorc_interpretation"),
    "RSAResult": ("rsa", "RSAResult"),
    "run_rsa": ("rsa", "run_rsa"),
    "save_rsa_result": ("rsa", "save_rsa_result"),
    "load_rsa_result": ("rsa", "load_rsa_result"),
    "StatisticalTestResult": ("statistics", "StatisticalTestResult"),
    "run_statistics": ("statistics", "run_statistics"),
    "run_group_statistics": ("statistics", "run_group_statistics"),
    "save_statistics": ("statistics", "save_statistics"),
    "load_statistics": ("statistics", "load_statistics"),
    "plot_decoding_time_series": ("plotting", "plot_decoding_time_series"),
    "plot_rsa_time_series": ("plotting", "plot_rsa_time_series"),
    "plot_group_time_series": ("plotting", "plot_group_time_series"),
    "plot_time_generalization": ("plotting", "plot_time_generalization"),
    "plot_multi_contrast_overlay": ("plotting", "plot_multi_contrast_overlay"),
    "plot_condition_comparison": ("plotting", "plot_condition_comparison"),
    "figure_stem": ("plotting", "figure_stem"),
    "compute_weight_projection": ("weight_projection", "compute_weight_projection"),
    "plot_weight_projection_topomaps": ("weight_projection", "plot_weight_projection_topomaps"),
    "run_onset_analysis": ("onset", "run_onset_analysis"),
    "run_asr_analysis": ("asr", "run_asr_analysis"),
    "build_trial_manifest": ("trial_selection", "build_trial_manifest"),
    "make_response_locked_epochs": ("response_lock", "make_response_locked_epochs"),
    "ResponseLockResult": ("response_lock", "ResponseLockResult"),
}

__all__ = [
    "AnalysisConfig",
    "make_config",
    "make_minimal_v2_config",
    "disable_statistics",
    "DATA_ROOT",
    "RESULTS_ROOT",
    *_LAZY_EXPORTS.keys(),
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
