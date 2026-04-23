"""EEG 预处理。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from .config import AnalysisConfig, PreprocessingConfig


PREPROCESSING_CACHE_VERSION = "event-before-resample-drop-status-v2"


@dataclass
class ChannelLayout:
    """解析后的通道布局。"""

    eeg_channels: list[str]
    eog_channels: list[str]
    status_channel: str


@dataclass
class PreprocessingResult:
    """预处理输出。"""

    epochs: mne.Epochs
    log: list[dict[str, Any]] = field(default_factory=list)
    cache_used: bool = False


def preprocess_session(
    config: AnalysisConfig,
    bdf_path: str | Path,
    marker_df: pd.DataFrame,
    session_name: str,
    trial_manifest_df: pd.DataFrame | None = None,
) -> PreprocessingResult:
    """执行单个 session 的预处理。"""
    bdf_path = Path(bdf_path)
    selection_signature = _trial_manifest_signature(trial_manifest_df)
    epochs_path, log_path = _cache_paths(config, session_name, bdf_path, selection_signature)

    if config.preprocessing.use_epoch_cache and epochs_path.exists() and log_path.exists():
        return _load_cached_result(epochs_path, log_path)

    epochs, log = _run_pipeline(config, bdf_path, marker_df, session_name, trial_manifest_df=trial_manifest_df)
    _save_cached_result(epochs_path, log_path, epochs, log)
    return PreprocessingResult(epochs=epochs, log=[log], cache_used=False)


def load_preprocessed_epochs(path: str | Path) -> mne.Epochs:
    """读取保存好的 epochs。"""
    return mne.read_epochs(str(path), preload=True, verbose=False)


def _run_pipeline(
    config: AnalysisConfig,
    bdf_path: Path,
    marker_df: pd.DataFrame,
    session_name: str,
    trial_manifest_df: pd.DataFrame | None = None,
) -> tuple[mne.Epochs, dict[str, Any]]:
    raw = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose=False)
    _rename_channels(raw, config.preprocessing)
    layout = _resolve_layout(raw, config.preprocessing)

    _prepare_channels(raw, layout)
    original_sfreq = float(raw.info["sfreq"])
    raw_events = _find_status_events(raw, layout.status_channel)
    raw.drop_channels([layout.status_channel])
    excluded_ica = _clean_with_ica(raw, config)
    events = _rescale_event_samples(raw_events, original_sfreq, float(raw.info["sfreq"]))
    if trial_manifest_df is None:
        events, metadata = _extract_events(events, marker_df)
        selection_log = {
            "selection_mode": "legacy",
            "selection_skipped_reason": "legacy dataset",
        }
    else:
        events, metadata, selection_log = _extract_events_with_manifest(
            events,
            marker_df,
            trial_manifest_df,
        )
    epochs = _build_epochs(raw, events, metadata, config.preprocessing)
    cleaned_epochs = _clean_epochs(epochs, config)

    log = {
        "session_name": session_name,
        "bdf_path": str(bdf_path),
        "excluded_ica_components": excluded_ica,
        **selection_log,
        "n_events": len(epochs),
        "n_clean_epochs": len(cleaned_epochs),
    }
    return cleaned_epochs, log


def _rename_channels(raw: mne.io.BaseRaw, config: PreprocessingConfig) -> None:
    rename_map: dict[str, str] = {}

    for channel_name in raw.ch_names:
        if channel_name.upper() in {"FP1", "FP2"}:
            rename_map[channel_name] = channel_name.upper().title()

    for canonical_name, aliases in config.eog_aliases.items():
        alias_set = {alias.upper() for alias in aliases}
        for channel_name in raw.ch_names:
            if channel_name.upper() in alias_set:
                rename_map[channel_name] = canonical_name

    if rename_map:
        raw.rename_channels(rename_map)


def _resolve_layout(raw: mne.io.BaseRaw, config: PreprocessingConfig) -> ChannelLayout:
    eeg_channels = [name for name in config.eeg_channels if name in raw.ch_names]
    eog_channels = [name for name in config.eog_channels if name in raw.ch_names]

    status_channel = next(
        (name for name in config.status_channels if name in raw.ch_names),
        None,
    )
    if status_channel is None:
        raise ValueError("未找到触发通道")
    if len(eeg_channels) < config.minimum_expected_eeg_channels:
        raise ValueError("EEG 通道数不足")

    return ChannelLayout(
        eeg_channels=eeg_channels,
        eog_channels=eog_channels,
        status_channel=status_channel,
    )


def _prepare_channels(raw: mne.io.BaseRaw, layout: ChannelLayout) -> None:
    if layout.eog_channels:
        raw.set_channel_types({name: "eog" for name in layout.eog_channels})

    raw.pick([*layout.eeg_channels, *layout.eog_channels, layout.status_channel])
    raw.set_montage("standard_1020")


def _clean_with_ica(raw: mne.io.BaseRaw, config: AnalysisConfig) -> list[int]:
    preprocessing = config.preprocessing
    raw.filter(
        preprocessing.highpass_hz,
        preprocessing.lowpass_hz,
        phase=preprocessing.filter_phase,
        verbose=False,
    )

    if preprocessing.skip_ica:
        raw.resample(preprocessing.resample_hz, verbose=False)
        return []

    from mne.preprocessing import ICA

    ica = ICA(
        n_components=preprocessing.ica_n_components,
        method=preprocessing.ica_method,
        random_state=config.reproducibility.random_seed,
    )
    ica.fit(raw, picks="eeg", verbose=False)

    reference_channels = [
        name for name in preprocessing.ica_eog_reference_channels if name in raw.ch_names
    ]
    excluded_components: list[int] = []
    if reference_channels:
        excluded_components, _ = ica.find_bads_eog(
            raw,
            ch_name=reference_channels,
            threshold=preprocessing.ica_eog_threshold,
            verbose=False,
        )

    ica.exclude = excluded_components
    ica.apply(raw, verbose=False)
    raw.resample(preprocessing.resample_hz, verbose=False)
    return excluded_components


def _find_status_events(raw: mne.io.BaseRaw, status_channel: str) -> np.ndarray:
    events = mne.find_events(raw, stim_channel=status_channel, verbose=False)
    events[:, 2] %= 256
    return events


def _rescale_event_samples(
    events: np.ndarray,
    old_sfreq: float,
    new_sfreq: float,
) -> np.ndarray:
    if np.isclose(old_sfreq, new_sfreq):
        return events.copy()
    scaled = events.copy()
    scaled[:, 0] = np.rint(scaled[:, 0].astype(float) * new_sfreq / old_sfreq).astype(int)
    return scaled


def _extract_events(
    events: np.ndarray,
    marker_df: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    valid_markers = marker_df["marker"].to_numpy()
    keep_mask = np.isin(events[:, 2], valid_markers)
    filtered_events = events[keep_mask]
    metadata = marker_df.set_index("marker").loc[filtered_events[:, 2]].reset_index()
    return filtered_events, metadata


def _extract_events_with_manifest(
    events: np.ndarray,
    marker_df: pd.DataFrame,
    trial_manifest_df: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    valid_markers = pd.to_numeric(marker_df["marker"], errors="coerce").dropna().astype(int).to_numpy()
    eeg_events = events[np.isin(events[:, 2], valid_markers)]

    formal = trial_manifest_df.copy()
    if "global_trial" in formal.columns:
        formal["global_trial"] = pd.to_numeric(formal["global_trial"], errors="coerce")
        formal = formal.sort_values(["global_trial", "block", "trial"], kind="stable")
    formal = formal.reset_index(drop=True)

    if "trigger" not in formal.columns and "marker" in formal.columns:
        formal["trigger"] = formal["marker"]
    if "trigger" not in formal.columns:
        raise ValueError("Trial manifest must include a trigger or marker column.")

    manifest_triggers = pd.to_numeric(formal["trigger"], errors="coerce").astype("Int64")
    if manifest_triggers.isna().any():
        bad_rows = manifest_triggers[manifest_triggers.isna()].index.tolist()[:5]
        raise ValueError(f"Trial manifest has non-numeric trigger values at rows: {bad_rows}")

    if len(eeg_events) != len(formal):
        raise ValueError(
            "EEG event count does not match formal trial count before epoching: "
            f"events={len(eeg_events)}, formal_trials={len(formal)}"
        )

    expected = manifest_triggers.astype(int).to_numpy()
    observed = eeg_events[:, 2].astype(int)
    mismatch_indices = np.flatnonzero(observed != expected)
    if len(mismatch_indices):
        idx = int(mismatch_indices[0])
        global_trial = formal.loc[idx, "global_trial"] if "global_trial" in formal.columns else idx
        raise ValueError(
            "EEG trigger order does not match formal trial order before epoching: "
            f"row={idx}, global_trial={global_trial}, eeg_trigger={observed[idx]}, "
            f"manifest_trigger={expected[idx]}"
        )

    if "keep_trial" not in formal.columns:
        raise ValueError("Trial manifest must include keep_trial before event filtering.")
    keep_mask = pd.to_numeric(formal["keep_trial"], errors="coerce").fillna(0).astype(int).to_numpy() == 1

    metadata = formal.loc[keep_mask].reset_index(drop=True).copy()
    if "marker" not in metadata.columns:
        metadata["marker"] = metadata["trigger"]

    # The manifest carries trial identifiers but not stimulus-level labels
    # (character, animacy, ...). Those live in the legacy marker_condition.csv
    # and must be joined on trigger/marker so downstream decoding / RSA that
    # reference them still work on modern sessions.
    marker_lookup = marker_df.copy()
    marker_lookup["marker"] = pd.to_numeric(marker_lookup["marker"], errors="coerce").astype(int)
    manifest_cols = set(metadata.columns)
    enrich_cols = [
        column for column in marker_lookup.columns
        if column != "marker" and column not in manifest_cols
    ]
    if enrich_cols:
        metadata = metadata.merge(
            marker_lookup[["marker", *enrich_cols]], on="marker", how="left"
        )

    drop_reasons: dict[str, int] = {}
    if "drop_reason" in formal.columns:
        dropped_reasons = formal.loc[~keep_mask, "drop_reason"].fillna("").replace("", "unspecified")
        drop_reasons = {str(key): int(value) for key, value in dropped_reasons.value_counts().to_dict().items()}

    log = {
        "selection_mode": "modern",
        "n_manifest_trials": int(len(formal)),
        "n_manifest_keep": int(keep_mask.sum()),
        "n_manifest_drop": int((~keep_mask).sum()),
        "drop_reason_counts": drop_reasons,
        "selection_version": str(formal["selection_version"].iloc[0])
        if "selection_version" in formal.columns and len(formal)
        else "",
    }
    return eeg_events[keep_mask], metadata, log


def _build_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    metadata: pd.DataFrame,
    config: PreprocessingConfig,
) -> mne.Epochs:
    return mne.Epochs(
        raw,
        events,
        tmin=config.epoch_tmin_s,
        tmax=config.epoch_tmax_s,
        baseline=None,
        preload=True,
        metadata=metadata,
        verbose=False,
    )


def _clean_epochs(epochs: mne.Epochs, config: AnalysisConfig) -> mne.Epochs:
    preprocessing = config.preprocessing
    if preprocessing.skip_autoreject:
        cleaned_epochs = epochs.copy()
    else:
        from autoreject import AutoReject

        autoreject = AutoReject(
            n_interpolate=list(preprocessing.autoreject_interpolate),
            consensus=list(preprocessing.autoreject_consensus),
            random_state=config.reproducibility.random_seed,
            n_jobs=config.statistics.n_jobs,
            verbose=False,
        )
        cleaned_epochs = autoreject.fit_transform(epochs)
    cleaned_epochs.set_eeg_reference("average", verbose=False)
    cleaned_epochs.pick("eeg")
    if preprocessing.baseline_window_s is not None:
        cleaned_epochs.apply_baseline(preprocessing.baseline_window_s, verbose=False)
    return cleaned_epochs


def _cache_paths(
    config: AnalysisConfig,
    session_name: str,
    bdf_path: Path,
    selection_signature: str,
) -> tuple[Path, Path]:
    preprocessing = config.preprocessing
    signature = "_".join(
        [
            PREPROCESSING_CACHE_VERSION,
            config.task,
            session_name,
            str(bdf_path.stat().st_mtime_ns),
            str(preprocessing.highpass_hz),
            str(preprocessing.lowpass_hz),
            str(preprocessing.resample_hz),
            str(preprocessing.filter_phase),
            str(preprocessing.epoch_tmin_s),
            str(preprocessing.epoch_tmax_s),
            str(preprocessing.baseline_window_s),
            str(preprocessing.ica_n_components),
            str(preprocessing.ica_method),
            str(preprocessing.ica_eog_threshold),
            str(preprocessing.autoreject_interpolate),
            str(preprocessing.autoreject_consensus),
            str(preprocessing.skip_ica),
            str(preprocessing.skip_autoreject),
            str(config.reproducibility.random_seed),
            selection_signature,
        ]
    )
    cache_hash = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    cache_dir = config.paths.cache_dir / session_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{session_name}_{cache_hash}"
    return cache_dir / f"{stem}-epo.fif", cache_dir / f"{stem}-log.json"


def _trial_manifest_signature(trial_manifest_df: pd.DataFrame | None) -> str:
    if trial_manifest_df is None:
        return "selection=legacy"
    signature_cols = [
        col
        for col in ["global_trial", "block", "trial", "trigger", "keep_trial", "drop_reason", "selection_version"]
        if col in trial_manifest_df.columns
    ]
    if not signature_cols:
        return f"selection=modern-emptycols-{len(trial_manifest_df)}"
    csv_bytes = trial_manifest_df[signature_cols].to_csv(index=False).encode("utf-8")
    digest = hashlib.sha1(csv_bytes).hexdigest()[:12]
    return f"selection=modern-{digest}"


def _load_cached_result(epochs_path: Path, log_path: Path) -> PreprocessingResult:
    epochs = mne.read_epochs(str(epochs_path), preload=True, verbose=False)
    with log_path.open("r", encoding="utf-8") as file:
        log = json.load(file)
    return PreprocessingResult(epochs=epochs, log=[log], cache_used=True)


def _save_cached_result(
    epochs_path: Path,
    log_path: Path,
    epochs: mne.Epochs,
    log: dict[str, Any],
) -> None:
    epochs.save(str(epochs_path), overwrite=True, verbose=False)
    with log_path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(log), file, indent=2)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return value
