"""Response-locked epoch construction (v2 pipeline).

Given event-locked epochs whose metadata carries a per-trial ``detected_latency_ms``
column (speech onset in ms, measured relative to stimulus onset), re-slice each
trial so that t=0 corresponds to the speech onset. Fast but approximate: the data
are not re-preprocessed; they are simply re-windowed from the existing epochs.
Trials whose response window falls outside the original epoch are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd


@dataclass
class ResponseLockResult:
    epochs: mne.EpochsArray
    n_input: int
    n_kept: int
    n_dropped_missing: int
    n_dropped_out_of_range: int
    rt_stats: dict


def make_response_locked_epochs(
    epochs: mne.Epochs,
    tmin_s: float = -1.0,
    tmax_s: float = 0.5,
    latency_column: str = "detected_latency_ms",
    min_latency_ms: float = 150.0,
    max_latency_ms: float = 2000.0,
    baseline_s: tuple[float, float] | None = None,
) -> ResponseLockResult:
    """Re-window ``epochs`` around per-trial speech onset.

    Parameters
    ----------
    epochs : event-locked epochs; must carry metadata with ``latency_column``.
    tmin_s, tmax_s : window relative to response (s).
    baseline_s : optional (tmin, tmax) baseline within the new window.
    """
    if epochs.metadata is None or latency_column not in epochs.metadata.columns:
        raise ValueError(
            f"epochs.metadata must contain '{latency_column}' for response lock."
        )

    metadata = epochs.metadata.reset_index(drop=True).copy()
    sfreq = float(epochs.info["sfreq"])
    original_tmin = float(epochs.tmin)
    data = epochs.get_data(copy=True)
    n_trials, n_channels, n_times = data.shape

    latencies_ms = pd.to_numeric(metadata[latency_column], errors="coerce").to_numpy(dtype=float)
    valid_latency = np.isfinite(latencies_ms) & (latencies_ms >= min_latency_ms) & (latencies_ms <= max_latency_ms)

    win_samples = int(round((tmax_s - tmin_s) * sfreq)) + 1
    kept_data: list[np.ndarray] = []
    kept_rows: list[int] = []
    kept_rts: list[float] = []
    n_out_of_range = 0
    n_missing = 0

    for trial_index in range(n_trials):
        if not valid_latency[trial_index]:
            n_missing += 1
            continue
        rt_s = latencies_ms[trial_index] / 1000.0
        start_time_s = rt_s + tmin_s
        start_sample = int(round((start_time_s - original_tmin) * sfreq))
        stop_sample = start_sample + win_samples
        if start_sample < 0 or stop_sample > n_times:
            n_out_of_range += 1
            continue
        kept_data.append(data[trial_index, :, start_sample:stop_sample])
        kept_rows.append(trial_index)
        kept_rts.append(rt_s)

    if not kept_data:
        raise RuntimeError(
            "Response lock: no trials survived. Check latency column and epoch window."
        )

    data_array = np.stack(kept_data, axis=0)
    new_metadata = metadata.iloc[kept_rows].reset_index(drop=True).copy()
    new_metadata["response_locked_rt_s"] = kept_rts

    # Rebuild an EpochsArray. Events are dummies; MVPA/RSA only use data+metadata.
    events = np.column_stack(
        [
            np.arange(len(kept_rows)) * win_samples,
            np.zeros(len(kept_rows), dtype=int),
            np.asarray(
                pd.to_numeric(new_metadata.get("marker", pd.Series([1] * len(kept_rows))), errors="coerce")
                .fillna(1)
                .astype(int)
            ),
        ]
    ).astype(int)

    new_epochs = mne.EpochsArray(
        data_array,
        info=epochs.info.copy(),
        events=events,
        tmin=tmin_s,
        metadata=new_metadata,
        verbose=False,
    )
    if baseline_s is not None:
        new_epochs.apply_baseline(baseline_s, verbose=False)

    rts = np.asarray(kept_rts)
    rt_stats = {
        "n": int(rts.size),
        "mean_s": float(rts.mean()),
        "median_s": float(np.median(rts)),
        "sd_s": float(rts.std(ddof=1)) if rts.size > 1 else 0.0,
        "min_s": float(rts.min()),
        "max_s": float(rts.max()),
    }

    return ResponseLockResult(
        epochs=new_epochs,
        n_input=n_trials,
        n_kept=len(kept_rows),
        n_dropped_missing=n_missing,
        n_dropped_out_of_range=n_out_of_range,
        rt_stats=rt_stats,
    )
