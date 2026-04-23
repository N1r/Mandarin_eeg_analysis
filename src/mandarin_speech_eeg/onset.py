"""Onset analysis for session timing logs and continuous audio.

Detection uses two parallel RMS envelopes: a full-band (LF) envelope tracking vowel/
sonorant energy, and a 2–8 kHz high-band (HF) envelope that captures obstruent frication
and aspiration. The earliest-crossing band determines onset, correcting the well-known
late bias for Mandarin unvoiced obstruents (/p t k s sh ch f h c q/) under broadband RMS.
After threshold crossing, onset is back-tracked within the same band to the first frame
exceeding baseline + 1 SD (Praat-style elbow), recovering the fast rise leading up to
the stable peak.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import wave
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .config import PROJECT_ROOT

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


FRAME_MS = 5                # hop between successive envelope frames
RMS_WINDOW_MS = 20          # RMS integration window (overlaps 4× at 5 ms hop)
HF_BAND_HZ = (2000.0, 8000.0)
BASELINE_START_MS = -500
BASELINE_END_MS = -50
SEARCH_START_MS = 100
SEARCH_PADDING_MS = 30
BASELINE_PRE_MS = 500
BASELINE_POST_MS = 50
MIN_LATENCY_MS = 100
MAX_LATENCY_MS = 1800
MIN_ABS_RMS = 0.010
BASELINE_MULTIPLIER = 2.5
STD_MULTIPLIER = 3.0
MIN_CONSEC_FRAMES = 2       # with 5 ms hop this equals 10 ms of sustained activity
BACKTRACK_SD_MULTIPLIER = 1.0
TRIM_PROPORTION = 0.05      # two-sided trim for latency aggregation
DEFAULT_OUTPUT_DIRNAME = "analysis_onset"
ANIMATE_MEANINGS = {
    "mother",
    "cat",
    "girl",
    "donkey",
    "snake",
    "butterfly",
    "monkey",
    "horse",
    "rat",
    "bird",
    "dog",
    "deer",
    "rabbit",
    "leopard",
    "crab",
}


def find_latest_session(base_dir: Path | str) -> Path:
    session_dirs = sorted(Path(base_dir).glob("sub-*/ses-*"))
    if not session_dirs:
        raise SystemExit(f"No session directories found under {base_dir}")
    return session_dirs[-1]


def resolve_session_dir(target: Path | str) -> Path:
    path = Path(target)
    return path if path.name.startswith("ses-") else find_latest_session(path)


def load_csv_rows(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No trial rows found in {path}")
    return rows


def load_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_float(value: str | None) -> float | None:
    if value in (None, "", "NA"):
        return None
    return float(value)


def safe_int(value: str | None) -> int | None:
    if value in (None, "", "NA"):
        return None
    return int(value)


def build_trial_records(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        record["block"] = safe_int(row["block"])
        record["trial"] = safe_int(row["trial"])
        record["global_trial"] = safe_int(row["global_trial"])
        record["tone"] = safe_int(row["tone"])
        record["trigger"] = safe_int(row["trigger"])
        record["audio_t_onset_in_segment"] = safe_float(row["audio_t_onset_in_segment"])
        record["audio_segment_start_time"] = safe_float(row["audio_segment_start_time"])
        record["t_onset"] = safe_float(row["t_onset"])
        record["t_blank_offset"] = safe_float(row["t_blank_offset"])
        records.append(record)
    return records


def group_by_block(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["block"]].append(record)
    for block_records in grouped.values():
        block_records.sort(key=lambda item: item["trial"])
    return grouped


def read_wav_mono(path: Path | str) -> tuple[int, array]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        raw = wav_file.readframes(n_frames)
    if channels != 1 or sample_width != 2:
        raise SystemExit(f"Only mono 16-bit PCM WAV is supported: {path}")
    samples = array("h")
    samples.frombytes(raw)
    return sample_rate, samples


def find_continuous_audio_file(session_dir: Path, block: int, rows: list[dict[str, Any]]) -> Path:
    """Find a block audio file while tolerating older participant name conventions."""
    audio_dir = session_dir / "continuous_audio"
    candidate_names: list[str] = []
    if rows:
        participant = str(rows[0].get("participant", "")).strip()
        if participant:
            candidate_names.append(f"sub-{participant}_block-{block:02d}_continuous.wav")
            if participant.isdigit():
                candidate_names.append(f"sub-{int(participant):02d}_block-{block:02d}_continuous.wav")
    candidate_names.append(f"{session_dir.parent.name}_block-{block:02d}_continuous.wav")

    for name in candidate_names:
        path = audio_dir / name
        if path.exists():
            return path

    matches = sorted(audio_dir.glob(f"*_block-{block:02d}_continuous.wav"))
    if matches:
        if rows:
            participant = str(rows[0].get("participant", "")).strip()
            preferred = [path for path in matches if participant and participant in path.name]
            if preferred:
                return preferred[0]
        return matches[0]

    raise FileNotFoundError(
        f"No continuous audio file found for block {block} under {audio_dir}"
    )


def build_rms_envelope(
    samples: array,
    sample_rate: int,
    frame_ms: int = FRAME_MS,
    window_ms: int = RMS_WINDOW_MS,
    bandpass_hz: tuple[float, float] | None = None,
) -> tuple[list[float], int]:
    """Overlapping-window RMS envelope, optionally band-pass filtered.

    ``bandpass_hz=(low, high)`` applies a zero-phase 4th-order Butterworth band-pass
    before RMS; used for the high-frequency (obstruent-sensitive) envelope. When
    ``None`` the envelope is computed on the raw signal (broadband).
    """
    signal = np.asarray(samples, dtype=np.float32)
    if bandpass_hz is not None:
        low, high = bandpass_hz
        nyq = sample_rate / 2.0
        high = min(high, nyq * 0.99)
        sos = butter(4, [low / nyq, high / nyq], btype="band", output="sos")
        signal = sosfiltfilt(sos, signal).astype(np.float32)

    hop = max(1, int(round(sample_rate * frame_ms / 1000)))
    win = max(hop, int(round(sample_rate * window_ms / 1000)))
    if signal.size < win:
        return [], hop

    # Cumulative-sum trick for O(n) moving RMS over `win`-sample windows.
    squared = (signal.astype(np.float64) ** 2)
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    starts = np.arange(0, signal.size - win + 1, hop)
    energy = (cumulative[starts + win] - cumulative[starts]) / float(win)
    envelope = (np.sqrt(energy) / 32768.0).tolist()
    return envelope, hop


def slice_values(envelope: list[float], start_ms: float, end_ms: float, frame_ms: int = FRAME_MS) -> list[float]:
    start_idx = max(0, int(start_ms // frame_ms))
    end_idx = min(len(envelope), int(math.ceil(end_ms / frame_ms)))
    if end_idx <= start_idx:
        return []
    return envelope[start_idx:end_idx]


def summarize_values(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    sorted_values = sorted(values)
    trimmed_count = max(5, int(len(sorted_values) * 0.8))
    trimmed_values = sorted_values[:trimmed_count]
    mean_value = statistics.fmean(trimmed_values)
    variance = statistics.fmean((value - mean_value) ** 2 for value in trimmed_values)
    return mean_value, math.sqrt(variance)


def _detect_onset_single_band(
    envelope: list[float], onset_ms: float, search_end_ms: float,
) -> dict[str, Any]:
    """Threshold-crossing + back-track for a single envelope band."""
    baseline_values = slice_values(envelope, onset_ms + BASELINE_START_MS, onset_ms + BASELINE_END_MS)
    baseline_mean, baseline_std = summarize_values(baseline_values)
    search_start_ms = onset_ms + SEARCH_START_MS
    search_stop_ms = min(onset_ms + MAX_LATENCY_MS, search_end_ms)

    result: dict[str, Any] = {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "threshold": None,
        "backtrack_threshold": None,
        "detected": False,
        "latency_ms": None,
        "crossing_latency_ms": None,
        "search_start_ms": search_start_ms,
        "search_end_ms": search_stop_ms,
        "peak_rms": None,
        "peak_latency_ms": None,
    }
    if baseline_mean is None or baseline_std is None:
        return result
    if search_stop_ms <= search_start_ms:
        return result

    threshold = max(MIN_ABS_RMS, baseline_mean * BASELINE_MULTIPLIER, baseline_mean + STD_MULTIPLIER * baseline_std)
    backtrack_threshold = baseline_mean + BACKTRACK_SD_MULTIPLIER * baseline_std
    result["threshold"] = threshold
    result["backtrack_threshold"] = backtrack_threshold

    start_idx = max(0, int(search_start_ms // FRAME_MS))
    stop_idx = min(len(envelope), int(math.ceil(search_stop_ms / FRAME_MS)))
    if stop_idx <= start_idx:
        return result

    # Peak within search window (diagnostic only)
    peak_value = None
    peak_time_ms = None
    for frame_index in range(start_idx, stop_idx):
        rms_value = envelope[frame_index]
        if peak_value is None or rms_value > peak_value:
            peak_value = rms_value
            peak_time_ms = frame_index * FRAME_MS
    result["peak_rms"] = peak_value
    result["peak_latency_ms"] = None if peak_time_ms is None else peak_time_ms - onset_ms

    # Locate first threshold crossing sustained for MIN_CONSEC_FRAMES
    consecutive = 0
    crossing_frame: int | None = None
    for frame_index in range(start_idx, stop_idx):
        if envelope[frame_index] >= threshold:
            if consecutive == 0:
                crossing_frame = frame_index
            consecutive += 1
            if consecutive >= MIN_CONSEC_FRAMES:
                break
        else:
            consecutive = 0
            crossing_frame = None
    if consecutive < MIN_CONSEC_FRAMES or crossing_frame is None:
        return result

    crossing_time_ms = crossing_frame * FRAME_MS

    # Back-track: walk backwards from the crossing frame while the envelope remains
    # above (baseline + 1 SD). This recovers the fast rise that precedes the firm
    # threshold and whose start is the perceptually correct onset for obstruents.
    backtrack_limit_idx = max(0, int(search_start_ms // FRAME_MS))
    onset_frame = crossing_frame
    probe = crossing_frame - 1
    while probe >= backtrack_limit_idx and envelope[probe] >= backtrack_threshold:
        onset_frame = probe
        probe -= 1

    onset_time_ms = onset_frame * FRAME_MS
    result.update(
        detected=True,
        latency_ms=onset_time_ms - onset_ms,
        crossing_latency_ms=crossing_time_ms - onset_ms,
    )
    return result


def detect_onset_latency(
    envelope_lf: list[float],
    envelope_hf: list[float],
    onset_ms: float,
    search_end_ms: float,
) -> dict[str, Any]:
    """Parallel two-band detection. Earliest detected latency wins."""
    lf = _detect_onset_single_band(envelope_lf, onset_ms, search_end_ms)
    hf = _detect_onset_single_band(envelope_hf, onset_ms, search_end_ms)

    lf_latency = lf["latency_ms"] if lf["detected"] else None
    hf_latency = hf["latency_ms"] if hf["detected"] else None
    if lf_latency is not None and (hf_latency is None or lf_latency <= hf_latency):
        winner, source = lf, "lf"
    elif hf_latency is not None:
        winner, source = hf, "hf"
    else:
        winner, source = lf, None  # neither band detected — keep LF diagnostics

    return {
        **winner,
        "source_band": source,
        "latency_ms_lf": lf["latency_ms"],
        "latency_ms_hf": hf["latency_ms"],
        "crossing_latency_ms_lf": lf["crossing_latency_ms"],
        "crossing_latency_ms_hf": hf["crossing_latency_ms"],
        "peak_rms_lf": lf["peak_rms"],
        "peak_rms_hf": hf["peak_rms"],
        "baseline_mean_lf": lf["baseline_mean"],
        "baseline_mean_hf": hf["baseline_mean"],
        "threshold_lf": lf["threshold"],
        "threshold_hf": hf["threshold"],
    }


def make_block_time_correction(block_records: list[dict[str, Any]], wav_duration_ms: float) -> dict[str, Any]:
    first_onset_ms = block_records[0]["audio_t_onset_in_segment"]
    csv_segment_start_ms = block_records[0]["audio_segment_start_time"]
    last_trial = block_records[-1]
    last_logged_blank_ms = None
    if last_trial["t_blank_offset"] is not None and csv_segment_start_ms is not None:
        last_logged_blank_ms = last_trial["t_blank_offset"] - csv_segment_start_ms

    wav_end_cap_ms = wav_duration_ms - SEARCH_PADDING_MS

    def correct(raw_ms: float | None) -> float | None:
        return raw_ms

    estimated_linear_scale = None
    if first_onset_ms is not None and last_logged_blank_ms is not None and last_logged_blank_ms > first_onset_ms and wav_end_cap_ms > first_onset_ms:
        estimated_linear_scale = (wav_end_cap_ms - first_onset_ms) / (last_logged_blank_ms - first_onset_ms)

    return {
        "mode": "raw_identity",
        "first_onset_ms": first_onset_ms,
        "last_logged_blank_ms": last_logged_blank_ms,
        "wav_end_cap_ms": wav_end_cap_ms,
        "scale": 1.0,
        "estimated_linear_scale": estimated_linear_scale,
        "correct": correct,
    }


def load_stimuli_table() -> list[dict[str, Any]]:
    stimuli_csv = PROJECT_ROOT / "Experiment" / "stimuli.csv"
    stimuli = []
    with stimuli_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["trigger"] = int(row["trigger"])
            row["tone"] = int(row["tone"])
            stimuli.append(row)
    return stimuli


def build_trial_analysis(session_dir: Path, records: list[dict[str, Any]], session_data: dict[str, Any]) -> list[dict[str, Any]]:
    grouped = group_by_block(records)
    analysis_rows = []
    config = session_data.get("config", {})
    image_window_ms = float(config.get("image_window_ms", 450))
    blank_window_ms = float(config.get("blank_window_ms", 1200))

    for block, block_records in sorted(grouped.items()):
        wav_path = find_continuous_audio_file(session_dir, block, block_records)
        sample_rate, samples = read_wav_mono(wav_path)
        envelope_lf, _ = build_rms_envelope(samples, sample_rate)
        envelope_hf, _ = build_rms_envelope(samples, sample_rate, bandpass_hz=HF_BAND_HZ)
        wav_duration_ms = len(samples) / sample_rate * 1000.0
        correction = make_block_time_correction(block_records, wav_duration_ms)

        for index, record in enumerate(block_records):
            raw_onset_ms = record["audio_t_onset_in_segment"]
            raw_design_trial_end_ms = raw_onset_ms + image_window_ms + blank_window_ms
            raw_blank_offset_ms = None
            if record["t_blank_offset"] is not None and record["audio_segment_start_time"] is not None:
                raw_blank_offset_ms = record["t_blank_offset"] - record["audio_segment_start_time"]

            raw_next_trial_ms = None
            if index + 1 < len(block_records):
                raw_next_trial_ms = block_records[index + 1]["audio_t_onset_in_segment"] - SEARCH_PADDING_MS

            onset_ms = correction["correct"](raw_onset_ms)
            design_trial_end_ms = correction["correct"](raw_design_trial_end_ms)
            blank_offset_ms = correction["correct"](raw_blank_offset_ms)
            next_trial_ms = correction["correct"](raw_next_trial_ms)

            preferred_trial_end_ms = blank_offset_ms if blank_offset_ms is not None else design_trial_end_ms
            wav_end_cap_ms = wav_duration_ms - SEARCH_PADDING_MS
            effective_trial_end_ms = preferred_trial_end_ms
            if next_trial_ms is not None:
                effective_trial_end_ms = min(effective_trial_end_ms, next_trial_ms)
            effective_trial_end_ms = min(effective_trial_end_ms, wav_end_cap_ms)

            trial_window_flags = []
            if blank_offset_ms is None:
                trial_window_flags.append("logged_end_missing")
            if preferred_trial_end_ms > wav_end_cap_ms:
                trial_window_flags.append("truncated_by_wav_end")
            if next_trial_ms is not None and preferred_trial_end_ms > next_trial_ms:
                trial_window_flags.append("capped_by_next_trial")
            if effective_trial_end_ms <= onset_ms:
                trial_window_flags.append("non_positive_trial_window")

            detection = detect_onset_latency(envelope_lf, envelope_hf, onset_ms, effective_trial_end_ms)
            analysis_rows.append(
                {
                    "participant": record["participant"],
                    "block": block,
                    "trial": record["trial"],
                    "global_trial": record["global_trial"],
                    "stimulus_id": record["stimulus_id"],
                    "char": record["char"],
                    "pinyin": record["pinyin"],
                    "tone": record["tone"],
                    "initial_type": record["initial_type"],
                    "rhyme_type": record["rhyme_type"],
                    "trigger": record["trigger"],
                    "audio_file": wav_path.name,
                    "correction_mode": correction["mode"],
                    "correction_scale": round(correction["scale"], 6),
                    "estimated_linear_scale": None if correction["estimated_linear_scale"] is None else round(correction["estimated_linear_scale"], 6),
                    "raw_picture_onset_in_segment_ms": round(raw_onset_ms, 3),
                    "picture_onset_in_segment_ms": round(onset_ms, 3),
                    "raw_design_trial_end_in_segment_ms": round(raw_design_trial_end_ms, 3),
                    "design_trial_end_in_segment_ms": round(design_trial_end_ms, 3),
                    "raw_logged_trial_end_in_segment_ms": None if raw_blank_offset_ms is None else round(raw_blank_offset_ms, 3),
                    "logged_trial_end_in_segment_ms": None if blank_offset_ms is None else round(blank_offset_ms, 3),
                    "raw_next_trial_cap_in_segment_ms": None if raw_next_trial_ms is None else round(raw_next_trial_ms, 3),
                    "next_trial_cap_in_segment_ms": None if next_trial_ms is None else round(next_trial_ms, 3),
                    "wav_end_cap_in_segment_ms": round(wav_end_cap_ms, 3),
                    "baseline_start_in_segment_ms": round(max(0.0, onset_ms - BASELINE_PRE_MS), 3),
                    "baseline_end_in_segment_ms": round(max(0.0, onset_ms - BASELINE_POST_MS), 3),
                    "effective_trial_end_in_segment_ms": round(effective_trial_end_ms, 3),
                    "search_end_in_segment_ms": round(effective_trial_end_ms, 3),
                    "search_window_ms": round(effective_trial_end_ms - onset_ms, 3),
                    "trial_end_source": "logged" if blank_offset_ms is not None else "design_fallback",
                    "trial_window_flags": ";".join(trial_window_flags) if trial_window_flags else "clean",
                    "baseline_mean_rms": None if detection["baseline_mean"] is None else round(detection["baseline_mean"], 6),
                    "baseline_std_rms": None if detection["baseline_std"] is None else round(detection["baseline_std"], 6),
                    "threshold_rms": None if detection["threshold"] is None else round(detection["threshold"], 6),
                    "peak_rms_in_window": None if detection["peak_rms"] is None else round(detection["peak_rms"], 6),
                    "peak_latency_ms": None if detection["peak_latency_ms"] is None else round(detection["peak_latency_ms"], 3),
                    "detected_onset": int(bool(detection["detected"])),
                    "detected_latency_ms": None if detection["latency_ms"] is None else round(detection["latency_ms"], 3),
                    "onset_source_band": detection.get("source_band") or "",
                    "detected_latency_ms_lf": None if detection.get("latency_ms_lf") is None else round(detection["latency_ms_lf"], 3),
                    "detected_latency_ms_hf": None if detection.get("latency_ms_hf") is None else round(detection["latency_ms_hf"], 3),
                    "crossing_latency_ms_lf": None if detection.get("crossing_latency_ms_lf") is None else round(detection["crossing_latency_ms_lf"], 3),
                    "crossing_latency_ms_hf": None if detection.get("crossing_latency_ms_hf") is None else round(detection["crossing_latency_ms_hf"], 3),
                }
            )

    return analysis_rows


def trimmed_mean(values: list[float], proportion: float = TRIM_PROPORTION) -> float | None:
    """Symmetric-trimmed mean, discarding `proportion` tail on each side.

    Used for speech-onset aggregation where anticipatory (<150 ms) and drifting
    (>1500 ms) trials pull the mean off the central mass. Median-like but keeps
    one-SD precision when n is modest.
    """
    clean = sorted(v for v in values if v is not None and math.isfinite(v))
    if not clean:
        return None
    k = int(len(clean) * proportion)
    trimmed = clean[k: len(clean) - k] if k > 0 else clean
    return statistics.fmean(trimmed) if trimmed else None


def summarize_group(rows: list[dict[str, Any]], group_key: str, group_value: Any) -> dict[str, Any]:
    latencies = [row["detected_latency_ms"] for row in rows if row["detected_latency_ms"] is not None]
    detection_rate = sum(row["detected_onset"] for row in rows) / len(rows)
    mean_latency = trimmed_mean(latencies)
    median_latency = statistics.median(latencies) if latencies else None
    sd_latency = statistics.stdev(latencies) if len(latencies) >= 2 else None
    return {
        "group_key": group_key,
        "group_value": group_value,
        "n_trials": len(rows),
        "n_detected": len(latencies),
        "detection_rate": round(detection_rate, 4),
        "mean_latency_ms": None if mean_latency is None else round(mean_latency, 3),
        "median_latency_ms": None if median_latency is None else round(median_latency, 3),
        "sd_latency_ms": None if sd_latency is None else round(sd_latency, 3),
        "min_latency_ms": None if not latencies else round(min(latencies), 3),
        "max_latency_ms": None if not latencies else round(max(latencies), 3),
    }


def summarize_blocks(analysis_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in analysis_rows:
        by_block[row["block"]].append(row)
    return [summarize_group(rows, "block", block) for block, rows in sorted(by_block.items())]


def summarize_conditions(analysis_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for key in ("tone", "initial_type", "rhyme_type"):
        by_value: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in analysis_rows:
            by_value[row[key]].append(row)
        for value, rows in sorted(by_value.items(), key=lambda item: str(item[0])):
            output.append(summarize_group(rows, key, value))
    return output


def assign_qc_metrics(analysis_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qc_rows = []
    for row in analysis_rows:
        detected = bool(row["detected_onset"])
        search_window_ms = row["search_window_ms"]
        baseline_mean = row["baseline_mean_rms"]
        baseline_std = row["baseline_std_rms"]
        threshold = row["threshold_rms"]
        peak_rms = row["peak_rms_in_window"]
        latency = row["detected_latency_ms"]
        peak_latency = row["peak_latency_ms"]

        flags = []
        score = 1.0

        if not detected:
            flags.append("not_detected")
            score = 0.0
            peak_ratio = None
        else:
            if search_window_ms < 450:
                flags.append("narrow_search_window")
                score -= 0.15
            if baseline_mean is not None and baseline_mean >= 0.004:
                flags.append("high_baseline_noise")
                score -= 0.2
            if baseline_std is not None and baseline_std >= 0.0025:
                flags.append("unstable_baseline")
                score -= 0.15
            if threshold is not None and peak_rms is not None:
                peak_ratio = peak_rms / threshold if threshold else None
                if peak_ratio is not None and peak_ratio < 1.5:
                    flags.append("weak_peak_margin")
                    score -= 0.3
                elif peak_ratio is not None and peak_ratio < 2.0:
                    flags.append("moderate_peak_margin")
                    score -= 0.15
            else:
                peak_ratio = None
            if latency is not None and latency <= 180:
                flags.append("near_lower_latency_bound")
                score -= 0.15
            if latency is not None and latency >= 1300:
                flags.append("near_upper_latency_bound")
                score -= 0.2
            if latency is not None and peak_latency is not None and (peak_latency - latency) >= 500:
                flags.append("diffuse_rise_to_peak")
                score -= 0.1
            score = max(0.0, round(score, 3))

        if not detected:
            confidence_label = "miss"
        elif score >= 0.75:
            confidence_label = "high"
        elif score >= 0.45:
            confidence_label = "medium"
        else:
            confidence_label = "low"

        qc_row = dict(row)
        qc_row["peak_over_threshold_ratio"] = None if peak_ratio is None else round(peak_ratio, 3)
        qc_row["qc_score"] = score
        qc_row["qc_label"] = confidence_label
        qc_row["qc_flags"] = ";".join(flags) if flags else "clean"
        qc_rows.append(qc_row)

    return qc_rows


def summarize_qc(qc_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qc_rows:
        by_label[row["qc_label"]].append(row)

    summary_rows = []
    for label in ("high", "medium", "low", "miss"):
        rows = by_label.get(label, [])
        if not rows:
            continue
        detected_latencies = [row["detected_latency_ms"] for row in rows if row["detected_latency_ms"] is not None]
        summary_rows.append(
            {
                "qc_label": label,
                "n_trials": len(rows),
                "mean_qc_score": round(statistics.fmean(row["qc_score"] for row in rows), 3),
                "mean_latency_ms": round(trimmed_mean(detected_latencies), 3) if detected_latencies else None,
            }
        )

    flag_counts: dict[str, int] = defaultdict(int)
    for row in qc_rows:
        for flag in row["qc_flags"].split(";"):
            if flag and flag != "clean":
                flag_counts[flag] += 1

    flag_rows = [{"flag": flag, "n_trials": count} for flag, count in sorted(flag_counts.items(), key=lambda item: (-item[1], item[0]))]
    flagged_trials = [row for row in qc_rows if row["qc_label"] in {"low", "miss"}]
    flagged_trials.sort(key=lambda row: (row["qc_score"], row["block"], row["trial"]))
    return summary_rows, flag_rows, flagged_trials


def summarize_trial_windows(analysis_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flag_counts: dict[str, int] = defaultdict(int)
    for row in analysis_rows:
        for flag in str(row["trial_window_flags"]).split(";"):
            if flag and flag != "clean":
                flag_counts[flag] += 1
    return [{"trial_window_flag": flag, "n_trials": count} for flag, count in sorted(flag_counts.items(), key=lambda item: (-item[1], item[0]))]


def summarize_animacy_tone(analysis_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stimuli = load_stimuli_table()
    animacy_by_stimulus = {row["stimulus_id"]: ("animate" if row["meaning"] in ANIMATE_MEANINGS else "inanimate") for row in stimuli}

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in analysis_rows:
        animacy = animacy_by_stimulus.get(row["stimulus_id"], "unknown")
        grouped[(animacy, row["tone"])].append(row)

    summary_rows = []
    for animacy in ("animate", "inanimate"):
        for tone in (1, 2, 3, 4):
            rows = grouped.get((animacy, tone), [])
            if not rows:
                continue
            latencies = [row["detected_latency_ms"] for row in rows if row["detected_latency_ms"] is not None]
            detection = sum(row["detected_onset"] for row in rows) / len(rows)
            mean_latency = trimmed_mean(latencies)
            sd_latency = statistics.stdev(latencies) if len(latencies) >= 2 else None
            summary_rows.append(
                {
                    "animacy": animacy,
                    "tone": tone,
                    "n_trials": len(rows),
                    "detection_rate": detection,
                    "mean_latency_ms": None if mean_latency is None else round(mean_latency, 3),
                    "sd_latency_ms": None if sd_latency is None else round(sd_latency, 3),
                }
            )
    return summary_rows


def generate_plots(output_dir: Path, qc_rows: list[dict[str, Any]], block_summary: list[dict[str, Any]], condition_summary: list[dict[str, Any]]) -> list[Path]:
    if plt is None:
        return []
    generated: list[Path] = []
    detected_rows = [row for row in qc_rows if row["detected_latency_ms"] is not None]
    if not detected_rows:
        return generated

    with plt.rc_context({"font.family": "serif"}):
        fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.9))
        fig.suptitle("Speech Onset Summary", fontsize=17, y=0.992)
        blocks = [row["group_value"] for row in block_summary]
        block_means = [row["mean_latency_ms"] for row in block_summary]
        detection_rates = [row["detection_rate"] * 100 for row in block_summary]

        ax = axes[0]
        ax.bar(blocks, block_means, color="#5B7FA3", alpha=0.92, width=0.68)
        ax.set_title("Block Mean Onset and Detection")
        ax.set_xlabel("Block")
        ax.set_ylabel("Latency (ms)")
        ax.grid(alpha=0.14, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax_rate = ax.twinx()
        ax_rate.plot(blocks, detection_rates, color="#C46A5A", marker="o", linewidth=1.8, markersize=4.5)
        ax_rate.set_ylabel("Detection rate (%)")
        ax_rate.set_ylim(0, 105)
        ax_rate.spines["top"].set_visible(False)

        tone_rows = [row for row in condition_summary if row["group_key"] == "tone"]
        tones = [str(row["group_value"]) for row in tone_rows]
        tone_means = [row["mean_latency_ms"] for row in tone_rows]
        ax = axes[1]
        ax.bar(tones, tone_means, color="#84A98C", alpha=0.92, width=0.68)
        ax.set_title("Mean Onset by Tone")
        ax.set_xlabel("Tone")
        ax.set_ylabel("Latency (ms)")
        ax.grid(alpha=0.14, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        block_latency_data = []
        block_labels = []
        for block in sorted({row["block"] for row in detected_rows}):
            block_labels.append(str(block))
            block_latency_data.append([row["detected_latency_ms"] for row in detected_rows if row["block"] == block])
        ax = axes[2]
        ax.boxplot(
            block_latency_data,
            tick_labels=block_labels,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#E8EEF5", "edgecolor": "#6C7A89", "linewidth": 1.0},
            medianprops={"color": "#B85C38", "linewidth": 1.4},
            whiskerprops={"color": "#6C7A89", "linewidth": 1.0},
            capprops={"color": "#6C7A89", "linewidth": 1.0},
        )
        ax.set_title("Latency Distribution by Block")
        ax.set_xlabel("Block")
        ax.set_ylabel("Latency (ms)")
        ax.grid(alpha=0.14, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(rect=[0, 0, 1, 0.975])
        overview_path = output_dir / "onset_overview.png"
        fig.savefig(overview_path, dpi=240, bbox_inches="tight")
        plt.close(fig)
    generated.append(overview_path)

    summary_rows = []
    for row in block_summary:
        mean_text = "NA" if row["mean_latency_ms"] is None or row["sd_latency_ms"] is None else f"{row['mean_latency_ms']:.0f} ms ({row['sd_latency_ms']:.0f})"
        summary_rows.append(["block", str(row["group_value"]), f"{row['detection_rate'] * 100:.1f}", mean_text])
    for key in ("tone", "initial_type", "rhyme_type"):
        for row in condition_summary:
            if row["group_key"] != key:
                continue
            mean_text = "NA" if row["mean_latency_ms"] is None or row["sd_latency_ms"] is None else f"{row['mean_latency_ms']:.0f} ms ({row['sd_latency_ms']:.0f})"
            summary_rows.append([key, str(row["group_value"]), f"{row['detection_rate'] * 100:.1f}", mean_text])

    fig_table, ax_table = plt.subplots(figsize=(13, max(6, 0.42 * len(summary_rows) + 1.8)))
    ax_table.axis("off")
    ax_table.set_title("Onset Summary by Block and Condition", fontsize=16, loc="left", pad=18)
    table = ax_table.table(cellText=summary_rows, colLabels=["Grouping", "Level", "Detection (%)", "Speech onset (SD)"], loc="center", cellLoc="left", colLoc="left", bbox=[0.02, 0.0, 0.96, 0.93])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("#888888")
        if row_idx == 0:
            cell.set_facecolor("#E9EEF3")
            cell.set_text_props(weight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#F8FAFC")
        else:
            cell.set_facecolor("#FFFFFF")
    table_path = output_dir / "onset_summary_table.png"
    fig_table.savefig(table_path, dpi=240, bbox_inches="tight")
    plt.close(fig_table)
    generated.append(table_path)

    fig_bar, axes_bar = plt.subplots(1, 2, figsize=(13, 5.5))
    tone_rows = [row for row in condition_summary if row["group_key"] == "tone"]
    tones = [str(row["group_value"]) for row in tone_rows]
    tone_detect = [row["detection_rate"] * 100 for row in tone_rows]
    tone_latency = [row["mean_latency_ms"] for row in tone_rows]
    axes_bar[0].bar(tones, tone_detect, color="#4E79A7", alpha=0.9)
    axes_bar[0].set_title("Detection by Tone")
    axes_bar[0].set_xlabel("Tone")
    axes_bar[0].set_ylabel("Detection (%)")
    axes_bar[0].set_ylim(0, 105)
    axes_bar[0].grid(alpha=0.2, axis="y")
    axes_bar[1].bar(tones, tone_latency, color="#E07A5F", alpha=0.9)
    axes_bar[1].set_title("Mean Onset by Tone")
    axes_bar[1].set_xlabel("Tone")
    axes_bar[1].set_ylabel("Latency (ms)")
    axes_bar[1].grid(alpha=0.2, axis="y")
    fig_bar.tight_layout()
    bar_path = output_dir / "onset_tone_bars.png"
    fig_bar.savefig(bar_path, dpi=240, bbox_inches="tight")
    plt.close(fig_bar)
    generated.append(bar_path)
    return generated


def generate_animacy_tone_plot(output_dir: Path, animacy_tone_summary: list[dict[str, Any]]) -> list[Path]:
    if plt is None or not animacy_tone_summary:
        return []
    generated = []
    tone_order = [1, 3, 2, 4]
    rows = []
    for animacy in ("animate", "inanimate"):
        for tone in tone_order:
            row = next((item for item in animacy_tone_summary if item["animacy"] == animacy and item["tone"] == tone), None)
            if row is None:
                continue
            mean_text = "NA" if row["mean_latency_ms"] is None or row["sd_latency_ms"] is None else f"{row['mean_latency_ms']:.0f} ms ({row['sd_latency_ms']:.0f} ms)"
            rows.append([animacy if tone == 1 else "", f"tone {tone}", f"{row['detection_rate'] * 100:.1f}", mean_text])

    with plt.rc_context({"font.family": "serif"}):
        fig, ax = plt.subplots(figsize=(13.5, 6.4))
        ax.axis("off")
        ax.text(0.01, 0.97, "Table. Accuracies and onset latencies by animacy and tone.", fontsize=17, transform=ax.transAxes, va="top")
        table = ax.table(cellText=rows, colLabels=["Animacy", "Tone", "Accuracy (%)", "Speech onset (SD)"], loc="center", cellLoc="left", colLoc="left", bbox=[0.03, 0.02, 0.94, 0.84])
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        for (r, c), cell in table.get_celld().items():
            cell.set_facecolor("#FFFFFF")
            cell.set_edgecolor("#888888")
            if r == 0:
                cell.visible_edges = "BT"
                cell.set_linewidth(1.1)
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#EEF3F8")
            else:
                cell.visible_edges = "B"
                cell.set_linewidth(0.6)
                if r == 5:
                    cell.set_linewidth(1.0)
                elif 1 <= r <= 4:
                    cell.set_facecolor("#F8FBFD")
                elif 5 <= r <= 8:
                    cell.set_facecolor("#FCFAF7")
            if c in (2, 3):
                cell._loc = "center"
        path = output_dir / "onset_animacy_tone_table.png"
        fig.savefig(path, dpi=280, bbox_inches="tight")
        plt.close(fig)
    generated.append(path)
    return generated


def generate_block_diagnostic_plots(session_dir: Path, output_dir: Path, qc_rows: list[dict[str, Any]]) -> list[Path]:
    if plt is None:
        return []
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in qc_rows:
        grouped[row["block"]].append(row)
    generated = []
    for block, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["trial"])
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.0, 1.0]})
        fig.suptitle(f"Block {block} Detailed Review", fontsize=15, y=0.995)
        wav_end_s = float(rows[0]["wav_end_cap_in_segment_ms"]) / 1000.0
        xs = [float(row["trial"]) for row in rows]
        raw_onsets = [float(row["raw_picture_onset_in_segment_ms"]) / 1000.0 for row in rows]
        corrected_onsets = [float(row["picture_onset_in_segment_ms"]) / 1000.0 for row in rows]
        logged_ends = [None if row["raw_logged_trial_end_in_segment_ms"] in ("", None) else float(row["raw_logged_trial_end_in_segment_ms"]) / 1000.0 for row in rows]
        latencies = [None if row["detected_latency_ms"] in ("", None) else float(row["detected_latency_ms"]) for row in rows]
        axes[0].plot(xs, raw_onsets, color="#C44E52", marker="o", linewidth=1.2, label="raw picture onset")
        axes[0].plot(xs, corrected_onsets, color="#2F6C8F", marker="o", linewidth=1.4, label="corrected picture onset")
        logged_x = [x for x, y in zip(xs, logged_ends) if y is not None]
        logged_y = [y for y in logged_ends if y is not None]
        axes[0].plot(logged_x, logged_y, color="#5C8E5F", linewidth=1.0, alpha=0.85, label="logged trial end")
        axes[0].axhline(wav_end_s, color="#111111", linestyle="--", linewidth=1.2, label="wav end cap")
        axes[0].legend(frameon=False, ncol=4, fontsize=8)
        axes[0].grid(alpha=0.2)
        det_x = [x for x, y in zip(xs, latencies) if y is not None]
        det_y = [y for y in latencies if y is not None]
        miss_x = [x for x, y in zip(xs, latencies) if y is None]
        axes[1].scatter(det_x, det_y, color="#2F6C8F", s=34, label="detected")
        if miss_x:
            axes[1].scatter(miss_x, [0] * len(miss_x), color="#C44E52", s=34, label="miss")
        axes[1].legend(frameon=False, fontsize=8)
        axes[1].grid(alpha=0.2)
        qc_scores = [float(row["qc_score"]) for row in rows]
        axes[2].bar(xs, qc_scores, color="#8A7DB8", alpha=0.9, width=0.8)
        axes[2].set_ylim(0, 1.05)
        axes[2].grid(alpha=0.2, axis="y")
        out_path = output_dir / f"block_{block:02d}_detail.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        generated.append(out_path)
    return generated


def summarize_segmentation(session_dir: Path, session_data: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = group_by_block(records)
    summary_rows = []
    config = session_data.get("config", {})
    image_window_ms = float(config.get("image_window_ms", 450))
    blank_window_ms = float(config.get("blank_window_ms", 1200))
    for block, block_records in sorted(grouped.items()):
        wav_path = find_continuous_audio_file(session_dir, block, block_records)
        sample_rate, samples = read_wav_mono(wav_path)
        wav_duration_ms = len(samples) / sample_rate * 1000.0
        correction = make_block_time_correction(block_records, wav_duration_ms)
        segment = session_data["audio_segments"][block - 1] if block - 1 < len(session_data.get("audio_segments", [])) else {}
        first_trial = block_records[0]
        last_trial = block_records[-1]
        csv_segment_start_ms = first_trial["audio_segment_start_time"]
        json_segment_start_ms = segment.get("start_time")
        last_trial_blank_offset_ms = None
        if last_trial["t_blank_offset"] is not None and csv_segment_start_ms is not None:
            last_trial_blank_offset_ms = last_trial["t_blank_offset"] - csv_segment_start_ms
        summary_rows.append(
            {
                "block": block,
                "audio_file": wav_path.name,
                "json_segment_start_ms": None if json_segment_start_ms is None else round(json_segment_start_ms, 3),
                "csv_segment_start_ms": None if csv_segment_start_ms is None else round(csv_segment_start_ms, 3),
                "start_offset_csv_minus_json_ms": None if json_segment_start_ms is None or csv_segment_start_ms is None else round(csv_segment_start_ms - json_segment_start_ms, 3),
                "json_duration_ms": None if segment.get("duration_ms") is None else round(segment["duration_ms"], 3),
                "wav_duration_ms": round(wav_duration_ms, 3),
                "first_picture_onset_ms": round(first_trial["audio_t_onset_in_segment"], 3),
                "correction_scale": round(correction["scale"], 6),
                "estimated_linear_scale": None if correction["estimated_linear_scale"] is None else round(correction["estimated_linear_scale"], 6),
                "design_last_trial_end_ms": round(last_trial["audio_t_onset_in_segment"] + image_window_ms + blank_window_ms, 3),
                "last_picture_onset_ms": round(last_trial["audio_t_onset_in_segment"], 3),
                "last_blank_offset_ms": None if last_trial_blank_offset_ms is None else round(last_trial_blank_offset_ms, 3),
                "last_trial_truncated_by_wav": int(last_trial_blank_offset_ms is not None and last_trial_blank_offset_ms > (wav_duration_ms - SEARCH_PADDING_MS)),
            }
        )
    return summary_rows


def summarize_block_details(analysis_rows: list[dict[str, Any]], qc_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    analysis_by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    qc_by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in analysis_rows:
        analysis_by_block[row["block"]].append(row)
    for row in qc_rows:
        qc_by_block[int(row["block"])].append(row)
    details = []
    for block in sorted(analysis_by_block):
        rows = sorted(analysis_by_block[block], key=lambda row: row["trial"])
        qc_rows_block = sorted(qc_by_block[block], key=lambda row: int(row["trial"]))
        detected = [row for row in rows if row["detected_latency_ms"] is not None]
        latencies = [row["detected_latency_ms"] for row in detected]
        misses = [row for row in qc_rows_block if row["qc_label"] == "miss"]
        low_conf = [row for row in qc_rows_block if row["qc_label"] in {"low", "medium"}]
        details.append({"block": block, "n_trials": len(rows), "n_detected": len(detected), "detection_rate": len(detected) / len(rows), "mean_latency_ms": round(trimmed_mean(latencies), 3) if latencies else None, "median_latency_ms": round(statistics.median(latencies), 3) if latencies else None, "min_latency_ms": round(min(latencies), 3) if latencies else None, "max_latency_ms": round(max(latencies), 3) if latencies else None, "correction_scale": rows[0]["correction_scale"], "estimated_linear_scale": rows[0].get("estimated_linear_scale"), "raw_first_onset_ms": rows[0]["raw_picture_onset_in_segment_ms"], "raw_last_onset_ms": rows[-1]["raw_picture_onset_in_segment_ms"], "corrected_last_onset_ms": rows[-1]["picture_onset_in_segment_ms"], "wav_end_cap_ms": rows[0]["wav_end_cap_in_segment_ms"], "miss_trials": [(row["trial"], row["char"]) for row in misses], "low_conf_trials": [(row["trial"], row["char"], row["qc_label"]) for row in low_conf[:8]], "late_trials": [(row["trial"], row["char"], row["detected_latency_ms"]) for row in sorted(detected, key=lambda item: item["detected_latency_ms"], reverse=True)[:5]], "clean_trials": sum(1 for row in qc_rows_block if row["qc_label"] == "high")})
    return details


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def linear_slope(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return None
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return numerator / denominator


def markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(str(row.get(header, "")) for header in headers) + " |" for row in rows]
    return "\n".join([header_line, separator_line] + body_lines)


def render_report(session_dir: Path, session_json: Path, analysis_rows: list[dict[str, Any]], segmentation_summary: list[dict[str, Any]], block_summary: list[dict[str, Any]], condition_summary: list[dict[str, Any]], qc_summary: list[dict[str, Any]], qc_flag_rows: list[dict[str, Any]], flagged_trials: list[dict[str, Any]], trial_window_summary: list[dict[str, Any]], plot_paths: list[Path], block_details: list[dict[str, Any]], block_plot_paths: list[Path]) -> str:
    detected_latencies = [row["detected_latency_ms"] for row in analysis_rows if row["detected_latency_ms"] is not None]
    overall_detection_rate = sum(row["detected_onset"] for row in analysis_rows) / len(analysis_rows)
    overall_mean = trimmed_mean(detected_latencies)
    overall_median = statistics.median(detected_latencies) if detected_latencies else None
    block_points = [(row["group_value"], row["mean_latency_ms"]) for row in block_summary if row["mean_latency_ms"] is not None]
    block_slope = linear_slope(block_points)
    qc_table = [{"qc_label": row["qc_label"], "n_trials": row["n_trials"], "mean_qc_score": row["mean_qc_score"], "mean_latency_ms": row["mean_latency_ms"]} for row in qc_summary]
    top_flag_rows = [{"block": row["block"], "trial": row["trial"], "char": row["char"], "tone": row["tone"], "latency_ms": row["detected_latency_ms"], "qc_label": row["qc_label"], "qc_flags": row["qc_flags"]} for row in flagged_trials[:12]]
    lines = [
        "# Onset Analysis Report",
        "",
        f"Session directory: `{session_dir}`",
        f"Session JSON: `{session_json.name}`",
        "",
        f"- Total trials: {len(analysis_rows)}",
        f"- Detected onsets: {sum(row['detected_onset'] for row in analysis_rows)} ({overall_detection_rate:.2%})",
        f"- Overall mean latency: {None if overall_mean is None else round(overall_mean, 3)} ms",
        f"- Overall median latency: {None if overall_median is None else round(overall_median, 3)} ms",
        f"- Block-level linear slope: {None if block_slope is None else round(block_slope, 3)} ms/block",
        "",
        "## Block summary",
        "",
        markdown_table([{"block": row["group_value"], "n_trials": row["n_trials"], "n_detected": row["n_detected"], "detection_rate": f"{row['detection_rate']:.2%}", "mean_latency_ms": row["mean_latency_ms"], "median_latency_ms": row["median_latency_ms"], "sd_latency_ms": row["sd_latency_ms"]} for row in block_summary], ["block", "n_trials", "n_detected", "detection_rate", "mean_latency_ms", "median_latency_ms", "sd_latency_ms"]),
        "",
        "## QC summary",
        "",
        markdown_table(qc_table, ["qc_label", "n_trials", "mean_qc_score", "mean_latency_ms"]),
        "",
        markdown_table(qc_flag_rows[:8], ["flag", "n_trials"]) if qc_flag_rows else "No QC flags.",
        "",
        markdown_table(top_flag_rows, ["block", "trial", "char", "tone", "latency_ms", "qc_label", "qc_flags"]) if top_flag_rows else "No low-confidence or missed trials.",
        "",
        "## Output files",
        "",
        "- onset_trial_level.csv",
        "- onset_trial_qc.csv",
        "- onset_qc_summary.csv",
        "- onset_segmentation_summary.csv",
        "- onset_block_summary.csv",
        "- onset_condition_summary.csv",
    ]
    return "\n".join(lines)


def run_onset_analysis(session_dir: Path | str, *, output_dir: Path | str | None = None, clear_output: bool = True) -> dict[str, Any]:
    session_dir = Path(session_dir).resolve()
    session_csvs = sorted(session_dir.glob("session_*.csv"))
    session_jsons = [path for path in sorted(session_dir.glob("session_*.json")) if "startup_report" not in path.name]
    if not session_csvs or not session_jsons:
        raise SystemExit(f"Could not find session CSV/JSON in {session_dir}")
    session_csv = next(path for path in session_csvs if not path.name.endswith("_practice.csv"))
    session_json = session_jsons[0]
    records = build_trial_records(load_csv_rows(session_csv))
    session_data = load_json(session_json)
    analysis_rows = build_trial_analysis(session_dir, records, session_data)
    segmentation_summary = summarize_segmentation(session_dir, session_data, records)
    block_summary = summarize_blocks(analysis_rows)
    condition_summary = summarize_conditions(analysis_rows)
    animacy_tone_summary = summarize_animacy_tone(analysis_rows)
    qc_rows = assign_qc_metrics(analysis_rows)
    qc_summary, qc_flag_rows, flagged_trials = summarize_qc(qc_rows)
    trial_window_summary = summarize_trial_windows(analysis_rows)
    block_details = summarize_block_details(analysis_rows, qc_rows)
    output_path = Path(output_dir).resolve() if output_dir else (session_dir / DEFAULT_OUTPUT_DIRNAME)
    if clear_output and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    write_csv(output_path / "onset_trial_level.csv", analysis_rows)
    write_csv(output_path / "onset_trial_qc.csv", qc_rows)
    write_csv(output_path / "onset_qc_summary.csv", qc_summary)
    if qc_flag_rows:
        write_csv(output_path / "onset_qc_flag_counts.csv", qc_flag_rows)
    if flagged_trials:
        write_csv(output_path / "onset_flagged_trials.csv", flagged_trials)
    write_csv(output_path / "onset_segmentation_summary.csv", segmentation_summary)
    write_csv(output_path / "onset_block_summary.csv", block_summary)
    write_csv(output_path / "onset_condition_summary.csv", condition_summary)
    if trial_window_summary:
        write_csv(output_path / "onset_trial_window_summary.csv", trial_window_summary)
    plot_paths = generate_plots(output_path, qc_rows, block_summary, condition_summary)
    plot_paths.extend(generate_animacy_tone_plot(output_path, animacy_tone_summary))
    block_plot_paths = generate_block_diagnostic_plots(session_dir, output_path, qc_rows)
    report = render_report(session_dir, session_json, analysis_rows, segmentation_summary, block_summary, condition_summary, qc_summary, qc_flag_rows, flagged_trials, trial_window_summary, plot_paths, block_details, block_plot_paths)
    report_path = output_path / "onset_report.md"
    report_path.write_text(report, encoding="utf-8")
    return {"session_dir": session_dir, "output_dir": output_path, "analysis_rows": analysis_rows, "report": report, "report_path": report_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze onset timing from session logs and continuous audio.")
    parser.add_argument("session_dir", nargs="?", default="data_ssh", help="Session directory or parent data directory. Default: data_ssh")
    parser.add_argument("--output-dir", default=None, help="Optional output directory")
    parser.add_argument("--keep-output", action="store_true", help="Keep existing output directory contents")
    args = parser.parse_args(argv)
    session_dir = resolve_session_dir(args.session_dir)
    result = run_onset_analysis(session_dir, output_dir=args.output_dir, clear_output=not args.keep_output)
    print(result["report"])
    print("")
    print(f"Wrote analysis files to: {result['output_dir']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
