"""Trial-level onset/ASR selection manifest for production EEG sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .asr import run_asr_analysis, tone3
from .onset import run_onset_analysis

logger = logging.getLogger(__name__)

DEFAULT_SELECTION_DIRNAME = "analysis_selection"
DEFAULT_MANIFEST_NAME = "trial_manifest.csv"
DEFAULT_SUMMARY_NAME = "selection_summary.json"
SELECTION_VERSION = "selection_v1_pinyin_fuzzy_0.75"
FUZZY_SIMILARITY_THRESHOLD = 0.75

FORMAL_COLUMNS = [
    "participant",
    "block",
    "trial",
    "global_trial",
    "stimulus_id",
    "trigger",
    "char",
    "pinyin",
    "tone",
    "initial_type",
    "rhyme_type",
]

ONSET_COLUMNS = [
    "detected_onset",
    "detected_latency_ms",
    "qc_label",
    "qc_flags",
]

ASR_COLUMNS = [
    "pred_char",
    "pred_pinyin_tone3",
    "alignment_status",
    "word_match",
    "pinyin_match",
    "tone_match",
]

MANIFEST_COLUMNS = [
    *FORMAL_COLUMNS,
    *ONSET_COLUMNS,
    *ASR_COLUMNS,
    "target_pinyin_tone3",
    "asr_match_mode",
    "keep_trial",
    "drop_reason",
    "strict_keep",
    "fuzzy_keep",
    "pinyin_similarity",
    "selection_version",
]


@dataclass(frozen=True)
class TrialManifestResult:
    """Paths and compact summary for a generated trial manifest."""

    session_dir: Path
    manifest_path: Path
    summary_path: Path
    summary: dict[str, Any]


def build_trial_manifest(
    session_dir: str | Path,
    *,
    asr_policy: str = "pinyin_fuzzy",
    output_dir: str | Path | None = None,
    run_missing: bool = True,
    force: bool = False,
) -> TrialManifestResult:
    """Build a trial manifest by merging formal trials, onset QC, and ASR QC."""

    if asr_policy != "pinyin_fuzzy":
        raise ValueError(f"Unsupported ASR policy: {asr_policy!r}")

    session_dir = Path(session_dir).resolve()
    if not is_modern_session_dir(session_dir):
        raise ValueError(f"Not a modern production session directory: {session_dir}")

    selection_dir = Path(output_dir).resolve() if output_dir else session_dir / DEFAULT_SELECTION_DIRNAME
    selection_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = selection_dir / DEFAULT_MANIFEST_NAME
    summary_path = selection_dir / DEFAULT_SUMMARY_NAME

    if manifest_path.exists() and summary_path.exists() and not force:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        return TrialManifestResult(session_dir, manifest_path, summary_path, summary)

    formal_df = load_formal_trials(session_dir)
    onset_df = _ensure_onset(session_dir, run_missing=run_missing)
    asr_df = _ensure_asr(session_dir, run_missing=run_missing)

    manifest = merge_trial_sources(formal_df, onset_df, asr_df)
    manifest = apply_asr_selection(manifest)
    manifest = order_manifest_columns(manifest)

    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    summary = summarize_manifest(manifest, session_dir=session_dir, asr_policy=asr_policy)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    logger.info(
        "selection manifest written: %s (keep=%s drop=%s)",
        manifest_path,
        summary["n_keep"],
        summary["n_drop"],
    )
    return TrialManifestResult(session_dir, manifest_path, summary_path, summary)


def load_formal_trials(session_dir: str | Path) -> pd.DataFrame:
    """Load formal trial order from the session CSV."""

    session_dir = Path(session_dir)
    csv_path = find_session_csv(session_dir)
    df = pd.read_csv(csv_path)
    if "phase" in df.columns:
        df = df[df["phase"].astype(str).str.lower() == "formal"].copy()
    if df.empty:
        raise ValueError(f"No formal trials found in {csv_path}")

    for col in FORMAL_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col not in {"block", "trial", "global_trial", "trigger", "tone"} else pd.NA

    df = df[FORMAL_COLUMNS].copy()
    for col in ["block", "trial", "global_trial", "trigger", "tone"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    missing = [col for col in ["block", "trial", "global_trial", "trigger"] if df[col].isna().any()]
    if missing:
        raise ValueError(f"Formal CSV is missing required numeric values: {missing}")

    df = df.sort_values(["global_trial", "block", "trial"]).reset_index(drop=True)
    return df


def merge_trial_sources(
    formal_df: pd.DataFrame,
    onset_df: pd.DataFrame | None,
    asr_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge source tables by block/trial while preserving formal trial order."""

    manifest = formal_df.copy()
    key_cols = ["block", "trial"]

    if onset_df is not None and not onset_df.empty:
        onset = _normalize_key_columns(onset_df)
        onset = _rename_first_existing(
            onset,
            {
                "detected_onset": ["detected_onset", "detected_onset_ms", "speech_onset_ms", "onset_ms"],
                "detected_latency_ms": ["detected_latency_ms", "latency_ms", "onset_latency_ms"],
                "qc_label": ["qc_label", "onset_qc_label"],
                "qc_flags": ["qc_flags", "onset_qc_flags"],
            },
        )
        manifest = manifest.merge(
            onset[[*key_cols, *[col for col in ONSET_COLUMNS if col in onset.columns]]],
            on=key_cols,
            how="left",
        )

    if asr_df is not None and not asr_df.empty:
        asr = _normalize_key_columns(asr_df)
        asr = _rename_first_existing(
            asr,
            {
                "pred_char": ["pred_char", "pred_text_window", "prediction"],
                "pred_pinyin_tone3": ["pred_pinyin_tone3", "pred_pinyin"],
                "alignment_status": ["alignment_status"],
                "word_match": ["word_match"],
                "pinyin_match": ["pinyin_match"],
                "tone_match": ["tone_match"],
                "target_pinyin_tone3": ["target_pinyin_tone3"],
            },
        )
        keep_cols = [col for col in [*ASR_COLUMNS, "target_pinyin_tone3"] if col in asr.columns]
        manifest = manifest.merge(asr[[*key_cols, *keep_cols]], on=key_cols, how="left")

    for col in ONSET_COLUMNS:
        if col not in manifest.columns:
            manifest[col] = ""
    for col in ASR_COLUMNS:
        if col not in manifest.columns:
            manifest[col] = ""
    if "target_pinyin_tone3" not in manifest.columns:
        manifest["target_pinyin_tone3"] = ""

    return manifest


def apply_asr_selection(df: pd.DataFrame, *, threshold: float = FUZZY_SIMILARITY_THRESHOLD) -> pd.DataFrame:
    """Apply the default relaxed pinyin fuzzy ASR policy."""

    rows = []
    for _, row in df.iterrows():
        target = _target_pinyin(row)
        predicted = row.get("pred_pinyin_tone3", "")
        status = str(row.get("alignment_status", "") or "").strip().lower()
        match = evaluate_asr_match(target, predicted, status, threshold=threshold)
        rows.append(match)

    result = df.copy()
    match_df = pd.DataFrame(rows)
    for col in match_df.columns:
        result[col] = match_df[col].to_numpy()
    result["selection_version"] = SELECTION_VERSION
    return result


def evaluate_asr_match(
    target_pinyin: Any,
    pred_pinyin: Any,
    alignment_status: str,
    *,
    threshold: float = FUZZY_SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate one aligned trial under the relaxed pinyin fuzzy policy."""

    target_norm = strip_pinyin_tone(target_pinyin)
    pred_norm = strip_pinyin_tone(pred_pinyin)
    status = (alignment_status or "").strip().lower()

    if status == "omission" or not pred_norm:
        return {
            "asr_match_mode": "omission",
            "keep_trial": 0,
            "drop_reason": "asr_omission",
            "strict_keep": 0,
            "fuzzy_keep": 0,
            "pinyin_similarity": 0.0,
        }

    similarity = normalized_levenshtein_similarity(target_norm, pred_norm)
    exact = bool(target_norm) and target_norm == pred_norm
    fuzzy = bool(target_norm) and similarity >= threshold

    if exact:
        mode = "exact"
        keep = 1
        reason = ""
    elif fuzzy:
        mode = "fuzzy"
        keep = 1
        reason = ""
    else:
        mode = "fail"
        keep = 0
        reason = "asr_pinyin_fail"

    return {
        "asr_match_mode": mode,
        "keep_trial": keep,
        "drop_reason": reason,
        "strict_keep": int(exact),
        "fuzzy_keep": int(keep),
        "pinyin_similarity": round(float(similarity), 6),
    }


def strip_pinyin_tone(value: Any) -> str:
    """Normalize pinyin to lowercase, tone-stripped, space-free syllable text."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    if not text or text == "nan":
        return ""
    text = text.replace("u:", "v").replace("ü", "v")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[1-5]", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^a-zv]", "", text)
    return text


def normalized_levenshtein_similarity(left: str, right: str) -> float:
    """Return 1 - normalized edit distance for two strings."""

    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for i, lch in enumerate(left, start=1):
        current = [i]
        for j, rch in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if lch == rch else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    distance = previous[-1]
    return max(0.0, 1.0 - distance / max(len(left), len(right)))


def summarize_manifest(
    manifest: pd.DataFrame,
    *,
    session_dir: Path,
    asr_policy: str,
) -> dict[str, Any]:
    """Build a compact JSON summary for QC and reproducibility."""

    keep = pd.to_numeric(manifest["keep_trial"], errors="coerce").fillna(0).astype(int)
    strict_keep = pd.to_numeric(manifest["strict_keep"], errors="coerce").fillna(0).astype(int)
    fuzzy_keep = pd.to_numeric(manifest["fuzzy_keep"], errors="coerce").fillna(0).astype(int)
    drop_reasons = (
        manifest.loc[keep == 0, "drop_reason"].fillna("").replace("", "unspecified").value_counts().to_dict()
    )
    match_modes = manifest["asr_match_mode"].fillna("").replace("", "missing").value_counts().to_dict()
    return {
        "selection_version": SELECTION_VERSION,
        "asr_policy": asr_policy,
        "session_dir": str(session_dir),
        "n_trials": int(len(manifest)),
        "n_keep": int(keep.sum()),
        "n_drop": int((keep == 0).sum()),
        "strict_keep": int(strict_keep.sum()),
        "fuzzy_keep": int(fuzzy_keep.sum()),
        "drop_reason_counts": {str(k): int(v) for k, v in drop_reasons.items()},
        "asr_match_mode_counts": {str(k): int(v) for k, v in match_modes.items()},
    }


def order_manifest_columns(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return manifest with stable public columns first."""

    result = manifest.copy()
    for col in MANIFEST_COLUMNS:
        if col not in result.columns:
            result[col] = ""
    extra_cols = [col for col in result.columns if col not in MANIFEST_COLUMNS]
    return result[[*MANIFEST_COLUMNS, *extra_cols]]


def try_resolve_session_dir_from_bdf(
    bdf_path: str | Path,
    *,
    search_root: str | Path | None = None,
) -> Path | None:
    """Return a modern session directory for a BDF path, otherwise None.

    If the supplied BDF is a legacy copy, ``search_root`` can be used to find
    a modern session containing a byte-identical BDF under ``ses-*/eeg_data``.
    """

    path = Path(bdf_path).resolve()
    candidates: list[Path] = []
    if path.is_file():
        candidates.extend([path.parent, path.parent.parent])
    candidates.extend(path.parents)
    for candidate in candidates:
        if candidate.name.startswith("ses-") and is_modern_session_dir(candidate):
            return candidate
    if search_root is not None and path.exists():
        return find_matching_modern_session(path, search_root)
    return None


def find_matching_modern_session(bdf_path: str | Path, search_root: str | Path) -> Path | None:
    """Find a modern session whose BDF is byte-identical to a legacy BDF copy."""

    bdf_path = Path(bdf_path).resolve()
    search_root = Path(search_root).resolve()
    if not bdf_path.exists() or not search_root.exists():
        return None

    target_size = bdf_path.stat().st_size
    target_hash: str | None = None
    for candidate_bdf in sorted(search_root.glob("sub-*/ses-*/eeg_data/*.bdf")):
        candidate_bdf = candidate_bdf.resolve()
        if candidate_bdf == bdf_path:
            continue
        if candidate_bdf.stat().st_size != target_size:
            continue
        session_dir = candidate_bdf.parent.parent
        if not is_modern_session_dir(session_dir):
            continue
        if target_hash is None:
            target_hash = _file_sha256(bdf_path)
        if _file_sha256(candidate_bdf) == target_hash:
            return session_dir
    return None


def is_modern_session_dir(session_dir: str | Path) -> bool:
    """Detect sessions that have formal metadata and continuous audio."""

    session_dir = Path(session_dir)
    if not session_dir.exists() or not session_dir.is_dir():
        return False
    has_csv = find_session_csv(session_dir, required=False) is not None
    has_json = find_session_json(session_dir, required=False) is not None
    has_audio = (session_dir / "continuous_audio").exists()
    return bool(has_csv and has_json and has_audio)


def find_session_csv(session_dir: str | Path, *, required: bool = True) -> Path | None:
    session_dir = Path(session_dir)
    candidates = [
        path
        for path in sorted(session_dir.glob("session_*.csv"))
        if "practice" not in path.name.lower()
        and "passive" not in path.name.lower()
        and "startup_report" not in path.name.lower()
    ]
    if candidates:
        return candidates[0]
    if required:
        raise FileNotFoundError(f"No formal session CSV found under {session_dir}")
    return None


def find_session_json(session_dir: str | Path, *, required: bool = True) -> Path | None:
    session_dir = Path(session_dir)
    candidates = [
        path
        for path in sorted(session_dir.glob("session_*.json"))
        if "practice" not in path.name.lower()
        and "passive" not in path.name.lower()
        and "startup_report" not in path.name.lower()
    ]
    if candidates:
        return candidates[0]
    if required:
        raise FileNotFoundError(f"No formal session JSON found under {session_dir}")
    return None


def load_trial_manifest(path: str | Path) -> pd.DataFrame:
    """Load a generated manifest and normalize its key numeric columns."""

    df = pd.read_csv(path)
    for col in ["block", "trial", "global_trial", "trigger", "keep_trial"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_onset(session_dir: Path, *, run_missing: bool) -> pd.DataFrame:
    path = session_dir / "analysis_onset" / "onset_trial_level.csv"
    if not path.exists():
        if not run_missing:
            return pd.DataFrame()
        logger.info("selection: running onset for %s", session_dir)
        run_onset_analysis(session_dir)
    if not path.exists():
        raise FileNotFoundError(f"Onset output not found after analysis: {path}")
    return pd.read_csv(path)


def _ensure_asr(session_dir: Path, *, run_missing: bool) -> pd.DataFrame:
    path = session_dir / "analysis_asr_qwen" / "asr_trial_level.csv"
    if not path.exists():
        if not run_missing:
            return pd.DataFrame()
        logger.info("selection: running ASR for %s", session_dir)
        run_asr_analysis(session_dir, ensure_onset=True)
    if not path.exists():
        raise FileNotFoundError(f"ASR output not found after analysis: {path}")
    return pd.read_csv(path)


def _normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in ["block", "trial"]:
        if col not in result.columns:
            raise ValueError(f"Source table is missing key column {col!r}")
        result[col] = pd.to_numeric(result[col], errors="coerce").astype("Int64")
    return result


def _rename_first_existing(df: pd.DataFrame, mapping: dict[str, list[str]]) -> pd.DataFrame:
    result = df.copy()
    for target, candidates in mapping.items():
        if target in result.columns:
            continue
        source = next((candidate for candidate in candidates if candidate in result.columns), None)
        if source is not None:
            result[target] = result[source]
    return result


def _target_pinyin(row: pd.Series) -> str:
    for col in ["pinyin", "target_pinyin_tone3"]:
        value = row.get(col, "")
        if strip_pinyin_tone(value):
            return str(value)
    char = row.get("char", "")
    if char is None or (isinstance(char, float) and pd.isna(char)) or str(char).strip() == "":
        return ""
    try:
        return tone3(str(char))
    except SystemExit:
        return ""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build onset/ASR trial selection manifest")
    parser.add_argument("session_dir", help="Modern production session directory")
    parser.add_argument("--output-dir", default=None, help="Optional output directory")
    parser.add_argument("--asr-policy", default="pinyin_fuzzy", choices=["pinyin_fuzzy"])
    parser.add_argument("--no-run-missing", action="store_true", help="Do not run missing onset/ASR analyses")
    parser.add_argument("--force", action="store_true", help="Rebuild even if manifest already exists")
    args = parser.parse_args(argv)

    result = build_trial_manifest(
        args.session_dir,
        asr_policy=args.asr_policy,
        output_dir=args.output_dir,
        run_missing=not args.no_run_missing,
        force=args.force,
    )
    print(f"Manifest: {result.manifest_path}")
    print(json.dumps(result.summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
