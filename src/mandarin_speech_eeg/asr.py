"""Qwen ASR evaluation aligned to onset-derived trial windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from pypinyin import Style, pinyin
except Exception:  # pragma: no cover
    Style = None
    pinyin = None

from .onset import build_trial_analysis, build_trial_records, load_csv_rows, load_json, run_onset_analysis

try:
    from qwen_asr import Qwen3ASRModel
except Exception:  # pragma: no cover
    Qwen3ASRModel = None


DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
DEFAULT_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
DEFAULT_OUTPUT_DIRNAME = "analysis_asr_qwen"


def load_trial_windows(session_dir: Path) -> list[dict[str, Any]]:
    analysis_csv = session_dir / "analysis_onset" / "onset_trial_level.csv"
    if analysis_csv.exists():
        with analysis_csv.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        return [{"participant": row["participant"], "block": int(row["block"]), "trial": int(row["trial"]), "global_trial": int(row["global_trial"]), "stimulus_id": row["stimulus_id"], "char": row["char"], "pinyin": row["pinyin"], "tone": int(row["tone"]), "initial_type": row["initial_type"], "rhyme_type": row["rhyme_type"], "picture_onset_in_segment_ms": float(row["picture_onset_in_segment_ms"]), "effective_trial_end_in_segment_ms": float(row["effective_trial_end_in_segment_ms"])} for row in rows]

    csv_candidates = sorted(session_dir.glob("session_*_*.csv"))
    json_candidates = sorted(session_dir.glob("session_*_*.json"))
    csv_path = next(path for path in csv_candidates if "practice" not in path.name)
    json_path = next(path for path in json_candidates if "practice" not in path.name)
    rows = load_csv_rows(csv_path)
    records = build_trial_records([row for row in rows if row.get("phase") == "formal"])
    session_data = load_json(json_path)
    analysis_rows = build_trial_analysis(session_dir, records, session_data)
    return [{"participant": row["participant"], "block": int(row["block"]), "trial": int(row["trial"]), "global_trial": int(row["global_trial"]), "stimulus_id": row["stimulus_id"], "char": row["char"], "pinyin": row["pinyin"], "tone": int(row["tone"]), "initial_type": row["initial_type"], "rhyme_type": row["rhyme_type"], "picture_onset_in_segment_ms": float(row["picture_onset_in_segment_ms"]), "effective_trial_end_in_segment_ms": float(row["effective_trial_end_in_segment_ms"])} for row in analysis_rows]


def tone3(text: str) -> str:
    if not text:
        return ""
    if pinyin is None or Style is None:
        raise SystemExit("pypinyin is not installed in this environment.")
    pieces = pinyin(text, style=Style.TONE3, strict=False, neutral_tone_with_five=True)
    return " ".join(piece[0] for piece in pieces if piece and piece[0])


def first_syllable_tone(text: str) -> str:
    if not text:
        return ""
    syll = text.split()[0]
    for ch in reversed(syll):
        if ch.isdigit():
            return ch
    return ""


def pick_device() -> tuple[str, Any]:
    if torch is None:
        raise SystemExit("torch is not installed in this environment.")
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_model(model_name: str, aligner_name: str) -> tuple[Any, str, str]:
    if torch is None:
        raise SystemExit("torch is not installed in this environment.")
    if pinyin is None or Style is None:
        raise SystemExit("pypinyin is not installed in this environment.")
    if Qwen3ASRModel is None:
        raise SystemExit("qwen_asr is not installed in this environment.")
    device, dtype = pick_device()
    kwargs = {"max_inference_batch_size": 1, "max_new_tokens": 512, "low_cpu_mem_usage": True, "dtype": dtype, "trust_remote_code": True}
    forced_aligner_kwargs = {"low_cpu_mem_usage": True, "dtype": dtype, "trust_remote_code": True}
    model = Qwen3ASRModel.from_pretrained(model_name, forced_aligner=aligner_name, forced_aligner_kwargs=forced_aligner_kwargs, **kwargs)
    if device != "cpu":
        model.model.to(device)
        if model.forced_aligner is not None:
            model.forced_aligner.model.to(device)
    return model, device, str(dtype).replace("torch.", "")


def item_midpoint_ms(item: Any) -> float:
    return (float(item.start_time) + float(item.end_time)) * 500.0


def substitution_cost(trial: dict[str, Any], item: Any) -> float:
    pred_char = item.text
    target_char = trial["char"]
    pred_pinyin = tone3(pred_char)
    target_pinyin = tone3(target_char)
    pred_tone = first_syllable_tone(pred_pinyin)
    target_tone = str(trial["tone"])
    if pred_char == target_char:
        lexical_cost = 0.0
    elif pred_pinyin and pred_pinyin == target_pinyin:
        lexical_cost = 0.18
    elif pred_tone and pred_tone == target_tone:
        lexical_cost = 0.95
    else:
        lexical_cost = 1.45
    midpoint = item_midpoint_ms(item)
    start_ms = trial["picture_onset_in_segment_ms"]
    end_ms = trial["effective_trial_end_in_segment_ms"]
    if start_ms <= midpoint <= end_ms:
        time_cost = 0.0
    else:
        distance_ms = min(abs(midpoint - start_ms), abs(midpoint - end_ms))
        time_cost = min(1.0, distance_ms / 1200.0)
    return lexical_cost + 0.55 * time_cost


def omission_cost(trial: dict[str, Any], items: list[Any]) -> float:
    start_ms = trial["picture_onset_in_segment_ms"]
    end_ms = trial["effective_trial_end_in_segment_ms"]
    near_items = [item for item in items if start_ms <= item_midpoint_ms(item) <= end_ms]
    return 1.0 if near_items else 0.7


def insertion_cost(item: Any, trials: list[dict[str, Any]]) -> float:
    midpoint = item_midpoint_ms(item)
    near_trial = any(trial["picture_onset_in_segment_ms"] <= midpoint <= trial["effective_trial_end_in_segment_ms"] for trial in trials)
    return 0.85 if near_trial else 0.55


def align_trials_to_items(trials: list[dict[str, Any]], items: list[Any]) -> tuple[list[Any | None], list[Any]]:
    n = len(trials)
    m = len(items)
    dp = [[math.inf] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + omission_cost(trials[i - 1], items)
        back[i][0] = ("omit", i - 1, None)
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + insertion_cost(items[j - 1], trials)
        back[0][j] = ("insert", None, j - 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = dp[i - 1][j - 1] + substitution_cost(trials[i - 1], items[j - 1])
            del_cost = dp[i - 1][j] + omission_cost(trials[i - 1], items)
            ins_cost = dp[i][j - 1] + insertion_cost(items[j - 1], trials)
            best = min(sub_cost, del_cost, ins_cost)
            dp[i][j] = best
            back[i][j] = ("match", i - 1, j - 1) if best == sub_cost else ("omit", i - 1, None) if best == del_cost else ("insert", None, j - 1)
    aligned = [None] * n
    insertions = []
    i, j = n, m
    while i > 0 or j > 0:
        action, trial_idx, item_idx = back[i][j]
        if action == "match":
            aligned[trial_idx] = items[item_idx]
            i -= 1
            j -= 1
        elif action == "omit":
            aligned[trial_idx] = None
            i -= 1
        else:
            insertions.append(items[item_idx])
            j -= 1
    return aligned, list(reversed(insertions))


def summarize_trial_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    predicted = [row for row in rows if row["pred_char"]]
    tone_den = sum(1 for row in rows if row["pred_tone"])
    return {
        "n_trials": total,
        "n_predicted": len(predicted),
        "char_accuracy": sum(row["word_match"] for row in rows) / total if total else 0.0,
        "pinyin_tone_accuracy": sum(row["pinyin_match"] for row in rows) / total if total else 0.0,
        "tone_accuracy": sum(row["tone_match"] for row in rows if row["pred_tone"]) / tone_den if tone_den else None,
        "orthographic_mismatch_count": sum(1 for row in rows if row["pred_char"] and not row["word_match"] and row["pinyin_match"]),
        "omission_count": sum(1 for row in rows if row["alignment_status"] == "omission"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_report(session_dir: Path, output_dir: Path, config_info: dict[str, str], block_results: dict[int, dict[str, Any]], trial_rows: list[dict[str, Any]]) -> Path:
    overall = summarize_trial_rows(trial_rows)
    report = "\n".join([
        "# Qwen3-ASR Block Evaluation",
        "",
        f"Session: `{session_dir}`",
        "",
        f"- ASR model: `{config_info['model_name']}`",
        f"- Forced aligner: `{config_info['aligner_name']}`",
        f"- Device: `{config_info['device']}`",
        f"- Dtype: `{config_info['dtype']}`",
        f"- Trials: {overall['n_trials']}",
        f"- Trials with aligned prediction: {overall['n_predicted']}",
        f"- pinyin_tone_accuracy: {overall['pinyin_tone_accuracy']:.2%}",
        f"- char_accuracy: {overall['char_accuracy']:.2%}",
        f"- Omission count: {overall['omission_count']}",
    ])
    report_path = output_dir / "asr_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def resolve_block_audio(session_dir: Path, block: int, rows: list[dict[str, Any]]) -> Path:
    """Find the continuous audio file for a block using trial metadata first."""

    audio_dir = session_dir / "continuous_audio"
    candidate_names: list[str] = []
    if rows:
        participant = str(rows[0].get("participant", "")).strip()
        if participant:
            candidate_names.extend(
                [
                    f"sub-{participant}_block-{block:02d}_continuous.wav",
                    f"sub-{int(participant):02d}_block-{block:02d}_continuous.wav" if participant.isdigit() else "",
                ]
            )
    candidate_names.append(f"{session_dir.parent.name}_block-{block:02d}_continuous.wav")

    for name in candidate_names:
        if not name:
            continue
        path = audio_dir / name
        if path.exists():
            return path

    matches = sorted(audio_dir.glob(f"*_block-{block:02d}_continuous.wav"))
    if matches:
        preferred = [
            path
            for path in matches
            if rows and str(rows[0].get("participant", "")).strip() in path.name
        ]
        return preferred[0] if preferred else matches[0]

    raise FileNotFoundError(
        f"No continuous audio file found for block {block:02d} under {audio_dir}"
    )


def run_asr_analysis(session_dir: Path | str, *, blocks: list[int] | None = None, model_name: str = DEFAULT_ASR_MODEL, aligner_name: str = DEFAULT_ALIGNER_MODEL, output_dir: Path | str | None = None, ensure_onset: bool = False) -> dict[str, Any]:
    session_dir = Path(session_dir).resolve()
    if ensure_onset and not (session_dir / "analysis_onset" / "onset_trial_level.csv").exists():
        run_onset_analysis(session_dir)
    output_path = Path(output_dir).resolve() if output_dir else (session_dir / DEFAULT_OUTPUT_DIRNAME)
    output_path.mkdir(parents=True, exist_ok=True)
    trial_rows = load_trial_windows(session_dir)
    if blocks:
        selected = set(blocks)
        trial_rows = [row for row in trial_rows if row["block"] in selected]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trial_rows:
        grouped[row["block"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["trial"])
    model, device, dtype_label = load_model(model_name, aligner_name)
    block_results: dict[int, dict[str, Any]] = {}
    all_trial_rows = []
    transcript_dump = []
    for block, rows in sorted(grouped.items()):
        wav_path = resolve_block_audio(session_dir, block, rows)
        result = model.transcribe(audio=str(wav_path), language="Chinese", return_time_stamps=True)[0]
        items = list(result.time_stamps) if result.time_stamps is not None else []
        aligned_items, orphan_items = align_trials_to_items(rows, items)
        block_results[block] = {"text": result.text, "language": result.language, "normalized_text": " ".join(item.text for item in items), "orphan_item_count": len(orphan_items), "trial_rows": []}
        transcript_dump.append({"block": block, "audio_file": wav_path.name, "language": result.language, "text": result.text})
        for row, aligned_item in zip(rows, aligned_items):
            pred_char = aligned_item.text if aligned_item else ""
            pred_pinyin_tone3 = tone3(pred_char)
            pred_tone = first_syllable_tone(pred_pinyin_tone3)
            trial_eval = {"participant": row["participant"], "block": row["block"], "trial": row["trial"], "global_trial": row["global_trial"], "stimulus_id": row["stimulus_id"], "char": row["char"], "target_pinyin": row["pinyin"], "target_pinyin_tone3": tone3(row["char"]), "tone": row["tone"], "initial_type": row["initial_type"], "rhyme_type": row["rhyme_type"], "trial_start_ms": round(row["picture_onset_in_segment_ms"], 3), "trial_end_ms": round(row["effective_trial_end_in_segment_ms"], 3), "pred_text_window": pred_char, "pred_char": pred_char, "pred_pinyin_tone3": pred_pinyin_tone3, "pred_tone": pred_tone, "aligned_item_count": 0 if aligned_item is None else 1, "alignment_status": "omission" if aligned_item is None else "aligned", "word_match": int(pred_char == row["char"]), "pinyin_match": int(pred_pinyin_tone3 == tone3(row["char"]) and bool(pred_pinyin_tone3)), "tone_match": int(pred_tone == str(row["tone"]) and bool(pred_tone))}
            block_results[block]["trial_rows"].append(trial_eval)
            all_trial_rows.append(trial_eval)
    write_csv(output_path / "asr_trial_level.csv", all_trial_rows)
    block_summary_rows = []
    for block in sorted(block_results):
        summary = summarize_trial_rows(block_results[block]["trial_rows"])
        block_summary_rows.append({"block": block, "n_trials": summary["n_trials"], "n_predicted": summary["n_predicted"], "pinyin_tone_accuracy": round(summary["pinyin_tone_accuracy"], 6), "char_accuracy": round(summary["char_accuracy"], 6), "tone_accuracy": None if summary["tone_accuracy"] is None else round(summary["tone_accuracy"], 6), "orthographic_mismatch_count": summary["orthographic_mismatch_count"], "omission_count": summary["omission_count"], "orphan_item_count": block_results[block]["orphan_item_count"], "language": block_results[block]["language"], "transcript_text": block_results[block]["text"], "normalized_transcript_text": block_results[block]["normalized_text"]})
    write_csv(output_path / "asr_block_summary.csv", block_summary_rows)
    transcript_path = output_path / "asr_block_transcripts.json"
    transcript_path.write_text(json.dumps(transcript_dump, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = render_report(session_dir, output_path, {"model_name": model_name, "aligner_name": aligner_name, "device": device, "dtype": dtype_label}, block_results, all_trial_rows)
    return {"session_dir": session_dir, "output_dir": output_path, "report_path": report_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate block-level Qwen3-ASR against single-character targets.")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--block", type=int, action="append", help="Only run selected block(s).")
    parser.add_argument("--model", default=DEFAULT_ASR_MODEL)
    parser.add_argument("--aligner", default=DEFAULT_ALIGNER_MODEL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ensure-onset", action="store_true", help="Generate onset analysis first if missing")
    args = parser.parse_args(argv)
    result = run_asr_analysis(args.session_dir, blocks=args.block, model_name=args.model, aligner_name=args.aligner, output_dir=args.output_dir, ensure_onset=args.ensure_onset)
    print(f"Wrote ASR analysis to: {result['output_dir']}")
    print(f"Report: {result['report_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
