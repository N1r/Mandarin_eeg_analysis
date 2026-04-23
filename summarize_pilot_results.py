"""Summarise pilot-level decoding / RSA results into a compact report.

Reads every per-subject `*_decoding.npz` / `*_rsa.npz` under
`results/batch_analysis/results/<group>/<task>/<subject>/` plus the group-level
`*_stats.json` files, and prints:

- Per contrast × task: group-mean peak, peak latency, first-significant-cluster
  onset (from cluster-based permutation stats if present).
- Per-subject peak table for eyeballing between-subject variability.
- Noise-ceiling status for RSA (does the group mean sit inside the ceiling?).

Run:  uv run python analysis/summarize_pilot_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from mandarin_speech_eeg import (  # noqa: E402
    RESULTS_ROOT,
    load_decoding_result,
    load_rsa_result,
)

RESULTS_DIR = RESULTS_ROOT / "batch_analysis" / "results"


def _first_cluster_onset_ms(stats_path: Path) -> float | None:
    if not stats_path.exists():
        return None
    payload = json.loads(stats_path.read_text())
    clusters = payload.get("cluster_masks") or []
    p_values = payload.get("cluster_p_values") or []
    times_s = np.asarray(payload.get("times_s", []))
    if not clusters or times_s.size == 0:
        return None
    onsets = []
    for cluster_indices, p_value in zip(clusters, p_values):
        if not cluster_indices or p_value is None or p_value >= 0.05:
            continue
        onset = float(times_s[int(cluster_indices[0])] * 1000.0)
        if onset >= 0:
            onsets.append(onset)
    return min(onsets) if onsets else None


def _load_group(task_dir: Path, contrast: str, suffix: str, loader) -> dict | None:
    per_subject: list[np.ndarray] = []
    chance = 0.0
    times_s = None
    noise_ceilings: list[np.ndarray] = []
    subjects: list[str] = []
    for subject_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
        path = subject_dir / f"{contrast}_{suffix}.npz"
        if not path.exists():
            continue
        result = loader(path)
        per_subject.append(result.scores)
        chance = result.chance_level
        times_s = result.times_s
        subjects.append(subject_dir.name)
        if suffix == "rsa" and getattr(result, "noise_ceiling", np.empty((0, 0))).size:
            noise_ceilings.append(result.noise_ceiling)
    if len(per_subject) < 1 or times_s is None:
        return None
    arr = np.stack(per_subject, axis=0)
    return {
        "subjects": subjects,
        "scores": arr,
        "chance": chance,
        "times_s": times_s,
        "noise_ceilings": noise_ceilings,
    }


def _peak(scores: np.ndarray, times_s: np.ndarray) -> tuple[float, float]:
    index = int(np.nanargmax(scores))
    return float(scores[index]), float(times_s[index] * 1000.0)


def _fmt_pct(value: float, chance: float) -> str:
    return f"{value * 100.0:5.1f}% (Δ{(value - chance) * 100.0:+.1f})"


def _fmt_rho(value: float) -> str:
    return f"ρ={value:+.3f}"


def summarise_task(group: str, task: str) -> None:
    task_dir = RESULTS_DIR / group / task
    if not task_dir.exists():
        return
    contrasts = sorted({p.stem.split("_decoding")[0] for p in task_dir.glob("*_decoding_group_stats.json")})
    if not contrasts:
        return

    print(f"\n── {group} / {task} ───────────────────────────────")
    for contrast in contrasts:
        dec = _load_group(task_dir, contrast, "decoding", load_decoding_result)
        rsa = _load_group(task_dir, contrast, "rsa", load_rsa_result)
        if dec is None and rsa is None:
            continue

        print(f"\n  ▸ {contrast}")

        if dec is not None:
            group_mean = dec["scores"].mean(axis=0)
            peak, peak_ms = _peak(group_mean, dec["times_s"])
            cluster_onset = _first_cluster_onset_ms(
                task_dir / f"{contrast}_decoding_group_stats.json"
            )
            print(
                f"    decoding | n={len(dec['subjects'])} "
                f"| peak {_fmt_pct(peak, dec['chance'])} @ {peak_ms:.0f} ms"
                + (f" | sig onset {cluster_onset:.0f} ms" if cluster_onset is not None else " | no sig cluster")
            )
            # Per-subject peaks
            per_subj = [
                f"{subj}:{_peak(dec['scores'][i], dec['times_s'])[0] * 100:.1f}%"
                for i, subj in enumerate(dec["subjects"])
            ]
            print("             subjects: " + "  ".join(per_subj))

        if rsa is not None:
            group_mean = rsa["scores"].mean(axis=0)
            peak, peak_ms = _peak(group_mean, rsa["times_s"])
            cluster_onset = _first_cluster_onset_ms(
                task_dir / f"{contrast}_rsa_group_stats.json"
            )
            ceiling_note = ""
            if rsa["noise_ceilings"]:
                avg_lower = np.mean([nc[0] for nc in rsa["noise_ceilings"]], axis=0)
                avg_upper = np.mean([nc[1] for nc in rsa["noise_ceilings"]], axis=0)
                peak_idx = int(np.argmax(group_mean))
                ceiling_note = (
                    f" | ceiling@peak [{avg_lower[peak_idx]:+.2f}, {avg_upper[peak_idx]:+.2f}]"
                )
            print(
                f"    rsa      | n={len(rsa['subjects'])} "
                f"| peak {_fmt_rho(peak)} @ {peak_ms:.0f} ms"
                + (f" | sig onset {cluster_onset:.0f} ms" if cluster_onset is not None else " | no sig cluster")
                + ceiling_note
            )


def main() -> None:
    print(f"Reading pilot results from: {RESULTS_DIR}")
    for group_dir in sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir()):
        for task_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            summarise_task(group_dir.name, task_dir.name)
    print()


if __name__ == "__main__":
    main()
