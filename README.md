# Mandarin Speech EEG Analysis

This repository contains the Python analysis package for a Mandarin speech EEG project. It supports production and perception tasks, including EEG preprocessing, ASR-based production-trial quality control, acoustic onset detection, event-locked and response-locked epoching, time-resolved decoding, RSA, HeteroRC decoding, visualization, and group-level pilot summaries.

The code is organized as a small package, `mandarin_speech_eeg`, with numbered workflow entry points for reproducible stepwise analysis.

## Installation

The recommended environment manager is [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

If `uv` is not available, use a standard Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The main external tools are:

- [MNE-Python](https://mne.tools/stable/index.html) for EEG loading, preprocessing, epoching, and cluster statistics.
- [qwen-asr](https://pypi.org/project/qwen-asr/) with [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) and [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) for production speech recognition and alignment.
- [scikit-learn](https://scikit-learn.org/stable/) for decoding and cross-validation.
- [rsatoolbox](https://rsatoolbox.readthedocs.io/) for representational similarity analysis utilities.
- [pypinyin](https://pypinyin.readthedocs.io/) for pinyin-based ASR trial selection.

All step scripts are configured to use a single worker by default to keep interactive runs responsive.

## Data Layout

Raw data are not tracked by git. The package expects data outside this repository, in the parent project folder:

```text
EEG_Full_analysis/
|-- Data/
|   |-- marker_condition.csv
|   `-- Production_Perception/
|       |-- *.bdf
|       `-- sub-*/ses-*/
|           |-- eeg_data/*.bdf
|           |-- continuous_audio/*.wav
|           |-- session_*.csv
|           `-- session_*.json
`-- Full_Analysis/
    `-- analysis/
```

Modern production sessions can use ASR, onset detection, and trial-level selection. Legacy BDF-only sessions remain supported through marker-based event metadata.

## Main Workflow

Run commands from the repository root, `Full_Analysis/analysis`.

```bash
python -m mandarin_speech_eeg.step_1_asr --group Production_Perception --subject <subject_id>
python -m mandarin_speech_eeg.step_2_selection --group Production_Perception --subject <subject_id>
python -m mandarin_speech_eeg.step_3_onset --group Production_Perception --subject <subject_id>
python -m mandarin_speech_eeg.step_4_epoch --group Production_Perception --task production --subject <subject_id>
python -m mandarin_speech_eeg.step_5_decoding --group Production_Perception --task production --subject <subject_id> --quick
python -m mandarin_speech_eeg.step_6_rsa --group Production_Perception --task production --subject <subject_id> --quick
python -m mandarin_speech_eeg.step_7_heterorc --group Production_Perception --task production --subject <subject_id> --quick
python -m mandarin_speech_eeg.step_8_integrated --group Production_Perception --task production --subject <subject_id>
python -m mandarin_speech_eeg.step_9_group --group Production_Perception --task production --subjects <subject_1> <subject_2> <subject_3>
```

For legacy BDF-only data, use the epoch step with `--legacy`:

```bash
python -m mandarin_speech_eeg.step_4_epoch --group Production_Perception --task perception --subject <subject_id> --legacy
```

## V2 Minimal Preprocessing

The v2 runner is a separate script that writes to `results/batch_analysis_v2/` and does not overwrite v1 outputs. It keeps preprocessing intentionally minimal: 0.5-30 Hz filtering, 100 Hz resampling, common-average reference, no ICA, and no AutoReject.

Use it stepwise while debugging:

```bash
python run_batch_analysis_v2.py --quick --no-stats --group Production_Perception --stage epoch --lock both
python run_batch_analysis_v2.py --quick --no-stats --group Production_Perception --stage decoding --lock event
python run_batch_analysis_v2.py --quick --no-stats --group Production_Perception --stage rsa --lock event
python run_batch_analysis_v2.py --quick --no-stats --group Production_Perception --stage heterorc --lock event
```

Available stages are `epoch`, `decoding`, `rsa`, `heterorc`, and `all`. Available locks are `event`, `response`, and `both`.

## Outputs

V1 stepwise outputs are written under:

```text
Full_Analysis/results/batch_analysis/
```

V2 outputs are written under:

```text
Full_Analysis/results/batch_analysis_v2/
```

Common subject-level files include:

```text
epoch_only/*-epo.fif
*_decoding.npz
*_rsa.npz
*_heterorc_decoding.npz
*_decoding.png
*_rsa.png
*_heterorc_decoding.png
*_integrated.png
```

Production quality-control outputs are stored next to the session data:

```text
analysis_asr_qwen/
analysis_onset/
analysis_selection/trial_manifest.csv
```

## Static Report

The repository includes a lightweight GitHub Pages report under:

```text
docs/index.html
docs/assets/
```

The report is not base64-embedded. Figures are stored as relative assets so the page can be served directly by GitHub Pages. In the repository settings, set GitHub Pages to deploy from the `docs/` folder on the main branch.

## Repository Hygiene

The repository intentionally excludes raw data, generated results, figures, local environments, model caches, and reference materials. Before uploading to GitHub, check:

```bash
git status --short --ignored
```

The only ignored large local folder expected inside the repository is usually `.venv/`.

## Contact

For questions about the analysis workflow, contact dingyr@hum.leidenuniv.nl.
