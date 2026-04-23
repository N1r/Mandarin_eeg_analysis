"""Unified plotting module.

All time-series figures (decoding, RSA, HeteroRC, overlays, integrated) share a
single rendering core so style tweaks stay in one place.

Visualization conventions follow Grootswagers, Wardle & Carlson (2017, JoCN):
pre-stim sanity shade, chance reference, cluster-based significance as bottom
dots, behavioural speech-onset as a dashed vertical, peak marker shown only
when a significant signal was found.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .config import AnalysisConfig
from .decoding import DecodingResult
from .rsa import RSAResult
from .statistics import StatisticalTestResult

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.labelpad": 3.5,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "legend.handlelength": 1.6,
        "legend.labelspacing": 0.25,
        "legend.borderaxespad": 0.4,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

FIGURE_SIZE = (6.1, 4.15)
LINE_WIDTH = 1.75
SEM_ALPHA = 0.18
STIMULUS_COLOR = "#bdbdbd"
STIMULUS_EDGE_COLOR = "#111111"
CHANCE_STYLE = {"color": "#1a1a1a", "linestyle": (0, (4, 3)), "linewidth": 0.9, "alpha": 0.95}
PRESTIM_COLOR = "#e6e6e6"
SIG_DOT_Y_FRAC = 0.105
SIG_DOT_SIZE = 8
STIM_BAR_Y_FRAC = 0.018
STIM_BAR_HEIGHT_FRAC = 0.042

# Okabe–Ito CVD-safe palette.
OKABE_ITO = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]

CONTRAST_COLORS = {
    "Tone": "#0072B2",
    "Animacy": "#D55E00",
    "Initial Type": "#009E73",
    "Rhyme Type": "#CC79A7",
}


@dataclass
class _CurveStyle:
    color: str
    ylabel: str
    as_percentage: bool  # percent scale => multiply by 100; zero_line treated as percent
    zero_line: float     # already in display units
    label: str = "signal"


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def plot_decoding_time_series(
    result: DecodingResult,
    config: AnalysisConfig,
    save_dir: str | Path,
    contrast_name: str,
    stats: StatisticalTestResult | None = None,
    speech_onset_ms: float | None = None,
) -> None:
    style = _CurveStyle(
        color=_contrast_color(contrast_name),
        ylabel="Classifier accuracy (% correct)",
        as_percentage=True,
        zero_line=result.chance_level * 100.0,
        label=f"{contrast_name} decoding (standard pipeline)",
    )
    sem = _fold_sem(result)
    _one_panel_figure(
        result=result, style=style, config=config,
        title=None,
        save_path=Path(save_dir) / f"{_stem(contrast_name)}_decoding",
        sem=sem, stats=stats, speech_onset_ms=speech_onset_ms,
    )


def plot_rsa_time_series(
    result: RSAResult,
    config: AnalysisConfig,
    save_dir: str | Path,
    contrast_name: str,
    stats: StatisticalTestResult | None = None,
    noise_ceiling: tuple[np.ndarray, np.ndarray] | None = None,
    speech_onset_ms: float | None = None,
    show_noise_ceiling: bool | None = None,
) -> None:
    if show_noise_ceiling is None:
        show_noise_ceiling = config.plotting.show_subject_rsa_noise_ceiling
    if not show_noise_ceiling:
        noise_ceiling = None

    style = _CurveStyle(
        color=_contrast_color(contrast_name),
        ylabel="Spearman rho",
        as_percentage=False,
        zero_line=0.0,
        label=f"{contrast_name} RSA",
    )
    _one_panel_figure(
        result=result, style=style, config=config,
        title=None,
        save_path=Path(save_dir) / f"{_stem(contrast_name)}_rsa",
        noise_ceiling=noise_ceiling, stats=stats, speech_onset_ms=speech_onset_ms,
    )


def plot_heterorc_decoding_time_series(
    result,
    config: AnalysisConfig,
    save_dir: str | Path,
    contrast_name: str,
    stats: StatisticalTestResult | None = None,
    speech_onset_ms: float | None = None,
) -> None:
    """HeteroRC decoding uses the same renderer as standard decoding."""
    style = _CurveStyle(
        color=_contrast_color(contrast_name), ylabel="Classifier accuracy (% correct)",
        as_percentage=True, zero_line=result.chance_level * 100.0,
        label=f"{contrast_name} HeteroRC decoding",
    )
    sem = _fold_sem(result)
    _one_panel_figure(
        result=result, style=style, config=config,
        title=f"{contrast_name} | {config.task} | HeteroRC",
        save_path=Path(save_dir) / f"{_stem(contrast_name)}_heterorc_decoding",
        sem=sem, stats=stats, speech_onset_ms=speech_onset_ms,
    )


def plot_group_time_series(
    subject_scores: np.ndarray,
    times_s: np.ndarray,
    chance_level: float,
    contrast_name: str,
    mode: str,
    config: AnalysisConfig,
    save_dir: str | Path,
    stats: StatisticalTestResult | None = None,
    speech_onset_ms: float | None = None,
) -> None:
    is_decoding = mode == "decoding"
    style = _CurveStyle(
        color=config.decoding_color if is_decoding else config.rsa_color,
        ylabel="Accuracy (%)" if is_decoding else "Spearman rho",
        as_percentage=is_decoding,
        zero_line=chance_level * 100.0 if is_decoding else 0.0,
    )
    mean = subject_scores.mean(axis=0)
    sem = subject_scores.std(axis=0) / np.sqrt(len(subject_scores))
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    _render_panel(
        axis=axis, times_ms=times_s * 1000.0, scores=mean, sem=sem,
        style=style, config=config, stats=stats,
        speech_onset_ms=speech_onset_ms,
    )
    _apply_axis_chrome(axis, title=f"{contrast_name} | group ({mode}) | {config.task}", x_label=True, show_legend=True, style=style, stats_drew_dots=_stats_has_clusters(stats))
    _save_figure(figure, Path(save_dir) / f"{_stem(contrast_name)}_{mode}_group", config.plotting.save_formats)


def plot_heterorc_group_time_series(
    subject_scores: np.ndarray,
    times_s: np.ndarray,
    chance_level: float,
    contrast_name: str,
    config: AnalysisConfig,
    save_dir: str | Path,
    stats: StatisticalTestResult | None = None,
    speech_onset_ms: float | None = None,
) -> None:
    style = _CurveStyle(
        color=config.heterorc_color, ylabel="Accuracy (%)",
        as_percentage=True, zero_line=chance_level * 100.0,
    )
    mean = subject_scores.mean(axis=0)
    sem = subject_scores.std(axis=0) / np.sqrt(len(subject_scores))
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    _render_panel(
        axis=axis, times_ms=times_s * 1000.0, scores=mean, sem=sem,
        style=style, config=config, stats=stats,
        speech_onset_ms=speech_onset_ms,
    )
    _apply_axis_chrome(axis, title=f"{contrast_name} | group | HeteroRC | {config.task}", x_label=True, show_legend=True, style=style, stats_drew_dots=_stats_has_clusters(stats))
    _save_figure(figure, Path(save_dir) / f"{_stem(contrast_name)}_heterorc_decoding_group", config.plotting.save_formats)


def plot_multi_contrast_overlay(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray | None]],
    times_s: np.ndarray,
    chance_level: float,
    mode: str,
    config: AnalysisConfig,
    save_dir: str | Path,
    title: str | None = None,
    stats_by_contrast: Mapping[str, StatisticalTestResult] | None = None,
    speech_onset_ms: float | None = None,
) -> None:
    style = _overlay_style(mode, chance_level)
    palette = _distinct_palette(len(curves))
    items = []
    for (name, (scores, sem)), color in zip(curves.items(), palette):
        stat = (stats_by_contrast or {}).get(name)
        items.append({"label": name, "scores": scores, "sem": sem, "stats": stat, "color": color})

    figure, axis = plt.subplots(figsize=(7.6, 3.6))
    _render_overlay_panel(
        axis=axis, times_ms=times_s * 1000.0, items=items, style=style,
        config=config, speech_onset_ms=speech_onset_ms,
    )
    _apply_axis_chrome(axis, title=title or f"Contrasts overlay | {config.task}", x_label=True, show_legend=True, style=style, overlay=True)
    _save_figure(figure, Path(save_dir) / f"overlay_{mode}_{config.task}", config.plotting.save_formats)


def plot_condition_comparison(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray | None]],
    times_s: np.ndarray,
    chance_level: float,
    mode: str,
    config: AnalysisConfig,
    save_dir: str | Path,
    title: str,
    stats_by_condition: Mapping[str, StatisticalTestResult] | None = None,
    speech_onset_ms: float | None = None,
) -> None:
    style = _overlay_style(mode, chance_level)
    is_decoding = mode == "decoding"
    color_map = {
        "production": config.plotting.decoding_color_production if is_decoding else config.plotting.rsa_color_production,
        "perception": config.plotting.decoding_color_perception if is_decoding else config.plotting.rsa_color_perception,
    }
    items = []
    for name, (scores, sem) in curves.items():
        color = color_map.get(name.lower(), "#555555")
        stat = (stats_by_condition or {}).get(name)
        items.append({"label": name, "scores": scores, "sem": sem, "stats": stat, "color": color})

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    _render_overlay_panel(
        axis=axis, times_ms=times_s * 1000.0, items=items, style=style,
        config=config, speech_onset_ms=speech_onset_ms,
    )
    _apply_axis_chrome(axis, title=title, x_label=True, show_legend=True, style=style, overlay=True)
    _save_figure(figure, Path(save_dir) / f"compare_{mode}_{_stem(title)}", config.plotting.save_formats)


def plot_contrast_integrated(
    contrast_name: str,
    decoding_result,
    rsa_result,
    heterorc_result,
    decoding_stats: StatisticalTestResult | None,
    rsa_stats: StatisticalTestResult | None,
    heterorc_stats: StatisticalTestResult | None,
    config: AnalysisConfig,
    save_dir: str | Path,
    speech_onset_ms: float | None = None,
    show_rsa_noise_ceiling: bool | None = None,
) -> None:
    """Three stacked panels (Decoding / RSA / HeteroRC) for one contrast."""
    if show_rsa_noise_ceiling is None:
        show_rsa_noise_ceiling = config.plotting.show_subject_rsa_noise_ceiling

    panels = []
    if decoding_result is not None:
        panels.append(("Decoding", decoding_result, decoding_stats, _CurveStyle(
            color=config.decoding_color, ylabel="Decoding\nAccuracy (%)",
            as_percentage=True, zero_line=decoding_result.chance_level * 100.0)))
    if rsa_result is not None:
        panels.append(("RSA", rsa_result, rsa_stats, _CurveStyle(
            color=config.rsa_color, ylabel="RSA\nSpearman rho",
            as_percentage=False, zero_line=0.0)))
    if heterorc_result is not None:
        panels.append(("HeteroRC", heterorc_result, heterorc_stats, _CurveStyle(
            color=config.heterorc_color, ylabel="HeteroRC\nAccuracy (%)",
            as_percentage=True, zero_line=heterorc_result.chance_level * 100.0)))
    if not panels:
        return

    figure, axes = plt.subplots(len(panels), 1, figsize=(7.0, 2.4 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)
    for i, (label, result, stats, style) in enumerate(panels):
        noise_ceiling = None
        if label == "RSA" and show_rsa_noise_ceiling:
            nc = getattr(result, "noise_ceiling", None)
            if nc is not None and nc.size:
                noise_ceiling = (nc[0], nc[1])
        _render_panel(
            axis=axes[i], times_ms=result.times_s * 1000.0, scores=result.scores,
            sem=_fold_sem(result), stats=stats,
            null_distribution=getattr(result, "null_distribution", None),
            noise_ceiling=noise_ceiling, style=style, config=config,
            speech_onset_ms=speech_onset_ms,
            draw_stim_bar=(i == 0),
        )
        _apply_axis_chrome(axes[i], title=None, x_label=(i == len(panels) - 1),
                           show_legend=False, style=style, stats_drew_dots=None)

    axes[0].set_title(f"{contrast_name} | {config.task}")
    figure.tight_layout()
    _save_figure(figure, Path(save_dir) / f"{_stem(contrast_name)}_integrated", _publication_formats(config))


def plot_modality_grid(
    contrasts: Sequence[str],
    results_by_modality: Mapping[str, Mapping[str, object]],
    stats_by_modality: Mapping[str, Mapping[str, StatisticalTestResult | None]] | None,
    config: AnalysisConfig,
    save_dir: str | Path,
    stem: str,
    title: str,
    speech_onset_ms: float | None = None,
) -> None:
    """Rows = modalities (decoding / RSA / HeteroRC), each overlaying contrasts."""
    modalities = []
    for mod_key, color_attr, ylabel, is_pct in (
        ("decoding", "decoding_color", "Decoding\nAccuracy (%)", True),
        ("rsa", "rsa_color", "RSA\nSpearman rho", False),
        ("heterorc", "heterorc_color", "HeteroRC\nAccuracy (%)", True),
    ):
        if results_by_modality.get(mod_key):
            modalities.append((mod_key, getattr(config, color_attr), ylabel, is_pct))
    if not modalities:
        return

    palette = _distinct_palette(len(contrasts))
    contrast_colors = {c: palette[i] for i, c in enumerate(contrasts)}

    figure, axes = plt.subplots(len(modalities), 1, figsize=(7.2, 2.5 * len(modalities)), sharex=True)
    axes = np.atleast_1d(axes)

    for i, (mod_key, _color, ylabel, is_pct) in enumerate(modalities):
        modality_results = results_by_modality[mod_key]
        modality_stats = (stats_by_modality or {}).get(mod_key, {})
        items = []
        chance_line = 0.0
        times_ms = None
        for contrast in contrasts:
            r = modality_results.get(contrast)
            if r is None:
                continue
            times_ms = r.times_s * 1000.0
            chance_line = r.chance_level * (100.0 if is_pct else 1.0)
            items.append({
                "label": contrast,
                "scores": r.scores,
                "sem": _fold_sem(r),
                "stats": modality_stats.get(contrast) if modality_stats else None,
                "null_distribution": getattr(r, "null_distribution", None),
                "color": contrast_colors[contrast],
            })
        if not items or times_ms is None:
            continue

        style = _CurveStyle(color="#444444", ylabel=ylabel, as_percentage=is_pct, zero_line=chance_line)
        _render_overlay_panel(
            axis=axes[i], times_ms=times_ms, items=items, style=style,
            config=config, speech_onset_ms=speech_onset_ms,
            draw_stim_bar=(i == 0),
        )
        _apply_axis_chrome(axes[i], title=None, x_label=(i == len(modalities) - 1),
                           show_legend=(i == 0), style=style, overlay=True)

    axes[0].set_title(title)
    figure.tight_layout()
    _save_figure(figure, Path(save_dir) / stem, _publication_formats(config))


def plot_time_generalization(
    tgm_scores: np.ndarray,
    times_s: np.ndarray,
    chance_level: float,
    title: str,
    config: AnalysisConfig,
    save_dir: str | Path,
    significance_mask: np.ndarray | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(5.6, 5.0))
    times_ms = times_s * 1000.0
    extent = [times_ms[0], times_ms[-1], times_ms[0], times_ms[-1]]
    data_pct = tgm_scores * 100.0
    chance_pct = chance_level * 100.0
    vabs = max(abs(data_pct.min() - chance_pct), abs(data_pct.max() - chance_pct), 2.0)
    norm = TwoSlopeNorm(vmin=chance_pct - vabs, vcenter=chance_pct, vmax=chance_pct + vabs)
    image = axis.imshow(
        data_pct, origin="lower", aspect="equal", cmap=config.plotting.tgm_cmap,
        interpolation="bilinear", extent=extent, norm=norm,
    )
    axis.axhline(0, color="black", linewidth=0.6, alpha=0.45)
    axis.axvline(0, color="black", linewidth=0.6, alpha=0.45)
    axis.plot([times_ms[0], times_ms[-1]], [times_ms[0], times_ms[-1]],
              color="black", linewidth=0.6, alpha=0.35, linestyle=":")
    if significance_mask is not None and significance_mask.any():
        axis.contour(times_ms, times_ms, significance_mask.astype(float),
                     levels=[0.5], colors="black", linewidths=0.8)
    colorbar = figure.colorbar(image, ax=axis, shrink=0.82, pad=0.03)
    colorbar.set_label("Accuracy (%)")
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.tick_params(width=0.6, length=2.5, labelsize=8)
    axis.set_title(title)
    axis.set_xlabel("Train time (ms)")
    axis.set_ylabel("Test time (ms)")
    axis.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    _save_figure(figure, Path(save_dir) / "time_generalization", config.plotting.save_formats)


# --------------------------------------------------------------------------
# Core renderers
# --------------------------------------------------------------------------


def _render_panel(
    axis,
    times_ms: np.ndarray,
    scores: np.ndarray,
    *,
    style: _CurveStyle,
    config: AnalysisConfig,
    sem: np.ndarray | None = None,
    stats: StatisticalTestResult | None = None,
    null_distribution: np.ndarray | None = None,
    noise_ceiling: tuple[np.ndarray, np.ndarray] | None = None,
    speech_onset_ms: float | None = None,
    draw_stim_bar: bool = True,
) -> bool:
    """Render one curve + all ornaments into `axis`. Returns True if sig dots were drawn."""
    scale = 100.0 if style.as_percentage else 1.0
    scores_disp = scores * scale
    sem_disp = None if sem is None else sem * scale
    nc_disp = None if noise_ceiling is None else (noise_ceiling[0] * scale, noise_ceiling[1] * scale)

    # Y-limits from the union of all displayed quantities.
    stack = [scores_disp]
    if sem_disp is not None:
        stack += [scores_disp + sem_disp, scores_disp - sem_disp]
    if nc_disp is not None:
        stack += [nc_disp[0], nc_disp[1]]
    y_lo, y_hi = _axis_limits(np.concatenate(stack), style)
    axis.set_ylim(y_lo, y_hi)

    axis.axhline(style.zero_line, **CHANCE_STYLE)
    axis.axvline(0.0, color="black", linewidth=0.75, alpha=0.45)

    if nc_disp is not None:
        axis.fill_between(times_ms, nc_disp[0], nc_disp[1],
                          facecolor="#bfbfbf", edgecolor="#7f7f7f",
                          alpha=0.30, linewidth=0.4, hatch="///", zorder=1)
    if sem_disp is not None:
        axis.fill_between(times_ms, scores_disp - sem_disp, scores_disp + sem_disp,
                          color=style.color, alpha=SEM_ALPHA, linewidth=0)

    axis.plot(times_ms, scores_disp, color=style.color, linewidth=LINE_WIDTH, zorder=3)

    # Significance (cluster first; pointwise fallback only when clusters empty/missing).
    sig_drawn = False
    if stats is not None and _stats_has_clusters(stats):
        sig_drawn = _draw_cluster_dots(axis, times_ms, stats, style.color)
    if not sig_drawn and null_distribution is not None and null_distribution.size:
        sig_drawn = _draw_pointwise_dots(axis, times_ms, scores_disp, null_distribution * scale, style.color)

    if draw_stim_bar:
        _draw_stimulus_bar(axis, config.plotting.stimulus_window_ms)

    if speech_onset_ms is not None and np.isfinite(speech_onset_ms):
        _draw_speech_onset(axis, float(speech_onset_ms))  # last: needs final ylim

    return sig_drawn


def _render_overlay_panel(
    axis,
    times_ms: np.ndarray,
    items: list[dict],
    *,
    style: _CurveStyle,
    config: AnalysisConfig,
    speech_onset_ms: float | None = None,
    draw_stim_bar: bool = True,
) -> bool:
    scale = 100.0 if style.as_percentage else 1.0
    stack: list[np.ndarray] = []
    for item in items:
        disp = item["scores"] * scale
        stack.append(disp)
        if item.get("sem") is not None:
            sd = item["sem"] * scale
            stack.append(disp + sd)
            stack.append(disp - sd)
    y_lo, y_hi = _axis_limits(np.concatenate(stack), style)
    axis.set_ylim(y_lo, y_hi)

    axis.axhline(style.zero_line, **CHANCE_STYLE)
    axis.axvline(0.0, color="black", linewidth=0.75, alpha=0.45)

    sig_any = False
    for item in items:
        disp = item["scores"] * scale
        color = item["color"]
        if item.get("sem") is not None:
            sd = item["sem"] * scale
            axis.fill_between(times_ms, disp - sd, disp + sd, color=color, alpha=SEM_ALPHA * 0.8, linewidth=0)
        axis.plot(times_ms, disp, color=color, linewidth=LINE_WIDTH, label=item["label"])
        stat = item.get("stats")
        if stat is not None and _stats_has_clusters(stat):
            if _draw_cluster_dots(axis, times_ms, stat, color):
                sig_any = True
        null_distribution = item.get("null_distribution")
        if (stat is None or not _stats_has_clusters(stat)) and null_distribution is not None and null_distribution.size:
            if _draw_pointwise_dots(axis, times_ms, disp, null_distribution * scale, color):
                sig_any = True

    if draw_stim_bar:
        _draw_stimulus_bar(axis, config.plotting.stimulus_window_ms)

    if speech_onset_ms is not None and np.isfinite(speech_onset_ms):
        _draw_speech_onset(axis, float(speech_onset_ms))

    return sig_any


def _one_panel_figure(
    result,
    style: _CurveStyle,
    config: AnalysisConfig,
    title: str,
    save_path: Path,
    sem: np.ndarray | None = None,
    stats: StatisticalTestResult | None = None,
    noise_ceiling: tuple[np.ndarray, np.ndarray] | None = None,
    speech_onset_ms: float | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    sig_drawn = _render_panel(
        axis=axis, times_ms=result.times_s * 1000.0, scores=result.scores,
        sem=sem, stats=stats,
        null_distribution=getattr(result, "null_distribution", None),
        noise_ceiling=noise_ceiling, style=style, config=config,
        speech_onset_ms=speech_onset_ms,
    )
    _apply_axis_chrome(axis, title=title, x_label=True, show_legend=True,
                       style=style, stats_drew_dots=sig_drawn)
    _save_figure(figure, save_path, config.plotting.save_formats)


# --------------------------------------------------------------------------
# Axis chrome / helpers
# --------------------------------------------------------------------------


def _apply_axis_chrome(
    axis,
    *,
    title: str | None,
    x_label: bool,
    show_legend: bool,
    style: _CurveStyle,
    stats_drew_dots: bool | None = None,
    overlay: bool = False,
) -> None:
    if title is not None:
        axis.set_title(title)
    if x_label:
        axis.set_xlabel("Time (ms)")
    axis.set_ylabel(style.ylabel)
    if style.as_percentage:
        axis.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    axis.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    axis.spines["left"].set_linewidth(1.1)
    axis.spines["bottom"].set_linewidth(1.1)
    axis.tick_params(axis="both", which="major", width=1.0, length=5.0, labelsize=8.5)
    axis.tick_params(axis="both", which="minor", width=0.8, length=2.5)

    if not show_legend:
        return

    if overlay:
        axis.legend(loc="upper left", fontsize=7.5, ncol=min(len(axis.get_lines()), 3))
        return

    handles = [
        Line2D([0], [0], color=style.color, lw=LINE_WIDTH, label=style.label),
        Rectangle((0, 0), 1, 1, facecolor=style.color, alpha=SEM_ALPHA, edgecolor="none",
                  label="Standard error"),
    ]
    if stats_drew_dots:
        handles.append(Line2D(
            [0], [0], marker="o", linestyle="", markerfacecolor=style.color,
            markeredgecolor=style.color, markersize=4.5, label="Decoding higher than chance (p < .05)",
        ))
    handles.extend([
        Line2D([0], [0], color=CHANCE_STYLE["color"], linestyle=CHANCE_STYLE["linestyle"],
               lw=CHANCE_STYLE["linewidth"], label="Chance"),
        Rectangle((0, 0), 1, 1, facecolor=STIMULUS_COLOR, edgecolor=STIMULUS_EDGE_COLOR,
                  label="Stimulus on"),
    ])
    axis.legend(handles=handles, loc="upper left", fontsize=7.5)


def _overlay_style(mode: str, chance_level: float) -> _CurveStyle:
    is_decoding = mode == "decoding"
    return _CurveStyle(
        color="#444444",
        ylabel="Accuracy (%)" if is_decoding else "Spearman rho",
        as_percentage=is_decoding,
        zero_line=chance_level * 100.0 if is_decoding else 0.0,
    )


def _fold_sem(result) -> np.ndarray | None:
    fold_scores = getattr(result, "fold_scores", None)
    if fold_scores is None or fold_scores.size == 0:
        return None
    return fold_scores.std(axis=0) / np.sqrt(len(fold_scores))


def _stats_has_clusters(stats: StatisticalTestResult | None) -> bool:
    if stats is None:
        return False
    masks = getattr(stats, "cluster_masks", None)
    return bool(masks)


def _axis_limits(values: np.ndarray, style: _CurveStyle) -> tuple[float, float]:
    finite = np.nan_to_num(values, nan=style.zero_line, posinf=style.zero_line, neginf=style.zero_line)
    vmin, vmax = float(finite.min()), float(finite.max())
    span_above = max(vmax - style.zero_line, 0.0)
    span_below = max(style.zero_line - vmin, 0.0)
    span = max(span_above, span_below)
    if style.as_percentage:
        # Decoding: ensure ±6 pp around chance; 25% headroom above the max excursion.
        pad = max(span * 0.25, 2.0)
        y_lo = min(style.zero_line - 6.0, vmin - pad)
        y_hi = max(style.zero_line + 6.0, vmax + pad)
    else:
        # RSA: data-driven. No hard floor; small effects stay legible.
        pad = max(span * 0.25, 0.003)
        y_lo = min(style.zero_line - pad, vmin - pad)
        y_hi = max(style.zero_line + pad, vmax + pad)
    return y_lo, y_hi


def _draw_prestim_shade(axis, config: AnalysisConfig) -> None:
    start_s, end_s = config.preprocessing.baseline_window_s
    start_ms = max(float(start_s) * 1000.0, axis.get_xlim()[0])
    end_ms = min(float(end_s) * 1000.0, axis.get_xlim()[1])
    if start_ms >= end_ms:
        return
    axis.axvspan(start_ms, end_ms, color=PRESTIM_COLOR, alpha=0.35, linewidth=0, zorder=0)


def _draw_stimulus_bar(axis, window_ms: tuple[float, float]) -> None:
    """Bottom stimulus-on bar in the Grootswagers-style decoding panel."""
    start, end = window_ms
    y_lo, y_hi = axis.get_ylim()
    y_span = y_hi - y_lo
    rect = Rectangle(
        (start, y_lo + y_span * STIM_BAR_Y_FRAC),
        end - start,
        y_span * STIM_BAR_HEIGHT_FRAC,
        facecolor=STIMULUS_COLOR,
        edgecolor=STIMULUS_EDGE_COLOR,
        linewidth=0.8,
        clip_on=False,
        zorder=4,
    )
    axis.add_patch(rect)


def _draw_cluster_dots(axis, times_ms: np.ndarray, stats: StatisticalTestResult, color: str) -> bool:
    if not _stats_has_clusters(stats):
        return False
    p_values = (
        stats.cluster_p_values.tolist()
        if getattr(stats, "cluster_p_values", np.empty(0)).size
        else [np.nan] * len(stats.cluster_masks)
    )
    sig_idx: list[int] = []
    for cluster_indices, p in zip(stats.cluster_masks, p_values):
        if cluster_indices.size == 0:
            continue
        if np.isfinite(p) and p >= 0.05:
            continue
        sig_idx.extend(int(i) for i in cluster_indices)
    if not sig_idx:
        return False
    y_lo, y_hi = axis.get_ylim()
    dot_y = y_lo + (y_hi - y_lo) * SIG_DOT_Y_FRAC
    axis.scatter(times_ms[sig_idx], np.full(len(sig_idx), dot_y),
                 s=SIG_DOT_SIZE, color=color, clip_on=False, zorder=6)
    return True


def _draw_pointwise_dots(axis, times_ms: np.ndarray, scores_disp: np.ndarray,
                         null_disp: np.ndarray, color: str) -> bool:
    p_values = (1.0 + np.sum(null_disp >= scores_disp[None, :], axis=0)) / (1.0 + len(null_disp))
    mask = p_values < 0.05
    if not mask.any():
        return False
    y_lo, y_hi = axis.get_ylim()
    dot_y = y_lo + (y_hi - y_lo) * SIG_DOT_Y_FRAC
    axis.scatter(times_ms[mask], np.full(int(mask.sum()), dot_y),
                 s=SIG_DOT_SIZE, color=color, clip_on=False, zorder=6)
    return True


def _annotate_peak(axis, times_ms: np.ndarray, scores_disp: np.ndarray,
                   color: str, as_percentage: bool) -> None:
    peak_idx = int(np.nanargmax(scores_disp))
    peak_t = float(times_ms[peak_idx])
    peak_v = float(np.nan_to_num(scores_disp[peak_idx], nan=0.0))
    label = f"peak {peak_v:.1f}% | {peak_t:.0f} ms" if as_percentage else f"peak {peak_v:.3f} | {peak_t:.0f} ms"
    axis.plot([peak_t], [peak_v], marker="o", markersize=4.2, markerfacecolor="white",
              markeredgecolor=color, markeredgewidth=1.1, zorder=5, clip_on=False)
    xy = (-7, 8) if peak_t > times_ms[-1] * 0.7 else (7, 8)
    ha = "right" if peak_t > times_ms[-1] * 0.7 else "left"
    axis.annotate(label, xy=(peak_t, peak_v), xytext=xy, textcoords="offset points",
                  fontsize=7.5, color=color, ha=ha)


def _draw_speech_onset(axis, onset_ms: float) -> None:
    x_lo, x_hi = axis.get_xlim()
    if onset_ms < x_lo or onset_ms > x_hi:
        return
    axis.axvline(onset_ms, color="#555555", linewidth=0.8, alpha=0.65, linestyle=(0, (3, 2)))
    y_lo, y_hi = axis.get_ylim()
    axis.text(onset_ms, y_hi - (y_hi - y_lo) * 0.04,
              f"speech onset {onset_ms:.0f} ms",
              fontsize=7, color="#444444",
              ha="left" if onset_ms < x_hi * 0.7 else "right", va="top")


def _distinct_palette(n: int) -> list[str]:
    if n <= len(OKABE_ITO):
        return OKABE_ITO[:n]
    cmap = plt.get_cmap("tab20")
    return [mpl.colors.to_hex(cmap(i % cmap.N)) for i in range(n)]


def _contrast_color(contrast_name: str) -> str:
    return CONTRAST_COLORS.get(contrast_name, OKABE_ITO[0])


def _save_figure(figure, save_path: Path, formats: Sequence[str]) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        if fmt in seen:
            continue
        seen.add(fmt)
        figure.savefig(save_path.with_suffix(f".{fmt}"))
    plt.close(figure)


def _publication_formats(config: AnalysisConfig) -> tuple[str, ...]:
    formats = list(config.plotting.save_formats)
    if "svg" not in formats:
        formats.append("svg")
    return tuple(formats)


def figure_stem(text: str) -> str:
    return (
        text.replace(" ", "_").replace("|", "_").replace("/", "_").replace("\\", "_").lower()
    )


def _stem(text: str) -> str:
    return figure_stem(text)
