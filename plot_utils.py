"""Centralised plotting module.

All figures produced by the training + evaluation pipeline are generated
here. Every plotting function applies the shared theme via
`_apply_plot_theme()` so the output has a consistent look-and-feel
(white background, black text, subtle grid lines).
"""
from __future__ import annotations

import os
import textwrap
from typing import Sequence

import numpy
import pandas
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from training_utils import (
    ensure_directory,
    best_auc_index,
    best_sign_consistency_index,
    knee_point_index,
)


# ---------------------------------------------------------------------------
# Shared theme -- applied at the start of every plotting function
# ---------------------------------------------------------------------------

def _apply_plot_theme() -> None:
    """Set a clean white-background, black-text theme for all plots.

    Combines the seaborn "whitegrid" paper theme with an explicit
    matplotlib rcParams override so background, axis edges and text
    remain consistent even inside notebook backends that override
    default rcParams.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "savefig.facecolor": "white",
        "text.color":       "black",
        "axes.labelcolor":  "black",
        "xtick.color":      "black",
        "ytick.color":      "black",
        "axes.edgecolor":   "black",
        "grid.color":       "#cccccc",
    })


# ---------------------------------------------------------------------------
# GA convergence plots
# ---------------------------------------------------------------------------

def plot_single_objective_convergence(stats: list[dict], filepath: str) -> None:
    ensure_directory(os.path.dirname(filepath))
    gens: list[int] = [s["gen"] for s in stats]
    max_vals: list[float] = [s["max"] for s in stats]
    avg_vals: list[float] = [s["avg"] for s in stats]

    _apply_plot_theme()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, max_vals, label="Max", linewidth=2)
    ax.plot(gens, avg_vals, label="Avg", linewidth=2, linestyle="--")
    if "min" in stats[0]:
        ax.fill_between(gens,
                        [s["min"] for s in stats],
                        max_vals,
                        alpha=0.15, label="Min-Max range")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (AUC)")
    ax.set_title("Single-Objective GA Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_multi_objective_convergence(stats: list[dict], filepath: str) -> None:
    ensure_directory(os.path.dirname(filepath))
    gens: list[int] = [s["gen"] for s in stats]

    _apply_plot_theme()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # AUC convergence
    axes[0].plot(gens, [s["auc_max"] for s in stats], label="Max", linewidth=2)
    axes[0].plot(gens, [s["auc_mean"] for s in stats], label="Mean", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("AUC")
    axes[0].set_title("AUC Convergence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Sign consistency convergence
    axes[1].plot(gens, [s["sign_max"] for s in stats], label="Max", linewidth=2)
    axes[1].plot(gens, [s["sign_mean"] for s in stats], label="Mean", linewidth=2, linestyle="--")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Sign Consistency")
    axes[1].set_title("Sign Consistency Convergence")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Pareto front size
    axes[2].plot(gens, [s["pareto_size"] for s in stats], linewidth=2, color="green")
    axes[2].set_xlabel("Generation")
    axes[2].set_ylabel("Pareto Front Size")
    axes[2].set_title("Pareto Front Size")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_pareto_front(pareto_individuals: list, filepath: str) -> None:
    """Plot the Pareto front and highlight the three canonical
    model-selection candidates: the knee point (balanced trade-off), the
    best sign-consistency point, and the best AUC point. All other Pareto
    solutions are shown in a muted neutral colour so the highlighted
    markers stand out.
    """
    ensure_directory(os.path.dirname(filepath))
    auc_vals: numpy.ndarray = numpy.array(
        [ind.fitness.values[0] for ind in pareto_individuals], dtype=float)
    sign_vals: numpy.ndarray = numpy.array(
        [ind.fitness.values[1] for ind in pareto_individuals], dtype=float)

    knee_idx: int = knee_point_index(pareto_individuals)
    best_sign_idx: int = best_sign_consistency_index(pareto_individuals)
    best_auc_idx: int = best_auc_index(pareto_individuals)

    # Filter background scatter by FITNESS COORDINATE so that any Pareto
    # individual whose (AUC, sign-consistency) tuple equals a highlighted
    # candidate's is suppressed -- prevents "ghost" background points
    # sitting exactly under the highlight markers when the GA population
    # has converged.
    highlight_coords: set[tuple[float, float]] = {
        (float(auc_vals[knee_idx]),      float(sign_vals[knee_idx])),
        (float(auc_vals[best_sign_idx]), float(sign_vals[best_sign_idx])),
        (float(auc_vals[best_auc_idx]),  float(sign_vals[best_auc_idx])),
    }
    other_mask: numpy.ndarray = numpy.array(
        [(float(auc_vals[i]), float(sign_vals[i])) not in highlight_coords
         for i in range(len(pareto_individuals))],
        dtype=bool,
    )

    _apply_plot_theme()
    fig, ax = plt.subplots(figsize=(8, 6))

    if other_mask.any():
        ax.scatter(auc_vals[other_mask], sign_vals[other_mask],
                   c="royalblue", alpha=0.75, edgecolors="black", s=55,
                   label="Other Pareto points", zorder=2)

    if best_auc_idx != knee_idx and best_auc_idx != best_sign_idx:
        ax.scatter(auc_vals[best_auc_idx], sign_vals[best_auc_idx],
                   c="tab:green", edgecolors="black", linewidths=1.0,
                   s=120, marker="^", zorder=4,
                   label=f"Best AUC ({auc_vals[best_auc_idx]:.4f}, "
                         f"{sign_vals[best_auc_idx]:.4f})")

    if best_sign_idx != knee_idx:
        ax.scatter(auc_vals[best_sign_idx], sign_vals[best_sign_idx],
                   c="tab:orange", edgecolors="black", linewidths=1.0,
                   s=100, marker="D", zorder=5,
                   label=f"Best sign-consistency ({auc_vals[best_sign_idx]:.4f}, "
                         f"{sign_vals[best_sign_idx]:.4f})")

    ax.scatter(auc_vals[knee_idx], sign_vals[knee_idx],
               c="tab:cyan", edgecolors="black", linewidths=1.0,
               s=120, marker="*", zorder=6,
               label=f"Knee point ({auc_vals[knee_idx]:.4f}, "
                     f"{sign_vals[knee_idx]:.4f})")

    ax.set_xlabel("AUC", fontweight="bold")
    ax.set_ylabel("Sign Consistency", fontweight="bold")
    ax.set_title("Pareto Front (AUC vs Sign Consistency)",
                 fontweight="bold", pad=10)
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Robustness suite (robustness_evaluation.py)
# ---------------------------------------------------------------------------

def _method_handles(method_styles: dict[str, tuple[str, str, str, str]]) -> list[Line2D]:
    """Legend entries for the methods: key -> (display name, colour, marker, line style)."""
    return [Line2D([0], [0], color=colour, marker=marker, linestyle=linestyle, linewidth=2, label=label)
            for label, colour, marker, linestyle in method_styles.values()]


def _panel_grid(n_panels: int, max_columns: int, panel_width: float, panel_height: float):
    n_columns: int = max(1, min(n_panels, max_columns))
    n_rows: int = int(numpy.ceil(n_panels / n_columns)) if n_panels else 1
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(panel_width * n_columns, panel_height * n_rows),
                             squeeze=False)
    for ax in list(axes.flat)[n_panels:]:
        ax.set_visible(False)
    return fig, axes


def plot_population_shift_curves(curves: pandas.DataFrame,
                                 panels: Sequence[tuple[str, str, str]],
                                 ess_levels: Sequence[float],
                                 method_styles: dict[str, tuple[str, str, str, str]],
                                 out_path: str,
                                 title: str) -> None:
    """Test ROC-AUC of every method along each population-shift axis, one panel per axis.

    Parameters
    ----------
    curves
        Columns `axis`, `position`, `method`, `mean`, `sd`. Position 0 is the clean test set; +k / -k is
        the k-th training-ESS level (1 = the mildest) of a re-weighting towards the positive / negative
        end of the axis. `sd` (the SD across runs) is drawn as a band where it exists.
    panels
        (axis, what the negative end emphasises, what the positive end emphasises), in panel order.
    ess_levels
        The training-ESS levels, mildest first (tick labels).
    method_styles
        Method key -> (display name, colour, marker, line style).
    """
    ensure_directory(os.path.dirname(out_path))
    _apply_plot_theme()
    fig, axes = _panel_grid(len(panels), 2, 7.2, 4.6)
    k: int = len(ess_levels)
    labels: list[str] = ([f"{level:.0%}" for level in reversed(ess_levels)] + ["clean"]
                         + [f"{level:.0%}" for level in ess_levels])
    for ax, (axis_name, negative, positive) in zip(axes.flat, panels):
        part: pandas.DataFrame = curves[curves["axis"] == axis_name]
        for method, (label, colour, marker, linestyle) in method_styles.items():
            line: pandas.DataFrame = part[part["method"] == method].sort_values("position")
            if line.empty:
                continue
            ax.plot(line["position"], line["mean"], color=colour, marker=marker, linestyle=linestyle,
                    linewidth=2, label=label)
            if line["sd"].notna().any():
                sd: pandas.Series = line["sd"].fillna(0.0)
                ax.fill_between(line["position"], line["mean"] - sd, line["mean"] + sd, color=colour, alpha=0.15)
        ax.set_xticks(numpy.arange(-k, k + 1))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlim(-k - 0.3, k + 0.3)
        ax.set_title(axis_name, fontweight="bold")
        ax.set_xlabel(f"← {negative}        training ESS of the re-weighting        {positive} →",
                      fontsize=9)
        ax.set_ylabel("Test ROC-AUC", fontweight="bold")
        ax.grid(True, alpha=0.3)
    fig.legend(handles=_method_handles(method_styles), loc="lower center", ncol=len(method_styles), frameon=True)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dependence_shift_summary(curves: pandas.DataFrame,
                                  differences: pandas.DataFrame,
                                  levels: dict[str, Sequence[float]],
                                  method_styles: dict[str, tuple[str, str, str, str]],
                                  out_path: str,
                                  title: str) -> None:
    """Dependence shifts, one row per direction (weakened / strengthened pairs). Left: the change of test
    ROC-AUC, averaged over the pairs of `curves`, against the share of the pair's correlation removed /
    added (mean over runs; band = SD across runs). Right: for every pair, MORSE's change minus the SO-GA's
    change (each averaged over the runs) -- above zero, MORSE lost less on that pair; the share of pairs
    above zero is printed over each box.

    Parameters
    ----------
    curves
        Columns `direction` ("weaken" / "strengthen"), `level`, `method`, `mean`, `sd`.
    differences
        Columns `direction`, `level`, `unit`, `difference` (may be empty).
    levels
        The levels of each direction, mildest first.
    """
    ensure_directory(os.path.dirname(out_path))
    _apply_plot_theme()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    titles: dict[str, str] = {"weaken": "Pairs weakened", "strengthen": "Pairs strengthened"}
    xlabels: dict[str, str] = {"weaken": "Share of the pair's correlation removed (100% = uncorrelated)",
                               "strengthen": "Share of the pair's correlation added"}
    box_colours: dict[str, str] = {"weaken": "#6baed6", "strengthen": "#fd8d3c"}
    for row, direction in enumerate(("weaken", "strengthen")):
        direction_levels: list[float] = list(levels.get(direction, ()))
        rank: dict[float, int] = {level: i + 1 for i, level in enumerate(direction_levels)}
        labels: list[str] = [f"{100 * level:g}%" for level in direction_levels]
        ax = axes[row, 0]
        part: pandas.DataFrame = curves[curves["direction"] == direction] if not curves.empty else curves
        if part.empty:
            for panel in axes[row]:
                panel.text(0.5, 0.5, "too few pairs are usable\nat every level", transform=panel.transAxes,
                           ha="center", va="center", color="gray")
        for method, (label, colour, marker, linestyle) in method_styles.items():
            line: pandas.DataFrame = part[part["method"] == method] if not part.empty else part
            if line.empty:
                continue
            x: numpy.ndarray = numpy.r_[0, line["level"].map(rank).to_numpy(dtype=float)]
            order: numpy.ndarray = numpy.argsort(x)
            mean: numpy.ndarray = numpy.r_[0.0, line["mean"].to_numpy(dtype=float)][order]
            sd: numpy.ndarray = numpy.r_[0.0, line["sd"].fillna(0.0).to_numpy(dtype=float)][order]
            ax.plot(x[order], mean, color=colour, marker=marker, linestyle=linestyle, linewidth=2, label=label)
            if line["sd"].notna().any():
                ax.fill_between(x[order], mean - sd, mean + sd, color=colour, alpha=0.15)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(numpy.arange(len(direction_levels) + 1))
        ax.set_xticklabels(["clean"] + labels)
        ax.set_title(titles[direction], fontweight="bold")
        ax.set_xlabel(xlabels[direction], fontweight="bold")
        ax.set_ylabel("Change of test ROC-AUC\n(mean over the pairs)", fontweight="bold")
        ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        if not differences.empty:
            for level in direction_levels:
                values: numpy.ndarray = differences.loc[
                    (differences["direction"] == direction) & numpy.isclose(differences["level"], level),
                    "difference"].dropna().to_numpy(dtype=float)
                if values.size == 0:
                    continue
                boxes = ax.boxplot([values], positions=[rank[level]], widths=0.5, patch_artist=True,
                                   showfliers=True, flierprops={"markersize": 3})
                for patch in boxes["boxes"]:
                    patch.set_facecolor(box_colours[direction])
                    patch.set_alpha(0.8)
                ax.annotate(f"{(values > 0).mean():.0%}", xy=(rank[level], float(numpy.max(values))),
                            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(numpy.arange(1, len(direction_levels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.4, len(direction_levels) + 0.6)
        ax.set_title(f"{titles[direction]}: MORSE minus SO-GA, per pair", fontweight="bold")
        ax.set_xlabel(xlabels[direction], fontweight="bold")
        ax.set_ylabel("Difference of the change of ROC-AUC\n(> 0: MORSE loses less)", fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.legend(handles=_method_handles(method_styles), loc="lower center", ncol=len(method_styles), frameon=True)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_corruption_curves(curves: pandas.DataFrame,
                           families: Sequence[tuple[str, str, str]],
                           method_styles: dict[str, tuple[str, str, str, str]],
                           out_path: str,
                           title: str) -> None:
    """Test ROC-AUC of every method against the corruption level, one panel per corruption family.

    Parameters
    ----------
    curves
        Columns `family`, `level` (0 = the clean test set), `method`, `mean`, `sd`: the ROC-AUC averaged
        over the corruption bank, then over the runs; `sd` is the SD across runs (drawn as a band).
    families
        (family, panel title, x-axis label), in panel order.
    """
    ensure_directory(os.path.dirname(out_path))
    _apply_plot_theme()
    fig, axes = _panel_grid(len(families), 2, 7.2, 4.6)
    for ax, (family, panel_title, xlabel) in zip(axes.flat, families):
        part: pandas.DataFrame = curves[curves["family"] == family]
        for method, (label, colour, marker, linestyle) in method_styles.items():
            line: pandas.DataFrame = part[part["method"] == method].sort_values("level")
            if line.empty:
                continue
            ax.plot(line["level"], line["mean"], color=colour, marker=marker, linestyle=linestyle,
                    linewidth=2, label=label)
            if line["sd"].notna().any():
                sd: pandas.Series = line["sd"].fillna(0.0)
                ax.fill_between(line["level"], line["mean"] - sd, line["mean"] + sd, color=colour, alpha=0.15)
        ax.set_title(panel_title, fontweight="bold")
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel("Test ROC-AUC", fontweight="bold")
        ax.grid(True, alpha=0.3)
    fig.legend(handles=_method_handles(method_styles), loc="lower center", ncol=len(method_styles), frameon=True)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_robustness_overview(overview: pandas.DataFrame,
                             family_labels: dict[str, str],
                             method_styles: dict[str, tuple[str, str, str, str]],
                             out_path: str,
                             title: str) -> None:
    """Clean vs stressed test ROC-AUC of every method, per stress family (x-axis; the tick label names
    the statistic, e.g. the worst scenario or the mean over the corruption bank). A second panel with
    the worst usable scenario is added when `worst_mean` is given. Hollow marker = clean, filled marker
    = stressed; error bars = SD across runs.

    Parameters
    ----------
    overview
        Columns `family`, `method`, `clean_mean`, `clean_sd`, `stressed_mean`, `stressed_sd`,
        `worst_mean`, `worst_sd` (NaN where not applicable).
    """
    ensure_directory(os.path.dirname(out_path))
    _apply_plot_theme()
    families: list[str] = list(dict.fromkeys(overview["family"]))
    worst_families: list[str] = [f for f in families
                                 if overview.loc[overview["family"] == f, "worst_mean"].notna().any()]
    panels: list[tuple[list[str], str, str, str]] = [
        (families, "stressed_mean", "stressed_sd", "Stressed = the statistic named under each family")]
    if worst_families:
        panels.append((worst_families, "worst_mean", "worst_sd", "Worst usable scenario (re-weighting)"))
    widths: list[float] = [max(4.0, 1.3 * len(p[0]) + 1.5) for p in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(sum(widths), 5.6), squeeze=False,
                             gridspec_kw={"width_ratios": widths})
    methods: list[str] = [m for m in method_styles if m in set(overview["method"])]
    step: float = 0.8 / max(len(methods), 1)
    for ax, (panel_families, column, sd_column, subtitle) in zip(axes.flat, panels):
        for j, method in enumerate(methods):
            label, colour, marker, _ = method_styles[method]
            for i, family in enumerate(panel_families):
                row: pandas.DataFrame = overview[(overview["family"] == family) & (overview["method"] == method)]
                if row.empty or numpy.isnan(row[column].iloc[0]):
                    continue
                x: float = i - 0.4 + step * (j + 0.5)
                clean_value: float = float(row["clean_mean"].iloc[0])
                stressed_value: float = float(row[column].iloc[0])
                ax.plot([x, x], [clean_value, stressed_value], color=colour, linewidth=1.5)
                ax.errorbar(x, clean_value, yerr=numpy.nan_to_num(row["clean_sd"].iloc[0]), fmt=marker,
                            markerfacecolor="white", markeredgecolor=colour, ecolor=colour, markersize=7, capsize=2)
                ax.errorbar(x, stressed_value, yerr=numpy.nan_to_num(row[sd_column].iloc[0]), fmt=marker,
                            color=colour, markersize=7, capsize=2)
        ax.set_xticks(numpy.arange(len(panel_families)))
        ax.set_xticklabels([textwrap.fill(family_labels.get(f, f), 16) for f in panel_families], fontsize=8)
        ax.set_ylabel("Test ROC-AUC", fontweight="bold")
        ax.set_title(subtitle, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
    fig.legend(handles=_method_handles({m: method_styles[m] for m in methods}), loc="lower center",
               ncol=len(methods), frameon=True)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sign_vs_degradation(points: pandas.DataFrame,
                             correlations: pandas.DataFrame,
                             family_labels: dict[str, str],
                             method_styles: dict[str, tuple[str, str, str, str]],
                             out_path: str,
                             title: str) -> None:
    """Sign consistency S of every GA model against its change of test ROC-AUC, one panel per stress
    family; the marker size grows with the number of inputs K. The Spearman correlation and the partial
    one given K are printed in every panel. Exploratory: a correlation does not show that S causes the
    difference.

    Parameters
    ----------
    points
        Columns `family`, `method`, `seed`, `sign_consistency`, `n_features`, `delta_roc_auc`.
    correlations
        Columns `family`, `n_models`, `spearman`, `partial_spearman_given_size`.
    """
    ensure_directory(os.path.dirname(out_path))
    _apply_plot_theme()
    families: list[str] = list(dict.fromkeys(points["family"]))
    fig, axes = _panel_grid(len(families), 4, 4.6, 4.2)
    scale: float = 160.0 / max(float(points["n_features"].max()), 1.0)
    for ax, family in zip(axes.flat, families):
        part: pandas.DataFrame = points[points["family"] == family]
        for method, (label, colour, marker, _) in method_styles.items():
            sub: pandas.DataFrame = part[part["method"] == method]
            ax.scatter(sub["sign_consistency"], sub["delta_roc_auc"], s=15 + scale * sub["n_features"],
                       c=colour, marker=marker, alpha=0.7, edgecolors="black", linewidths=0.4, label=label)
        row: pandas.DataFrame = correlations[correlations["family"] == family]
        if not row.empty:
            ax.text(0.02, 0.02, f"Spearman {row['spearman'].iloc[0]:+.2f}\n"
                                f"given K {row['partial_spearman_given_size'].iloc[0]:+.2f}",
                    transform=ax.transAxes, fontsize=8, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(textwrap.fill(family_labels.get(family, family), 30), fontsize=10, fontweight="bold")
        ax.set_xlabel("Sign consistency S")
        ax.set_ylabel("Change of test ROC-AUC")
        ax.grid(True, alpha=0.3)
    handles: list[Line2D] = [Line2D([0], [0], color=colour, marker=marker, linestyle="", markersize=8, label=label)
                             for label, colour, marker, _ in method_styles.values()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=True)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Legacy stress grid (legacy_stress_evaluation.py): noise x covariate shift, binary re-draw noise
# ---------------------------------------------------------------------------

def plot_noise_comparison(agg_df: pandas.DataFrame,
                          model_styles: dict[str, tuple[str, str, str, str]],
                          xlabel: str,
                          ylabel: str,
                          title: str,
                          out_path: str) -> None:
    """Line plot of mean AUC vs. a 1D noise level for multiple models with
    shaded +/- 1 std bands.

    Parameters
    ----------
    agg_df
        Multi-index-column DataFrame produced by
        `.groupby(level_col).agg(["mean", "std"])`. Expected to have
        `(f"auc_{key}", "mean")` and `(f"auc_{key}", "std")` columns for
        every `key` in `model_styles`.
    model_styles
        Dict mapping model key -> (display name, colour, marker, linestyle).
    """
    ensure_directory(os.path.dirname(out_path))
    _apply_plot_theme()
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, (name, colour, marker, ls) in model_styles.items():
        mean_series: pandas.Series = agg_df[(f"auc_{key}", "mean")]
        std_series:  pandas.Series = agg_df[(f"auc_{key}", "std")].fillna(0.0)
        x_vals: numpy.ndarray = mean_series.index.values
        m_vals: numpy.ndarray = mean_series.values
        s_vals: numpy.ndarray = std_series.values
        ax.plot(x_vals, m_vals, label=name,
                color=colour, marker=marker, linestyle=ls, linewidth=2)
        ax.fill_between(x_vals, m_vals - s_vals, m_vals + s_vals,
                        color=colour, alpha=0.15)
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_2d_heatmap_grid(heatmap_agg: pandas.DataFrame,
                         model_styles: dict[str, tuple[str, str, str, str]],
                         cbar_label: str,
                         xlabel: str,
                         ylabel: str,
                         title: str,
                         out_path: str,
                         aurs_scores: dict[str, float] | None = None) -> None:
    """Render a 2x2 grid of heatmaps -- one panel per model -- sharing the
    same vmin/vmax so the panels are directly visually comparable.

    Parameters
    ----------
    heatmap_agg
        A pandas DataFrame with a two-level MultiIndex whose LEVELS are
        (noise_level, mean_shift) and whose columns include one
        `auc_{key}` for every `key` in `model_styles`.
    model_styles
        Same shape as in `plot_noise_comparison`; only the display name
        (index 0) is used for the panel title -- colour/marker/linestyle
        are irrelevant for the heatmap and get ignored.
    aurs_scores
        Optional dict mapping each `model_styles` key to its AURS score
        (see `evaluation_utils.compute_aurs`) -- the fraction of that
        model's own clean-test score retained on average across this whole
        grid. When given, each panel's title gets a second line showing it
        as a percentage. `None` (the default) omits the subtitle.
    """
    ensure_directory(os.path.dirname(out_path))
    _apply_plot_theme()

    def _pivot(model_key: str) -> pandas.DataFrame:
        col_name: str = f"auc_{model_key}"
        m: pandas.DataFrame = heatmap_agg[[col_name]].reset_index()
        pivot: pandas.DataFrame = m.pivot(
            index="mean_shift", columns="noise_level", values=col_name)
        # Order rows so the most-positive shift is on top.
        return pivot.sort_index(ascending=False)

    pivots: dict[str, pandas.DataFrame] = {
        key: _pivot(key) for key in model_styles.keys()
    }
    shared_vmin: float = min(float(p.values.min()) for p in pivots.values())
    shared_vmax: float = max(float(p.values.max()) for p in pivots.values())

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    for ax, (key, pivot) in zip(axes.flat, pivots.items()):
        display_name: str = model_styles[key][0]
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="viridis",
            vmin=shared_vmin, vmax=shared_vmax,
            cbar_kws={"label": cbar_label},
            annot=False,
            square=False,
        )
        panel_title: str = display_name
        if aurs_scores is not None:
            panel_title += f"\nAURS = {aurs_scores[key]:.1%}"
        ax.set_title(panel_title, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")

    fig.suptitle(title, fontweight="bold", fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Balanced sensitivity / specificity curve for the deployed model
# ---------------------------------------------------------------------------

def plot_specificity_sensitivity_curve(sorted_scores: numpy.ndarray,
                                       sensitivity_curve: numpy.ndarray,
                                       specificity_curve: numpy.ndarray,
                                       intersection_idx: int,
                                       out_path: str) -> None:
    """Render the sensitivity and specificity curves as functions of the
    classification threshold rank, and mark the balanced (Sens ≈ Spec)
    threshold with a square marker + legend annotation.

    Consumes the output of `evaluation_utils.find_balanced_threshold`. Saves
    a PDF for high-quality inclusion in the paper.
    """
    ensure_directory(os.path.dirname(out_path))
    _apply_plot_theme()

    n: int = len(sorted_scores)
    crossover_value: float = float(sorted_scores[intersection_idx])
    crossover_sens: float = float(sensitivity_curve[intersection_idx])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(range(1, n + 1), specificity_curve,
            label="Specificity", color="green", linestyle="-")
    ax.plot(range(1, n + 1), sensitivity_curve,
            label="Sensitivity", color="blue", linestyle="--")
    ax.scatter(intersection_idx + 1, crossover_sens,
               color="green", marker="s", zorder=5,
               label=(f"Balanced Sens/Spec = {crossover_sens:.4f}\n"
                      f"Threshold = {crossover_value:.4f}"))
    ax.set_xlabel("Threshold rank (low -> high)")
    ax.set_ylabel("Sensitivity / Specificity")
    ax.set_ylim(0, 1)
    ax.set_xlim(1, n)
    ax.set_xticks([1, n])
    ax.set_xticklabels(["Lowest threshold", "Highest threshold"])
    ax.set_title("Specificity and Sensitivity curve", fontweight="bold", pad=10)
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Model-parsimony plot
# ---------------------------------------------------------------------------

def plot_feature_count_comparison(feature_count_df: pandas.DataFrame,
                                  method_order: Sequence[str],
                                  palette: dict[str, str],
                                  n_features_total: int,
                                  n_seeds: int,
                                  out_path: str) -> None:
    """Boxplot + stripplot overlay comparing per-seed feature counts across
    methods. Each box is annotated with its median value for at-a-glance
    readability.

    Parameters
    ----------
    feature_count_df
        Long-format frame with columns {"method", "seed", "n_features"}.
    method_order
        Explicit x-axis order for the methods.
    palette
        Dict mapping method name -> matplotlib colour string.
    n_features_total
        Total number of candidate features (goes into the y-axis label).
    n_seeds
        Number of seeds contributing to the distribution (goes into the
        title).
    """
    _apply_plot_theme()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=feature_count_df,
        x="method", y="n_features",
        order=list(method_order), hue="method", palette=palette,
        legend=False, width=0.5, ax=ax,
    )
    sns.stripplot(
        data=feature_count_df,
        x="method", y="n_features",
        order=list(method_order),
        color="black", size=4, alpha=0.6, jitter=0.15, ax=ax,
    )

    for i, method in enumerate(method_order):
        med: float = feature_count_df.loc[
            feature_count_df["method"] == method, "n_features"
        ].median()
        ax.annotate(
            f"median={int(med)}",
            xy=(i, med), xytext=(0, 14),
            textcoords="offset points", ha="center",
            fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
        )

    ax.set_xlabel("Method", fontweight="bold")
    ax.set_ylabel(f"Number of selected features (out of {n_features_total})",
                  fontweight="bold")
    ax.set_title(
        f"Feature Count per Method Across {n_seeds} Seeds\n"
        f"(box = IQR with median line, dots = individual seeds)",
        fontweight="bold", pad=10,
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sign consistency of the final models
# ---------------------------------------------------------------------------

def plot_sign_consistency_comparison(sign_df: pandas.DataFrame,
                                     method_order: Sequence[str],
                                     palette: dict[str, str],
                                     n_seeds: int,
                                     out_path: str) -> None:
    """Boxplot + stripplot overlay comparing the sign consistency of the final
    (full-training-set refit) model across methods, one dot per seed; each box
    is annotated with its median.

    Parameters
    ----------
    sign_df
        Long-format frame with columns {"method", "seed", "sign_consistency"}
        (see `evaluation_utils.compute_model_sign_consistency`).
    method_order
        Explicit x-axis order for the methods.
    palette
        Dict mapping method name -> matplotlib colour string.
    n_seeds
        Number of seeds contributing to the distribution (goes into the title).
    """
    ensure_directory(os.path.dirname(out_path))
    _apply_plot_theme()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=sign_df,
        x="method", y="sign_consistency",
        order=list(method_order), hue="method", palette=palette,
        legend=False, width=0.5, ax=ax,
    )
    sns.stripplot(
        data=sign_df,
        x="method", y="sign_consistency",
        order=list(method_order),
        color="black", size=4, alpha=0.6, jitter=0.15, ax=ax,
    )

    for i, method in enumerate(method_order):
        med: float = sign_df.loc[
            sign_df["method"] == method, "sign_consistency"
        ].median()
        ax.annotate(
            f"median={med:.3f}",
            xy=(i, med), xytext=(0, 14),
            textcoords="offset points", ha="center",
            fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
        )

    ax.set_xlabel("Method", fontweight="bold")
    ax.set_ylabel("Sign consistency of the final model\n(fraction of selected features)",
                  fontweight="bold")
    ax.set_title(
        f"Sign Consistency of the Final Refit Model per Method Across {n_seeds} Seeds\n"
        f"(box = IQR with median line, dots = individual seeds)",
        fontweight="bold", pad=10,
    )
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
