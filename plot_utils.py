"""Centralised plotting module.

All figures produced by the training + evaluation pipeline are generated
here. Every plotting function applies the shared theme via
`_apply_plot_theme()` so the output has a consistent look-and-feel
(white background, black text, subtle grid lines).
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy
import pandas
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
# All-models noise-robustness plots
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
