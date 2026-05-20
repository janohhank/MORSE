import csv
import os

import numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_directory(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def _apply_plot_theme() -> None:
    """Set a clean white-background, black-text theme for all plots."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.edgecolor": "black",
        "grid.color": "#cccccc",
    })


def save_stats_csv(stats: list[dict], filepath: str) -> None:
    if not stats:
        return
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, "w", newline="") as f:
        writer: csv.DictWriter = csv.DictWriter(f, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)


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
    """Plot the Pareto front and highlight the three canonical model-selection
    candidates: the knee point (balanced trade-off), the best sign-consistency
    point, and the best AUC point. All other Pareto solutions are shown in a
    muted neutral colour so the highlighted markers stand out."""
    ensure_directory(os.path.dirname(filepath))
    auc_vals: numpy.ndarray = numpy.array(
        [ind.fitness.values[0] for ind in pareto_individuals], dtype=float)
    sign_vals: numpy.ndarray = numpy.array(
        [ind.fitness.values[1] for ind in pareto_individuals], dtype=float)

    # Resolve the three canonical selection indices. These helpers handle
    # edge cases (single-point fronts, coincident endpoints) themselves.
    knee_idx: int = knee_point_index(pareto_individuals)
    best_sign_idx: int = best_sign_consistency_index(pareto_individuals)
    best_auc_idx: int = best_auc_index(pareto_individuals)
    highlight_indices: set[int] = {knee_idx, best_sign_idx, best_auc_idx}

    other_mask: numpy.ndarray = numpy.array(
        [i not in highlight_indices for i in range(len(pareto_individuals))], dtype=bool)

    _apply_plot_theme()
    fig, ax = plt.subplots(figsize=(8, 6))

    # Background: all other Pareto solutions in a muted colour.
    if other_mask.any():
        ax.scatter(auc_vals[other_mask], sign_vals[other_mask],
                   c="lightsteelblue", alpha=0.75, edgecolors="black", s=55,
                   label=f"Other Pareto points",
                   zorder=2)

    # Highlight: best AUC (green triangle) -- only if distinct from the others.
    if best_auc_idx != knee_idx and best_auc_idx != best_sign_idx:
        ax.scatter(auc_vals[best_auc_idx], sign_vals[best_auc_idx],
                   c="tab:green", edgecolors="black", linewidths=1.0,
                   s=120, marker="^", zorder=4,
                   label=f"Best AUC ({auc_vals[best_auc_idx]:.4f}, "
                         f"{sign_vals[best_auc_idx]:.4f})")

    # Highlight: best sign-consistency (orange diamond) -- only if distinct from knee.
    if best_sign_idx != knee_idx:
        ax.scatter(auc_vals[best_sign_idx], sign_vals[best_sign_idx],
                   c="tab:orange", edgecolors="black", linewidths=1.0,
                   s=100, marker="D", zorder=5,
                   label=f"Best sign-consistency ({auc_vals[best_sign_idx]:.4f}, "
                         f"{sign_vals[best_sign_idx]:.4f})")

    # Highlight: knee point (red star) -- always drawn on top with the largest marker.
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
# Pareto front model selection
# Fitness convention (multi-objective GA):
#   ind.fitness.values[0] = AUC              (higher is better)
#   ind.fitness.values[1] = Sign consistency (higher is better)
# ---------------------------------------------------------------------------

def best_auc_index(pareto_front: list) -> int:
    """Return the index of the Pareto individual with the highest AUC."""
    if not pareto_front:
        raise ValueError("Pareto front is empty")
    return max(range(len(pareto_front)),
               key=lambda i: pareto_front[i].fitness.values[0])


def best_sign_consistency_index(pareto_front: list) -> int:
    """Return the index of the Pareto individual with the highest sign-consistency score."""
    if not pareto_front:
        raise ValueError("Pareto front is empty")
    return max(range(len(pareto_front)),
               key=lambda i: pareto_front[i].fitness.values[1])


def knee_point_index(pareto_front: list) -> int:
    """Return the index of the knee-point individual on the Pareto front.

    The knee point is the Pareto solution with the largest perpendicular
    distance from the straight line connecting the two extreme points
    (best-AUC and best-sign-consistency). Both objectives are min-max
    normalized to [0, 1] first so they are comparable. For a single-point
    front, index 0 is returned; if the two extremes coincide, the best-AUC
    index is returned.
    """
    n: int = len(pareto_front)
    if n == 0:
        raise ValueError("Pareto front is empty")
    if n == 1:
        return 0

    auc: numpy.ndarray = numpy.array(
        [ind.fitness.values[0] for ind in pareto_front], dtype=float)
    sign: numpy.ndarray = numpy.array(
        [ind.fitness.values[1] for ind in pareto_front], dtype=float)

    def _norm(v: numpy.ndarray) -> numpy.ndarray:
        rng: float = float(v.max() - v.min())
        if rng == 0.0:
            return numpy.zeros_like(v)
        return (v - v.min()) / rng

    auc_n: numpy.ndarray = _norm(auc)
    sign_n: numpy.ndarray = _norm(sign)

    p_auc_idx: int = int(numpy.argmax(auc_n))
    p_sign_idx: int = int(numpy.argmax(sign_n))
    if p_auc_idx == p_sign_idx:
        return p_auc_idx

    p1: numpy.ndarray = numpy.array([auc_n[p_auc_idx], sign_n[p_auc_idx]])
    p2: numpy.ndarray = numpy.array([auc_n[p_sign_idx], sign_n[p_sign_idx]])
    line_vec: numpy.ndarray = p2 - p1
    line_len: float = float(numpy.linalg.norm(line_vec))
    if line_len == 0.0:
        return p_auc_idx

    # Perpendicular distance from each (auc_n, sign_n) point to the line p1-p2
    points: numpy.ndarray = numpy.column_stack([auc_n, sign_n])
    rel: numpy.ndarray = points - p1
    cross: numpy.ndarray = rel[:, 0] * line_vec[1] - rel[:, 1] * line_vec[0]
    distances: numpy.ndarray = numpy.abs(cross) / line_len

    return int(numpy.argmax(distances))


def select_pareto_individual(pareto_front: list, use_knee_point: bool = True):
    """Pick one individual from a Pareto front using a consistent strategy.

    use_knee_point=True  -> knee-point (balanced trade-off via knee_point_index).
    use_knee_point=False -> best-sign-consistency (max fitness.values[1]).

    Centralising the choice here guarantees that every call site in the
    notebook (evaluation, stability, all-models comparison, ...) picks the
    *same* individual for a given Pareto front.
    """
    if use_knee_point:
        return pareto_front[knee_point_index(pareto_front)]
    return pareto_front[best_sign_consistency_index(pareto_front)]
