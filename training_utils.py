"""Small, dependency-light utilities used by the GA training classes and
the notebook.
"""
import csv
import os

import numpy


def ensure_directory(directory: str) -> None:
    """Create a directory (and any missing parents) if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def save_stats_csv(stats: list[dict], filepath: str) -> None:
    """Persist a list of homogeneous dict rows as a CSV file. Used by the
    single- and multi-objective GA training classes to save per-generation
    statistics.
    """
    if not stats:
        return
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, "w", newline="") as f:
        writer: csv.DictWriter = csv.DictWriter(f, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)


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
