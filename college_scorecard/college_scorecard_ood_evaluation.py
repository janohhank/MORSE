"""Out-of-domain evaluation of a College Scorecard run: TableShift's held-out Carnegie classes.

college_scorecard_data_preparation.ipynb writes three files. The training file and the in-domain (ID)
test file hold institutions of the 25 Carnegie classes that TableShift treats as in-domain; the
out-of-domain (OOD) file holds all 945 institutions of the 8 classes that TableShift holds out
(seminaries, schools of art and design, master's universities with larger programmes, some associate's
colleges, ...). training_notebook.ipynb never looks at the OOD file. This script takes the models of a
finished, checkpointed run -- nothing is retrained -- and scores them on both test files. That is a
real, not synthetic, distribution shift: whole types of institutions the models have never seen.

What is evaluated
    * The final models of every seed, rebuilt exactly as the run built them from the checkpointed
      masks (refit on the whole training file): the SO-GA best individual, SFS, all features, and three
      positions on every MORSE front -- the max-f1 end, the knee point and the max-S end. "MORSE" in the
      comparisons is the position the run itself used (USE_KNEE_POINT_SELECTION of its notebook).
    * Every distinct solution of every MORSE front, to see whether sign consistency goes together with
      out-of-domain performance within a front.

Metrics
    ROC-AUC is the primary metric of the ID -> OOD comparison. It does not depend on the class
    prevalence, which differs a lot between the two test files (0.644 ID, 0.411 OOD), so its drop
    measures the effect of the shift itself.
    PR-AUC (average precision) is the metric the PR runs optimised, but its level depends on the
    prevalence (a random ranking scores the prevalence), so a raw ID -> OOD difference mixes the shift
    with the change of the class prior. It is therefore also reported *calibrated* to the ID prevalence
    (Siblini et al., "Master your metrics with calibration", IDA 2020): the OOD institutions are
    re-weighted class-wise so that the positives carry the ID share of the total weight. Every precision
    then becomes the precision at the ID prevalence, while every recall stays the same.
    The sign consistency of every model is that of the refitted model (evaluation_utils.
    compute_model_sign_consistency, marginal correlations of the training file).

Statistics
    MORSE against every baseline: paired two-sided Wilcoxon signed-rank tests over the seeds (for the
    deterministic SFS and all-features models the one-sample test against their constant value), and a
    bootstrap over the institutions -- the ID and the OOD file resampled independently -- of the
    seed-averaged difference, for the OOD ROC-AUC and for the ID -> OOD drop of the ROC-AUC.

Checks before anything is scored
    * the training file must match the run's checkpoint fingerprint (feature names and the hashes of the
      standardised training matrix and of the labels), so the models are refit on exactly the data the
      run was trained on;
    * the recomputed ID test scores must reproduce the run's own per-seed results
      (evaluation/all_models_comparison/gaussian_2d_per_seed.csv), when the run folder has them.

Usage (from any working directory):
    python college_scorecard/college_scorecard_ood_evaluation.py
    python college_scorecard/college_scorecard_ood_evaluation.py --run 2026-09-25_14-06-20_college_scorecard_pr
Without --run, the newest run folder in the repository root whose checkpoints were trained on the
College Scorecard training file is used.

Outputs (default folder: <run>/evaluation/ood_test/):
    ood_per_seed.csv          every seed x model: size, sign consistency, ID and OOD metrics
    ood_summary.csv           mean and standard deviation per model
    ood_tests.csv             MORSE against every baseline: Wilcoxon tests and bootstrap intervals
    ood_front_solutions.csv   every distinct Pareto solution: CV fitness, size, ID and OOD metrics
    ood_evaluation.png/.pdf   the figure
    ood_report.txt            the printed report
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
import warnings
from typing import Any, Callable

import numpy
import pandas
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# The pipeline modules live in the repository root, one level above this folder.
REPOSITORY_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.model_selection import StratifiedKFold  # noqa: E402

from checkpoint_utils import (FINGERPRINT_FILE, SeedTrainingResult, TrainingCheckpointStore,  # noqa: E402
                              array_fingerprint, read_json)
from evaluation_utils import (build_model_package, compute_marginal_correlations,  # noqa: E402
                              compute_model_sign_consistency, predict_scores)
from multi_objective_training import MultiObjectiveTraining  # noqa: E402
from run_manifest import read_run_manifest  # noqa: E402
from training_config import TrainingConfig  # noqa: E402
from training_utils import (best_auc_index, best_sign_consistency_index, ensure_directory,  # noqa: E402
                            knee_point_index)

DATA_DIRECTORY: str = os.path.join(REPOSITORY_ROOT, "college_scorecard")
DEFAULT_TRAIN_CSV: str = os.path.join(DATA_DIRECTORY, "college_scorecard_preprocessed_train_data.csv")
DEFAULT_ID_TEST_CSV: str = os.path.join(DATA_DIRECTORY, "college_scorecard_preprocessed_test_data.csv")
DEFAULT_OOD_CSV: str = os.path.join(DATA_DIRECTORY, "college_scorecard_preprocessed_ood_test_data.csv")
DEFAULT_TARGET: str = "label"

# The three positions on a MORSE front: key -> (label, index function of training_utils).
FRONT_POSITIONS: dict[str, tuple[str, Callable[[list], int]]] = {
    "max_f1": ("MORSE max-f1 end", best_auc_index),
    "knee": ("MORSE knee point", knee_point_index),
    "max_s": ("MORSE max-S end", best_sign_consistency_index),
}
BASELINES: dict[str, str] = {"soga": "SO-GA", "sfs": "SFS", "all": "All features"}
MODEL_ORDER: list[str] = ["max_f1", "knee", "max_s", "soga", "sfs", "all"]
LABELS: dict[str, str] = {**{key: label for key, (label, _) in FRONT_POSITIONS.items()}, **BASELINES}
# Colour, marker (the notebook's colours for MORSE / SO-GA / SFS / all features).
STYLE: dict[str, tuple[str, str]] = {
    "max_f1": ("#9ecae1", "o"), "knee": ("#4292c6", "s"), "max_s": ("#08519c", "v"),
    "soga": ("tab:orange", "X"), "sfs": ("tab:red", "D"), "all": ("tab:green", "^")}

# Metrics compared between MORSE and the baselines: column -> description. For the two drops a
# NEGATIVE difference (MORSE - baseline) means that MORSE loses less.
TESTED_METRICS: dict[str, str] = {
    "ood_roc_auc": "OOD ROC-AUC",
    "roc_auc_drop": "ROC-AUC drop (ID - OOD)",
    "ood_pr_auc_calibrated": "OOD PR-AUC, calibrated to the ID prevalence",
    "pr_auc_calibrated_drop": "PR-AUC drop (ID - calibrated OOD)",
}


# ---------------------------------------------------------------------------
# Data and run discovery
# ---------------------------------------------------------------------------

def load_split(path: str, target: str,
               feature_names: list[str] | None = None) -> tuple[pandas.DataFrame, numpy.ndarray]:
    """Features and 0/1 labels of one preprocessed CSV. With `feature_names` the columns are checked
    against them and put in their order (the order the run's masks refer to)."""
    frame: pandas.DataFrame = pandas.read_csv(path)
    if target not in frame.columns:
        raise ValueError(f"{path}: there is no target column {target!r}")
    y: numpy.ndarray = frame[target].to_numpy()
    if not set(numpy.unique(y)) <= {0, 1}:
        raise ValueError(f"{path}: the target must be 0/1, found {sorted(numpy.unique(y))[:5]}")
    X: pandas.DataFrame = frame.drop(columns=[target])
    if feature_names is not None:
        missing: list[str] = [name for name in feature_names if name not in X.columns]
        if missing:
            raise ValueError(f"{path}: {len(missing)} input(s) of the run are missing, e.g. {missing[:3]}")
        X = X[feature_names]
    return X, y.astype(int)


def training_data_fingerprint(X_train: pandas.DataFrame, y_train: numpy.ndarray) -> dict[str, Any]:
    """The `features` and `data` parts of a checkpoint fingerprint, recomputed for a training file
    exactly as training_notebook.ipynb computes them: the feature names, the float64 training matrix
    AFTER the notebook's StandardScaler, and the float64 labels
    (checkpoint_utils.build_training_fingerprint)."""
    names: list[str] = list(X_train.columns)
    X_search: numpy.ndarray = numpy.ascontiguousarray(X_train.to_numpy(), dtype=numpy.float64)
    y_search: numpy.ndarray = numpy.ascontiguousarray(y_train, dtype=numpy.float64)
    return {
        "features": {"count": len(names),
                     "sha256": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()},
        "data": {"X_train": array_fingerprint(StandardScaler().fit_transform(X_search)),
                 "y_train": array_fingerprint(y_search)},
    }


def fingerprint_differences(saved: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """What differs between a run's saved fingerprint and the recomputed data fingerprint ([] = the
    run was trained on this training file). Only the data and the feature names are compared: the GA
    settings and the code do not matter for re-scoring the saved models."""
    settings: dict[str, Any] = saved.get("settings", {})
    differences: list[str] = []
    if settings.get("features") != current["features"]:
        differences.append("the input variables (names or order) differ")
    for part in ("X_train", "y_train"):
        if settings.get("data", {}).get(part) != current["data"][part]:
            differences.append(f"the training data ({part}) differ")
    return differences


def find_matching_runs(current: dict[str, Any]) -> list[str]:
    """Run folders in the repository root whose checkpoints were trained on this training file,
    oldest first (the folder names start with a timestamp)."""
    runs: list[str] = []
    for name in sorted(os.listdir(REPOSITORY_ROOT)):
        path: str = os.path.join(REPOSITORY_ROOT, name, "checkpoints", "training", FINGERPRINT_FILE)
        if os.path.isfile(path) and not fingerprint_differences(read_json(path), current):
            runs.append(os.path.join(REPOSITORY_ROOT, name))
    return runs


def run_selection_rule(run_directory: str) -> str | None:
    """The front-selection rule of a run -- "knee" or "max_s" -- from the run manifest (run_manifest.py),
    or else from USE_KNEE_POINT_SELECTION in the copy of training_notebook.ipynb archived in the run
    folder; None if neither can be read."""
    manifest: dict[str, Any] | None = read_run_manifest(run_directory)
    if manifest is not None and manifest.get("settings", {}).get("pareto_rule") in ("knee", "max_s"):
        return manifest["settings"]["pareto_rule"]
    path: str = os.path.join(run_directory, "training_notebook.ipynb")
    if not os.path.isfile(path):
        return None
    for cell in read_json(path).get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        match = re.search(r"^\s*USE_KNEE_POINT_SELECTION\s*(?::\s*bool)?\s*=\s*(True|False)",
                          "".join(cell.get("source", [])), re.MULTILINE)
        if match:
            return "knee" if match.group(1) == "True" else "max_s"
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calibration_weights(y: numpy.ndarray, reference_prevalence: float) -> numpy.ndarray:
    """Class-wise weights under which the positives carry `reference_prevalence` of the total weight.
    Weighted average precision is then the calibrated PR-AUC of Siblini et al. (2020): every precision
    becomes the precision at the reference prevalence, every recall is unchanged."""
    prevalence: float = float(numpy.mean(y))
    return numpy.where(y == 1, reference_prevalence / prevalence,
                       (1.0 - reference_prevalence) / (1.0 - prevalence))


def score(y: numpy.ndarray, probabilities: numpy.ndarray, reference_prevalence: float) -> dict[str, float]:
    """ROC-AUC, raw PR-AUC and PR-AUC calibrated to `reference_prevalence` of one prediction vector."""
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "pr_auc_calibrated": float(average_precision_score(
            y, probabilities, sample_weight=calibration_weights(y, reference_prevalence))),
    }


def roc_auc_rows(y: numpy.ndarray, scores: numpy.ndarray) -> numpy.ndarray:
    """ROC-AUC of every row of `scores` (models x observations) at once, through the Mann-Whitney
    statistic with average ranks for ties -- the same value as sklearn's roc_auc_score, fast enough for
    thousands of bootstrap samples."""
    ranks: numpy.ndarray = rankdata(scores, axis=1)
    positives: numpy.ndarray = y == 1
    n_positive: int = int(positives.sum())
    n_negative: int = y.size - n_positive
    return (ranks[:, positives].sum(axis=1) - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def partial_spearman(x: numpy.ndarray, y: numpy.ndarray, z: numpy.ndarray) -> float:
    """Spearman correlation of x and y after removing the linear effect of the ranks of z from both."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    design: numpy.ndarray = numpy.column_stack([numpy.ones_like(rz), rz])
    ex: numpy.ndarray = rx - design @ numpy.linalg.lstsq(design, rx, rcond=None)[0]
    ey: numpy.ndarray = ry - design @ numpy.linalg.lstsq(design, ry, rcond=None)[0]
    # undefined when x or y is constant or fully explained by z; the residuals are then ~1e-16 rounding
    # noise rather than exactly zero, so compare with a tolerance (ranks are of the order of n)
    if ex.std() < 1e-9 or ey.std() < 1e-9:
        return float("nan")
    return float(numpy.corrcoef(ex, ey)[0, 1])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def seed_masks(result: SeedTrainingResult) -> dict[str, list[int]]:
    """The feature masks of one seed: the three MORSE front positions and the three baselines."""
    front: list = result.pareto_front
    masks: dict[str, list[int]] = {
        key: list(front[index_of(front)]) for key, (_, index_of) in FRONT_POSITIONS.items()}
    masks["soga"] = list(result.single_best)
    masks["sfs"] = list(result.forward_mask)
    masks["all"] = list(result.all_mask)
    return masks


def evaluate_models(results: dict[int, SeedTrainingResult], features: list[str],
                    X_train: pandas.DataFrame, y_train: numpy.ndarray, marginal_corr: pandas.Series,
                    X_id: pandas.DataFrame, y_id: numpy.ndarray,
                    X_ood: pandas.DataFrame, y_ood: numpy.ndarray
                    ) -> tuple[pandas.DataFrame, dict[str, dict[str, numpy.ndarray]]]:
    """Score the six models of every seed on both test files.

    Returns the per-seed table and the predictions {model: {"id": seeds x n_id, "ood": seeds x n_ood}}
    for the bootstrap. Every model is refit on the whole training file with the seed as random_state,
    exactly as the notebook's evaluation does (evaluation_utils.build_model_package)."""
    reference_prevalence: float = float(numpy.mean(y_id))
    rows: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, list[numpy.ndarray]]] = {
        key: {"id": [], "ood": []} for key in MODEL_ORDER}
    for seed, result in results.items():
        for key, mask in seed_masks(result).items():
            package: dict[str, Any] = build_model_package(mask, features, X_train, y_train, seed=seed)
            p_id: numpy.ndarray = predict_scores(package, X_id)
            p_ood: numpy.ndarray = predict_scores(package, X_ood)
            predictions[key]["id"].append(p_id)
            predictions[key]["ood"].append(p_ood)
            id_scores: dict[str, float] = score(y_id, p_id, reference_prevalence)
            ood_scores: dict[str, float] = score(y_ood, p_ood, reference_prevalence)
            sign: dict[str, Any] = compute_model_sign_consistency(package, marginal_corr)
            rows.append({
                "seed": seed, "model": key, "label": LABELS[key],
                "n_features": sign["n_features"], "sign_consistency": sign["sign_consistency"],
                "n_inconsistent": sign["n_inconsistent"],
                "id_roc_auc": id_scores["roc_auc"], "ood_roc_auc": ood_scores["roc_auc"],
                "roc_auc_drop": id_scores["roc_auc"] - ood_scores["roc_auc"],
                "id_pr_auc": id_scores["pr_auc"], "ood_pr_auc": ood_scores["pr_auc"],
                "ood_pr_auc_calibrated": ood_scores["pr_auc_calibrated"],
                "pr_auc_calibrated_drop": id_scores["pr_auc"] - ood_scores["pr_auc_calibrated"],
            })
    stacked = {key: {part: numpy.vstack(values) for part, values in parts.items()}
               for key, parts in predictions.items()}
    return pandas.DataFrame(rows), stacked


def cv_evaluator(fingerprint: dict[str, Any], X_train: pandas.DataFrame, y_train: numpy.ndarray,
                 features: list[str]) -> MultiObjectiveTraining:
    """The run's own fitness function (MultiObjectiveTraining.evaluate_multi) on the run's data and folds:
    the training matrix standardised as in the notebook, the CV splitter of the checkpoint fingerprint."""
    cv_settings: dict[str, Any] = fingerprint["settings"]["cv"]
    cv = StratifiedKFold(n_splits=cv_settings["n_splits"], shuffle=cv_settings["shuffle"],
                         random_state=cv_settings["random_state"])
    X_search: numpy.ndarray = StandardScaler().fit_transform(
        numpy.ascontiguousarray(X_train.to_numpy(), dtype=numpy.float64))
    y_search: numpy.ndarray = numpy.ascontiguousarray(y_train, dtype=numpy.float64)
    config = TrainingConfig(seed=0, use_roc_auc=bool(fingerprint["settings"]["training_config"]["use_roc_auc"]))
    return MultiObjectiveTraining(config, features, X_search, y_search, cv)


def evaluate_front_solutions(results: dict[int, SeedTrainingResult], features: list[str],
                             X_train: pandas.DataFrame, y_train: numpy.ndarray,
                             marginal_corr: pandas.Series, evaluator: MultiObjectiveTraining,
                             X_id: pandas.DataFrame, y_id: numpy.ndarray,
                             X_ood: pandas.DataFrame, y_ood: numpy.ndarray) -> pandas.DataFrame:
    """Every distinct solution of every seed's MORSE front, with its CV fitness and test scores.

    The CV values are re-evaluated with the run's fitness function rather than read from the checkpoint:
    runs trained before the reference-sign fix of evaluation_utils.compute_marginal_correlations (2026-09-26)
    stored a sign consistency whose reference was reversed for 0/1 inputs with a prevalence above 0.5. The
    stored CV objective (AUC) is not affected by the fix and must be reproduced exactly."""
    reference_prevalence: float = float(numpy.mean(y_id))
    rows: list[dict[str, Any]] = []
    for seed, result in results.items():
        front: list = result.pareto_front
        position_masks: dict[str, tuple[int, ...]] = {
            key: tuple(front[index_of(front)]) for key, (_, index_of) in FRONT_POSITIONS.items()}
        seen: set[tuple[int, ...]] = set()
        for individual in front:
            mask: tuple[int, ...] = tuple(individual)
            if mask in seen:
                continue
            seen.add(mask)
            cv_objective, cv_sign_consistency = evaluator.evaluate_multi(list(mask))
            if abs(cv_objective - individual.fitness.values[0]) > 1e-12:
                raise RuntimeError("the run's stored CV objective is not reproduced: other data or folds")
            package: dict[str, Any] = build_model_package(list(mask), features, X_train, y_train, seed=seed)
            id_scores = score(y_id, predict_scores(package, X_id), reference_prevalence)
            ood_scores = score(y_ood, predict_scores(package, X_ood), reference_prevalence)
            rows.append({
                "seed": seed,
                "positions": "|".join(key for key, m in position_masks.items() if m == mask),
                "cv_objective": cv_objective,
                "cv_sign_consistency": cv_sign_consistency,
                "cv_sign_consistency_stored": individual.fitness.values[1],
                "n_features": len(package["features"]),
                "sign_consistency": compute_model_sign_consistency(package, marginal_corr)["sign_consistency"],
                "id_roc_auc": id_scores["roc_auc"], "ood_roc_auc": ood_scores["roc_auc"],
                "roc_auc_drop": id_scores["roc_auc"] - ood_scores["roc_auc"],
                "id_pr_auc": id_scores["pr_auc"],
                "ood_pr_auc_calibrated": ood_scores["pr_auc_calibrated"],
            })
    return pandas.DataFrame(rows)


def verify_against_run(run_directory: str, per_seed: pandas.DataFrame, run_rule: str | None,
                       use_roc_auc: bool) -> tuple[str, str | None]:
    """Compare the recomputed ID test scores with the run's own per-seed results.

    The run's MORSE column belongs to the front position of the run's rule; when that rule is unknown,
    both rules are tried. Raises if the scores do not agree -- then the models or the data are not the
    ones the run evaluated. Returns a message and the rule the run evaluated (None if not checked)."""
    path: str = os.path.join(run_directory, "evaluation", "all_models_comparison", "gaussian_2d_per_seed.csv")
    if not os.path.isfile(path):
        return "not checked: the run folder has no per-seed evaluation results", None
    grid: pandas.DataFrame = pandas.read_csv(path)
    clean: pandas.DataFrame = grid[numpy.isclose(grid["noise_level"], 0.0)
                                   & numpy.isclose(grid["mean_shift"], 0.0)].set_index("seed")
    metric: str = "id_roc_auc" if use_roc_auc else "id_pr_auc"

    def deviation(column: str, key: str) -> float:
        ours: pandas.Series = per_seed[per_seed["model"] == key].set_index("seed")[metric]
        return float((ours - clean.loc[ours.index, f"auc_{column}"]).abs().max())

    baseline_deviation: float = max(deviation("single", "soga"), deviation("forward", "sfs"),
                                    deviation("all", "all"))
    candidates: list[str] = [run_rule] if run_rule else ["max_s", "knee"]
    morse_deviation: dict[str, float] = {key: deviation("multi", key) for key in candidates}
    matched: list[str] = [key for key, value in morse_deviation.items() if value <= 1e-9]
    if baseline_deviation > 1e-9 or not matched:
        raise RuntimeError(
            f"the recomputed ID test scores do not reproduce the run's own results (baselines: largest "
            f"deviation {baseline_deviation:.3g}; MORSE {morse_deviation}); the models or the data files "
            f"are not the ones this run evaluated")
    largest: float = max(baseline_deviation, morse_deviation[matched[0]])
    return (f"reproduced (largest deviation {largest:.1e}; {len(clean)} seeds x 4 models, MORSE = "
            f"{LABELS[matched[0]]})"), matched[0]


def display_path(path: str) -> str:
    """A path relative to the repository root when possible (not across drives on Windows)."""
    try:
        return os.path.relpath(path, REPOSITORY_ROOT)
    except ValueError:
        return os.path.abspath(path)


def paired_tests(per_seed: pandas.DataFrame, morse_key: str) -> pandas.DataFrame:
    """Wilcoxon signed-rank tests of MORSE against every baseline over the seeds."""
    reference: pandas.DataFrame = per_seed[per_seed["model"] == morse_key].set_index("seed").sort_index()
    rows: list[dict[str, Any]] = []
    for other in BASELINES:
        baseline: pandas.DataFrame = per_seed[per_seed["model"] == other].set_index("seed").sort_index()
        for metric, description in TESTED_METRICS.items():
            difference: pandas.Series = reference[metric] - baseline[metric]
            p_value: float = float("nan") if numpy.allclose(difference, 0.0) else float(wilcoxon(difference).pvalue)
            rows.append({
                "baseline": BASELINES[other], "metric": metric, "description": description,
                "morse_mean": reference[metric].mean(), "baseline_mean": baseline[metric].mean(),
                "mean_difference": difference.mean(), "wilcoxon_p": p_value,
                "morse_greater_in": int((difference > 0).sum()), "n_seeds": int(len(difference)),
            })
    return pandas.DataFrame(rows)


def bootstrap_roc(predictions: dict[str, dict[str, numpy.ndarray]], y_id: numpy.ndarray,
                  y_ood: numpy.ndarray, morse_key: str, n_bootstrap: int,
                  random_state: int) -> pandas.DataFrame:
    """Percentile intervals of the seed-averaged differences MORSE - baseline of the OOD ROC-AUC and
    of the ROC-AUC drop, resampling the institutions of the ID and the OOD file independently."""
    if n_bootstrap <= 0:
        return pandas.DataFrame()
    keys: list[str] = [morse_key, *BASELINES]
    rng: numpy.random.Generator = numpy.random.default_rng(random_state)
    level: dict[str, list[float]] = {key: [] for key in BASELINES}
    drop: dict[str, list[float]] = {key: [] for key in BASELINES}
    for _ in range(n_bootstrap):
        i_id: numpy.ndarray = rng.integers(0, y_id.size, y_id.size)
        i_ood: numpy.ndarray = rng.integers(0, y_ood.size, y_ood.size)
        if numpy.ptp(y_id[i_id]) == 0 or numpy.ptp(y_ood[i_ood]) == 0:
            continue  # a resample with a single class has no ROC-AUC
        auc_id: dict[str, float] = {key: float(roc_auc_rows(y_id[i_id], predictions[key]["id"][:, i_id]).mean())
                                    for key in keys}
        auc_ood: dict[str, float] = {key: float(roc_auc_rows(y_ood[i_ood], predictions[key]["ood"][:, i_ood]).mean())
                                     for key in keys}
        for other in BASELINES:
            level[other].append(auc_ood[morse_key] - auc_ood[other])
            drop[other].append((auc_id[morse_key] - auc_ood[morse_key]) - (auc_id[other] - auc_ood[other]))
    rows: list[dict[str, Any]] = []
    for other in BASELINES:
        for metric, values in (("ood_roc_auc", level[other]), ("roc_auc_drop", drop[other])):
            array: numpy.ndarray = numpy.asarray(values)
            rows.append({"baseline": BASELINES[other], "metric": metric,
                         "bootstrap_ci_low": float(numpy.percentile(array, 2.5)),
                         "bootstrap_ci_high": float(numpy.percentile(array, 97.5)),
                         "n_bootstrap": int(array.size)})
    return pandas.DataFrame(rows)


def front_correlations(fronts: pandas.DataFrame) -> dict[str, numpy.ndarray]:
    """Per seed, Spearman correlations between sign consistency and OOD performance within the front."""
    out: dict[str, list[float]] = {
        "cv_s_vs_ood_roc": [], "cv_s_vs_ood_roc_given_k": [], "cv_s_vs_roc_drop": [], "final_s_vs_ood_roc": []}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a constant column gives nan, which is reported as such
        for _, group in fronts.groupby("seed"):
            out["cv_s_vs_ood_roc"].append(spearmanr(group["cv_sign_consistency"], group["ood_roc_auc"]).statistic)
            out["cv_s_vs_ood_roc_given_k"].append(partial_spearman(
                group["cv_sign_consistency"].to_numpy(), group["ood_roc_auc"].to_numpy(),
                group["n_features"].to_numpy(dtype=float)))
            out["cv_s_vs_roc_drop"].append(spearmanr(group["cv_sign_consistency"], group["roc_auc_drop"]).statistic)
            out["final_s_vs_ood_roc"].append(spearmanr(group["sign_consistency"], group["ood_roc_auc"]).statistic)
    return {key: numpy.asarray(values, dtype=float) for key, values in out.items()}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _dumbbell(ax: plt.Axes, per_seed: pandas.DataFrame, keys: list[str], id_column: str, ood_column: str,
              ylabel: str, title: str) -> None:
    """Mean ID and OOD value of every model (+-1 SD over the seeds), joined by a line."""
    offsets: numpy.ndarray = numpy.linspace(-0.12, 0.12, len(keys))
    for offset, key in zip(offsets, keys):
        colour, marker = STYLE[key]
        rows: pandas.DataFrame = per_seed[per_seed["model"] == key]
        means = [rows[id_column].mean(), rows[ood_column].mean()]
        sds = [rows[id_column].std(ddof=1), rows[ood_column].std(ddof=1)]
        ax.errorbar([offset, 1 + offset], means, yerr=sds, color=colour, marker=marker, markersize=7,
                    linewidth=1.6, capsize=3, label=LABELS[key])
    ax.set_xticks([0, 1], ["in-domain test", "out-of-domain test"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)


def plot_results(per_seed: pandas.DataFrame, fronts: pandas.DataFrame, morse_key: str, run_name: str,
                 id_prevalence: float, ood_prevalence: float, out_base: str) -> None:
    """Four panels: ID vs OOD ROC-AUC, ID vs OOD calibrated PR-AUC, the per-seed ROC-AUC drops, and the
    OOD ROC-AUC of every Pareto solution against its sign consistency."""
    main_keys: list[str] = [morse_key, *BASELINES]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))

    _dumbbell(axes[0, 0], per_seed, main_keys, "id_roc_auc", "ood_roc_auc", "ROC-AUC",
              "(a) ROC-AUC, in-domain vs out-of-domain")
    _dumbbell(axes[0, 1], per_seed, main_keys, "id_pr_auc", "ood_pr_auc_calibrated", "PR-AUC",
              f"(b) PR-AUC, OOD calibrated to the ID prevalence {id_prevalence:.3f}\n"
              f"(raw OOD prevalence {ood_prevalence:.3f})")
    axes[0, 0].legend(loc="lower left", fontsize=9)

    ax = axes[1, 0]
    jitter = numpy.random.default_rng(0)
    for position, key in enumerate(MODEL_ORDER):
        colour, marker = STYLE[key]
        drops: numpy.ndarray = per_seed[per_seed["model"] == key]["roc_auc_drop"].to_numpy()
        ax.scatter(position + jitter.uniform(-0.15, 0.15, drops.size), drops, color=colour, marker=marker,
                   s=22, alpha=0.75, edgecolors="none")
        ax.hlines(drops.mean(), position - 0.3, position + 0.3, color="black", linewidth=2)
    ax.set_xticks(range(len(MODEL_ORDER)),
                  [LABELS[key].replace("MORSE ", "MORSE\n") + (" *" if key == morse_key else "")
                   for key in MODEL_ORDER], fontsize=8.5)
    ax.set_ylabel("ROC-AUC drop, in-domain minus out-of-domain")
    ax.set_title("(c) Loss under the shift per seed (bar = mean; * = run's MORSE rule)")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    ax.scatter(fronts["sign_consistency"], fronts["ood_roc_auc"], s=6, color="0.6", alpha=0.45,
               edgecolors="none", label="Pareto solutions (all seeds)")
    for key in MODEL_ORDER:
        colour, marker = STYLE[key]
        rows = per_seed[per_seed["model"] == key]
        ax.scatter(rows["sign_consistency"], rows["ood_roc_auc"], color=colour, marker=marker,
                   s=40 if key in BASELINES else 30, edgecolors="black", linewidths=0.4,
                   label=LABELS[key], zorder=3)
    ax.set_xlabel("sign consistency S of the refitted model")
    ax.set_ylabel("out-of-domain ROC-AUC")
    ax.set_title("(d) Out-of-domain ROC-AUC vs sign consistency")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"College Scorecard: in-domain vs out-of-domain (TableShift held-out Carnegie classes)\n"
                 f"run {run_name}; {per_seed['seed'].nunique()} seeds; models refit from the checkpoints")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_base + ".png", dpi=200)
    fig.savefig(out_base + ".pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="run folder (absolute, or relative to the repository root); "
                                      "default: the newest run trained on the College Scorecard training file")
    parser.add_argument("--selection", choices=("auto", "max_s", "knee"), default="auto",
                        help="the MORSE front position compared with the baselines; auto = the run's own rule")
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--id-test-csv", default=DEFAULT_ID_TEST_CSV)
    parser.add_argument("--ood-csv", default=DEFAULT_OOD_CSV)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap samples (0 = no bootstrap)")
    parser.add_argument("--random-state", type=int, default=0, help="seed of the bootstrap")
    parser.add_argument("--out", help="output folder; default: <run>/evaluation/ood_test")
    return parser.parse_args(argv)


def resolve_run(argument: str | None, train_csv: str, current: dict[str, Any]) -> str:
    """The run folder to evaluate: --run if given (it must have been trained on the training file),
    otherwise the newest run folder in the repository root whose checkpoints match the training file."""
    if argument:
        run_directory: str = os.path.abspath(os.path.join(REPOSITORY_ROOT, argument))
        fingerprint_path: str = os.path.join(run_directory, "checkpoints", "training", FINGERPRINT_FILE)
        if not os.path.isfile(fingerprint_path):
            raise FileNotFoundError(f"{run_directory} has no training checkpoints ({fingerprint_path})")
        differences: list[str] = fingerprint_differences(read_json(fingerprint_path), current)
        if differences:
            raise ValueError(f"{run_directory} was not trained on {train_csv}: " + "; ".join(differences))
        return run_directory
    candidates: list[str] = find_matching_runs(current)
    if not candidates:
        raise FileNotFoundError(f"no run folder in {REPOSITORY_ROOT} has checkpoints trained on {train_csv}; "
                                f"train one or pass --run")
    if len(candidates) > 1:
        print("runs trained on this data: " + ", ".join(os.path.basename(c) for c in candidates)
              + " -> using the newest; choose another one with --run", flush=True)
    return candidates[-1]


def main(argv: list[str] | None = None) -> None:
    arguments: argparse.Namespace = parse_arguments(argv)
    started: float = time.time()
    report: list[str] = []

    def say(line: str = "") -> None:
        print(line, flush=True)
        report.append(line)

    # ---- data, and the run that was trained on it
    X_train, y_train = load_split(arguments.train_csv, arguments.target)
    run_directory: str = resolve_run(arguments.run, arguments.train_csv,
                                     training_data_fingerprint(X_train, y_train))
    run_name: str = os.path.basename(os.path.normpath(run_directory))
    fingerprint: dict[str, Any] = read_json(os.path.join(run_directory, "checkpoints", "training", FINGERPRINT_FILE))
    use_roc_auc: bool = bool(fingerprint["settings"]["training_config"]["use_roc_auc"])
    features: list[str] = list(X_train.columns)
    X_id, y_id = load_split(arguments.id_test_csv, arguments.target, features)
    X_ood, y_ood = load_split(arguments.ood_csv, arguments.target, features)

    store = TrainingCheckpointStore(os.path.join(run_directory, "checkpoints", "training"), fingerprint)
    seeds: list[int] = store.completed_seeds()
    if not seeds:
        raise FileNotFoundError(f"{run_directory} has no completed seed checkpoint")
    results: dict[int, SeedTrainingResult] = {seed: store.load_seed(seed) for seed in seeds}

    # ---- the six models of every seed, and the check against the run's own results
    marginal_corr = pandas.Series(compute_marginal_correlations(X_train, y_train), index=X_train.columns)
    per_seed, predictions = evaluate_models(results, features, X_train, y_train, marginal_corr,
                                            X_id, y_id, X_ood, y_ood)
    notebook_rule: str | None = run_selection_rule(run_directory)
    check, verified_rule = verify_against_run(run_directory, per_seed, notebook_rule, use_roc_auc)
    for key in MODEL_ORDER:  # the fast ROC-AUC of the bootstrap must equal sklearn's
        exact: numpy.ndarray = per_seed[per_seed["model"] == key]["ood_roc_auc"].to_numpy()
        assert numpy.allclose(roc_auc_rows(y_ood, predictions[key]["ood"]), exact, atol=1e-12)

    run_rule: str | None = notebook_rule or verified_rule
    if arguments.selection != "auto":
        morse_key, rule_note = arguments.selection, "chosen with --selection"
    elif run_rule is not None:
        morse_key, rule_note = run_rule, "the run's own front-selection rule"
    else:
        morse_key, rule_note = "max_s", "default -- the run's own rule could not be determined"

    say("=" * 110)
    say("College Scorecard out-of-domain evaluation (TableShift's held-out Carnegie classes)")
    say("=" * 110)
    say(f"run:            {run_name}  ({len(seeds)} seeds {seeds[0]}..{seeds[-1]}; GA objective "
        f"{'ROC-AUC' if use_roc_auc else 'PR-AUC'})")
    say(f"training file:  {display_path(arguments.train_csv)}  ({len(y_train)} institutions, {len(features)} "
        f"inputs; matches the run's checkpoint fingerprint)")
    say(f"ID test file:   {display_path(arguments.id_test_csv)}  ({len(y_id)} institutions, "
        f"prevalence {y_id.mean():.3f})")
    say(f"OOD test file:  {display_path(arguments.ood_csv)}  ({len(y_ood)} institutions, "
        f"prevalence {y_ood.mean():.3f})")
    say(f"check of the recomputed ID test scores against the run's own results: {check}")
    say(f"MORSE model compared with the baselines: {LABELS[morse_key]} ({rule_note})")

    # ---- summary per model
    columns: list[str] = ["n_features", "sign_consistency", "id_roc_auc", "ood_roc_auc", "roc_auc_drop",
                          "id_pr_auc", "ood_pr_auc", "ood_pr_auc_calibrated", "pr_auc_calibrated_drop"]
    grouped: pandas.DataFrame = per_seed.groupby("model")[columns].agg(["mean", "std"]).reindex(MODEL_ORDER)
    summary: pandas.DataFrame = pandas.DataFrame(
        {f"{column}_{statistic}": grouped[(column, statistic)] for column in columns
         for statistic in ("mean", "std")})
    summary.insert(0, "label", [LABELS[key] for key in summary.index])
    summary.index.name = "model"

    say("\n-- Final models: mean over the seeds (* = the MORSE model compared below) --")
    say(f"{'model':20s} {'K':>6s} {'S':>6s} | {'ROC ID':>7s} {'ROC OOD':>8s} {'drop':>7s} | "
        f"{'PR ID':>6s} {'PR OOD':>7s} {'PR OODc':>8s} {'drop c':>7s} | {'SD of ROC OOD':>13s}")
    for key, row in summary.iterrows():
        label: str = row["label"] + (" *" if key == morse_key else "")
        say(f"{label:20s} {row['n_features_mean']:6.1f} {row['sign_consistency_mean']:6.3f} | "
            f"{row['id_roc_auc_mean']:7.4f} {row['ood_roc_auc_mean']:8.4f} {row['roc_auc_drop_mean']:7.4f} | "
            f"{row['id_pr_auc_mean']:6.4f} {row['ood_pr_auc_mean']:7.4f} {row['ood_pr_auc_calibrated_mean']:8.4f} "
            f"{row['pr_auc_calibrated_drop_mean']:7.4f} | {row['ood_roc_auc_std']:13.4f}")
    say(f"K = inputs, S = sign consistency of the refitted model, drop = ID - OOD. PR OOD is the raw OOD "
        f"PR-AUC (prevalence {y_ood.mean():.3f}, not comparable with PR ID); PR OODc is calibrated to the "
        f"ID prevalence {y_id.mean():.3f}; drop c = PR ID - PR OODc.")

    # ---- MORSE against the baselines
    tests: pandas.DataFrame = paired_tests(per_seed, morse_key)
    intervals: pandas.DataFrame = bootstrap_roc(predictions, y_id, y_ood, morse_key, arguments.bootstrap,
                                                arguments.random_state)
    if not intervals.empty:
        tests = tests.merge(intervals, on=["baseline", "metric"], how="left")
    say(f"\n-- {LABELS[morse_key]} against the baselines: difference = MORSE - baseline, averaged over the "
        f"seeds (for the drops a negative difference means that MORSE loses less) --")
    for _, row in tests.iterrows():
        interval: str = ""
        if pandas.notna(row.get("bootstrap_ci_low", numpy.nan)):
            interval = (f"; bootstrap 95% CI [{row['bootstrap_ci_low']:+.4f}, {row['bootstrap_ci_high']:+.4f}] "
                        f"({int(row['n_bootstrap'])} samples)")
        say(f"  vs {row['baseline']:12s} {row['description']:44s} {row['mean_difference']:+.4f}  "
            f"(Wilcoxon p = {row['wilcoxon_p']:.3g}, MORSE greater in {row['morse_greater_in']}/"
            f"{row['n_seeds']} seeds{interval})")

    # ---- every solution of every front
    fronts: pandas.DataFrame = evaluate_front_solutions(
        results, features, X_train, y_train, marginal_corr, cv_evaluator(fingerprint, X_train, y_train, features),
        X_id, y_id, X_ood, y_ood)
    correlations: dict[str, numpy.ndarray] = front_correlations(fronts)
    say(f"\n-- Within the MORSE fronts: {len(fronts)} distinct solutions ({len(fronts) / len(seeds):.1f} per "
        f"seed, {int(fronts['n_features'].min())}-{int(fronts['n_features'].max())} inputs). Spearman "
        f"correlation per seed: median [range], seeds with a positive value --")
    descriptions: dict[str, str] = {
        "cv_s_vs_ood_roc": "CV sign consistency vs OOD ROC-AUC",
        "cv_s_vs_ood_roc_given_k": "  ... with the number of inputs controlled for",
        "cv_s_vs_roc_drop": "CV sign consistency vs ROC-AUC drop",
        "final_s_vs_ood_roc": "sign consistency of the refit vs OOD ROC-AUC",
    }
    for key, description in descriptions.items():
        finite: numpy.ndarray = correlations[key][numpy.isfinite(correlations[key])]
        if finite.size == 0:
            say(f"  {description:48s} not defined")
            continue
        say(f"  {description:48s} {numpy.median(finite):+.3f} [{finite.min():+.3f}, {finite.max():+.3f}]  "
            f"positive in {(finite > 0).sum()}/{finite.size}")

    # ---- outputs
    out_directory: str = os.path.abspath(arguments.out or os.path.join(run_directory, "evaluation", "ood_test"))
    ensure_directory(out_directory)
    per_seed.to_csv(os.path.join(out_directory, "ood_per_seed.csv"), index=False)
    summary.to_csv(os.path.join(out_directory, "ood_summary.csv"))
    tests.to_csv(os.path.join(out_directory, "ood_tests.csv"), index=False)
    fronts.to_csv(os.path.join(out_directory, "ood_front_solutions.csv"), index=False)
    plot_results(per_seed, fronts, morse_key, run_name, float(y_id.mean()), float(y_ood.mean()),
                 os.path.join(out_directory, "ood_evaluation"))
    say(f"\nresults written to {out_directory} ({time.time() - started:.0f} s)")
    with open(os.path.join(out_directory, "ood_report.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
