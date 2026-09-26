"""Robustness suite: the final models of a run under test-set stresses that are generated automatically.

Every final model of every seed -- MORSE (the run's Pareto selection rule), the SO-GA, SFS and the
all-features model -- is scored on the test set under two kinds of stress (robustness_utils,
docs/robustness.md):

  re-weighted test populations  population shifts (leading principal components, typical vs atypical
                                rows) and dependence shifts (pairs of correlated inputs made less or more
                                dependent); severity = the effective sample size left on the training rows
  corrupted test sets           measurement noise, recording noise of 0/1 inputs, under-recording, values
                                that are not available; one fixed bank of realisations for every model

No feature is named anywhere: the scenarios come from the training data by fixed rules
(robustness_config.RobustnessConfig), the same for every dataset. ROC-AUC is the primary metric (it does
not depend on the class prevalence); PR-AUC (average precision) is reported as well, under re-weighting
at the unshifted prevalence (robustness_utils.normalise_within_classes).

Statistics: the unit of replication is the GA run (seed). For every family and severity each model gets
one number -- the mean over the family's usable scenarios (re-weighting) or over the corruption bank
(corruption) -- and MORSE is compared with the SO-GA by a paired two-sided Wilcoxon signed-rank test over
the seeds, and with the deterministic SFS and all-features models by the one-sample test against their
single value.

Two ways to run it
  * in training_notebook.ipynb, right after training: `build_final_models` + `run_robustness_suite`;
  * later, on any checkpointed run, without retraining:
        python robustness_evaluation.py --run 2026-09-25_14-06-20_college_scorecard_pr
    The run's data are rebuilt by executing the configuration and data-loading cells of the notebook
    copy archived in the run folder, and checked against the run's checkpoint fingerprint.

Outputs (default: <run>/evaluation/robustness/)
  config.json                 the suite settings and the provenance of the evaluation
  schema.json                 the inferred feature schema
  models.csv                  every final model: size, sign consistency, clean ROC-AUC / PR-AUC
  scenarios.csv               every re-weighting scenario: strength, ESS (training / test / per class),
                              support, and what actually moved (statistic, correlation, means, SDs)
  reweighting_scores.csv      every model under every re-weighting scenario
  corruption_scores.csv       every model under every corruption level and repetition
  corruption_diagnostics.csv  what every corrupted test set changed (and what had to be undone)
  model_family_scores.csv     every model per family and severity (mean over scenarios / repetitions)
  summary.csv                 per family, severity and method: mean over runs, SDs, change, worst case
  tests.csv                   MORSE against every baseline, per family and severity
  sign_vs_degradation.csv     sign consistency vs degradation of the GA models (exploratory)
  report.txt                  the printed report
  robustness_*.png            five figures
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence

import numpy
import pandas
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from checkpoint_utils import (FINGERPRINT_FILE, TrainingCheckpointStore, array_fingerprint,
                              atomic_write_json, atomic_write_text, read_json, source_fingerprint)
from evaluation_utils import (build_model_package, compute_marginal_correlations,
                              compute_model_sign_consistency, predict_scores)
import plot_utils
import robustness_utils
from plot_utils import (plot_corruption_curves, plot_dependence_shift_summary, plot_population_shift_curves,
                        plot_robustness_overview, plot_sign_vs_degradation)
from robustness_config import RobustnessConfig
from robustness_utils import (CorruptionBank, DependenceTilt, FeatureSchema, calibrate_strengths,
                              class_effective_sizes, dependence_pairs, effective_sample_fraction,
                              exponential_tilt, infer_feature_schema, normalise_within_classes,
                              population_axes, weighted_average_precision, weighted_correlation,
                              weighted_roc_auc)
from training_utils import ensure_directory, repository_root, select_pareto_individual

METHODS: tuple[str, ...] = ("multi", "single", "forward", "all")
# Display name, colour, marker, line style -- the notebook's colours for the four methods.
METHOD_STYLES: dict[str, tuple[str, str, str, str]] = {
    "multi":   ("MORSE", "tab:blue", "o", "-"),
    "single":  ("SO-GA", "tab:orange", "s", "--"),
    "forward": ("SFS", "tab:red", "D", ":"),
    "all":     ("All features", "tab:green", "^", "-."),
}
REWEIGHTING_FAMILIES: tuple[str, ...] = ("population", "dependence_decorrelate", "dependence_strengthen")
FAMILY_LABELS: dict[str, str] = {
    "population": "Population shift",
    "dependence_decorrelate": "Dependence: decorrelated pairs",
    "dependence_strengthen": "Dependence: strengthened pairs",
    "gaussian_noise": "Measurement noise",
    "binary_redraw": "Recording noise (0/1 re-drawn)",
    "under_recording": "Under-recording (1 lost)",
    "value_masking": "Values not available",
}
FAMILY_ORDER: tuple[str, ...] = REWEIGHTING_FAMILIES + robustness_utils.CORRUPTION_FAMILIES
CORRUPTION_LEVEL_LABELS: dict[str, str] = {
    "gaussian_noise": "noise SD (fraction of the training SD)",
    "binary_redraw": "probability that a 0/1 input is re-drawn",
    "under_recording": "probability that a recorded 1 is lost",
    "value_masking": "probability that an available value is withheld",
}


# ---------------------------------------------------------------------------
# The final models
# ---------------------------------------------------------------------------

def build_final_models(
        feature_names: list[str],
        X_train: pandas.DataFrame,
        y_train: pandas.Series,
        seeds: Sequence[int],
        pareto_fronts: dict[int, list],
        single_best: dict[int, Any],
        forward_masks: dict[int, list[int]],
        all_masks: dict[int, list[int]],
        use_knee_point: bool) -> dict[int, dict[str, dict]]:
    """The final model of every method and seed, refit on the whole training set:
    `{seed: {"multi": package, "single": ..., "all": ..., "forward": ...}}`. MORSE is the individual of
    the seed's Pareto front picked by the pipeline's rule (`select_pareto_individual`)."""
    packages: dict[int, dict[str, dict]] = {}
    for seed in seeds:
        morse = select_pareto_individual(pareto_fronts[seed], use_knee_point=use_knee_point)
        packages[seed] = {
            "multi":   build_model_package(morse, feature_names, X_train, y_train, seed=seed),
            "single":  build_model_package(single_best[seed], feature_names, X_train, y_train, seed=seed),
            "all":     build_model_package(all_masks[seed], feature_names, X_train, y_train, seed=seed),
            "forward": build_model_package(forward_masks[seed], feature_names, X_train, y_train, seed=seed),
        }
    return packages


@dataclass
class _Model:
    method: str
    seed: int | None           # None: a deterministic baseline evaluated once
    package: dict[str, Any]
    columns: numpy.ndarray     # positions of the model's inputs in the feature order

    @property
    def key(self) -> str:
        return f"{self.method}/{'-' if self.seed is None else self.seed}"

    def predict(self, X: numpy.ndarray) -> numpy.ndarray:
        scaled: numpy.ndarray = self.package["scaler"].transform(X[:, self.columns])
        return self.package["model"].predict_proba(scaled)[:, 1]


def _model_instances(model_packages: dict[int, dict[str, dict]], features: list[str],
                     say: Callable[[str], None]) -> list[_Model]:
    """MORSE and the SO-GA once per seed; SFS and the all-features model once, as they do not depend on
    the seed (identical inputs in every seed, and the lbfgs fit ignores its random_state)."""
    position: dict[str, int] = {name: j for j, name in enumerate(features)}
    seeds: list[int] = sorted(model_packages)
    models: list[_Model] = []
    for method in METHODS:
        per_seed: list[tuple[int, dict]] = [(seed, model_packages[seed][method]) for seed in seeds
                                            if method in model_packages[seed]]
        if not per_seed:
            continue
        single_instance: bool = method in ("forward", "all") and len(
            {tuple(package["features"]) for _, package in per_seed}) == 1
        if method in ("forward", "all") and not single_instance:
            say(f"NOTE: the {METHOD_STYLES[method][0]} inputs differ between the seeds; it is evaluated per seed.")
        for seed, package in (per_seed[:1] if single_instance else per_seed):
            models.append(_Model(method=method, seed=None if single_instance else seed, package=package,
                                 columns=numpy.array([position[name] for name in package["features"]], dtype=int)))
    return models


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------

@dataclass
class RobustnessResults:
    output_directory: str
    schema: FeatureSchema
    models: pandas.DataFrame
    scenarios: pandas.DataFrame
    model_family_scores: pandas.DataFrame
    summary: pandas.DataFrame
    tests: pandas.DataFrame
    report: str


def run_robustness_suite(
        X_train: pandas.DataFrame,
        y_train: Sequence[int],
        X_test: pandas.DataFrame,
        y_test: Sequence[int],
        model_packages: dict[int, dict[str, dict]],
        output_directory: str,
        config: RobustnessConfig | None = None,
        run_directory: str | None = None,
        log: Callable[[str], None] = print) -> RobustnessResults:
    """Evaluate the final models (`build_final_models`) under every automatically generated stress and
    write the tables, the figures and the report to `output_directory` (see the module docstring).

    X_train / y_train: the training data the models were fit on (the scenarios are derived from them);
    X_test / y_test:   the test set that is re-weighted and corrupted;
    run_directory:     the run folder, recorded in config.json together with its checkpoint fingerprint.
    """
    config = config or RobustnessConfig()
    started: float = time.time()
    lines: list[str] = []

    def say(line: str = "") -> None:
        lines.append(line)
        log(line)

    features: list[str] = list(X_train.columns)
    X_test = X_test[features]
    y_tr: numpy.ndarray = numpy.asarray(y_train).astype(int)
    y_te: numpy.ndarray = numpy.asarray(y_test).astype(int)
    ensure_directory(output_directory)

    # ---- models and their clean scores
    models: list[_Model] = _model_instances(model_packages, features, say)
    X_clean: numpy.ndarray = X_test.to_numpy(dtype=float)
    clean: dict[str, numpy.ndarray] = {model.key: model.predict(X_clean) for model in models}
    # self-check: the fast paths reproduce evaluation_utils.predict_scores and sklearn's metrics
    reference: _Model = models[0]
    if not (numpy.allclose(clean[reference.key], predict_scores(reference.package, X_test), rtol=0, atol=1e-12)
            and abs(weighted_roc_auc(y_te, clean[reference.key]) - roc_auc_score(y_te, clean[reference.key])) < 1e-10
            and abs(weighted_average_precision(y_te, clean[reference.key])
                    - average_precision_score(y_te, clean[reference.key])) < 1e-10):
        raise RuntimeError("the robustness suite's predictions or metrics do not reproduce evaluation_utils / "
                           "sklearn on the clean test set")
    marginal: pandas.Series = pandas.Series(compute_marginal_correlations(X_train, y_tr), index=features)
    model_rows: list[dict[str, Any]] = []
    for model in models:
        model_rows.append({
            "method": model.method, "seed": model.seed,
            **compute_model_sign_consistency(model.package, marginal),
            "clean_roc_auc": weighted_roc_auc(y_te, clean[model.key]),
            "clean_pr_auc": weighted_average_precision(y_te, clean[model.key])})
    models_frame: pandas.DataFrame = pandas.DataFrame(model_rows)
    models_frame["seed"] = models_frame["seed"].astype("Int64")

    counts: dict[str, int] = {method: sum(model.method == method for model in models) for method in METHODS}
    say(f"Robustness suite: {len(y_tr)} training rows, {len(y_te)} test rows "
        f"({int(y_te.sum())} positive), {len(features)} inputs; models: "
        + ", ".join(f"{METHOD_STYLES[m][0]} x{counts[m]}" for m in METHODS if counts[m]))

    # ---- automatic schema
    schema: FeatureSchema = infer_feature_schema(X_train, config)
    summary_counts: dict[str, int] = schema.summary()
    say(f"Automatic schema (training rows): {summary_counts['binary']} 0/1 inputs "
        f"({summary_counts['one_hot_groups']} one-hot groups with {summary_counts['one_hot_levels']} levels, "
        f"{summary_counts['availability_flags']} availability flags, {summary_counts['stand_alone_binary']} "
        f"stand-alone), {summary_counts['continuous']} continuous, {summary_counts['constant']} constant, "
        f"{summary_counts['other']} other; {summary_counts['forbidden_combinations']} forbidden 0/1 combinations.")

    # ---- re-weighted populations
    scenarios, reweighting_scores, n_eligible_pairs, n_pairs = _reweighting(
        X_train, X_test, y_te, schema, models, clean, config)
    n_usable: int = int((scenarios["supported"] & ~scenarios["saturated"]).sum()) if not scenarios.empty else 0
    say(f"Re-weighting: {len(scenarios)} scenarios ({n_usable} usable = supported by the test set "
        f"and not saturated); dependence pairs: {n_eligible_pairs} eligible, {n_pairs} used.")

    # ---- corrupted test sets
    corruption_scores, diagnostics, applicable = _corruption(X_train, X_test, y_te, schema, models, clean, config)
    say("Corruption families: " + (", ".join(applicable) if applicable else "none applicable") + ".")

    # ---- summaries
    all_family_scores: pandas.DataFrame = _model_family_scores(scenarios, reweighting_scores, corruption_scores)
    # a dependence family is summarised at a severity only with enough usable pairs there
    all_family_scores["summarised"] = (
        ~(all_family_scores["family"].astype(str).str.startswith("dependence_")
          & (all_family_scores["n_scenarios"] < config.min_pairs)) if not all_family_scores.empty else [])
    family_scores: pandas.DataFrame = all_family_scores[all_family_scores["summarised"].astype(bool)].drop(
        columns="summarised").reset_index(drop=True)
    skipped: pandas.DataFrame = (
        all_family_scores.loc[~all_family_scores["summarised"].astype(bool), ["family", "severity", "n_scenarios"]]
        .drop_duplicates() if not all_family_scores.empty else pandas.DataFrame())
    summary: pandas.DataFrame = _summary(family_scores, models_frame)
    tests: pandas.DataFrame = _tests(family_scores, scenarios, reweighting_scores)
    sign_points, sign_summary = _sign_analysis(models_frame, family_scores, config)

    # ---- files
    atomic_write_json(os.path.join(output_directory, "config.json"),
                      _provenance(config, run_directory, X_train, X_test, models_frame))
    atomic_write_json(os.path.join(output_directory, "schema.json"), schema.to_dict())
    tables: dict[str, pandas.DataFrame] = {
        "models.csv": models_frame, "scenarios.csv": scenarios, "reweighting_scores.csv": reweighting_scores,
        "corruption_scores.csv": corruption_scores, "corruption_diagnostics.csv": diagnostics,
        "model_family_scores.csv": all_family_scores, "summary.csv": summary, "tests.csv": tests,
        "sign_vs_degradation.csv": sign_summary}
    for name, frame in tables.items():
        frame.to_csv(os.path.join(output_directory, name), index=False)
    sign_points.to_csv(os.path.join(output_directory, "sign_vs_degradation_points.csv"), index=False)

    _figures(output_directory, config, scenarios, reweighting_scores, corruption_scores, family_scores,
             models_frame, sign_points, sign_summary)

    for line in _report_lines(summary, tests, sign_summary, skipped, config):
        say(line)
    say(f"\nRobustness suite finished in {time.time() - started:.0f} s; outputs in {output_directory}")
    report: str = "\n".join(lines) + "\n"
    atomic_write_text(os.path.join(output_directory, "report.txt"), report)
    return RobustnessResults(output_directory=output_directory, schema=schema, models=models_frame,
                             scenarios=scenarios, model_family_scores=family_scores, summary=summary,
                             tests=tests, report=report)


# ---------------------------------------------------------------------------
# Re-weighting and corruption
# ---------------------------------------------------------------------------

def _reweighting(X_train: pandas.DataFrame, X_test: pandas.DataFrame, y_test: numpy.ndarray,
                 schema: FeatureSchema, models: list[_Model], clean: dict[str, numpy.ndarray],
                 config: RobustnessConfig) -> tuple[pandas.DataFrame, pandas.DataFrame, int, int]:
    scenario_rows: list[dict[str, Any]] = []
    score_rows: list[tuple] = []
    clean_roc: dict[str, float] = {key: weighted_roc_auc(y_test, p) for key, p in clean.items()}
    clean_ap: dict[str, float] = {key: weighted_average_precision(y_test, p) for key, p in clean.items()}

    def add(row: dict[str, Any], w_train: numpy.ndarray | None, w_test: numpy.ndarray | None) -> None:
        """Record a scenario and score every model under it; a saturated scenario (its severity cannot
        be reached, so there are no weights) is only recorded."""
        if w_train is None or w_test is None:
            row.update({"train_ess": numpy.nan, "test_ess": numpy.nan, "ess_negatives": numpy.nan,
                        "ess_positives": numpy.nan, "supported": False, "weighted_prevalence": numpy.nan})
            scenario_rows.append(row)
            return
        negatives, positives = class_effective_sizes(w_test, y_test)
        test_ess: float = effective_sample_fraction(w_test)
        row.update({
            "train_ess": effective_sample_fraction(w_train), "test_ess": test_ess,
            "ess_negatives": negatives, "ess_positives": positives,
            "supported": bool(test_ess >= config.min_test_ess_fraction
                              and min(negatives, positives) >= config.min_class_ess),
            "weighted_prevalence": float(w_test @ y_test / w_test.sum())})
        scenario_rows.append(row)
        class_weights: numpy.ndarray = normalise_within_classes(w_test, y_test)
        for model in models:
            p: numpy.ndarray = clean[model.key]
            roc: float = weighted_roc_auc(y_test, p, class_weights)
            ap: float = weighted_average_precision(y_test, p, class_weights)
            score_rows.append((row["scenario_id"], model.method, model.seed, roc, ap,
                               roc - clean_roc[model.key], ap - clean_ap[model.key]))

    # population shifts
    for axis in population_axes(X_train, X_test, config):
        for direction, sign in (("+", 1.0), ("-", -1.0)):
            calibrated = calibrate_strengths(
                lambda s: effective_sample_fraction(exponential_tilt(axis.train, sign * s)), config.ess_levels)
            for level, (strength, saturated) in zip(config.ess_levels, calibrated):
                row: dict[str, Any] = {
                    "scenario_id": f"population|{axis.name}|{direction}|{level:g}", "family": "population",
                    "scenario": axis.name, "direction": direction,
                    "emphasis": axis.positive if sign > 0 else axis.negative, "ess_level": level,
                    "strength": numpy.nan if saturated else sign * strength, "saturated": saturated,
                    "description": axis.description}
                if saturated:
                    add(row, None, None)
                    continue
                w_train: numpy.ndarray = exponential_tilt(axis.train, sign * strength)
                w_test: numpy.ndarray = exponential_tilt(axis.test, sign * strength)
                row["statistic_shift"] = float(w_test @ axis.test / w_test.sum() - axis.test.mean())
                add(row, w_train, w_test)

    # dependence shifts
    pairs, n_eligible = dependence_pairs(X_train, X_test, schema, config)
    binary: set[str] = set(schema.binary)
    for pair in pairs:
        a_train, b_train = X_train[pair.a].to_numpy(dtype=float), X_train[pair.b].to_numpy(dtype=float)
        a_test, b_test = X_test[pair.a].to_numpy(dtype=float), X_test[pair.b].to_numpy(dtype=float)
        tilt: DependenceTilt = DependenceTilt(a_train, b_train, a_test, b_test,
                                              pair.a in binary, pair.b in binary, config.score_clip)
        clean_correlation: float = weighted_correlation(a_test, b_test)
        sd_a, sd_b = float(a_train.std()), float(b_train.std())
        towards_positive: float = 1.0 if pair.correlation > 0 else -1.0
        for direction, sign in (("decorrelate", -towards_positive), ("strengthen", towards_positive)):
            calibrated = calibrate_strengths(lambda s: tilt.train_ess(sign * s), config.ess_levels)
            for level, (strength, saturated) in zip(config.ess_levels, calibrated):
                row = {"scenario_id": f"dependence|{pair.a} x {pair.b}|{direction}|{level:g}",
                       "family": f"dependence_{direction}", "scenario": f"{pair.a} x {pair.b}",
                       "direction": direction, "emphasis": direction, "ess_level": level,
                       "strength": numpy.nan if saturated else sign * strength, "saturated": saturated,
                       "pair_a": pair.a, "pair_b": pair.b, "pair_kind": pair.kind,
                       "correlation_train": pair.correlation, "correlation_test": clean_correlation}
                if saturated:
                    add(row, None, None)
                    continue
                w_train, w_test = tilt.weights(sign * strength)
                shares: numpy.ndarray = w_test / w_test.sum()
                row.update({
                    "correlation_weighted": weighted_correlation(a_test, b_test, w_test),
                    "mean_shift_a": float((shares @ a_test - a_test.mean()) / sd_a),
                    "mean_shift_b": float((shares @ b_test - b_test.mean()) / sd_b),
                    "sd_ratio_a": _sd_ratio(a_test, shares),
                    "sd_ratio_b": _sd_ratio(b_test, shares),
                    "balance_error": tilt.balance_error})
                add(row, w_train, w_test)

    scenarios: pandas.DataFrame = pandas.DataFrame(scenario_rows)
    scores: pandas.DataFrame = pandas.DataFrame(score_rows, columns=[
        "scenario_id", "method", "seed", "roc_auc", "pr_auc", "delta_roc_auc", "delta_pr_auc"])
    scores["seed"] = scores["seed"].astype("Int64")
    return scenarios, scores, n_eligible, len(pairs)


def _sd_ratio(values: numpy.ndarray, shares: numpy.ndarray) -> float:
    """SD of a column under the weights (`shares` sum to 1) relative to its unweighted SD."""
    unweighted: float = float(values.std())
    if unweighted == 0.0:
        return float("nan")
    return float(numpy.sqrt(shares @ (values - shares @ values) ** 2)) / unweighted


def _corruption(X_train: pandas.DataFrame, X_test: pandas.DataFrame, y_test: numpy.ndarray,
                schema: FeatureSchema, models: list[_Model], clean: dict[str, numpy.ndarray],
                config: RobustnessConfig) -> tuple[pandas.DataFrame, pandas.DataFrame, list[str]]:
    bank: CorruptionBank = CorruptionBank(schema, X_train, X_test, config.corruption_repetitions,
                                          config.corruption_seed)
    clean_roc: dict[str, float] = {key: weighted_roc_auc(y_test, p) for key, p in clean.items()}
    clean_ap: dict[str, float] = {key: weighted_average_precision(y_test, p) for key, p in clean.items()}
    score_rows: list[tuple] = []
    diagnostic_rows: list[dict[str, Any]] = []
    applicable: list[str] = bank.applicable_families()
    for family in applicable:
        for level in config.corruption_levels:
            for repetition in range(config.corruption_repetitions):
                corrupted, counts = bank.corrupt(family, level, repetition)
                X: numpy.ndarray = corrupted.to_numpy(dtype=float)
                for model in models:
                    p: numpy.ndarray = model.predict(X)
                    roc: float = weighted_roc_auc(y_test, p)
                    ap: float = weighted_average_precision(y_test, p)
                    score_rows.append((family, level, repetition, model.method, model.seed, roc, ap,
                                       roc - clean_roc[model.key], ap - clean_ap[model.key]))
                diagnostic_rows.append({"family": family, "level": level, "repetition": repetition, **counts})
    scores: pandas.DataFrame = pandas.DataFrame(score_rows, columns=[
        "family", "level", "repetition", "method", "seed", "roc_auc", "pr_auc", "delta_roc_auc", "delta_pr_auc"])
    scores["seed"] = scores["seed"].astype("Int64")
    return scores, pandas.DataFrame(diagnostic_rows), applicable


# ---------------------------------------------------------------------------
# Summaries and statistics
# ---------------------------------------------------------------------------

def headline_statistic(family: str) -> tuple[str, str, str]:
    """(score column, change column, description) of the statistic that summarises a family. For the
    population shifts it is the WORST scenario: their scenarios move the population in opposite
    directions along each axis, so a mean would net harmful shifts against beneficial ones. For a
    dependence family (one direction each) it is the mean over the pairs, for a corruption family the
    mean over the corruption bank."""
    if family == "population":
        return "worst_roc_auc", "worst_delta_roc_auc", "worst scenario"
    if family.startswith("dependence_"):
        return "roc_auc", "delta_roc_auc", "mean over pairs"
    return "roc_auc", "delta_roc_auc", "mean over the bank"


def _in_order(frame: pandas.DataFrame) -> pandas.DataFrame:
    """Families in FAMILY_ORDER; within a family the mildest severity first (re-weighting: the highest
    ESS; corruption: the lowest level); methods in METHODS order."""
    if frame.empty:
        return frame
    family_rank: pandas.Series = frame["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
    severity: pandas.Series = frame["severity"].astype(float)
    severity_rank: pandas.Series = severity.where(~frame["family"].isin(REWEIGHTING_FAMILIES), -severity)
    keys: dict[str, pandas.Series] = {"_family": family_rank, "_severity": severity_rank}
    if "method" in frame:
        keys["_method"] = frame["method"].map({m: i for i, m in enumerate(METHODS)})
    return (frame.assign(**keys).sort_values(list(keys), kind="stable")
            .drop(columns=list(keys)).reset_index(drop=True))

def _model_family_scores(scenarios: pandas.DataFrame, reweighting_scores: pandas.DataFrame,
                         corruption_scores: pandas.DataFrame) -> pandas.DataFrame:
    """One row per family, severity and model: re-weighting -> mean and minimum over the family's usable
    scenarios; corruption -> mean over the repetitions, and their SD (the corruption variability)."""
    frames: list[pandas.DataFrame] = []
    if not scenarios.empty:
        usable: pandas.DataFrame = scenarios.loc[scenarios["supported"] & ~scenarios["saturated"],
                                                 ["scenario_id", "family", "ess_level"]]
        merged: pandas.DataFrame = reweighting_scores.merge(usable, on="scenario_id")
        if not merged.empty:
            frames.append(merged.groupby(["family", "ess_level", "method", "seed"], dropna=False).agg(
                roc_auc=("roc_auc", "mean"), delta_roc_auc=("delta_roc_auc", "mean"),
                pr_auc=("pr_auc", "mean"), delta_pr_auc=("delta_pr_auc", "mean"),
                worst_roc_auc=("roc_auc", "min"), worst_delta_roc_auc=("delta_roc_auc", "min"),
                n_scenarios=("roc_auc", "size")).reset_index().rename(columns={"ess_level": "severity"}))
    if not corruption_scores.empty:
        frames.append(corruption_scores.groupby(["family", "level", "method", "seed"], dropna=False).agg(
            roc_auc=("roc_auc", "mean"), delta_roc_auc=("delta_roc_auc", "mean"),
            pr_auc=("pr_auc", "mean"), delta_pr_auc=("delta_pr_auc", "mean"),
            corruption_sd=("roc_auc", "std"),
            n_scenarios=("roc_auc", "size")).reset_index().rename(columns={"level": "severity"}))
    if not frames:
        return pandas.DataFrame(columns=["family", "severity", "method", "seed"])
    scores: pandas.DataFrame = pandas.concat(frames, ignore_index=True)
    scores["seed"] = scores["seed"].astype("Int64")
    return _in_order(scores)


def _summary(family_scores: pandas.DataFrame, models_frame: pandas.DataFrame) -> pandas.DataFrame:
    """Per family, severity and method: the mean over the runs and the SD ACROSS RUNS (the algorithmic
    variability, after averaging over scenarios / repetitions; empty for SFS and all features), the
    clean score, the change, the worst usable scenario (re-weighting) and the corruption SD
    (corruption: SD over the repetitions, averaged over the runs)."""
    clean: pandas.DataFrame = models_frame.groupby("method").agg(
        clean_roc_auc=("clean_roc_auc", "mean"), clean_pr_auc=("clean_pr_auc", "mean"))
    rows: list[dict[str, Any]] = []
    for (family, severity, method), part in family_scores.groupby(["family", "severity", "method"], sort=False):
        row: dict[str, Any] = {"family": family, "severity": severity, "method": method,
                               "n_runs": len(part),
                               "clean_roc_auc": clean.loc[method, "clean_roc_auc"],
                               "roc_auc": part["roc_auc"].mean(),
                               "roc_auc_sd": part["roc_auc"].std(ddof=1) if len(part) > 1 else numpy.nan,
                               "delta_roc_auc": part["delta_roc_auc"].mean(),
                               "delta_roc_auc_sd": part["delta_roc_auc"].std(ddof=1) if len(part) > 1 else numpy.nan,
                               "clean_pr_auc": clean.loc[method, "clean_pr_auc"],
                               "pr_auc": part["pr_auc"].mean(), "delta_pr_auc": part["delta_pr_auc"].mean(),
                               "n_scenarios": int(part["n_scenarios"].max())}
        for column in ("worst_roc_auc", "worst_delta_roc_auc"):
            if column in part and part[column].notna().any():
                row[column] = part[column].mean()
                row[f"{column}_sd"] = part[column].std(ddof=1) if len(part) > 1 else numpy.nan
        if "corruption_sd" in part and part["corruption_sd"].notna().any():
            row["corruption_sd"] = part["corruption_sd"].mean()
        rows.append(row)
    return _in_order(pandas.DataFrame(rows))


def _wilcoxon(differences: numpy.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value (exact for small samples without ties); NaN if every
    difference is zero or there are fewer than two."""
    differences = differences[~numpy.isnan(differences)]
    if differences.size < 2 or numpy.allclose(differences, 0.0):
        return float("nan")
    return float(wilcoxon(differences).pvalue)


def _tests(family_scores: pandas.DataFrame, scenarios: pandas.DataFrame,
           reweighting_scores: pandas.DataFrame) -> pandas.DataFrame:
    """MORSE against every baseline, per family, severity and statistic (the per-run mean ROC-AUC, its
    change from the model's clean score, and for re-weighting the worst usable scenario and its change;
    `headline_statistic` says which one summarises a family). A positive difference means MORSE is
    better (higher score, smaller loss). Against the SO-GA: paired over the seeds; against SFS / all
    features: MORSE's seeds against the baseline's single value. For the re-weighting families the table
    also gives the share of usable scenarios in which MORSE's run-averaged change is better than the
    SO-GA's (descriptive only: the scenarios share their test rows, so they are not independent)."""
    share: dict[tuple[str, float], tuple[float, int]] = {}
    if not scenarios.empty:
        usable: pandas.DataFrame = scenarios.loc[scenarios["supported"] & ~scenarios["saturated"],
                                                 ["scenario_id", "family", "ess_level"]]
        merged: pandas.DataFrame = reweighting_scores.merge(usable, on="scenario_id")
        per_scenario: pandas.DataFrame = merged[merged["method"].isin(["multi", "single"])].groupby(
            ["family", "ess_level", "scenario_id", "method"])["delta_roc_auc"].mean().unstack("method")
        if {"multi", "single"} <= set(per_scenario.columns):
            better: pandas.Series = (per_scenario["multi"] > per_scenario["single"]).groupby(
                level=["family", "ess_level"]).agg(["mean", "size"])
            share = {key: (float(value["mean"]), int(value["size"])) for key, value in better.iterrows()}

    rows: list[dict[str, Any]] = []
    for (family, severity), part in family_scores.groupby(["family", "severity"], sort=False):
        morse: pandas.DataFrame = part[part["method"] == "multi"].set_index("seed")
        if morse.empty:
            continue
        for statistic in ("roc_auc", "delta_roc_auc", "worst_roc_auc", "worst_delta_roc_auc"):
            if statistic not in part or part[statistic].isna().all():
                continue
            for baseline in ("single", "forward", "all"):
                base: pandas.DataFrame = part[part["method"] == baseline]
                if base.empty:
                    continue
                if base["seed"].isna().all():
                    differences: numpy.ndarray = (morse[statistic] - base[statistic].iloc[0]).to_numpy(dtype=float)
                else:
                    aligned: pandas.Series = morse[statistic] - base.set_index("seed")[statistic]
                    differences = aligned.dropna().to_numpy(dtype=float)
                row: dict[str, Any] = {
                    "family": family, "severity": severity, "statistic": statistic, "baseline": baseline,
                    "n_runs": int(differences.size), "mean_difference": float(numpy.mean(differences)),
                    "median_difference": float(numpy.median(differences)),
                    "n_morse_better": int((differences > 0).sum()), "p_value": _wilcoxon(differences)}
                if baseline == "single" and statistic == "delta_roc_auc" and (family, severity) in share:
                    row["scenario_share_morse_better"], row["n_scenarios"] = share[(family, severity)]
                rows.append(row)
    return _in_order(pandas.DataFrame(rows))


def partial_spearman(x: numpy.ndarray, y: numpy.ndarray, z: numpy.ndarray) -> float:
    """Spearman correlation of x and y after removing the (rank-)linear effect of z."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    design: numpy.ndarray = numpy.column_stack([numpy.ones_like(rz), rz])
    ex: numpy.ndarray = rx - design @ numpy.linalg.lstsq(design, rx, rcond=None)[0]
    ey: numpy.ndarray = ry - design @ numpy.linalg.lstsq(design, ry, rcond=None)[0]
    # undefined when x or y is constant or fully explained by z; the residuals are then ~1e-16 rounding
    # noise rather than exactly zero, so compare with a tolerance (ranks are of the order of n)
    if ex.std() < 1e-9 or ey.std() < 1e-9:
        return float("nan")
    return float(numpy.corrcoef(ex, ey)[0, 1])


def _sign_analysis(models_frame: pandas.DataFrame, family_scores: pandas.DataFrame,
                   config: RobustnessConfig) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """Sign consistency S of every GA model (MORSE and SO-GA, all seeds) against its change of ROC-AUC
    per family (the family's headline statistic, at the headline severity), also given the model size K
    (partial Spearman). Exploratory: a correlation does not show that S causes the difference."""
    ga: pandas.DataFrame = models_frame[models_frame["method"].isin(["multi", "single"])][
        ["method", "seed", "sign_consistency", "n_features"]]
    points: list[pandas.DataFrame] = []
    rows: list[dict[str, Any]] = []
    present: set[str] = set(family_scores["family"]) if not family_scores.empty else set()
    for family in [f for f in FAMILY_ORDER if f in present]:
        severity: float = config.headline_ess if family in REWEIGHTING_FAMILIES else config.headline_corruption_level
        change: str = headline_statistic(family)[1]
        part: pandas.DataFrame = family_scores[(family_scores["family"] == family)
                                               & numpy.isclose(family_scores["severity"].astype(float), severity)]
        merged: pandas.DataFrame = ga.merge(part[["method", "seed", change]], on=["method", "seed"]).rename(
            columns={change: "delta_roc_auc"})
        if len(merged) < 4:
            continue
        merged.insert(0, "family", family)
        merged.insert(1, "severity", severity)
        points.append(merged)
        s = merged["sign_consistency"].to_numpy(dtype=float)
        d = merged["delta_roc_auc"].to_numpy(dtype=float)
        k = merged["n_features"].to_numpy(dtype=float)
        rows.append({"family": family, "severity": severity, "n_models": len(merged),
                     "spearman": float(spearmanr(s, d)[0]) if numpy.ptp(s) > 0 and numpy.ptp(d) > 0 else numpy.nan,
                     "partial_spearman_given_size": partial_spearman(s, d, k)})
    return (pandas.concat(points, ignore_index=True) if points else pandas.DataFrame(),
            pandas.DataFrame(rows))


# ---------------------------------------------------------------------------
# Figures and report
# ---------------------------------------------------------------------------

def _figures(directory: str, config: RobustnessConfig, scenarios: pandas.DataFrame,
             reweighting_scores: pandas.DataFrame, corruption_scores: pandas.DataFrame,
             family_scores: pandas.DataFrame, models_frame: pandas.DataFrame,
             sign_points: pandas.DataFrame, sign_summary: pandas.DataFrame) -> None:
    clean_per_model: pandas.DataFrame = models_frame[["method", "seed", "clean_roc_auc"]]
    styles: dict[str, tuple[str, str, str, str]] = {m: METHOD_STYLES[m] for m in METHODS
                                                    if m in set(models_frame["method"])}
    levels: list[float] = list(config.ess_levels)

    def across_runs(frame: pandas.DataFrame, keys: list[str], value: str) -> pandas.DataFrame:
        grouped = frame.groupby(keys + ["method"], dropna=False)[value]
        return grouped.agg(mean="mean", sd=lambda v: v.std(ddof=1) if len(v) > 1 else numpy.nan).reset_index()

    clean_curve: pandas.DataFrame = across_runs(clean_per_model.assign(position=0), ["position"], "clean_roc_auc")

    usable: pandas.DataFrame = scenarios.loc[scenarios["supported"] & ~scenarios["saturated"]] \
        if not scenarios.empty else scenarios
    if not usable.empty:
        merged: pandas.DataFrame = reweighting_scores.merge(
            usable[["scenario_id", "family", "scenario", "direction", "ess_level"]], on="scenario_id")

        # population: one panel per axis, both directions
        population: pandas.DataFrame = merged[merged["family"] == "population"].copy()
        if not population.empty:
            rank: dict[float, int] = {level: i + 1 for i, level in enumerate(levels)}
            population["position"] = population["ess_level"].map(rank) * population["direction"].map({"+": 1, "-": -1})
            curves: pandas.DataFrame = across_runs(population, ["scenario", "position"], "roc_auc").rename(
                columns={"scenario": "axis"})
            axes_order: list[str] = list(dict.fromkeys(scenarios.loc[scenarios["family"] == "population", "scenario"]))
            with_clean: pandas.DataFrame = pandas.concat(
                [curves] + [clean_curve.assign(axis=axis) for axis in axes_order], ignore_index=True)
            emphasis: pandas.DataFrame = scenarios[scenarios["family"] == "population"].drop_duplicates(
                ["scenario", "direction"]).set_index(["scenario", "direction"])["emphasis"]
            panels: list[tuple[str, str, str]] = [(axis, emphasis.get((axis, "-"), "-"), emphasis.get((axis, "+"), "+"))
                                                  for axis in axes_order]
            plot_population_shift_curves(with_clean, panels, levels, styles,
                                         os.path.join(directory, "robustness_population.png"),
                                         "Population shifts: test ROC-AUC of the re-weighted test population\n"
                                         "(mean over runs; band = SD across runs)")

        # dependence: mean change over the pairs usable at EVERY level (so that the curves compare the same
        # pairs across the levels), and MORSE - SO-GA per pair
        dependence: pandas.DataFrame = merged[merged["family"].str.startswith("dependence_")].copy()
        levels_per_pair: pandas.Series = usable[usable["family"].str.startswith("dependence_")].groupby(
            ["direction", "scenario"])["ess_level"].nunique()
        common: pandas.DataFrame = levels_per_pair[levels_per_pair == len(levels)].reset_index()[
            ["direction", "scenario"]]
        n_common: dict[str, int] = common["direction"].value_counts().to_dict()
        enough: list[str] = [direction for direction, count in n_common.items() if count >= config.min_pairs]
        dependence = dependence.merge(common[common["direction"].isin(enough)], on=["direction", "scenario"])
        if not dependence.empty:
            per_run: pandas.DataFrame = dependence.groupby(["direction", "ess_level", "method", "seed"],
                                                           dropna=False)["delta_roc_auc"].mean().reset_index()
            curves = across_runs(per_run, ["direction", "ess_level"], "delta_roc_auc")
            per_pair: pandas.DataFrame = dependence[dependence["method"].isin(["multi", "single"])].groupby(
                ["direction", "ess_level", "scenario", "method"])["delta_roc_auc"].mean().unstack("method")
            differences: pandas.DataFrame = pandas.DataFrame()
            if {"multi", "single"} <= set(per_pair.columns):
                differences = (per_pair["multi"] - per_pair["single"]).rename("difference").reset_index()
            plot_dependence_shift_summary(curves, differences, levels, styles,
                                          os.path.join(directory, "robustness_dependence.png"),
                                          f"Dependence shifts: change of test ROC-AUC over the pairs usable at every "
                                          f"level ({n_common.get('decorrelate', 0)} decorrelated, "
                                          f"{n_common.get('strengthen', 0)} strengthened; a direction with fewer than "
                                          f"{config.min_pairs} is left out)\nmean over runs, band = SD across runs")

    if not corruption_scores.empty:
        per_run = corruption_scores.groupby(["family", "level", "method", "seed"], dropna=False)[
            "roc_auc"].mean().reset_index()
        curves = across_runs(per_run, ["family", "level"], "roc_auc")
        families: list[str] = list(dict.fromkeys(corruption_scores["family"]))
        with_clean = pandas.concat([curves] + [clean_curve.drop(columns="position").assign(family=f, level=0.0)
                                               for f in families], ignore_index=True)
        plot_corruption_curves(with_clean, [(f, FAMILY_LABELS[f], CORRUPTION_LEVEL_LABELS[f]) for f in families],
                               styles, os.path.join(directory, "robustness_corruption.png"),
                               "Corrupted test sets: test ROC-AUC (mean over the corruption bank, then over runs; "
                               "band = SD across runs)")

    if not family_scores.empty:
        overview_rows: list[dict[str, Any]] = []
        clean_stats: pandas.DataFrame = across_runs(clean_per_model, [], "clean_roc_auc").set_index("method")
        overview_labels: dict[str, str] = {}
        for family in [f for f in FAMILY_ORDER if f in set(family_scores["family"])]:
            reweighting: bool = family in REWEIGHTING_FAMILIES
            severity: float = config.headline_ess if reweighting else config.headline_corruption_level
            score, _, description = headline_statistic(family)
            overview_labels[family] = (f"{FAMILY_LABELS[family]} ({description}, "
                                       + (f"ESS {severity:.0%})" if reweighting else f"level {severity:g})"))
            part: pandas.DataFrame = family_scores[(family_scores["family"] == family)
                                                   & numpy.isclose(family_scores["severity"].astype(float), severity)]
            for method, group in part.groupby("method"):
                overview_rows.append({
                    "family": family, "method": method,
                    "clean_mean": clean_stats.loc[method, "mean"], "clean_sd": clean_stats.loc[method, "sd"],
                    "stressed_mean": group[score].mean(),
                    "stressed_sd": group[score].std(ddof=1) if len(group) > 1 else numpy.nan,
                    "worst_mean": numpy.nan, "worst_sd": numpy.nan})
        if overview_rows:
            plot_robustness_overview(pandas.DataFrame(overview_rows), overview_labels, styles,
                                     os.path.join(directory, "robustness_overview.png"),
                                     "Clean (hollow) vs stressed (filled) test ROC-AUC per stress family "
                                     "(error bars = SD across runs)")

    if not sign_points.empty:
        plot_sign_vs_degradation(sign_points, sign_summary, FAMILY_LABELS,
                                 {m: METHOD_STYLES[m] for m in ("multi", "single")},
                                 os.path.join(directory, "robustness_sign_vs_degradation.png"),
                                 "Sign consistency vs change of test ROC-AUC of the GA models "
                                 "(marker size = number of inputs; exploratory)")


def _format_p(p: float) -> str:
    if numpy.isnan(p):
        return "  --  "
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def _report_lines(summary: pandas.DataFrame, tests: pandas.DataFrame, sign_summary: pandas.DataFrame,
                  skipped: pandas.DataFrame, config: RobustnessConfig) -> list[str]:
    lines: list[str] = []
    if summary.empty:
        return ["\nNo scenario could be evaluated."]
    lines.append("\nTest ROC-AUC under stress, mean over runs, with its change from each model's clean score. A family")
    lines.append("is summarised by its worst scenario (population shifts) or by the mean over its pairs / corruption")
    lines.append("bank (the others). MORSE-SOGA: difference of that change, paired over the runs (> 0: MORSE loses")
    lines.append("less), Wilcoxon p, and the runs in which MORSE is better; n = usable scenarios (re-weighting) or")
    lines.append("corrupted test sets per level (corruption). clean ROC-AUC: "
                 + ", ".join(f"{METHOD_STYLES[m][0]} {summary.loc[summary['method'] == m, 'clean_roc_auc'].iloc[0]:.3f}"
                             for m in METHODS if (summary["method"] == m).any()))
    header: str = (f"{'family':32s} {'severity':>8s} {'n':>4s} "
                   + " ".join(f"{METHOD_STYLES[m][0]:>14s}" for m in METHODS)
                   + f" {'MORSE-SOGA':>11s} {'p':>7s} {'better':>6s}")
    lines.append(header)
    for (family, severity), part in summary.groupby(["family", "severity"], sort=False):
        score, change, _ = headline_statistic(family)
        cells: list[str] = []
        for method in METHODS:
            row: pandas.DataFrame = part[part["method"] == method]
            cells.append(f"{row[score].iloc[0]:.3f} ({row[change].iloc[0]:+.3f})"
                         if not row.empty and score in row and not numpy.isnan(row[score].iloc[0]) else f"{'':>14s}")
        test: pandas.DataFrame = tests[(tests["family"] == family) & numpy.isclose(tests["severity"].astype(float), severity)
                                       & (tests["statistic"] == change) & (tests["baseline"] == "single")] \
            if not tests.empty else pandas.DataFrame()
        comparison: str = (f"{test['mean_difference'].iloc[0]:+11.4f} {_format_p(test['p_value'].iloc[0]):>7s} "
                           f"{int(test['n_morse_better'].iloc[0]):>3d}/{int(test['n_runs'].iloc[0]):<2d}"
                           if not test.empty else "")
        severity_text: str = f"{severity:.0%} ESS" if family in REWEIGHTING_FAMILIES else f"{severity:g}"
        lines.append(f"{FAMILY_LABELS.get(family, family):32s} {severity_text:>8s} {int(part['n_scenarios'].max()):>4d} "
                     + " ".join(f"{cell:>14s}" for cell in cells) + " " + comparison)
    for _, row in skipped.iterrows():
        lines.append(f"{FAMILY_LABELS.get(row['family'], row['family']):32s} {row['severity']:.0%} ESS "
                     f"{int(row['n_scenarios']):>4d}   not summarised: fewer than {config.min_pairs} usable pairs")
    if not sign_summary.empty:
        lines.append("\nSign consistency vs change of ROC-AUC across the GA models (headline severity; exploratory):")
        for _, row in sign_summary.iterrows():
            lines.append(f"  {FAMILY_LABELS.get(row['family'], row['family']):32s} Spearman {row['spearman']:+.2f}, "
                         f"given the model size {row['partial_spearman_given_size']:+.2f} (n = {row['n_models']})")
    return lines


def _git_state() -> dict[str, Any]:
    try:
        commit: str = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository_root(), capture_output=True,
                                     text=True, timeout=10).stdout.strip()
        dirty: str = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repository_root(),
                                    capture_output=True, text=True, timeout=10).stdout.strip()
        return {"commit": commit or "unknown", "uncommitted_changes": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "uncommitted_changes": None}


def _provenance(config: RobustnessConfig, run_directory: str | None, X_train: pandas.DataFrame,
                X_test: pandas.DataFrame, models_frame: pandas.DataFrame) -> dict[str, Any]:
    training_fingerprint: str | None = None
    if run_directory:
        path: str = os.path.join(run_directory, "checkpoints", "training", FINGERPRINT_FILE)
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                training_fingerprint = hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": config.to_dict(),
        "run_directory": run_directory,
        "training_fingerprint_sha256": training_fingerprint,
        "data": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "inputs": int(X_train.shape[1])},
        "models": {method: int((models_frame["method"] == method).sum()) for method in METHODS},
        "seeds": sorted(int(s) for s in models_frame["seed"].dropna().unique()),
        "git": _git_state(),
        "source_sha256": source_fingerprint((robustness_utils, plot_utils, sys.modules[__name__],
                                             sys.modules[RobustnessConfig.__module__])),
    }


# ---------------------------------------------------------------------------
# Stand-alone use on a finished run
# ---------------------------------------------------------------------------

def load_archived_run(run_directory: str) -> dict[str, Any]:
    """Rebuild a run's training and test data by executing the configuration and data-loading cells of
    the copy of training_notebook.ipynb archived in the run folder: every code cell from the first one
    that assigns TARGET_COLUMN to the first one that assigns X_test. This reproduces the exact inputs
    of the run (paths, merged validation file, column whitelist) without restating them. The result is
    checked against the run's checkpoint fingerprint by `verify_training_data`."""
    path: str = os.path.join(run_directory, "training_notebook.ipynb")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} does not exist: the run folder has no archived copy of the notebook, "
                                f"so its data cannot be rebuilt")
    sources: list[str] = ["".join(cell.get("source", [])) for cell in read_json(path).get("cells", [])
                          if cell.get("cell_type") == "code"]
    start: int | None = next((i for i, source in enumerate(sources)
                              if re.search(r"^TARGET_COLUMN\b", source, re.MULTILINE)), None)
    end: int | None = None if start is None else next(
        (i for i in range(start, len(sources)) if re.search(r"^X_test\b", sources[i], re.MULTILINE)), None)
    if start is None or end is None:
        raise ValueError(f"{path}: no configuration cell (TARGET_COLUMN = ...) followed by a data cell "
                         f"(X_test = ...) was found")
    namespace: dict[str, Any] = {"os": os, "time": time, "numpy": numpy, "pandas": pandas,
                                 "repository_root": repository_root, "__name__": "archived_notebook"}
    previous: str = os.getcwd()
    os.chdir(repository_root())   # the cells use paths relative to the repository root
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for source in sources[start:end + 1]:
                exec(compile(source, path, "exec"), namespace)
    finally:
        os.chdir(previous)
    target: str = namespace["TARGET_COLUMN"]
    df_train: pandas.DataFrame = namespace["df_train"]
    return {"X_train": df_train.drop(columns=[target]), "y_train": df_train[target],
            "X_test": namespace["X_test"], "y_test": namespace["y_test"],
            "use_knee_point": bool(namespace.get("USE_KNEE_POINT_SELECTION", True))}


def verify_training_data(run_directory: str, X_train: pandas.DataFrame, y_train: Sequence[float]) -> dict[str, Any]:
    """Check that the training data are those the run's checkpoints were trained on -- the feature names
    and the hashes of the standardised training matrix and of the labels, recomputed exactly as the
    notebook computes them (checkpoint_utils.build_training_fingerprint) -- and return the fingerprint."""
    fingerprint: dict[str, Any] = read_json(os.path.join(run_directory, "checkpoints", "training", FINGERPRINT_FILE))
    names: list[str] = list(X_train.columns)
    X_search: numpy.ndarray = numpy.ascontiguousarray(X_train.to_numpy(), dtype=numpy.float64)
    current: dict[str, Any] = {
        "features": {"count": len(names), "sha256": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()},
        "X_train": array_fingerprint(StandardScaler().fit_transform(X_search)),
        "y_train": array_fingerprint(numpy.ascontiguousarray(numpy.asarray(y_train), dtype=numpy.float64)),
    }
    settings: dict[str, Any] = fingerprint.get("settings", {})
    problems: list[str] = []
    if settings.get("features") != current["features"]:
        problems.append("the input variables (names or order) differ")
    for part in ("X_train", "y_train"):
        if settings.get("data", {}).get(part) != current[part]:
            problems.append(f"the training data ({part}) differ")
    if problems:
        raise ValueError(f"the rebuilt data do not match the checkpoints of {run_directory}: "
                         + "; ".join(problems))
    return fingerprint


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the robustness suite on the final models of a checkpointed run (no retraining).")
    parser.add_argument("--run", required=True,
                        help="the run's result folder (absolute, or relative to the repository root)")
    parser.add_argument("--selection", choices=("auto", "knee", "max_s"), default="auto",
                        help="which Pareto solution is MORSE's final model; auto = the run's own rule")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="evaluate only these seeds (default: every seed with a complete checkpoint)")
    parser.add_argument("--out", default=None, help="output folder (default: <run>/evaluation/robustness)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    arguments: argparse.Namespace = parse_arguments(argv)
    run: str = arguments.run if os.path.isabs(arguments.run) else os.path.join(repository_root(), arguments.run)
    run = os.path.normpath(run)
    data: dict[str, Any] = load_archived_run(run)
    fingerprint: dict[str, Any] = verify_training_data(run, data["X_train"], data["y_train"])
    store: TrainingCheckpointStore = TrainingCheckpointStore(os.path.join(run, "checkpoints", "training"), fingerprint)
    seeds: list[int] = sorted(arguments.seeds) if arguments.seeds else store.completed_seeds()
    fronts, single, forward, everything = store.load_all(seeds)
    use_knee_point: bool = data["use_knee_point"] if arguments.selection == "auto" else arguments.selection == "knee"
    print(f"Run {run}: {len(seeds)} seeds; MORSE = the {'knee point' if use_knee_point else 'max-S end'} "
          f"of every Pareto front.")
    packages: dict[int, dict[str, dict]] = build_final_models(
        list(data["X_train"].columns), data["X_train"], data["y_train"], seeds, fronts, single, forward,
        everything, use_knee_point)
    run_robustness_suite(data["X_train"], data["y_train"], data["X_test"], data["y_test"], packages,
                         output_directory=arguments.out or os.path.join(run, "evaluation", "robustness"),
                         run_directory=run)


if __name__ == "__main__":
    main()
