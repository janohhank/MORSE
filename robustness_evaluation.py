"""Robustness suite: the final models of a run under test-set stresses that are generated automatically.

Every final model of every seed -- MORSE (the run's Pareto selection rule), the SO-GA, SFS and the
all-features model -- is scored on the test set under two kinds of stress (robustness_utils,
docs/robustness.md):

  re-weighted test populations  population shifts (leading principal components, typical vs atypical
                                rows; severity = the effective sample size left on the training rows) and
                                dependence shifts (pairs of correlated inputs weakened -- a share of the
                                correlation removed, 100% = uncorrelated -- or strengthened)
  corrupted test sets           measurement noise, recording noise of 0/1 inputs, under-recording, values
                                that are not available; one fixed bank of realisations for every model

No feature is named anywhere: the scenarios come from the training data by fixed rules
(robustness_config.RobustnessConfig), the same for every dataset. ROC-AUC is the primary metric (it does
not depend on the class prevalence); PR-AUC (average precision) is reported as well, under re-weighting
at the unshifted prevalence (robustness_utils.normalise_within_classes).

Statistics: the unit of replication is the GA run (seed). For every family and severity each model gets
one number -- the worst scenario / the mean over the pairs of the family's COHORT (re-weighting: the
scenarios usable at every level of the family, so every level compares the same scenarios) or the mean
over the corruption bank (corruption) -- and MORSE is compared with the SO-GA by a paired two-sided
Wilcoxon signed-rank test over the seeds, and with the deterministic SFS and all-features models by the
one-sample test against their single value.

Two ways to run it
  * in training_notebook.ipynb, right after training: `build_final_models` + `run_robustness_suite`;
  * later, on any checkpointed run, without retraining:
        python robustness_evaluation.py --run 2026-09-25_14-06-20_college_scorecard_pr
    The run's data are rebuilt by executing the configuration and data-loading cells of the notebook
    copy archived in the run folder (or of a copy given with --notebook), and checked against the run's
    checkpoint fingerprint.

Outputs (default: <run>/evaluation/robustness/; robustness_<rule>/ for a Pareto rule other than the run's)
  config.json                 the suite settings and the provenance of the evaluation
  schema.json                 the inferred feature schema
  models.csv                  every final model: size, sign consistency, clean ROC-AUC / PR-AUC
  scenarios.csv               every re-weighting scenario: level, strength, ESS (training / test / per
                              class), attainable / supported / usable / in the cohort, and what actually
                              moved (statistic, target and reached correlation, means, SDs)
  reweighting_scores.csv      every model under every re-weighting scenario
  corruption_scores.csv       every model under every corruption level and repetition
  corruption_diagnostics.csv  what every corrupted test set changed (and what had to be undone)
  model_family_scores.csv     every model per family and severity (mean over scenarios / repetitions)
  summary.csv                 per family, severity and method: mean over runs, SDs, change, worst case
  tests.csv                   MORSE against every baseline, per family and severity
  sign_vs_degradation.csv     sign consistency vs degradation of the GA models (exploratory)
  supplementary_per_level_*.csv  the re-weighting families over every scenario usable at each level
  report.txt                  the printed report
  robustness_*.png            five figures
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence

import numpy
import pandas
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

from checkpoint_utils import (FINGERPRINT_FILE, TrainingCheckpointStore, _library_versions, array_fingerprint,
                              atomic_write_json, atomic_write_text, read_json, source_fingerprint)
from evaluation_utils import (build_model_package, compute_marginal_correlations,
                              compute_model_sign_consistency, predict_scores)
import plot_utils
import robustness_utils
from plot_utils import (plot_corruption_curves, plot_dependence_shift_summary, plot_population_shift_curves,
                        plot_robustness_overview, plot_sign_vs_degradation)
from robustness_config import RobustnessConfig
from robustness_utils import (CorruptionBank, DependenceTilt, FeatureSchema, calibrate_correlation, calibrate_strengths,
                              class_effective_sizes, dependence_pairs, effective_sample_fraction,
                              exponential_tilt, infer_feature_schema, normalise_within_classes,
                              population_axes, weighted_average_precision, weighted_correlation,
                              weighted_roc_auc)
from run_manifest import (git_state, load_archived_run, load_run_data, manifest_sha256,  # noqa: F401
                          verify_training_data)
from training_utils import ensure_directory, repository_root, select_pareto_individual

METHODS: tuple[str, ...] = ("multi", "single", "forward", "all")
# Display name, colour, marker, line style -- the notebook's colours for the four methods.
METHOD_STYLES: dict[str, tuple[str, str, str, str]] = {
    "multi":   ("MORSE", "tab:blue", "o", "-"),
    "single":  ("SO-GA", "tab:orange", "s", "--"),
    "forward": ("SFS", "tab:red", "D", ":"),
    "all":     ("All features", "tab:green", "^", "-."),
}
REWEIGHTING_FAMILIES: tuple[str, ...] = ("population", "dependence_weaken", "dependence_strengthen")
FAMILY_LABELS: dict[str, str] = {
    "population": "Population shift",
    "dependence_weaken": "Dependence: weakened pairs",
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


def family_levels(family: str, config: RobustnessConfig) -> tuple[float, ...]:
    """The severity levels of a family, mildest first: the training ESS of a population shift, the share
    of the correlation removed / added by a dependence shift, the corruption level."""
    if family == "population":
        return tuple(config.ess_levels)
    if family == "dependence_weaken":
        return tuple(config.dependence_weaken_levels)
    if family == "dependence_strengthen":
        return tuple(config.dependence_strengthen_levels)
    return tuple(config.corruption_levels)


def headline_level(family: str, config: RobustnessConfig) -> float:
    """The severity at which a family enters the overview figure and the sign-consistency analysis."""
    return {"population": config.headline_ess, "dependence_weaken": config.headline_weaken,
            "dependence_strengthen": config.headline_strengthen}.get(family, config.headline_corruption_level)


def severity_text(family: str, level: float) -> str:
    """"60% ESS" (population), "r -100%" (the pair's correlation removed), "r +50%" (added), "0.5"."""
    if family == "population":
        return f"{level:.0%} ESS"
    if family == "dependence_weaken":
        return f"r -{100 * level:g}%"
    if family == "dependence_strengthen":
        return f"r +{100 * level:g}%"
    return f"{level:g}"


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
        use_knee_point: bool,
        record_directory: str | None = None) -> dict[int, dict[str, dict]]:
    """The final model of every method and seed, refit on the whole training set:
    `{seed: {"multi": package, "single": ..., "all": ..., "forward": ...}}`. MORSE is the individual of
    the seed's Pareto front picked by the pipeline's rule (`select_pareto_individual`).

    With `record_directory` (the run's evaluation folder), the fitted models are recorded there as
    final_models_<rule>.json the first time and checked against that record every later time
    (`record_final_models`), so a re-evaluation provably uses the same models."""
    packages: dict[int, dict[str, dict]] = {}
    for seed in seeds:
        morse = select_pareto_individual(pareto_fronts[seed], use_knee_point=use_knee_point)
        packages[seed] = {
            "multi":   build_model_package(morse, feature_names, X_train, y_train, seed=seed),
            "single":  build_model_package(single_best[seed], feature_names, X_train, y_train, seed=seed),
            "all":     build_model_package(all_masks[seed], feature_names, X_train, y_train, seed=seed),
            "forward": build_model_package(forward_masks[seed], feature_names, X_train, y_train, seed=seed),
        }
    if record_directory is not None:
        ensure_directory(record_directory)
        record_final_models(packages, final_models_path(record_directory, "knee" if use_knee_point else "max_s"))
    return packages


def final_models_path(evaluation_directory: str, rule: str) -> str:
    """Where the fitted final models of a Pareto rule are recorded."""
    return os.path.join(evaluation_directory, f"final_models_{rule}.json")


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
        log: Callable[[str], None] = print,
        selection_rule: str | None = None) -> RobustnessResults:
    """Evaluate the final models (`build_final_models`) under every automatically generated stress and
    write the tables, the figures and the report to `output_directory` (see the module docstring).

    X_train / y_train: the training data the models were fit on (the scenarios are derived from them);
    X_test / y_test:   the test set that is re-weighted and corrupted;
    run_directory:     the run folder, recorded in config.json together with its checkpoint fingerprint
                       and its run manifest;
    selection_rule:    MORSE's Pareto rule ("knee" / "max_s"), recorded in config.json.

    config.json is written first (status "running") and every stage's tables as soon as the stage is
    done, so an interrupted evaluation keeps what it finished; config.json says "complete" at the end.
    A folder that holds an evaluation of other data or of another rule is not overwritten.
    """
    config = config or RobustnessConfig()
    started: float = time.time()
    started_at: datetime = datetime.now()
    lines: list[str] = []

    def say(line: str = "") -> None:
        lines.append(line)
        log(line)

    features: list[str] = list(X_train.columns)
    X_test = X_test[features]
    y_tr: numpy.ndarray = numpy.asarray(y_train).astype(int)
    y_te: numpy.ndarray = numpy.asarray(y_test).astype(int)
    fingerprints: dict[str, Any] = data_fingerprints(X_train, y_tr, X_test, y_te)
    ensure_directory(output_directory)
    _refuse_other_evaluations(output_directory, fingerprints, selection_rule, say)

    def save(frames: dict[str, pandas.DataFrame]) -> None:
        for name, frame in frames.items():
            frame.to_csv(os.path.join(output_directory, name), index=False)

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

    atomic_write_json(os.path.join(output_directory, "config.json"),
                      _provenance(config, run_directory, fingerprints, models_frame, selection_rule, "running",
                                  started_at))
    save({"models.csv": models_frame})

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
        f"{summary_counts['other']} other.")
    say(f"Availability: {summary_counts['values_with_flag']} values with a flag "
        f"({summary_counts['values_paired_by_name']} paired by name, "
        f"{summary_counts['values_paired_statistically']} statistically); "
        f"{summary_counts['unresolved_availability']} unresolved (schema.json). "
        f"{summary_counts['unseen_combinations']} combinations of 0/1 values never seen in training "
        f"(a diagnostic: the corruptions may create them).")
    atomic_write_json(os.path.join(output_directory, "schema.json"), schema.to_dict())

    # ---- re-weighted populations
    scenarios, reweighting_scores, n_eligible_pairs, n_pairs = _reweighting(
        X_train, X_test, y_te, schema, models, clean, config)
    n_usable: int = int(scenarios["usable"].sum()) if not scenarios.empty else 0
    say(f"Re-weighting: {len(scenarios)} scenarios, {n_usable} usable (attainable on the training rows and "
        f"supported by enough effective rows); dependence pairs: {n_eligible_pairs} eligible, {n_pairs} used.")
    say(f"Cohorts -- the units usable at every level of their family, which the main analysis uses: "
        f"{_cohort_line(scenarios)}.")
    save({"scenarios.csv": scenarios, "reweighting_scores.csv": reweighting_scores})

    # ---- corrupted test sets
    corruption_scores, diagnostics, applicable, clean_violations = _corruption(
        X_train, X_test, y_te, schema, models, clean, config)
    say("Corruption families: " + (", ".join(applicable) if applicable else "none applicable") + ".")
    save({"corruption_scores.csv": corruption_scores, "corruption_diagnostics.csv": diagnostics})
    if clean_violations:
        say(f"NOTE: {clean_violations} clean test values do not hold their fill value although their availability "
            f"flag says 'not available'; they are left as they are.")
    if not diagnostics.empty:
        headline: pandas.DataFrame = diagnostics[numpy.isclose(diagnostics["level"], config.headline_corruption_level)]
        shares: pandas.Series = headline.groupby("family", sort=False)["rows_with_unseen_combination"].mean() / len(y_te)
        say(f"Test rows with a combination of 0/1 values never seen in training, at level "
            f"{config.headline_corruption_level:g} (diagnostic): "
            + ", ".join(f"{FAMILY_LABELS[family]} {share:.0%}" for family, share in shares.items()) + ".")

    # ---- summaries: every family over its cohort (the scenarios usable at every level of the family)
    all_family_scores: pandas.DataFrame = _model_family_scores(scenarios, reweighting_scores, corruption_scores)
    cohort_sizes: dict[str, int] = ({family: int(part.loc[part["in_cohort"], "unit"].nunique())
                                     for family, part in scenarios.groupby("family")} if not scenarios.empty else {})
    # a dependence family is summarised only if its cohort has enough pairs
    too_small: list[str] = [family for family in FAMILY_ORDER
                            if family.startswith("dependence_") and cohort_sizes.get(family, config.min_pairs)
                            < config.min_pairs]
    all_family_scores["summarised"] = (~all_family_scores["family"].isin(too_small)
                                       if not all_family_scores.empty else [])
    family_scores: pandas.DataFrame = all_family_scores[all_family_scores["summarised"].astype(bool)].drop(
        columns="summarised").reset_index(drop=True)
    skipped: pandas.DataFrame = pandas.DataFrame([{"family": family, "n_scenarios": cohort_sizes[family]}
                                                  for family in too_small])
    summary: pandas.DataFrame = _summary(family_scores, models_frame)
    tests: pandas.DataFrame = _tests(family_scores, scenarios, reweighting_scores)
    sign_points, sign_summary = _sign_analysis(models_frame, family_scores, config)
    # supplementary: the re-weighting families over every scenario usable at each level (a different
    # set of scenarios per level -- answers another question than the main analysis)
    per_level_scores: pandas.DataFrame = _model_family_scores(scenarios, reweighting_scores, pandas.DataFrame(),
                                                              cohort_only=False)
    per_level_summary: pandas.DataFrame = _summary(per_level_scores, models_frame)
    per_level_tests: pandas.DataFrame = _tests(per_level_scores, scenarios, reweighting_scores, cohort_only=False)

    # ---- files
    save({"model_family_scores.csv": all_family_scores, "summary.csv": summary, "tests.csv": tests,
          "sign_vs_degradation.csv": sign_summary, "sign_vs_degradation_points.csv": sign_points,
          "supplementary_per_level_summary.csv": per_level_summary,
          "supplementary_per_level_tests.csv": per_level_tests})

    _figures(output_directory, config, scenarios, reweighting_scores, corruption_scores, family_scores,
             models_frame, sign_points, sign_summary)

    for line in _report_lines(summary, tests, sign_summary, skipped, config):
        say(line)
    say(f"\nRobustness suite finished in {time.time() - started:.0f} s; outputs in {output_directory}")
    report: str = "\n".join(lines) + "\n"
    atomic_write_text(os.path.join(output_directory, "report.txt"), report)
    atomic_write_json(os.path.join(output_directory, "config.json"),
                      _provenance(config, run_directory, fingerprints, models_frame, selection_rule, "complete",
                                  started_at, time.time() - started))
    return RobustnessResults(output_directory=output_directory, schema=schema, models=models_frame,
                             scenarios=scenarios, model_family_scores=family_scores, summary=summary,
                             tests=tests, report=report)


# ---------------------------------------------------------------------------
# Re-weighting and corruption
# ---------------------------------------------------------------------------

def _reweighting(X_train: pandas.DataFrame, X_test: pandas.DataFrame, y_test: numpy.ndarray,
                 schema: FeatureSchema, models: list[_Model], clean: dict[str, numpy.ndarray],
                 config: RobustnessConfig) -> tuple[pandas.DataFrame, pandas.DataFrame, int, int]:
    """Every re-weighting scenario (scenarios.csv) and every model under it (reweighting_scores.csv).

    A scenario is `attainable` if its severity can be realised on the training rows, `supported` if the
    re-weighting leaves enough effective rows (training ESS, test ESS, per class), `usable` if both, and
    `in_cohort` if its unit -- a population axis and direction, or a dependence pair -- is usable at
    EVERY level of its family: the main summaries, tests and figures use exactly those scenarios."""
    scenario_rows: list[dict[str, Any]] = []
    score_rows: list[tuple] = []
    clean_roc: dict[str, float] = {key: weighted_roc_auc(y_test, p) for key, p in clean.items()}
    clean_ap: dict[str, float] = {key: weighted_average_precision(y_test, p) for key, p in clean.items()}

    def add(row: dict[str, Any], w_train: numpy.ndarray | None, w_test: numpy.ndarray | None) -> None:
        """Record a scenario and score every model under it; an unattainable scenario (its severity cannot
        be realised, so there are no weights) is only recorded."""
        if w_train is None or w_test is None:
            row.update({"train_ess": numpy.nan, "test_ess": numpy.nan, "ess_negatives": numpy.nan,
                        "ess_positives": numpy.nan, "supported": False, "weighted_prevalence": numpy.nan})
            scenario_rows.append(row)
            return
        negatives, positives = class_effective_sizes(w_test, y_test)
        train_ess: float = effective_sample_fraction(w_train)
        test_ess: float = effective_sample_fraction(w_test)
        row.update({
            "train_ess": train_ess, "test_ess": test_ess, "ess_negatives": negatives, "ess_positives": positives,
            "supported": bool(train_ess >= config.min_train_ess_fraction
                              and test_ess >= config.min_test_ess_fraction
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

    # population shifts: severity = the training ESS
    for axis in population_axes(X_train, X_test, config):
        for direction, sign in (("+", 1.0), ("-", -1.0)):
            calibrated = calibrate_strengths(
                lambda s: effective_sample_fraction(exponential_tilt(axis.train, sign * s)), config.ess_levels)
            for level, (strength, saturated) in zip(config.ess_levels, calibrated):
                row: dict[str, Any] = {
                    "scenario_id": f"population|{axis.name}|{direction}|{level:g}", "family": "population",
                    "scenario": axis.name, "unit": f"{axis.name} {direction}", "direction": direction,
                    "emphasis": axis.positive if sign > 0 else axis.negative, "level": level,
                    "strength": numpy.nan if saturated else sign * strength, "attainable": not saturated,
                    "description": axis.description}
                if saturated:
                    add(row, None, None)
                    continue
                w_train: numpy.ndarray = exponential_tilt(axis.train, sign * strength)
                w_test: numpy.ndarray = exponential_tilt(axis.test, sign * strength)
                row["statistic_shift"] = float(w_test @ axis.test / w_test.sum() - axis.test.mean())
                add(row, w_train, w_test)

    # dependence shifts: severity = the share of the pair's training (score) correlation removed / added
    pairs, n_eligible = dependence_pairs(X_train, X_test, schema, config)
    binary: set[str] = set(schema.binary)
    for pair in pairs:
        a_train, b_train = X_train[pair.a].to_numpy(dtype=float), X_train[pair.b].to_numpy(dtype=float)
        a_test, b_test = X_test[pair.a].to_numpy(dtype=float), X_test[pair.b].to_numpy(dtype=float)
        tilt: DependenceTilt = DependenceTilt(a_train, b_train, a_test, b_test,
                                              pair.a in binary, pair.b in binary, config.score_clip)
        score_correlation: float = tilt.score_correlation(0.0)
        clean_test_correlation: float = weighted_correlation(a_test, b_test)
        sd_a, sd_b = float(a_train.std()), float(b_train.std())
        for direction, levels, sign in (("weaken", config.dependence_weaken_levels, -1.0),
                                        ("strengthen", config.dependence_strengthen_levels, 1.0)):
            targets: list[float] = [score_correlation * (1.0 + sign * share) for share in levels]
            # a correlation cannot reach +-1; the targets beyond are unattainable without a search
            possible: int = next((i for i, target in enumerate(targets) if abs(target) >= 0.999), len(targets))
            calibrated = calibrate_correlation(tilt, targets[:possible]) if possible else []
            calibrated += [(float("nan"), False)] * (len(targets) - len(calibrated))
            for level, target, (strength, reached) in zip(levels, targets, calibrated):
                row = {"scenario_id": f"dependence|{pair.a} x {pair.b}|{direction}|{level:g}",
                       "family": f"dependence_{direction}", "scenario": f"{pair.a} x {pair.b}",
                       "unit": f"{pair.a} x {pair.b}", "direction": direction, "emphasis": direction, "level": level,
                       "strength": strength, "attainable": reached,
                       "pair_a": pair.a, "pair_b": pair.b, "pair_kind": pair.kind,
                       "correlation_train": pair.correlation, "correlation_test": clean_test_correlation,
                       "score_correlation_clean": score_correlation, "score_correlation_target": target}
                if not reached:
                    add(row, None, None)
                    continue
                w_train, w_test = tilt.weights(strength)
                shares: numpy.ndarray = w_test / w_test.sum()
                row.update({
                    "score_correlation_reached": weighted_correlation(
                        tilt._train_stats[:, 0], tilt._train_stats[:, 1], w_train),
                    "correlation_train_weighted": weighted_correlation(a_train, b_train, w_train),
                    "correlation_test_weighted": weighted_correlation(a_test, b_test, w_test),
                    "mean_shift_a": float((shares @ a_test - a_test.mean()) / sd_a),
                    "mean_shift_b": float((shares @ b_test - b_test.mean()) / sd_b),
                    "sd_ratio_a": _sd_ratio(a_test, shares),
                    "sd_ratio_b": _sd_ratio(b_test, shares),
                    "balance_error": tilt.balance_error})
                add(row, w_train, w_test)

    scenarios: pandas.DataFrame = pandas.DataFrame(scenario_rows)
    if not scenarios.empty:
        scenarios["usable"] = scenarios["attainable"].astype(bool) & scenarios["supported"].astype(bool)
        needed: dict[str, int] = {family: len(family_levels(family, config)) for family in REWEIGHTING_FAMILIES}
        counts: pandas.Series = scenarios[scenarios["usable"]].groupby(["family", "unit"])["level"].nunique()
        complete: set[tuple[str, str]] = {key for key, n in counts.items() if n == needed[key[0]]}
        scenarios["in_cohort"] = [(family, unit) in complete
                                  for family, unit in zip(scenarios["family"], scenarios["unit"])]
    scores: pandas.DataFrame = pandas.DataFrame(score_rows, columns=[
        "scenario_id", "method", "seed", "roc_auc", "pr_auc", "delta_roc_auc", "delta_pr_auc"])
    scores["seed"] = scores["seed"].astype("Int64")
    return scenarios, scores, n_eligible, len(pairs)


def _cohort_line(scenarios: pandas.DataFrame) -> str:
    """How many units every re-weighting family has, and how many of them are in its cohort."""
    if scenarios.empty:
        return "no re-weighting scenario"
    parts: list[str] = []
    names: dict[str, str] = {"population": "population axis-directions", "dependence_weaken": "weakened pairs",
                             "dependence_strengthen": "strengthened pairs"}
    for family in REWEIGHTING_FAMILIES:
        part: pandas.DataFrame = scenarios[scenarios["family"] == family]
        if part.empty:
            continue
        in_cohort: int = part.loc[part["in_cohort"], "unit"].nunique()
        parts.append(f"{names[family]} {in_cohort}/{part['unit'].nunique()}")
    return ", ".join(parts)
def _sd_ratio(values: numpy.ndarray, shares: numpy.ndarray) -> float:
    """SD of a column under the weights (`shares` sum to 1) relative to its unweighted SD."""
    unweighted: float = float(values.std())
    if unweighted == 0.0:
        return float("nan")
    return float(numpy.sqrt(shares @ (values - shares @ values) ** 2)) / unweighted


def _corruption(X_train: pandas.DataFrame, X_test: pandas.DataFrame, y_test: numpy.ndarray,
                schema: FeatureSchema, models: list[_Model], clean: dict[str, numpy.ndarray],
                config: RobustnessConfig) -> tuple[pandas.DataFrame, pandas.DataFrame, list[str], int]:
    """Every model on every corrupted test set of the bank: (scores, diagnostics per corrupted test set,
    the applicable families, the number of clean test cells that already break the availability rule)."""
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
    return scores, pandas.DataFrame(diagnostic_rows), applicable, bank.clean_availability_violations


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
    """Families in FAMILY_ORDER; within a family the mildest severity first (population: the highest
    ESS; the others: the lowest level); methods in METHODS order."""
    if frame.empty:
        return frame
    family_rank: pandas.Series = frame["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
    severity: pandas.Series = frame["severity"].astype(float)
    severity_rank: pandas.Series = severity.where(frame["family"] != "population", -severity)
    keys: dict[str, pandas.Series] = {"_family": family_rank, "_severity": severity_rank}
    if "method" in frame:
        keys["_method"] = frame["method"].map({m: i for i, m in enumerate(METHODS)})
    return (frame.assign(**keys).sort_values(list(keys), kind="stable")
            .drop(columns=list(keys)).reset_index(drop=True))


def _selected_scenarios(scenarios: pandas.DataFrame, cohort_only: bool) -> pandas.DataFrame:
    """The re-weighting scenarios a summary uses: the family's cohort (usable at every level), or -- for
    the supplementary per-level analysis -- every usable scenario of each level."""
    chosen: pandas.Series = scenarios["usable"] & (scenarios["in_cohort"] if cohort_only else True)
    return scenarios.loc[chosen, ["scenario_id", "family", "level"]]


def _model_family_scores(scenarios: pandas.DataFrame, reweighting_scores: pandas.DataFrame,
                         corruption_scores: pandas.DataFrame, cohort_only: bool = True) -> pandas.DataFrame:
    """One row per family, severity and model: re-weighting -> mean and minimum over the family's cohort
    (`cohort_only`) or over every usable scenario of the level; corruption -> mean over the repetitions,
    and their SD (the corruption variability)."""
    frames: list[pandas.DataFrame] = []
    if not scenarios.empty:
        merged: pandas.DataFrame = reweighting_scores.merge(_selected_scenarios(scenarios, cohort_only),
                                                            on="scenario_id")
        if not merged.empty:
            frames.append(merged.groupby(["family", "level", "method", "seed"], dropna=False).agg(
                roc_auc=("roc_auc", "mean"), delta_roc_auc=("delta_roc_auc", "mean"),
                pr_auc=("pr_auc", "mean"), delta_pr_auc=("delta_pr_auc", "mean"),
                worst_roc_auc=("roc_auc", "min"), worst_delta_roc_auc=("delta_roc_auc", "min"),
                n_scenarios=("roc_auc", "size")).reset_index().rename(columns={"level": "severity"}))
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
    difference is zero or there are fewer than two. The differences are rounded to 12 decimals first:
    ROC-AUC values are discrete (multiples of 1 / (m1 m0) on an unweighted test set), so exact ties and
    zeros between the differences are common, and floating-point noise of ~1e-17 would break them and
    decide between scipy's exact (no ties) and approximate (tie-corrected) distribution. On the College
    Scorecard run of 2026-09-27 that moved p for under-recording at level 0.1 from 0.126 to 0.133."""
    differences = numpy.round(differences[~numpy.isnan(differences)], 12)
    if differences.size < 2 or numpy.allclose(differences, 0.0):
        return float("nan")
    return float(wilcoxon(differences).pvalue)


def _tests(family_scores: pandas.DataFrame, scenarios: pandas.DataFrame,
           reweighting_scores: pandas.DataFrame, cohort_only: bool = True) -> pandas.DataFrame:
    """MORSE against every baseline, per family, severity and statistic (the per-run mean ROC-AUC, its
    change from the model's clean score, and for re-weighting the worst scenario and its change;
    `headline_statistic` says which one summarises a family). A positive difference means MORSE is
    better (higher score, smaller loss). Against the SO-GA: paired over the seeds; against SFS / all
    features: MORSE's seeds against the baseline's single value. For the re-weighting families the table
    also gives the share of the scenarios (the same ones as `family_scores`: the cohort, or every usable
    one) in which MORSE's run-averaged change is better than the SO-GA's (descriptive only: the scenarios
    share their test rows, so they are not independent)."""
    share: dict[tuple[str, float], tuple[float, int]] = {}
    if not scenarios.empty:
        merged: pandas.DataFrame = reweighting_scores.merge(_selected_scenarios(scenarios, cohort_only),
                                                            on="scenario_id")
        per_scenario: pandas.DataFrame = merged[merged["method"].isin(["multi", "single"])].groupby(
            ["family", "level", "scenario_id", "method"])["delta_roc_auc"].mean().unstack("method")
        if {"multi", "single"} <= set(per_scenario.columns):
            better: pandas.Series = (per_scenario["multi"] > per_scenario["single"]).groupby(
                level=["family", "level"]).agg(["mean", "size"])
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
                    "n_morse_better": int((numpy.round(differences, 12) > 0).sum()),   # a 1e-17 "win" is a tie
                    "p_value": _wilcoxon(differences)}
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
    per family (the family's headline statistic, at its headline severity), also given the model size K
    (partial Spearman). Exploratory: a correlation does not show that S causes the difference."""
    ga: pandas.DataFrame = models_frame[models_frame["method"].isin(["multi", "single"])][
        ["method", "seed", "sign_consistency", "n_features"]]
    points: list[pandas.DataFrame] = []
    rows: list[dict[str, Any]] = []
    present: set[str] = set(family_scores["family"]) if not family_scores.empty else set()
    for family in [f for f in FAMILY_ORDER if f in present]:
        severity: float = headline_level(family, config)
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

def _dependence_plot_data(scenarios: pandas.DataFrame, reweighting_scores: pandas.DataFrame,
                          config: RobustnessConfig) -> tuple[pandas.DataFrame, pandas.DataFrame, dict[str, int]]:
    """What the dependence figure shows -- computed from the same cohort as summary.csv and tests.csv:
    (curves: mean change over the cohort per run, then mean and SD across runs; differences: MORSE minus
    SO-GA per pair, each averaged over the runs; cohort size per direction). A direction whose cohort
    has fewer than `min_pairs` pairs is left out, as in the report."""
    cohort: pandas.DataFrame = scenarios[scenarios["family"].str.startswith("dependence_")
                                         & scenarios["usable"] & scenarios["in_cohort"]]
    sizes: dict[str, int] = cohort.groupby("direction")["unit"].nunique().to_dict()
    cohort = cohort[cohort["direction"].isin([d for d, size in sizes.items() if size >= config.min_pairs])]
    merged: pandas.DataFrame = reweighting_scores.merge(
        cohort[["scenario_id", "direction", "unit", "level"]], on="scenario_id")
    if merged.empty:
        return pandas.DataFrame(), pandas.DataFrame(), sizes
    per_run: pandas.DataFrame = merged.groupby(["direction", "level", "method", "seed"], dropna=False)[
        "delta_roc_auc"].mean().reset_index()
    curves: pandas.DataFrame = per_run.groupby(["direction", "level", "method"], dropna=False)["delta_roc_auc"].agg(
        mean="mean", sd=lambda v: v.std(ddof=1) if len(v) > 1 else numpy.nan).reset_index()
    per_pair: pandas.DataFrame = merged[merged["method"].isin(["multi", "single"])].groupby(
        ["direction", "level", "unit", "method"])["delta_roc_auc"].mean().unstack("method")
    differences: pandas.DataFrame = pandas.DataFrame()
    if {"multi", "single"} <= set(per_pair.columns):
        differences = (per_pair["multi"] - per_pair["single"]).rename("difference").reset_index()
    return curves, differences, sizes


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

    if not scenarios.empty:
        # population: one panel per axis, both directions, over the axis-directions usable at every level
        cohort: pandas.DataFrame = scenarios[(scenarios["family"] == "population") & scenarios["usable"]
                                             & scenarios["in_cohort"]]
        population: pandas.DataFrame = reweighting_scores.merge(
            cohort[["scenario_id", "scenario", "direction", "level"]], on="scenario_id")
        if not population.empty:
            rank: dict[float, int] = {level: i + 1 for i, level in enumerate(levels)}
            population["position"] = population["level"].map(rank) * population["direction"].map({"+": 1, "-": -1})
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
                                         "(axis-directions usable at every level; mean over runs, band = SD across runs)")

        # dependence: the same cohort as the report and the tests
        curves, differences, sizes = _dependence_plot_data(scenarios, reweighting_scores, config)
        if not curves.empty:
            plot_dependence_shift_summary(
                curves, differences, {"weaken": config.dependence_weaken_levels,
                                      "strengthen": config.dependence_strengthen_levels},
                styles, os.path.join(directory, "robustness_dependence.png"),
                f"Dependence shifts: change of test ROC-AUC over the pairs usable at every level "
                f"({sizes.get('weaken', 0)} weakened, {sizes.get('strengthen', 0)} strengthened; a direction with "
                f"fewer than {config.min_pairs} is left out)\nmean over runs, band = SD across runs")

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
            severity: float = headline_level(family, config)
            score, _, description = headline_statistic(family)
            overview_labels[family] = f"{FAMILY_LABELS[family]} ({description}, {severity_text(family, severity)})"
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
    lines.append("less), Wilcoxon p, and the runs in which MORSE is better. n = the family's cohort -- the scenarios")
    lines.append("usable at EVERY level, the same at each level (re-weighting) -- or the corrupted test sets per level.")
    lines.append("Severity: population = training ESS; dependence = share of the pair's correlation removed (r -) or")
    lines.append("added (r +); corruption = level. clean ROC-AUC: "
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
        lines.append(f"{FAMILY_LABELS.get(family, family):32s} {severity_text(family, severity):>8s} "
                     f"{int(part['n_scenarios'].max()):>4d} " + " ".join(f"{cell:>14s}" for cell in cells) + " "
                     + comparison)
    if not skipped.empty:
        for family, part in skipped.groupby("family", sort=False):
            lines.append(f"{FAMILY_LABELS.get(family, family):32s} not summarised: only {int(part['n_scenarios'].max())} "
                         f"pairs are usable at every level (at least {config.min_pairs} needed)")
    lines.append("Supplementary (not the main analysis): every scenario usable at each level, a different set per")
    lines.append("level -- supplementary_per_level_summary.csv / supplementary_per_level_tests.csv.")
    if not sign_summary.empty:
        lines.append("\nSign consistency vs change of ROC-AUC across the GA models (headline severity; exploratory):")
        for _, row in sign_summary.iterrows():
            lines.append(f"  {FAMILY_LABELS.get(row['family'], row['family']):32s} Spearman {row['spearman']:+.2f}, "
                         f"given the model size {row['partial_spearman_given_size']:+.2f} (n = {row['n_models']})")
    return lines


def data_fingerprints(X_train: pandas.DataFrame, y_train: Sequence[float], X_test: pandas.DataFrame,
                      y_test: Sequence[float]) -> dict[str, Any]:
    """Fingerprints of the evaluated data (inputs in the training column order), as run_manifest.json
    records them."""
    features: list[str] = list(X_train.columns)

    def fingerprint(values: Any) -> dict[str, Any]:
        return array_fingerprint(numpy.ascontiguousarray(numpy.asarray(values, dtype=numpy.float64)))

    return {"X_train": fingerprint(X_train.to_numpy(dtype=numpy.float64)), "y_train": fingerprint(y_train),
            "X_test": fingerprint(X_test[features].to_numpy(dtype=numpy.float64)), "y_test": fingerprint(y_test)}


def _provenance(config: RobustnessConfig, run_directory: str | None, fingerprints: dict[str, Any],
                models_frame: pandas.DataFrame, selection_rule: str | None, status: str,
                started: datetime, runtime_seconds: float | None = None) -> dict[str, Any]:
    training_fingerprint: str | None = None
    if run_directory:
        path: str = os.path.join(run_directory, "checkpoints", "training", FINGERPRINT_FILE)
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                training_fingerprint = hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()
    shape: tuple[int, ...] = tuple(fingerprints["X_train"]["shape"])
    return {
        "status": status,
        "created": started.isoformat(timespec="seconds"),
        "runtime_seconds": None if runtime_seconds is None else round(runtime_seconds, 1),
        "config": config.to_dict(),
        "run_directory": run_directory,
        "selection_rule": selection_rule,
        "training_fingerprint_sha256": training_fingerprint,
        "run_manifest_sha256": manifest_sha256(run_directory) if run_directory else None,
        "data": {"train_rows": int(shape[0]), "test_rows": int(fingerprints["X_test"]["shape"][0]),
                 "inputs": int(shape[1]) if len(shape) > 1 else 0, "fingerprints": fingerprints},
        "models": {method: int((models_frame["method"] == method).sum()) for method in METHODS},
        "seeds": sorted(int(s) for s in models_frame["seed"].dropna().unique()),
        "git": git_state(),
        "source_sha256": source_fingerprint((robustness_utils, plot_utils, sys.modules[__name__],
                                             sys.modules[RobustnessConfig.__module__])),
    }


def _refuse_other_evaluations(output_directory: str, fingerprints: dict[str, Any], selection_rule: str | None,
                              say: Callable[[str], None]) -> None:
    """An output folder that already holds an evaluation of other data or of another Pareto rule is not
    overwritten (the same evaluation repeated is fine)."""
    path: str = os.path.join(output_directory, "config.json")
    if not os.path.isfile(path):
        return
    previous: dict[str, Any] = read_json(path)
    previous_fingerprints: Any = previous.get("data", {}).get("fingerprints")
    previous_rule: Any = previous.get("selection_rule")
    problems: list[str] = []
    if previous_fingerprints is not None and previous_fingerprints != fingerprints:
        problems.append("other data")
    if previous_rule is not None and selection_rule is not None and previous_rule != selection_rule:
        problems.append(f"the Pareto rule {previous_rule!r} (now {selection_rule!r})")
    if problems:
        raise FileExistsError(f"{output_directory} holds an evaluation of {' and of '.join(problems)}; choose "
                              f"another output folder (--out) or remove that one")
    if previous_fingerprints is None:
        say(f"NOTE: {output_directory} holds an evaluation made before the data were recorded; it is replaced.")


def record_final_models(packages: dict[int, dict[str, dict]], path: str, tolerance: float = 1e-8) -> str:
    """Record the fitted final models -- inputs, scaler, coefficients, intercept of every method and seed --
    in `path` the first time, and check them against the record every later time: a re-evaluation refits
    the models from the checkpointed masks, and a newer library or code could fit them differently. Returns
    the record's SHA-256. Raises RuntimeError when a refit differs by more than `tolerance`."""
    models: dict[str, dict[str, Any]] = {}
    for seed in sorted(packages):
        for method, package in packages[seed].items():
            models[f"{method}/{seed}"] = {
                "features": list(package["features"]),
                "scaler_mean": package["scaler"].mean_.tolist(), "scaler_scale": package["scaler"].scale_.tolist(),
                "coef": package["model"].coef_.ravel().tolist(), "intercept": float(package["model"].intercept_[0])}
    record: dict[str, Any] = {"schema_version": 1, "libraries": _library_versions(), "models": models}
    if os.path.isfile(path):
        stored: dict[str, Any] = read_json(path)
        problems: list[str] = []
        for key, model in models.items():
            before: dict[str, Any] | None = stored.get("models", {}).get(key)
            if before is None:
                continue                            # a seed evaluated for the first time: added below
            if before["features"] != model["features"]:
                problems.append(f"{key}: other inputs")
                continue
            difference: float = max(float(numpy.max(numpy.abs(numpy.subtract(before[part], model[part]), dtype=float),
                                                    initial=0.0))
                                    for part in ("scaler_mean", "scaler_scale", "coef", "intercept"))
            if difference > tolerance:
                problems.append(f"{key}: parameters differ by up to {difference:.2g}")
        if problems:
            raise RuntimeError(f"the final models refit now differ from those recorded in {path} ("
                               + "; ".join(problems[:5]) + ("; ..." if len(problems) > 5 else "")
                               + "): the library versions or the code changed since. Remove the file to "
                                 "accept the new models.")
        record["models"] = {**stored.get("models", {}), **models}
        record["libraries"] = stored.get("libraries", record["libraries"])
    atomic_write_json(path, record)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()


# ---------------------------------------------------------------------------
# Stand-alone use on a finished run
# ---------------------------------------------------------------------------

@dataclass
class RunModels:
    """A finished run, rebuilt without retraining: its data and the final model of every method and seed."""
    directory: str
    X_train: pandas.DataFrame
    y_train: pandas.Series
    X_test: pandas.DataFrame
    y_test: pandas.Series
    packages: dict[int, dict[str, dict]]
    seeds: list[int]
    use_knee_point: bool        # MORSE's Pareto rule used here ...
    own_rule: str               # ... and the run's own rule ("knee" / "max_s")
    use_roc_auc: bool           # the run's main objective
    fingerprint: dict[str, Any]
    source: str = ""            # where the data came from: run manifest / archived notebook / notebook copy

    @property
    def rule(self) -> str:
        return "knee" if self.use_knee_point else "max_s"

    @property
    def folder_suffix(self) -> str:
        """"" for the run's own rule, "_knee" / "_max_s" otherwise: an evaluation with another rule gets
        its own output folder instead of overwriting the run's."""
        return "" if self.rule == self.own_rule else f"_{self.rule}"


def load_run_models(run: str, selection: str = "auto", seeds: Sequence[int] | None = None,
                    notebook: str | None = None, say: Callable[[str], None] = print) -> RunModels:
    """Rebuild a checkpointed run: its data (from the run manifest, the archived notebook or the notebook
    copy `notebook`; checked against the checkpoint fingerprint) and the final models of `seeds` (default:
    every seed with a complete checkpoint), recorded / checked in the run's evaluation folder. `selection`
    picks MORSE's Pareto solution: "auto" = the run's own rule, or "knee" / "max_s"."""
    directory: str = os.path.normpath(run if os.path.isabs(run) else os.path.join(repository_root(), run))
    data: dict[str, Any] = load_run_data(directory, notebook)
    fingerprint: dict[str, Any] = verify_training_data(directory, data["X_train"], data["y_train"])
    store: TrainingCheckpointStore = TrainingCheckpointStore(os.path.join(directory, "checkpoints", "training"),
                                                             fingerprint)
    chosen: list[int] = sorted(seeds) if seeds else store.completed_seeds()
    fronts, single, forward, everything = store.load_all(chosen)
    use_knee_point: bool = data["use_knee_point"] if selection == "auto" else selection == "knee"
    say(f"Run {directory} (data from its {data['source']}): {len(chosen)} seeds; MORSE = the "
        f"{'knee point' if use_knee_point else 'max-S end'} of every Pareto front.")
    packages: dict[int, dict[str, dict]] = build_final_models(
        list(data["X_train"].columns), data["X_train"], data["y_train"], chosen, fronts, single, forward,
        everything, use_knee_point, record_directory=os.path.join(directory, "evaluation"))
    use_roc_auc: Any = data.get("use_roc_auc")
    if use_roc_auc is None:
        use_roc_auc = fingerprint.get("settings", {}).get("training_config", {}).get("use_roc_auc", True)
    return RunModels(directory=directory, X_train=data["X_train"], y_train=data["y_train"], X_test=data["X_test"],
                     y_test=data["y_test"], packages=packages, seeds=chosen, use_knee_point=use_knee_point,
                     own_rule="knee" if data["use_knee_point"] else "max_s", use_roc_auc=bool(use_roc_auc),
                     fingerprint=fingerprint, source=data["source"])


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the robustness suite on the final models of a checkpointed run (no retraining).")
    parser.add_argument("--run", required=True,
                        help="the run's result folder (absolute, or relative to the repository root)")
    parser.add_argument("--selection", choices=("auto", "knee", "max_s"), default="auto",
                        help="which Pareto solution is MORSE's final model; auto = the run's own rule")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="evaluate only these seeds (default: every seed with a complete checkpoint)")
    parser.add_argument("--notebook", default=None,
                        help="a copy of training_notebook.ipynb to rebuild the data from, for a run folder without "
                             "an archived copy of the notebook (checked against the checkpoints)")
    parser.add_argument("--out", default=None,
                        help="output folder (default: <run>/evaluation/robustness, or robustness_<rule> for a "
                             "rule other than the run's own)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    arguments: argparse.Namespace = parse_arguments(argv)
    run: RunModels = load_run_models(arguments.run, arguments.selection, arguments.seeds, arguments.notebook)
    run_robustness_suite(run.X_train, run.y_train, run.X_test, run.y_test, run.packages,
                         output_directory=arguments.out or os.path.join(run.directory, "evaluation",
                                                                        "robustness" + run.folder_suffix),
                         run_directory=run.directory, selection_rule=run.rule)


if __name__ == "__main__":
    main()
