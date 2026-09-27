"""Tests of robustness_evaluation: the final models, the whole suite on synthetic data, the aggregation
(corruption repetitions are averaged within a run before runs are compared), and the stand-alone loading
of a finished run (archived notebook + checkpoint fingerprint).

Run from the repository root:

    python -m unittest discover -s tests -v
"""
import contextlib
import io
import runpy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from unittest import mock

import numpy
import pandas
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from deap import creator  # noqa: E402

import robustness_evaluation as evaluation  # noqa: E402
import run_manifest  # noqa: E402
from checkpoint_utils import (SeedTrainingResult, TrainingCheckpointStore, atomic_write_json,  # noqa: E402
                              build_training_fingerprint)
from deap_types import ensure_multi_objective_types, ensure_single_objective_types  # noqa: E402
from evaluation_utils import build_model_package, predict_scores  # noqa: E402
from robustness_config import RobustnessConfig  # noqa: E402
from robustness_evaluation import (_model_family_scores, _summary, build_final_models,  # noqa: E402
                                   load_archived_run, run_robustness_suite, verify_training_data)
from test_robustness_utils import make_data  # noqa: E402
from training_config import TrainingConfig  # noqa: E402

SEEDS = [1, 2, 3]


def masks(n_features: int) -> dict[str, dict[int, list[int]]]:
    rng = numpy.random.default_rng(12)
    per_seed = {method: {seed: list((rng.random(n_features) < 0.6).astype(int)) for seed in SEEDS}
                for method in ("multi", "single")}
    fixed = {"forward": list((rng.random(n_features) < 0.4).astype(int)), "all": [1] * n_features}
    for method, mask in fixed.items():
        per_seed[method] = {seed: mask for seed in SEEDS}
    return per_seed


class SuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.y_train = make_data(900, seed=21)
        cls.X_test, cls.y_test = make_data(400, seed=22)
        features = list(cls.X_train.columns)
        chosen = masks(len(features))
        cls.packages = {seed: {method: build_model_package(chosen[method][seed], features, cls.X_train,
                                                           pandas.Series(cls.y_train), seed=seed)
                               for method in ("multi", "single", "all", "forward")} for seed in SEEDS}
        cls.config = RobustnessConfig(ess_levels=(0.8,), headline_ess=0.8, corruption_levels=(0.5, 1.0),
                                      headline_corruption_level=0.5, corruption_repetitions=2, max_pairs=5,
                                      min_pairs=1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.results = run_robustness_suite(cls.X_train, cls.y_train, cls.X_test, cls.y_test, cls.packages,
                                           cls.directory.name, config=cls.config, log=lambda line: None)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_every_output_is_written(self):
        for name in ("config.json", "schema.json", "models.csv", "scenarios.csv", "reweighting_scores.csv",
                     "corruption_scores.csv", "corruption_diagnostics.csv", "model_family_scores.csv",
                     "summary.csv", "tests.csv", "sign_vs_degradation.csv", "report.txt",
                     "robustness_population.png", "robustness_dependence.png", "robustness_corruption.png",
                     "robustness_overview.png", "robustness_sign_vs_degradation.png"):
            self.assertTrue(os.path.isfile(os.path.join(self.directory.name, name)), name)
        with open(os.path.join(self.directory.name, "config.json"), encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["config"]["corruption_repetitions"], 2)
        diagnostics = pandas.read_csv(os.path.join(self.directory.name, "corruption_diagnostics.csv"))
        self.assertTrue({"eligible", "selected", "changed", "unseen_combinations",
                         "rows_with_unseen_combination"} <= set(diagnostics.columns))
        self.assertIn("Availability: 1 values with a flag (1 paired by name, 0 statistically); 0 unresolved",
                      self.results.report)
        self.assertIn("never seen in training, at level 0.5 (diagnostic): Measurement noise 0%", self.results.report)

    def test_models_and_clean_scores(self):
        models = self.results.models
        self.assertEqual(models["method"].value_counts().to_dict(), {"multi": 3, "single": 3, "forward": 1, "all": 1})
        self.assertTrue(models.loc[models["method"].isin(["forward", "all"]), "seed"].isna().all())
        for _, row in models[models["method"] == "multi"].iterrows():
            expected = roc_auc_score(self.y_test, predict_scores(self.packages[int(row["seed"])]["multi"], self.X_test))
            self.assertAlmostEqual(row["clean_roc_auc"], expected, places=12)

    def test_scenarios_and_families(self):
        scenarios = self.results.scenarios
        self.assertEqual(int((scenarios["family"] == "population").sum()), 8)       # 4 axes x 2 directions x 1 level
        self.assertLessEqual(int(scenarios["family"].str.startswith("dependence_").sum()), 5 * 8)   # pairs x levels
        families = set(self.results.summary["family"])
        self.assertTrue({"population", "gaussian_noise", "binary_redraw", "under_recording",
                         "value_masking"} <= families)
        corruption = pandas.read_csv(os.path.join(self.directory.name, "corruption_scores.csv"))
        self.assertEqual(len(corruption), 4 * 2 * 2 * 8)     # families x levels x repetitions x models

    def test_tests_compare_morse_with_every_baseline(self):
        tests = self.results.tests
        self.assertEqual(set(tests["baseline"]), {"single", "forward", "all"})
        self.assertTrue((tests["n_runs"] == 3).all())


class AggregationTests(unittest.TestCase):
    def test_repetitions_are_averaged_within_a_run_before_runs_are_compared(self):
        rows = []
        for seed, offset in ((1, 0.00), (2, 0.10)):
            for repetition, noise in enumerate((-0.05, 0.05)):
                rows.append({"family": "gaussian_noise", "level": 0.5, "repetition": repetition, "method": "multi",
                             "seed": seed, "roc_auc": 0.8 + offset + noise, "pr_auc": 0.7,
                             "delta_roc_auc": offset + noise, "delta_pr_auc": 0.0})
        corruption = pandas.DataFrame(rows)
        corruption["seed"] = corruption["seed"].astype("Int64")
        family_scores = _model_family_scores(pandas.DataFrame(), pandas.DataFrame(), corruption)
        self.assertEqual(len(family_scores), 2)
        numpy.testing.assert_allclose(sorted(family_scores["roc_auc"]), [0.8, 0.9])
        models = pandas.DataFrame({"method": ["multi", "multi"], "seed": [1, 2],
                                   "clean_roc_auc": [0.8, 0.9], "clean_pr_auc": [0.7, 0.7]})
        summary = _summary(family_scores, models)
        # SD across runs of the run means (0.8, 0.9), not of the four single corrupted test sets
        self.assertAlmostEqual(summary["roc_auc_sd"].iloc[0], numpy.std([0.8, 0.9], ddof=1), places=12)
        self.assertAlmostEqual(summary["corruption_sd"].iloc[0], numpy.std([-0.05, 0.05], ddof=1), places=12)


class FinalModelTests(unittest.TestCase):
    def test_build_final_models_uses_the_selection_rule(self):
        ensure_multi_objective_types()
        ensure_single_objective_types()
        X, y = make_data(300, seed=31)
        features = list(X.columns)
        n = len(features)
        low_s = creator.Individual([1] * n)
        low_s.fitness.values = (0.9, 0.5)
        high_s = creator.Individual([1] * (n // 2) + [0] * (n - n // 2))
        high_s.fitness.values = (0.8, 0.9)
        single = creator.IndividualSingle([0] * (n - 2) + [1, 1])
        single.fitness.values = (0.85,)
        packages = build_final_models(features, X, pandas.Series(y), [7], {7: [low_s, high_s]}, {7: single},
                                      {7: [1] + [0] * (n - 1)}, {7: [1] * n}, use_knee_point=False)
        self.assertEqual(set(packages[7]), {"multi", "single", "all", "forward"})
        self.assertEqual(packages[7]["multi"]["features"], features[: n // 2])      # the max-S end
        self.assertEqual(packages[7]["single"]["features"], features[-2:])


class StandAloneTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        X, y = make_data(200, seed=41)
        frame = X.assign(label=y)
        self.train_csv = os.path.join(self.directory.name, "train.csv")
        self.test_csv = os.path.join(self.directory.name, "test.csv")
        frame.iloc[:150].to_csv(self.train_csv, index=False)
        frame.iloc[150:].to_csv(self.test_csv, index=False)
        cells = [
            {"cell_type": "code", "metadata": {}, "source": ["import os\n"], "outputs": [], "execution_count": None},
            {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": [
                "#TARGET_COLUMN: str = 'something else'\n",
                f"CSV_TRAIN_PATH: str = {self.train_csv!r}\n",
                f"CSV_TEST_PATH: str = {self.test_csv!r}\n",
                "TARGET_COLUMN: str = 'label'\n",
                "USE_KNEE_POINT_SELECTION: bool = False\n"]},
            {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": [
                "df_train: pandas.DataFrame = pandas.read_csv(CSV_TRAIN_PATH)\n",
                "df_train = df_train.drop(columns=['med_b'])\n"]},
            {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": [
                "df_test = pandas.read_csv(CSV_TEST_PATH).drop(columns=['med_b'])\n",
                "y_test: pandas.Series = df_test[TARGET_COLUMN]\n",
                "X_test: pandas.DataFrame = df_test.drop(columns=[TARGET_COLUMN])\n"]},
            {"cell_type": "code", "metadata": {}, "source": ["raise RuntimeError('training must not run')\n"],
             "outputs": [], "execution_count": None},
        ]
        with open(os.path.join(self.directory.name, "training_notebook.ipynb"), "w", encoding="utf-8") as handle:
            json.dump({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, handle)

    def tearDown(self):
        self.directory.cleanup()

    def test_a_run_folder_without_a_usable_notebook_is_refused(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(FileNotFoundError, "nor an archived copy"):
                load_archived_run(empty)
            cells = [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                      "source": ["X_test = None\n"]}]
            with open(os.path.join(empty, "training_notebook.ipynb"), "w", encoding="utf-8") as handle:
                json.dump({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, handle)
            with self.assertRaisesRegex(ValueError, "no configuration cell"):
                load_archived_run(empty)

    def test_the_script_entry_point(self):
        path = os.path.join(os.path.dirname(HERE), "robustness_evaluation.py")
        with mock.patch.object(sys, "argv", [path, "--help"]):
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                with self.assertRaises(SystemExit) as stopped:
                    runpy.run_path(path, run_name="__main__")
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("--run", printed.getvalue())

    def test_the_archived_notebook_rebuilds_the_data_and_the_fingerprint_is_checked(self):
        data = load_archived_run(self.directory.name)
        self.assertFalse(data["use_knee_point"])
        self.assertEqual(len(data["X_train"]), 150)
        self.assertEqual(len(data["X_test"]), 50)
        self.assertNotIn("med_b", data["X_train"].columns)
        self.assertNotIn("label", data["X_train"].columns)

        X = numpy.ascontiguousarray(data["X_train"].to_numpy(), dtype=numpy.float64)
        y = numpy.ascontiguousarray(data["y_train"].to_numpy(), dtype=numpy.float64)
        fingerprint = build_training_fingerprint(
            config=TrainingConfig(seed=0), cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            feature_names=list(data["X_train"].columns), X_train=StandardScaler().fit_transform(X), y_train=y)
        atomic_write_json(os.path.join(self.directory.name, "checkpoints", "training", "fingerprint.json"),
                          fingerprint)
        verify_training_data(self.directory.name, data["X_train"], data["y_train"])
        changed = data["X_train"].copy()
        changed.iloc[0, 0] += 1.0
        with self.assertRaises(ValueError):
            verify_training_data(self.directory.name, changed, data["y_train"])
        with self.assertRaises(ValueError):
            verify_training_data(self.directory.name, data["X_train"].iloc[:, ::-1], data["y_train"])


def write_run_folder(directory: str) -> dict:
    """A finished, checkpointed run: an archived notebook whose data cells read two CSVs, the checkpoint
    fingerprint of those data, and two seeds with a three-point Pareto front each."""
    X, y = make_data(260, seed=51)
    frame = X.assign(label=y)
    train_csv, test_csv = os.path.join(directory, "train.csv"), os.path.join(directory, "test.csv")
    frame.iloc[:180].to_csv(train_csv, index=False)
    frame.iloc[180:].to_csv(test_csv, index=False)
    cells = [
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": [
            f"CSV_TRAIN_PATH: str = {train_csv!r}\n", f"CSV_TEST_PATH: str = {test_csv!r}\n",
            "TARGET_COLUMN: str = 'label'\n", "USE_KNEE_POINT_SELECTION: bool = False\n"]},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": [
            "df_train: pandas.DataFrame = pandas.read_csv(CSV_TRAIN_PATH)\n",
            "df_test = pandas.read_csv(CSV_TEST_PATH)\n",
            "y_test: pandas.Series = df_test[TARGET_COLUMN]\n",
            "X_test: pandas.DataFrame = df_test.drop(columns=[TARGET_COLUMN])\n"]}]
    with open(os.path.join(directory, "training_notebook.ipynb"), "w", encoding="utf-8") as handle:
        json.dump({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, handle)

    data = load_archived_run(directory)
    features = list(data["X_train"].columns)
    X_search = StandardScaler().fit_transform(numpy.ascontiguousarray(data["X_train"].to_numpy(), dtype=numpy.float64))
    fingerprint = build_training_fingerprint(
        config=TrainingConfig(seed=0), cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        feature_names=features, X_train=X_search,
        y_train=numpy.ascontiguousarray(data["y_train"].to_numpy(), dtype=numpy.float64))
    store = TrainingCheckpointStore(os.path.join(directory, "checkpoints", "training"), fingerprint)
    store.prepare()
    ensure_multi_objective_types()
    ensure_single_objective_types()
    n = len(features)
    for seed in (1, 2):
        front = []
        for size, (auc, sign) in ((n, (0.90, 0.60)), (8, (0.85, 0.80)), (4 + seed, (0.80, 0.95))):
            individual = creator.Individual([1] * size + [0] * (n - size))
            individual.fitness.values = (auc, sign)
            front.append(individual)
        single = creator.IndividualSingle([1, 0] * (n // 2) + [1] * (n % 2))
        single.fitness.values = (0.88,)
        store.save_seed(SeedTrainingResult(seed, front, single, [1, 1, 1] + [0] * (n - 3), [1] * n))
    return {"features": features, "n": n}


class CommandLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.run_directory = os.path.join(cls.temporary.name, "2026-01-01_00-00-00_run")
        os.makedirs(cls.run_directory)
        cls.info = write_run_folder(cls.run_directory)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_main(self, *arguments: str) -> str:
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            evaluation.main(["--run", self.run_directory, *arguments])
        return printed.getvalue()

    def test_a_finished_run_is_evaluated_from_its_checkpoints(self):
        out = os.path.join(self.temporary.name, "out_auto")
        printed = self.run_main("--out", out)
        self.assertIn("MORSE = the max-S end", printed)                 # the archived notebook's rule
        self.assertIn("Robustness suite finished", printed)
        models = pandas.read_csv(os.path.join(out, "models.csv"))
        # max-S end of seed s has 4 + s inputs
        morse = models[models["method"] == "multi"].set_index("seed")["n_features"].to_dict()
        self.assertEqual(morse, {1: 5, 2: 6})
        with open(os.path.join(out, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        self.assertEqual(os.path.normpath(config["run_directory"]), os.path.normpath(self.run_directory))
        self.assertIsNotNone(config["training_fingerprint_sha256"])
        self.assertEqual(config["seeds"], [1, 2])

    def test_the_selection_and_the_seeds_can_be_chosen(self):
        out = os.path.join(self.temporary.name, "out_knee")
        printed = self.run_main("--out", out, "--selection", "knee", "--seeds", "2")
        self.assertIn("1 seeds; MORSE = the knee point", printed)
        models = pandas.read_csv(os.path.join(out, "models.csv"))
        self.assertEqual(models.loc[models["method"] == "multi", "seed"].tolist(), [2])
        self.assertEqual(models.loc[models["method"] == "multi", "n_features"].tolist(), [8])   # the knee point

    def test_the_default_output_folder_is_inside_the_run(self):
        self.run_main("--seeds", "1")
        self.assertTrue(os.path.isfile(os.path.join(self.run_directory, "evaluation", "robustness", "report.txt")))

    def test_another_rule_gets_its_own_default_folder(self):
        self.run_main("--seeds", "1", "--selection", "knee")
        self.assertTrue(os.path.isfile(os.path.join(self.run_directory, "evaluation", "robustness_knee", "report.txt")))

    def test_a_run_with_a_manifest_needs_no_notebook(self):
        other = os.path.join(self.temporary.name, "run_with_manifest")
        os.makedirs(other)
        write_run_folder(other)
        data = load_archived_run(other)
        run_manifest.write_run_manifest(other, run_manifest.build_run_manifest(
            [[os.path.join(other, "train.csv")]], os.path.join(other, "test.csv"), "label", data["X_train"],
            data["y_train"], data["X_test"], data["y_test"], use_roc_auc=False, use_knee_point=True),
            log=lambda line: None)
        os.remove(os.path.join(other, "training_notebook.ipynb"))
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            run = evaluation.load_run_models(other, seeds=[1])
        self.assertIn("data from its run manifest", printed.getvalue())
        self.assertEqual((run.source, run.own_rule, run.use_roc_auc), ("run manifest", "knee", False))
        self.assertEqual(run.packages[1]["multi"]["features"], self.info["features"][:8])   # the knee point
        self.assertTrue(os.path.isfile(os.path.join(other, "evaluation", "final_models_knee.json")))

    def test_a_notebook_copy_can_stand_in_for_a_missing_archived_one(self):
        other = os.path.join(self.temporary.name, "run_without_notebook")
        os.makedirs(other)
        write_run_folder(other)
        copy = os.path.join(self.temporary.name, "notebook_copy.ipynb")
        os.replace(os.path.join(other, "training_notebook.ipynb"), copy)
        with self.assertRaisesRegex(FileNotFoundError, "--notebook"):
            evaluation.load_run_models(other, say=lambda line: None)
        run = evaluation.load_run_models(other, "max_s", [2], notebook=copy, say=lambda line: None)
        self.assertEqual((run.rule, run.own_rule, run.folder_suffix, run.seeds, run.use_roc_auc),
                         ("max_s", "max_s", "", [2], True))
        self.assertEqual(run.packages[2]["multi"]["features"], self.info["features"][:6])     # max-S end: 4 + seed
        knee = evaluation.load_run_models(other, "knee", [2], notebook=copy, say=lambda line: None)
        self.assertEqual((knee.rule, knee.folder_suffix), ("knee", "_knee"))

    def test_other_data_are_refused(self):
        other = os.path.join(self.temporary.name, "other_run")
        os.makedirs(other)
        write_run_folder(other)
        fingerprint_path = os.path.join(other, "checkpoints", "training", "fingerprint.json")
        with open(fingerprint_path, encoding="utf-8") as handle:
            fingerprint = json.load(handle)
        fingerprint["settings"]["data"]["y_train"]["sha256"] = "0" * 64
        atomic_write_json(fingerprint_path, fingerprint)
        with self.assertRaisesRegex(ValueError, "do not match the checkpoints"):
            evaluation.main(["--run", other])


class HelperTests(unittest.TestCase):
    def test_headline_statistics(self):
        self.assertEqual(evaluation.headline_statistic("population")[:2], ("worst_roc_auc", "worst_delta_roc_auc"))
        self.assertEqual(evaluation.headline_statistic("dependence_strengthen")[2], "mean over pairs")
        self.assertEqual(evaluation.headline_statistic("gaussian_noise")[:2], ("roc_auc", "delta_roc_auc"))

    def test_small_helpers(self):
        self.assertTrue(numpy.isnan(evaluation._sd_ratio(numpy.ones(5), numpy.full(5, 0.2))))
        self.assertAlmostEqual(evaluation._sd_ratio(numpy.array([0.0, 1.0]), numpy.array([0.5, 0.5])), 1.0)
        self.assertTrue(numpy.isnan(evaluation._wilcoxon(numpy.zeros(5))))
        self.assertTrue(numpy.isnan(evaluation._wilcoxon(numpy.array([0.1]))))
        self.assertLess(evaluation._wilcoxon(numpy.arange(1.0, 21.0)), 0.001)
        self.assertTrue(numpy.isnan(evaluation.partial_spearman(numpy.ones(6), numpy.arange(6.0), numpy.arange(6.0))))
        self.assertEqual(evaluation._format_p(float("nan")).strip(), "--")
        self.assertEqual(evaluation._format_p(0.0001), "1.0e-04")
        self.assertEqual(evaluation._format_p(0.25), "0.250")
        empty = pandas.DataFrame()
        self.assertTrue(evaluation._in_order(empty).empty)
        self.assertEqual(list(evaluation._model_family_scores(empty, empty, empty).columns),
                         ["family", "severity", "method", "seed"])

    def test_summaries_skip_what_they_cannot_compare(self):
        scores = pandas.DataFrame([
            {"family": "gaussian_noise", "severity": 0.5, "method": "single", "seed": s, "roc_auc": 0.8,
             "delta_roc_auc": -0.01, "worst_roc_auc": numpy.nan} for s in (1, 2)])
        scores["seed"] = scores["seed"].astype("Int64")
        # no MORSE rows: nothing to test
        self.assertTrue(evaluation._tests(scores, pandas.DataFrame(), pandas.DataFrame()).empty)
        morse = scores.assign(method="multi", roc_auc=0.82)
        tests = evaluation._tests(pandas.concat([scores, morse]), pandas.DataFrame(), pandas.DataFrame())
        self.assertEqual(set(tests["statistic"]), {"roc_auc", "delta_roc_auc"})   # worst_* are all NaN
        self.assertTrue(numpy.isnan(tests.loc[tests["statistic"] == "delta_roc_auc", "p_value"].iloc[0]))
        # fewer than four GA models: no sign-consistency correlation
        models = pandas.DataFrame({"method": ["multi", "single"], "seed": pandas.array([1, 1], dtype="Int64"),
                                   "sign_consistency": [0.9, 0.7], "n_features": [5, 9]})
        points, summary = evaluation._sign_analysis(models, pandas.concat([scores, morse]), RobustnessConfig(
            corruption_levels=(0.5,), headline_corruption_level=0.5))
        self.assertTrue(points.empty and summary.empty)

    def test_report_lines(self):
        self.assertEqual(evaluation._report_lines(pandas.DataFrame(), pandas.DataFrame(), pandas.DataFrame(),
                                                  pandas.DataFrame(), RobustnessConfig()),
                         ["\nNo scenario could be evaluated."])
        summary = pandas.DataFrame([{"family": "gaussian_noise", "severity": 0.5, "method": method,
                                     "clean_roc_auc": 0.8, "roc_auc": 0.75, "delta_roc_auc": -0.05, "n_scenarios": 10}
                                    for method in ("multi", "single")])
        skipped = pandas.DataFrame([{"family": "dependence_strengthen", "n_scenarios": 2}])
        lines = evaluation._report_lines(summary, pandas.DataFrame(), pandas.DataFrame(), skipped, RobustnessConfig())
        self.assertTrue(any("not summarised: only 2 pairs are usable at every level (at least 10 needed)" in line
                            for line in lines))
        self.assertTrue(any(line.startswith("Measurement noise") for line in lines))

    def test_severity_names_and_levels(self):
        config = RobustnessConfig()
        self.assertEqual(evaluation.severity_text("population", 0.6), "60% ESS")
        self.assertEqual(evaluation.severity_text("dependence_weaken", 1.0), "r -100%")
        self.assertEqual(evaluation.severity_text("dependence_strengthen", 0.125), "r +12.5%")
        self.assertEqual(evaluation.severity_text("value_masking", 0.5), "0.5")
        self.assertEqual(evaluation.family_levels("dependence_weaken", config), (0.25, 0.5, 0.75, 1.0))
        self.assertEqual(evaluation.family_levels("gaussian_noise", config), config.corruption_levels)
        self.assertEqual([evaluation.headline_level(f, config) for f in
                          ("population", "dependence_weaken", "dependence_strengthen", "binary_redraw")],
                         [0.6, 1.0, 0.5, 0.5])

    def test_provenance(self):
        with mock.patch.object(run_manifest.subprocess, "run", side_effect=OSError("no git")):
            self.assertEqual(run_manifest.git_state(), {"commit": "unknown", "uncommitted_changes": None})
        with mock.patch.object(run_manifest.subprocess, "run", side_effect=subprocess.TimeoutExpired("git", 10)):
            self.assertEqual(run_manifest.git_state()["commit"], "unknown")
        models = pandas.DataFrame({"method": ["multi", "all"], "seed": pandas.array([3, None], dtype="Int64")})
        frame = pandas.DataFrame({"a": [1.0, 2.0]})
        fingerprints = evaluation.data_fingerprints(frame, [0, 1], frame, [1, 0])
        started = datetime.now()
        with tempfile.TemporaryDirectory() as run:
            record = evaluation._provenance(RobustnessConfig(), run, fingerprints, models, None, "running", started)
            self.assertIsNone(record["training_fingerprint_sha256"])
            self.assertIsNone(record["run_manifest_sha256"])
            self.assertIsNone(record["runtime_seconds"])
            atomic_write_json(os.path.join(run, "checkpoints", "training", "fingerprint.json"), {"a": 1})
            atomic_write_json(os.path.join(run, run_manifest.MANIFEST_FILE), {"b": 2})
            record = evaluation._provenance(RobustnessConfig(), run, fingerprints, models, "knee", "complete",
                                            started, 12.34)
        self.assertEqual(len(record["training_fingerprint_sha256"]), 64)
        self.assertEqual(len(record["run_manifest_sha256"]), 64)
        self.assertEqual((record["status"], record["selection_rule"], record["runtime_seconds"]),
                         ("complete", "knee", 12.3))
        self.assertEqual((record["data"]["train_rows"], record["data"]["test_rows"], record["data"]["inputs"]),
                         (2, 2, 1))
        self.assertEqual(record["data"]["fingerprints"]["y_test"]["shape"], [2])
        self.assertEqual((record["seeds"], record["models"]["multi"], record["models"]["all"]), ([3], 1, 1))
        self.assertIn("robustness_utils.py", record["source_sha256"])


class ModelInstanceTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_data(200, seed=61)
        self.features = list(self.X.columns)
        n = len(self.features)

        def package(mask, seed):
            return build_model_package(mask, self.features, self.X, pandas.Series(self.y), seed=seed)

        self.packages = {seed: {"multi": package([1] * n, seed), "single": package([1] * 5 + [0] * (n - 5), seed),
                                "forward": package([1] * (2 + seed) + [0] * (n - 2 - seed), seed)}
                         for seed in (1, 2)}

    def test_a_missing_method_is_skipped_and_differing_baseline_inputs_are_kept_per_seed(self):
        notes = []
        models = evaluation._model_instances(self.packages, self.features, notes.append)
        self.assertEqual([(m.method, m.seed) for m in models],
                         [("multi", 1), ("multi", 2), ("single", 1), ("single", 2), ("forward", 1), ("forward", 2)])
        self.assertTrue(any("SFS inputs differ between the seeds" in note for note in notes))

    def test_the_self_check_stops_a_wrong_prediction_path(self):
        with mock.patch.object(evaluation, "predict_scores", side_effect=lambda package, X: numpy.zeros(len(X))):
            with tempfile.TemporaryDirectory() as out:
                with self.assertRaisesRegex(RuntimeError, "do not reproduce"):
                    run_robustness_suite(self.X, self.y, self.X, self.y, self.packages, out, log=lambda line: None)


class ReproducibilityTests(unittest.TestCase):
    """Review point 9: the fitted models are recorded and checked, an evaluation never overwrites one of
    other data or of another rule, and every stage is saved as soon as it is done."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.X, self.y = make_data(300, seed=95)
        self.X_test, self.y_test = make_data(150, seed=96)
        features = list(self.X.columns)
        self.packages = {seed: {m: build_model_package([1] * len(features), features, self.X, pandas.Series(self.y),
                                                       seed=seed) for m in ("multi", "single")} for seed in (1, 2)}
        self.config = RobustnessConfig(ess_levels=(0.8,), headline_ess=0.8, corruption_levels=(0.5,),
                                       headline_corruption_level=0.5, corruption_repetitions=1, max_pairs=2)

    def run_suite(self, out, X_test=None, rule="knee", log=lambda line: None):
        return run_robustness_suite(self.X, self.y, self.X_test if X_test is None else X_test, self.y_test,
                                    self.packages, out, config=self.config, log=log, selection_rule=rule)

    def test_the_final_models_are_recorded_and_checked(self):
        path = os.path.join(self.temporary.name, "final_models_knee.json")
        first = evaluation.record_final_models(self.packages, path)
        self.assertEqual(evaluation.record_final_models(self.packages, path), first)      # the same refit
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
        self.assertEqual(set(stored["models"]), {"multi/1", "single/1", "multi/2", "single/2"})
        stored["models"]["multi/1"]["coef"][0] += 1e-3
        atomic_write_json(path, stored)
        with self.assertRaisesRegex(RuntimeError, "multi/1: parameters differ"):
            evaluation.record_final_models(self.packages, path)
        stored["models"]["multi/1"]["features"] = ["x1"]
        atomic_write_json(path, stored)
        with self.assertRaisesRegex(RuntimeError, "other inputs"):
            evaluation.record_final_models(self.packages, path)
        # a seed evaluated for the first time is added to the record
        fresh = os.path.join(self.temporary.name, "fresh.json")
        evaluation.record_final_models({1: self.packages[1]}, fresh)
        evaluation.record_final_models(self.packages, fresh)
        with open(fresh, encoding="utf-8") as handle:
            self.assertEqual(len(json.load(handle)["models"]), 4)

    def test_build_final_models_writes_the_record_of_its_rule(self):
        ensure_multi_objective_types()
        ensure_single_objective_types()
        features = list(self.X.columns)
        n = len(features)
        individual = creator.Individual([1] * n)
        individual.fitness.values = (0.9, 0.9)
        single = creator.IndividualSingle([1] * n)
        single.fitness.values = (0.9,)
        folder = os.path.join(self.temporary.name, "evaluation")
        build_final_models(features, self.X, pandas.Series(self.y), [5], {5: [individual]}, {5: single},
                           {5: [1] * n}, {5: [1] * n}, use_knee_point=False, record_directory=folder)
        self.assertTrue(os.path.isfile(os.path.join(folder, "final_models_max_s.json")))

    def test_an_evaluation_of_other_data_or_another_rule_is_not_overwritten(self):
        out = os.path.join(self.temporary.name, "out")
        self.run_suite(out)
        self.run_suite(out)                                                              # the same: fine
        with self.assertRaisesRegex(FileExistsError, "other data"):
            self.run_suite(out, X_test=self.X_test.assign(x1=self.X_test["x1"] + 1.0))
        with self.assertRaisesRegex(FileExistsError, "the Pareto rule 'knee'"):
            self.run_suite(out, rule="max_s")
        # an evaluation made before the data were recorded is replaced, with a note
        with open(os.path.join(out, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        del config["data"]["fingerprints"], config["selection_rule"]
        atomic_write_json(os.path.join(out, "config.json"), config)
        notes = []
        self.run_suite(out, rule="max_s", log=notes.append)
        self.assertTrue(any("made before the data were recorded" in note for note in notes))
        with open(os.path.join(out, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        self.assertEqual((config["status"], config["selection_rule"]), ("complete", "max_s"))
        self.assertGreater(config["runtime_seconds"], 0)

    def test_every_stage_is_saved_when_it_is_done(self):
        out = os.path.join(self.temporary.name, "interrupted")
        with mock.patch.object(evaluation, "_corruption", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_suite(out)
        written = set(os.listdir(out))
        self.assertTrue({"config.json", "models.csv", "schema.json", "scenarios.csv", "reweighting_scores.csv"}
                        <= written)
        self.assertFalse({"corruption_scores.csv", "summary.csv", "report.txt"} & written)
        with open(os.path.join(out, "config.json"), encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["status"], "running")


class CleanViolationTests(unittest.TestCase):
    def test_a_test_value_that_already_breaks_the_availability_rule_is_reported(self):
        X, y = make_data(300, seed=81)
        X_test, y_test = make_data(150, seed=82)
        row = X_test.index[X_test["lab_measured"] == 0][0]
        X_test.loc[row, "lab"] = 7.0
        features = list(X.columns)
        packages = {1: {m: build_model_package([1] * len(features), features, X, pandas.Series(y), seed=1)
                        for m in ("multi", "single")}}
        config = RobustnessConfig(ess_levels=(0.8,), headline_ess=0.8, corruption_levels=(0.5,),
                                  headline_corruption_level=0.5, corruption_repetitions=1, max_pairs=2)
        with tempfile.TemporaryDirectory() as out:
            results = run_robustness_suite(X, y, X_test, y_test, packages, out, config=config, log=lambda line: None)
        self.assertIn("NOTE: 1 clean test values do not hold their fill value", results.report)


class SaturatedPopulationTests(unittest.TestCase):
    def test_a_population_shift_the_data_cannot_support_is_recorded_but_not_scored(self):
        rng = numpy.random.default_rng(71)
        b = (rng.random(300) < 0.7).astype(int)
        X = pandas.DataFrame({"b": b, "c": b})                    # two identical 0/1 inputs: two distinct rows
        y = (rng.random(300) < numpy.where(b == 1, 0.7, 0.3)).astype(int)
        packages = {1: {m: build_model_package([1, 1], ["b", "c"], X, pandas.Series(y), seed=1)
                        for m in ("multi", "single", "forward", "all")}}
        config = RobustnessConfig(ess_levels=(0.8, 0.6), headline_ess=0.6, corruption_levels=(0.5,),
                                  headline_corruption_level=0.5, corruption_repetitions=2)
        with tempfile.TemporaryDirectory() as out:
            results = run_robustness_suite(X, y, X, y, packages, out, config=config, log=lambda line: None)
            scores = pandas.read_csv(os.path.join(out, "reweighting_scores.csv"))
        scenarios = results.scenarios.set_index("scenario_id")
        # tilting towards the 70% of rows with b = 1 can never leave less than 70% ESS
        saturated = scenarios.loc["population|PC1|+|0.6"]
        self.assertFalse(saturated["attainable"])
        self.assertFalse(saturated["supported"] or saturated["usable"])
        self.assertTrue(numpy.isnan(saturated["strength"]))
        self.assertNotIn("population|PC1|+|0.6", set(scores["scenario_id"]))
        self.assertTrue(scenarios.loc["population|PC1|+|0.8", "attainable"])
        self.assertIn("population|PC1|+|0.8", set(scores["scenario_id"]))
        # the direction that cannot reach 60% is out of the population cohort at every level, so the
        # main summary does not compare "PC1 +" at 80% with a set of scenarios that lacks it at 60%
        self.assertFalse(scenarios.loc["population|PC1|+|0.8", "in_cohort"])
        self.assertTrue(scenarios.loc["population|PC1|-|0.8", "in_cohort"])


class CohortTests(unittest.TestCase):
    """Review point 6: the report, the tests and the figures summarise the same scenarios -- the cohort
    usable at every level -- while every scenario usable at a level goes to a supplementary table."""

    @classmethod
    def setUpClass(cls):
        X, y = make_data(1500, seed=91)
        X_test, y_test = make_data(500, seed=92)
        features = list(X.columns)
        chosen = masks(len(features))
        packages = {seed: {method: build_model_package(chosen[method][seed], features, X, pandas.Series(y), seed=seed)
                           for method in ("multi", "single")} for seed in SEEDS}
        cls.config = RobustnessConfig(ess_levels=(0.9, 0.6), headline_ess=0.6, corruption_levels=(0.5,),
                                      headline_corruption_level=0.5, corruption_repetitions=1, min_pairs=1,
                                      dependence_weaken_levels=(0.5, 1.0), headline_weaken=1.0,
                                      dependence_strengthen_levels=(0.25, 0.5), headline_strengthen=0.5)
        cls.directory = tempfile.TemporaryDirectory()
        cls.results = run_robustness_suite(X, y, X_test, y_test, packages, cls.directory.name, config=cls.config,
                                           log=lambda line: None)
        cls.scores = pandas.read_csv(os.path.join(cls.directory.name, "reweighting_scores.csv"))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_weakening_reaches_its_target_and_never_reverses_the_dependence(self):
        weakened = self.results.scenarios[(self.results.scenarios["family"] == "dependence_weaken")
                                          & self.results.scenarios["attainable"]]
        self.assertFalse(weakened.empty)
        numpy.testing.assert_allclose(weakened["score_correlation_reached"], weakened["score_correlation_target"],
                                      atol=1e-6)
        self.assertTrue((weakened["score_correlation_reached"] * weakened["score_correlation_clean"] >= -1e-9).all())
        full = weakened[weakened["level"] == 1.0]
        numpy.testing.assert_allclose(full["score_correlation_reached"], 0.0, atol=1e-6)
        strengthened = self.results.scenarios[(self.results.scenarios["family"] == "dependence_strengthen")
                                              & self.results.scenarios["attainable"]]
        numpy.testing.assert_allclose(strengthened["score_correlation_reached"],
                                      strengthened["score_correlation_clean"] * (1 + strengthened["level"]), atol=1e-6)

    def test_the_cohort_is_the_same_at_every_level(self):
        scenarios = self.results.scenarios
        for family, part in scenarios[scenarios["in_cohort"]].groupby("family"):
            units = [set(level_part["unit"]) for _, level_part in part.groupby("level")]
            self.assertTrue(all(u == units[0] for u in units), family)
            self.assertTrue(part["usable"].all())
        summary = self.results.summary
        for family, part in summary[summary["family"].str.startswith("dependence_")].groupby("family"):
            self.assertEqual(part["n_scenarios"].nunique(), 1)
            self.assertEqual(int(part["n_scenarios"].iloc[0]),
                             scenarios.loc[(scenarios["family"] == family) & scenarios["in_cohort"], "unit"].nunique())

    def test_the_figure_shows_the_numbers_of_the_summary(self):
        curves, _, _ = evaluation._dependence_plot_data(self.results.scenarios, self.scores, self.config)
        self.assertFalse(curves.empty)
        summary = self.results.summary.set_index(["family", "severity", "method"])
        for _, row in curves.iterrows():
            self.assertAlmostEqual(row["mean"], summary.loc[(f"dependence_{row['direction']}", row["level"],
                                                             row["method"]), "delta_roc_auc"], places=12)

    def test_the_per_level_cohort_is_supplementary(self):
        supplementary = pandas.read_csv(os.path.join(self.directory.name, "supplementary_per_level_summary.csv"))
        self.assertTrue(set(supplementary["family"]) <= set(evaluation.REWEIGHTING_FAMILIES))
        usable = self.results.scenarios[self.results.scenarios["usable"]]
        for (family, level), part in supplementary.groupby(["family", "severity"]):
            expected = usable[(usable["family"] == family) & numpy.isclose(usable["level"], level)]["unit"].nunique()
            self.assertEqual(int(part["n_scenarios"].max()), expected)
        self.assertTrue(os.path.isfile(os.path.join(self.directory.name, "supplementary_per_level_tests.csv")))
        self.assertIn("Cohorts -- the units usable at every level", self.results.report)
        self.assertIn("Supplementary (not the main analysis)", self.results.report)


if __name__ == "__main__":
    unittest.main()
