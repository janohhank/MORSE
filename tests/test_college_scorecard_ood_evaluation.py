"""Tests of college_scorecard/college_scorecard_ood_evaluation.py: the helpers (data loading, fingerprints,
run discovery, metrics, statistics) and the whole command line on a synthetic checkpointed run whose
fronts carry the true CV objectives and whose per-seed results the script must reproduce.

Run from the repository root:

    python -m unittest discover -s tests -v
"""
import contextlib
import importlib.util
import io
import json
import os
import runpy
import sys
import tempfile
import unittest
from unittest import mock

import numpy
import pandas
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from deap import creator  # noqa: E402

from checkpoint_utils import (SeedTrainingResult, TrainingCheckpointStore, atomic_write_json,  # noqa: E402
                              build_training_fingerprint)
from deap_types import ensure_multi_objective_types, ensure_single_objective_types  # noqa: E402
from evaluation_utils import build_model_package, predict_scores  # noqa: E402
from multi_objective_training import MultiObjectiveTraining  # noqa: E402
from training_config import TrainingConfig  # noqa: E402
from training_utils import best_sign_consistency_index  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "college_scorecard_ood_evaluation", os.path.join(ROOT, "college_scorecard", "college_scorecard_ood_evaluation.py"))
ood = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ood)

N_FEATURES = 6
SEEDS = (3, 4)


def make_split(n: int, seed: int, shift: float = 0.0) -> pandas.DataFrame:
    """College-Scorecard-like data; `shift` moves the inputs and the prevalence (the out-of-domain file)."""
    rng = numpy.random.default_rng(seed)
    X = rng.normal(loc=shift, size=(n, N_FEATURES))
    X[:, 5] = (rng.random(n) < 0.4).astype(float)
    logit = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.6 * X[:, 5] - 1.5 * shift
    frame = pandas.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)])
    frame["label"] = (rng.random(n) < 1.0 / (1.0 + numpy.exp(-logit))).astype(int)
    return frame


def write_fixture(directory: str) -> dict:
    """Three CSVs and a checkpointed run trained on the first one, with the run's own per-seed results."""
    paths = {"train": os.path.join(directory, "train.csv"), "id": os.path.join(directory, "id.csv"),
             "ood": os.path.join(directory, "ood.csv")}
    make_split(240, 1).to_csv(paths["train"], index=False)
    make_split(160, 2).to_csv(paths["id"], index=False)
    make_split(160, 3, shift=0.7).to_csv(paths["ood"], index=False)
    X_train, y_train = ood.load_split(paths["train"], "label")
    features = list(X_train.columns)

    run = os.path.join(directory, "2026-01-01_00-00-00_cs")
    fingerprint = build_training_fingerprint(
        config=TrainingConfig(seed=0, use_roc_auc=False), cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        feature_names=features, X_train=numpy.ascontiguousarray(X_train.to_numpy(), dtype=numpy.float64),
        y_train=numpy.ascontiguousarray(y_train, dtype=numpy.float64))
    store = TrainingCheckpointStore(os.path.join(run, "checkpoints", "training"), fingerprint)
    store.prepare()
    evaluator = ood.cv_evaluator(fingerprint, X_train, y_train, features)
    ensure_multi_objective_types()
    ensure_single_objective_types()
    X_id, y_id = ood.load_split(paths["id"], "label", features)
    rows = []
    for seed in SEEDS:
        front = []
        for mask in ([1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 1], [1, 0, 0, 0, 0, seed % 2]):
            individual = creator.Individual(mask)
            individual.fitness.values = evaluator.evaluate_multi(mask)     # the true CV objectives
            front.append(individual)
        single = creator.IndividualSingle([1, 1, 0, 1, 0, 1])
        single.fitness.values = (0.8,)
        result = SeedTrainingResult(seed, front, single, [1, 1, 0, 0, 0, 0], [1] * N_FEATURES)
        store.save_seed(result)
        # the run's own clean test results (PR-AUC, MORSE = the max-S end), as the notebook wrote them
        masks = {"multi": list(front[best_sign_consistency_index(front)]), "single": list(single),
                 "forward": result.forward_mask, "all": result.all_mask}
        row = {"seed": seed, "noise_level": 0.0, "mean_shift": 0.0}
        for key, mask in masks.items():
            package = build_model_package(mask, features, X_train, y_train, seed=seed)
            row[f"auc_{key}"] = average_precision_score(y_id, predict_scores(package, X_id))
        rows.append(row)
    evaluation_directory = os.path.join(run, "evaluation", "all_models_comparison")
    os.makedirs(evaluation_directory)
    pandas.DataFrame(rows).to_csv(os.path.join(evaluation_directory, "gaussian_2d_per_seed.csv"), index=False)
    write_notebook(run, "USE_KNEE_POINT_SELECTION: bool = False")
    return {"paths": paths, "run": run, "features": features, "fingerprint": fingerprint}


def write_notebook(directory: str, config_line: str) -> None:
    notebook = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# run"]},
                          {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                           "source": ["N_JOBS = -1\n", config_line + "\n"]}],
                "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    with open(os.path.join(directory, "training_notebook.ipynb"), "w", encoding="utf-8") as handle:
        json.dump(notebook, handle)


class LoadingAndFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, "data.csv")
        make_split(50, 5).to_csv(self.path, index=False)

    def test_load_split_orders_and_checks_the_inputs(self):
        X, y = ood.load_split(self.path, "label", ["f2", "f0"])
        self.assertEqual(list(X.columns), ["f2", "f0"])
        self.assertTrue(set(numpy.unique(y)) <= {0, 1})
        with self.assertRaisesRegex(ValueError, "no target column"):
            ood.load_split(self.path, "outcome")
        with self.assertRaisesRegex(ValueError, "input"):
            ood.load_split(self.path, "label", ["f0", "missing_input"])
        pandas.read_csv(self.path).assign(label=2).to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "must be 0/1"):
            ood.load_split(self.path, "label")

    def test_the_data_fingerprint_matches_the_checkpoint_fingerprint(self):
        X, y = ood.load_split(self.path, "label")
        current = ood.training_data_fingerprint(X, y)
        other = ood.training_data_fingerprint(X.iloc[:, ::-1], 1 - y)
        unscaled = numpy.ascontiguousarray(X.to_numpy(), dtype=numpy.float64)
        # the trainers receive the unscaled inputs; runs made before 2026-09-28 received them standardised
        for X_train, legacy in ((unscaled, False), (StandardScaler().fit_transform(unscaled), True)):
            full = build_training_fingerprint(
                config=TrainingConfig(seed=0), cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                feature_names=list(X.columns), X_train=X_train, y_train=numpy.ascontiguousarray(y, dtype=numpy.float64))
            if legacy:
                del full["settings"]["standardisation"]
            self.assertEqual(ood.fingerprint_differences(full, current), [])
            self.assertEqual(len(ood.fingerprint_differences(full, other)), 3)
            # the other convention's matrix is another matrix
            full["settings"]["data"]["X_train"] = current["data_standardised" if not legacy else "data"]["X_train"]
            self.assertEqual(ood.fingerprint_differences(full, current), ["the training data (X_train) differ"])

    def test_the_cv_evaluator_standardises_as_the_run_did(self):
        X, y = ood.load_split(self.path, "label")
        features = list(X.columns)
        unscaled = numpy.ascontiguousarray(X.to_numpy(), dtype=numpy.float64)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fingerprint = build_training_fingerprint(config=TrainingConfig(seed=0), cv=cv, feature_names=features,
                                                 X_train=unscaled, y_train=numpy.asarray(y, dtype=numpy.float64))
        legacy = json.loads(json.dumps(fingerprint))
        del legacy["settings"]["standardisation"]
        mask = [1, 1, 0, 1, 0, 1]
        config = TrainingConfig(seed=0)
        per_fold = MultiObjectiveTraining(config, features, unscaled, y.astype(float), cv).evaluate_multi(mask)
        as_a_whole = MultiObjectiveTraining(config, features, StandardScaler().fit_transform(unscaled), y.astype(float),
                                            cv, standardise_folds=False).evaluate_multi(mask)
        self.assertEqual(ood.cv_evaluator(fingerprint, X, y, features).evaluate_multi(mask), per_fold)
        self.assertEqual(ood.cv_evaluator(legacy, X, y, features).evaluate_multi(mask), as_a_whole)


class RunDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.fixture = write_fixture(cls.temporary.name)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def current(self) -> dict:
        X, y = ood.load_split(self.fixture["paths"]["train"], "label")
        return ood.training_data_fingerprint(X, y)

    def test_matching_runs_are_found_and_the_newest_is_used(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ("2026-01-01_run", "2026-02-01_run"):
                target = os.path.join(root, name, "checkpoints", "training")
                os.makedirs(target)
                atomic_write_json(os.path.join(target, "fingerprint.json"), self.fixture["fingerprint"])
            unrelated = os.path.join(root, "2026-03-01_other", "checkpoints", "training")
            os.makedirs(unrelated)
            other = json.loads(json.dumps(self.fixture["fingerprint"]))
            other["settings"]["features"]["count"] += 1
            atomic_write_json(os.path.join(unrelated, "fingerprint.json"), other)
            with mock.patch.object(ood, "REPOSITORY_ROOT", root):
                self.assertEqual([os.path.basename(r) for r in ood.find_matching_runs(self.current())],
                                 ["2026-01-01_run", "2026-02-01_run"])
                with contextlib.redirect_stdout(io.StringIO()) as printed:
                    chosen = ood.resolve_run(None, "train.csv", self.current())
                self.assertEqual(os.path.basename(chosen), "2026-02-01_run")
                self.assertIn("using the newest", printed.getvalue())
            with mock.patch.object(ood, "REPOSITORY_ROOT", self.temporary.name + "_does_not_exist"):
                with self.assertRaises(FileNotFoundError):
                    ood.find_matching_runs(self.current())

    def test_an_explicit_run_must_have_checkpoints_of_this_data(self):
        self.assertEqual(ood.resolve_run(self.fixture["run"], "train.csv", self.current()), self.fixture["run"])
        with self.assertRaises(FileNotFoundError):
            ood.resolve_run(self.temporary.name, "train.csv", self.current())
        different = ood.training_data_fingerprint(*ood.load_split(self.fixture["paths"]["id"], "label"))
        with self.assertRaisesRegex(ValueError, "was not trained on"):
            ood.resolve_run(self.fixture["run"], "train.csv", different)
        with tempfile.TemporaryDirectory() as empty_root:
            with mock.patch.object(ood, "REPOSITORY_ROOT", empty_root):
                with self.assertRaisesRegex(FileNotFoundError, "no run folder"):
                    ood.resolve_run(None, "train.csv", self.current())

    def test_the_selection_rule_is_read_from_the_archived_notebook(self):
        self.assertEqual(ood.run_selection_rule(self.fixture["run"]), "max_s")
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(ood.run_selection_rule(directory))
            write_notebook(directory, "USE_KNEE_POINT_SELECTION = True")
            self.assertEqual(ood.run_selection_rule(directory), "knee")
            write_notebook(directory, "N = 1")
            self.assertIsNone(ood.run_selection_rule(directory))
            # the run manifest comes first
            atomic_write_json(os.path.join(directory, "run_manifest.json"), {"settings": {"pareto_rule": "max_s"}})
            self.assertEqual(ood.run_selection_rule(directory), "max_s")
            atomic_write_json(os.path.join(directory, "run_manifest.json"), {"settings": {}})
            self.assertIsNone(ood.run_selection_rule(directory))

    def test_display_path(self):
        inside = os.path.join(ood.REPOSITORY_ROOT, "college_scorecard", "x.csv")
        self.assertEqual(ood.display_path(inside), os.path.join("college_scorecard", "x.csv"))
        with mock.patch.object(ood.os.path, "relpath", side_effect=ValueError("different drives")):
            self.assertEqual(ood.display_path("x.csv"), os.path.abspath("x.csv"))


class MetricAndStatisticTests(unittest.TestCase):
    def setUp(self):
        rng = numpy.random.default_rng(8)
        self.y = (rng.random(200) < 0.35).astype(int)
        self.scores = numpy.round(rng.random((4, 200)) + 0.4 * self.y, 1)     # ties

    def test_calibration_weights_and_calibrated_pr_auc(self):
        weights = ood.calibration_weights(self.y, 0.6)
        self.assertAlmostEqual(weights[self.y == 1].sum() / weights.sum(), 0.6)
        same = ood.score(self.y, self.scores[0], float(self.y.mean()))
        self.assertAlmostEqual(same["pr_auc_calibrated"], same["pr_auc"])
        self.assertAlmostEqual(same["roc_auc"], roc_auc_score(self.y, self.scores[0]))
        higher = ood.score(self.y, self.scores[0], 0.8)
        self.assertGreater(higher["pr_auc_calibrated"], same["pr_auc"])     # a higher prevalence, higher precision

    def test_row_wise_roc_auc_equals_sklearn(self):
        expected = [roc_auc_score(self.y, row) for row in self.scores]
        numpy.testing.assert_allclose(ood.roc_auc_rows(self.y, self.scores), expected, atol=1e-12)

    def test_partial_spearman(self):
        rng = numpy.random.default_rng(9)
        z = rng.normal(size=100)
        x, y = z + rng.normal(size=100), z + rng.normal(size=100)
        self.assertLess(abs(ood.partial_spearman(x, y, z)), 0.3)                # x and y only share z
        self.assertTrue(numpy.isnan(ood.partial_spearman(numpy.ones(10), numpy.arange(10.0), numpy.arange(10.0))))

    def per_seed(self, morse_offset: float) -> pandas.DataFrame:
        rows = []
        for seed in range(6):
            for key, base in (("max_s", 0.80 + morse_offset), ("soga", 0.80), ("sfs", 0.79), ("all", 0.78)):
                value = base + 0.001 * seed
                rows.append({"seed": seed, "model": key, "ood_roc_auc": value, "roc_auc_drop": 0.1 - value / 10,
                             "ood_pr_auc_calibrated": value, "pr_auc_calibrated_drop": 0.1})
        return pandas.DataFrame(rows)

    def test_paired_tests(self):
        tests = ood.paired_tests(self.per_seed(0.01), "max_s")
        self.assertEqual(len(tests), 3 * 4)
        roc = tests[(tests["metric"] == "ood_roc_auc") & (tests["baseline"] == "SO-GA")].iloc[0]
        self.assertAlmostEqual(roc["mean_difference"], 0.01)
        self.assertEqual((roc["morse_greater_in"], roc["n_seeds"]), (6, 6))
        # identical values: no test
        drop = tests[(tests["metric"] == "pr_auc_calibrated_drop")]
        self.assertTrue(drop["wilcoxon_p"].isna().all())

    def test_bootstrap_intervals(self):
        predictions = {key: {"id": self.scores[:2] + 0.05 * i, "ood": self.scores[2:] + 0.05 * i}
                       for i, key in enumerate(["max_s", "soga", "sfs", "all"])}
        self.assertTrue(ood.bootstrap_roc(predictions, self.y, self.y, "max_s", 0, 0).empty)
        intervals = ood.bootstrap_roc(predictions, self.y, self.y, "max_s", 50, 1)
        self.assertEqual(len(intervals), 3 * 2)
        self.assertTrue((intervals["bootstrap_ci_low"] <= intervals["bootstrap_ci_high"]).all())
        self.assertTrue((intervals["n_bootstrap"] == 50).all())
        # a single positive: most resamples have one class and are skipped, never crash
        rare = numpy.zeros(200, dtype=int)
        rare[0] = 1
        few = ood.bootstrap_roc(predictions, rare, rare, "max_s", 20, 2)
        self.assertTrue((few["n_bootstrap"] < 20).all())

    def test_front_correlations(self):
        fronts = pandas.DataFrame({"seed": [1, 1, 1, 2, 2, 2], "cv_sign_consistency": [0.6, 0.8, 1.0, 0.7, 0.7, 0.7],
                                   "ood_roc_auc": [0.70, 0.72, 0.75, 0.7, 0.71, 0.72], "n_features": [9, 6, 3, 5, 4, 3],
                                   "roc_auc_drop": [0.10, 0.08, 0.05, 0.1, 0.1, 0.1],
                                   "sign_consistency": [0.6, 0.8, 1.0, 0.8, 0.9, 1.0]})
        correlations = ood.front_correlations(fronts)
        self.assertEqual(correlations["cv_s_vs_ood_roc"][0], 1.0)
        self.assertEqual(correlations["cv_s_vs_roc_drop"][0], -1.0)
        self.assertTrue(numpy.isnan(correlations["cv_s_vs_ood_roc"][1]))       # constant S in seed 2


class CommandLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.fixture = write_fixture(cls.temporary.name)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def arguments(self, out: str, *extra: str) -> list[str]:
        paths = self.fixture["paths"]
        return ["--run", self.fixture["run"], "--train-csv", paths["train"], "--id-test-csv", paths["id"],
                "--ood-csv", paths["ood"], "--bootstrap", "30", "--out", out, *extra]

    def test_the_run_is_reproduced_and_evaluated_out_of_domain(self):
        out = os.path.join(self.temporary.name, "out")
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            ood.main(self.arguments(out))
        report = printed.getvalue()
        self.assertIn("reproduced (largest deviation", report)
        self.assertIn("the run's own front-selection rule", report)
        for name in ("ood_per_seed.csv", "ood_summary.csv", "ood_tests.csv", "ood_front_solutions.csv",
                     "ood_evaluation.png", "ood_evaluation.pdf", "ood_report.txt"):
            self.assertGreater(os.path.getsize(os.path.join(out, name)), 0, name)
        per_seed = pandas.read_csv(os.path.join(out, "ood_per_seed.csv"))
        self.assertEqual(len(per_seed), len(SEEDS) * 6)
        self.assertTrue(numpy.allclose(per_seed["roc_auc_drop"], per_seed["id_roc_auc"] - per_seed["ood_roc_auc"]))
        fronts = pandas.read_csv(os.path.join(out, "ood_front_solutions.csv"))
        self.assertEqual(len(fronts), len(SEEDS) * 3)
        numpy.testing.assert_allclose(fronts["cv_sign_consistency"], fronts["cv_sign_consistency_stored"])
        tests = pandas.read_csv(os.path.join(out, "ood_tests.csv"))
        self.assertIn("bootstrap_ci_low", tests.columns)

    def test_an_explicit_selection_overrides_the_run_rule(self):
        out = os.path.join(self.temporary.name, "out_knee")
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            ood.main(self.arguments(out, "--selection", "knee"))
        self.assertIn("chosen with --selection", printed.getvalue())

    def test_other_models_than_the_run_evaluated_are_refused(self):
        run = self.fixture["run"]
        path = os.path.join(run, "evaluation", "all_models_comparison", "gaussian_2d_per_seed.csv")
        results = pandas.read_csv(path)
        try:
            results.assign(auc_single=results["auc_single"] + 0.01).to_csv(path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "do not reproduce the run's own results"):
                    ood.main(self.arguments(os.path.join(self.temporary.name, "out_bad")))
        finally:
            results.to_csv(path, index=False)

    def test_without_the_run_results_the_check_is_skipped(self):
        per_seed = pandas.DataFrame({"seed": [1], "model": ["soga"], "id_pr_auc": [0.5]})
        message, rule = ood.verify_against_run(self.temporary.name, per_seed, None, use_roc_auc=False)
        self.assertEqual((message.startswith("not checked"), rule), (True, None))

    def test_a_stored_cv_objective_that_is_not_reproduced_is_an_error(self):
        store_directory = os.path.join(self.fixture["run"], "checkpoints", "training")
        store = TrainingCheckpointStore(store_directory, self.fixture["fingerprint"])
        result = store.load_seed(SEEDS[0])
        result.pareto_front[0].fitness.values = (0.123, 0.5)
        X_train, y_train = ood.load_split(self.fixture["paths"]["train"], "label")
        X_id, y_id = ood.load_split(self.fixture["paths"]["id"], "label")
        evaluator = ood.cv_evaluator(self.fixture["fingerprint"], X_train, y_train, self.fixture["features"])
        marginal = pandas.Series(0.1, index=self.fixture["features"])
        with self.assertRaisesRegex(RuntimeError, "not reproduced"):
            ood.evaluate_front_solutions({SEEDS[0]: result}, self.fixture["features"], X_train, y_train, marginal,
                                         evaluator, X_id, y_id, X_id, y_id)

    def test_duplicate_front_members_are_evaluated_once(self):
        store = TrainingCheckpointStore(os.path.join(self.fixture["run"], "checkpoints", "training"),
                                        self.fixture["fingerprint"])
        result = store.load_seed(SEEDS[0])
        duplicate = creator.Individual(list(result.pareto_front[1]))
        duplicate.fitness.values = result.pareto_front[1].fitness.values
        result.pareto_front.append(duplicate)
        X_train, y_train = ood.load_split(self.fixture["paths"]["train"], "label")
        X_id, y_id = ood.load_split(self.fixture["paths"]["id"], "label")
        evaluator = ood.cv_evaluator(self.fixture["fingerprint"], X_train, y_train, self.fixture["features"])
        fronts = ood.evaluate_front_solutions({SEEDS[0]: result}, self.fixture["features"], X_train, y_train,
                                              pandas.Series(0.1, index=self.fixture["features"]), evaluator,
                                              X_id, y_id, X_id, y_id)
        self.assertEqual(len(fronts), 3)                        # three distinct masks, the duplicate once
        positions = {p for joined in fronts["positions"].fillna("") for p in joined.split("|") if p}
        self.assertEqual(positions, {"max_f1", "knee", "max_s"})

    def test_without_notebook_and_run_results_the_max_s_end_is_the_default(self):
        run = self.fixture["run"]
        notebook = os.path.join(run, "training_notebook.ipynb")
        results = os.path.join(run, "evaluation", "all_models_comparison", "gaussian_2d_per_seed.csv")
        os.replace(notebook, notebook + ".away")
        os.replace(results, results + ".away")
        try:
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                ood.main(self.arguments(os.path.join(self.temporary.name, "out_default"), "--bootstrap", "0"))
        finally:
            os.replace(notebook + ".away", notebook)
            os.replace(results + ".away", results)
        report = printed.getvalue()
        self.assertIn("not checked: the run folder has no per-seed evaluation results", report)
        self.assertIn("default -- the run's own rule could not be determined", report)

    def test_the_script_entry_point(self):
        path = os.path.join(ROOT, "college_scorecard", "college_scorecard_ood_evaluation.py")
        with mock.patch.object(sys, "argv", [path, "--help"]):
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                with self.assertRaises(SystemExit) as stopped:
                    runpy.run_path(path, run_name="__main__")
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("--ood-csv", printed.getvalue())

    def test_a_run_without_completed_seeds_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            run = os.path.join(directory, "empty_run")
            store = TrainingCheckpointStore(os.path.join(run, "checkpoints", "training"), self.fixture["fingerprint"])
            store.prepare()
            paths = self.fixture["paths"]
            with self.assertRaisesRegex(FileNotFoundError, "no completed seed"):
                ood.main(["--run", run, "--train-csv", paths["train"], "--id-test-csv", paths["id"],
                          "--ood-csv", paths["ood"]])


if __name__ == "__main__":
    unittest.main()
