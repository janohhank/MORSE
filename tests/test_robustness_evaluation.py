"""Tests of robustness_evaluation: the final models, the whole suite on synthetic data, the aggregation
(corruption repetitions are averaged within a run before runs are compared), and the stand-alone loading
of a finished run (archived notebook + checkpoint fingerprint).

Run from the repository root:

    python -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import unittest

import numpy
import pandas
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from deap import creator  # noqa: E402

from checkpoint_utils import atomic_write_json, build_training_fingerprint  # noqa: E402
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
        self.assertLessEqual(int(scenarios["family"].str.startswith("dependence_").sum()), 10)
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


if __name__ == "__main__":
    unittest.main()
