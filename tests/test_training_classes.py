"""Tests of the four trainers: MultiObjectiveTraining (NSGA-II, CV AUC + sign consistency),
SingleObjectiveTraining (the AUC-only GA), ForwardStepwiseTraining and AllFeaturesTraining.

The fitness functions are checked against a direct re-computation on the same folds, every fold standardised on
its own training rows; the GAs are run with a tiny population so that the tests take seconds.

Run from the repository root:

    python -m unittest discover -s tests -v
"""
import contextlib
import csv
import io
import os
import random
import sys
import tempfile
import unittest

import numpy
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deap import creator  # noqa: E402

from all_features_training import AllFeaturesTraining  # noqa: E402
from evaluation_utils import compute_marginal_correlations  # noqa: E402
from forward_stepwise_training import ForwardStepwiseTraining  # noqa: E402
from multi_objective_training import MultiObjectiveTraining  # noqa: E402
from single_objective_training import SingleObjectiveTraining  # noqa: E402
from training_config import TrainingConfig  # noqa: E402

N_FEATURES: int = 8


def make_data(n: int = 300, seed: int = 0) -> tuple[numpy.ndarray, numpy.ndarray, list[str]]:
    """Informative inputs (x0 strongly, x1 as a suppressor of x0's correlated twin x2, a rare 0/1 input x6) and
    noise, unscaled and on very different scales; x6 (4% ones) and the heavy-tailed x7 have SDs that differ
    between the CV folds, so standardising every fold on its own training rows and standardising the whole
    training set once give different fitness values."""
    rng = numpy.random.default_rng(seed)
    X = rng.normal(size=(n, N_FEATURES))
    X[:, 2] = 0.8 * X[:, 0] + 0.6 * rng.normal(size=n)
    X[:, 6] = (rng.random(n) < 0.04).astype(float)
    logit = 2.0 * X[:, 0] - 1.0 * X[:, 2] + 1.0 * X[:, 1] + 1.5 * X[:, 6]
    y = (rng.random(n) < 1.0 / (1.0 + numpy.exp(-logit))).astype(float)
    X[:, 7] = rng.standard_t(df=1.5, size=n)
    scales = numpy.array([1.0, 10.0, 0.1, 100.0, 1.0, 5.0, 1.0, 1.0])
    offsets = numpy.array([0.0, 5.0, -3.0, 50.0, 0.0, 0.0, 0.0, 0.0])
    return X * scales + offsets, y, [f"x{i}" for i in range(N_FEATURES)]


def cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def quietly(function, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args)


def manual_fitness(X, y, mask, use_roc_auc: bool, seed: int = 0, per_fold: bool = True) -> tuple[float, float]:
    """The CV fitness recomputed directly: mean fold AUC (or AP) and mean fold sign consistency, with the
    marginal correlations of each fold's training part. `per_fold`: the selected inputs are standardised on
    each fold's training rows (otherwise `X` is used as it is, e.g. standardised as a whole beforehand)."""
    columns = [i for i, bit in enumerate(mask) if bit]
    aucs, signs = [], []
    for train, validation in cv().split(X, y):
        X_train, X_validation = X[train][:, columns], X[validation][:, columns]
        if per_fold:
            scaler = StandardScaler().fit(X_train)
            X_train, X_validation = scaler.transform(X_train), scaler.transform(X_validation)
        model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed)
        model.fit(X_train, y[train])
        probabilities = model.predict_proba(X_validation)[:, 1]
        aucs.append(roc_auc_score(y[validation], probabilities) if use_roc_auc
                    else average_precision_score(y[validation], probabilities))
        product = compute_marginal_correlations(X[train], y[train])[columns] * model.coef_[0]
        signs.append(1.0 - numpy.sum((product < 0) | numpy.isclose(product, 0.0, atol=1e-12)) / len(columns))
    return float(numpy.mean(aucs)), float(numpy.mean(signs))


def dominates(a, b) -> bool:
    return all(x >= z for x, z in zip(a, b)) and any(x > z for x, z in zip(a, b))


class AllFeaturesTrainingTests(unittest.TestCase):
    def test_every_feature_is_selected(self):
        names = [f"f{i}" for i in range(5)]
        self.assertEqual(AllFeaturesTraining(TrainingConfig(seed=0), names).run(), [1] * 5)


class ForwardStepwiseTrainingTests(unittest.TestCase):
    def test_it_selects_the_informative_inputs_and_stops(self):
        X, y, _ = make_data(600)
        for use_roc_auc in (True, False):
            mask = ForwardStepwiseTraining(TrainingConfig(seed=0, use_roc_auc=use_roc_auc), X, y, cv()).run()
            self.assertEqual(len(mask), N_FEATURES)
            self.assertTrue(set(mask) <= {0, 1})
            self.assertEqual(mask[0], 1, "the strongest input must be selected")
            self.assertLess(sum(mask), N_FEATURES, "the tol-based stopping must leave noise inputs out")

    def test_every_fold_is_standardised_inside_the_pipeline(self):
        X, y, _ = make_data(600)

        def selection(estimator, inputs) -> list[int]:
            selector = SequentialFeatureSelector(estimator, n_features_to_select="auto", tol=1e-3,
                                                 direction="forward", scoring="roc_auc", cv=cv())
            return [int(bit) for bit in selector.fit(inputs, y).get_support()]

        mask = ForwardStepwiseTraining(TrainingConfig(seed=0), X, y, cv()).run()
        model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=0)
        self.assertEqual(mask, selection(Pipeline([("scaler", StandardScaler()), ("lr", model)]), X))
        # the scaler is refitted in every fold, so the selection does not depend on the scale of the inputs ...
        rescaled = X * numpy.linspace(0.5, 20.0, N_FEATURES) - 7.0
        self.assertEqual(ForwardStepwiseTraining(TrainingConfig(seed=0), rescaled, y, cv()).run(), mask)
        # ... whereas without a scaler these unscaled inputs lead to another selection
        self.assertNotEqual(selection(model, X), mask)


class MultiObjectiveTrainingTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.names = make_data()

    def trainer(self, use_roc_auc: bool = True, **settings) -> MultiObjectiveTraining:
        config = TrainingConfig(seed=0, use_roc_auc=use_roc_auc, **settings)
        return MultiObjectiveTraining(config, self.names, self.X, self.y, cv())

    def test_an_empty_mask_has_zero_fitness(self):
        self.assertEqual(self.trainer().evaluate_multi([0] * N_FEATURES), (0.0, 0.0))

    def test_the_fitness_is_the_mean_over_the_folds(self):
        for use_roc_auc in (True, False):
            for mask in ([1, 1, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 1, 1],
                         [1] * N_FEATURES):
                auc, sign = self.trainer(use_roc_auc).evaluate_multi(mask)
                expected_auc, expected_sign = manual_fitness(self.X, self.y, mask, use_roc_auc)
                self.assertAlmostEqual(auc, expected_auc, places=12)
                self.assertAlmostEqual(sign, expected_sign, places=12)

    def test_runs_made_before_2026_09_28_are_re_evaluated_with_standardise_folds_false(self):
        # those runs passed the training set standardised as a whole, and the folds were slices of it
        standardised = StandardScaler().fit_transform(self.X)
        legacy = MultiObjectiveTraining(TrainingConfig(seed=0), self.names, standardised, self.y, cv(),
                                        standardise_folds=False)
        mask = [1, 1, 0, 1, 0, 0, 1, 1]
        auc, sign = legacy.evaluate_multi(mask)
        expected_auc, expected_sign = manual_fitness(standardised, self.y, mask, True, per_fold=False)
        self.assertAlmostEqual(auc, expected_auc, places=12)
        self.assertAlmostEqual(sign, expected_sign, places=12)
        # the validation rows of every fold contributed to that scaling: standardising per fold gives another value
        self.assertNotAlmostEqual(auc, self.trainer().evaluate_multi(mask)[0], places=6)

    def test_sign_consistency_sees_the_suppressor(self):
        trainer = self.trainer()
        # x2 is positively correlated with y through x0 but has a negative conditional effect
        _, with_suppressor = trainer.evaluate_multi([1, 0, 1, 0, 0, 0, 0, 0])
        _, single_input = trainer.evaluate_multi([1, 0, 0, 0, 0, 0, 0, 0])
        self.assertLess(with_suppressor, 1.0)
        self.assertEqual(single_input, 1.0)

    def test_results_are_cached_until_cleared(self):
        trainer = self.trainer()
        mask = [1, 1, 0, 0, 0, 0, 0, 0]
        first = trainer.evaluate_multi(mask)
        self.assertIs(trainer.evaluate_multi(mask), first)
        trainer.clear_cache()
        self.assertEqual(trainer._cache, {})
        self.assertEqual(trainer.evaluate_multi(mask), first)

    def run_ga(self, directory: str = "") -> list:
        random.seed(5)
        numpy.random.seed(5)
        return quietly(self.trainer(pop_size=8, ngen=2, result_directory=directory).run)

    def test_run_returns_a_non_dominated_front_and_writes_its_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            front = self.run_ga(directory)
            seed_directory = os.path.join(directory, "seed_0")
            with open(os.path.join(seed_directory, "convergence.csv"), newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["gen"]) for row in rows], [0, 1, 2])
            for name in ("convergence.png", "pareto_front.png"):
                self.assertGreater(os.path.getsize(os.path.join(seed_directory, name)), 0)
        self.assertTrue(front)
        for individual in front:
            self.assertIsInstance(individual, creator.Individual)
            self.assertEqual(len(individual), N_FEATURES)
            auc, sign = individual.fitness.values
            self.assertTrue(0.0 <= auc <= 1.0 and 0.0 <= sign <= 1.0)
        for a in front:
            for b in front:
                self.assertFalse(dominates(a.fitness.values, b.fitness.values))
        self.assertLessEqual(max(float(row["pareto_size"]) for row in rows), 8 * 3)

    def test_run_is_reproducible_for_a_seed(self):
        first = [(list(ind), ind.fitness.values) for ind in self.run_ga()]
        second = [(list(ind), ind.fitness.values) for ind in self.run_ga()]
        self.assertEqual(first, second)


class SingleObjectiveTrainingTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.names = make_data()

    def trainer(self, use_roc_auc: bool = True, **settings) -> SingleObjectiveTraining:
        config = TrainingConfig(seed=0, use_roc_auc=use_roc_auc, **settings)
        return SingleObjectiveTraining(config, self.names, self.X, self.y, cv())

    def test_an_empty_mask_has_zero_fitness(self):
        self.assertEqual(self.trainer().evaluate_single([0] * N_FEATURES), (0.0,))

    def test_the_fitness_is_the_mean_fold_auc(self):
        for use_roc_auc in (True, False):
            for mask in ([1, 1, 0, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 1, 1]):
                (auc,) = self.trainer(use_roc_auc).evaluate_single(mask)
                self.assertAlmostEqual(auc, manual_fitness(self.X, self.y, mask, use_roc_auc)[0], places=12)

    def test_runs_made_before_2026_09_28_are_re_evaluated_with_standardise_folds_false(self):
        standardised = StandardScaler().fit_transform(self.X)
        legacy = SingleObjectiveTraining(TrainingConfig(seed=0), self.names, standardised, self.y, cv(),
                                         standardise_folds=False)
        mask = [1, 1, 0, 1, 0, 0, 1, 1]
        (auc,) = legacy.evaluate_single(mask)
        self.assertAlmostEqual(auc, manual_fitness(standardised, self.y, mask, True, per_fold=False)[0], places=12)
        self.assertNotAlmostEqual(auc, self.trainer().evaluate_single(mask)[0], places=6)

    def test_results_are_cached_until_cleared(self):
        trainer = self.trainer()
        mask = [1, 0, 0, 0, 0, 0, 0, 1]
        first = trainer.evaluate_single(mask)
        self.assertIs(trainer.evaluate_single(mask), first)
        trainer.clear_cache()
        self.assertEqual(trainer._cache, {})

    def run_ga(self, directory: str = ""):
        random.seed(7)
        numpy.random.seed(7)
        return quietly(self.trainer(pop_size=8, ngen=2, result_directory=directory).run)

    def test_run_returns_the_best_individual_and_writes_its_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            best = self.run_ga(directory)
            seed_directory = os.path.join(directory, "seed_0")
            with open(os.path.join(seed_directory, "convergence.csv"), newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(os.path.getsize(os.path.join(seed_directory, "convergence.png")), 0)
        self.assertIsInstance(best, creator.IndividualSingle)
        self.assertEqual(len(best), N_FEATURES)
        self.assertAlmostEqual(best.fitness.values[0], self.trainer().evaluate_single(list(best))[0], places=12)
        # the hall of fame keeps the best individual ever seen
        self.assertAlmostEqual(best.fitness.values[0], max(float(row["max"]) for row in rows), places=12)

    def test_run_is_reproducible_for_a_seed(self):
        self.assertEqual(list(self.run_ga()), list(self.run_ga()))


if __name__ == "__main__":
    unittest.main()
