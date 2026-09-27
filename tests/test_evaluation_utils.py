"""Tests of evaluation_utils: column-type detection, the final model package and its scores, the sign
consistency of a final model, the balanced sensitivity / specificity threshold, and the stress tests of
the legacy stress grid (Gaussian noise, PC1 covariate shift, AURS).
(compute_marginal_correlations and apply_dummy_noise have their own test files.)

Run from the repository root:

    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation_utils import (apply_dummy_noise, apply_proportional_noise, build_model_package,  # noqa: E402
                              compute_aurs, compute_marginal_correlations, compute_model_sign_consistency,
                              covariate_shift_weights, evaluate_model, find_balanced_threshold,
                              fit_covariate_shift_axis, get_continuous_columns, get_dummy_columns,
                              predict_scores, score_predictions, select_deployment_model)
from deap import creator  # noqa: E402
from deap_types import ensure_multi_objective_types  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402


def make_frame(n: int = 400, seed: int = 0) -> tuple[pandas.DataFrame, pandas.Series]:
    rng = numpy.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 0.8 * x1 + 0.6 * rng.normal(size=n)
    frame = pandas.DataFrame({
        "x1": x1, "x2": x2, "count": rng.integers(0, 5, size=n),
        "flag": (rng.random(n) < 0.3).astype(int), "flag_bool": rng.random(n) < 0.5,
        "constant_one": numpy.ones(n, dtype=int), "two_values": rng.choice([3.0, 7.0], size=n)})
    logit = 1.2 * x1 - 0.8 * x2 + 0.7 * frame["flag"]
    y = pandas.Series((rng.random(n) < 1.0 / (1.0 + numpy.exp(-logit))).astype(int))
    return frame, y


class ColumnTypeTests(unittest.TestCase):
    def test_continuous_and_dummy_columns(self):
        frame, _ = make_frame()
        self.assertEqual(get_continuous_columns(frame), ["x1", "x2", "count"])
        # 0/1 columns (also bool and a constant 1) are dummies; a two-valued {3, 7} column is neither
        self.assertEqual(get_dummy_columns(frame), ["flag", "flag_bool", "constant_one"])


class ModelPackageTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.y = make_frame()
        self.features = list(self.frame.columns)
        self.mask = [1 if name in ("x1", "x2", "flag") else 0 for name in self.features]

    def test_package_refits_on_the_selected_inputs_only(self):
        package = build_model_package(self.mask, self.features, self.frame, self.y, seed=3)
        self.assertEqual(package["features"], ["x1", "x2", "flag"])
        self.assertEqual(package["model"].coef_.shape, (1, 3))
        scaled = StandardScaler().fit_transform(self.frame[["x1", "x2", "flag"]].to_numpy())
        reference = LogisticRegression(solver="lbfgs", max_iter=1000).fit(scaled, self.y)
        numpy.testing.assert_allclose(package["model"].coef_, reference.coef_, rtol=1e-8)

    def test_scores_and_weighted_scores(self):
        package = build_model_package(self.mask, self.features, self.frame, self.y, seed=3)
        probabilities = predict_scores(package, self.frame)
        self.assertTrue(((probabilities > 0) & (probabilities < 1)).all())
        self.assertAlmostEqual(score_predictions(self.y, probabilities, use_roc_auc=True),
                               roc_auc_score(self.y, probabilities))
        self.assertAlmostEqual(score_predictions(self.y, probabilities, use_roc_auc=False),
                               average_precision_score(self.y, probabilities))
        weights = numpy.linspace(0.5, 1.5, len(self.y))
        self.assertAlmostEqual(score_predictions(self.y, probabilities, True, weights),
                               roc_auc_score(self.y, probabilities, sample_weight=weights))
        self.assertAlmostEqual(evaluate_model(package, self.frame, self.y, use_roc_auc=False, sample_weight=weights),
                               average_precision_score(self.y, probabilities, sample_weight=weights))

    def test_sign_consistency_of_a_final_model(self):
        package = build_model_package(self.mask, self.features, self.frame, self.y, seed=3)
        marginal = pandas.Series(compute_marginal_correlations(self.frame, self.y), index=self.features)
        result = compute_model_sign_consistency(package, marginal)
        expected_inconsistent = int(sum(marginal[name] * coefficient <= 0 for name, coefficient
                                        in zip(package["features"], package["model"].coef_[0])))
        self.assertEqual(result["n_features"], 3)
        self.assertEqual(result["n_inconsistent"], expected_inconsistent)
        self.assertEqual(result["n_consistent"], 3 - expected_inconsistent)
        self.assertAlmostEqual(result["sign_consistency"], 1 - expected_inconsistent / 3)
        # x2 is a suppressor here (positive marginal correlation, negative conditional effect)
        self.assertGreaterEqual(result["n_inconsistent"], 1)

    def test_a_zero_product_counts_as_inconsistent(self):
        package = build_model_package(self.mask, self.features, self.frame, self.y, seed=3)
        marginal = pandas.Series(1.0, index=self.features)
        marginal["x1"] = 0.0
        self.assertGreaterEqual(compute_model_sign_consistency(package, marginal)["n_inconsistent"], 1)


class BalancedThresholdTests(unittest.TestCase):
    def test_perfect_separation(self):
        y = numpy.array([0, 0, 0, 1, 1, 1])
        scores = numpy.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        result = find_balanced_threshold(y, scores)
        self.assertEqual(result["threshold"], 0.7)
        self.assertEqual((result["sensitivity"], result["specificity"]), (1.0, 1.0))
        numpy.testing.assert_array_equal(result["y_pred"], y)
        self.assertEqual(len(result["sensitivity_curve"]), 6)
        numpy.testing.assert_array_equal(result["sorted_scores"], numpy.sort(scores))

    def test_threshold_balances_the_two_rates(self):
        rng = numpy.random.default_rng(1)
        y = (rng.random(300) < 0.4).astype(int)
        scores = rng.random(300) + 0.6 * y
        result = find_balanced_threshold(pandas.Series(y), pandas.Series(scores))
        gaps = numpy.abs(result["sensitivity_curve"] - result["specificity_curve"])
        self.assertAlmostEqual(abs(result["sensitivity"] - result["specificity"]), gaps.min())
        predicted = result["y_pred"]
        self.assertAlmostEqual(result["sensitivity"], predicted[y == 1].mean())
        self.assertAlmostEqual(result["specificity"], 1 - predicted[y == 0].mean())


class LegacyStressTests(unittest.TestCase):
    """The stress tests of runs made before the robustness suite."""

    def setUp(self):
        self.frame, self.y = make_frame(2000, seed=2)
        self.train_std = self.frame.std()

    def test_gaussian_noise_scales_with_the_training_sd_of_each_continuous_column(self):
        numpy.random.seed(0)
        noisy = apply_proportional_noise(self.frame, self.train_std, 0.5, ["x1", "count", "absent"])
        for column in ("x1", "count"):
            self.assertAlmostEqual((noisy[column] - self.frame[column]).std(), 0.5 * self.train_std[column],
                                   delta=0.05 * self.train_std[column])
        pandas.testing.assert_series_equal(noisy["x2"], self.frame["x2"])       # not in the list
        pandas.testing.assert_frame_equal(apply_proportional_noise(self.frame, self.train_std, 0.0, ["x1"]),
                                          self.frame)

    def test_dummy_noise_without_matching_columns_changes_nothing(self):
        pandas.testing.assert_frame_equal(apply_dummy_noise(self.frame, 0.5, ["absent"], pandas.Series(dtype=float)),
                                          self.frame)

    def test_the_covariate_shift_axis_is_the_first_principal_component(self):
        axis = fit_covariate_shift_axis(self.frame)
        self.assertEqual(axis["features"], list(self.frame.columns))
        self.assertAlmostEqual(float(numpy.linalg.norm(axis["loading"])), 1.0, places=12)
        self.assertGreater(axis["loading"].sum(), 0.0)
        self.assertEqual(axis["std"][list(self.frame.columns).index("constant_one")], 1.0)   # zero SD -> 1
        Z = (self.frame.to_numpy(dtype=float) - axis["mean"]) / axis["std"]
        eigenvalues, eigenvectors = numpy.linalg.eigh(Z.T @ Z / len(Z))
        self.assertAlmostEqual(abs(float(eigenvectors[:, -1] @ axis["loading"])), 1.0, places=8)
        self.assertAlmostEqual(axis["score_std"], float((Z @ axis["loading"]).std()), places=12)

    def test_covariate_shift_weights(self):
        axis = fit_covariate_shift_axis(self.frame)
        numpy.testing.assert_allclose(covariate_shift_weights(axis, self.frame, self.y, 0.0), 1.0)
        weights = covariate_shift_weights(axis, self.frame, self.y, 1.0)
        # the class totals are kept, so the outcome prevalence is unchanged
        for cls in (0, 1):
            self.assertAlmostEqual(weights[self.y == cls].sum(), float((self.y == cls).sum()), places=8)
        # within a class the weight grows with the (clipped) PC1 score
        score = ((self.frame.to_numpy(dtype=float) - axis["mean"]) / axis["std"]) @ axis["loading"] / axis["score_std"]
        negatives = self.y.to_numpy() == 0
        order = numpy.argsort(score[negatives])
        self.assertTrue(numpy.all(numpy.diff(weights[negatives][order]) >= -1e-12))
        # scores beyond the clip share one weight
        clipped = weights[negatives][score[negatives] >= 2.0]
        if clipped.size > 1:
            self.assertAlmostEqual(clipped.min(), clipped.max(), places=12)


class DeploymentModelTests(unittest.TestCase):
    """The best MORSE model is chosen without the test set (review point 3)."""

    def setUp(self):
        ensure_multi_objective_types()
        self.X, self.y = make_frame(400, seed=3)
        self.features = list(self.X.columns)
        self.cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        def front(mask, cv_objective):
            individual = creator.Individual(mask + [0] * (len(self.features) - len(mask)))
            individual.fitness.values = (cv_objective, 0.9)
            return [individual]

        # seed 3 ties with seed 2 on the cross-validated objective: the smaller seed wins
        self.fronts = {1: front([1, 1], 0.80), 2: front([1, 1, 1, 1], 0.85), 3: front([1, 0, 1], 0.85)}

    def choose(self):
        return select_deployment_model(self.fronts, [1, 2, 3], self.features, self.X, self.y, self.cv,
                                       use_knee_point=True)

    def test_the_seed_with_the_best_cross_validated_objective_is_chosen(self):
        chosen = self.choose()
        self.assertEqual((chosen["seed"], chosen["cv_objective"]), (2, 0.85))
        self.assertEqual(chosen["package"]["features"], self.features[:4])
        self.assertIs(chosen["individual"], self.fronts[2][0])

    def test_the_threshold_balances_the_out_of_fold_predictions(self):
        chosen = self.choose()
        X = self.X[self.features[:4]].to_numpy(dtype=float)
        y = self.y.to_numpy()
        expected = numpy.zeros(len(y))
        for train, held_out in self.cv.split(X, y):
            scaler = StandardScaler().fit(X[train])
            model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=2).fit(scaler.transform(X[train]),
                                                                                            y[train])
            expected[held_out] = model.predict_proba(scaler.transform(X[held_out]))[:, 1]
        numpy.testing.assert_allclose(chosen["oof_scores"], expected, rtol=0, atol=1e-12)
        self.assertEqual(chosen["threshold"], find_balanced_threshold(y, expected)["threshold"])
        self.assertEqual(chosen["balanced"]["threshold"], chosen["threshold"])


class AursTests(unittest.TestCase):
    @staticmethod
    def surface(function, noise=numpy.linspace(0.0, 1.0, 11), shift=numpy.linspace(-1.0, 1.0, 11)) -> pandas.DataFrame:
        rows = [{"noise_level": n, "mean_shift": s, "auc_m": function(n, s)} for n in noise for s in shift]
        return pandas.DataFrame(rows).groupby(["noise_level", "mean_shift"]).mean()

    def test_a_flat_surface_keeps_everything(self):
        self.assertAlmostEqual(compute_aurs(self.surface(lambda n, s: 0.8), "m"), 1.0, places=12)

    def test_a_linear_loss_is_integrated_exactly(self):
        # retention 1 - 0.2 * noise: the mean over noise in [0, 1] is 0.9, whatever the shift
        self.assertAlmostEqual(compute_aurs(self.surface(lambda n, s: 0.8 * (1 - 0.2 * n)), "m"), 0.9, places=12)
        # a gain on one side of the shift cancels a loss on the other: the limitation the docstring names
        self.assertAlmostEqual(compute_aurs(self.surface(lambda n, s: 0.8 * (1 + 0.1 * s)), "m"), 1.0, places=12)

    def test_uneven_levels_are_weighted_by_their_spacing(self):
        noise = numpy.array([0.0, 0.1, 1.0])
        # retention 1 on [0, 0.1], then falls linearly to 0.5 at 1.0: area 0.1 + 0.9 * 0.75 = 0.775
        value = compute_aurs(self.surface(lambda n, s: 0.8 if n <= 0.1 else 0.4, noise=noise), "m")
        self.assertAlmostEqual(value, 0.775, places=12)

    def test_the_clean_cell_is_required(self):
        with self.assertRaisesRegex(ValueError, "clean cell"):
            compute_aurs(self.surface(lambda n, s: 0.8, noise=numpy.array([0.1, 0.5])), "m")
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            compute_aurs(self.surface(lambda n, s: 0.0), "m")


if __name__ == "__main__":
    unittest.main()
