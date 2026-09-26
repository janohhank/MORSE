"""Tests of robustness_utils: weighted metrics, tilts and their calibration, the automatic feature schema,
the dependence tilt and the corruption bank.

Run from the repository root:

    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy
import pandas
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robustness_config import RobustnessConfig  # noqa: E402
from robustness_utils import (CorruptionBank, DependenceTilt, NormalScores, calibrate_strength,  # noqa: E402
                              calibrate_strengths, dependence_pairs, effective_sample_fraction,
                              exponential_tilt, infer_feature_schema, normalise_within_classes,
                              population_axes, weighted_average_precision, weighted_correlation,
                              weighted_roc_auc)

FILL: float = 5.0


def make_data(n: int, seed: int) -> tuple[pandas.DataFrame, numpy.ndarray]:
    """Inputs of every kind the schema has to recognise:
      x1, x2                 correlated continuous inputs (r about 0.7)
      lab, lab_measured      a value with its availability flag (the value is FILL when not measured)
      state_A/B/C            an exhaustive one-hot group
      region_N/S             a one-hot group with a reference level (row sum 0 or 1)
      med_a, med_b           share a name prefix but are independent: NOT a one-hot group
      rare, common, common2  stand-alone 0/1 inputs (common2 is correlated with common)
      general, specific      specific = 1 only together with general = 1
      female, male_code      never both 1
    """
    rng = numpy.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 0.7 * x1 + 0.7 * rng.normal(size=n)
    measured = rng.random(n) < 0.7
    lab = numpy.where(measured, rng.normal(10.0, 2.0, size=n), FILL)
    state = rng.choice(3, size=n, p=[0.5, 0.3, 0.2])
    region = rng.choice(3, size=n, p=[0.4, 0.4, 0.2])
    common = (rng.random(n) < 0.7).astype(int)
    general = (rng.random(n) < 0.4).astype(int)
    female = (rng.random(n) < 0.6).astype(int)
    frame = pandas.DataFrame({
        "x1": x1, "x2": x2, "lab": lab, "lab_measured": measured.astype(int),
        "state_A": (state == 0).astype(int), "state_B": (state == 1).astype(int), "state_C": (state == 2).astype(int),
        "region_N": (region == 0).astype(int), "region_S": (region == 1).astype(int),
        "med_a": (rng.random(n) < 0.3).astype(int), "med_b": (rng.random(n) < 0.3).astype(int),
        "rare": (rng.random(n) < 0.1).astype(int), "common": common,
        "common2": numpy.where(rng.random(n) < 0.7, common, (rng.random(n) < 0.7).astype(int)),
        "general": general, "specific": general * (rng.random(n) < 0.5).astype(int),
        "female": female, "male_code": (1 - female) * (rng.random(n) < 0.5).astype(int),
    })
    logit = x1 - 0.5 * x2 + 0.5 * measured + 0.8 * frame["rare"] - 0.3 * frame["state_B"]
    y = (rng.random(n) < 1.0 / (1.0 + numpy.exp(-logit))).astype(int)
    return frame, y


class WeightedMetricTests(unittest.TestCase):
    def setUp(self):
        rng = numpy.random.default_rng(1)
        self.y = (rng.random(400) < 0.4).astype(int)
        self.scores = numpy.round(rng.random(400) + 0.5 * self.y, 1)     # rounded: many ties
        self.weights = rng.random(400) + 0.2

    def test_weighted_metrics_match_sklearn(self):
        for weights in (None, self.weights):
            self.assertAlmostEqual(weighted_roc_auc(self.y, self.scores, weights),
                                   roc_auc_score(self.y, self.scores, sample_weight=weights), places=12)
            self.assertAlmostEqual(weighted_average_precision(self.y, self.scores, weights),
                                   average_precision_score(self.y, self.scores, sample_weight=weights), places=12)

    def test_per_class_normalisation_keeps_roc_auc_and_prevalence(self):
        balanced = normalise_within_classes(self.weights, self.y)
        self.assertAlmostEqual(weighted_roc_auc(self.y, self.scores, balanced),
                               weighted_roc_auc(self.y, self.scores, self.weights), places=12)
        self.assertAlmostEqual(balanced @ self.y / balanced.sum(), self.y.mean(), places=12)

    def test_effective_sample_fraction(self):
        self.assertAlmostEqual(effective_sample_fraction(numpy.ones(50)), 1.0)
        one_row = numpy.zeros(50)
        one_row[3] = 1.0
        self.assertAlmostEqual(effective_sample_fraction(one_row), 1.0 / 50)

    def test_weighted_correlation(self):
        a, b = self.scores, self.weights
        self.assertAlmostEqual(weighted_correlation(a, b), numpy.corrcoef(a, b)[0, 1], places=12)


class TiltTests(unittest.TestCase):
    def test_normal_scores(self):
        values = numpy.random.default_rng(2).exponential(size=5000)
        transform = NormalScores(values)
        scores = transform(values)
        self.assertAlmostEqual(scores.mean(), 0.0, delta=0.01)
        self.assertAlmostEqual(scores.std(), 1.0, delta=0.02)
        order = numpy.argsort(values)
        self.assertTrue(numpy.all(numpy.diff(scores[order]) >= 0))
        tied = transform(numpy.array([values[0], values[0]]))
        self.assertEqual(tied[0], tied[1])
        extreme = transform(numpy.array([-1e9, 1e9]))
        self.assertAlmostEqual(extreme[0], scores.min())
        self.assertAlmostEqual(extreme[1], scores.max())

    def test_calibration_follows_the_normal_formula(self):
        statistic = NormalScores(numpy.random.default_rng(3).normal(size=20000))(
            numpy.random.default_rng(3).normal(size=20000))
        targets = [0.9, 0.8, 0.6]
        results = calibrate_strengths(lambda s: effective_sample_fraction(exponential_tilt(statistic, s)), targets)
        for target, (strength, saturated) in zip(targets, results):
            self.assertFalse(saturated)
            self.assertAlmostEqual(strength, numpy.sqrt(-numpy.log(target)), delta=0.02)
            self.assertAlmostEqual(effective_sample_fraction(exponential_tilt(statistic, strength)), target, places=4)

    def test_calibration_reports_an_unreachable_target(self):
        statistic = numpy.r_[numpy.zeros(300), numpy.ones(700)]     # at most 70% of the rows can share the weight
        strength, saturated = calibrate_strength(lambda s: effective_sample_fraction(exponential_tilt(statistic, s)), 0.6)
        self.assertTrue(saturated)
        strength, saturated = calibrate_strength(lambda s: effective_sample_fraction(exponential_tilt(statistic, s)), 0.9)
        self.assertFalse(saturated)
        # a NaN (the tilt cannot be realised) also stops the search
        _, saturated = calibrate_strength(lambda s: float("nan") if s > 0.1 else 1.0 - s, 0.5)
        self.assertTrue(saturated)


class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_data(3000, seed=4)
        cls.schema = infer_feature_schema(cls.X, RobustnessConfig())

    def test_types(self):
        self.assertEqual(set(self.schema.continuous), {"x1", "x2", "lab"})
        self.assertIn("lab_measured", self.schema.binary)
        self.assertFalse(self.schema.constant)

    def test_one_hot_groups_need_the_data_to_confirm_the_name(self):
        self.assertEqual(self.schema.dummy_families, {"state": ["state_A", "state_B", "state_C"],
                                                      "region": ["region_N", "region_S"]})
        self.assertEqual(self.schema.family_frequencies["state"][-1], 0.0)          # exhaustive
        self.assertAlmostEqual(self.schema.family_frequencies["region"][-1], 0.2, delta=0.03)

    def test_value_with_availability_flag(self):
        self.assertEqual(len(self.schema.value_indicators), 1)
        indicator = self.schema.value_indicators[0]
        self.assertEqual((indicator.flag, indicator.off_level, indicator.fills), ("lab_measured", 0, {"lab": FILL}))

    def test_forbidden_combinations(self):
        forbidden = {(f.a, f.a_value, f.b, f.b_value) for f in self.schema.forbidden}
        self.assertTrue(("general", 0, "specific", 1) in forbidden or ("specific", 1, "general", 0) in forbidden)
        self.assertTrue(("female", 1, "male_code", 1) in forbidden or ("male_code", 1, "female", 1) in forbidden)
        self.assertTrue(("state_A", 1, "state_B", 1) in forbidden)
        self.assertFalse(any({f[0], f[2]} == {"med_a", "med_b"} for f in forbidden))

    def test_stand_alone_and_under_recording_inputs(self):
        self.assertEqual(set(self.schema.redraw_columns),
                         {"med_a", "med_b", "rare", "common", "common2", "general", "specific", "female", "male_code"})
        self.assertEqual(set(self.schema.under_recording_columns),
                         {"med_a", "med_b", "rare", "general", "specific", "male_code"})


class ShiftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_data(3000, seed=5)
        cls.X_test, cls.y_test = make_data(800, seed=6)
        cls.config = RobustnessConfig()
        cls.schema = infer_feature_schema(cls.X, cls.config)

    def test_population_axes_depend_on_the_training_data_and_the_row_only(self):
        axes = population_axes(self.X, self.X_test, self.config)
        self.assertEqual([axis.name for axis in axes], ["PC1", "PC2", "PC3", "extremes"])
        subset = population_axes(self.X, self.X_test.iloc[:50], self.config)
        for full, part in zip(axes, subset):
            numpy.testing.assert_allclose(full.train, part.train)
            numpy.testing.assert_allclose(full.test[:50], part.test)
            self.assertAlmostEqual(full.train.mean(), 0.0, delta=0.01)

    def test_dependence_pairs_follow_the_rules(self):
        pairs, _ = dependence_pairs(self.X, self.X_test, self.schema, self.config)
        names = {frozenset((p.a, p.b)) for p in pairs}
        self.assertIn(frozenset(("x1", "x2")), names)
        self.assertIn(frozenset(("common", "common2")), names)
        self.assertNotIn(frozenset(("lab", "lab_measured")), names)       # a value with its own flag
        self.assertFalse(any(len(pair) == 2 and all(n.startswith("state_") for n in pair) for pair in names))
        self.assertTrue(all(self.config.pair_min_abs_correlation <= abs(p.correlation)
                            < self.config.pair_max_abs_correlation for p in pairs))
        self.assertEqual([abs(p.correlation) for p in pairs], sorted((abs(p.correlation) for p in pairs), reverse=True))

    def check_tilt(self, a: str, b: str, binary: tuple[bool, bool]):
        tilt = DependenceTilt(self.X[a], self.X[b], self.X_test[a], self.X_test[b], *binary, clip=2.5)
        base = weighted_correlation(self.X[a], self.X[b])
        for strength, direction in ((0.3, 1.0), (-0.3, -1.0)):
            w_train, w_test = tilt.weights(strength)
            # the balanced moments are those of the scores: exact for a 0/1 input (its score is linear in
            # the value), approximate for the raw values of a continuous input (normal scores)
            self.assertLess(tilt.balance_error, 1e-8)
            for column, is_binary in zip((a, b), binary):
                values = self.X[column].to_numpy(dtype=float)
                tolerance = 1e-8 if is_binary else 0.02 * values.std()
                self.assertAlmostEqual(w_train @ values / w_train.sum(), values.mean(), delta=tolerance)
            change = weighted_correlation(self.X[a], self.X[b], w_train) - base
            self.assertGreater(direction * change, 0.02)
            self.assertEqual(w_test.shape, (len(self.X_test),))
        w_train, _ = tilt.weights(0.0)
        numpy.testing.assert_allclose(w_train, 1.0, atol=1e-9)

    def test_dependence_tilt_changes_the_dependence_and_keeps_the_means(self):
        self.check_tilt("x1", "x2", (False, False))
        self.check_tilt("common", "common2", (True, True))
        self.check_tilt("x1", "rare", (False, True))

    def test_reweighting_never_uses_labels(self):
        axes = population_axes(self.X, self.X_test, self.config)
        weights = exponential_tilt(axes[0].test, 0.7)
        shuffled = numpy.random.default_rng(7).permutation(self.y_test)
        # the weights are a function of X alone; only the per-class normalisation (used for PR-AUC) sees y,
        # and ROC-AUC is the same with or without it
        scores = self.X_test["x1"].to_numpy()
        for labels in (self.y_test, shuffled):
            self.assertAlmostEqual(weighted_roc_auc(labels, scores, weights),
                                   weighted_roc_auc(labels, scores, normalise_within_classes(weights, labels)), places=12)


class CorruptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, _ = make_data(2000, seed=8)
        cls.X_test, _ = make_data(600, seed=9)
        cls.schema = infer_feature_schema(cls.X, RobustnessConfig())
        cls.bank = CorruptionBank(cls.schema, cls.X, cls.X_test, repetitions=3, seed=11)

    def test_every_family_applies_and_level_zero_is_the_clean_test_set(self):
        self.assertEqual(self.bank.applicable_families(),
                         ["gaussian_noise", "binary_redraw", "under_recording", "value_masking"])
        for family in self.bank.applicable_families():
            corrupted, counts = self.bank.corrupt(family, 0.0, 0)
            pandas.testing.assert_frame_equal(corrupted, self.X_test.astype(float))
            self.assertEqual(counts["changed"], 0)

    def test_common_random_numbers(self):
        clean = self.X_test.astype(float)
        low, _ = self.bank.corrupt("gaussian_noise", 0.3, 1)
        high, _ = self.bank.corrupt("gaussian_noise", 0.6, 1)
        numpy.testing.assert_allclose((high - clean)[["x1", "x2"]].to_numpy(), 2 * (low - clean)[["x1", "x2"]].to_numpy())
        for family in ("binary_redraw", "under_recording", "value_masking"):
            low_changed = (self.bank.corrupt(family, 0.3, 2)[0] != clean).to_numpy()
            high_changed = (self.bank.corrupt(family, 0.7, 2)[0] != clean).to_numpy()
            self.assertTrue(low_changed.any())
            self.assertFalse((low_changed & ~high_changed).any(), family)

    def test_repetitions_differ_and_the_bank_is_reproducible(self):
        first, _ = self.bank.corrupt("binary_redraw", 0.5, 0)
        second, _ = self.bank.corrupt("binary_redraw", 0.5, 1)
        again, _ = CorruptionBank(self.schema, self.X, self.X_test, repetitions=3, seed=11).corrupt("binary_redraw", 0.5, 0)
        self.assertFalse(first.equals(second))
        pandas.testing.assert_frame_equal(first, again)

    def test_redraw_keeps_the_data_valid(self):
        clean = self.X_test
        corrupted, counts = self.bank.corrupt("binary_redraw", 1.0, 0)
        self.assertTrue((corrupted[["state_A", "state_B", "state_C"]].sum(axis=1) == 1).all())
        self.assertTrue((corrupted[["region_N", "region_S"]].sum(axis=1) <= 1).all())
        for (a, u, b, v) in (("specific", 1, "general", 0), ("female", 1, "male_code", 1)):
            created = ((corrupted[a] == u) & (corrupted[b] == v)) & ~((clean[a] == u) & (clean[b] == v))
            self.assertFalse(created.any(), (a, b))
        self.assertGreater(counts["reverted"], 0)
        # the availability flag and the continuous inputs are not touched by the re-draw
        pandas.testing.assert_series_equal(corrupted["lab_measured"], clean["lab_measured"].astype(float))
        pandas.testing.assert_series_equal(corrupted["x1"], clean["x1"].astype(float))

    def test_under_recording_only_loses_ones_of_eligible_inputs(self):
        clean = self.X_test.astype(float)
        corrupted, _ = self.bank.corrupt("under_recording", 0.8, 0)
        changed = corrupted != clean
        self.assertTrue(changed.to_numpy().any())
        self.assertTrue(set(changed.columns[changed.any()]) <= set(self.schema.under_recording_columns))
        self.assertTrue((clean.to_numpy()[changed.to_numpy()] == 1.0).all())
        self.assertTrue((corrupted.to_numpy()[changed.to_numpy()] == 0.0).all())

    def test_value_masking_and_noise_respect_availability(self):
        clean = self.X_test.astype(float)
        masked, _ = self.bank.corrupt("value_masking", 1.0, 0)
        self.assertTrue((masked["lab_measured"] == 0).all())
        self.assertTrue((masked["lab"] == FILL).all())
        noisy, _ = self.bank.corrupt("gaussian_noise", 1.0, 0)
        unavailable = clean["lab_measured"] == 0
        self.assertTrue((noisy.loc[unavailable, "lab"] == FILL).all())
        self.assertTrue((noisy.loc[~unavailable, "lab"] != clean.loc[~unavailable, "lab"]).all())


if __name__ == "__main__":
    unittest.main()
