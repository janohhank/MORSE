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
from robustness_utils import (BALANCE_TOLERANCE, CorruptionBank, DependenceTilt, NormalScores,  # noqa: E402
                              calibrate_correlation, calibrate_strength, calibrate_strengths,
                              class_effective_sizes, dependence_pairs,
                              effective_sample_fraction, exponential_tilt, infer_feature_schema,
                              normalise_within_classes, population_axes, weighted_average_precision,
                              weighted_correlation, weighted_roc_auc)

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
        self.assertEqual(indicator.named, ["lab"])                     # `lab_measured` names `lab`
        self.assertEqual(self.schema.unresolved_availability, [])

    def test_unseen_combinations_are_found_as_a_diagnostic(self):
        unseen = {(f.a, f.a_value, f.b, f.b_value) for f in self.schema.unseen_combinations}
        self.assertTrue(("general", 0, "specific", 1) in unseen or ("specific", 1, "general", 0) in unseen)
        self.assertTrue(("female", 1, "male_code", 1) in unseen or ("male_code", 1, "female", 1) in unseen)
        self.assertTrue(("state_A", 1, "state_B", 1) in unseen)
        self.assertFalse(any({f[0], f[2]} == {"med_a", "med_b"} for f in unseen))

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

    def test_redraw_keeps_the_structure_and_counts_unseen_combinations(self):
        clean = self.X_test
        corrupted, counts = self.bank.corrupt("binary_redraw", 1.0, 0)
        self.assertTrue((corrupted[["state_A", "state_B", "state_C"]].sum(axis=1) == 1).all())
        self.assertTrue((corrupted[["region_N", "region_S"]].sum(axis=1) <= 1).all())
        # a combination never seen in training is an association, not an impossibility: the re-draw of
        # independent inputs creates it, and the diagnostics count it
        created = (corrupted["specific"] == 1) & (corrupted["general"] == 0) & ~(
            (clean["specific"] == 1) & (clean["general"] == 0))
        self.assertTrue(created.any())
        self.assertGreater(counts["unseen_combinations"], 0)
        self.assertGreaterEqual(counts["unseen_combinations"], counts["rows_with_unseen_combination"])
        self.assertGreater(counts["rows_with_unseen_combination"], 0)
        self.assertEqual(self.bank.corrupt("binary_redraw", 0.0, 0)[1]["unseen_combinations"], 0)
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
        masked, counts = self.bank.corrupt("value_masking", 1.0, 0)
        self.assertTrue((masked["lab_measured"] == 0).all())
        self.assertTrue((masked["lab"] == FILL).all())
        # level 1 withholds every available value: nothing is undone
        self.assertEqual(counts["changed"], counts["eligible"])
        self.assertEqual(counts["eligible"], int((clean["lab_measured"] == 1).sum()))
        noisy, _ = self.bank.corrupt("gaussian_noise", 1.0, 0)
        unavailable = clean["lab_measured"] == 0
        self.assertTrue((noisy.loc[unavailable, "lab"] == FILL).all())
        self.assertTrue((noisy.loc[~unavailable, "lab"] != clean.loc[~unavailable, "lab"]).all())


def make_availability_data(n: int, seed: int) -> pandas.DataFrame:
    """Two values with availability flags that are linked, like College Scorecard's admission rates:
      rate / rate:missing           missing in 20% of the rows (fill 0.75), never when reports_rate = 1
      rate_all / rate_all:missing   missing in 10% of the rows (fill 0.5), only where rate is missing too
      reports_rate                  a 0/1 input: 1 = the rate is always reported
    so rate_all:missing = 1 without rate:missing = 1, and rate:missing = 1 with reports_rate = 1, never occur."""
    rng = numpy.random.default_rng(seed)
    reports = (rng.random(n) < 0.4).astype(int)
    rate_missing = ((rng.random(n) < 1 / 3) & (reports == 0)).astype(int)
    rate_all_missing = rate_missing * (rng.random(n) < 0.5).astype(int)
    return pandas.DataFrame({
        "x": rng.normal(size=n),
        "rate": numpy.where(rate_missing == 1, 0.75, rng.uniform(0.1, 0.9, n)),
        "rate_all": numpy.where(rate_all_missing == 1, 0.5, rng.uniform(0.1, 0.9, n)),
        "rate:missing": rate_missing, "rate_all:missing": rate_all_missing, "reports_rate": reports})


class ConfigTests(unittest.TestCase):
    def test_defaults_are_valid_and_serialisable(self):
        config = RobustnessConfig()
        self.assertEqual(config.to_dict()["ess_levels"], (0.9, 0.8, 0.7, 0.6))
        self.assertIn(config.headline_ess, config.ess_levels)

    def test_invalid_settings_are_refused(self):
        invalid = [
            {"ess_levels": ()}, {"ess_levels": (0.9, 1.0)}, {"ess_levels": (0.6, 0.8), "headline_ess": 0.6},
            {"corruption_levels": (0.0, 0.5)}, {"corruption_levels": (0.5, 0.2), "headline_corruption_level": 0.5},
            {"headline_ess": 0.5}, {"headline_corruption_level": 0.55},
            {"pair_min_abs_correlation": 0.6, "pair_max_abs_correlation": 0.5},
            {"corruption_repetitions": 0}, {"max_pairs": -1}, {"population_components": -1},
            {"dependence_weaken_levels": ()}, {"dependence_weaken_levels": (0.5, 1.2)},
            {"dependence_weaken_levels": (1.0, 0.5)}, {"dependence_strengthen_levels": (0.0, 0.5)},
            {"dependence_strengthen_levels": (0.25, 0.25, 0.5)}, {"headline_weaken": 0.3},
            {"headline_strengthen": 0.3}, {"min_train_ess_fraction": 0.0}]
        for settings in invalid:
            with self.assertRaises(ValueError, msg=str(settings)):
                RobustnessConfig(**settings)


class CalibrationEdgeCaseTests(unittest.TestCase):
    def test_invalid_targets_are_refused(self):
        with self.assertRaises(ValueError):
            calibrate_strengths(lambda s: 1.0, [1.2])
        with self.assertRaises(ValueError):
            calibrate_strengths(lambda s: 1.0, [0.6, 0.8])
        with self.assertRaises(ValueError):
            NormalScores([])

    def test_an_inconsistent_ess_function_is_reported_as_saturated(self):
        calls = {"n": 0}

        def ess(strength: float) -> float:
            # decreases while the bracket is searched (5 calls), then answers "no shift" for every
            # strength, so the root finder sees no sign change inside the bracket
            calls["n"] += 1
            return max(0.0, 1.0 - strength) if calls["n"] <= 5 else 1.0

        self.assertEqual(calibrate_strengths(ess, [0.6]), [(20.0, True)])

    def test_a_target_the_function_jumps_over_is_reported_as_saturated(self):
        results = calibrate_strengths(lambda s: 1.0 if s < 0.5 else 0.3, [0.9, 0.6])
        self.assertEqual(results, [(20.0, True), (20.0, True)])

    def test_class_effective_sizes(self):
        negatives, positives = class_effective_sizes(numpy.array([1.0, 1.0, 2.0, 0.0]), numpy.array([0, 0, 1, 1]))
        self.assertEqual((negatives, positives), (2.0, 1.0))


class SchemaEdgeCaseTests(unittest.TestCase):
    def test_missing_values_are_refused(self):
        with self.assertRaises(ValueError):
            infer_feature_schema(pandas.DataFrame({"a": [1.0, numpy.nan, 2.0]}))

    def test_constant_and_two_valued_inputs(self):
        rng = numpy.random.default_rng(3)
        frame = pandas.DataFrame({"const": numpy.ones(100), "two": rng.choice([2.0, 5.0], 100),
                                  "flag": (rng.random(100) < 0.5).astype(int)})
        schema = infer_feature_schema(frame)
        self.assertEqual((schema.constant, schema.other, schema.binary, schema.continuous),
                         (["const"], ["two"], ["flag"], []))
        # no continuous input: no value/availability pairs; a single 0/1 input: no unseen combinations
        self.assertEqual((schema.value_indicators, schema.unseen_combinations), ([], []))
        summary = schema.summary()
        self.assertEqual((summary["constant"], summary["other"], summary["binary"]), (1, 1, 1))
        self.assertEqual(set(schema.to_dict()), {"summary", "binary", "continuous", "constant", "other",
                                                 "one_hot_groups", "value_indicators", "unresolved_availability",
                                                 "unseen_combinations", "stand_alone_binary",
                                                 "under_recording_inputs"})

    def test_a_flag_level_with_too_few_rows_is_no_statistical_availability_flag(self):
        rng = numpy.random.default_rng(4)
        flag = numpy.zeros(300, dtype=int)
        flag[:5] = 1                                              # 5 rows < value_indicator_min_rows
        value = numpy.where(flag == 1, 9.0, rng.normal(size=300))
        schema = infer_feature_schema(pandas.DataFrame({"value": value, "flag": flag}))
        self.assertEqual(schema.value_indicators, [])
        # ... but the same flag named after its value is paired: the name is the evidence
        schema = infer_feature_schema(pandas.DataFrame({"value": value, "value:missing": flag}))
        self.assertEqual([(i.flag, i.off_level, i.fills) for i in schema.value_indicators],
                         [("value:missing", 1, {"value": 9.0})])

    def test_dependence_pairs_need_two_inputs_and_enough_rows(self):
        rng = numpy.random.default_rng(5)
        single = pandas.DataFrame({"x": rng.normal(size=200)})
        self.assertEqual(dependence_pairs(single, single, infer_feature_schema(single)), ([], 0))
        x = rng.normal(size=400)
        rare = numpy.zeros(400, dtype=int)
        rare[numpy.argsort(x)[-15:]] = 1                          # 15 ones: correlated with x, but < 20 rows
        frame = pandas.DataFrame({"x": x, "rare": rare})
        self.assertGreater(abs(numpy.corrcoef(x, rare)[0, 1]), 0.3)
        self.assertEqual(dependence_pairs(frame, frame, infer_feature_schema(frame)), ([], 0))


class UnbalanceableTiltTests(unittest.TestCase):
    def test_a_dependence_that_cannot_be_strengthened_with_fixed_moments_gives_nan(self):
        rng = numpy.random.default_rng(0)
        a = rng.integers(0, 10, 400).astype(float)
        b = numpy.round(a + rng.normal(scale=0.3, size=a.size))       # r = 0.995 with ten values
        tilt = DependenceTilt(a, b, a, b, False, False, clip=2.5)
        self.assertLess(tilt.train_ess(-5.0), 0.99)                    # weakening works
        self.assertLess(tilt.balance_error, BALANCE_TOLERANCE)
        self.assertTrue(numpy.isnan(tilt.train_ess(20.0)))             # strengthening that far does not
        self.assertGreater(tilt.balance_error, BALANCE_TOLERANCE)
        self.assertEqual(calibrate_strengths(tilt.train_ess, [0.8, 0.6]), [(20.0, True), (20.0, True)])


class CorrelationCalibrationTests(unittest.TestCase):
    """Dependence severities on the correlation scale (review point 5): weakening stops at zero."""

    def setUp(self):
        rng = numpy.random.default_rng(41)
        x = rng.normal(size=2000)
        z = 0.5 * x + 0.85 * rng.normal(size=2000)
        self.tilt = DependenceTilt(x, z, x[:500], z[:500], False, False, clip=2.5)
        self.clean = self.tilt.score_correlation(0.0)

    def test_the_clean_score_correlation(self):
        scores = self.tilt._train_stats[:, :2]
        self.assertAlmostEqual(self.clean, float(numpy.corrcoef(scores.T)[0, 1]), places=12)
        self.assertGreater(self.clean, 0.4)

    def test_weakening_reaches_every_target_and_stops_at_zero(self):
        targets = [0.75 * self.clean, 0.5 * self.clean, 0.25 * self.clean, 0.0]
        results = calibrate_correlation(self.tilt, targets)
        self.assertTrue(all(reached for _, reached in results))
        strengths = [strength for strength, _ in results]
        self.assertTrue(all(s < 0 for s in strengths))                         # a positive r is lowered
        self.assertEqual(strengths, sorted(strengths, reverse=True))           # further = stronger
        for (strength, _), target in zip(results, targets):
            self.assertAlmostEqual(self.tilt.score_correlation(strength), target, delta=1e-6)

    def test_an_impossible_strengthening_is_unattainable(self):
        results = calibrate_correlation(self.tilt, [1.2 * self.clean, 0.9999])
        self.assertTrue(results[0][1])
        self.assertAlmostEqual(self.tilt.score_correlation(results[0][0]), 1.2 * self.clean, delta=1e-6)
        self.assertFalse(results[1][1])
        self.assertTrue(numpy.isnan(results[1][0]))

    def test_invalid_targets(self):
        self.assertEqual(calibrate_correlation(self.tilt, []), [])
        with self.assertRaises(ValueError):
            calibrate_correlation(self.tilt, [0.5 * self.clean, 0.8 * self.clean])      # moving back
        with self.assertRaises(ValueError):
            calibrate_correlation(self.tilt, [0.5 * self.clean, 1.2 * self.clean])      # both sides
        with self.assertRaises(ValueError):
            calibrate_correlation(self.tilt, [self.clean])                              # no move at all

    def test_a_target_the_balancing_cannot_reach(self):
        rng = numpy.random.default_rng(0)
        a = rng.integers(0, 10, 400).astype(float)
        b = numpy.round(a + rng.normal(scale=0.3, size=a.size))       # score correlation 0.990, ten values
        tilt = DependenceTilt(a, b, a, b, False, False, clip=2.5)
        self.assertAlmostEqual(tilt.score_correlation(0.0), 0.990, delta=0.001)
        strength, reached = calibrate_correlation(tilt, [0.995])[0]
        self.assertFalse(reached)
        self.assertTrue(numpy.isnan(strength))
        at_most: float = tilt.score_correlation(20.0)                   # the strongest tilt stays short of it
        self.assertTrue(numpy.isnan(at_most) or at_most < 0.995)


class CorruptionBankErrorTests(unittest.TestCase):
    def setUp(self):
        X, _ = make_data(300, seed=12)
        self.bank = CorruptionBank(infer_feature_schema(X), X, X, repetitions=2, seed=1)

    def test_invalid_requests_are_refused(self):
        with self.assertRaises(ValueError):
            self.bank.corrupt("gaussian_noise", 1.5, 0)
        with self.assertRaises(ValueError):
            self.bank.corrupt("gaussian_noise", 0.5, 2)                # only repetitions 0 and 1 exist
        with self.assertRaises(ValueError):
            self.bank.corrupt("salt_and_pepper", 0.5, 0)

    def test_describe_counts_the_units(self):
        self.assertEqual(self.bank.describe(), {"gaussian_noise": 3, "binary_redraw": 11, "under_recording": 6,
                                                "value_masking": 1})

    def test_only_continuous_inputs(self):
        rng = numpy.random.default_rng(13)
        frame = pandas.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
        bank = CorruptionBank(infer_feature_schema(frame), frame, frame, repetitions=1, seed=2)
        self.assertEqual(bank.applicable_families(), ["gaussian_noise"])
        noisy, counts = bank.corrupt("gaussian_noise", 0.5, 0)
        self.assertEqual(counts["changed"], 200)
        self.assertFalse(noisy.equals(frame))


class LinkedAvailabilityTests(unittest.TestCase):
    """Two values with linked availability flags, like College Scorecard's admission rates. Each flag is
    masked on its own: "flag off => fill value" always holds, and a combination never seen in training
    (rate_all withheld while rate is reported) is counted, not prevented."""

    @classmethod
    def setUpClass(cls):
        cls.X = make_availability_data(3000, seed=14)
        cls.X_test = make_availability_data(1000, seed=15)
        cls.schema = infer_feature_schema(cls.X)
        cls.bank = CorruptionBank(cls.schema, cls.X, cls.X_test, repetitions=2, seed=3)

    def test_every_flag_is_paired_with_its_own_value_by_name(self):
        indicators = {i.flag: (i.off_level, i.fills, i.named) for i in self.schema.value_indicators}
        self.assertEqual(indicators, {"rate:missing": (1, {"rate": 0.75}, ["rate"]),
                                      "rate_all:missing": (1, {"rate_all": 0.5}, ["rate_all"])})
        unseen = {(f.a, f.a_value, f.b, f.b_value) for f in self.schema.unseen_combinations}
        self.assertIn(("rate:missing", 0, "rate_all:missing", 1), unseen)     # rate_all missing => rate missing
        self.assertIn(("rate:missing", 1, "reports_rate", 1), unseen)          # reported => not missing

    def check_valid(self, corrupted: pandas.DataFrame) -> None:
        self.assertTrue((corrupted.loc[corrupted["rate:missing"] == 1, "rate"] == 0.75).all())
        self.assertTrue((corrupted.loc[corrupted["rate_all:missing"] == 1, "rate_all"] == 0.5).all())

    def test_each_flag_is_masked_on_its_own_and_level_one_withholds_every_value(self):
        for level in (0.3, 0.6, 1.0):
            for repetition in (0, 1):
                corrupted, counts = self.bank.corrupt("value_masking", level, repetition)
                self.check_valid(corrupted)
                self.assertEqual(counts["changed"], counts["selected"])
        full, counts = self.bank.corrupt("value_masking", 1.0, 0)
        self.assertEqual(counts["changed"], counts["eligible"])
        self.assertTrue((full[["rate:missing", "rate_all:missing"]] == 1).all().all())
        self.assertTrue(((full["rate"] == 0.75) & (full["rate_all"] == 0.5)).all())

    def test_an_unseen_combination_is_created_and_counted(self):
        clean = self.X_test
        corrupted, counts = self.bank.corrupt("value_masking", 0.4, 0)
        created = (corrupted["rate_all:missing"] == 1) & (corrupted["rate:missing"] == 0)
        self.assertTrue(created.any())
        self.assertFalse(((clean["rate_all:missing"] == 1) & (clean["rate:missing"] == 0)).any())
        self.assertGreaterEqual(counts["rows_with_unseen_combination"], int(created.sum()))

    def test_masked_cells_stay_nested_across_levels(self):
        clean = self.X_test.astype(float)
        previous = None
        for level in (0.2, 0.4, 0.6, 0.8, 1.0):
            changed = (self.bank.corrupt("value_masking", level, 1)[0] != clean).to_numpy()
            if previous is not None:
                self.assertFalse((previous & ~changed).any(), level)
            previous = changed


def indicators_of(frame: pandas.DataFrame) -> dict[str, tuple[int, dict[str, float], list[str]]]:
    return {i.flag: (i.off_level, i.fills, i.named) for i in infer_feature_schema(frame).value_indicators}


class AvailabilityPairingTests(unittest.TestCase):
    """The pairing of values with their availability flags (review point 1 on RadFusion and College
    Scorecard): by name first, then the statistical rules for values without a flag of their own."""

    def test_a_median_fill_that_is_also_a_common_measured_value_is_paired_by_name(self):
        rng = numpy.random.default_rng(21)
        measured = rng.random(2000) < 0.6
        # like RadFusion's sodium: the fill 136 is the median AND 30% of the measured values
        sodium = numpy.where(measured, rng.choice([134.0, 135.0, 136.0, 137.0, 138.0], 2000,
                                                  p=[0.15, 0.2, 0.3, 0.2, 0.15]), 136.0)
        frame = pandas.DataFrame({"sodium:Value": sodium, "sodium:Binary": measured.astype(int)})
        self.assertEqual(indicators_of(frame), {"sodium:Binary": (0, {"sodium:Value": 136.0}, ["sodium:Value"])})
        # the statistical rule alone misses it -- the failure the names fix -- and says nothing about it
        renamed = frame.rename(columns={"sodium:Binary": "flag"})
        self.assertEqual(indicators_of(renamed), {})
        self.assertEqual(infer_feature_schema(renamed).unresolved_availability, [])

    def test_a_value_with_its_own_flag_is_not_claimed_by_another_flag(self):
        rng = numpy.random.default_rng(22)
        measured = rng.random(2000) < 0.7                       # INR and PTT are measured together
        inr = numpy.where(measured, numpy.round(rng.normal(1.1, 0.1, 2000), 1), 1.1)
        ptt = numpy.where(measured, rng.normal(30.0, 4.0, 2000), 14.2)
        frame = pandas.DataFrame({"inr:Binary": measured.astype(int), "inr:Value": inr,
                                  "ptt:Binary": measured.astype(int), "ptt:Value": ptt})
        # ptt:Value is constant on inr:Binary = 0 too, and inr's fill is common: the old rule paired
        # inr:Binary with ptt:Value and left inr:Value unpaired
        self.assertEqual(indicators_of(frame), {"inr:Binary": (0, {"inr:Value": 1.1}, ["inr:Value"]),
                                                "ptt:Binary": (0, {"ptt:Value": 14.2}, ["ptt:Value"])})

    def test_the_name_rule_needs_the_same_separator(self):
        rng = numpy.random.default_rng(23)
        missing = rng.random(1000) < 0.3
        frame = pandas.DataFrame({"rate:missing": missing.astype(int),
                                  "rate": numpy.where(missing, 0.7, rng.uniform(0, 1, 1000)),
                                  "rate_all": rng.uniform(0, 1, 1000)})           # not governed by the flag
        schema = infer_feature_schema(frame)
        self.assertEqual([(i.flag, i.fills) for i in schema.value_indicators], [("rate:missing", {"rate": 0.7})])
        self.assertEqual(schema.unresolved_availability, [])                    # rate_all was never proposed

    def test_generic_names_propose_nothing(self):
        # like Arrhythmia's feature_1 ... feature_278: the stem "feature" says nothing about availability,
        # and feature_1 = 1 on a few rows where feature_3 happens to be constant is no pairing
        rng = numpy.random.default_rng(26)
        rare = numpy.zeros(300, dtype=int)
        rare[:12] = 1
        frame = pandas.DataFrame({"feature_1": rare, "feature_2": (rng.random(300) < 0.5).astype(int),
                                  "feature_3": numpy.where(rare == 1, 0.0, rng.normal(size=300)),
                                  "feature_4": rng.normal(size=300)})
        schema = infer_feature_schema(frame)
        self.assertFalse(any(indicator.named for indicator in schema.value_indicators))
        self.assertEqual(schema.unresolved_availability, [])

    def test_a_value_the_names_pair_with_two_flags_is_reported(self):
        rng = numpy.random.default_rng(27)
        missing = rng.random(500) < 0.3
        frame = pandas.DataFrame({"rate": numpy.where(missing, 0.5, rng.uniform(0, 1, 500)),
                                  "rate:missing": missing.astype(int), "rate_flag": missing.astype(int)})
        schema = infer_feature_schema(frame)
        self.assertFalse(any("rate" in indicator.named for indicator in schema.value_indicators))
        self.assertEqual(sorted(u.flag for u in schema.unresolved_availability), ["rate:missing", "rate_flag"])
        self.assertIn("several flags", schema.unresolved_availability[0].reason)

    def test_a_pair_the_data_do_not_confirm_is_reported(self):
        rng = numpy.random.default_rng(24)
        frame = pandas.DataFrame({"x": rng.normal(size=500), "x:missing": (rng.random(500) < 0.2).astype(int)})
        schema = infer_feature_schema(frame)
        self.assertEqual(schema.value_indicators, [])
        self.assertEqual([(u.flag, u.value) for u in schema.unresolved_availability], [("x:missing", "x")])
        self.assertIn("not constant", schema.unresolved_availability[0].reason)
        self.assertIn("x:missing", schema.redraw_columns)                       # an ordinary 0/1 input then

    def test_a_flag_shared_by_a_block_of_values(self):
        rng = numpy.random.default_rng(25)
        n = 3000
        act_missing = rng.random(n) < 0.4
        fac_missing = act_missing & (rng.random(n) < 0.1)       # nested: no faculty data => no ACT scores
        small_missing = numpy.zeros(n, dtype=bool)
        small_missing[rng.choice(n, 15, replace=False)] = True  # 15 rows
        measured_math = numpy.round(rng.normal(23.0, 2.0, n))   # the fill 23 is 20% of the measured values
        frame = pandas.DataFrame({
            "act:missing": act_missing.astype(int), "act": numpy.where(act_missing, 22.0, rng.normal(22, 3, n)),
            "act_math": numpy.where(act_missing, 23.0, measured_math),
            "block_a": numpy.where(act_missing, 0.37, rng.uniform(0, 1, n)),   # a rare fill: the share rule
            # a structural zero, not an imputation: 0 is the minimum of a sparse share
            "pct_sparse": numpy.where(act_missing, 0.0, numpy.where(rng.random(n) < 0.4, 0.0, rng.uniform(0, 1, n))),
            "fac:missing": fac_missing.astype(int), "fac": numpy.where(fac_missing, 0.5, rng.uniform(0, 1, n)),
            "small:missing": small_missing.astype(int),
            "small": numpy.where(small_missing, 3.0, rng.normal(3.0, 1.0, n)),
            # the interior value 2 on the 15 rows, and on half of the other rows: 0.5^15 = 3e-5 is no proof
            "small_other": numpy.where(small_missing, 2.0, rng.choice([1.0, 2.0, 3.0], n, p=[0.25, 0.5, 0.25])),
        })
        schema = infer_feature_schema(frame)
        indicators = {i.flag: i for i in schema.value_indicators}
        # act_math: an interior fill on all ~1200 rows where the ACT block is missing, although 20% of the
        # measured rows hold it too -- no chance; it is claimed by fac:missing as well, but belongs to the
        # flag with the most off rows
        self.assertEqual(indicators["act:missing"].fills, {"act": 22.0, "act_math": 23.0, "block_a": 0.37})
        self.assertEqual(indicators["act:missing"].named, ["act"])
        self.assertEqual(indicators["fac:missing"].fills, {"fac": 0.5})
        # the structural zero is neither paired nor reported
        self.assertFalse(any("pct_sparse" in i.fills for i in schema.value_indicators))
        # an interior fill on 15 rows that half of the other rows hold too can be chance: reported, not used
        self.assertEqual(indicators["small:missing"].fills, {"small": 3.0})
        self.assertEqual([(u.flag, u.value) for u in schema.unresolved_availability], [("small:missing", "small_other")])
        self.assertIn("can be chance", schema.unresolved_availability[0].reason)
        self.assertEqual(schema.summary()["values_paired_statistically"], 2)


class ValidityCheckTests(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_data(400, seed=31)
        self.X_test, _ = make_data(200, seed=32)
        self.schema = infer_feature_schema(self.X)
        self.bank = CorruptionBank(self.schema, self.X, self.X_test, repetitions=1, seed=4)
        self.clean = self.X_test.to_numpy(dtype=float)
        self.column = {name: j for j, name in enumerate(self.X_test.columns)}

    def broken(self, change) -> str:
        X = self.clean.copy()
        change(X)
        with self.assertRaises(RuntimeError) as raised:
            self.bank._check(X, "test", 0.5)
        return str(raised.exception)

    def test_every_structural_rule_is_checked(self):
        c = self.column
        self.assertIn("0/1 input", self.broken(lambda X: X.__setitem__((0, c["rare"]), 0.5)))
        self.assertIn("two levels", self.broken(lambda X: X.__setitem__((slice(None), c["region_N"]), 1.0)
                                                or X.__setitem__((slice(None), c["region_S"]), 1.0)))
        self.assertIn("lost its level", self.broken(
            lambda X: [X.__setitem__((0, c[name]), 0.0) for name in ("state_A", "state_B", "state_C")]))
        measured_row = int(numpy.flatnonzero(self.clean[:, c["lab_measured"]] == 1)[0])
        self.assertIn("'lab_measured'", self.broken(lambda X: X.__setitem__((measured_row, c["lab_measured"]), 0.0)))
        self.bank._check(self.clean.copy(), "test", 0.5)          # the clean test set is valid

    def test_a_test_row_that_already_breaks_the_rule_is_left_alone(self):
        X_test = self.X_test.copy()
        row = X_test.index[X_test["lab_measured"] == 0][0]
        X_test.loc[row, "lab"] = 7.0                               # "not measured", yet not the fill value
        bank = CorruptionBank(self.schema, self.X, X_test, repetitions=1, seed=4)
        self.assertEqual(bank.clean_availability_violations, 1)
        for family in bank.applicable_families():
            corrupted, _ = bank.corrupt(family, 1.0, 0)            # no RuntimeError
            self.assertEqual(corrupted.loc[row, "lab"], 7.0)


if __name__ == "__main__":
    unittest.main()
