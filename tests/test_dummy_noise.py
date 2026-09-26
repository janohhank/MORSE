"""Tests of evaluation_utils.apply_dummy_noise (prevalence-preserving noise on binary features).

Run from the repository root:

    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy
import pandas

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation_utils import apply_dummy_noise  # noqa: E402

PREVALENCES: dict[str, float] = {"rare": 0.03, "mid": 0.30, "dense": 0.90}
N_ROWS: int = 20000


def make_frame() -> pandas.DataFrame:
    rng = numpy.random.RandomState(123)
    data = {name: (rng.rand(N_ROWS) < prevalence).astype(int) for name, prevalence in PREVALENCES.items()}
    data["continuous"] = rng.randn(N_ROWS)
    return pandas.DataFrame(data)


class DummyNoiseTests(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)
        self.frame = make_frame()
        self.columns = list(PREVALENCES)
        self.prevalence = pandas.Series(PREVALENCES)

    def test_prevalence_is_preserved_and_every_column_keeps_correlation_one_minus_p(self):
        for p in (0.1, 0.3, 0.5):
            noisy = apply_dummy_noise(self.frame, p, self.columns, self.prevalence)
            for name in self.columns:
                self.assertAlmostEqual(noisy[name].mean(), PREVALENCES[name], delta=0.01, msg=f"{name} p={p}")
                correlation = numpy.corrcoef(self.frame[name], noisy[name])[0, 1]
                self.assertAlmostEqual(correlation, 1.0 - p, delta=0.03, msg=f"{name} p={p}")

    def test_full_noise_removes_all_information_but_keeps_the_prevalence(self):
        noisy = apply_dummy_noise(self.frame, 1.0, self.columns, self.prevalence)
        for name in self.columns:
            self.assertAlmostEqual(noisy[name].mean(), PREVALENCES[name], delta=0.01)
            self.assertAlmostEqual(numpy.corrcoef(self.frame[name], noisy[name])[0, 1], 0.0, delta=0.03)

    def test_the_earlier_three_argument_call_still_works(self):
        noisy = apply_dummy_noise(self.frame, 0.3, self.columns)
        for name in self.columns:
            self.assertAlmostEqual(noisy[name].mean(), self.frame[name].mean(), delta=0.01)

    def test_the_redraw_follows_the_given_training_prevalence(self):
        # test prevalence 0.30, training prevalence 0.10: at p = 1 the column becomes Bernoulli(0.10)
        noisy = apply_dummy_noise(self.frame, 1.0, ["mid"], pandas.Series({"mid": 0.10}))
        self.assertAlmostEqual(noisy["mid"].mean(), 0.10, delta=0.01)

    def test_contract(self):
        before = self.frame.copy(deep=True)
        noisy = apply_dummy_noise(self.frame, 0.4, self.columns, self.prevalence)
        self.assertTrue(self.frame.equals(before), "the input frame must not be modified")
        self.assertTrue(noisy["continuous"].equals(self.frame["continuous"]), "other columns must be untouched")
        self.assertEqual(list(noisy.columns), list(self.frame.columns))
        self.assertTrue((noisy.dtypes == self.frame.dtypes).all())
        self.assertTrue(set(numpy.unique(noisy[self.columns].to_numpy())) <= {0, 1})

        unchanged = apply_dummy_noise(self.frame, 0.0, self.columns, self.prevalence)
        self.assertTrue(unchanged.equals(self.frame))
        self.assertIsNot(unchanged, self.frame)

        # columns that are not in the frame are ignored
        apply_dummy_noise(self.frame, 0.2, self.columns + ["absent"], self.prevalence)

    def test_dtypes_bool_int_and_float_survive(self):
        small = pandas.DataFrame({"b": [True, False] * 50, "i": [1, 0, 0, 0] * 25, "f": [0.0, 1.0] * 50})
        prevalence = pandas.Series({"b": 0.5, "i": 0.25, "f": 0.5})
        noisy = apply_dummy_noise(small, 0.5, ["b", "i", "f"], prevalence)
        self.assertTrue((noisy.dtypes == small.dtypes).all(), noisy.dtypes)

    def test_invalid_noise_fraction(self):
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                apply_dummy_noise(self.frame, bad, self.columns, self.prevalence)

    def test_reproducible_with_the_global_seed(self):
        numpy.random.seed(7)
        first = apply_dummy_noise(self.frame, 0.3, self.columns, self.prevalence)
        numpy.random.seed(7)
        second = apply_dummy_noise(self.frame, 0.3, self.columns, self.prevalence)
        self.assertTrue(first.equals(second))


if __name__ == "__main__":
    unittest.main()
