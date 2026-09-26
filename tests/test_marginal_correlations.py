"""Tests of evaluation_utils.compute_marginal_correlations, the reference direction of the sign-consistency
objective.

The regression they guard against: the GA computes the correlations on STANDARDISED fold data, and an earlier
version cast two-valued columns to int before a Matthews coefficient, which reversed the sign for every 0/1 column
with a prevalence above 0.5.

Run from the repository root:

    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy
import pandas
from scipy.stats import pointbiserialr
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation_utils import compute_marginal_correlations  # noqa: E402

N_ROWS: int = 5000
PREVALENCES: tuple[float, ...] = (0.03, 0.2, 0.45, 0.55, 0.8, 0.97)


def make_data(seed: int = 0) -> tuple[pandas.DataFrame, numpy.ndarray]:
    """Binary columns of several prevalences (positively and negatively associated with y), two
    continuous columns and a constant column."""
    rng = numpy.random.default_rng(seed)
    latent = rng.normal(size=N_ROWS)
    y = (latent + rng.normal(size=N_ROWS) > 0).astype(int)
    columns: dict[str, numpy.ndarray] = {}
    for prevalence in PREVALENCES:
        threshold = numpy.quantile(latent + rng.normal(size=N_ROWS), 1.0 - prevalence)
        columns[f"up_{prevalence}"] = (latent + rng.normal(size=N_ROWS) > threshold).astype(int)
        columns[f"down_{prevalence}"] = (-latent + rng.normal(size=N_ROWS) > -threshold).astype(int)
    columns["continuous_up"] = latent + rng.normal(size=N_ROWS)
    columns["continuous_down"] = -2.0 * latent + rng.normal(size=N_ROWS) + 5.0
    columns["constant"] = numpy.full(N_ROWS, 3.0)
    return pandas.DataFrame(columns), y


class MarginalCorrelationTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_data()

    def test_binary_columns_give_the_phi_coefficient(self):
        result = pandas.Series(compute_marginal_correlations(self.X, self.y), index=self.X.columns)
        for name in self.X.columns:
            if set(numpy.unique(self.X[name])) == {0, 1}:
                self.assertAlmostEqual(result[name], matthews_corrcoef(self.y, self.X[name]), places=10, msg=name)

    def test_continuous_columns_give_the_point_biserial_correlation(self):
        result = pandas.Series(compute_marginal_correlations(self.X, self.y), index=self.X.columns)
        for name in ("continuous_up", "continuous_down"):
            self.assertAlmostEqual(result[name], pointbiserialr(self.y, self.X[name])[0], places=10, msg=name)

    def test_standardising_the_inputs_changes_nothing(self):
        # The regression: the GA passes standardised data, the evaluation raw data.
        raw = compute_marginal_correlations(self.X, self.y)
        standardised = compute_marginal_correlations(StandardScaler().fit_transform(self.X), self.y)
        numpy.testing.assert_allclose(standardised, raw, atol=1e-12)

    def test_dense_binary_columns_keep_their_sign_after_standardisation(self):
        standardised = pandas.Series(
            compute_marginal_correlations(StandardScaler().fit_transform(self.X), self.y), index=self.X.columns)
        for prevalence in (0.55, 0.8, 0.97):
            self.assertGreater(standardised[f"up_{prevalence}"], 0.0, msg=prevalence)
            self.assertLess(standardised[f"down_{prevalence}"], 0.0, msg=prevalence)

    def test_constant_column_and_constant_target_give_zero(self):
        result = pandas.Series(compute_marginal_correlations(self.X, self.y), index=self.X.columns)
        self.assertEqual(result["constant"], 0.0)
        numpy.testing.assert_array_equal(
            compute_marginal_correlations(self.X, numpy.ones(N_ROWS, dtype=int)), numpy.zeros(self.X.shape[1]))

    def test_array_and_dataframe_inputs_agree(self):
        numpy.testing.assert_array_equal(compute_marginal_correlations(self.X, self.y),
                                         compute_marginal_correlations(self.X.to_numpy(), pandas.Series(self.y)))


if __name__ == "__main__":
    unittest.main()
