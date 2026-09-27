"""Tests of legacy_stress_evaluation: the noise x covariate-shift grid, the 0/1 re-draw sweep and AURS of the
earlier notebook, on synthetic data and on a synthetic checkpointed run (command line).

The grid was also checked against real runs: on the four paper runs of 2026-09-25 it reproduces the stored
CSVs of evaluation/all_models_comparison/ byte for byte.

Run from the repository root:

    python -m unittest discover -s tests -v
"""
import contextlib
import io
import os
import runpy
import sys
import tempfile
import unittest
from unittest import mock

import numpy
import pandas
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import legacy_stress_evaluation as legacy  # noqa: E402
from evaluation_utils import (apply_proportional_noise, compute_aurs, covariate_shift_weights,  # noqa: E402
                              build_model_package, fit_covariate_shift_axis, get_continuous_columns,
                              predict_scores)
from test_robustness_evaluation import write_run_folder  # noqa: E402
from test_robustness_utils import make_data  # noqa: E402

METHODS = ("multi", "single", "all", "forward")
PNGS = ("gaussian_noise_comparison_test.png", "covariate_shift_comparison_test.png",
        "dummy_flip_comparison_test.png", "gaussian_2d_heatmap_grid_test.png")
CSVS = ("gaussian_2d_per_seed.csv", "gaussian_noise_per_seed.csv", "covariate_shift_per_seed.csv",
        "dummy_flip_per_seed.csv", "aurs_scores.csv")


def packages_for(X: pandas.DataFrame, y: numpy.ndarray, seeds, methods=METHODS) -> dict:
    features = list(X.columns)
    n = len(features)
    rng = numpy.random.default_rng(5)
    return {seed: {method: build_model_package(list((rng.random(n) < 0.6).astype(int)) if method != "all" else [1] * n,
                                               features, X, pandas.Series(y), seed=seed)
                   for method in methods} for seed in seeds}


class GridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.y_train = make_data(400, seed=81)
        cls.X_test, y_test = make_data(200, seed=82)
        cls.y_test = pandas.Series(y_test)
        cls.packages = packages_for(cls.X_train, cls.y_train, (3, 4))
        cls.directory = tempfile.TemporaryDirectory()
        cls.out = os.path.join(cls.directory.name, "grid")
        cls.results = legacy.run_legacy_stress_grid(cls.X_train, cls.X_test, cls.y_test, cls.packages, cls.out,
                                                    log=lambda line: None)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_every_output_is_written(self):
        for name in CSVS + PNGS:
            self.assertTrue(os.path.isfile(os.path.join(self.out, name)), name)
        grid = pandas.read_csv(os.path.join(self.out, "gaussian_2d_per_seed.csv"))
        self.assertEqual(list(grid.columns), ["seed", "noise_level", "mean_shift"] + [f"auc_{m}" for m in METHODS])
        self.assertEqual(len(grid), 2 * 11 * 11)
        aurs = pandas.read_csv(os.path.join(self.out, "aurs_scores.csv"))
        self.assertEqual(aurs["model_key"].tolist(), list(METHODS))
        self.assertEqual(aurs["method"].iloc[0], "Multi-Objective (MORSE)")

    def test_the_clean_cell_and_the_slices(self):
        grid = self.results.grid
        clean = grid[(grid["noise_level"] == 0.0) & (grid["mean_shift"] == 0.0)].set_index("seed")
        redraw = self.results.redraw
        for seed in (3, 4):
            for method in METHODS:
                expected = roc_auc_score(self.y_test, predict_scores(self.packages[seed][method], self.X_test))
                self.assertAlmostEqual(clean.loc[seed, f"auc_{method}"], expected, places=12)
                self.assertAlmostEqual(redraw[(redraw["seed"] == seed) & (redraw["flip_rate"] == 0.0)][
                    f"auc_{method}"].iloc[0], expected, places=12)
        self.assertEqual(len(self.results.gaussian), 2 * 11)
        self.assertEqual(len(self.results.shift), 2 * 11)
        self.assertTrue(numpy.allclose(self.results.shift["mean_shift"].unique(), numpy.round(numpy.arange(-1, 1.1, 0.2), 2)))

    def test_the_draws_are_those_of_the_earlier_notebook(self):
        # the global state is seeded with the seed, then the noise levels are drawn in order (none at 0)
        seed = 3
        numpy.random.seed(seed)
        train_std = self.X_train.std()
        continuous = get_continuous_columns(self.X_train)
        apply_proportional_noise(self.X_test, train_std, 0.0, continuous)
        noisy = apply_proportional_noise(self.X_test, train_std, 0.1, continuous)
        weights = covariate_shift_weights(fit_covariate_shift_axis(self.X_train), self.X_test, self.y_test, 0.4)
        expected = roc_auc_score(self.y_test, predict_scores(self.packages[seed]["multi"], noisy), sample_weight=weights)
        grid = self.results.grid
        row = grid[(grid["seed"] == seed) & (grid["noise_level"] == 0.1) & numpy.isclose(grid["mean_shift"], 0.4)]
        self.assertAlmostEqual(row["auc_multi"].iloc[0], expected, places=12)

    def test_a_second_run_is_identical_and_aurs_matches(self):
        out = os.path.join(self.directory.name, "again")
        legacy.run_legacy_stress_grid(self.X_train, self.X_test, self.y_test, self.packages, out, log=lambda line: None)
        for name in CSVS:
            with open(os.path.join(self.out, name), "rb") as first, open(os.path.join(out, name), "rb") as second:
                self.assertEqual(first.read(), second.read(), name)
        heatmap_agg = self.results.grid.drop(columns=["seed"]).groupby(["noise_level", "mean_shift"]).mean()
        for method in METHODS:
            self.assertAlmostEqual(self.results.aurs[method], compute_aurs(heatmap_agg, method), places=12)

    def test_pr_auc_and_a_missing_method(self):
        packages = {seed: {m: p for m, p in methods.items() if m in ("multi", "single")}
                    for seed, methods in self.packages.items()}
        printed = []
        results = legacy.run_legacy_stress_grid(self.X_train, self.X_test, self.y_test, packages,
                                                os.path.join(self.directory.name, "pr"), use_roc_auc=False,
                                                log=printed.append)
        self.assertEqual(set(results.aurs), {"multi", "single"})
        clean = results.grid[(results.grid["noise_level"] == 0.0) & (results.grid["mean_shift"] == 0.0)]
        expected = average_precision_score(self.y_test, predict_scores(packages[3]["single"], self.X_test))
        self.assertAlmostEqual(clean.loc[clean["seed"] == 3, "auc_single"].iloc[0], expected, places=12)
        self.assertTrue(any("clean-test PR-AUC" in line for line in printed))


class CommandLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.run_directory = os.path.join(cls.temporary.name, "2026-01-01_00-00-00_run")
        os.makedirs(cls.run_directory)
        write_run_folder(cls.run_directory)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_main(self, *arguments: str) -> str:
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            legacy.main(["--run", self.run_directory, "--seeds", "1", *arguments])
        return printed.getvalue()

    def test_the_default_folder_is_the_one_earlier_runs_used(self):
        printed = self.run_main()
        self.assertIn("MORSE = the max-S end", printed)                 # the archived notebook's rule
        folder = os.path.join(self.run_directory, "evaluation", "all_models_comparison")
        for name in CSVS + PNGS:
            self.assertTrue(os.path.isfile(os.path.join(folder, name)), name)

    def test_another_rule_or_metric_gets_its_own_folder(self):
        self.run_main("--selection", "knee")
        self.assertTrue(os.path.isfile(os.path.join(self.run_directory, "evaluation", "all_models_comparison_knee",
                                                    "aurs_scores.csv")))
        printed = self.run_main("--metric", "pr")
        self.assertIn("clean-test PR-AUC", printed)
        self.assertTrue(os.path.isfile(os.path.join(self.run_directory, "evaluation", "all_models_comparison_pr",
                                                    "aurs_scores.csv")))
        out = os.path.join(self.temporary.name, "chosen")
        self.run_main("--out", out, "--selection", "max_s", "--metric", "roc")
        self.assertTrue(os.path.isfile(os.path.join(out, "gaussian_2d_per_seed.csv")))

    def test_the_script_entry_point(self):
        path = os.path.join(os.path.dirname(HERE), "legacy_stress_evaluation.py")
        with mock.patch.object(sys, "argv", [path, "--help"]):
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                with self.assertRaises(SystemExit) as stopped:
                    runpy.run_path(path, run_name="__main__")
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("--metric", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
