"""Tests of plot_utils: every figure is written, as a non-empty image of the expected format, for typical
inputs and for the edge cases the callers produce (a missing method, a direction without pairs, no
spread across runs, a single panel, ...).

Run from the repository root:

    python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

import numpy
import pandas

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from deap import creator  # noqa: E402

from deap_types import ensure_multi_objective_types  # noqa: E402
from plot_utils import (_panel_grid, plot_2d_heatmap_grid, plot_corruption_curves,  # noqa: E402
                        plot_dependence_shift_summary, plot_feature_count_comparison,
                        plot_multi_objective_convergence, plot_noise_comparison, plot_pareto_front,
                        plot_population_shift_curves, plot_robustness_overview, plot_sign_consistency_comparison,
                        plot_sign_vs_degradation, plot_single_objective_convergence,
                        plot_specificity_sensitivity_curve)

STYLES = {"multi": ("MORSE", "tab:blue", "o", "-"), "single": ("SO-GA", "tab:orange", "s", "--"),
          "forward": ("SFS", "tab:red", "D", ":"), "all": ("All features", "tab:green", "^", "-.")}
LEVELS = [0.9, 0.8, 0.7, 0.6]
PNG = b"\x89PNG"
PDF = b"%PDF"


class PlotTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(plt.close, "all")

    def path(self, name: str) -> str:
        # a sub-folder that does not exist yet: the plots that create their folder must do so
        return os.path.join(self.directory.name, "figures", name)

    def assertImage(self, path: str, signature: bytes = PNG) -> None:
        self.assertTrue(os.path.isfile(path), path)
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(len(signature)), signature, path)
        self.assertGreater(os.path.getsize(path), 1000)
        self.assertEqual(plt.get_fignums(), [], "the figure must be closed")


class TrainingFigureTests(PlotTestCase):
    def test_single_objective_convergence_with_and_without_the_minimum(self):
        stats = [{"gen": g, "max": 0.7 + 0.01 * g, "avg": 0.6 + 0.01 * g, "min": 0.5} for g in range(5)]
        plot_single_objective_convergence(stats, self.path("single.png"))
        self.assertImage(self.path("single.png"))
        plot_single_objective_convergence([{k: v for k, v in s.items() if k != "min"} for s in stats],
                                          self.path("single_no_min.png"))
        self.assertImage(self.path("single_no_min.png"))

    def test_multi_objective_convergence(self):
        stats = [{"gen": g, "auc_max": 0.8, "auc_mean": 0.7, "sign_max": 0.9, "sign_mean": 0.8,
                  "pareto_size": 3 + g} for g in range(4)]
        plot_multi_objective_convergence(stats, self.path("multi.png"))
        self.assertImage(self.path("multi.png"))

    def front(self, points):
        ensure_multi_objective_types()
        individuals = []
        for auc, sign in points:
            individual = creator.Individual([1, 0, 1])
            individual.fitness.values = (auc, sign)
            individuals.append(individual)
        return individuals

    def test_pareto_front_with_distinct_and_with_coinciding_candidates(self):
        # knee, best AUC and best sign consistency are three different points, plus other points
        plot_pareto_front(self.front([(1.0, 0.5), (0.6, 1.0), (0.96, 0.95), (0.8, 0.75)]), self.path("front.png"))
        self.assertImage(self.path("front.png"))
        # one point is the knee and both extremes; no other points
        plot_pareto_front(self.front([(0.9, 0.9)]), self.path("front_single.png"))
        self.assertImage(self.path("front_single.png"))

    def test_specificity_sensitivity_curve_is_a_pdf(self):
        n = 50
        plot_specificity_sensitivity_curve(numpy.linspace(0, 1, n), numpy.linspace(1, 0, n), numpy.linspace(0, 1, n),
                                           25, self.path("curve.pdf"))
        self.assertImage(self.path("curve.pdf"), PDF)

    def test_feature_count_and_sign_consistency_boxplots(self):
        rows = [{"method": method, "seed": seed, "n_features": count + seed % 3,
                 "sign_consistency": value - 0.01 * (seed % 2)}
                for seed in range(6) for method, count, value in (("MORSE", 10, 0.95), ("SO-GA", 20, 0.7))]
        frame = pandas.DataFrame(rows)
        palette = {"MORSE": "tab:blue", "SO-GA": "tab:orange"}
        os.makedirs(os.path.dirname(self.path("x")), exist_ok=True)     # this plot expects an existing folder
        plot_feature_count_comparison(frame, ["MORSE", "SO-GA"], palette, 30, 6, self.path("counts.png"))
        self.assertImage(self.path("counts.png"))
        plot_sign_consistency_comparison(frame, ["MORSE", "SO-GA"], palette, 6, self.path("sign.png"))
        self.assertImage(self.path("sign.png"))


class RobustnessFigureTests(PlotTestCase):
    def curves(self, keys: dict, value: float = 0.8) -> pandas.DataFrame:
        rows = []
        for key_values in keys:
            for method in STYLES:
                rows.append({**key_values, "method": method, "mean": value,
                             "sd": numpy.nan if method in ("forward", "all") else 0.01})
        return pandas.DataFrame(rows)

    def test_population_curves_with_several_and_with_one_panel(self):
        keys = [{"axis": axis, "position": position} for axis in ("PC1", "PC2", "PC3") for position in range(-4, 5)]
        curves = self.curves(keys)
        curves = curves[~((curves["axis"] == "PC3") & (curves["method"] == "forward"))]     # a missing line
        panels = [(axis, "low", "high") for axis in ("PC1", "PC2", "PC3")]
        plot_population_shift_curves(curves, panels, LEVELS, STYLES, self.path("population.png"), "t")
        self.assertImage(self.path("population.png"))
        plot_population_shift_curves(curves, panels[:1], LEVELS, STYLES, self.path("population1.png"), "t")
        self.assertImage(self.path("population1.png"))

    def test_dependence_summary_with_both_directions_and_with_one_missing(self):
        keys = [{"direction": d, "ess_level": level} for d in ("decorrelate", "strengthen") for level in LEVELS]
        curves = self.curves(keys, value=-0.01)
        differences = pandas.DataFrame([{"direction": d, "ess_level": level, "scenario": f"p{i}",
                                         "difference": 0.001 * (i - 3)}
                                        for d in ("decorrelate", "strengthen") for level in LEVELS for i in range(8)])
        plot_dependence_shift_summary(curves, differences, LEVELS, STYLES, self.path("dependence.png"), "t")
        self.assertImage(self.path("dependence.png"))
        only_decorrelate = curves[curves["direction"] == "decorrelate"]
        plot_dependence_shift_summary(only_decorrelate, differences[differences["direction"] == "decorrelate"],
                                      LEVELS, STYLES, self.path("dependence1.png"), "t")
        self.assertImage(self.path("dependence1.png"))
        plot_dependence_shift_summary(only_decorrelate, pandas.DataFrame(), LEVELS,
                                      {k: STYLES[k] for k in ("multi", "single")}, self.path("dependence0.png"), "t")
        self.assertImage(self.path("dependence0.png"))

    def test_corruption_curves(self):
        keys = [{"family": family, "level": level} for family in ("gaussian_noise", "binary_redraw", "value_masking")
                for level in (0.0, 0.5, 1.0)]
        families = [("gaussian_noise", "Noise", "level"), ("binary_redraw", "Redraw", "level"),
                    ("value_masking", "Masking", "level")]
        curves = self.curves(keys)
        curves = curves[~((curves["family"] == "value_masking") & (curves["method"] == "all"))]   # a missing line
        plot_corruption_curves(curves, families, STYLES, self.path("corruption.png"), "t")
        self.assertImage(self.path("corruption.png"))

    def test_overview_with_and_without_a_worst_case_panel(self):
        rows = [{"family": family, "method": method, "clean_mean": 0.85, "clean_sd": 0.01, "stressed_mean": 0.8,
                 "stressed_sd": numpy.nan if method == "all" else 0.02,
                 "worst_mean": 0.75 if family == "population" else numpy.nan, "worst_sd": 0.02}
                for family in ("population", "gaussian_noise") for method in STYLES]
        overview = pandas.DataFrame(rows)
        overview = overview[~((overview["family"] == "gaussian_noise") & (overview["method"] == "forward"))]
        labels = {"population": "Population", "gaussian_noise": "Noise"}
        plot_robustness_overview(overview, labels, STYLES, self.path("overview.png"), "t")
        self.assertImage(self.path("overview.png"))
        plot_robustness_overview(overview.assign(worst_mean=numpy.nan), labels, STYLES,
                                 self.path("overview1.png"), "t")
        self.assertImage(self.path("overview1.png"))

    def test_sign_vs_degradation_over_several_rows_of_panels(self):
        families = ["population", "dependence_decorrelate", "gaussian_noise", "binary_redraw", "under_recording"]
        rng = numpy.random.default_rng(0)
        points = pandas.DataFrame([{"family": family, "method": method, "seed": seed,
                                    "sign_consistency": rng.uniform(0.6, 1.0), "n_features": rng.integers(5, 50),
                                    "delta_roc_auc": rng.normal(-0.02, 0.01)}
                                   for family in families for method in ("multi", "single") for seed in range(5)])
        correlations = pandas.DataFrame([{"family": family, "n_models": 10, "spearman": 0.1,
                                          "partial_spearman_given_size": -0.1} for family in families[:-1]])
        plot_sign_vs_degradation(points, correlations, {f: f for f in families},
                                 {k: STYLES[k] for k in ("multi", "single")}, self.path("sign_deg.png"), "t")
        self.assertImage(self.path("sign_deg.png"))

    def test_panel_grid_hides_the_unused_panels(self):
        fig, axes = _panel_grid(5, 4, 2.0, 2.0)
        self.assertEqual(axes.shape, (2, 4))
        self.assertEqual([ax.get_visible() for ax in axes.flat], [True] * 5 + [False] * 3)
        plt.close(fig)


class LegacyGridFigureTests(PlotTestCase):
    def grid(self) -> pandas.DataFrame:
        rows = [{"seed": seed, "noise_level": n, "mean_shift": s,
                 **{f"auc_{key}": 0.8 - 0.1 * n + 0.02 * s + 0.01 * seed for key in STYLES}}
                for seed in (1, 2) for n in (0.0, 0.5, 1.0) for s in (-1.0, 0.0, 1.0)]
        return pandas.DataFrame(rows)

    def test_noise_comparison_with_and_without_spread(self):
        grid = self.grid()
        agg = grid[grid["mean_shift"] == 0.0].drop(columns=["seed", "mean_shift"]).groupby("noise_level").agg(["mean", "std"])
        plot_noise_comparison(agg, STYLES, "noise", "AUC", "t", self.path("noise.png"))
        self.assertImage(self.path("noise.png"))
        one_seed = grid[(grid["seed"] == 1) & (grid["mean_shift"] == 0.0)].drop(columns=["seed", "mean_shift"])
        plot_noise_comparison(one_seed.groupby("noise_level").agg(["mean", "std"]), STYLES, "noise", "AUC", "t",
                              self.path("noise1.png"))           # std is NaN with one seed: no band
        self.assertImage(self.path("noise1.png"))

    def test_heatmap_grid_with_and_without_aurs(self):
        agg = self.grid().drop(columns=["seed"]).groupby(["noise_level", "mean_shift"]).mean()
        plot_2d_heatmap_grid(agg, STYLES, "AUC", "noise", "shift", "t", self.path("heat.png"),
                             aurs_scores={key: 0.95 for key in STYLES})
        self.assertImage(self.path("heat.png"))
        plot_2d_heatmap_grid(agg, STYLES, "AUC", "noise", "shift", "t", self.path("heat0.png"))
        self.assertImage(self.path("heat0.png"))


if __name__ == "__main__":
    unittest.main()
