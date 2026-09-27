"""Tests of training_utils: repository_root, the CSV writer, and the Pareto-front selection helpers.

Run from the repository root:

    python -m unittest discover -s tests -v
"""
import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deap import creator  # noqa: E402

from deap_types import ensure_multi_objective_types  # noqa: E402
from training_utils import (best_auc_index, best_sign_consistency_index, ensure_directory,  # noqa: E402
                            knee_point_index, repository_root, save_stats_csv, select_pareto_individual)


def front(*points: tuple[float, float]) -> list:
    """A Pareto front of individuals with the given (AUC, sign consistency) fitness values."""
    ensure_multi_objective_types()
    individuals = []
    for position, (auc, sign) in enumerate(points):
        individual = creator.Individual([position % 2, 1])
        individual.fitness.values = (auc, sign)
        individuals.append(individual)
    return individuals


class RepositoryRootTests(unittest.TestCase):
    def test_it_is_the_folder_of_the_pipeline_modules(self):
        root = repository_root()
        self.assertTrue(os.path.isabs(root))
        for module in ("training_utils.py", "checkpoint_utils.py", "evaluation_utils.py", "training_config.py"):
            self.assertTrue(os.path.isfile(os.path.join(root, module)), module)

    def test_it_does_not_depend_on_the_working_directory(self):
        # a notebook opened from inside a result folder must still write into the repository root
        expected = repository_root()
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            try:
                os.chdir(elsewhere)
                self.assertEqual(repository_root(), expected)
            finally:
                os.chdir(original)   # leave the folder before it is deleted (Windows cannot delete the cwd)


class FileHelperTests(unittest.TestCase):
    def test_ensure_directory_creates_nested_folders_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "a", "b", "c")
            ensure_directory(target)
            ensure_directory(target)
            self.assertTrue(os.path.isdir(target))

    def test_save_stats_csv_writes_a_header_and_one_row_per_record(self):
        stats = [{"gen": 0, "max": 0.5, "avg": 0.4}, {"gen": 1, "max": 0.6, "avg": 0.45}]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "seed_1", "convergence.csv")      # the folder is created
            save_stats_csv(stats, path)
            with open(path, newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([row["gen"] for row in rows], ["0", "1"])
        self.assertEqual(float(rows[1]["max"]), 0.6)

    def test_save_stats_csv_writes_nothing_for_no_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "empty.csv")
            save_stats_csv([], path)
            self.assertFalse(os.path.exists(path))


class ParetoSelectionTests(unittest.TestCase):
    def test_extremes(self):
        points = front((0.90, 0.60), (0.85, 0.80), (0.70, 0.95))
        self.assertEqual(best_auc_index(points), 0)
        self.assertEqual(best_sign_consistency_index(points), 2)

    def test_empty_fronts_are_refused(self):
        for function in (best_auc_index, best_sign_consistency_index, knee_point_index):
            with self.assertRaises(ValueError):
                function([])

    def test_single_point_front(self):
        self.assertEqual(knee_point_index(front((0.8, 0.9))), 0)

    def test_knee_is_the_point_farthest_from_the_line_between_the_extremes(self):
        # after min-max normalisation the extremes are (1, 0) and (0, 1); (0.9, 0.9) bulges out the most
        points = front((1.00, 0.50), (0.60, 1.00), (0.96, 0.95), (0.80, 0.75))
        self.assertEqual(knee_point_index(points), 2)

    def test_knee_when_both_extremes_are_the_same_point(self):
        points = front((0.9, 0.9), (0.8, 0.8), (0.7, 0.7))
        self.assertEqual(knee_point_index(points), 0)

    def test_knee_with_one_constant_objective(self):
        # all AUCs equal: the normalised AUC is 0 everywhere, so the best-AUC index is the first point and
        # the best-sign index another one; every point lies on the line between them (distance 0), and the
        # first of the tied points is returned
        points = front((0.8, 0.5), (0.8, 0.9), (0.8, 0.7))
        self.assertEqual(knee_point_index(points), 0)

    def test_select_pareto_individual_follows_the_rule(self):
        points = front((1.00, 0.50), (0.60, 1.00), (0.96, 0.95))
        self.assertIs(select_pareto_individual(points, use_knee_point=True), points[2])
        self.assertIs(select_pareto_individual(points, use_knee_point=False), points[1])


if __name__ == "__main__":
    unittest.main()
