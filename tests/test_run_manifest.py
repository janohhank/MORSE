"""Tests of run_manifest: the manifest a run writes when it starts (which data files, verified by exact
reproduction; fingerprints; settings), resuming with the same / other data, and rebuilding a finished run's
data from the manifest -- or from a notebook copy when there is none.

Run from the repository root:

    python -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy
import pandas
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import run_manifest  # noqa: E402
from run_manifest import (MANIFEST_FILE, ManifestMismatchError, build_run_manifest,  # noqa: E402
                          candidate_train_files, load_run_data, manifest_sha256, read_run_manifest,
                          write_run_manifest)
from test_robustness_utils import make_data  # noqa: E402
from training_config import TrainingConfig  # noqa: E402


class ManifestTestCase(unittest.TestCase):
    """Three CSV files like RadFusion's: train and validation are concatenated for training, and an `idx`
    column that is not an input is dropped by the column list."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = self.temporary.name
        X, y = make_data(300, seed=61)
        frame = X.assign(label=y, idx=numpy.arange(300))
        self.paths = {name: os.path.join(self.folder, f"{name}.csv") for name in ("train", "validation", "test")}
        frame.iloc[:180].to_csv(self.paths["train"], index=False)
        frame.iloc[180:230].to_csv(self.paths["validation"], index=False)
        frame.iloc[230:].to_csv(self.paths["test"], index=False)
        self.features = [c for c in X.columns if c != "med_b"]            # a column list, as the notebook's
        merged = pandas.concat([pandas.read_csv(self.paths["train"]), pandas.read_csv(self.paths["validation"])],
                               ignore_index=True)
        test = pandas.read_csv(self.paths["test"])
        self.X_train, self.y_train = merged[self.features], merged["label"].astype(int)
        self.X_test, self.y_test = test[self.features], test["label"]
        self.run = os.path.join(self.folder, "run")
        os.makedirs(self.run)

    def manifest(self, candidates=None, **overrides) -> dict:
        arguments = dict(
            train_file_candidates=candidates if candidates is not None else [
                [self.paths["test"]],                                       # a stale candidate: not the data
                [self.paths["train"], self.paths["validation"]], [self.paths["train"]]],
            test_file=self.paths["test"], target="label", X_train=self.X_train, y_train=self.y_train,
            X_test=self.X_test, y_test=self.y_test, use_roc_auc=True, use_knee_point=False,
            training_config=TrainingConfig(seed=0), cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            seeds=[42, 43])
        arguments.update(overrides)
        return build_run_manifest(**arguments)


class BuildTests(ManifestTestCase):
    def test_the_candidate_that_reproduces_the_data_is_recorded(self):
        manifest = self.manifest()
        data = manifest["data"]
        self.assertEqual([record["path"] for record in data["train_files"]],
                         [self.paths["train"], self.paths["validation"]])
        self.assertEqual(data["test_file"]["path"], self.paths["test"])
        self.assertTrue(all(len(record["sha256"]) == 64 for record in data["train_files"]))
        self.assertEqual((data["target"], data["features"]), ("label", self.features))
        self.assertEqual(data["arrays"]["X_train"]["shape"], [230, len(self.features)])
        self.assertEqual(manifest["settings"]["pareto_rule"], "max_s")
        self.assertEqual(manifest["settings"]["main_objective"], "ROC-AUC")
        self.assertEqual(manifest["settings"]["seeds"], [42, 43])
        self.assertEqual(manifest["settings"]["cv"]["random_state"], 42)
        self.assertEqual(set(manifest["settings"]["training_config"]), {"pop_size", "ngen", "cxpb", "mutpb"})
        json.dumps(manifest)                                               # JSON-ready

    def test_data_that_no_file_reproduces_are_recorded_without_files(self):
        changed = self.X_train.copy()
        changed.iloc[0, 0] += 1.0                                          # changed after reading
        manifest = self.manifest(X_train=changed, candidates=[[self.paths["train"]], []], test_file=None)
        self.assertIsNone(manifest["data"]["train_files"])
        self.assertIsNone(manifest["data"]["test_file"])
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            write_run_manifest(self.run, manifest)
        self.assertIn("could not be traced back to their CSV files", printed.getvalue())
        with self.assertRaisesRegex(FileNotFoundError, "--notebook"):
            load_run_data(self.run)                                        # nothing to rebuild the data from

    def test_candidate_file_lists_from_a_notebook_namespace(self):
        namespace = {"files": [self.paths["train"], self.paths["validation"]], "CSV_TRAIN_PATH": self.paths["train"],
                     "CSV_VALIDATION_PATH": self.paths["validation"]}
        self.assertEqual(candidate_train_files(namespace),
                         [[self.paths["train"], self.paths["validation"]], [self.paths["train"], self.paths["validation"]],
                          [self.paths["train"]]])
        self.assertEqual(candidate_train_files({"CSV_TRAIN_PATH": "a.csv", "files": "not a list"}), [["a.csv"]])
        self.assertEqual(candidate_train_files({}), [])

    def test_paths_relative_to_the_repository_root_are_found(self):
        with mock.patch.object(run_manifest, "repository_root", return_value=self.folder):
            self.assertEqual(run_manifest._resolve("train.csv"), os.path.abspath(self.paths["train"]))
            with self.assertRaisesRegex(FileNotFoundError, "also not relative to"):
                run_manifest._resolve("no/such/file.csv")


class WriteTests(ManifestTestCase):
    def test_write_read_and_resume(self):
        path = write_run_manifest(self.run, self.manifest(), log=lambda line: None)
        self.assertEqual(path, os.path.join(self.run, MANIFEST_FILE))
        first = read_run_manifest(self.run)
        self.assertEqual(len(manifest_sha256(self.run)), 64)
        notes = []
        write_run_manifest(self.run, self.manifest(use_knee_point=True, seeds=[42, 43, 44]), log=notes.append)
        second = read_run_manifest(self.run)
        self.assertTrue(any("settings differ" in note for note in notes))
        self.assertEqual(second["settings"]["pareto_rule"], "knee")
        self.assertEqual(second["created"], first["created"])
        self.assertIn("updated", second)

    def test_resuming_with_other_data_is_refused(self):
        write_run_manifest(self.run, self.manifest(), log=lambda line: None)
        other = self.X_test.copy()
        other.iloc[0, 0] += 1.0
        with self.assertRaises(ManifestMismatchError):
            write_run_manifest(self.run, self.manifest(X_test=other), log=lambda line: None)

    def test_no_manifest(self):
        self.assertIsNone(read_run_manifest(self.run))
        self.assertIsNone(manifest_sha256(self.run))


class LoadTests(ManifestTestCase):
    def setUp(self):
        super().setUp()
        write_run_manifest(self.run, self.manifest(), log=lambda line: None)

    def test_the_data_are_rebuilt_from_the_manifest(self):
        data = load_run_data(self.run)
        self.assertEqual(data["source"], "run manifest")
        pandas.testing.assert_frame_equal(data["X_train"], self.X_train)
        numpy.testing.assert_array_equal(data["y_train"].to_numpy(), self.y_train.to_numpy())
        pandas.testing.assert_frame_equal(data["X_test"], self.X_test)
        self.assertEqual((data["use_knee_point"], data["use_roc_auc"]), (False, True))

    def test_a_changed_file_that_still_gives_the_data_is_accepted(self):
        frame = pandas.read_csv(self.paths["test"]).assign(unused=1)
        frame.to_csv(self.paths["test"], index=False)
        printed = io.StringIO()
        with contextlib.redirect_stderr(printed):
            data = load_run_data(self.run)
        self.assertIn("changed since the run", printed.getvalue())
        pandas.testing.assert_frame_equal(data["X_test"], self.X_test)

    def test_changed_data_are_refused(self):
        frame = pandas.read_csv(self.paths["validation"])
        frame.loc[0, self.features[0]] += 1.0
        frame.to_csv(self.paths["validation"], index=False)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "no longer give the run's X_train"):
                load_run_data(self.run)

    def test_a_missing_file_is_reported(self):
        os.remove(self.paths["test"])
        with self.assertRaises(FileNotFoundError):
            load_run_data(self.run)

    def test_a_notebook_copy_takes_precedence(self):
        cells = [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": [
            f"CSV_TRAIN_PATH: str = {self.paths['train']!r}\n", "TARGET_COLUMN: str = 'label'\n",
            "USE_KNEE_POINT_SELECTION: bool = True\n", "df_train = pandas.read_csv(CSV_TRAIN_PATH)\n",
            f"df_test = pandas.read_csv({self.paths['test']!r})\n", "y_test = df_test[TARGET_COLUMN]\n",
            "X_test = df_test.drop(columns=[TARGET_COLUMN])\n"]}]
        notebook = os.path.join(self.folder, "copy.ipynb")
        with open(notebook, "w", encoding="utf-8") as handle:
            json.dump({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, handle)
        data = load_run_data(self.run, notebook)
        self.assertEqual((data["source"], len(data["X_train"]), data["use_knee_point"]), ("notebook copy", 180, True))


if __name__ == "__main__":
    unittest.main()
