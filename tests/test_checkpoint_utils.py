"""Tests of checkpoint_utils and deap_types.

Run from the repository root:

    python -m unittest discover -s tests -v
"""
import inspect
import json
import os
import sys
import tempfile
import unittest
import warnings

import numpy
from deap import creator
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import checkpoint_utils as cu  # noqa: E402
from deap_types import ensure_multi_objective_types, ensure_single_objective_types  # noqa: E402
from multi_objective_training import MultiObjectiveTraining  # noqa: E402
from single_objective_training import SingleObjectiveTraining  # noqa: E402
from training_config import TrainingConfig  # noqa: E402
from training_utils import select_pareto_individual  # noqa: E402

N_FEATURES: int = 12
FEATURE_NAMES: list[str] = [f"feature_{i}" for i in range(N_FEATURES)]


def make_fingerprint(ngen: int = 10, x_offset: float = 0.0, feature_names: list[str] = FEATURE_NAMES,
                     code_objects=()) -> dict:
    rng = numpy.random.RandomState(0)
    X = rng.randn(40, N_FEATURES) + x_offset
    y = (rng.rand(40) > 0.5).astype(float)
    return cu.build_training_fingerprint(
        config=TrainingConfig(seed=0, ngen=ngen, use_roc_auc=False),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        feature_names=feature_names, X_train=X, y_train=y, code_objects=code_objects)


def make_seed_training(seed: int) -> tuple:
    """What the notebook's `_train_one_seed` returns, with fully valid DEAP individuals."""
    ensure_multi_objective_types()
    ensure_single_objective_types()
    front = []
    for k, (auc, sign) in enumerate([(0.90 - 0.02 * seed, 0.60), (0.85, 0.75), (0.70, 0.95), (0.6, 1.0)]):
        individual = creator.Individual([1] * (k + 1) + [0] * (N_FEATURES - k - 1))
        individual.fitness.values = (auc, sign)
        front.append(individual)
    single = creator.IndividualSingle([1, 0] * (N_FEATURES // 2))
    single.fitness.values = (0.9123456789012345,)
    forward = [1 if i % 3 == 0 else 0 for i in range(N_FEATURES)]
    everything = [1] * N_FEATURES
    return seed, front, single, forward, everything


def result_from_tuple(values: tuple, seconds: float = 1.5) -> cu.SeedTrainingResult:
    seed, front, single, forward, everything = values
    return cu.SeedTrainingResult(seed, front, single, forward, everything, seconds)


def make_seed_only(seed: int) -> tuple:
    """What the notebook's `_train_one_seed` returns now: only the parts that depend on the seed."""
    seed, front, single, _, _ = make_seed_training(seed)
    return seed, front, single


def make_baselines(seconds: float = 0.5) -> cu.SharedBaselines:
    """The seed-independent baselines, equal to the masks `make_seed_training` produces."""
    _, _, _, forward, everything = make_seed_training(0)
    return cu.SharedBaselines(forward_mask=forward, all_mask=everything, seconds=seconds)


class MaskTests(unittest.TestCase):
    def test_round_trip(self):
        mask = [1, 0, 0, 1, 1, 0]
        self.assertEqual(cu.string_to_mask(cu.mask_to_string(mask), 6), mask)

    def test_rejects_invalid_masks(self):
        with self.assertRaises(ValueError):
            cu.mask_to_string([1, 2, 0])
        with self.assertRaises(ValueError):
            cu.string_to_mask("1012", 4)
        with self.assertRaises(ValueError):
            cu.string_to_mask("101", 4)


class FingerprintTests(unittest.TestCase):
    def test_identical_fingerprints_do_not_differ(self):
        problems, notes = cu.compare_fingerprints(make_fingerprint(), make_fingerprint())
        self.assertEqual((problems, notes), ([], []))

    def test_settings_changes_are_problems(self):
        base = make_fingerprint()
        problems, _ = cu.compare_fingerprints(base, make_fingerprint(ngen=99))
        self.assertTrue(any("training_config.ngen" in p for p in problems), problems)
        problems, _ = cu.compare_fingerprints(base, make_fingerprint(x_offset=1e-9))
        self.assertTrue(any("data.X_train.sha256" in p for p in problems), problems)
        problems, _ = cu.compare_fingerprints(base, make_fingerprint(feature_names=FEATURE_NAMES[::-1]))
        self.assertTrue(any("features.sha256" in p for p in problems), problems)

    def test_environment_changes_are_only_notes(self):
        base = make_fingerprint(code_objects=(cu.TrainingCheckpointStore,))
        changed = json.loads(json.dumps(base))
        changed["environment"]["libraries"]["numpy"] = "0.0.1"
        changed["environment"]["source_sha256"]["checkpoint_utils.py"] = "0" * 64
        problems, notes = cu.compare_fingerprints(base, changed)
        self.assertEqual(problems, [])
        self.assertTrue(any("numpy" in n for n in notes), notes)
        self.assertTrue(any("checkpoint_utils.py changed" in n for n in notes), notes)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = os.path.join(self._temporary.name, "checkpoints", "training")
        self.store = cu.TrainingCheckpointStore(self.directory, make_fingerprint())
        self.store.prepare()

    def tearDown(self):
        self._temporary.cleanup()

    def test_round_trip_is_exact_and_types_match_the_trainers(self):
        original = make_seed_training(7)
        self.store.save_seed(result_from_tuple(original, seconds=12.34))
        self.assertTrue(self.store.is_complete(7))

        loaded = self.store.load_seed(7)
        _, front, single, forward, everything = original
        self.assertEqual(loaded.seed, 7)
        self.assertEqual(len(loaded.pareto_front), len(front))
        for a, b in zip(front, loaded.pareto_front):
            self.assertEqual(list(a), list(b))
            self.assertEqual(a.fitness.values, b.fitness.values)          # bit-exact floats
            self.assertTrue(b.fitness.valid)
            self.assertIsInstance(b, creator.Individual)
        self.assertIsInstance(loaded.single_best, creator.IndividualSingle)
        self.assertEqual(list(loaded.single_best), list(single))
        self.assertEqual(loaded.single_best.fitness.values, single.fitness.values)
        self.assertEqual(loaded.forward_mask, forward)
        self.assertEqual(loaded.all_mask, everything)
        self.assertAlmostEqual(loaded.seconds, 12.3)

        # the downstream selection picks the same individual from the restored front
        for knee in (True, False):
            self.assertEqual(list(select_pareto_individual(front, use_knee_point=knee)),
                             list(select_pareto_individual(loaded.pareto_front, use_knee_point=knee)))

    def test_files_are_plain_readable_text(self):
        self.store.save_seed(result_from_tuple(make_seed_training(3)))
        seed_directory = self.store.seed_directory(3)
        self.assertEqual(sorted(os.listdir(seed_directory)),
                         sorted([cu.FRONT_FILE, cu.SELECTIONS_FILE, cu.MARKER_FILE]))
        with open(os.path.join(seed_directory, cu.FRONT_FILE), encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        self.assertEqual(lines[0], "auc,sign_consistency,n_features,mask")
        self.assertEqual(lines[1].split(",")[3], "1" + "0" * (N_FEATURES - 1))
        self.assertEqual([f for f in os.listdir(self.directory) if f.endswith(".tmp")], [])

    def test_a_seed_without_marker_is_not_complete_and_is_replaced_on_save(self):
        seed_directory = self.store.seed_directory(5)
        os.makedirs(seed_directory)
        with open(os.path.join(seed_directory, cu.FRONT_FILE), "w") as handle:
            handle.write("half written")
        self.assertFalse(self.store.is_complete(5))
        self.assertEqual(self.store.missing_seeds([5, 6]), [5, 6])
        with self.assertRaises(FileNotFoundError):
            self.store.load_seed(5)

        self.store.save_seed(result_from_tuple(make_seed_training(5)))
        self.assertTrue(self.store.is_complete(5))
        self.assertEqual(len(self.store.load_seed(5).pareto_front), 4)

    def test_save_rejects_wrong_length_masks_and_empty_fronts(self):
        seed, front, single, forward, everything = make_seed_training(1)
        with self.assertRaises(ValueError):
            self.store.save_seed(cu.SeedTrainingResult(seed, [], single, forward, everything))
        with self.assertRaises(ValueError):
            self.store.save_seed(cu.SeedTrainingResult(seed, front, single, forward[:-1], everything))
        self.assertFalse(self.store.is_complete(1))

    def test_load_all_shapes_and_missing_seeds(self):
        for seed in (1, 2):
            self.store.save_seed(result_from_tuple(make_seed_training(seed)))
        multi, single, forward, everything = self.store.load_all([1, 2])
        self.assertEqual(sorted(multi), [1, 2])
        self.assertEqual(sorted(single), [1, 2])
        self.assertEqual(sorted(forward), [1, 2])
        self.assertEqual(sorted(everything), [1, 2])
        with self.assertRaisesRegex(FileNotFoundError, r"\[3\]"):
            self.store.load_all([1, 3])

    def test_shared_baselines_round_trip_and_validation(self):
        self.assertIsNone(self.store.load_shared_baselines())
        self.store.save_shared_baselines(make_baselines(seconds=12.34))
        loaded = self.store.load_shared_baselines()
        self.assertEqual(loaded.forward_mask, make_baselines().forward_mask)
        self.assertEqual(loaded.all_mask, make_baselines().all_mask)
        self.assertAlmostEqual(loaded.seconds, 12.3)
        with self.assertRaises(ValueError):
            self.store.save_shared_baselines(cu.SharedBaselines(forward_mask=[1, 0], all_mask=[1] * N_FEATURES))

    def test_import_results_from_memory(self):
        seeds = [4, 5]
        run = {seed: make_seed_training(seed) for seed in seeds}
        cu.import_results(self.store, seeds,
                          multi={s: run[s][1] for s in seeds}, single={s: run[s][2] for s in seeds},
                          forward={s: run[s][3] for s in seeds}, everything={s: run[s][4] for s in seeds})
        self.assertEqual(self.store.completed_seeds(), seeds)
        multi, single, forward, everything = self.store.load_all(seeds)
        self.assertEqual([list(i) for i in multi[5]], [list(i) for i in run[5][1]])
        self.assertEqual(forward[4], run[4][3])
        # the identical SFS / all-features masks of all seeds are also saved as the shared baselines
        shared = self.store.load_shared_baselines()
        self.assertEqual(shared.forward_mask, run[4][3])
        self.assertEqual(shared.all_mask, run[4][4])

    def test_import_results_warns_and_saves_no_shared_baselines_when_the_seeds_disagree(self):
        seeds = [4, 5]
        run = {seed: make_seed_training(seed) for seed in seeds}
        other_forward = list(run[5][3])
        other_forward[0] = 1 - other_forward[0]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cu.import_results(self.store, seeds,
                              multi={s: run[s][1] for s in seeds}, single={s: run[s][2] for s in seeds},
                              forward={4: run[4][3], 5: other_forward}, everything={s: run[s][4] for s in seeds})
        self.assertTrue(any(issubclass(w.category, cu.CheckpointWarning) for w in caught))
        self.assertEqual(self.store.completed_seeds(), seeds)
        self.assertIsNone(self.store.load_shared_baselines())

    def test_prepare_verifies_the_fingerprint(self):
        # the same fingerprint is fine, also from a second store object (a "resumed" run)
        cu.TrainingCheckpointStore(self.directory, make_fingerprint()).prepare()
        # other settings are refused, with a readable reason
        with self.assertRaisesRegex(cu.CheckpointMismatchError, "training_config.ngen"):
            cu.TrainingCheckpointStore(self.directory, make_fingerprint(ngen=11)).prepare()
        # a warning, not an error, for an environment change
        changed = make_fingerprint()
        changed["environment"]["libraries"]["numpy"] = "0.0.1"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cu.TrainingCheckpointStore(self.directory, changed).prepare()
        self.assertTrue(any(issubclass(w.category, cu.CheckpointWarning) for w in caught))

    def test_seed_folders_without_a_fingerprint_are_refused(self):
        other = os.path.join(self._temporary.name, "other")
        os.makedirs(os.path.join(other, "seed_1"))
        with self.assertRaises(cu.CheckpointMismatchError):
            cu.TrainingCheckpointStore(other, make_fingerprint()).prepare()

    def test_expect_existing_refuses_an_empty_folder(self):
        with self.assertRaises(FileNotFoundError):
            self.store.prepare(expect_existing=True)
        self.store.save_seed(result_from_tuple(make_seed_training(1)))
        self.store.prepare(expect_existing=True)  # now there is something to resume

    def test_expect_existing_does_not_create_anything(self):
        missing = os.path.join(self._temporary.name, "never_trained", "checkpoints", "training")
        with self.assertRaises(FileNotFoundError):
            cu.TrainingCheckpointStore(missing, make_fingerprint()).prepare(expect_existing=True)
        self.assertFalse(os.path.exists(os.path.join(self._temporary.name, "never_trained")))


class TrainMissingSeedsTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.store = cu.TrainingCheckpointStore(os.path.join(self._temporary.name, "training"), make_fingerprint())
        self.store.prepare()
        self.messages: list[str] = []
        self.seed_calls: list[int] = []
        self.baseline_calls: list[int] = []

    def tearDown(self):
        self._temporary.cleanup()

    def train_one_seed(self, seed):
        self.seed_calls.append(seed)
        return make_seed_only(seed)

    def compute_baselines(self):
        self.baseline_calls.append(1)
        return make_baselines()

    def run_training(self, seeds, n_jobs=1, train_one_seed=None, compute_baselines=None):
        return cu.train_missing_seeds(
            self.store, seeds, train_one_seed or self.train_one_seed, compute_baselines or self.compute_baselines,
            n_jobs=n_jobs, log=self.messages.append)

    def test_sequential_run_then_resume_trains_only_what_is_missing(self):
        self.assertEqual(self.run_training([10, 11, 12]), [10, 11, 12])
        self.assertEqual(self.seed_calls, [10, 11, 12])
        self.assertEqual(self.store.completed_seeds(), [10, 11, 12])

        # a second call finds everything and trains nothing
        self.assertEqual(self.run_training([10, 11, 12]), [])
        self.assertEqual(self.seed_calls, [10, 11, 12])

        # a crashed seed (marker missing) is retrained, an added seed is trained, the rest is kept
        os.remove(os.path.join(self.store.seed_directory(11), cu.MARKER_FILE))
        self.assertEqual(self.run_training([10, 11, 12, 13]), [11, 13])
        self.assertEqual(self.seed_calls, [10, 11, 12, 11, 13])
        self.assertTrue(any("already have a checkpoint" in m for m in self.messages))

    def test_the_seed_independent_baselines_are_computed_once_and_copied_into_every_seed(self):
        self.run_training([10, 11, 12])
        self.assertEqual(len(self.baseline_calls), 1)

        expected = make_baselines()
        multi, single, forward, everything = self.store.load_all([10, 11, 12])
        for seed in (10, 11, 12):
            self.assertEqual(forward[seed], expected.forward_mask)
            self.assertEqual(everything[seed], expected.all_mask)
        self.assertEqual(self.store.load_shared_baselines().forward_mask, expected.forward_mask)
        self.assertTrue(os.path.isfile(os.path.join(self.store.directory, cu.SHARED_BASELINES_FILE)))

        # ... and they are reused, not recomputed, when more seeds are trained later (even by another store object)
        resumed = cu.TrainingCheckpointStore(self.store.directory, make_fingerprint())
        resumed.prepare()

        def must_not_run():
            raise AssertionError("the baselines must be loaded from the store, not computed again")

        cu.train_missing_seeds(resumed, [10, 11, 12, 13], self.train_one_seed, must_not_run, n_jobs=1,
                               log=self.messages.append)
        self.assertEqual(resumed.load_all([13])[2][13], expected.forward_mask)
        self.assertTrue(any("Reusing the seed-independent baselines" in m for m in self.messages))

    def test_the_baselines_are_not_computed_when_there_is_nothing_to_train(self):
        for seed in (1, 2):
            self.store.save_seed(result_from_tuple(make_seed_training(seed)))
        self.assertEqual(self.run_training([1, 2]), [])
        self.assertEqual(self.baseline_calls, [])
        self.assertIsNone(self.store.load_shared_baselines())

    def test_seeds_finished_before_a_failure_stay_saved(self):
        def train_one_seed(seed):
            if seed == 22:
                raise RuntimeError("boom")
            return make_seed_only(seed)

        with self.assertRaisesRegex(RuntimeError, "boom"):
            self.run_training([21, 22, 23], train_one_seed=train_one_seed)
        self.assertEqual(self.store.completed_seeds(), [21])
        self.assertEqual(self.store.missing_seeds([21, 22, 23]), [22, 23])
        self.assertIsNotNone(self.store.load_shared_baselines())  # computed before the first seed

    def test_a_result_of_the_wrong_seed_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "results of seed"):
            self.run_training([30], train_one_seed=lambda seed: make_seed_only(seed + 1))

    def test_parallel_workers_save_their_own_seeds_with_the_shared_baselines(self):
        def train_one_seed(seed):
            return make_seed_only(seed)

        trained = self.run_training([40, 41], n_jobs=2, train_one_seed=train_one_seed)
        self.assertEqual(sorted(trained), [40, 41])
        self.assertEqual(len(self.baseline_calls), 1)  # in the parent process, once
        multi, single, forward, everything = self.store.load_all([40, 41])
        self.assertEqual(sorted(multi), [40, 41])
        self.assertEqual(list(single[41]), [1, 0] * (N_FEATURES // 2))
        self.assertEqual(forward[40], make_baselines().forward_mask)
        self.assertEqual(forward[41], make_baselines().forward_mask)
        self.assertEqual(everything[41], [1] * N_FEATURES)


class RequireFixedCvTests(unittest.TestCase):
    def test_fixed_splitters_are_accepted(self):
        cu.require_fixed_cv(StratifiedKFold(n_splits=3, shuffle=True, random_state=42))
        cu.require_fixed_cv(StratifiedKFold(n_splits=3, shuffle=True, random_state=numpy.int64(7)))
        cu.require_fixed_cv(StratifiedKFold(n_splits=3))                  # no shuffling: always the same folds

    def test_splitters_that_depend_on_the_global_random_state_are_refused(self):
        for random_state in (None, numpy.random.RandomState(1)):
            with self.assertRaisesRegex(ValueError, "cannot be computed once"):
                cu.require_fixed_cv(StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state))


class AtomicWriteTests(unittest.TestCase):
    def test_replaces_the_target_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "nested", "file.json")
            cu.atomic_write_json(path, {"a": 1})
            cu.atomic_write_json(path, {"a": 2})
            self.assertEqual(cu.read_json(path), {"a": 2})
            self.assertEqual(os.listdir(os.path.dirname(path)), ["file.json"])


class SharedTypesTests(unittest.TestCase):
    def test_weights(self):
        ensure_multi_objective_types()
        ensure_single_objective_types()
        self.assertEqual(creator.FitnessMulti.weights, (1.0, 1.0))
        self.assertEqual(creator.FitnessSingle.weights, (1.0,))

    def test_the_trainers_use_the_shared_definitions(self):
        self.assertIn("ensure_multi_objective_types()", inspect.getsource(MultiObjectiveTraining.run))
        self.assertIn("ensure_single_objective_types()", inspect.getsource(SingleObjectiveTraining.run))


if __name__ == "__main__":
    unittest.main()
