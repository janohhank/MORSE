"""Checkpointing of the expensive training stage.

Training is by far the slowest part of the pipeline (tens of minutes for 20 seeds);
the evaluation after it takes minutes. A failure in the evaluation -- or a lost
kernel -- must therefore never force a retrain. This module writes the result of
every seed to disk the moment it is ready and lets a later run pick those results
up instead of training again.

Layout, under `<run directory>/checkpoints/training/`:

    fingerprint.json      what the checkpoints belong to (training configuration, CV
                          settings, feature names, hashes of the training data)
    seed_<N>/
        morse_front.csv   every Pareto individual: AUC, sign consistency, feature
                          count and the feature mask as a 0/1 string
        selections.json   the SO-GA best individual, the SFS mask, the all-features mask
        complete.json     written LAST -- a seed without it counts as not trained

Design rules
  * Plain CSV / JSON, never a pickle of DEAP objects: the files are readable, useful
    for the paper (fronts, masks) and independent of library versions. The DEAP
    individuals are rebuilt on load through `deap_types`.
  * Every file is written atomically (temporary file, flush, rename), and the marker
    is written after all the other files, so a crash can never leave a checkpoint
    that looks complete but is not.
  * Each seed is saved by the worker that trained it, at the moment it finishes.
  * A checkpoint folder carries a fingerprint. Resuming with different data or
    different settings raises `CheckpointMismatchError` instead of silently mixing
    results; a change of the algorithm source code only warns.
  * The fitted logistic-regression packages of the evaluation are deliberately NOT
    checkpointed: they are rebuilt from the saved masks in seconds, and they depend on
    evaluation settings (knee point vs best sign consistency) that a cache could get
    out of sync with.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import inspect
import io
import json
import os
import platform
import shutil
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from typing import Any, Callable, Iterable, Sequence

import numpy
from deap import creator
from joblib import Parallel, delayed

from deap_types import ensure_multi_objective_types, ensure_single_objective_types
from training_utils import ensure_directory

SCHEMA_VERSION: int = 1

FINGERPRINT_FILE: str = "fingerprint.json"
FRONT_FILE: str = "morse_front.csv"
SELECTIONS_FILE: str = "selections.json"
MARKER_FILE: str = "complete.json"
SEED_DIRECTORY_PREFIX: str = "seed_"


class CheckpointMismatchError(RuntimeError):
    """The checkpoints on disk were written for other data or other settings."""


class CheckpointWarning(UserWarning):
    """The checkpoints are usable but were written in a different environment."""


# ---------------------------------------------------------------------------
# Atomic file helpers
# ---------------------------------------------------------------------------

def atomic_write_text(path: str, text: str) -> None:
    """Write `text` to `path` so that neither a reader nor a crash ever sees a
    half-written file: the text goes to a temporary file next to the target, is
    flushed to disk and only then renamed over the target."""
    ensure_directory(os.path.dirname(os.path.abspath(path)))
    temporary_path: str = f"{path}.{os.getpid()}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def atomic_write_json(path: str, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Feature masks
# ---------------------------------------------------------------------------

def mask_to_string(mask: Sequence[int]) -> str:
    """A feature mask as a compact 0/1 string, e.g. [1, 0, 1] -> "101"."""
    bits: list[str] = []
    for bit in mask:
        if bit not in (0, 1):
            raise ValueError(f"a feature mask may only contain 0 and 1, found {bit!r}")
        bits.append("1" if bit else "0")
    return "".join(bits)


def string_to_mask(text: str, expected_length: int) -> list[int]:
    """Inverse of `mask_to_string`, checked against the number of features."""
    if len(text) != expected_length or set(text) - {"0", "1"}:
        raise ValueError(
            f"invalid feature mask: expected {expected_length} characters of 0/1, "
            f"got {len(text)} characters {text[:20]!r}...")
    return [int(character) for character in text]


# ---------------------------------------------------------------------------
# Fingerprint: what a set of checkpoints belongs to
# ---------------------------------------------------------------------------

def array_fingerprint(array: numpy.ndarray) -> dict[str, Any]:
    """Shape, dtype and SHA-256 of the raw bytes of an array."""
    contiguous: numpy.ndarray = numpy.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def source_fingerprint(objects: Iterable[Any]) -> dict[str, str]:
    """SHA-256 of the source file of every given class / function / module, keyed
    by file name (line endings are normalised, so the hash is the same on every
    operating system)."""
    hashes: dict[str, str] = {}
    for obj in objects:
        path: str | None = inspect.getsourcefile(obj)
        if path is None:
            continue
        with open(path, "rb") as handle:
            content: bytes = handle.read().replace(b"\r\n", b"\n")
        hashes[os.path.basename(path)] = hashlib.sha256(content).hexdigest()
    return hashes


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in ("numpy", "scikit-learn", "deap"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def build_training_fingerprint(
        config: Any,
        cv: Any,
        feature_names: Sequence[str],
        X_train: numpy.ndarray,
        y_train: numpy.ndarray,
        code_objects: Iterable[Any] = ()) -> dict[str, Any]:
    """Describe everything the per-seed results depend on.

    `settings` are the things that make old results unusable when they change:
    every field of the training configuration (except the per-seed `seed` and the
    output directory), the cross-validation splitter, the feature names and the
    training data itself. `environment` is informational (library versions and the
    hash of the algorithm source files): a difference there only produces a warning.

    config:       the `TrainingConfig` (any dataclass) used for the seeds.
    cv:           the shared cross-validation splitter (StratifiedKFold).
    X_train:      the array the trainers receive (after scaling).
    code_objects: classes / functions whose source files are hashed.
    """
    training_config: dict[str, Any] = {
        name: value for name, value in dataclasses.asdict(config).items()
        if name not in ("seed", "result_directory")}
    random_state: Any = cv.random_state
    return {
        "schema_version": SCHEMA_VERSION,
        "settings": {
            "training_config": training_config,
            "cv": {
                "n_splits": int(cv.get_n_splits()),
                "shuffle": bool(getattr(cv, "shuffle", False)),
                "random_state": random_state if isinstance(random_state, (int, type(None))) else repr(random_state),
            },
            "features": {
                "count": len(feature_names),
                "sha256": hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest(),
            },
            "data": {
                "X_train": array_fingerprint(X_train),
                "y_train": array_fingerprint(y_train),
            },
        },
        "environment": {
            "libraries": _library_versions(),
            "source_sha256": source_fingerprint(code_objects),
        },
    }


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    else:
        out[prefix] = value
    return out


def _short(value: Any) -> str:
    text: str = repr(value)
    return text if len(text) <= 26 else text[:14] + "..."


def _describe_difference(key: str, before: Any, after: Any) -> str:
    source_prefix: str = "environment.source_sha256."
    if key.startswith(source_prefix):
        name: str = key[len(source_prefix):]
        if before == "<absent>":
            return f"{name} is new since the checkpoints were written"
        if after == "<absent>":
            return f"{name} is no longer part of the code"
        return f"{name} changed since the checkpoints were written"
    return f"{key}: checkpoint has {_short(before)}, current run has {_short(after)}"


def compare_fingerprints(saved: dict[str, Any], current: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (problems, notes): `problems` are differences in the schema or the
    settings (the checkpoints must not be reused), `notes` are differences in the
    environment (library versions, algorithm source code)."""
    problems: list[str] = []
    notes: list[str] = []

    if saved.get("schema_version") != current["schema_version"]:
        problems.append(
            f"schema_version: checkpoint has {saved.get('schema_version')!r}, "
            f"this code writes {current['schema_version']!r}")

    for section, target in (("settings", problems), ("environment", notes)):
        saved_flat: dict[str, Any] = _flatten(section, saved.get(section, {}), {})
        current_flat: dict[str, Any] = _flatten(section, current[section], {})
        for key in sorted(set(saved_flat) | set(current_flat)):
            before: Any = saved_flat.get(key, "<absent>")
            after: Any = current_flat.get(key, "<absent>")
            if before != after:
                target.append(_describe_difference(key, before, after))

    return problems, notes


# ---------------------------------------------------------------------------
# Per-seed training results
# ---------------------------------------------------------------------------

@dataclass
class SeedTrainingResult:
    """Everything one seed's training produces, i.e. what the evaluation needs.

    pareto_front: MORSE's final Pareto front, a list of `creator.Individual`
                  (feature mask + fitness values (AUC, sign consistency)).
    single_best:  the SO-GA best `creator.IndividualSingle` (mask + AUC fitness).
    forward_mask: the forward-stepwise-selection mask.
    all_mask:     the all-features mask.
    seconds:      wall-clock training time of the seed (informational).
    """
    seed: int
    pareto_front: list
    single_best: Any
    forward_mask: list[int]
    all_mask: list[int]
    seconds: float = 0.0


class TrainingCheckpointStore:
    """A directory of per-seed training checkpoints (see the module docstring).

    Typical use, in the notebook:

        store = TrainingCheckpointStore(directory, fingerprint)
        store.prepare()                                  # create / verify the folder
        train_missing_seeds(store, seeds, train_one_seed)  # only the missing ones
        multi, single, forward, everything = store.load_all(seeds)
    """

    def __init__(self, directory: str, fingerprint: dict[str, Any]) -> None:
        self._directory: str = directory
        self._fingerprint: dict[str, Any] = fingerprint
        self._n_features: int = int(fingerprint["settings"]["features"]["count"])

    @property
    def directory(self) -> str:
        return self._directory

    def seed_directory(self, seed: int) -> str:
        return os.path.join(self._directory, f"{SEED_DIRECTORY_PREFIX}{seed}")

    # ---- folder / fingerprint ---------------------------------------------------------------
    def prepare(self, expect_existing: bool = False) -> None:
        """Create the folder and its fingerprint, or verify an existing folder.

        Raises `CheckpointMismatchError` if the folder was written for other data
        or settings, and warns (`CheckpointWarning`) about environment changes.
        With `expect_existing=True` -- the caller means to resume a run -- it
        raises `FileNotFoundError` (before creating anything) when the folder holds
        no completed seed, which is the case for every run made before checkpointing
        existed: better an error than a silent retrain of hours.
        """
        if expect_existing and not self.completed_seeds():
            raise FileNotFoundError(
                f"no completed seed checkpoint found in {self._directory}. A run made before "
                f"checkpointing existed cannot be resumed -- start a fresh run instead.")

        ensure_directory(self._directory)
        fingerprint_path: str = os.path.join(self._directory, FINGERPRINT_FILE)

        if os.path.exists(fingerprint_path):
            problems, notes = compare_fingerprints(read_json(fingerprint_path), self._fingerprint)
            if problems:
                raise CheckpointMismatchError(
                    f"the checkpoints in {self._directory} were written for different data or "
                    f"settings and must not be reused:\n  - " + "\n  - ".join(problems) +
                    "\nResume only with the same data and settings, or start a fresh run.")
            for note in notes:
                warnings.warn(f"checkpoints in {self._directory}: {note}", CheckpointWarning, stacklevel=2)
        else:
            if self._seed_directories():
                raise CheckpointMismatchError(
                    f"{self._directory} contains seed checkpoints but no {FINGERPRINT_FILE}, so "
                    f"it cannot be verified that they belong to this data and configuration.")
            atomic_write_json(fingerprint_path, self._fingerprint)

    # ---- which seeds are done ---------------------------------------------------------------
    def _seed_directories(self) -> list[int]:
        if not os.path.isdir(self._directory):
            return []
        seeds: list[int] = []
        for name in os.listdir(self._directory):
            suffix: str = name[len(SEED_DIRECTORY_PREFIX):]
            if name.startswith(SEED_DIRECTORY_PREFIX) and suffix.isdigit() \
                    and os.path.isdir(os.path.join(self._directory, name)):
                seeds.append(int(suffix))
        return sorted(seeds)

    def is_complete(self, seed: int) -> bool:
        return os.path.isfile(os.path.join(self.seed_directory(seed), MARKER_FILE))

    def completed_seeds(self) -> list[int]:
        return [seed for seed in self._seed_directories() if self.is_complete(seed)]

    def missing_seeds(self, seeds: Iterable[int]) -> list[int]:
        return [seed for seed in seeds if not self.is_complete(seed)]

    # ---- save -------------------------------------------------------------------------------
    def save_seed(self, result: SeedTrainingResult) -> None:
        """Write one seed's results. Replaces any earlier (possibly partial)
        checkpoint of the same seed; the completion marker is written last."""
        if not result.pareto_front:
            raise ValueError(f"seed {result.seed}: the Pareto front is empty")

        rows: list[tuple[float, float, int, str]] = []
        for individual in result.pareto_front:
            values: tuple = tuple(individual.fitness.values)
            if len(values) != 2:
                raise ValueError(f"seed {result.seed}: a Pareto individual has no valid fitness")
            mask_text: str = mask_to_string(individual)
            string_to_mask(mask_text, self._n_features)  # length check
            rows.append((float(values[0]), float(values[1]), mask_text.count("1"), mask_text))

        single_mask: str = mask_to_string(result.single_best)
        forward_mask: str = mask_to_string(result.forward_mask)
        all_mask: str = mask_to_string(result.all_mask)
        for text in (single_mask, forward_mask, all_mask):
            string_to_mask(text, self._n_features)

        seed_directory: str = self.seed_directory(result.seed)
        if os.path.isdir(seed_directory):
            shutil.rmtree(seed_directory)
        ensure_directory(seed_directory)

        buffer: io.StringIO = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["auc", "sign_consistency", "n_features", "mask"])
        writer.writerows(rows)
        atomic_write_text(os.path.join(seed_directory, FRONT_FILE), buffer.getvalue())

        atomic_write_json(os.path.join(seed_directory, SELECTIONS_FILE), {
            "soga": {"mask": single_mask, "fitness": float(result.single_best.fitness.values[0])},
            "sfs": {"mask": forward_mask},
            "all_features": {"mask": all_mask},
        })

        atomic_write_json(os.path.join(seed_directory, MARKER_FILE), {
            "schema_version": SCHEMA_VERSION,
            "seed": int(result.seed),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "training_seconds": round(float(result.seconds), 1),
            "front_size": len(rows),
            "n_features": {
                "soga": single_mask.count("1"),
                "sfs": forward_mask.count("1"),
                "all_features": all_mask.count("1"),
            },
        })

    # ---- load -------------------------------------------------------------------------------
    def load_seed(self, seed: int) -> SeedTrainingResult:
        """Rebuild one seed's results, as the very same DEAP types the trainers return."""
        if not self.is_complete(seed):
            raise FileNotFoundError(f"seed {seed} has no complete checkpoint in {self._directory}")

        ensure_multi_objective_types()
        ensure_single_objective_types()
        seed_directory: str = self.seed_directory(seed)
        marker: dict[str, Any] = read_json(os.path.join(seed_directory, MARKER_FILE))

        pareto_front: list = []
        with open(os.path.join(seed_directory, FRONT_FILE), "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                individual = creator.Individual(string_to_mask(row["mask"], self._n_features))
                individual.fitness.values = (float(row["auc"]), float(row["sign_consistency"]))
                pareto_front.append(individual)
        if len(pareto_front) != marker["front_size"]:
            raise ValueError(
                f"seed {seed}: {FRONT_FILE} has {len(pareto_front)} rows but the marker says {marker['front_size']}")

        selections: dict[str, Any] = read_json(os.path.join(seed_directory, SELECTIONS_FILE))
        single_best = creator.IndividualSingle(string_to_mask(selections["soga"]["mask"], self._n_features))
        single_best.fitness.values = (float(selections["soga"]["fitness"]),)

        return SeedTrainingResult(
            seed=int(seed),
            pareto_front=pareto_front,
            single_best=single_best,
            forward_mask=string_to_mask(selections["sfs"]["mask"], self._n_features),
            all_mask=string_to_mask(selections["all_features"]["mask"], self._n_features),
            seconds=float(marker["training_seconds"]),
        )

    def load_all(self, seeds: Sequence[int]) -> tuple[dict[int, list], dict[int, Any], dict[int, list[int]], dict[int, list[int]]]:
        """Load every seed into the four dictionaries the evaluation uses:
        (MORSE fronts, SO-GA best individuals, SFS masks, all-features masks)."""
        missing: list[int] = self.missing_seeds(seeds)
        if missing:
            raise FileNotFoundError(f"no complete checkpoint for seeds {missing} in {self._directory}")

        multi: dict[int, list] = {}
        single: dict[int, Any] = {}
        forward: dict[int, list[int]] = {}
        everything: dict[int, list[int]] = {}
        for seed in seeds:
            result: SeedTrainingResult = self.load_seed(seed)
            multi[seed] = result.pareto_front
            single[seed] = result.single_best
            forward[seed] = result.forward_mask
            everything[seed] = result.all_mask
        return multi, single, forward, everything


def import_results(
        store: TrainingCheckpointStore,
        seeds: Sequence[int],
        multi: dict[int, list],
        single: dict[int, Any],
        forward: dict[int, list[int]],
        everything: dict[int, list[int]]) -> None:
    """Write results that already exist in memory to `store` -- for example the
    `training_results_*` dictionaries of a run that started before checkpointing
    existed and is still alive in the kernel -- so that they survive a restart."""
    for seed in seeds:
        store.save_seed(SeedTrainingResult(
            seed=seed, pareto_front=multi[seed], single_best=single[seed],
            forward_mask=forward[seed], all_mask=everything[seed]))


# ---------------------------------------------------------------------------
# Orchestration: train what is missing, save each seed as it finishes
# ---------------------------------------------------------------------------

def _print_flushed(message: str) -> None:
    print(message, flush=True)


def train_missing_seeds(
        store: TrainingCheckpointStore,
        seeds: Sequence[int],
        train_one_seed: Callable[[int], tuple],
        n_jobs: int = -1,
        log: Callable[[str], None] = _print_flushed) -> list[int]:
    """Train every seed that has no complete checkpoint yet and write each result
    to `store` the moment it is ready (by the worker that trained it), so nothing
    is lost if a later seed -- or the rest of the pipeline -- fails.

    train_one_seed: seed -> (seed, pareto_front, single_best, forward_mask, all_mask),
                    the per-seed training function of the notebook.
    n_jobs:         1 = sequential in this process, otherwise a joblib/loky pool
                    (-1 = all cores), one seed per worker.

    Returns the seeds that were trained in this call.
    """
    missing: list[int] = store.missing_seeds(seeds)
    log(f"{len(seeds) - len(missing)} of {len(seeds)} seeds already have a checkpoint in "
        f"{store.directory}; training {len(missing)}: {missing}")
    if not missing:
        return []

    def train_and_save(seed: int) -> int:
        started: float = time.time()
        finished_seed, pareto_front, single_best, forward_mask, all_mask = train_one_seed(seed)
        if finished_seed != seed:
            raise RuntimeError(f"train_one_seed({seed}) returned the results of seed {finished_seed}")
        store.save_seed(SeedTrainingResult(
            seed=seed, pareto_front=pareto_front, single_best=single_best,
            forward_mask=forward_mask, all_mask=all_mask, seconds=time.time() - started))
        return seed

    start: float = time.time()

    def report(done: int) -> None:
        elapsed: float = time.time() - start
        eta: float = elapsed / done * (len(missing) - done)
        log(f"Progress: {done}/{len(missing)} seeds done and saved | elapsed {elapsed / 60:6.1f} min | "
            f"avg {elapsed / done / 60:5.1f} min/seed | ETA ~{eta / 60:5.1f} min")

    if n_jobs == 1:
        for done, seed in enumerate(missing, 1):
            train_and_save(seed)
            report(done)
    else:
        log(f"Training {len(missing)} seeds in parallel (n_jobs={n_jobs}, backend=loky)...")
        parallel = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered")
        for done, _ in enumerate(parallel(delayed(train_and_save)(seed) for seed in missing), 1):
            report(done)

    return missing
