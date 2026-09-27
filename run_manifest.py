"""The run manifest: what a run was trained on, written automatically when the run starts, so that the
evaluation can be repeated later on the run folder alone -- without an archived copy of the notebook and
without retraining.

run_manifest.json (in the run folder)
  data      the target column and the input names in their order; the CSV files the training and the
            test data were read from (path as given, absolute path, SHA-256, rows) -- found by trying the
            notebook's candidate file lists and keeping the one that reproduces the data exactly -- and
            fingerprints of the training and test arrays (checkpoint_utils.array_fingerprint);
  settings  the main objective (ROC-AUC / PR-AUC), MORSE's Pareto rule (knee / max_s), the GA settings,
            the cross-validation, the seeds;
  plus the creation time, the git commit and the library versions.

A run that is resumed (RESUME_FROM) must have been made from the same data: a different data block raises
ManifestMismatchError. The settings block is updated (with a note) when they changed.

`load_run_data` rebuilds a finished run's data -- from the manifest (the files are read again and must
reproduce the recorded arrays), or, for runs made before the manifest existed, from the data cells of the
notebook copy archived in the run folder (`load_archived_run`) -- and `verify_training_data` checks the
training data against the run's checkpoint fingerprint.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Callable, Sequence

import numpy
import pandas
from sklearn.preprocessing import StandardScaler

from checkpoint_utils import FINGERPRINT_FILE, _library_versions, array_fingerprint, atomic_write_json, read_json
from training_utils import repository_root

MANIFEST_FILE: str = "run_manifest.json"
MANIFEST_SCHEMA_VERSION: int = 1


class ManifestMismatchError(ValueError):
    """The run folder's manifest records other data than the notebook is running on."""


# ---------------------------------------------------------------------------
# Writing the manifest
# ---------------------------------------------------------------------------

def _file_record(path: str) -> dict[str, Any]:
    absolute: str = _resolve(path)
    with open(absolute, "rb") as handle:
        digest: str = hashlib.sha256(handle.read()).hexdigest()
    return {"path": path, "absolute": absolute, "sha256": digest}


def _resolve(path: str) -> str:
    """An existing file: the path itself, or relative to the repository root (the notebook's paths are)."""
    if os.path.isfile(path):
        return os.path.abspath(path)
    candidate: str = os.path.join(repository_root(), path)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)
    raise FileNotFoundError(f"{path} does not exist (also not relative to {repository_root()})")


def _arrays(X: pandas.DataFrame, y: Sequence[float]) -> tuple[numpy.ndarray, numpy.ndarray]:
    return (numpy.ascontiguousarray(X.to_numpy(dtype=numpy.float64)),
            numpy.ascontiguousarray(numpy.asarray(y, dtype=numpy.float64)))


def _read(files: Sequence[str], target: str, features: Sequence[str]) -> tuple[pandas.DataFrame, pandas.Series]:
    """The inputs (in `features` order) and the target of one or several CSV files, concatenated in order."""
    frame: pandas.DataFrame = pandas.concat([pandas.read_csv(_resolve(path)) for path in files], ignore_index=True)
    return frame[list(features)], frame[target]


def _reproduces(files: Sequence[str], target: str, X: pandas.DataFrame, y: Sequence[float]) -> bool:
    try:
        X_read, y_read = _read(files, target, list(X.columns))
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return False
    X_expected, y_expected = _arrays(X, y)
    X_found, y_found = _arrays(X_read, y_read)
    return (X_found.shape == X_expected.shape and y_found.shape == y_expected.shape
            and numpy.array_equal(X_found, X_expected, equal_nan=True) and numpy.array_equal(y_found, y_expected))


def candidate_train_files(namespace: dict[str, Any]) -> list[list[str]]:
    """The file lists the notebook's training data may have been read from, in the order to try them: a
    `files` list (train + validation files concatenated), CSV_TRAIN_PATH with CSV_VALIDATION_PATH, and
    CSV_TRAIN_PATH alone. Stale names left in a kernel are harmless: a candidate is only used if it
    reproduces the training data exactly."""
    candidates: list[list[str]] = []
    files: Any = namespace.get("files")
    if isinstance(files, (list, tuple)) and files and all(isinstance(f, str) for f in files):
        candidates.append(list(files))
    train: Any = namespace.get("CSV_TRAIN_PATH")
    validation: Any = namespace.get("CSV_VALIDATION_PATH")
    if isinstance(train, str):
        if isinstance(validation, str):
            candidates.append([train, validation])
        candidates.append([train])
    return candidates


def build_run_manifest(train_file_candidates: Sequence[Sequence[str]], test_file: str | None, target: str,
                       X_train: pandas.DataFrame, y_train: Sequence[float],
                       X_test: pandas.DataFrame, y_test: Sequence[float],
                       use_roc_auc: bool, use_knee_point: bool, training_config: Any = None, cv: Any = None,
                       seeds: Sequence[int] = ()) -> dict[str, Any]:
    """The manifest of a run (see the module docstring). The first candidate file list that reproduces
    X_train / y_train exactly is recorded; if none does (or the test file does not reproduce X_test /
    y_test), the files are recorded as None and only a notebook copy can rebuild the run's data later."""
    features: list[str] = list(X_train.columns)
    train_files: list[dict[str, Any]] | None = None
    for candidate in train_file_candidates:
        if candidate and _reproduces(candidate, target, X_train, y_train):
            train_files = [_file_record(path) for path in candidate]
            break
    test_record: dict[str, Any] | None = None
    if test_file and _reproduces([test_file], target, X_test[features], y_test):
        test_record = _file_record(test_file)
    X_tr, y_tr = _arrays(X_train, y_train)
    X_te, y_te = _arrays(X_test[features], y_test)
    config: dict[str, Any] | None = None
    if training_config is not None:
        config = {name: getattr(training_config, name) for name in ("pop_size", "ngen", "cxpb", "mutpb")
                  if hasattr(training_config, name)}
    cv_settings: dict[str, Any] | None = None
    if cv is not None:
        cv_settings = {"splitter": type(cv).__name__, "n_splits": getattr(cv, "n_splits", None),
                       "shuffle": getattr(cv, "shuffle", None), "random_state": getattr(cv, "random_state", None)}
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
        "environment": _library_versions(),
        "data": {
            "target": target,
            "features": features,
            "train_files": train_files,
            "test_file": test_record,
            "arrays": {"X_train": array_fingerprint(X_tr), "y_train": array_fingerprint(y_tr),
                       "X_test": array_fingerprint(X_te), "y_test": array_fingerprint(y_te)},
        },
        "settings": {
            "main_objective": "ROC-AUC" if use_roc_auc else "PR-AUC",
            "use_roc_auc": bool(use_roc_auc),
            "pareto_rule": "knee" if use_knee_point else "max_s",
            "training_config": config,
            "cv": cv_settings,
            "seeds": [int(seed) for seed in seeds],
        },
    }


def write_run_manifest(run_directory: str, manifest: dict[str, Any],
                       log: Callable[[str], None] = print) -> str:
    """Write the manifest into the run folder and return its path. An existing manifest (a resumed run)
    must record the same data; changed settings are updated with a note."""
    path: str = os.path.join(run_directory, MANIFEST_FILE)
    existing: dict[str, Any] | None = read_run_manifest(run_directory)
    if existing is not None:
        old, new = existing.get("data", {}), manifest["data"]
        differences: list[str] = [part for part in ("target", "features", "arrays") if old.get(part) != new.get(part)]
        if differences:
            raise ManifestMismatchError(
                f"{path} records other data than this notebook runs on ({', '.join(differences)} differ): "
                f"resume a run only with its own data, or start a new run")
        if existing.get("settings") != manifest["settings"]:
            log(f"NOTE: the run settings differ from those recorded in {path}; the manifest is updated.")
        manifest = {**manifest, "created": existing.get("created", manifest["created"]),
                    "updated": datetime.now().isoformat(timespec="seconds")}
    if manifest["data"]["train_files"] is None or manifest["data"]["test_file"] is None:
        log("NOTE: the training / test data could not be traced back to their CSV files (they were changed "
            "after reading); the run manifest keeps only their fingerprints, so a later re-evaluation of this "
            "run needs a notebook copy (--notebook).")
    atomic_write_json(path, manifest)
    return path


def read_run_manifest(run_directory: str) -> dict[str, Any] | None:
    path: str = os.path.join(run_directory, MANIFEST_FILE)
    return read_json(path) if os.path.isfile(path) else None


def manifest_sha256(run_directory: str) -> str | None:
    path: str = os.path.join(run_directory, MANIFEST_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()


def git_state() -> dict[str, Any]:
    """The repository's commit and whether tracked files have uncommitted changes."""
    try:
        commit: str = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository_root(), capture_output=True,
                                     text=True, timeout=10).stdout.strip()
        dirty: str = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repository_root(),
                                    capture_output=True, text=True, timeout=10).stdout.strip()
        return {"commit": commit or "unknown", "uncommitted_changes": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "uncommitted_changes": None}


# ---------------------------------------------------------------------------
# Rebuilding a finished run's data
# ---------------------------------------------------------------------------

def load_archived_run(run_directory: str, notebook: str | None = None) -> dict[str, Any]:
    """Rebuild a run's training and test data by executing the configuration and data-loading cells of
    the copy of training_notebook.ipynb archived in the run folder (or of the copy `notebook`): every
    code cell from the first one that assigns TARGET_COLUMN to the first one that assigns X_test. This
    reproduces the exact inputs of the run (paths, merged validation file, column whitelist) without
    restating them. The result is checked against the run's checkpoint fingerprint by
    `verify_training_data`."""
    path: str = notebook or os.path.join(run_directory, "training_notebook.ipynb")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} does not exist: the run folder has neither a run manifest nor an archived "
                                f"copy of the notebook, so its data cannot be rebuilt (pass a copy of the notebook "
                                f"that has the run's data settings with --notebook)")
    sources: list[str] = ["".join(cell.get("source", [])) for cell in read_json(path).get("cells", [])
                          if cell.get("cell_type") == "code"]
    start: int | None = next((i for i, source in enumerate(sources)
                              if re.search(r"^TARGET_COLUMN\b", source, re.MULTILINE)), None)
    end: int | None = None if start is None else next(
        (i for i in range(start, len(sources)) if re.search(r"^X_test\b", sources[i], re.MULTILINE)), None)
    if start is None or end is None:
        raise ValueError(f"{path}: no configuration cell (TARGET_COLUMN = ...) followed by a data cell "
                         f"(X_test = ...) was found")
    namespace: dict[str, Any] = {"os": os, "time": time, "numpy": numpy, "pandas": pandas,
                                 "repository_root": repository_root, "__name__": "archived_notebook"}
    previous: str = os.getcwd()
    os.chdir(repository_root())   # the cells use paths relative to the repository root
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for source in sources[start:end + 1]:
                exec(compile(source, path, "exec"), namespace)
    finally:
        os.chdir(previous)
    target: str = namespace["TARGET_COLUMN"]
    df_train: pandas.DataFrame = namespace["df_train"]
    return {"X_train": df_train.drop(columns=[target]), "y_train": df_train[target],
            "X_test": namespace["X_test"], "y_test": namespace["y_test"],
            "use_knee_point": bool(namespace.get("USE_KNEE_POINT_SELECTION", True)),
            "use_roc_auc": namespace.get("USE_ROC_AUC"), "source": "notebook copy" if notebook else "archived notebook"}


def _from_manifest(run_directory: str, manifest: dict[str, Any]) -> dict[str, Any]:
    data: dict[str, Any] = manifest["data"]
    target: str = data["target"]
    features: list[str] = data["features"]

    def read(records: list[dict[str, Any]]) -> tuple[pandas.DataFrame, pandas.Series]:
        paths: list[str] = []
        for record in records:
            path: str = record["absolute"] if os.path.isfile(record["absolute"]) else record["path"]
            paths.append(_resolve(path))
            with open(paths[-1], "rb") as handle:
                if hashlib.sha256(handle.read()).hexdigest() != record["sha256"]:
                    print(f"NOTE: {paths[-1]} changed since the run; the data it gives are checked below.",
                          file=sys.stderr)
        return _read(paths, target, features)

    X_train, y_train = read(data["train_files"])
    X_test, y_test = read([data["test_file"]])
    arrays: dict[str, Any] = data["arrays"]
    X_tr, y_tr = _arrays(X_train, y_train)
    X_te, y_te = _arrays(X_test, y_test)
    for name, array in (("X_train", X_tr), ("y_train", y_tr), ("X_test", X_te), ("y_test", y_te)):
        if array_fingerprint(array) != arrays[name]:
            raise ValueError(f"the data files of {run_directory} no longer give the run's {name} "
                             f"(its fingerprint in {MANIFEST_FILE} differs): the files changed after the run")
    settings: dict[str, Any] = manifest.get("settings", {})
    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
            "use_knee_point": settings.get("pareto_rule", "knee") == "knee",
            "use_roc_auc": settings.get("use_roc_auc"), "source": "run manifest"}


def load_run_data(run_directory: str, notebook: str | None = None) -> dict[str, Any]:
    """A finished run's data: {X_train, y_train, X_test, y_test, use_knee_point, use_roc_auc, source}.
    With `notebook`, from the data cells of that notebook copy; otherwise from the run manifest when it
    names the data files, else from the notebook copy archived in the run folder."""
    if notebook is None:
        manifest: dict[str, Any] | None = read_run_manifest(run_directory)
        if manifest is not None and manifest["data"].get("train_files") and manifest["data"].get("test_file"):
            return _from_manifest(run_directory, manifest)
    return load_archived_run(run_directory, notebook)


def verify_training_data(run_directory: str, X_train: pandas.DataFrame, y_train: Sequence[float]) -> dict[str, Any]:
    """Check that the training data are those the run's checkpoints were trained on -- the feature names
    and the hashes of the standardised training matrix and of the labels, recomputed exactly as the
    notebook computes them (checkpoint_utils.build_training_fingerprint) -- and return the fingerprint."""
    fingerprint: dict[str, Any] = read_json(os.path.join(run_directory, "checkpoints", "training", FINGERPRINT_FILE))
    names: list[str] = list(X_train.columns)
    X_search: numpy.ndarray = numpy.ascontiguousarray(X_train.to_numpy(), dtype=numpy.float64)
    current: dict[str, Any] = {
        "features": {"count": len(names), "sha256": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()},
        "X_train": array_fingerprint(StandardScaler().fit_transform(X_search)),
        "y_train": array_fingerprint(numpy.ascontiguousarray(numpy.asarray(y_train), dtype=numpy.float64)),
    }
    settings: dict[str, Any] = fingerprint.get("settings", {})
    problems: list[str] = []
    if settings.get("features") != current["features"]:
        problems.append("the input variables (names or order) differ")
    for part in ("X_train", "y_train"):
        if settings.get("data", {}).get(part) != current[part]:
            problems.append(f"the training data ({part}) differ")
    if problems:
        raise ValueError(f"the rebuilt data do not match the checkpoints of {run_directory}: "
                         + "; ".join(problems))
    return fingerprint
