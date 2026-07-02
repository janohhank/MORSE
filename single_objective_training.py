from __future__ import annotations

import os
import random
from typing import Sequence, Any

import numpy
from deap import creator, base, tools, algorithms
from numpy import floating
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from training_config import TrainingConfig
from training_utils import save_stats_csv, plot_single_objective_convergence


class SingleObjectiveTraining:
    """Single-objective AUC-only GA baseline (DEAP `eaMuPlusLambda`).
    """

    def __init__(self,
                 config: TrainingConfig,
                 feature_names: list[str],
                 X_train: numpy.ndarray,
                 y_train: numpy.ndarray,
                 cv: StratifiedKFold) -> None:
        self._config: TrainingConfig = config
        self._feature_names: list[str] = feature_names
        self._X_train: numpy.ndarray = X_train
        self._y_train: numpy.ndarray = y_train
        self._cv: StratifiedKFold = cv

        # Per-fold materialisation is lazy: the first call to _evaluate_single
        # builds the (scaled_train, scaled_val, y_train, y_val) tuples once.
        self._folds: list[tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]] | None = None

        # Per-instance evaluation cache.
        self._cache: dict[tuple[int, ...], tuple[float]] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def _ensure_folds(self) -> None:
        if self._folds is not None:
            return
        folds: list[tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]] = []
        for train_idx, val_idx in self._cv.split(self._X_train, self._y_train):
            X_fold_train: numpy.ndarray = self._X_train[train_idx]
            X_fold_val: numpy.ndarray = self._X_train[val_idx]
            y_fold_train: numpy.ndarray = self._y_train[train_idx]
            y_fold_val: numpy.ndarray = self._y_train[val_idx]

            scaler: StandardScaler = StandardScaler()
            X_fold_train_scaled: numpy.ndarray = scaler.fit_transform(X_fold_train)
            X_fold_val_scaled: numpy.ndarray = scaler.transform(X_fold_val)

            folds.append((X_fold_train_scaled, X_fold_val_scaled, y_fold_train, y_fold_val))
        self._folds = folds

    def evaluate_single(self, individual: Sequence[int]) -> tuple[float]:
        key: tuple[int, ...] = tuple(individual)
        if key in self._cache:
            return self._cache[key]
        result = self._evaluate_single(key)
        self._cache[key] = result
        return result

    def _evaluate_single(self, individual: Sequence[int]) -> tuple[float] | tuple[floating[Any]]:
        if sum(individual) == 0:
            return (0.0,)

        self._ensure_folds()

        cols: numpy.ndarray = numpy.where(numpy.array(individual) == 1)[0]

        auc_scores: list[float] = []
        for X_fold_train_scaled, X_fold_val_scaled, y_fold_train, y_fold_val in self._folds:
            X_fold_train_sub: numpy.ndarray = X_fold_train_scaled[:, cols]
            X_fold_val_sub: numpy.ndarray = X_fold_val_scaled[:, cols]

            model: LogisticRegression = LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                random_state=self._config.seed)
            model.fit(X_fold_train_sub, y_fold_train)

            probs: numpy.ndarray = model.predict_proba(X_fold_val_sub)[:, 1]
            if self._config.use_roc_auc:
                fold_auc = roc_auc_score(y_fold_val, probs)
            else:
                fold_auc = average_precision_score(y_fold_val, probs)
            auc_scores.append(fold_auc)

        return (numpy.mean(auc_scores),)

    def run(self) -> creator.Individual:
        if "FitnessSingle" not in creator.__dict__:
            creator.create("FitnessSingle", base.Fitness, weights=(1.0,))

        if "IndividualSingle" not in creator.__dict__:
            creator.create("IndividualSingle", list, fitness=creator.FitnessSingle)

        toolbox: base.Toolbox = base.Toolbox()

        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.IndividualSingle,
            toolbox.attr_bool,
            n=len(self._feature_names),
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.evaluate_single)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / len(self._feature_names))
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop: list[creator.IndividualSingle] = toolbox.population(n=self._config.pop_size)
        hof: tools.HallOfFame = tools.HallOfFame(1)

        stats: tools.Statistics = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", numpy.max)
        stats.register("avg", numpy.mean)
        stats.register("min", numpy.min)
        stats.register("std", numpy.std)

        _, logbook = algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            mu=self._config.pop_size,
            lambda_=self._config.pop_size,
            cxpb=self._config.cxpb,
            mutpb=self._config.mutpb,
            ngen=self._config.ngen,
            stats=stats,
            halloffame=hof,
            verbose=True
        )

        # Save statistics and plots
        if self._config.result_directory:
            gen_stats: list[dict] = [
                {"gen": rec["gen"], "nevals": rec["nevals"],
                 "max": rec["max"], "avg": rec["avg"],
                 "min": rec["min"], "std": rec["std"]}
                for rec in logbook
            ]
            seed_dir: str = os.path.join(self._config.result_directory, f"seed_{self._config.seed}")
            save_stats_csv(gen_stats, os.path.join(seed_dir, "convergence.csv"))
            plot_single_objective_convergence(gen_stats, os.path.join(seed_dir, "convergence.png"))

        best_individual: creator.IndividualSingle = hof[0]
        return best_individual
