from __future__ import annotations

import os
import random
from typing import Sequence, Any

import numpy
from numpy import floating
from deap import base, creator, tools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from training_config import TrainingConfig
from training_utils import save_stats_csv
from plot_utils import plot_multi_objective_convergence, plot_pareto_front
from evaluation_utils import compute_marginal_correlations


class MultiObjectiveTraining:
    """NSGA-II GA that co-optimises ROC/PR-AUC and sign-consistency over binary
    feature-subset masks.
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

        # Per-fold materialisation is lazy: the first call to _evaluate_multi
        # builds the (X_fold_train, X_fold_val, y_fold_train, y_fold_val)
        # tuples AND the fold-level marginal correlation arrays once.
        self._folds: list[tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]] | None = None
        # Per-fold marginal-correlation arrays; index-aligned with self._folds.
        # Computed on each fold's TRAINING partition -- the same data slice the
        # per-fold LR coefficients are fitted on. Keeping both estimators on
        # the same slice keeps the sign-consistency fitness (a) leakage-free
        # -- no validation-partition information enters the fitness -- and
        # (b) an unbiased CV estimate of the out-of-sample sign agreement.
        self._fold_corrs: list[numpy.ndarray] | None = None

        # Per-instance evaluation cache.
        self._cache: dict[tuple[int, ...], tuple[float, float]] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def _ensure_folds(self) -> None:
        """Materialise the CV folds AND their per-fold marginal correlations
        on demand.

        Per-fold correlations are computed on the fold's TRAINING partition
        only, i.e. the same data that the fold's LR coefficients are fitted
        on. This is the methodologically correct pairing:

          * `sign(fold_corr) == sign(fold_beta)` measures whether the two
            estimators agree on the SAME sample slice -- an unbiased CV
            estimate of the population-level sign agreement.
          * Using a full-train correlation reference instead would leak
            validation-partition information into the fitness function
            (the same rows are held out from the LR fit but contribute
            to the correlation reference), which is a subtle but real
            metholodogical flaw.
        """
        if self._folds is not None:
            return

        folds: list[tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]] = []
        fold_corrs: list[numpy.ndarray] = []
        for train_idx, val_idx in self._cv.split(self._X_train, self._y_train):
            X_fold_train: numpy.ndarray = self._X_train[train_idx]
            X_fold_val: numpy.ndarray = self._X_train[val_idx]
            y_fold_train: numpy.ndarray = self._y_train[train_idx]
            y_fold_val: numpy.ndarray = self._y_train[val_idx]

            folds.append((X_fold_train, X_fold_val, y_fold_train, y_fold_val))
            fold_corrs.append(compute_marginal_correlations(X_fold_train, y_fold_train))
        self._folds = folds
        self._fold_corrs = fold_corrs

    def evaluate_multi(self, individual: Sequence[int]) -> tuple[float, float]:
        key: tuple[int, ...] = tuple(individual)
        if key in self._cache:
            return self._cache[key]
        result = self._evaluate_multi(key)
        self._cache[key] = result
        return result

    def _evaluate_multi(self, individual: Sequence[int]) -> tuple[float, float] | tuple[floating[Any], floating[Any]]:
        if sum(individual) == 0:
            return 0.0, 0.0

        self._ensure_folds()

        cols: numpy.ndarray = numpy.where(numpy.array(individual) == 1)[0]
        n_selected_features: int = len(cols)

        auc_scores: list[float] = []
        sign_scores: list[float] = []

        for (X_fold_train_scaled, X_fold_val_scaled, y_fold_train, y_fold_val), fold_corr in zip(
                self._folds, self._fold_corrs):
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
                fold_auc: float = roc_auc_score(y_fold_val, probs)
            else:
                fold_auc: float = average_precision_score(y_fold_val, probs)
            auc_scores.append(fold_auc)

            # Sign-consistency: per-fold agreement between the fold-computed
            # marginal correlation sign and the fold-fitted coefficient sign
            # for the selected features -- both estimators use the same fold
            # training partition, so the check is leakage-free.
            selected_fold_corr: numpy.ndarray = fold_corr[cols]
            check: numpy.ndarray = selected_fold_corr * model.coef_[0]
            penalties: int = numpy.sum((check < 0) | numpy.isclose(check, 0.0, atol=1e-12))
            fold_sign: float = 1.0 - (penalties / n_selected_features)
            sign_scores.append(fold_sign)

        return numpy.mean(auc_scores), numpy.mean(sign_scores)

    def run(self) -> list[creator.Individual]:
        if "FitnessMulti" not in creator.__dict__:
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))

        if "Individual" not in creator.__dict__:
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox: base.Toolbox = base.Toolbox()

        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_bool,
            n=len(self._feature_names),
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate_multi)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / len(self._feature_names))
        toolbox.register("select", tools.selNSGA2)

        pop: list[creator.Individual] = toolbox.population(n=self._config.pop_size)

        # Initial evaluation
        invalid: list[creator.Individual] = [ind for ind in pop if not ind.fitness.valid]
        fitnesses: list[tuple[float, float]] = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        # Crowding distance assignment
        pop: list[creator.Individual] = toolbox.select(pop, len(pop))

        # Pareto archive
        hof: tools.ParetoFront = tools.ParetoFront()
        hof.update(pop)

        # Collect generation 0 stats
        gen_stats: list[dict] = [self._collect_gen_stats(0, pop, hof, len(invalid))]

        # Genetic algorithm
        for gen in range(self._config.ngen):
            # Binary tournament selection (NSGA-II)
            offspring: list[creator.Individual] = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            # Crossover
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= self._config.cxpb:
                    toolbox.mate(ind1, ind2)
                    del ind1.fitness.values
                    del ind2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() <= self._config.mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Fitness evaluation
            invalid: list[creator.Individual] = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses: list[tuple[float, float]] = list(map(toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            # NSGA-II survival selection
            pop: list[creator.Individual] = toolbox.select(pop + offspring, self._config.pop_size)

            # Update global Pareto archive
            hof.update(pop)

            gen_stats.append(self._collect_gen_stats(gen + 1, pop, hof, len(invalid)))

            print(f"Generation {gen + 1} done | Pareto size: {len(hof)}")

        # Save statistics and plots
        front: list[creator.Individual] = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        if self._config.result_directory:
            seed_dir: str = os.path.join(self._config.result_directory, f"seed_{self._config.seed}")
            save_stats_csv(gen_stats, os.path.join(seed_dir, "convergence.csv"))
            plot_multi_objective_convergence(gen_stats, os.path.join(seed_dir, "convergence.png"))
            plot_pareto_front(front, os.path.join(seed_dir, "pareto_front.png"))

        return front

    @staticmethod
    def _collect_gen_stats(gen: int, pop: list, hof: tools.ParetoFront, nevals: int) -> dict:
        fitness_values: list[tuple[float, float]] = [ind.fitness.values for ind in pop]
        auc_values: list[float] = [f[0] for f in fitness_values]
        sign_values: list[float] = [f[1] for f in fitness_values]
        return {
            "gen": gen,
            "nevals": nevals,
            "auc_max": float(numpy.max(auc_values)),
            "auc_mean": float(numpy.mean(auc_values)),
            "auc_std": float(numpy.std(auc_values)),
            "sign_max": float(numpy.max(sign_values)),
            "sign_mean": float(numpy.mean(sign_values)),
            "sign_std": float(numpy.std(sign_values)),
            "pareto_size": len(hof),
        }
