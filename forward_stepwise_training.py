from __future__ import annotations

import numpy
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from training_config import TrainingConfig


class ForwardStepwiseTraining:
    """Forward stepwise feature selection baseline.

    A thin OO wrapper around sklearn's `SequentialFeatureSelector` that
    matches the shape of `SingleObjectiveTraining` and
    `MultiObjectiveTraining` so all three baselines can be invoked from the
    per-seed training loop with a uniform pattern:

        result = ForwardStepwiseTraining(...).run()

    Refit on each inner CV fold (no leakage across folds).

    `inner_n_jobs` maps to sklearn's `n_jobs` for SFS's own CV; it should be
    forced to 1 when the outer seed pool is parallel (loky x sklearn would
    otherwise oversubscribe cores).
    """

    def __init__(self,
                 config: TrainingConfig,
                 X_train: numpy.ndarray,
                 y_train: numpy.ndarray,
                 cv: StratifiedKFold,
                 inner_n_jobs: int = 1) -> None:
        self._config: TrainingConfig = config
        self._X_train: numpy.ndarray = X_train
        self._y_train: numpy.ndarray = y_train
        self._cv: StratifiedKFold = cv
        self._inner_n_jobs: int = inner_n_jobs

    def run(self) -> list[int]:
        """Fit SFS and return the selected feature mask as a bit-vector
        (list[int], values in {0, 1}) aligned with the column order of
        `X_train`."""
        pipe: Pipeline = Pipeline([
            ("lr", LogisticRegression(
                penalty="l2", solver="lbfgs",
                max_iter=1000, random_state=self._config.seed)),
        ])

        sfs: SequentialFeatureSelector = SequentialFeatureSelector(
            pipe,
            n_features_to_select="auto",
            # With n_features_to_select="auto", scikit-learn only applies a
            # tol-based stopping rule (keep adding features while the CV
            # score improves by more than `tol`) when `tol` is explicitly
            # set. Leaving `tol=None` (the default) silently falls back to
            # selecting a FIXED n_features // 2, regardless of whether those
            # features still help -- not the adaptive baseline this class is
            # meant to provide.
            tol=1e-3,
            direction="forward",
            scoring="roc_auc" if self._config.use_roc_auc else "average_precision",
            cv=self._cv,
            n_jobs=self._inner_n_jobs,
        )
        sfs.fit(self._X_train, self._y_train)

        return [1 if m else 0 for m in sfs.get_support()]
