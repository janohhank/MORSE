from __future__ import annotations

from training_config import TrainingConfig


class AllFeaturesTraining:
    """No-feature-selection baseline.

    Trivially returns the all-ones bit-vector. Kept as an explicit class so
    that the per-seed training loop can call all compared methods --
    MultiObjectiveTraining, SingleObjectiveTraining, ForwardStepwiseTraining
    and AllFeaturesTraining -- through the same `.run()` shape and populate
    the same `training_results_*` dictionaries.
    """

    def __init__(self,
                 config: TrainingConfig,
                 feature_names: list[str]) -> None:
        self._config: TrainingConfig = config
        self._feature_names: list[str] = feature_names

    def run(self) -> list[int]:
        """Return the all-ones mask (every feature selected)."""
        return [1] * len(self._feature_names)
