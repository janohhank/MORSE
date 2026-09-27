"""Settings of the robustness suite (robustness_evaluation.py, robustness_utils.py).

Every setting is global. The suite generates its test scenarios from the data by fixed rules, with the
same settings for every dataset -- nothing here (or anywhere in the suite) names a feature. See
docs/robustness.md for what each family of tests does and how to read its results.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RobustnessConfig:
    # ---- Re-weighted test populations (population and dependence shifts) --------------------------------
    # Severity of a POPULATION shift = the effective sample size (ESS) it leaves on the TRAINING rows, as a
    # fraction of the rows. ESS = (sum w)^2 / sum(w^2): a weighted average over the rows is as precise as
    # a plain average over ESS rows. For an exponential tilt of a normal-scored statistic,
    # ESS / n = exp(-strength^2), and the tilt moves that statistic by `strength` standard deviations:
    # 90% -> 0.32 SD, 80% -> 0.47 SD, 70% -> 0.60 SD, 60% -> 0.71 SD. Lower ESS = a larger shift but a
    # noisier estimate, which matters most for the small test sets (136 rows on Arrhythmia).
    ess_levels: tuple[float, ...] = (0.9, 0.8, 0.7, 0.6)
    # Population shifts: the leading principal components of the standardised training covariates ...
    population_components: int = 3
    # ... and the distance from the centre in the leading components that explain this share of the
    # variance ("typical" vs "extreme" rows).
    extremes_variance_share: float = 0.8
    # Dependence shifts: the pairs of inputs whose dependence is changed. Eligible are pairs with a
    # training correlation |r| in [min, max) -- a weaker one has little dependence to change, a stronger
    # one is (nearly) deterministic -- that are not two levels of one one-hot group or a value with its own
    # availability flag, and whose 0/1 members have enough rows in every cell. The strongest `max_pairs`
    # pairs are used.
    pair_min_abs_correlation: float = 0.3
    pair_max_abs_correlation: float = 0.95
    pair_min_train_count: int = 20
    pair_min_test_count: int = 5
    max_pairs: int = 200
    # Severity of a DEPENDENCE shift = how far the pair's correlation (of the scores the tilt uses) moves
    # on the TRAINING rows, as a share of its clean value r:
    #   weakening:     r -> (1 - share) r; share 1 = the pair made uncorrelated, never beyond (a reversal
    #                  of the correlation beyond -0.25 r needs extreme weights for almost every pair);
    #   strengthening: r -> (1 + share) r; strengthening needs far more re-weighting per unit of
    #                  correlation than weakening (and |r| < 1), hence the smaller shares.
    # The ESS each scenario costs is recorded with it.
    dependence_weaken_levels: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    dependence_strengthen_levels: tuple[float, ...] = (0.125, 0.25, 0.375, 0.5)
    # Continuous inputs enter a dependence tilt as normal scores clipped to +-score_clip, so that a few
    # extreme test rows cannot take all the weight.
    score_clip: float = 2.5
    # A scenario is SUPPORTED if the re-weighted test rows keep at least this share of the rows as ESS and
    # this many effective rows in each class, and -- for a dependence shift, whose severity is not an ESS
    # -- the re-weighted training rows at least `min_train_ess_fraction`. Unsupported scenarios are
    # reported but left out of the summaries.
    min_test_ess_fraction: float = 0.3
    min_class_ess: float = 20.0
    min_train_ess_fraction: float = 0.3
    # Every family is summarised over ONE cohort: the scenarios (population axis and direction, dependence
    # pair) usable at EVERY level of the family, so that the curves and the tests across the levels compare
    # the same scenarios. A dependence family is summarised only if its cohort has at least this many pairs.
    min_pairs: int = 10

    # ---- Corrupted test sets ------------------------------------------------------------------------------
    # Severity levels of every corruption family (the clean test set is level 0 and is always included):
    #   measurement noise  - noise SD as a fraction of the input's training SD
    #   recording noise    - probability that a 0/1 input (or one-hot group) of a row is re-drawn
    #   under-recording    - probability that a recorded 1 is lost
    #   value masking      - probability that an available value becomes "not available"
    corruption_levels: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    # Fixed bank of corrupted test sets: every model sees the same realisations, and the same random
    # draws are reused at every level (common random numbers).
    corruption_repetitions: int = 10
    corruption_seed: int = 20260926
    # Under-recording applies to the stand-alone 0/1 inputs whose training prevalence is below this value,
    # i.e. those where 1 is the recorded event.
    under_recording_max_prevalence: float = 0.5

    # ---- Automatic feature schema (inferred from the training rows) ----------------------------------------
    # Availability pairs are found by their names first: a 0/1 flag `<stem>:<suffix>` (or `<stem>_<suffix>`)
    # with the value `<stem>` or `<stem>:<suffix>`, confirmed by the data (the value holds one fill value on
    # every row of one flag level). A value that has no flag of its own name (a flag the preprocessing
    # shared between several values) is paired statistically: it must hold ONE value on every row of one
    # flag level (at least `value_indicator_min_rows` rows) and that value on at most
    # `value_indicator_max_other_share` of the other rows. The share rule would miss imputed medians that
    # are common measured values (e.g. sodium 136), which is why the names come first. For a flag already
    # confirmed by its name, a value strictly inside its range on all k off rows also joins when k chance
    # matches are that unlikely: (share of the other rows holding it)^k <= `value_indicator_max_chance`.
    value_indicator_min_rows: int = 10
    value_indicator_max_other_share: float = 0.05
    value_indicator_max_chance: float = 1e-6
    # DIAGNOSTIC ONLY: a combination of two 0/1 values that never occurs in training although independence
    # predicts at least this many rows is counted when a corruption creates it (corruption_diagnostics.csv).
    # It is an unusual association, not proof of an impossible combination, so it is never prevented.
    unseen_combination_min_expected: float = 5.0

    # ---- Summaries --------------------------------------------------------------------------------------------
    # The severities used for the overview figure and the sign-consistency analysis; the CSV summaries
    # and tests cover every level.
    headline_ess: float = 0.6
    headline_weaken: float = 1.0
    headline_strengthen: float = 0.5
    headline_corruption_level: float = 0.5

    def __post_init__(self) -> None:
        if not self.ess_levels or any(not 0.0 < level < 1.0 for level in self.ess_levels):
            raise ValueError(f"ess_levels must lie in (0, 1), got {self.ess_levels}")
        if list(self.ess_levels) != sorted(self.ess_levels, reverse=True):
            raise ValueError(f"ess_levels must be in decreasing order (milder first), got {self.ess_levels}")
        for name, levels, upper in (("dependence_weaken_levels", self.dependence_weaken_levels, 1.0),
                                    ("dependence_strengthen_levels", self.dependence_strengthen_levels, None)):
            if not levels or any(level <= 0.0 or (upper is not None and level > upper) for level in levels):
                raise ValueError(f"{name} must be positive shares" + (" of at most 1" if upper else "")
                                 + f", got {levels}")
            if list(levels) != sorted(levels) or len(set(levels)) != len(levels):
                raise ValueError(f"{name} must be increasing (milder first), got {levels}")
        if self.headline_weaken not in self.dependence_weaken_levels:
            raise ValueError(f"headline_weaken {self.headline_weaken} is not one of dependence_weaken_levels")
        if self.headline_strengthen not in self.dependence_strengthen_levels:
            raise ValueError(f"headline_strengthen {self.headline_strengthen} is not one of "
                             f"dependence_strengthen_levels")
        if not 0.0 < self.min_train_ess_fraction <= 1.0:
            raise ValueError(f"min_train_ess_fraction must lie in (0, 1], got {self.min_train_ess_fraction}")
        if any(not 0.0 < level <= 1.0 for level in self.corruption_levels):
            raise ValueError(f"corruption_levels must lie in (0, 1], got {self.corruption_levels}")
        if list(self.corruption_levels) != sorted(self.corruption_levels):
            raise ValueError(f"corruption_levels must be increasing, got {self.corruption_levels}")
        if self.headline_ess not in self.ess_levels:
            raise ValueError(f"headline_ess {self.headline_ess} is not one of ess_levels {self.ess_levels}")
        if self.headline_corruption_level not in self.corruption_levels:
            raise ValueError(f"headline_corruption_level {self.headline_corruption_level} is not one of "
                             f"corruption_levels {self.corruption_levels}")
        if not 0.0 <= self.pair_min_abs_correlation < self.pair_max_abs_correlation <= 1.0:
            raise ValueError("need 0 <= pair_min_abs_correlation < pair_max_abs_correlation <= 1")
        if self.corruption_repetitions < 1 or self.max_pairs < 0 or self.population_components < 0:
            raise ValueError("corruption_repetitions must be >= 1, max_pairs and population_components >= 0")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
