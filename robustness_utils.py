"""Building blocks of the robustness suite (robustness_evaluation.py; docs/robustness.md).

Every scenario is generated from the TRAINING data by fixed rules (robustness_config.RobustnessConfig).
Nothing here names a feature or a dataset, so the same code runs on every dataset and no scenario can be
picked after seeing which one favours a method. There are two kinds of stress, with different meanings:

* RE-WEIGHTING (population and dependence shifts) changes which rows the test population consists of,
  i.e. P(x), and never a row itself. Every row keeps its own (x, y) pair, so P(y | x) is unchanged (a
  covariate shift). Only real rows are re-emphasised, so a re-weighting can never produce an impossible
  combination of values and needs no knowledge of what the features mean.
* CORRUPTION changes recorded values: measurement noise, recording errors of 0/1 inputs, lost records,
  values that are not available. It does not keep P(y | observed x), and it has to keep the data valid
  by its STRUCTURAL rules: a one-hot group keeps at most one level (exactly one if it is exhaustive), a
  "not available" flag keeps its value at the fill value. Those rules are inferred from the names and the
  training data (`infer_feature_schema`) and checked after every corruption. A combination of 0/1 values
  that merely never occurs in training is an association, not an impossibility: it is counted when a
  corruption creates it, never prevented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy
import pandas
from scipy.optimize import brentq
from scipy.stats import norm

from robustness_config import RobustnessConfig

# Upper end of the strength search of a tilt. A tilt that still leaves more than the target effective
# sample size at this strength is "saturated": the training data cannot support a shift that large in
# this direction (e.g. strengthening a dependence that is already close to its maximum).
MAX_TILT_STRENGTH: float = 20.0
# Largest deviation of a balanced moment (of standardised scores) from its target that still counts as
# balanced (DependenceTilt).
BALANCE_TOLERANCE: float = 1e-6

CORRUPTION_FAMILIES: tuple[str, ...] = ("gaussian_noise", "binary_redraw", "under_recording", "value_masking")


# ---------------------------------------------------------------------------
# Weighted metrics and effective sample size
# ---------------------------------------------------------------------------

def effective_sample_fraction(weights: Sequence[float]) -> float:
    """ESS / n of a weight vector, ESS = (sum w)^2 / sum(w^2): a weighted average over the n rows is as
    precise as a plain average over ESS rows. 1.0 for equal weights, 1/n when one row has all the weight."""
    w: numpy.ndarray = numpy.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum() / w.size)


def class_effective_sizes(weights: Sequence[float], y: Sequence[int]) -> tuple[float, float]:
    """ESS in rows of the negatives and of the positives: a ranking metric compares the two classes, so
    its precision depends on both."""
    w: numpy.ndarray = numpy.asarray(weights, dtype=float)
    labels: numpy.ndarray = numpy.asarray(y).astype(int)
    sizes: list[float] = []
    for cls in (0, 1):
        part: numpy.ndarray = w[labels == cls]
        sizes.append(float(part.sum() ** 2 / (part ** 2).sum()) if part.size else 0.0)
    return sizes[0], sizes[1]


def normalise_within_classes(weights: Sequence[float], y: Sequence[int]) -> numpy.ndarray:
    """Scale the weights of each class to sum to that class's row count, so that the weighted outcome
    prevalence equals the unweighted one. ROC-AUC does not change (it compares positives with negatives,
    so a constant factor per class cancels). Average precision then measures the shifted population at
    the unshifted prevalence, instead of mixing the shift with a change of the class prior."""
    w: numpy.ndarray = numpy.asarray(weights, dtype=float).copy()
    labels: numpy.ndarray = numpy.asarray(y).astype(int)
    for cls in (0, 1):
        rows: numpy.ndarray = labels == cls
        if rows.any():
            w[rows] *= rows.sum() / w[rows].sum()
    return w


def weighted_roc_auc(y: Sequence[int], scores: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """ROC-AUC under per-row weights (sklearn's `roc_auc_score(..., sample_weight=...)`, ties counted
    half), computed in O(n log n) without building the curve."""
    labels: numpy.ndarray = numpy.asarray(y).astype(int)
    values: numpy.ndarray = numpy.asarray(scores, dtype=float)
    w: numpy.ndarray = numpy.ones(values.size) if weights is None else numpy.asarray(weights, dtype=float)
    order: numpy.ndarray = numpy.argsort(values, kind="mergesort")
    values, labels, w = values[order], labels[order], w[order]
    starts: numpy.ndarray = numpy.r_[0, numpy.flatnonzero(numpy.diff(values)) + 1]
    negatives: numpy.ndarray = numpy.add.reduceat(numpy.where(labels == 0, w, 0.0), starts)
    positives: numpy.ndarray = numpy.add.reduceat(numpy.where(labels == 1, w, 0.0), starts)
    below: numpy.ndarray = numpy.cumsum(negatives) - negatives
    return float(numpy.sum(positives * (below + 0.5 * negatives)) / (positives.sum() * negatives.sum()))


def weighted_average_precision(y: Sequence[int], scores: Sequence[float],
                               weights: Sequence[float] | None = None) -> float:
    """Average precision (PR-AUC) under per-row weights, identical to sklearn's
    `average_precision_score(..., sample_weight=...)` including its handling of tied scores."""
    labels: numpy.ndarray = numpy.asarray(y).astype(float)
    values: numpy.ndarray = numpy.asarray(scores, dtype=float)
    w: numpy.ndarray = numpy.ones(values.size) if weights is None else numpy.asarray(weights, dtype=float)
    order: numpy.ndarray = numpy.argsort(-values, kind="mergesort")
    values, labels, w = values[order], labels[order], w[order]
    last: numpy.ndarray = numpy.r_[numpy.flatnonzero(numpy.diff(values)), values.size - 1]
    true_positives: numpy.ndarray = numpy.cumsum(labels * w)[last]
    false_positives: numpy.ndarray = numpy.cumsum((1.0 - labels) * w)[last]
    precision: numpy.ndarray = true_positives / (true_positives + false_positives)
    recall: numpy.ndarray = true_positives / true_positives[-1]
    return float(numpy.sum(numpy.diff(numpy.r_[0.0, recall]) * precision))


def weighted_correlation(a: Sequence[float], b: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Pearson correlation of two columns under per-row weights."""
    x: numpy.ndarray = numpy.asarray(a, dtype=float)
    z: numpy.ndarray = numpy.asarray(b, dtype=float)
    w: numpy.ndarray = numpy.ones(x.size) if weights is None else numpy.asarray(weights, dtype=float)
    w = w / w.sum()
    dx: numpy.ndarray = x - w @ x
    dz: numpy.ndarray = z - w @ z
    denominator: float = float(numpy.sqrt((w @ dx ** 2) * (w @ dz ** 2)))
    return float(w @ (dx * dz)) / denominator if denominator > 0 else float("nan")


# ---------------------------------------------------------------------------
# Tilts and their severity
# ---------------------------------------------------------------------------

class NormalScores:
    """Maps values of a statistic to normal scores under its TRAINING distribution: the standard-normal
    quantile of the mid-rank empirical CDF of the training values. Tied values share one score, and a
    value outside the training range gets the most extreme training score (the CDF is clipped to
    [1/(2n), 1 - 1/(2n)]). On the training rows the scores are standard normal up to ties, whatever the
    distribution of the statistic -- which is what puts every tilt on one severity scale: a tilt
    exp(s * score) moves the score by s standard deviations and leaves ESS / n = exp(-s^2)."""

    def __init__(self, train_values: Sequence[float]) -> None:
        self._ordered: numpy.ndarray = numpy.sort(numpy.asarray(train_values, dtype=float))
        if self._ordered.size == 0:
            raise ValueError("NormalScores needs at least one training value")

    def __call__(self, values: Sequence[float]) -> numpy.ndarray:
        x: numpy.ndarray = numpy.asarray(values, dtype=float)
        n: int = self._ordered.size
        below: numpy.ndarray = numpy.searchsorted(self._ordered, x, side="left")
        up_to: numpy.ndarray = numpy.searchsorted(self._ordered, x, side="right")
        cdf: numpy.ndarray = (below + 0.5 * (up_to - below)) / n
        return norm.ppf(numpy.clip(cdf, 0.5 / n, 1.0 - 0.5 / n))


def exponential_tilt(statistic: Sequence[float], strength: float) -> numpy.ndarray:
    """Weights proportional to exp(strength * statistic), scaled to mean 1 (overflow-safe)."""
    exponent: numpy.ndarray = strength * numpy.asarray(statistic, dtype=float)
    w: numpy.ndarray = numpy.exp(exponent - exponent.max())
    return w / w.mean()


def calibrate_strengths(ess_at: Callable[[float], float], targets: Sequence[float],
                        max_strength: float = MAX_TILT_STRENGTH,
                        initial_strength: float = 0.05) -> list[tuple[float, bool]]:
    """(strength, saturated) for every target ESS fraction, in the given (decreasing) order: the
    strength >= 0 at which the TRAINING effective sample fraction `ess_at(strength)` falls to the
    target (ess_at(0) = 1). The strength is searched upwards from `initial_strength`, doubling until the
    target is crossed, and then refined by root finding -- so consecutive evaluations stay close to each
    other, which keeps a warm-started solver (DependenceTilt) on its path.

    `saturated` is True when the target cannot be reached: even `max_strength` leaves more than the
    target, or `ess_at` returns NaN before the target is crossed (the tilt cannot be realised, e.g.
    because a dependence cannot be strengthened further with the marginal moments held fixed). A
    saturated target -- and every stronger one after it -- gets `max_strength` and must not be used."""
    if any(not 0.0 < target < 1.0 for target in targets):
        raise ValueError(f"target ESS fractions must lie in (0, 1), got {list(targets)}")
    if list(targets) != sorted(targets, reverse=True):
        raise ValueError(f"target ESS fractions must be decreasing, got {list(targets)}")
    results: list[tuple[float, bool]] = []
    low: float = 0.0
    high: float = min(initial_strength, max_strength)
    high_ess: float = ess_at(high)
    exhausted: bool = False
    for target in targets:
        while not exhausted:
            if numpy.isnan(high_ess):
                exhausted = True
            elif high_ess < target:
                break
            elif high >= max_strength:
                exhausted = True
            else:
                low, high = high, min(2.0 * high, max_strength)
                high_ess = ess_at(high)
        if exhausted:
            results.append((max_strength, True))
            continue

        def distance(strength: float, goal: float = target) -> float:
            ess: float = ess_at(strength)
            return -1.0 if numpy.isnan(ess) else ess - goal

        try:
            strength: float = float(brentq(distance, low, high, xtol=1e-7))
        except (ValueError, RuntimeError):
            strength = float("nan")
        reached: float = ess_at(strength) if not numpy.isnan(strength) else float("nan")
        if numpy.isnan(reached) or abs(reached - target) > 1e-3:
            exhausted = True
            results.append((max_strength, True))
            continue
        results.append((strength, False))
        low = strength
    return results


def calibrate_strength(ess_at: Callable[[float], float], target: float,
                       max_strength: float = MAX_TILT_STRENGTH) -> tuple[float, bool]:
    """`calibrate_strengths` for a single target."""
    return calibrate_strengths(ess_at, [target], max_strength)[0]


# ---------------------------------------------------------------------------
# Automatic feature schema (training rows only)
# ---------------------------------------------------------------------------

@dataclass
class ValueIndicator:
    """A 0/1 flag that marks values as not available (e.g. `ADM_RATE:missing` = 1 or `albumin:Binary` = 0).
    Whenever `flag == off_level`, every column of `fills` holds its fill value (the imputed value).
    `named` lists the values paired with the flag by their names (`albumin:Binary` / `albumin:Value`); the
    other values of `fills` were paired statistically: values without a flag of their own that the
    preprocessing covered with a shared flag (e.g. a census block with one `:missing` flag)."""
    flag: str
    off_level: int
    fills: dict[str, float]
    named: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class UnresolvedAvailability:
    """A possible flag / value pair that is NOT used: proposed by the names but not confirmed by the data,
    or a value that looks imputed on a flag level but fails the statistical rule. Reported (schema.json and
    the report) so that a missed relationship is visible instead of silently corrupted."""
    flag: str
    value: str
    reason: str


@dataclass(frozen=True)
class UnseenCombination:
    """`a == a_value` together with `b == b_value` never occurs in the training rows although independence
    predicts `expected` of them. A DIAGNOSTIC only: it is an unusual association, not proof that the
    combination is impossible (such combinations do occur in test sets), so the corruptions may create it;
    corruption_diagnostics.csv counts how often they do. The genuine structural rules -- one-hot groups
    and availability pairs -- are enforced by construction instead."""
    a: str
    a_value: int
    b: str
    b_value: int
    expected: float


@dataclass
class FeatureSchema:
    """What the corruptions need to know about the inputs, inferred from the training rows by
    `infer_feature_schema`."""
    features: list[str]
    binary: list[str]
    continuous: list[str]
    constant: list[str]
    other: list[str]
    # one-hot groups: name prefix -> member columns (at most one of them is 1 in every training row) ...
    dummy_families: dict[str, list[str]]
    # ... and the training frequency of every member, followed by that of "none of them" (0 if the
    # group is exhaustive)
    family_frequencies: dict[str, list[float]]
    value_indicators: list[ValueIndicator]
    # stand-alone 0/1 inputs (not in a one-hot group, not an availability flag)
    redraw_columns: list[str]
    # the stand-alone 0/1 inputs whose 1 is the recorded event (training prevalence below the threshold)
    under_recording_columns: list[str]
    training_prevalence: dict[str, float] = field(default_factory=dict)
    unresolved_availability: list[UnresolvedAvailability] = field(default_factory=list)
    unseen_combinations: list[UnseenCombination] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Counts, for the report."""
        named: int = sum(len(indicator.named) for indicator in self.value_indicators)
        return {
            "inputs": len(self.features),
            "binary": len(self.binary),
            "continuous": len(self.continuous),
            "constant": len(self.constant),
            "other": len(self.other),
            "one_hot_groups": len(self.dummy_families),
            "one_hot_levels": sum(len(members) for members in self.dummy_families.values()),
            "availability_flags": len(self.value_indicators),
            "values_with_flag": len({c for indicator in self.value_indicators for c in indicator.fills}),
            "values_paired_by_name": named,
            "values_paired_statistically": sum(len(i.fills) for i in self.value_indicators) - named,
            "unresolved_availability": len(self.unresolved_availability),
            "unseen_combinations": len(self.unseen_combinations),
            "stand_alone_binary": len(self.redraw_columns),
            "under_recording_inputs": len(self.under_recording_columns),
        }

    def to_dict(self) -> dict[str, Any]:
        """Everything, JSON-ready (schema.json)."""
        return {
            "summary": self.summary(),
            "binary": self.binary,
            "continuous": self.continuous,
            "constant": self.constant,
            "other": self.other,
            "one_hot_groups": {prefix: {"members": members, "frequencies": self.family_frequencies[prefix]}
                               for prefix, members in self.dummy_families.items()},
            "value_indicators": [{"flag": i.flag, "off_level": i.off_level, "fills": i.fills,
                                  "paired_by_name": i.named,
                                  "paired_statistically": [c for c in i.fills if c not in i.named]}
                                 for i in self.value_indicators],
            "unresolved_availability": [{"flag": u.flag, "value": u.value, "reason": u.reason}
                                        for u in self.unresolved_availability],
            "unseen_combinations": [{"a": f.a, "a_value": f.a_value, "b": f.b, "b_value": f.b_value,
                                     "expected_rows": round(f.expected, 2)} for f in self.unseen_combinations],
            "stand_alone_binary": self.redraw_columns,
            "under_recording_inputs": self.under_recording_columns,
        }


def infer_feature_schema(X_train: pandas.DataFrame, config: RobustnessConfig | None = None) -> FeatureSchema:
    """Infer the input types and the rules that keep corrupted data valid, from the TRAINING rows only.

    * Types: 0/1 inputs, continuous inputs (more than two distinct values), constants, and "other"
      (two values that are not 0/1; left alone by the corruptions).
    * Value/availability pairs (an imputed value with its missing / measured indicator; the flag level on
      which the value is constant means "not available"):
        - by name first: a 0/1 flag `<stem><sep><suffix>` (sep one of `:_. `) and a continuous value named
          `<stem>` or `<stem><sep><suffix>` -- e.g. `albumin:Binary` / `albumin:Value`, `ADM_RATE:missing` /
          `ADM_RATE` -- used when the data confirm it: the value holds one value on every row of exactly
          one level of the flag. The name is the evidence, so no share rule applies (an imputed median is
          often a common measured value) and any number of rows suffices;
        - statistically, for continuous values that have no flag of their own name: the value holds ONE
          value on every row of one level of a 0/1 input (at least `value_indicator_min_rows` rows) and
          that value on at most `value_indicator_max_other_share` of the other rows. Such a value may
          belong to several flags (a flag shared by a block of values). A value with a flag of its own
          belongs to that flag only.
      Name-proposed pairs the data do not confirm, and values that look imputed but fail the statistical
      rule, are listed as unresolved.
    * One-hot groups: 0/1 inputs named `<prefix>_<level>` (pandas' dummy encoding) with at most one of
      them equal to 1 in every training row. The name only proposes a group; the data must confirm it.
    * Unseen combinations (a diagnostic): pairs of 0/1 values that never occur together although
      independence predicts at least `unseen_combination_min_expected` rows.
    """
    config = config or RobustnessConfig()
    features: list[str] = list(X_train.columns)
    A: numpy.ndarray = X_train.to_numpy(dtype=float)
    if numpy.isnan(A).any():
        raise ValueError("the training data contain missing values; the robustness suite expects the "
                         "preprocessed (imputed) data the models were trained on")
    index: dict[str, int] = {name: j for j, name in enumerate(features)}

    binary: list[str] = []
    continuous: list[str] = []
    constant: list[str] = []
    other: list[str] = []
    for j, name in enumerate(features):
        values: numpy.ndarray = numpy.unique(A[:, j])
        if values.size <= 1:
            constant.append(name)
        elif values.size == 2 and values[0] == 0.0 and values[1] == 1.0:
            binary.append(name)
        elif values.size == 2:
            other.append(name)
        else:
            continuous.append(name)

    indicators, unresolved = _find_value_indicators(A, index, binary, continuous, config)
    flags: set[str] = {indicator.flag for indicator in indicators}

    candidates: dict[str, list[str]] = {}
    for name in binary:
        if name not in flags and "_" in name:
            candidates.setdefault(name.rsplit("_", 1)[0], []).append(name)
    families: dict[str, list[str]] = {}
    frequencies: dict[str, list[float]] = {}
    n: int = A.shape[0]
    for prefix, members in candidates.items():
        if len(members) < 2:
            continue
        block: numpy.ndarray = A[:, [index[m] for m in members]]
        if block.sum(axis=1).max() <= 1:
            families[prefix] = members
            counts: numpy.ndarray = block.sum(axis=0)
            frequencies[prefix] = [float(c / n) for c in counts] + [float((n - counts.sum()) / n)]

    unseen: list[UnseenCombination] = _find_unseen_combinations(
        A, index, binary, config.unseen_combination_min_expected)
    in_family: set[str] = {m for members in families.values() for m in members}
    prevalence: dict[str, float] = {name: float(A[:, index[name]].mean()) for name in binary}
    redraw: list[str] = [name for name in binary if name not in in_family and name not in flags]
    under: list[str] = [name for name in redraw if prevalence[name] < config.under_recording_max_prevalence]

    return FeatureSchema(
        features=features, binary=binary, continuous=continuous, constant=constant, other=other,
        dummy_families=families, family_frequencies=frequencies, value_indicators=indicators,
        redraw_columns=redraw, under_recording_columns=under, training_prevalence=prevalence,
        unresolved_availability=unresolved, unseen_combinations=unseen)


# Characters that separate the parts of a column name (`albumin:Binary`, `ADM_RATE:missing`, `x.flag`).
NAME_SEPARATORS: str = ":_. "


def _split_name(name: str) -> tuple[str, str] | None:
    """(stem, separator) of a name, split at its last separator: "albumin:Binary" -> ("albumin", ":"),
    "ADM_RATE:missing" -> ("ADM_RATE", ":"), "bun_cre:Value" -> ("bun_cre", ":"); None without one."""
    cut: int = max(name.rfind(separator) for separator in NAME_SEPARATORS)
    return (name[:cut], name[cut]) if cut > 0 else None


def _named_values(flag: str, continuous: list[str]) -> list[str]:
    """The continuous inputs whose names pair them with `flag` = `<stem><sep><suffix>`: the value `<stem>`
    itself, or a value `<stem><sep><other suffix>` with the same separator (so `ADM_RATE:missing` pairs
    with `ADM_RATE` but not with `ADM_RATE_ALL`)."""
    split: tuple[str, str] | None = _split_name(flag)
    if split is None:
        return []
    return [value for value in continuous if value == split[0] or (value != flag and _split_name(value) == split)]


def _find_value_indicators(A: numpy.ndarray, index: dict[str, int], binary: list[str], continuous: list[str],
                           config: RobustnessConfig) -> tuple[list[ValueIndicator], list[UnresolvedAvailability]]:
    if not continuous or not binary:
        return [], []
    found: dict[str, ValueIndicator] = {}
    unresolved: list[UnresolvedAvailability] = []

    # 1. pairs proposed by the names, confirmed by the data
    named: set[str] = set()
    for flag in binary:
        x: numpy.ndarray = A[:, index[flag]]
        for value in _named_values(flag, continuous):
            v: numpy.ndarray = A[:, index[value]]
            # a continuous value has more than two distinct values, so it is constant on at most one level
            levels: list[int] = [level for level in (0, 1) if numpy.ptp(v[x == level]) == 0]
            if not levels:
                unresolved.append(UnresolvedAvailability(
                    flag, value, "paired by name, but the value is not constant on either level of the flag"))
                continue
            level: int = levels[0]
            indicator: ValueIndicator | None = found.get(flag)
            if indicator is None:
                indicator = found[flag] = ValueIndicator(flag=flag, off_level=level, fills={}, named=[])
            elif indicator.off_level != level:
                unresolved.append(UnresolvedAvailability(
                    flag, value, f"paired by name, but constant on flag level {level} while the flag's other "
                                 f"values are constant on level {indicator.off_level}"))
                continue
            indicator.fills[value] = float(v[x == level][0])
            indicator.named.append(value)
            named.add(value)

    # 2. values without a flag of their own name, statistically:
    #    rule S (any 0/1 input as the flag): one value on every row of a flag level and on at most
    #           `value_indicator_max_other_share` of the other rows;
    #    rule B (a flag confirmed by name, i.e. a known availability flag shared by a block of values):
    #           one value strictly inside the value's range (an imputed centre, not a structural zero at
    #           the boundary) on all k rows of the flag's off level, where a share s of the other rows
    #           holds it too -- k chance matches have probability s^k <= `value_indicator_max_chance`.
    #    Missingness is often nested (no faculty data => no SAT scores either), so several flags can claim
    #    one value; it belongs to the claiming flag with the most off rows -- the one that explains its
    #    imputation -- or to all of them on a tie (with one fill value).
    remaining: list[str] = [c for c in continuous if c not in named]
    if remaining:
        C: numpy.ndarray = A[:, [index[c] for c in remaining]]
        low: numpy.ndarray = C.min(axis=0)
        high: numpy.ndarray = C.max(axis=0)
        near_miss: dict[str, UnresolvedAvailability] = {}
        claims: dict[str, list[tuple[int, str, int, float]]] = {}      # value -> (off rows, flag, level, fill)
        for flag in binary:
            x = A[:, index[flag]]
            confirmed: ValueIndicator | None = found.get(flag)
            best: tuple[int, dict[str, float], int] | None = None
            for level in ((confirmed.off_level,) if confirmed is not None else (0, 1)):
                rows: numpy.ndarray = x == level
                n_rows: int = int(rows.sum())
                if n_rows < config.value_indicator_min_rows or (~rows).sum() < config.value_indicator_min_rows:
                    continue
                inside: numpy.ndarray = C[rows]
                outside: numpy.ndarray = C[~rows]
                fills: dict[str, float] = {}
                for k in numpy.flatnonzero(numpy.ptp(inside, axis=0) == 0):
                    fill: float = float(inside[0, k])
                    share: float = float(numpy.mean(outside[:, k] == fill))
                    if share <= config.value_indicator_max_other_share:
                        fills[remaining[k]] = fill
                    elif confirmed is not None and low[k] < fill < high[k]:
                        if share ** n_rows <= config.value_indicator_max_chance:
                            fills[remaining[k]] = fill
                        elif remaining[k] not in near_miss:
                            near_miss[remaining[k]] = UnresolvedAvailability(
                                flag, remaining[k], f"holds {fill:g} on all {n_rows} rows where {flag} says 'not "
                                                    f"available', but also on {share:.0%} of the other rows, so "
                                                    f"this can be chance")
                if fills and (best is None or len(fills) > len(best[1])):
                    best = (level, fills, n_rows)
            if best is not None:
                for value, fill in best[1].items():
                    claims.setdefault(value, []).append((best[2], flag, best[0], fill))
        for value, options in claims.items():
            largest: int = max(option[0] for option in options)
            winners: list[tuple[int, str, int, float]] = [option for option in options if option[0] == largest]
            for _, flag, level, fill in winners:
                if fill != winners[0][3]:
                    continue                                 # one fill value per value
                indicator = found.get(flag)
                if indicator is None:
                    indicator = found[flag] = ValueIndicator(flag=flag, off_level=level, fills={}, named=[])
                indicator.fills[value] = fill
        paired: set[str] = {value for indicator in found.values() for value in indicator.fills}
        unresolved.extend(entry for value, entry in near_miss.items() if value not in paired)
    return [found[flag] for flag in binary if flag in found], unresolved


def _find_unseen_combinations(A: numpy.ndarray, index: dict[str, int], binary: list[str],
                              min_expected: float) -> list[UnseenCombination]:
    if len(binary) < 2:
        return []
    B: numpy.ndarray = A[:, [index[c] for c in binary]]
    n: int = B.shape[0]
    ones: numpy.ndarray = B.sum(axis=0)
    both: numpy.ndarray = B.T @ B
    counts: dict[tuple[int, int], numpy.ndarray] = {
        (1, 1): both,
        (1, 0): ones[:, None] - both,
        (0, 1): ones[None, :] - both,
        (0, 0): n - ones[:, None] - ones[None, :] + both,
    }
    share: dict[int, numpy.ndarray] = {1: ones / n, 0: 1.0 - ones / n}
    upper: numpy.ndarray = numpy.triu(numpy.ones((len(binary), len(binary)), dtype=bool), k=1)
    found: list[UnseenCombination] = []
    for (u, v), count in counts.items():
        expected: numpy.ndarray = n * numpy.outer(share[u], share[v])
        hit: numpy.ndarray = upper & (numpy.rint(count) == 0) & (expected >= min_expected)
        for i, j in zip(*numpy.nonzero(hit)):
            found.append(UnseenCombination(binary[i], u, binary[j], v, float(expected[i, j])))
    return found


# ---------------------------------------------------------------------------
# Population shifts: principal directions of the training covariates
# ---------------------------------------------------------------------------

@dataclass
class ShiftAxis:
    """A direction along which the test population is re-weighted. `train` / `test` are the normal
    scores of the rows (under the training distribution of the statistic); `positive` / `negative`
    describe what a positive / negative tilt emphasises."""
    name: str
    description: str
    positive: str
    negative: str
    train: numpy.ndarray
    test: numpy.ndarray


def population_axes(X_train: pandas.DataFrame, X_test: pandas.DataFrame,
                    config: RobustnessConfig | None = None) -> list[ShiftAxis]:
    """The population-shift axes, fitted on the standardised TRAINING covariates (every input):
    the leading `population_components` principal components (the sign of each is fixed so that its
    loadings sum to a positive value, which makes "positive" reproducible), and "extremes": the
    Mahalanobis distance from the centre within the leading components that explain
    `extremes_variance_share` of the variance -- a positive tilt emphasises atypical rows, a negative one
    typical rows."""
    config = config or RobustnessConfig()
    features: list[str] = list(X_train.columns)
    A: numpy.ndarray = X_train.to_numpy(dtype=float)
    T: numpy.ndarray = X_test[features].to_numpy(dtype=float)
    mean: numpy.ndarray = A.mean(axis=0)
    sd: numpy.ndarray = A.std(axis=0)
    sd[sd == 0.0] = 1.0
    Z: numpy.ndarray = (A - mean) / sd
    ZT: numpy.ndarray = (T - mean) / sd
    eigenvalues, eigenvectors = numpy.linalg.eigh(Z.T @ Z / Z.shape[0])
    order: numpy.ndarray = numpy.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    keep: int = int((eigenvalues > 1e-10 * eigenvalues[0]).sum())
    total: float = float(eigenvalues[:keep].sum())

    axes: list[ShiftAxis] = []
    for k in range(min(config.population_components, keep)):
        loading: numpy.ndarray = eigenvectors[:, k]
        if loading.sum() < 0.0:
            loading = -loading
        train_statistic: numpy.ndarray = Z @ loading
        transform: NormalScores = NormalScores(train_statistic)
        axes.append(ShiftAxis(
            name=f"PC{k + 1}",
            description=f"principal component {k + 1} of the standardised training inputs "
                        f"({eigenvalues[k] / total:.1%} of the variance)",
            positive=f"high PC{k + 1}", negative=f"low PC{k + 1}",
            train=transform(train_statistic), test=transform(ZT @ loading)))

    if keep > 0:
        leading: int = int(numpy.searchsorted(numpy.cumsum(eigenvalues[:keep]) / total,
                                              config.extremes_variance_share) + 1)
        leading = min(leading, keep)
        scale: numpy.ndarray = numpy.sqrt(eigenvalues[:leading])
        train_distance: numpy.ndarray = (((Z @ eigenvectors[:, :leading]) / scale) ** 2).sum(axis=1)
        test_distance: numpy.ndarray = (((ZT @ eigenvectors[:, :leading]) / scale) ** 2).sum(axis=1)
        transform = NormalScores(train_distance)
        axes.append(ShiftAxis(
            name="extremes",
            description=f"distance from the centre in the leading {leading} principal components "
                        f"({eigenvalues[:leading].sum() / total:.0%} of the variance)",
            positive="atypical rows", negative="typical rows",
            train=transform(train_distance), test=transform(test_distance)))
    return axes


# ---------------------------------------------------------------------------
# Dependence shifts: pairs of correlated inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DependencePair:
    a: str
    b: str
    correlation: float      # Pearson correlation on the training rows
    kind: str               # "continuous-continuous", "binary-continuous" or "binary-binary"


def dependence_pairs(X_train: pandas.DataFrame, X_test: pandas.DataFrame, schema: FeatureSchema,
                     config: RobustnessConfig | None = None) -> tuple[list[DependencePair], int]:
    """The pairs whose dependence is shifted, chosen by a fixed rule from the TRAINING correlations and
    the same for every method and seed: |r| in [pair_min_abs_correlation, pair_max_abs_correlation), not
    two levels of one one-hot group and not a value with its own availability flag, and -- for 0/1
    members -- at least `pair_min_train_count` training rows and `pair_min_test_count` test rows in every
    cell of the 2 x 2 table (every level of a single 0/1 member). The strongest `max_pairs` pairs by |r|
    are returned, with the number of eligible pairs."""
    config = config or RobustnessConfig()
    usable: set[str] = set(schema.binary) | set(schema.continuous)
    names: list[str] = [name for name in schema.features if name in usable]
    if len(names) < 2:
        return [], 0
    A: numpy.ndarray = X_train[names].to_numpy(dtype=float)
    T: numpy.ndarray = X_test[names].to_numpy(dtype=float)
    Z: numpy.ndarray = (A - A.mean(axis=0)) / A.std(axis=0)
    R: numpy.ndarray = Z.T @ Z / Z.shape[0]
    is_binary: numpy.ndarray = numpy.array([name in set(schema.binary) for name in names])
    family_of: dict[str, str] = {m: prefix for prefix, members in schema.dummy_families.items() for m in members}
    with_flag: set[frozenset[str]] = {frozenset((indicator.flag, column))
                                      for indicator in schema.value_indicators for column in indicator.fills}

    magnitude: numpy.ndarray = numpy.abs(R)
    candidates: numpy.ndarray = numpy.triu(
        (magnitude >= config.pair_min_abs_correlation) & (magnitude < config.pair_max_abs_correlation), k=1)
    pairs: list[DependencePair] = []
    for i, j in zip(*numpy.nonzero(candidates)):
        a, b = names[i], names[j]
        if a in family_of and family_of[a] == family_of.get(b):
            continue
        if frozenset((a, b)) in with_flag:
            continue
        if not (_enough_rows(A, i, j, is_binary, config.pair_min_train_count)
                and _enough_rows(T, i, j, is_binary, config.pair_min_test_count)):
            continue
        kind: str = {2: "binary-binary", 1: "binary-continuous", 0: "continuous-continuous"}[
            int(is_binary[i]) + int(is_binary[j])]
        pairs.append(DependencePair(a=a, b=b, correlation=float(R[i, j]), kind=kind))
    pairs.sort(key=lambda pair: (-abs(pair.correlation), pair.a, pair.b))
    return pairs[:config.max_pairs], len(pairs)


def _enough_rows(M: numpy.ndarray, i: int, j: int, is_binary: numpy.ndarray, minimum: int) -> bool:
    if is_binary[i] and is_binary[j]:
        cells: list[int] = [int(((M[:, i] == u) & (M[:, j] == v)).sum()) for u in (0, 1) for v in (0, 1)]
        return min(cells) >= minimum
    for k in (i, j):
        if is_binary[k] and min(int((M[:, k] == 0).sum()), int((M[:, k] == 1).sum())) < minimum:
            return False
    return True


class DependenceTilt:
    """A re-weighting that changes the dependence of two inputs and keeps their location and spread.

    Each input enters as a score: a 0/1 input standardised with its training mean and SD, a continuous
    input as its normal score under the training distribution, clipped to +-clip (so that a few extreme
    rows cannot take all the weight). The weights are the minimum-KL (maximum-entropy) re-weighting of
    the TRAINING rows of the form

        w  proportional to  exp(strength * s_a * s_b + lambda . t),   t = (s_a, s_b [, s_a^2, s_b^2]),

    where lambda is solved so that the weighted training rows keep the unweighted training means of the
    scores and, for continuous inputs, their second moments (entropy balancing). Only the product
    moment -- the dependence -- moves: strength > 0 raises it, strength < 0 lowers it. The same function
    (the training score transforms and the fitted lambda) is then applied to the test rows, so the test
    population is re-weighted by a rule learned from the training data only.

    Without the balancing terms a plain exp(strength * s_a * s_b) tilt mostly moves the means: it
    shifted the two means by up to 0.7-0.8 SD on College Scorecard and RadFusion pairs at a training
    ESS of 60%, i.e. it was largely a shift of the marginals rather than of the dependence.
    """

    def __init__(self, train_a: Sequence[float], train_b: Sequence[float], test_a: Sequence[float],
                 test_b: Sequence[float], a_binary: bool, b_binary: bool, clip: float = 2.5) -> None:
        train_scores: list[numpy.ndarray] = []
        test_scores: list[numpy.ndarray] = []
        for train_values, test_values, binary in ((train_a, test_a, a_binary), (train_b, test_b, b_binary)):
            x: numpy.ndarray = numpy.asarray(train_values, dtype=float)
            t: numpy.ndarray = numpy.asarray(test_values, dtype=float)
            if binary:
                centre, spread = x.mean(), x.std()
                train_scores.append((x - centre) / spread)
                test_scores.append((t - centre) / spread)
            else:
                transform: NormalScores = NormalScores(x)
                train_scores.append(numpy.clip(transform(x), -clip, clip))
                test_scores.append(numpy.clip(transform(t), -clip, clip))
        self._train_stats: numpy.ndarray = self._statistics(train_scores, (a_binary, b_binary))
        self._test_stats: numpy.ndarray = self._statistics(test_scores, (a_binary, b_binary))
        self._target: numpy.ndarray = self._train_stats.mean(axis=0)
        self._train_product: numpy.ndarray = train_scores[0] * train_scores[1]
        self._test_product: numpy.ndarray = test_scores[0] * test_scores[1]
        self._lambda: numpy.ndarray = numpy.zeros(self._train_stats.shape[1])
        self.balance_error: float = 0.0

    @staticmethod
    def _statistics(scores: list[numpy.ndarray], binary: tuple[bool, bool]) -> numpy.ndarray:
        columns: list[numpy.ndarray] = list(scores)
        columns += [scores[k] ** 2 for k in (0, 1) if not binary[k]]
        return numpy.column_stack(columns)

    def _objective(self, lam: numpy.ndarray, strength: float) -> float:
        exponent: numpy.ndarray = self._train_stats @ lam + strength * self._train_product
        top: float = float(exponent.max())
        return float(numpy.log(numpy.mean(numpy.exp(exponent - top))) + top - self._target @ lam)

    def _solve(self, strength: float) -> numpy.ndarray:
        """Solve the (convex) dual by Newton's method, warm-started at the last solution; if that does
        not balance the moments, start again from lambda = 0 and keep the better solution."""
        best_lambda: numpy.ndarray = self._lambda
        best_error: float = float("inf")
        for start in (self._lambda, numpy.zeros_like(self._lambda)):
            lam: numpy.ndarray = self._newton(start.copy(), strength)
            error: float = float(numpy.max(numpy.abs(self._moments(lam, strength)[2])))
            if error < best_error:
                best_lambda, best_error = lam, error
            if error < 1e-8:
                break
        self._lambda = best_lambda
        # largest deviation of a balanced training moment from its target (0 = exactly balanced)
        self.balance_error = best_error
        return best_lambda

    def _newton(self, lam: numpy.ndarray, strength: float) -> numpy.ndarray:
        S: numpy.ndarray = self._train_stats
        value: float = self._objective(lam, strength)
        for _ in range(200):
            w, mean, gradient = self._moments(lam, strength)
            if numpy.max(numpy.abs(gradient)) < 1e-10:
                break
            centred: numpy.ndarray = S - mean
            hessian: numpy.ndarray = centred.T @ (centred * w[:, None])
            step: numpy.ndarray = numpy.linalg.lstsq(hessian, gradient, rcond=None)[0]
            decrease: float = float(gradient @ step)
            if decrease < 1e-18:            # the Newton decrement is at the rounding level: at the optimum
                break
            t: float = 1.0
            while True:
                candidate: numpy.ndarray = lam - t * step
                candidate_value: float = self._objective(candidate, strength)
                if candidate_value <= value - 1e-4 * t * decrease or t < 1e-12:
                    break
                t *= 0.5
            if candidate_value > value:     # no further progress: at the numerical optimum
                break
            lam, value = candidate, candidate_value
        return lam

    def _moments(self, lam: numpy.ndarray, strength: float) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """(normalised training weights, weighted means of the statistics, their deviation from the targets)."""
        exponent: numpy.ndarray = self._train_stats @ lam + strength * self._train_product
        w: numpy.ndarray = numpy.exp(exponent - exponent.max())
        w /= w.sum()
        mean: numpy.ndarray = w @ self._train_stats
        return w, mean, mean - self._target

    def train_ess(self, strength: float) -> float:
        """Effective sample fraction of the training weights at `strength`, or NaN when the moments
        cannot be balanced there (the shift cannot be realised with the marginals held fixed); the
        function `calibrate_strengths` expects."""
        w_train, _ = self.weights(strength)
        return effective_sample_fraction(w_train) if self.balance_error <= BALANCE_TOLERANCE else float("nan")

    def score_correlation(self, strength: float) -> float:
        """Correlation of the two scores on the training rows re-weighted at `strength` -- the dependence
        the tilt moves -- or NaN when the moments cannot be balanced there. With the means (and second
        moments) of the scores held at their clean values it is a linear function of the product moment,
        so it increases with the strength."""
        w_train, _ = self.weights(strength)
        if self.balance_error > BALANCE_TOLERANCE:
            return float("nan")
        a: numpy.ndarray = self._train_stats[:, 0]
        b: numpy.ndarray = self._train_stats[:, 1]
        return weighted_correlation(a, b, w_train)

    def weights(self, strength: float) -> tuple[numpy.ndarray, numpy.ndarray]:
        """(training weights, test weights), each scaled to mean 1."""
        lam: numpy.ndarray = self._solve(strength)
        train: numpy.ndarray = self._train_stats @ lam + strength * self._train_product
        test: numpy.ndarray = self._test_stats @ lam + strength * self._test_product
        w_train: numpy.ndarray = numpy.exp(train - train.max())
        w_test: numpy.ndarray = numpy.exp(test - test.max())
        return w_train / w_train.mean(), w_test / w_test.mean()


def calibrate_correlation(tilt: DependenceTilt, targets: Sequence[float],
                          max_strength: float = MAX_TILT_STRENGTH,
                          initial_strength: float = 0.05) -> list[tuple[float, bool]]:
    """(strength, reached) for every target training score correlation of `tilt`, in the given order --
    each target further from the clean correlation than the one before, all on one side of it. The
    strength is searched from 0 towards the target (doubling until the correlation passes it), then
    refined by root finding, starting each target from the previous one's strength.

    `reached` is False when the target cannot be realised: the moments cannot be balanced before it, or
    even `max_strength` stays short of it; that target -- and every further one -- gets NaN."""
    clean: float = tilt.score_correlation(0.0)
    if not targets:
        return []
    direction: float = 1.0 if targets[0] > clean else -1.0
    steps: list[float] = [direction * (target - clean) for target in targets]
    if any(step <= 0.0 for step in steps) or steps != sorted(steps):
        raise ValueError(f"the targets must move away from the clean correlation {clean:.4f} in one direction, "
                         f"got {list(targets)}")
    results: list[tuple[float, bool]] = []
    low: float = 0.0
    for target in targets:
        def progress(s: float, goal: float = target) -> float:
            # < 0 until the correlation has moved past the goal; NaN where the tilt cannot be balanced
            value: float = tilt.score_correlation(direction * s)
            return float("nan") if numpy.isnan(value) else direction * (value - goal)

        high: float = max(low, initial_strength)
        value_high: float = progress(high)
        while not numpy.isnan(value_high) and value_high < 0.0 and high < max_strength:
            low, high = high, min(2.0 * high, max_strength)
            value_high = progress(high)
        try:
            if numpy.isnan(value_high) or value_high < 0.0:
                raise ValueError("the target is not reached")
            strength: float = float(brentq(progress, low, high, xtol=1e-9))
        except (ValueError, RuntimeError):
            results.extend([(float("nan"), False)] * (len(targets) - len(results)))
            break
        results.append((direction * strength, True))
        low = strength
    return results


# ---------------------------------------------------------------------------
# Corrupted test sets
# ---------------------------------------------------------------------------

@dataclass
class _Unit:
    """Columns that are corrupted together: one stand-alone 0/1 input, a one-hot group, or an
    availability flag with its values."""
    columns: numpy.ndarray
    cumulative: numpy.ndarray | None = None     # one-hot group: cumulative level frequencies
    prevalence: float = 0.0                      # stand-alone 0/1 input: training prevalence
    flag: int = -1                               # availability flag: its column ...
    off_level: int = 0                           # ... the level meaning "not available" ...
    value_columns: numpy.ndarray = field(default_factory=lambda: numpy.zeros(0, dtype=int))
    fills: numpy.ndarray = field(default_factory=lambda: numpy.zeros(0))   # ... and the fill values


class CorruptionBank:
    """A fixed bank of corrupted versions of the test set.

    For every family and every repetition the random draws are made ONCE and reused at every severity
    level (common random numbers): the Gaussian noise at level 0.4 is exactly twice the noise at level
    0.2, and the cells corrupted at a level are a subset of those corrupted at any higher level. Every
    model is scored on the very same corrupted data, so method differences are never confounded with
    different draws, and the draws do not depend on the GA seed. Each family has its own random stream,
    so adding or changing one family never changes the draws of another.

    Families (level in [0, 1]; level 0 is the clean test set):
      gaussian_noise  x + level * training SD * N(0, 1) on every continuous input; values that are not
                      available (their flag says so) keep their fill value.
      binary_redraw   every stand-alone 0/1 input -- and every one-hot group, as ONE categorical input --
                      is re-drawn from its training distribution with probability `level`.
      under_recording every recorded 1 of a stand-alone 0/1 input with training prevalence below the
                      threshold is lost (set to 0) with probability `level`.
      value_masking   every available value with an availability flag becomes "not available" (the flag
                      set to its off level, its values to their fill values) with probability `level`, each
                      flag independently: at level 1 every value is withheld.
    Availability flags are never re-drawn or under-recorded, so a flag and its values always agree. The
    structural rules (one-hot groups, availability pairs, 0/1 values) are checked after every corruption
    (a violation raises RuntimeError). A combination of 0/1 values that never occurs in training
    (`FeatureSchema.unseen_combinations`) may be created -- it is an association, not an impossibility --
    and is counted in the diagnostics of every corrupted test set.
    """

    def __init__(self, schema: FeatureSchema, X_train: pandas.DataFrame, X_test: pandas.DataFrame,
                 repetitions: int, seed: int) -> None:
        self._features: list[str] = list(schema.features)
        self._index: pandas.Index = X_test.index
        self._clean: numpy.ndarray = X_test[self._features].to_numpy(dtype=float).copy()
        self._repetitions: int = int(repetitions)
        column: dict[str, int] = {name: j for j, name in enumerate(self._features)}
        n_rows: int = self._clean.shape[0]

        # -- measurement noise
        self._noise_columns: numpy.ndarray = numpy.array([column[c] for c in schema.continuous], dtype=int)
        self._noise_scale: numpy.ndarray = (X_train[schema.continuous].std().to_numpy(dtype=float)
                                            if schema.continuous else numpy.zeros(0))
        self._unavailable: numpy.ndarray = numpy.zeros_like(self._clean, dtype=bool)
        for indicator in schema.value_indicators:
            off_rows: numpy.ndarray = self._clean[:, column[indicator.flag]] == indicator.off_level
            for name in indicator.fills:
                self._unavailable[off_rows, column[name]] = True

        # -- recording noise: one-hot groups first, then stand-alone inputs
        self._redraw_units: list[_Unit] = []
        self._exhaustive_groups: list[numpy.ndarray] = []
        self._groups: list[numpy.ndarray] = []
        for prefix, members in schema.dummy_families.items():
            frequencies: numpy.ndarray = numpy.asarray(schema.family_frequencies[prefix], dtype=float)
            columns: numpy.ndarray = numpy.array([column[m] for m in members])
            self._groups.append(columns)
            if frequencies[-1] <= 0.0:                  # exhaustive group: never "none of them"
                frequencies = frequencies[:-1]
                self._exhaustive_groups.append(columns)
            cumulative: numpy.ndarray = numpy.cumsum(frequencies / frequencies.sum())[:-1]
            self._redraw_units.append(_Unit(columns=columns, cumulative=cumulative))
        for name in schema.redraw_columns:
            self._redraw_units.append(_Unit(columns=numpy.array([column[name]]),
                                            prevalence=schema.training_prevalence[name]))

        # -- under-recording
        self._under_units: list[_Unit] = [_Unit(columns=numpy.array([column[name]]))
                                          for name in schema.under_recording_columns]

        # -- value masking: one unit per availability flag, with its values
        self._mask_units: list[_Unit] = []
        for indicator in schema.value_indicators:
            values: list[str] = list(indicator.fills)
            value_columns: numpy.ndarray = numpy.array([column[v] for v in values], dtype=int)
            self._mask_units.append(_Unit(
                columns=numpy.r_[column[indicator.flag], value_columns].astype(int),
                flag=column[indicator.flag], off_level=int(indicator.off_level),
                value_columns=value_columns,
                fills=numpy.array([indicator.fills[v] for v in values], dtype=float)))

        # -- validity checks: 0/1 inputs, and the availability rule on the clean rows (a test row that
        # already breaks it, e.g. imputed differently, is left as it is and not counted as a violation)
        self._binary_columns: numpy.ndarray = numpy.array([column[c] for c in schema.binary], dtype=int)
        self._clean_violations: list[numpy.ndarray] = [self._availability_violations(self._clean, unit)
                                                       for unit in self._mask_units]
        self.clean_availability_violations: int = int(sum(v.sum() for v in self._clean_violations))

        # -- unseen combinations (diagnostic), in column indices, and whether a clean row already has them
        self._unseen: numpy.ndarray = numpy.array(
            [[column[f.a], f.a_value, column[f.b], f.b_value] for f in schema.unseen_combinations], dtype=int
        ).reshape(-1, 4)
        if self._unseen.size:
            fa, fu, fb, fv = self._unseen.T
            self._clean_has: numpy.ndarray = (self._clean[:, fa] == fu) & (self._clean[:, fb] == fv)
        else:
            self._clean_has = numpy.zeros((n_rows, 0), dtype=bool)

        # -- one independent random stream per family and repetition
        root: numpy.random.SeedSequence = numpy.random.SeedSequence(int(seed))
        self._streams: dict[str, list[numpy.random.SeedSequence]] = {
            family: sequence.spawn(self._repetitions)
            for family, sequence in zip(CORRUPTION_FAMILIES, root.spawn(len(CORRUPTION_FAMILIES)))}
        self._draws: dict[tuple[str, int], dict[str, numpy.ndarray]] = {}

    # ---- what can be corrupted -----------------------------------------------------------------------
    def applicable_families(self) -> list[str]:
        available: dict[str, bool] = {
            "gaussian_noise": self._noise_columns.size > 0,
            "binary_redraw": bool(self._redraw_units),
            "under_recording": bool(self._under_units),
            "value_masking": bool(self._mask_units),
        }
        return [family for family in CORRUPTION_FAMILIES if available[family]]

    def describe(self) -> dict[str, int]:
        return {
            "gaussian_noise": int(self._noise_columns.size),
            "binary_redraw": len(self._redraw_units),
            "under_recording": len(self._under_units),
            "value_masking": len(self._mask_units),
        }

    # ---- random draws (made once per family and repetition) --------------------------------------------
    def _draw(self, family: str, repetition: int) -> dict[str, numpy.ndarray]:
        if family not in CORRUPTION_FAMILIES:
            raise ValueError(f"unknown corruption family {family!r}; known: {CORRUPTION_FAMILIES}")
        key: tuple[str, int] = (family, repetition)
        if key not in self._draws:
            if not 0 <= repetition < self._repetitions:
                raise ValueError(f"repetition must lie in [0, {self._repetitions}), got {repetition}")
            rng: numpy.random.Generator = numpy.random.default_rng(self._streams[family][repetition])
            n_rows: int = self._clean.shape[0]
            if family == "gaussian_noise":
                self._draws[key] = {"z": rng.standard_normal((n_rows, self._noise_columns.size))}
            elif family == "binary_redraw":
                self._draws[key] = {"select": rng.random((n_rows, len(self._redraw_units))),
                                    "value": rng.random((n_rows, len(self._redraw_units)))}
            elif family == "under_recording":
                self._draws[key] = {"select": rng.random((n_rows, len(self._under_units)))}
            else:   # value_masking
                self._draws[key] = {"select": rng.random((n_rows, len(self._mask_units)))}
        return self._draws[key]

    # ---- corrupted data ---------------------------------------------------------------------------------
    def corrupt(self, family: str, level: float, repetition: int) -> tuple[pandas.DataFrame, dict[str, int]]:
        """The test set under `family` at `level` in the given repetition, and counts of what changed:
        `eligible` cells (or units), `selected` by the random draw, `changed` in the end, and the
        combinations of 0/1 values unseen in training that the corruption created
        (`unseen_combinations`, in `rows_with_unseen_combination` rows)."""
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"level must lie in [0, 1], got {level}")
        draws: dict[str, numpy.ndarray] = self._draw(family, repetition)
        X: numpy.ndarray = self._clean.copy()
        if family == "gaussian_noise":
            counts = self._gaussian(X, level, draws)
        elif family == "binary_redraw":
            counts = self._redraw(X, level, draws)
        elif family == "under_recording":
            counts = self._under_record(X, level, draws)
        else:
            counts = self._mask(X, level, draws)
        self._check(X, family, level)
        counts.update(self._unseen_created(X))
        return pandas.DataFrame(X, columns=self._features, index=self._index), counts

    def _gaussian(self, X: numpy.ndarray, level: float, draws: dict[str, numpy.ndarray]) -> dict[str, int]:
        columns: numpy.ndarray = self._noise_columns
        available: numpy.ndarray = ~self._unavailable[:, columns]
        noise: numpy.ndarray = level * draws["z"] * self._noise_scale[None, :]
        X[:, columns] += numpy.where(available, noise, 0.0)
        changed: int = int((available & (noise != 0.0)).sum())
        return {"eligible": int(available.sum()), "selected": changed, "changed": changed}

    def _redraw(self, X: numpy.ndarray, level: float, draws: dict[str, numpy.ndarray]) -> dict[str, int]:
        select: numpy.ndarray = draws["select"] < level
        changed: int = 0
        for k, unit in enumerate(self._redraw_units):
            rows: numpy.ndarray = numpy.flatnonzero(select[:, k])
            if rows.size == 0:
                continue
            u: numpy.ndarray = draws["value"][rows, k]
            old: numpy.ndarray = X[numpy.ix_(rows, unit.columns)]
            if unit.cumulative is not None:
                level_index: numpy.ndarray = numpy.searchsorted(unit.cumulative, u, side="right")
                new: numpy.ndarray = numpy.zeros((rows.size, unit.columns.size))
                chosen: numpy.ndarray = level_index < unit.columns.size      # else: "none of them"
                new[numpy.flatnonzero(chosen), level_index[chosen]] = 1.0
            else:
                new = (u < unit.prevalence).astype(float)[:, None]
            X[numpy.ix_(rows, unit.columns)] = new
            changed += int((new != old).any(axis=1).sum())
        return {"eligible": int(select.size), "selected": int(select.sum()), "changed": changed}

    def _under_record(self, X: numpy.ndarray, level: float, draws: dict[str, numpy.ndarray]) -> dict[str, int]:
        columns: numpy.ndarray = numpy.array([unit.columns[0] for unit in self._under_units], dtype=int)
        recorded: numpy.ndarray = X[:, columns] == 1.0
        lost: numpy.ndarray = recorded & (draws["select"] < level)
        X[:, columns] = numpy.where(lost, 0.0, X[:, columns])
        return {"eligible": int(recorded.sum()), "selected": int(lost.sum()), "changed": int(lost.sum())}

    def _mask(self, X: numpy.ndarray, level: float, draws: dict[str, numpy.ndarray]) -> dict[str, int]:
        flags: numpy.ndarray = numpy.array([unit.flag for unit in self._mask_units], dtype=int)
        off: numpy.ndarray = numpy.array([unit.off_level for unit in self._mask_units], dtype=float)
        available: numpy.ndarray = X[:, flags] != off[None, :]
        select: numpy.ndarray = available & (draws["select"] < level)
        for k, unit in enumerate(self._mask_units):
            rows: numpy.ndarray = numpy.flatnonzero(select[:, k])
            if rows.size:
                X[rows, unit.flag] = unit.off_level
                X[numpy.ix_(rows, unit.value_columns)] = unit.fills[None, :]
        return {"eligible": int(available.sum()), "selected": int(select.sum()), "changed": int(select.sum())}

    # ---- validity and diagnostics ---------------------------------------------------------------------------
    @staticmethod
    def _availability_violations(X: numpy.ndarray, unit: _Unit) -> numpy.ndarray:
        """(rows x values) cells whose flag says "not available" but which do not hold the fill value."""
        off_rows: numpy.ndarray = X[:, unit.flag] == unit.off_level
        return off_rows[:, None] & (X[:, unit.value_columns] != unit.fills[None, :])

    def _check(self, X: numpy.ndarray, family: str, level: float) -> None:
        """The structural rules: 0/1 inputs stay 0/1; a one-hot group keeps at most one level (exactly one
        if it is exhaustive and the clean row had one); a value whose flag says "not available" holds its
        fill value (unless the clean test row already broke that rule and the cell is unchanged)."""
        problems: list[str] = []
        binary: numpy.ndarray = X[:, self._binary_columns]
        if not numpy.isin(binary, (0.0, 1.0)).all():
            problems.append("a 0/1 input holds another value")
        for columns in self._groups:
            if (X[:, columns].sum(axis=1) > 1).any():
                problems.append("a one-hot group has two levels in a row")
        for columns in self._exhaustive_groups:
            if ((X[:, columns].sum(axis=1) != 1) & (self._clean[:, columns].sum(axis=1) == 1)).any():
                problems.append("an exhaustive one-hot group lost its level")
        for unit, clean_violation in zip(self._mask_units, self._clean_violations):
            violation: numpy.ndarray = self._availability_violations(X, unit)
            unchanged: numpy.ndarray = X[:, unit.value_columns] == self._clean[:, unit.value_columns]
            if (violation & ~(clean_violation & unchanged)).any():
                problems.append(f"a value of the availability flag {self._features[unit.flag]!r} is not at "
                                f"its fill value")
        if problems:
            raise RuntimeError(f"{family} at level {level} broke the structural rules of the data: "
                               + "; ".join(dict.fromkeys(problems)))

    def _unseen_created(self, X: numpy.ndarray) -> dict[str, int]:
        if not self._unseen.size:
            return {"unseen_combinations": 0, "rows_with_unseen_combination": 0}
        fa, fu, fb, fv = self._unseen.T
        created: numpy.ndarray = (X[:, fa] == fu) & (X[:, fb] == fv) & ~self._clean_has
        return {"unseen_combinations": int(created.sum()),
                "rows_with_unseen_combination": int(created.any(axis=1).sum())}
