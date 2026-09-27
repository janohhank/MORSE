from __future__ import annotations

from typing import Any, Union
import numpy
import pandas
import matplotlib
matplotlib.use("Agg")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from training_utils import select_pareto_individual

# ---------------------------------------------------------------------------
# Sign consistency score calculation
# ---------------------------------------------------------------------------

def compute_marginal_correlations(
    X: Union[numpy.ndarray, pandas.DataFrame],
    y: Union[numpy.ndarray, pandas.Series],
) -> numpy.ndarray:
    """Per-feature marginal correlation with the binary target `y`.

    The Pearson correlation of every column with `y`. For a binary (0/1)
    column this IS the phi (Matthews) coefficient and for any other column it
    IS the point-biserial correlation -- the two measures the method
    prescribes -- so one formula covers both. Constant columns get 0.

    The value does not depend on the location or the scale of a column, and it
    must not: the GA calls this function on STANDARDISED fold data
    (MultiObjectiveTraining), the evaluation on the raw training data. An
    earlier version computed the Matthews coefficient of `feat.astype(int)` for
    two-valued columns. On standardised data that cast turns a 0/1 column whose
    prevalence exceeds 0.5 into {-k, 0} -- the 1s become 0 -- and sklearn's
    multiclass Matthews coefficient then has the OPPOSITE sign (and half the
    magnitude when the 1s cast to 2 or more). The GA's sign-consistency
    objective therefore used the reversed reference direction for exactly those
    inputs (7 inputs of College Scorecard, 1 of arrhythmia, 15 of RadFusion in
    the 2026-09-25 runs), while the evaluation on raw data was right.

    Returns a numpy array of length `X.shape[1]`, aligned with the column
    order of `X`.
    """
    X_arr: numpy.ndarray = (X.to_numpy(dtype=float) if hasattr(X, "to_numpy")
                            else numpy.asarray(X, dtype=float))
    y_arr: numpy.ndarray = numpy.asarray(y, dtype=float)

    out: numpy.ndarray = numpy.zeros(X_arr.shape[1], dtype=float)
    # Constant columns (and a constant target) have no correlation. Test it
    # exactly with ptp: centring a constant column can leave rounding noise.
    varying: numpy.ndarray = numpy.ptp(X_arr, axis=0) > 0
    if numpy.ptp(y_arr) == 0 or not varying.any():
        return out
    X_centred: numpy.ndarray = X_arr[:, varying] - X_arr[:, varying].mean(axis=0)
    y_centred: numpy.ndarray = y_arr - y_arr.mean()
    out[varying] = (y_centred @ X_centred) / numpy.sqrt(
        (X_centred ** 2).sum(axis=0) * (y_centred ** 2).sum())
    return out


def compute_model_sign_consistency(
        model_pkg: dict[str, Any],
        marginal_corr: pandas.Series) -> dict[str, Any]:
    """Sign consistency of a FINAL (full-training-set refit) model.

    The fraction of the model's selected features whose fitted logistic
    regression coefficient has the same sign as that feature's marginal
    correlation with the target (Matthews / point-biserial, see
    `compute_marginal_correlations`). It is the very quantity MORSE optimises,
    with the same strict rule as the GA fitness in
    `MultiObjectiveTraining._evaluate_multi`: a feature counts as INCONSISTENT
    if `marginal_corr * coefficient` is negative or numerically zero, and as
    consistent otherwise. Two differences from the fitness value, both on
    purpose: it is measured on the deployed model, i.e. the coefficients of the
    refit on the WHOLE training set and the marginal correlations of the whole
    training set (the GA fitness averages three fold-wise estimates), and it can
    be computed for every method -- including the baselines, which never see the
    quantity during selection.

    Read it together with the number of selected features: a model with a single
    feature is trivially 100% consistent (with one predictor the coefficient
    always has the sign of the marginal correlation), so the measure is only
    informative between models of comparable size.

    Parameters
    ----------
    model_pkg
        A package from `build_model_package` ("model", "scaler", "features").
    marginal_corr
        Marginal correlation of EVERY candidate feature with the target on the
        training set, indexed by feature name -- compute it once with
        `pandas.Series(compute_marginal_correlations(X_train, y_train),
        index=X_train.columns)` and reuse it for all models.

    Returns
    -------
    dict with `n_features`, `n_consistent`, `n_inconsistent` and
    `sign_consistency` (= n_consistent / n_features).
    """
    corr: numpy.ndarray = marginal_corr.loc[model_pkg["features"]].to_numpy(dtype=float)
    coef: numpy.ndarray = model_pkg["model"].coef_[0]

    check: numpy.ndarray = corr * coef
    inconsistent: numpy.ndarray = (check < 0) | numpy.isclose(check, 0.0, atol=1e-12)

    n_features: int = int(len(check))
    n_inconsistent: int = int(inconsistent.sum())
    return {
        "n_features":       n_features,
        "n_consistent":     n_features - n_inconsistent,
        "n_inconsistent":   n_inconsistent,
        "sign_consistency": 1.0 - n_inconsistent / n_features,
    }


# ---------------------------------------------------------------------------
# Column type detection
# ---------------------------------------------------------------------------

def get_continuous_columns(df: pandas.DataFrame) -> list[str]:
    """Return columns with more than 2 unique numeric values (continuous features)."""
    return [col for col in df.select_dtypes(include=[numpy.number]).columns
            if df[col].nunique() > 2]


def get_dummy_columns(df: pandas.DataFrame) -> list[str]:
    """Return binary columns whose values are a subset of {0, 1}."""
    dummy_cols: list[str] = []
    for col in df.columns:
        unique_vals: set[Any] = set(df[col].dropna().unique())
        if len(unique_vals) <= 2 and unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
            dummy_cols.append(col)
    return dummy_cols


# ---------------------------------------------------------------------------
# Noise injection and covariate shift -- LEGACY
#
# The pipeline evaluates robustness with the robustness suite (robustness_evaluation.py,
# robustness_utils.py; docs/robustness.md). The stress tests below are the ones runs made before the
# suite used. They drive the legacy stress grid (legacy_stress_evaluation.py), which the notebook still
# runs as an optional block, so that results stay comparable with earlier runs.
# ---------------------------------------------------------------------------

def apply_proportional_noise(
        X_test: pandas.DataFrame,
        train_std: pandas.Series,
        noise_fraction: float,
        continuous_cols: list[str]) -> pandas.DataFrame:
    """
    Adds zero-mean Gaussian measurement noise to the continuous variables,
    proportional to each column's training-set standard deviation.

    noise_fraction: 0.0-1.0 (noise standard deviation as a fraction of the
                    training standard deviation of that column)

    This function deliberately models NOISE only. An earlier version also
    translated every continuous column by a constant ("mean shift"); that was
    removed because, for a logistic-regression score and a rank-based metric
    (ROC-AUC / PR-AUC), adding the same constant to a feature for every test
    row shifts every logit by one and the same amount, which cannot change the
    ranking of the patients -- the shift axis was verified to have exactly zero
    effect on the scores. Covariate shift is modelled by re-weighting the test
    population instead: see `covariate_shift_weights`.
    """
    if numpy.isclose(noise_fraction, 0.0, atol=1e-09):
        return X_test.copy()

    X_out: pandas.DataFrame = X_test.copy()

    for col in continuous_cols:
        if col not in X_out.columns or col not in train_std.index:
            continue
        std_val: float = train_std[col]

        noise: numpy.ndarray = numpy.random.normal(
            loc=0.0, scale=noise_fraction * std_val, size=len(X_out))
        X_out[col] += noise

    return X_out


def apply_dummy_noise(
        X_test: pandas.DataFrame,
        noise_fraction: float,
        dummy_cols: list[str],
        train_prevalence: pandas.Series | None = None) -> pandas.DataFrame:
    """
    Re-draw noise on dummy (binary) variables, from each column's training prevalence.

    Every dummy cell is, independently, **re-drawn with probability
    `noise_fraction`** from a Bernoulli distribution with the column's own
    TRAINING prevalence `pi_j = train_prevalence[j]`; the remaining cells keep
    their value. The re-draw does not look at the cell's value or at the label.

    * 0.0  -> original data, no noise
    * 0.5  -> half of the cells are re-drawn from their column's marginal
    * 1.0  -> every column is independent of the truth and follows its training
              prevalence (no information left; the information falls monotonically
              with the fraction -- fractions above 0.5 cannot invert the signal)

    WHY NOT A FAIR COIN (the earlier implementation re-drew with P(1) = 0.5)
    A fair coin is far from the marginal of a rare flag, so it does not add
    "a little noise", it floods the column with false positives. With a 4%
    prevalence flag and 10% of the cells re-drawn, about 4.8% of all patients
    turn into false positives while only 3.8% are true positives that survived:
    a recorded "1" is right only 44% of the time (94% for a 45%-prevalence flag
    at the same setting), and the column's prevalence more than doubles.
    Measured on the 273 binary RadFusion columns (57% of them below 5%
    prevalence) at 10% re-drawn cells, the noise variance was 2.8 times the
    column's own signal variance for the rare columns (correlation with the
    clean column 0.51) but only 0.4 times for the dense ones (correlation 0.84).
    The corruption therefore hurt a model according to how SPARSE its features
    happen to be, not according to how much it relies on them, and it changed
    every column's prevalence -- a prevalence shift on top of the noise.

    PROPERTIES OF THE RE-DRAW (p = noise_fraction, pi = the column prevalence)
      * prevalence is preserved: E[X'] = pi whenever the test prevalence equals
        the training prevalence (a small, documented difference otherwise);
      * every column keeps the SAME correlation 1 - p with its clean version
        and gets the same noise-to-signal variance ratio 2p, whatever its
        prevalence (verified on the RadFusion columns: 0.90 and 0.20 at p = 0.1
        for the rare, medium and dense columns alike). It is the binary
        counterpart of `apply_proportional_noise`, which scales its Gaussian
        noise by every column's standard deviation;
      * a recorded 1 is right with probability 1 - p*(1 - pi), i.e. about
        1 - p for rare flags, and false positives affect only about p*pi of
        the patients (0.4% instead of 4.8% in the example above);
      * the information a column keeps falls monotonically in p (its correlation
        with the clean column is 1 - p), and the scheme needs no cap for very
        dense columns (a scheme that keeps sensitivity = precision = 1 - p by
        flipping 0 -> 1 with probability p*pi/(1 - pi) would exceed probability 1
        for every column with pi > 1/(1 + p), e.g. the seven RadFusion lab flags
        with 91-94% prevalence at p = 0.1). A frozen model's measured score on
        one finite test set need not fall monotonically in every realisation,
        though.

    LIMITATION: every column is re-drawn independently, so the noise breaks
    the association BETWEEN columns for the re-drawn cells (a real recording
    error would often be correlated across related codes).

    Uses the global `numpy.random` state (seed it with `set_seed`), like
    `apply_proportional_noise`. The dtype of every column (bool / int / float)
    is preserved.

    X_test:           the (clean) test features.
    noise_fraction:   p in [0, 1], the share of dummy cells that are re-drawn.
    dummy_cols:       the binary columns to corrupt (see `get_dummy_columns`).
    train_prevalence: per-column mean of the TRAINING data, e.g.
                      `X_train[dummy_cols].mean()`. Estimated on the training
                      data only, so the noise never looks at test data. Optional
                      so that the earlier three-argument call keeps working: if
                      omitted, the prevalence of `X_test` itself is used (the
                      noise then preserves the TEST prevalence in expectation).
    """
    if not 0.0 <= noise_fraction <= 1.0:
        raise ValueError(f"noise_fraction must be in [0, 1], got {noise_fraction}")

    if numpy.isclose(noise_fraction, 0.0, atol=1e-09):
        return X_test.copy()

    X_out: pandas.DataFrame = X_test.copy()
    if train_prevalence is None:
        train_prevalence = X_out[[col for col in dummy_cols if col in X_out.columns]].mean()
    cols: list[str] = [
        col for col in dummy_cols if col in X_out.columns and col in train_prevalence.index]
    if not cols:
        return X_out

    clean: numpy.ndarray = X_out[cols].to_numpy(dtype=float)
    prevalence: numpy.ndarray = train_prevalence[cols].to_numpy(dtype=float)

    redraw: numpy.ndarray = numpy.random.random_sample(clean.shape) < noise_fraction
    redrawn: numpy.ndarray = (
        numpy.random.random_sample(clean.shape) < prevalence).astype(float)
    noisy: numpy.ndarray = numpy.where(redraw, redrawn, clean)

    for position, col in enumerate(cols):
        X_out[col] = noisy[:, position].astype(X_out[col].dtype)

    return X_out


# ---------------------------------------------------------------------------
# Covariate shift (re-weighted test population) -- LEGACY, see above
# ---------------------------------------------------------------------------

def fit_covariate_shift_axis(X_train: pandas.DataFrame) -> dict[str, Any]:
    """Fit the axis along which the test population is shifted: the first
    principal component (PC1) of the standardised TRAINING covariates (all
    features, continuous and binary alike).

    PC1 is the dominant direction of variation of the patient population (on
    RadFusion, for example, an overall medication / hospitalisation burden), so
    shifting the population along it is a model-agnostic, dataset-agnostic
    notion of "a different case mix". The axis is fitted on the training data
    only, and the very same axis is used for every model, seed and noise level.

    Returns a dict with the feature order, the training mean / std used for the
    standardisation (zero-variance columns get std=1 and therefore contribute
    nothing), the unit-length PC1 `loading`, and `score_std`, the training
    standard deviation of the projection, so that the PC1 score of any patient
    can be expressed in training-SD units. The sign of a principal axis is
    arbitrary; it is fixed here so that the loadings sum to a positive value,
    which makes "positive shift strength" reproducible across runs and machines.
    """
    features: list[str] = list(X_train.columns)
    X: numpy.ndarray = X_train.to_numpy(dtype=float)

    mean: numpy.ndarray = X.mean(axis=0)
    std: numpy.ndarray = X.std(axis=0)
    std[std == 0.0] = 1.0
    Z: numpy.ndarray = (X - mean) / std

    covariance: numpy.ndarray = (Z.T @ Z) / Z.shape[0]
    _, eigenvectors = numpy.linalg.eigh(covariance)   # eigenvalues ascending
    loading: numpy.ndarray = eigenvectors[:, -1]
    if loading.sum() < 0.0:
        loading = -loading

    return {
        "features":  features,
        "mean":      mean,
        "std":       std,
        "loading":   loading,
        "score_std": float((Z @ loading).std()),
    }


def covariate_shift_weights(
        shift_axis: dict[str, Any],
        X_test: pandas.DataFrame,
        y_test: Union[numpy.ndarray, pandas.Series],
        strength: float,
        max_abs_score: float = 2.0) -> numpy.ndarray:
    """Importance weights that shift the TEST population along the PC1 axis by
    `strength` standard deviations (a covariate shift). Pass them as
    `sample_weight` to the metric (see `score_predictions`).

    WHY THE FEATURE VALUES ARE NOT SHIFTED
    A constant translation of features (x -> x + c for every test row) cannot
    change the ranking produced by a logistic-regression score -- every logit
    moves by the same constant -- so ROC-AUC / PR-AUC are exactly invariant to
    it (verified on real runs: bit-identical scores at zero noise). Editing the
    feature values of a real patient would also silently change what that
    patient's (unchanged) label means. A genuine covariate shift changes which
    patients are represented, i.e. P(x), while every patient keeps their own
    (x, y) pair, i.e. P(y | x) stays intact. Re-weighting does exactly that.

    DEFINITION
        z_i = PC1 score of test patient i in training-SD units, clipped to
              [-max_abs_score, +max_abs_score]
        w_i is proportional to exp(strength * z_i)
    The weights are then normalised WITHIN each class so that the total weight
    of the positives and of the negatives equals their original counts.
    Consequences:
      * the outcome prevalence is preserved exactly, so PR-AUC is not
        contaminated by a change of the label prior;
      * both classes are tilted by the same factor, so P(y | x) changes at most
        by a constant log-odds offset, which cannot alter any ranking metric;
      * strength = 0 gives w_i = 1 and reproduces the unweighted metric.
    For a Gaussian score, exponential tilting by `strength` moves the mean of
    the weighted PC1 score by `strength` standard deviations and leaves its
    variance unchanged, so `strength` keeps the meaning of the former "mean
    shift in standard deviations": positive = towards the high end of PC1,
    negative = towards the low end.

    WHY THE CLIP
    Heavy-tailed axes (e.g. medication burden) let a handful of outlying
    patients dominate an untruncated tilt. Measured on the RadFusion test set
    at strength +1, the effective sample size was 26% of n with a +-3 SD clip
    and 39% with the default +-2 SD clip (max weight 9.4 vs 4.5). Truncating
    the score is the standard weight-truncation remedy.

    The effective sample size ESS = (sum w)^2 / sum(w^2) tells how many
    unweighted patients the re-weighted test set is worth; the metric under a
    strong shift is correspondingly noisier (a property of the finite test
    set, shared by all models).
    """
    Z: numpy.ndarray = (
        X_test[shift_axis["features"]].to_numpy(dtype=float) - shift_axis["mean"]
    ) / shift_axis["std"]
    score: numpy.ndarray = (Z @ shift_axis["loading"]) / shift_axis["score_std"]
    score = numpy.clip(score, -max_abs_score, max_abs_score)

    weights: numpy.ndarray = numpy.exp(strength * score)

    y: numpy.ndarray = numpy.asarray(y_test).astype(int)
    for cls in (0, 1):
        in_class: numpy.ndarray = y == cls
        if in_class.any():
            weights[in_class] *= in_class.sum() / weights[in_class].sum()
    return weights


# ---------------------------------------------------------------------------
# Model building and evaluation
# ---------------------------------------------------------------------------

def build_model_package(
        individual: list[int],
        feature_names: list[str],
        X_train: pandas.DataFrame,
        y_train: pandas.Series,
        seed: int) -> dict[str, Any]:
    """Retrain a LogisticRegression on the full training set for the features
    selected by *individual* and return a ready-to-evaluate model package."""
    selected_features: list[str] = [
        f for f, bit in zip(feature_names, individual) if bit == 1
    ]

    scaler: StandardScaler = StandardScaler()
    X_scaled: numpy.ndarray = scaler.fit_transform(X_train[selected_features].to_numpy())

    # L2 is scikit-learn's default penalty. Passing `penalty="l2"` explicitly is
    # deprecated since scikit-learn 1.8 (removed in 1.10) and warns on every fit.
    model: LogisticRegression = LogisticRegression(
        solver="lbfgs", max_iter=1000, random_state=seed)
    model.fit(X_scaled, y_train)

    return {"model": model, "scaler": scaler, "features": selected_features}


def predict_scores(
        model_pkg: dict[str, Any],
        X: pandas.DataFrame) -> numpy.ndarray:
    """Predicted positive-class probabilities of a model package on (possibly
    noisy) data. Split from the metric so that one prediction can be scored
    under many test-population weightings (see `covariate_shift_weights`)."""
    features: list[str] = model_pkg["features"]
    X_scaled: numpy.ndarray = model_pkg["scaler"].transform(X[features].to_numpy())
    return model_pkg["model"].predict_proba(X_scaled)[:, 1]


def score_predictions(
        y_true: Union[numpy.ndarray, pandas.Series],
        y_prob: numpy.ndarray,
        use_roc_auc: bool = True,
        sample_weight: Union[numpy.ndarray, None] = None) -> float:
    """ROC-AUC (or PR-AUC / average precision) of a prediction vector,
    optionally under per-patient `sample_weight`s (a re-weighted test
    population). `sample_weight=None` is the ordinary unweighted metric."""
    if use_roc_auc:
        return float(roc_auc_score(y_true, y_prob, sample_weight=sample_weight))
    return float(average_precision_score(y_true, y_prob, sample_weight=sample_weight))


def evaluate_model(
        model_pkg: dict[str, Any],
        X_test: pandas.DataFrame,
        y_test: pandas.Series,
        use_roc_auc: bool = True,
        sample_weight: Union[numpy.ndarray, None] = None) -> float:
    """Score a model package on (possibly noisy) test data."""
    return score_predictions(
        y_test, predict_scores(model_pkg, X_test), use_roc_auc, sample_weight)


# ---------------------------------------------------------------------------
# Balanced sensitivity / specificity threshold
# ---------------------------------------------------------------------------

def find_balanced_threshold(
        y_test: Union[numpy.ndarray, pandas.Series],
        y_probs: Union[numpy.ndarray, pandas.Series]) -> dict[str, Any]:
    """Find the classification threshold at which sensitivity is closest to
    specificity.

    Sweeps all possible thresholds (each `y_probs` value acts as a cut-off,
    lowest first). At every candidate threshold we compute the confusion
    matrix, then sensitivity and specificity. The threshold with the smallest
    absolute (sensitivity - specificity) gap is returned, along with the full
    curves for plotting and the corresponding binary predictions.

    Returns a dict with keys:
      - `threshold`         : the balanced cut-off value.
      - `sensitivity`       : sensitivity at that threshold.
      - `specificity`       : specificity at that threshold.
      - `intersection_idx`  : the sorted-index of the balanced threshold.
      - `sorted_scores`     : the sorted y_probs array.
      - `sensitivity_curve` : per-threshold sensitivity (same length as scores).
      - `specificity_curve` : per-threshold specificity (same length as scores).
      - `y_pred`            : binary predictions using the balanced threshold.
    """
    y_true: numpy.ndarray = numpy.asarray(y_test).astype(int)
    y_probs_np: numpy.ndarray = numpy.asarray(y_probs, dtype=float)

    sorted_indices: numpy.ndarray = numpy.argsort(y_probs_np)
    sorted_scores: numpy.ndarray = y_probs_np[sorted_indices]
    sorted_y_true: numpy.ndarray = y_true[sorted_indices]

    n: int = len(y_true)
    sensitivity: numpy.ndarray = numpy.zeros(n)
    specificity: numpy.ndarray = numpy.zeros(n)

    for i in range(n):
        predicted_positive: numpy.ndarray = numpy.zeros(n)
        predicted_positive[i:] = 1

        tp: int = int(numpy.sum((predicted_positive == 1) & (sorted_y_true == 1)))
        fn: int = int(numpy.sum((predicted_positive == 0) & (sorted_y_true == 1)))
        tn: int = int(numpy.sum((predicted_positive == 0) & (sorted_y_true == 0)))
        fp: int = int(numpy.sum((predicted_positive == 1) & (sorted_y_true == 0)))

        sensitivity[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    diff: numpy.ndarray = numpy.abs(sensitivity - specificity)
    intersection_idx: int = int(numpy.argmin(diff))
    threshold: float = float(sorted_scores[intersection_idx])

    # Binary predictions on the ORIGINAL (unsorted) probability vector.
    y_pred: numpy.ndarray = (y_probs_np >= threshold).astype(int)

    return {
        "threshold":         threshold,
        "sensitivity":       float(sensitivity[intersection_idx]),
        "specificity":       float(specificity[intersection_idx]),
        "intersection_idx":  intersection_idx,
        "sorted_scores":     sorted_scores,
        "sensitivity_curve": sensitivity,
        "specificity_curve": specificity,
        "y_pred":            y_pred,
    }


# ---------------------------------------------------------------------------
# The single deployment model, chosen without the test set
# ---------------------------------------------------------------------------

def out_of_fold_scores(
        package: dict[str, Any],
        X_train: pandas.DataFrame,
        y_train: Union[numpy.ndarray, pandas.Series],
        cv: Any,
        seed: int) -> numpy.ndarray:
    """Out-of-fold predicted probabilities of a model package's specification on the training data: for
    every fold of `cv`, a fresh StandardScaler and the same logistic regression as `build_model_package`
    are fit on the other folds and predict the held-out rows."""
    features: list[str] = package["features"]
    X: numpy.ndarray = X_train[features].to_numpy(dtype=float)
    y: numpy.ndarray = numpy.asarray(y_train).astype(int)
    scores: numpy.ndarray = numpy.zeros(len(y))
    for train_rows, held_out in cv.split(X, y):
        scaler: StandardScaler = StandardScaler().fit(X[train_rows])
        model: LogisticRegression = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed)
        model.fit(scaler.transform(X[train_rows]), y[train_rows])
        scores[held_out] = model.predict_proba(scaler.transform(X[held_out]))[:, 1]
    return scores


def select_deployment_model(
        pareto_fronts: dict[int, list],
        seeds: list[int],
        feature_names: list[str],
        X_train: pandas.DataFrame,
        y_train: Union[numpy.ndarray, pandas.Series],
        cv: Any,
        use_knee_point: bool) -> dict[str, Any]:
    """The single MORSE model to deploy, chosen WITHOUT the test set.

    * Seed: the one whose selected Pareto solution (knee point or max-S end, like everywhere else) has the
      best cross-validated main objective -- the GA's own first fitness value, computed on the same
      folds for every seed; a tie goes to the smallest seed. Choosing the seed by its TEST score instead
      picks the luckiest of the seeds on that test set, so the reported test score is optimistic.
    * Threshold: the one at which sensitivity and specificity are closest on OUT-OF-FOLD predictions of
      the training data (`out_of_fold_scores`, the same CV splitter). A threshold found on the test labels
      is fitted to the very rows it is then evaluated on.
    The test set is then used once, to evaluate the chosen model at the chosen threshold.

    Returns {"seed", "individual", "cv_objective", "package" (refit on the whole training set),
    "oof_scores", "threshold", "balanced" (find_balanced_threshold on the out-of-fold predictions)}.
    """
    candidates: list[tuple[float, int, Any]] = []
    for seed in seeds:
        individual = select_pareto_individual(pareto_fronts[seed], use_knee_point=use_knee_point)
        candidates.append((float(individual.fitness.values[0]), seed, individual))
    cv_objective, seed, individual = max(candidates, key=lambda candidate: (candidate[0], -candidate[1]))
    package: dict[str, Any] = build_model_package(individual, feature_names, X_train, y_train, seed=seed)
    oof: numpy.ndarray = out_of_fold_scores(package, X_train, y_train, cv, seed)
    balanced: dict[str, Any] = find_balanced_threshold(y_train, oof)
    return {"seed": seed, "individual": individual, "cv_objective": cv_objective, "package": package,
            "oof_scores": oof, "threshold": balanced["threshold"], "balanced": balanced}


# ---------------------------------------------------------------------------
# AURS: Area Under the Robustness Surface -- LEGACY
#
# The summary of the legacy stress grid (legacy_stress_evaluation.py). The robustness suite
# (robustness_evaluation.py) does not use it.
# ---------------------------------------------------------------------------

# numpy 2.0 renamed `trapz` to `trapezoid` and later fully removed `trapz`
# (accessing it raises AttributeError instead of merely warning); numpy < 2.0
# only has `trapz`. Resolve once here so `compute_aurs` works unmodified on
# either numpy major version, regardless of exactly what's installed.
_trapezoid = getattr(numpy, "trapezoid", None) or numpy.trapz


def compute_aurs(heatmap_agg: pandas.DataFrame, model_key: str) -> float:
    """Summarise a model's noise-robustness 2-D sweep into a single score:
    AURS, the **Area Under the Robustness Surface**.

    WHAT AURS IS
    ------------
    The 2-D noise sweep (`gaussian_2d_heatmap_grid_test.png` / `heatmap_agg`
    of `legacy_stress_evaluation.run_legacy_stress_grid`) evaluates every model
    at each combination of (Gaussian noise level, covariate-shift strength),
    giving one AUC/PR-AUC number per grid cell. The second axis is stored in
    the column named `mean_shift`; it holds the strength of the
    re-weighted-population covariate shift of `covariate_shift_weights` (in SD
    of the dominant covariate axis), not a translation of feature values. AURS
    collapses that whole grid into ONE number per model: the average fraction
    of the model's OWN clean-test score that it retains, averaged over every
    stress condition in the swept grid.

    It deliberately does *not* just average the raw AUC values across the
    grid. Two models can have different clean-test AUCs, so a plain average
    of raw degraded AUCs would conflate two different questions: "how good
    is this model to begin with?" (already reported elsewhere -- e.g. the
    Pareto front, or the clean-cell entry of this same grid) and "how much
    does stress hurt it, relative to where it started?" AURS isolates the
    second question.

    HOW IT IS CALCULATED
    ---------------------
    1. Pivot `heatmap_agg[f"auc_{model_key}"]` -- already averaged across
       seeds by the caller -- into a 2-D grid indexed by (mean_shift rows x
       noise_level columns).
    2. Locate the CLEAN cell of that grid, i.e. noise_level == 0 AND
       mean_shift == 0 (the unperturbed test set), and read the model's
       clean-test score there: `clean_value`.
    3. Convert every cell into a RETENTION ratio relative to that clean
       baseline: `retention[i, j] = auc[i, j] / clean_value`. A retention of
       1.0 means "no degradation at all at this stress level"; 0.5 means
       "half of the clean-test score is lost here".
    4. Numerically integrate the retention surface over the full 2-D grid
       with the composite trapezoidal rule (`numpy.trapezoid`, or
       `numpy.trapz` on numpy < 2.0): first along the noise axis for every
       fixed shift level, then integrate that resulting 1-D profile along
       the shift axis. This is a genuine double integral of the (noise,
       shift) -> retention surface, not a naive flat average, so it would
       still weight the grid correctly even if the swept noise/shift levels
       were not evenly spaced.
    5. Divide the raw integral by the grid's total area,
       `(noise_range) x (shift_range)`, to renormalise it back onto the same
       [~0, ~1] retention scale that a single cell lives on -- an integral
       by itself scales with the size of the grid, not just its shape. The
       result of this division IS the AURS score.

    INTERPRETING THE SCORE -- AND ITS LIMITS
    -----------------------------------------
    - AURS = 1.0 (100%): on average over the grid the model keeps its whole
      clean-test score.
    - AURS can land ABOVE 1.0, and that is the main limitation of the score:
      the covariate shift re-weights the test population in BOTH directions
      along the axis, and a shift towards easier cases raises every model's
      score (on RadFusion, for example, all four methods score higher at
      strength -1 than on the clean test set). The average over the grid nets
      those gains against the losses in the harmful direction, so a model can
      look robust although it degrades badly on one side. The robustness suite
      reports the worst scenario of a shift axis instead.
    - AURS is only meaningful together with the specific grid it was
      computed over. Widening the swept noise/shift range will generally
      LOWER every model's AURS even if nothing about the model itself
      changed, simply because more (harsher) grid is now being averaged
      in. Always report the swept `noise_levels` / `shift_levels` extent
      alongside the score.

    Parameters
    ----------
    heatmap_agg
        A DataFrame with a two-level MultiIndex `(noise_level, mean_shift)`
        and at least the column `f"auc_{model_key}"`, already averaged
        across seeds -- exactly the `heatmap_agg` built by the legacy grid via
        `heatmap_df.groupby(["noise_level", "mean_shift"]).mean()`.
    model_key
        The model key whose column (`auc_{model_key}`) to score, e.g.
        `"multi"`, `"single"`, `"all"`, `"forward"`.

    Returns
    -------
    float
        The AURS score as a fraction (multiply by 100 for a percentage).

    Raises
    ------
    ValueError
        If the grid has no exact (noise_level=0, mean_shift=0) clean cell to
        normalise against, or if that cell's score is not strictly positive.
    """
    col: str = f"auc_{model_key}"
    pivot: pandas.DataFrame = (
        heatmap_agg[[col]]
        .reset_index()
        .pivot(index="mean_shift", columns="noise_level", values=col)
        .sort_index(axis=0)   # ascending mean_shift
        .sort_index(axis=1)   # ascending noise_level
    )

    shift_levels: numpy.ndarray = pivot.index.to_numpy(dtype=float)
    noise_levels: numpy.ndarray = pivot.columns.to_numpy(dtype=float)
    grid: numpy.ndarray = pivot.to_numpy(dtype=float)  # shape (n_shift, n_noise)

    i0: int = int(numpy.argmin(numpy.abs(shift_levels)))
    j0: int = int(numpy.argmin(numpy.abs(noise_levels)))
    if not (numpy.isclose(shift_levels[i0], 0.0) and numpy.isclose(noise_levels[j0], 0.0)):
        raise ValueError(
            f"compute_aurs requires an exact (noise_level=0, mean_shift=0) "
            f"clean cell to normalise against; closest grid point was "
            f"(noise_level={noise_levels[j0]}, mean_shift={shift_levels[i0]})."
        )

    clean_value: float = float(grid[i0, j0])
    if clean_value <= 0.0:
        raise ValueError(
            f"Clean-cell '{col}' value must be strictly positive to "
            f"normalise by; got {clean_value}."
        )

    retention: numpy.ndarray = grid / clean_value

    # Double trapezoidal integration: integrate along the noise axis for
    # every shift level, then integrate the resulting 1-D profile along the
    # shift axis. `_trapezoid` resolves to numpy.trapezoid (numpy >= 2.0) or
    # numpy.trapz (numpy < 2.0) -- see the comment above.
    inner: numpy.ndarray = _trapezoid(retention, x=noise_levels, axis=1)
    total: float = float(_trapezoid(inner, x=shift_levels))

    grid_area: float = float(noise_levels[-1] - noise_levels[0]) * float(shift_levels[-1] - shift_levels[0])
    return total / grid_area
