from __future__ import annotations

from typing import Any, Union
import numpy
import pandas
import matplotlib
matplotlib.use("Agg")
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

# numpy 2.0 renamed `trapz` to `trapezoid` and later fully removed `trapz`
# (accessing it raises AttributeError instead of merely warning); numpy < 2.0
# only has `trapz`. Resolve once here so `compute_aurs` works unmodified on
# either numpy major version, regardless of exactly what's installed.
_trapezoid = getattr(numpy, "trapezoid", None) or numpy.trapz

# ---------------------------------------------------------------------------
# Sign consistency score calculation
# ---------------------------------------------------------------------------

def compute_marginal_correlations(
    X: Union[numpy.ndarray, pandas.DataFrame],
    y: Union[numpy.ndarray, pandas.Series],
) -> numpy.ndarray:
    """Per-feature marginal correlation with the binary target `y`.

    - Matthews correlation for columns with exactly 2 unique values.
    - Point-biserial correlation for columns with more than 2 unique values.
    - 0 for constant columns.

    Returns a numpy array of length `X.shape[1]`, aligned with the column
    order of `X`.
    """
    X_arr: numpy.ndarray = X.to_numpy() if hasattr(X, "to_numpy") else numpy.asarray(X)
    y_int: numpy.ndarray = numpy.asarray(y, dtype=int)
    n_features: int = X_arr.shape[1]

    out: numpy.ndarray = numpy.zeros(n_features, dtype=float)
    for j in range(n_features):
        feat: numpy.ndarray = X_arr[:, j]
        u: int = len(numpy.unique(feat))
        if u <= 1:
            out[j] = 0.0
        elif u == 2:
            out[j] = float(matthews_corrcoef(y_int, feat.astype(int)))
        else:
            corr, _ = pointbiserialr(y_int, feat)
            out[j] = float(corr)
    return out


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
# Noise injection
# ---------------------------------------------------------------------------

def apply_proportional_noise(
        X_test: pandas.DataFrame,
        train_std: pandas.Series,
        noise_fraction: float,
        mean_shift_fraction: float,
        continuous_cols: list[str]) -> pandas.DataFrame:
    """
    Adds Gaussian noise and/or systematic mean shift to continuous variables,
    proportional to the training-set standard deviation.

    noise_fraction:      0.0-1.0  (scale of additive Gaussian noise)
    mean_shift_fraction: -1.0-1.0 (direction and magnitude of covariate shift)
    """
    no_noise: bool = numpy.isclose(noise_fraction, 0.0, atol=1e-09)
    no_shift: bool = numpy.isclose(mean_shift_fraction, 0.0, atol=1e-09)

    if no_noise and no_shift:
        return X_test.copy()

    X_out: pandas.DataFrame = X_test.copy()

    for col in continuous_cols:
        if col not in X_out.columns or col not in train_std.index:
            continue
        std_val: float = train_std[col]

        if not no_shift:
            X_out[col] += mean_shift_fraction * std_val

        if not no_noise:
            noise: numpy.ndarray = numpy.random.normal(
                loc=0.0, scale=noise_fraction * std_val, size=len(X_out))
            X_out[col] += noise

    return X_out


def apply_dummy_noise(
        X_test: pandas.DataFrame,
        noise_fraction: float,
        dummy_cols: list[str]) -> pandas.DataFrame:
    """
    Randomisation noise on dummy (binary) variables.

    noise_fraction: fraction [0.0-1.0] of cells that are **replaced** with a
    random coin flip (0 or 1 with equal probability).

    * 0.0  → original data, no noise
    * 0.5  → half the dummy cells are replaced with random values
    * 1.0  → all dummy cells are uniformly random (maximum entropy, no signal)

    This guarantees monotonic signal degradation: unlike deterministic bit-flip,
    fractions above 0.5 cannot invert and recover the original signal.
    """
    if numpy.isclose(noise_fraction, 0.0, atol=1e-09):
        return X_test.copy()

    X_out: pandas.DataFrame = X_test.copy()

    for col in dummy_cols:
        if col not in X_out.columns:
            continue
        # Select which cells to corrupt
        corrupt_mask: numpy.ndarray = numpy.random.rand(len(X_out)) < noise_fraction
        n_corrupt: int = int(corrupt_mask.sum())

        if n_corrupt == 0:
            continue

        # Replace selected cells with uniform random {0, 1}
        random_vals: numpy.ndarray = numpy.random.randint(0, 2, size=n_corrupt)

        if pandas.api.types.is_bool_dtype(X_out[col]):
            X_out.loc[corrupt_mask, col] = random_vals.astype(bool)
        else:
            X_out.loc[corrupt_mask, col] = random_vals

    return X_out


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

    model: LogisticRegression = LogisticRegression(
        penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed)
    model.fit(X_scaled, y_train)

    return {"model": model, "scaler": scaler, "features": selected_features}


def evaluate_model(
        model_pkg: dict[str, Any],
        X_test: pandas.DataFrame,
        y_test: pandas.Series,
        use_roc_auc: bool = True) -> float:
    """Score a model package on (possibly noisy) test data."""
    features: list[str] = model_pkg["features"]
    X_scaled: numpy.ndarray = model_pkg["scaler"].transform(
        X_test[features].to_numpy())
    y_prob: numpy.ndarray = model_pkg["model"].predict_proba(X_scaled)[:, 1]

    if use_roc_auc:
        return float(roc_auc_score(y_test, y_prob))
    return float(average_precision_score(y_test, y_prob))


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
# AURS: Area Under the Robustness Surface
# ---------------------------------------------------------------------------

def compute_aurs(heatmap_agg: pandas.DataFrame, model_key: str) -> float:
    """Summarise a model's noise-robustness 2-D sweep into a single score:
    AURS, the **Area Under the Robustness Surface**.

    WHAT AURS IS
    ------------
    The 2-D noise sweep (`gaussian_2d_heatmap_grid_test.png` / `heatmap_agg`
    in the notebook) evaluates every model at each combination of
    (Gaussian noise level, covariate mean-shift), giving one AUC/PR-AUC
    number per grid cell. AURS collapses that whole grid into ONE number per
    model: the average fraction of the model's OWN clean-test score that it
    retains, averaged over every stress condition in the swept grid.

    It deliberately does *not* just average the raw AUC values across the
    grid. Two models can have different clean-test AUCs, so a plain average
    of raw degraded AUCs would conflate two different questions: "how good
    is this model to begin with?" (already reported elsewhere -- e.g. the
    Pareto front, or the clean-cell entry of this same grid) and "how much
    does stress hurt it, relative to where it started?" AURS isolates the
    second question, which is the one MORSE's own hypothesis is about: "a
    modest cost in clean-test AUC in exchange for better robustness to
    noise" needs the clean-cost and the robustness measured separately, and
    AURS is the robustness half of that comparison.

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

    INTERPRETING THE SCORE
    -----------------------
    - AURS = 1.0 (100%): the model's score is completely flat across the
      ENTIRE swept range of noise and covariate shift -- effectively
      perfect robustness within the tested grid.
    - AURS = 0.0 (0%): the model's score collapses to zero everywhere in
      the grid except the clean cell itself.
    - AURS can occasionally land slightly ABOVE 1.0. This is not a bug: a
      small amount of injected noise can sometimes marginally *improve* a
      held-out score (a mild regularisation-like effect), and heatmap_agg
      is itself an across-seed mean, which carries its own sampling noise
      -- both effects are most visible in the lightly-perturbed cells
      right next to the clean cell.
    - AURS is only meaningful together with the specific grid it was
      computed over. Widening the swept noise/shift range will generally
      LOWER every model's AURS even if nothing about the model itself
      changed, simply because more (harsher) grid is now being averaged
      in. Always report the swept `noise_levels` / `shift_levels` extent
      alongside the score.

    This normalise-by-own-clean-baseline approach is conceptually similar
    to the relative-degradation metrics used in the image-corruption
    robustness literature (e.g. Hendrycks & Dietterich's benchmarks of
    common image corruptions), which likewise score robustness against a
    model's own clean performance rather than comparing raw corrupted
    scores across models directly.

    Parameters
    ----------
    heatmap_agg
        A DataFrame with a two-level MultiIndex `(noise_level, mean_shift)`
        and at least the column `f"auc_{model_key}"`, already averaged
        across seeds -- exactly the `heatmap_agg` built in the notebook via
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
    # numpy.trapz (numpy < 2.0) -- see the module-level comment above.
    inner: numpy.ndarray = _trapezoid(retention, x=noise_levels, axis=1)
    total: float = float(_trapezoid(inner, x=shift_levels))

    grid_area: float = float(noise_levels[-1] - noise_levels[0]) * float(shift_levels[-1] - shift_levels[0])
    return total / grid_area
