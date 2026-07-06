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
