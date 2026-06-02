from __future__ import annotations

import os
import warnings
from typing import Any, Union

import numpy
import pandas
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor

from training_utils import ensure_directory


# ---------------------------------------------------------------------------
# Marginal correlations + sign-consistency primitives
# Shared by the multi-objective GA fitness (multi_objective_training.py) and
# the post-hoc verification / sign-inconsistency reports in the notebook.
# ---------------------------------------------------------------------------

def compute_marginal_correlations(
    X: Union[numpy.ndarray, pandas.DataFrame],
    y: Union[numpy.ndarray, pandas.Series],
) -> numpy.ndarray:
    """Per-feature marginal correlation with the binary target `y`.

    Uses Matthews correlation when a column has exactly two unique values
    (and is therefore effectively binary), point-biserial correlation when
    a column has more than two unique values, and 0 for constant columns.

    Returns a numpy array of length `X.shape[1]`, aligned with the column
    order of `X`. Numerically identical to the inline loops previously
    embedded in cells 5, 18 and 30 of training_notebook.ipynb.
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


def compute_sign_consistency_score(
    coefficients: numpy.ndarray,
    correlations: numpy.ndarray,
    zero_atol: float = 1e-12,
) -> float:
    """Sign-consistency score in [0, 1] for a fitted model.

    A feature contributes a penalty when its (marginal correlation x model
    coefficient) product is negative OR approximately zero (within `zero_atol`).
    The score is `1 - penalised_fraction`.

    This is the canonical definition used both as the second objective of the
    multi-objective GA (per fold inside MultiObjectiveTraining) and as the
    post-hoc verification metric in the fold-vs-full-train report. Keeping
    a single definition here removes the risk of the two paths drifting
    apart.
    """
    coefs: numpy.ndarray = numpy.asarray(coefficients).ravel()
    corrs: numpy.ndarray = numpy.asarray(correlations).ravel()
    n: int = int(coefs.shape[0])
    if n == 0:
        return 0.0
    if corrs.shape[0] != n:
        raise ValueError(
            f"coefficients and correlations must have equal length, "
            f"got {n} vs {corrs.shape[0]}"
        )
    check: numpy.ndarray = corrs * coefs
    penalties: int = int(numpy.sum((check < 0) | numpy.isclose(check, 0.0, atol=zero_atol)))
    return 1.0 - penalties / n


# ---------------------------------------------------------------------------
# Matplotlib global theme: white background, black text
# ---------------------------------------------------------------------------

def _apply_plot_theme() -> None:
    """Set a clean white-background, black-text theme for all plots."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.edgecolor": "black",
        "grid.color": "#cccccc",
    })


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
# Full noise sweep
# ---------------------------------------------------------------------------

def run_noise_evaluation(
        model_pkg_multi: dict[str, Any],
        model_pkg_single: dict[str, Any],
        X_test: pandas.DataFrame,
        y_test: pandas.Series,
        train_std: pandas.Series,
        continuous_cols: list[str],
        dummy_cols: list[str],
        use_roc_auc: bool = True) -> pandas.DataFrame:
    """Run Gaussian-noise + mean-shift sweep and dummy bit-flip sweep,
    returning a tidy DataFrame with all results."""

    noise_levels: numpy.ndarray = numpy.arange(0.0, 1.1, 0.1)
    shift_levels: numpy.ndarray = numpy.arange(-1.0, 1.1, 0.2)
    flip_levels: numpy.ndarray = numpy.arange(0.0, 1.05, 0.1)

    results: list[dict[str, Any]] = []

    # --- Gaussian noise + mean-shift on continuous features ---
    for shift in shift_levels:
        for noise in noise_levels:
            clean_shift: float = round(float(shift), 2)
            clean_noise: float = round(float(noise), 2)

            X_noisy: pandas.DataFrame = apply_proportional_noise(
                X_test, train_std, clean_noise, clean_shift, continuous_cols)

            results.append({
                "noise_type": "gaussian",
                "noise_level": clean_noise,
                "mean_shift": clean_shift,
                "flip_rate": 0.0,
                "auc_multi": evaluate_model(model_pkg_multi, X_noisy, y_test, use_roc_auc),
                "auc_single": evaluate_model(model_pkg_single, X_noisy, y_test, use_roc_auc),
            })

    # --- Bit-flip on dummy features ---
    for flip in flip_levels:
        clean_flip: float = round(float(flip), 2)

        X_noisy = apply_dummy_noise(X_test, clean_flip, dummy_cols)

        results.append({
            "noise_type": "dummy_flip",
            "noise_level": 0.0,
            "mean_shift": 0.0,
            "flip_rate": clean_flip,
            "auc_multi": evaluate_model(model_pkg_multi, X_noisy, y_test, use_roc_auc),
            "auc_single": evaluate_model(model_pkg_single, X_noisy, y_test, use_roc_auc),
        })

    return pandas.DataFrame(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_robustness_heatmaps(results_df: pandas.DataFrame, filepath: str,
                             dataset_label: str = "Test", use_roc_auc: bool = True) -> None:
    MAIN_OBJECTIVE: str = "ROC-AUC" if use_roc_auc == True else "PR-AUC"

    """Side-by-side heatmaps: multi vs single under Gaussian noise + shift."""
    ensure_directory(os.path.dirname(filepath))

    df_gauss: pandas.DataFrame = results_df[results_df["noise_type"] == "gaussian"]

    pivot_multi: pandas.DataFrame = df_gauss.pivot_table(
        index="mean_shift", columns="noise_level", values="auc_multi")
    pivot_multi = pivot_multi.sort_index(ascending=False)

    pivot_single: pandas.DataFrame = df_gauss.pivot_table(
        index="mean_shift", columns="noise_level", values="auc_single")
    pivot_single = pivot_single.sort_index(ascending=False)

    vmin: float = min(df_gauss["auc_multi"].min(), df_gauss["auc_single"].min())
    vmax: float = max(df_gauss["auc_multi"].max(), df_gauss["auc_single"].max())

    clean_multi: float = pivot_multi.loc[0.0, 0.0] if 0.0 in pivot_multi.index and 0.0 in pivot_multi.columns else float("nan")
    clean_single: float = pivot_single.loc[0.0, 0.0] if 0.0 in pivot_single.index and 0.0 in pivot_single.columns else float("nan")

    _apply_plot_theme()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(pivot_multi, cmap="viridis", vmin=vmin, vmax=vmax, annot=False,
                cbar_kws={"label": f"{dataset_label} {MAIN_OBJECTIVE}"}, ax=axes[0])
    axes[0].set_title(f"Multi-Objective (MORSE) Robustness\n"
                      f"Clean AUC = {clean_multi:.4f} | AUC range [{vmin:.4f}, {vmax:.4f}]",
                      fontweight="bold", pad=15)
    axes[0].set_xlabel("Gaussian Noise Level (fraction of training std)", fontweight="bold")
    axes[0].set_ylabel("Systematic Mean Shift (fraction of training std)", fontweight="bold")

    sns.heatmap(pivot_single, cmap="viridis", vmin=vmin, vmax=vmax, annot=False,
                cbar_kws={"label": f"{dataset_label} {MAIN_OBJECTIVE}"}, ax=axes[1])
    axes[1].set_title(f"Single-Objective (AUC-only GA) Robustness\n"
                      f"Clean AUC = {clean_single:.4f} | AUC range [{vmin:.4f}, {vmax:.4f}]",
                      fontweight="bold", pad=15)
    axes[1].set_xlabel("Gaussian Noise Level (fraction of training std)", fontweight="bold")
    axes[1].set_ylabel("Systematic Mean Shift (fraction of training std)", fontweight="bold")

    fig.suptitle(f"Robustness on {dataset_label} Set: Gaussian Noise + Covariate Shift on Continuous Features",
                 fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dummy_flip_comparison(results_df: pandas.DataFrame, filepath: str,
                               dataset_label: str = "Test", use_roc_auc: bool = True) -> None:
    MAIN_OBJECTIVE: str = "ROC-AUC" if use_roc_auc == True else "PR-AUC"

    """Line plot: multi vs single AUC under increasing bit-flip rate."""
    ensure_directory(os.path.dirname(filepath))

    df_flip: pandas.DataFrame = results_df[results_df["noise_type"] == "dummy_flip"].sort_values("flip_rate")

    _apply_plot_theme()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_flip["flip_rate"], df_flip["auc_multi"],
            label="Multi-Objective (MORSE)", linewidth=2, marker="o", color="tab:blue")
    ax.plot(df_flip["flip_rate"], df_flip["auc_single"],
            label="Single-Objective (AUC-only GA)", linewidth=2, marker="s", linestyle="--", color="tab:orange")
    ax.set_xlabel("Corruption Fraction (proportion of dummy cells randomised)", fontweight="bold")
    ax.set_ylabel(f"{dataset_label} {MAIN_OBJECTIVE}", fontweight="bold")
    ax.set_title(f"Robustness on {dataset_label} Set: Random Corruption Noise on Binary (Dummy) Features\n"
                 f"(corrupted cells replaced with uniform random {{0, 1}})",
                 fontweight="bold", pad=10)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    if not df_flip.empty:
        ax.annotate(f"Clean AUC: Multi={df_flip['auc_multi'].iloc[0]:.4f}, "
                    f"Single={df_flip['auc_single'].iloc[0]:.4f}",
                    xy=(0.02, 0.02), xycoords="axes fraction", fontsize=9,
                    fontstyle="italic", color="gray")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_gaussian_noise_comparison(results_df: pandas.DataFrame, filepath: str,
                                    dataset_label: str = "Test", use_roc_auc: bool = True) -> None:
    MAIN_OBJECTIVE: str = "ROC-AUC" if use_roc_auc == True else "PR-AUC"

    """Line plot: multi vs single AUC under Gaussian noise (zero mean-shift)."""
    ensure_directory(os.path.dirname(filepath))

    df_gauss: pandas.DataFrame = results_df[
        (results_df["noise_type"] == "gaussian") &
        (numpy.isclose(results_df["mean_shift"], 0.0))
    ].sort_values("noise_level")

    _apply_plot_theme()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_gauss["noise_level"], df_gauss["auc_multi"],
            label="Multi-Objective (MORSE)", linewidth=2, marker="o", color="tab:blue")
    ax.plot(df_gauss["noise_level"], df_gauss["auc_single"],
            label="Single-Objective (AUC-only GA)", linewidth=2, marker="s", linestyle="--", color="tab:orange")
    ax.set_xlabel("Gaussian Noise Level (fraction of training std, no mean shift)",
                  fontweight="bold")
    ax.set_ylabel(f"{dataset_label} {MAIN_OBJECTIVE}", fontweight="bold")
    ax.set_title(f"Robustness on {dataset_label} Set: Additive Gaussian Noise on Continuous Features\n"
                 f"(zero systematic mean shift)",
                 fontweight="bold", pad=10)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    if not df_gauss.empty:
        ax.annotate(f"Clean AUC: Multi={df_gauss['auc_multi'].iloc[0]:.4f}, "
                    f"Single={df_gauss['auc_single'].iloc[0]:.4f}",
                    xy=(0.02, 0.02), xycoords="axes fraction", fontsize=9,
                    fontstyle="italic", color="gray")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator: evaluate one seed and save everything
# ---------------------------------------------------------------------------

def evaluate_and_save(
        seed: int,
        model_pkg_multi: dict[str, Any],
        model_pkg_single: dict[str, Any],
        X_eval: pandas.DataFrame,
        y_eval: pandas.Series,
        train_std: pandas.Series,
        continuous_cols: list[str],
        dummy_cols: list[str],
        result_directory: str,
        use_roc_auc: bool = True,
        dataset_label: str = "Test") -> pandas.DataFrame:
    """Run the full noise evaluation for one seed and persist CSV + plots.

    dataset_label: human-readable name for the dataset being evaluated
    (e.g. "Test" or "Validation").  Used in plot titles, file names, and
    the console summary line.
    """
    results_df: pandas.DataFrame = run_noise_evaluation(
        model_pkg_multi, model_pkg_single,
        X_eval, y_eval, train_std,
        continuous_cols, dummy_cols, use_roc_auc)

    label_lower: str = dataset_label.lower()
    seed_dir: str = os.path.join(result_directory, f"seed_{seed}")
    ensure_directory(seed_dir)

    results_df.to_csv(os.path.join(seed_dir, f"noise_evaluation_{label_lower}.csv"), index=False)
    plot_robustness_heatmaps(results_df, os.path.join(seed_dir, f"robustness_heatmaps_{label_lower}.png"), dataset_label, use_roc_auc)
    plot_gaussian_noise_comparison(results_df, os.path.join(seed_dir, f"gaussian_noise_comparison_{label_lower}.png"), dataset_label, use_roc_auc)
    plot_dummy_flip_comparison(results_df, os.path.join(seed_dir, f"dummy_flip_comparison_{label_lower}.png"), dataset_label, use_roc_auc)

    # Print summary for this seed
    clean: pandas.DataFrame = results_df[
        (results_df["noise_type"] == "gaussian") &
        (numpy.isclose(results_df["noise_level"], 0.0)) &
        (numpy.isclose(results_df["mean_shift"], 0.0))
    ]
    if not clean.empty:
        row: pandas.Series = clean.iloc[0]
        print(f"  Seed {seed} [{dataset_label}] | Multi features: {len(model_pkg_multi['features']):3d} | "
              f"Single features: {len(model_pkg_single['features']):3d} | "
              f"Clean AUC  Multi: {row['auc_multi']:.4f}  Single: {row['auc_single']:.4f}")

    return results_df


# ---------------------------------------------------------------------------
# Feature selection stability (Jaccard similarity)
# ---------------------------------------------------------------------------

def _individual_to_feature_set(individual: list[int], feature_names: list[str]) -> set[str]:
    """Convert a binary individual to a set of selected feature names."""
    return {f for f, bit in zip(feature_names, individual) if bit == 1}


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity: |A intersect B| / |A union B|.  Returns 0.0 if both empty."""
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_stability_matrix(
        individuals_by_seed: dict[int, list[int]],
        feature_names: list[str]) -> tuple[list[int], numpy.ndarray]:
    """Compute pairwise Jaccard similarity matrix across seeds.
    Returns (seed_list, similarity_matrix)."""
    seeds: list[int] = sorted(individuals_by_seed.keys())
    n: int = len(seeds)
    feature_sets: list[set[str]] = [
        _individual_to_feature_set(individuals_by_seed[s], feature_names)
        for s in seeds
    ]

    matrix: numpy.ndarray = numpy.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = jaccard_similarity(feature_sets[i], feature_sets[j])

    return seeds, matrix


def plot_stability_heatmap(
        seeds_multi: list[int],
        matrix_multi: numpy.ndarray,
        seeds_single: list[int],
        matrix_single: numpy.ndarray,
        filepath: str) -> None:
    """Side-by-side Jaccard similarity heatmaps for multi and single objective."""
    ensure_directory(os.path.dirname(filepath))

    mean_multi: float = float(matrix_multi[numpy.triu_indices_from(matrix_multi, k=1)].mean()) \
        if len(seeds_multi) > 1 else 1.0
    mean_single: float = float(matrix_single[numpy.triu_indices_from(matrix_single, k=1)].mean()) \
        if len(seeds_single) > 1 else 1.0

    # Shared color scale across both panels: identical colormap and value range
    # so the two heatmaps are directly visually comparable.
    cmap_name: str = "viridis"
    vmin: float = 0.0
    vmax: float = 1.0

    _apply_plot_theme()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(matrix_multi, annot=True, fmt=".3f", cmap=cmap_name,
                vmin=vmin, vmax=vmax, square=True, ax=axes[0],
                xticklabels=[f"Seed {s}" for s in seeds_multi],
                yticklabels=[f"Seed {s}" for s in seeds_multi],
                cbar=False)
    axes[0].set_title(f"Multi-Objective (MORSE)\n"
                      f"Mean pairwise Jaccard = {mean_multi:.3f}",
                      fontweight="bold", pad=15)
    axes[0].set_xlabel("Seed", fontweight="bold")
    axes[0].set_ylabel("Seed", fontweight="bold")

    sns.heatmap(matrix_single, annot=True, fmt=".3f", cmap=cmap_name,
                vmin=vmin, vmax=vmax, square=True, ax=axes[1],
                xticklabels=[f"Seed {s}" for s in seeds_single],
                yticklabels=[f"Seed {s}" for s in seeds_single],
                cbar=False)
    axes[1].set_title(f"Single-Objective (AUC-only GA)\n"
                      f"Mean pairwise Jaccard = {mean_single:.3f}",
                      fontweight="bold", pad=15)
    axes[1].set_xlabel("Seed", fontweight="bold")
    axes[1].set_ylabel("Seed", fontweight="bold")

    fig.suptitle("Feature Selection Stability Across Seeds (Jaccard Similarity)",
                 fontweight="bold", fontsize=14, y=1.02)

    # Single shared colorbar on the right so both panels read against the same scale.
    fig.tight_layout(rect=(0.0, 0.0, 0.9, 1.0))
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    sm = plt.cm.ScalarMappable(
        cmap=cmap_name, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Jaccard Similarity")

    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_frequency(
        individuals_by_seed: dict[int, list[int]],
        feature_names: list[str],
        label: str,
        filepath: str) -> None:
    """Bar chart showing how often each feature is selected across seeds."""
    ensure_directory(os.path.dirname(filepath))

    n_seeds: int = len(individuals_by_seed)
    counts: numpy.ndarray = numpy.zeros(len(feature_names))
    for ind in individuals_by_seed.values():
        counts += numpy.array(ind)

    order: numpy.ndarray = numpy.argsort(-counts)
    # Keep only features selected at least once
    mask: numpy.ndarray = counts[order] > 0
    order: numpy.ndarray = order[mask]

    _apply_plot_theme()
    fig, ax = plt.subplots(figsize=(max(10, len(order) * 0.4), 6))
    x_pos: numpy.ndarray = numpy.arange(len(order))
    bars = ax.bar(x_pos, counts[order], color="steelblue", edgecolor="black", alpha=0.8)

    # Highlight features selected by all seeds
    for i, idx in enumerate(order):
        if counts[idx] == n_seeds:
            bars[i].set_color("darkgreen")
            bars[i].set_alpha(1.0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([feature_names[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Feature Name", fontweight="bold")
    ax.set_ylabel(f"Selection Count (out of {n_seeds} seeds)", fontweight="bold")
    ax.set_title(f"Feature Selection Frequency - {label}\n"
                 f"(green = selected by all {n_seeds} seeds, "
                 f"blue = selected by some seeds)",
                 fontweight="bold", pad=10)
    ax.set_ylim(0, n_seeds + 0.5)
    ax.axhline(y=n_seeds, color="darkgreen", linestyle="--", alpha=0.5, label=f"All {n_seeds} seeds")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Variance Inflation Factor (VIF) analysis
# ---------------------------------------------------------------------------

def compute_vif(X: pandas.DataFrame, features: list[str]) -> pandas.DataFrame:
    """Compute VIF for each selected feature.
    Returns a DataFrame with columns ['feature', 'vif'].
    Zero-variance columns are excluded; inf/NaN VIF values are capped."""
    X_subset: pandas.DataFrame = X[features].copy().astype(numpy.float64)

    # Drop zero-variance columns (they cause division by zero in R²)
    zero_var_cols: list[str] = [col for col in X_subset.columns if X_subset[col].std() == 0]
    if zero_var_cols:
        X_subset = X_subset.drop(columns=zero_var_cols)

    if X_subset.shape[1] < 2:
        return pandas.DataFrame(columns=["feature", "vif"])

    # Add constant column for intercept
    X_subset.insert(0, "_const", 1.0)

    values: numpy.ndarray = X_subset.to_numpy(dtype=numpy.float64)
    vif_data: list[dict[str, Any]] = []
    for i in range(1, X_subset.shape[1]):  # skip constant
        # Suppress the "divide by zero" RuntimeWarning that statsmodels emits
        # when R^2 == 1 (perfect collinearity); we cap the inf result below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vif_val: float = float(variance_inflation_factor(values, i))
        # Cap inf/NaN (perfect collinearity) to a large finite value
        if not numpy.isfinite(vif_val):
            vif_val = 1e6
        vif_data.append({
            "feature": X_subset.columns[i],
            "vif": vif_val,
        })

    return pandas.DataFrame(vif_data)


def compute_vif_across_seeds(
        individuals_by_seed: dict[int, list[int]],
        feature_names: list[str],
        X_train: pandas.DataFrame) -> pandas.DataFrame:
    """Compute VIF for each seed's selected features.
    Returns a tidy DataFrame with columns ['seed', 'feature', 'vif']."""
    all_rows: list[dict[str, Any]] = []
    for seed, individual in individuals_by_seed.items():
        selected: list[str] = [f for f, bit in zip(feature_names, individual) if bit == 1]
        if len(selected) < 2:
            continue
        vif_df: pandas.DataFrame = compute_vif(X_train, selected)
        vif_df["seed"] = seed
        all_rows.extend(vif_df.to_dict("records"))

    return pandas.DataFrame(all_rows)


def plot_vif_boxplot(
        vif_multi: pandas.DataFrame,
        vif_single: pandas.DataFrame,
        filepath: str) -> None:
    """Side-by-side boxplots comparing VIF distributions of MORSE vs SOGA selected features."""
    ensure_directory(os.path.dirname(filepath))

    vif_multi_plot: pandas.DataFrame = vif_multi.copy()
    vif_multi_plot["method"] = "Multi-Objective\n(MORSE)"
    vif_single_plot: pandas.DataFrame = vif_single.copy()
    vif_single_plot["method"] = "Single-Objective\n(AUC-only GA)"

    combined: pandas.DataFrame = pandas.concat([vif_multi_plot, vif_single_plot], ignore_index=True)

    _apply_plot_theme()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: boxplot by method
    sns.boxplot(data=combined, x="method", y="vif", hue="method", ax=axes[0],
            palette={"Multi-Objective\n(MORSE)": "tab:blue",
                     "Single-Objective\n(AUC-only GA)": "tab:orange"},
            width=0.5, legend=False)
    sns.stripplot(data=combined, x="method", y="vif", ax=axes[0],
                  color="black", alpha=0.4, size=4, jitter=True)
    axes[0].axhline(y=5.0, color="red", linestyle="--", alpha=0.7, label="VIF = 5 (moderate)")
    axes[0].axhline(y=10.0, color="darkred", linestyle=":", alpha=0.7, label="VIF = 10 (high)")
    axes[0].set_xlabel("Feature Selection Method", fontweight="bold")
    axes[0].set_ylabel("Variance Inflation Factor (VIF)", fontweight="bold")
    axes[0].set_title("VIF Distribution by Method\n"
                      "(lower VIF = less multicollinearity)",
                      fontweight="bold", pad=10)
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Summary statistics annotation
    for i, (method_name, df_method) in enumerate([
        ("Multi-Objective\n(MORSE)", vif_multi),
        ("Single-Objective\n(AUC-only GA)", vif_single)
    ]):
        if not df_method.empty:
            median_val: float = df_method["vif"].median()
            mean_val: float = df_method["vif"].mean()
            max_val: float = df_method["vif"].max()
            axes[0].annotate(
                f"Median={median_val:.1f}\nMean={mean_val:.1f}\nMax={max_val:.1f}",
                xy=(i, max_val), xytext=(i + 0.3, max_val),
                fontsize=8, color="black", fontstyle="italic",
                ha="center", va="bottom")

    # Right: per-feature VIF comparison (averaged across seeds)
    avg_multi: pandas.DataFrame = vif_multi.groupby("feature")["vif"].mean().reset_index()
    avg_multi.columns = ["feature", "vif_multi"]
    avg_single: pandas.DataFrame = vif_single.groupby("feature")["vif"].mean().reset_index()
    avg_single.columns = ["feature", "vif_single"]

    merged: pandas.DataFrame = pandas.merge(avg_multi, avg_single, on="feature", how="outer").fillna(0.0)
    merged = merged.sort_values("vif_multi", ascending=True)

    y_pos: numpy.ndarray = numpy.arange(len(merged))
    bar_height: float = 0.35

    axes[1].barh(y_pos - bar_height / 2, merged["vif_multi"],
                 bar_height, label="Multi-Objective (MORSE)", color="tab:blue", edgecolor="black")
    axes[1].barh(y_pos + bar_height / 2, merged["vif_single"],
                 bar_height, label="Single-Objective (AUC-only GA)", color="tab:orange", edgecolor="black")
    axes[1].axvline(x=5.0, color="red", linestyle="--", alpha=0.7, label="VIF = 5")
    axes[1].axvline(x=10.0, color="darkred", linestyle=":", alpha=0.7, label="VIF = 10")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(merged["feature"], fontsize=8)
    axes[1].set_xlabel("Mean VIF (averaged across seeds)", fontweight="bold")
    axes[1].set_ylabel("Feature", fontweight="bold")
    axes[1].set_title("Per-Feature VIF Comparison\n"
                      "(features selected by either method)",
                      fontweight="bold", pad=10)
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.suptitle("Multicollinearity Analysis: Variance Inflation Factor (VIF)",
                 fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
