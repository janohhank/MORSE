"""Legacy stress grid: the robustness evaluation the notebook ran before the robustness suite.

The robustness suite (robustness_evaluation.py, docs/robustness.md) is the pipeline's robustness
evaluation. This module keeps the earlier one, unchanged, because its outputs stay useful for comparing
with runs made before the suite (and paper_assets reads them). It scores the final model of every method
and seed on the test set under

  * additive Gaussian noise on the continuous inputs (noise level 0.0 .. 1.0 in steps of 0.1, as a
    fraction of each input's training SD; evaluation_utils.apply_proportional_noise),
  * a covariate shift of the test population: a re-weighting along the first principal component of the
    training inputs, strength -1.0 .. +1.0 in steps of 0.2 (evaluation_utils.covariate_shift_weights),
  * re-draw noise on the 0/1 inputs: a share p = 0.0 .. 1.0 of the cells is re-drawn from the column's
    training prevalence (evaluation_utils.apply_dummy_noise).

Noise and shift form one 2-D grid; the Gaussian-noise line (shift 0) and the covariate-shift line (noise
0.3) are slices of it. AURS (evaluation_utils.compute_aurs) summarises each method's grid in one number.
The metric is the run's main objective (ROC-AUC or PR-AUC).

Known limitations, and the reasons the suite replaced it: every 0/1 input is re-drawn independently,
including availability flags, whose values then no longer match them; AURS averages the two directions
of the shift, so a gain on one side offsets a loss on the other; the Gaussian axis barely moves datasets
that are mostly 0/1 inputs.

Two ways to run it
  * in training_notebook.ipynb, as the optional block after the robustness suite
    (`RUN_LEGACY_STRESS_GRID`);
  * later, on any checkpointed run, without retraining:
        python legacy_stress_evaluation.py --run 2026-09-25_16-05-04_radfusion_pr

Outputs (default: <run>/evaluation/all_models_comparison/, the folder earlier runs used)
  gaussian_2d_per_seed.csv       every seed, noise level and shift strength
  gaussian_noise_per_seed.csv    the slice at shift 0
  covariate_shift_per_seed.csv   the slice at noise level 0.3
  dummy_flip_per_seed.csv        the 0/1 re-draw sweep ("flip" is the historical name)
  aurs_scores.csv                AURS per method
  *_comparison_test.png, gaussian_2d_heatmap_grid_test.png   the four figures
"""
from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy
import pandas

from evaluation_utils import (apply_dummy_noise, apply_proportional_noise, compute_aurs, covariate_shift_weights,
                              evaluate_model, fit_covariate_shift_axis, get_continuous_columns, get_dummy_columns,
                              predict_scores, score_predictions)
from plot_utils import plot_2d_heatmap_grid, plot_noise_comparison
from robustness_evaluation import load_run_models
from training_utils import ensure_directory

NOISE_LEVELS: numpy.ndarray = numpy.arange(0.0, 1.1, 0.1)
SHIFT_LEVELS: numpy.ndarray = numpy.arange(-1.0, 1.1, 0.2)
# Share p of the 0/1 cells that is re-drawn from the column's training prevalence; the "flip" names of
# the columns and files are historical and kept so that code reading earlier runs keeps working.
REDRAW_LEVELS: numpy.ndarray = numpy.arange(0.0, 1.05, 0.1)
# Noise level of the covariate-shift line (a slice of the 2-D grid).
FIXED_NOISE_LEVEL_FOR_SHIFT_SWEEP: float = 0.3
# Display name, colour, marker, line style -- as in the earlier notebook (aurs_scores.csv carries the names).
MODEL_STYLES: dict[str, tuple[str, str, str, str]] = {
    "multi":   ("Multi-Objective (MORSE)",        "tab:blue",   "o", "-"),
    "single":  ("Single-Objective (AUC-only GA)", "tab:orange", "s", "--"),
    "all":     ("All features (no selection)",    "tab:green",  "^", "-."),
    "forward": ("Forward stepwise selection",     "tab:red",    "D", ":"),
}
DEFAULT_FOLDER: str = "all_models_comparison"


@dataclass
class LegacyStressResults:
    output_directory: str
    grid: pandas.DataFrame          # every seed x noise level x shift strength
    gaussian: pandas.DataFrame      # slice at shift 0
    shift: pandas.DataFrame         # slice at noise level FIXED_NOISE_LEVEL_FOR_SHIFT_SWEEP
    redraw: pandas.DataFrame        # the 0/1 re-draw sweep
    aurs: dict[str, float]


def _grid(levels: numpy.ndarray) -> list[float]:
    """Rounded grid values (the "+ 0.0" turns a possible -0.0 into 0.0)."""
    return [round(float(v), 2) + 0.0 for v in levels]


def run_legacy_stress_grid(
        X_train: pandas.DataFrame,
        X_test: pandas.DataFrame,
        y_test: pandas.Series | numpy.ndarray,
        model_packages: dict[int, dict[str, dict]],
        output_directory: str,
        use_roc_auc: bool = True,
        log: Callable[[str], None] = print) -> LegacyStressResults:
    """Run the legacy stress grid on the final models (`robustness_evaluation.build_final_models`) and
    write the tables and figures to `output_directory` (see the module docstring).

    The random draws reproduce the earlier notebook exactly: the global random state is seeded with the
    seed before each seed's sweeps, and the noise is drawn in the same order."""
    metric: str = "ROC-AUC" if use_roc_auc else "PR-AUC"
    seeds: list[int] = sorted(model_packages)
    keys: list[str] = [key for key in MODEL_STYLES if all(key in model_packages[s] for s in seeds)]
    styles: dict[str, tuple[str, str, str, str]] = {key: MODEL_STYLES[key] for key in keys}
    noise_grid, shift_grid, redraw_grid = _grid(NOISE_LEVELS), _grid(SHIFT_LEVELS), _grid(REDRAW_LEVELS)

    continuous_cols: list[str] = get_continuous_columns(X_train)
    dummy_cols: list[str] = get_dummy_columns(X_train)
    train_std: pandas.Series = X_train.std()
    dummy_prevalence: pandas.Series = X_train[dummy_cols].mean()

    # The shift axis (fitted on the training inputs) and the test weights per strength depend only on the
    # data, so they are computed once.
    shift_axis: dict[str, Any] = fit_covariate_shift_axis(X_train)
    shift_weights: dict[float, numpy.ndarray] = {
        sv: covariate_shift_weights(shift_axis, X_test, y_test, sv) for sv in shift_grid}
    log(f"Covariate shift: effective sample size of the re-weighted test set (n = {len(y_test)}):")
    for sv in (shift_grid[0], shift_grid[len(shift_grid) // 2], shift_grid[-1]):
        w: numpy.ndarray = shift_weights[sv]
        log(f"  strength {sv:+.1f}: ESS = {w.sum() ** 2 / (w ** 2).sum():6.1f}, max weight = {w.max():.2f}")

    heatmap_rows: list[dict] = []
    redraw_rows: list[dict] = []
    log(f"Running the legacy stress grid for {len(keys)} methods across {len(seeds)} seeds...")
    for s in seeds:
        random.seed(s)
        numpy.random.seed(s)
        models: dict[str, dict] = {key: model_packages[s][key] for key in keys}
        # 2-D sweep: Gaussian noise level x covariate-shift strength; each prediction is made once per
        # noise level and scored under every shift weighting
        for nv in noise_grid:
            X_noisy: pandas.DataFrame = apply_proportional_noise(X_test, train_std, nv, continuous_cols)
            probs: dict[str, numpy.ndarray] = {key: predict_scores(pkg, X_noisy) for key, pkg in models.items()}
            for sv in shift_grid:
                row: dict = {"seed": s, "noise_level": nv, "mean_shift": sv}
                for key in models:
                    row[f"auc_{key}"] = score_predictions(y_test, probs[key], use_roc_auc=use_roc_auc,
                                                          sample_weight=shift_weights[sv])
                heatmap_rows.append(row)
        # 1-D sweep: re-draw noise on the 0/1 inputs
        for fv in redraw_grid:
            X_noisy = apply_dummy_noise(X_test, fv, dummy_cols, dummy_prevalence)
            row = {"seed": s, "flip_rate": fv}
            for key, pkg in models.items():
                row[f"auc_{key}"] = evaluate_model(pkg, X_noisy, y_test, use_roc_auc=use_roc_auc)
            redraw_rows.append(row)
        log(f"  Seed {s} done (features: "
            + ", ".join(f"{key}={len(models[key]['features'])}" for key in keys) + ")")

    heatmap_df: pandas.DataFrame = pandas.DataFrame(heatmap_rows)
    redraw_df: pandas.DataFrame = pandas.DataFrame(redraw_rows)
    gauss_df: pandas.DataFrame = heatmap_df[numpy.isclose(heatmap_df["mean_shift"], 0.0)].drop(
        columns=["mean_shift"]).reset_index(drop=True)
    shift_df: pandas.DataFrame = heatmap_df[numpy.isclose(heatmap_df["noise_level"], FIXED_NOISE_LEVEL_FOR_SHIFT_SWEEP)
                                            ].drop(columns=["noise_level"]).reset_index(drop=True)

    gauss_agg: pandas.DataFrame = gauss_df.drop(columns=["seed"]).groupby("noise_level").agg(["mean", "std"])
    shift_agg: pandas.DataFrame = shift_df.drop(columns=["seed"]).groupby("mean_shift").agg(["mean", "std"])
    redraw_agg: pandas.DataFrame = redraw_df.drop(columns=["seed"]).groupby("flip_rate").agg(["mean", "std"])
    heatmap_agg: pandas.DataFrame = heatmap_df.drop(columns=["seed"]).groupby(["noise_level", "mean_shift"]).mean()

    # AURS: the average fraction of each method's own clean-test score retained over the whole grid
    # (evaluation_utils.compute_aurs; note its limits there)
    aurs: dict[str, float] = {key: compute_aurs(heatmap_agg, key) for key in keys}

    ensure_directory(output_directory)
    heatmap_df.to_csv(os.path.join(output_directory, "gaussian_2d_per_seed.csv"), index=False)
    gauss_df.to_csv(os.path.join(output_directory, "gaussian_noise_per_seed.csv"), index=False)
    shift_df.to_csv(os.path.join(output_directory, "covariate_shift_per_seed.csv"), index=False)
    redraw_df.to_csv(os.path.join(output_directory, "dummy_flip_per_seed.csv"), index=False)
    pandas.DataFrame([{"model_key": key, "method": styles[key][0], "aurs": aurs[key]} for key in keys]).to_csv(
        os.path.join(output_directory, "aurs_scores.csv"), index=False)

    log(f"\nAURS (Area Under the Robustness Surface) per model -- avg. fraction of clean-test {metric} "
        f"retained over noise_level in [{noise_grid[0]:.1f}, {noise_grid[-1]:.1f}] x covariate-shift strength "
        f"in [{shift_grid[0]:.1f}, {shift_grid[-1]:.1f}]:")
    for key in keys:
        log(f"  {styles[key][0]:35s} AURS = {aurs[key]:.1%}")

    ylabel: str = f"Test {metric} (mean across seeds; shaded = +/- 1 std)"
    plot_noise_comparison(
        agg_df=gauss_agg, model_styles=styles,
        xlabel="Gaussian Noise Level (fraction of training std, no covariate shift)", ylabel=ylabel,
        title=("Robustness on Test Set: Additive Gaussian Noise on Continuous Features\n"
               "(all 4 models, averaged across seeds)"),
        out_path=os.path.join(output_directory, "gaussian_noise_comparison_test.png"))
    plot_noise_comparison(
        agg_df=shift_agg, model_styles=styles,
        xlabel=(f"Covariate-shift strength (test population shifted along its dominant axis, in SD; "
                f"fixed Gaussian noise level = {FIXED_NOISE_LEVEL_FOR_SHIFT_SWEEP})"),
        ylabel=ylabel,
        title=("Robustness on Test Set: Covariate Shift of the Test Population (re-weighted)\n"
               f"(fixed Gaussian noise level = {FIXED_NOISE_LEVEL_FOR_SHIFT_SWEEP}, "
               "all 4 models, averaged across seeds)"),
        out_path=os.path.join(output_directory, "covariate_shift_comparison_test.png"))
    plot_noise_comparison(
        agg_df=redraw_agg, model_styles=styles,
        xlabel="Noise fraction p (share of dummy cells re-drawn from the column's training prevalence)",
        ylabel=ylabel,
        title=("Robustness on Test Set: Prevalence-Preserving Noise on Binary (Dummy) Features\n"
               "(all 4 models, averaged across seeds)"),
        out_path=os.path.join(output_directory, "dummy_flip_comparison_test.png"))
    plot_2d_heatmap_grid(
        heatmap_agg=heatmap_agg, model_styles=styles, cbar_label=f"Test {metric}",
        xlabel="Gaussian Noise Level (fraction of training std)",
        ylabel="Covariate-shift strength (SD along the dominant axis)",
        title=("Robustness on Test Set: 2D sweep (Gaussian noise x covariate shift)\n"
               f"per-cell mean {metric} across seeds; shared colour scale for direct comparison\n"
               "per-panel AURS = avg. fraction of that model's own clean-test score "
               "retained across the whole grid (see evaluation_utils.compute_aurs)"),
        out_path=os.path.join(output_directory, "gaussian_2d_heatmap_grid_test.png"),
        aurs_scores=aurs)
    log(f"\nLegacy stress grid: plots and CSVs saved to {output_directory}")
    return LegacyStressResults(output_directory=output_directory, grid=heatmap_df, gaussian=gauss_df,
                               shift=shift_df, redraw=redraw_df, aurs=aurs)


# ---------------------------------------------------------------------------
# Stand-alone use on a finished run
# ---------------------------------------------------------------------------

def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the legacy stress grid (noise x covariate shift, 0/1 re-draw noise, AURS) on the "
                    "final models of a checkpointed run (no retraining).")
    parser.add_argument("--run", required=True,
                        help="the run's result folder (absolute, or relative to the repository root)")
    parser.add_argument("--selection", choices=("auto", "knee", "max_s"), default="auto",
                        help="which Pareto solution is MORSE's final model; auto = the run's own rule")
    parser.add_argument("--metric", choices=("auto", "roc", "pr"), default="auto",
                        help="ROC-AUC or PR-AUC; auto = the run's main objective")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="evaluate only these seeds (default: every seed with a complete checkpoint)")
    parser.add_argument("--notebook", default=None,
                        help="a copy of training_notebook.ipynb to rebuild the data from, for a run folder without "
                             "an archived copy of the notebook (checked against the checkpoints)")
    parser.add_argument("--out", default=None,
                        help=f"output folder (default: <run>/evaluation/{DEFAULT_FOLDER}, with the rule / metric "
                             f"appended when they differ from the run's own)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    arguments: argparse.Namespace = parse_arguments(argv)
    run = load_run_models(arguments.run, arguments.selection, arguments.seeds, arguments.notebook)
    use_roc_auc: bool = run.use_roc_auc if arguments.metric == "auto" else arguments.metric == "roc"
    suffix: str = run.folder_suffix + ("" if use_roc_auc == run.use_roc_auc else ("_roc" if use_roc_auc else "_pr"))
    output: str = arguments.out or os.path.join(run.directory, "evaluation", DEFAULT_FOLDER + suffix)
    started: float = time.time()
    run_legacy_stress_grid(run.X_train, run.X_test, run.y_test, run.packages, output_directory=output,
                           use_roc_auc=use_roc_auc)
    print(f"Legacy stress grid finished in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
