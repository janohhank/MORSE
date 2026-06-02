# MORSE

**M**ulti-objective **O**ptimization with **R**obust **S**election of **E**xplanatory variables.

A reference implementation and full empirical study of a **sign-consistency-aware
feature selection framework** for logistic regression, built on the NSGA-II
multi-objective evolutionary algorithm. MORSE accepts the AUC trade-off that
classical feature selection methods optimise alone and adds a structural
robustness objective: the **sign agreement between each feature's marginal
correlation with the target and its fitted regression coefficient**. The two
objectives are co-optimised over the space of binary feature-subset
indicators, producing a Pareto front of (AUC, sign-consistency) trade-offs
from which a single deployment model is then chosen — for example via the
**knee point** of the front.

---

## Why this matters: the problem MORSE addresses

A logistic regression coefficient `β_k` represents the partial effect of
feature `k` on the log-odds of the target, conditional on all other features
in the model. In datasets with **suppressor variables**, **strong
multicollinearity**, or **small samples** relative to the feature count, the
fitted sign of `β_k` can flip relative to the sign of the feature's marginal
association with the target. The model achieves a high in-sample AUC by
relying on a delicate statistical cancellation between correlated noise
predictors and the genuine signal carriers; the price is a brittle model.

Such sign-inconsistent models tend to:

- Generalise poorly under **covariate shift** (out-of-distribution data).
- Degrade rapidly under **moderate test-time noise**, because the cancellation
  that propped up the in-sample AUC is destroyed.
- Conflict with **domain knowledge**: a clinician or financial analyst can
  inspect the fitted coefficients and immediately spot the implausible signs.

MORSE's hypothesis — empirically tested across three clinical / risk-scoring
datasets in this repository — is that **explicitly penalising sign
inconsistency during feature selection yields more parsimonious models with
better robustness to noise**, at a modest cost in clean-test AUC.

---

## Method, in one paragraph

For every candidate binary feature mask `m ∈ {0, 1}^p` the GA evaluates two
fitness values on stratified 3-fold cross-validation of the training set:

1. **Predictive performance** — mean ROC-AUC (or PR-AUC) of an L2-penalised
   logistic regression fit on each fold's training partition and scored on the
   fold's validation partition.
2. **Sign consistency** — for the fitted coefficients `β_k` of the selected
   features, the fraction whose product `corr(x_k, y) · β_k` is strictly
   positive (computed per fold on the fold's training partition; the
   negative-or-near-zero fraction is the penalty).

NSGA-II is then run over the population of feature masks. From each seed's
final Pareto front, the deployable model is selected by the **knee-point**
heuristic (maximum perpendicular distance from the line connecting the two
extreme candidates of the Pareto front). The final logistic regression is
**refit on the full training set** with the selected feature subset for
downstream evaluation.

The repository compares MORSE against three baselines: a single-objective GA
(AUC only), forward stepwise selection via scikit-learn, and the no-selection
all-features logistic regression. Every method is evaluated on **20 random
seeds**, on both clean and progressively noised versions of the test set
(Gaussian noise on continuous features, random binary corruption on dummy
features). Statistical significance of the pairwise differences is reported
via **paired Wilcoxon signed-rank tests** on the per-seed AUCs.

---

## Repository layout

```
.
├── training_notebook.ipynb          # Single end-to-end pipeline notebook
│
├── multi_objective_training.py      # NSGA-II GA with AUC + sign-consistency objectives
├── single_objective_training.py     # Baseline AUC-only GA
├── training_config.py               # GA hyperparameter dataclass
├── training_utils.py                # Plotting, Pareto-front selection helpers
├── evaluation_utils.py              # Noise injection, model evaluation, VIF, stability
│
├── arrhythmia/                      # UCI Arrhythmia dataset + preparation notebook
│   ├── arrhythmia.data
│   ├── arrhythmia.names
│   └── arrhythmia_data_preparation.ipynb
│
└── readmit/                         # UCI Diabetes 130-US Hospitals dataset
    ├── diabetic_data.csv
    └── readmit_130_hospitals_data_preparation.ipynb
```

A third dataset (**RadFusion** — Electronic Health Records and CTPA imaging
for pulmonary-embolism detection) is referenced in the notebook configuration
but **not redistributed** here because of its access-controlled licence; see
the corresponding selector lines in `training_notebook.ipynb` (cell 2) for
the path placeholders.

### Key modules at a glance

| Module | Responsibility |
|---|---|
| `multi_objective_training.MultiObjectiveTraining` | NSGA-II loop with the dual AUC + sign-consistency fitness. Uses DEAP's `selNSGA2` for the multi-objective survival selection. |
| `single_objective_training.SingleObjectiveTraining` | Single-objective AUC-only GA (DEAP `eaMuPlusLambda`) used as the SO-GA baseline. |
| `training_utils` | Selection helpers (`knee_point_index`, `best_sign_consistency_index`, `best_auc_index`, `select_pareto_individual`), convergence plots, and the Pareto-front visualisation that highlights the three canonical selection candidates. |
| `evaluation_utils` | `build_model_package` (final-model retraining on full train), `apply_proportional_noise` (Gaussian + mean-shift), `apply_dummy_noise` (random binary corruption), Jaccard stability, VIF analysis, all per-seed plots and CSV writers. |

---

## How to reproduce

### Prerequisites

- Python 3.10+
- The standard scientific stack:
  - `numpy`, `pandas`, `scipy`
  - `scikit-learn`
  - `matplotlib`, `seaborn`
  - `statsmodels` (for the VIF analysis)
  - `deap` (the evolutionary algorithms library)
  - `joblib` (seed-level parallelism via the loky backend)

A reasonably fast multi-core workstation is recommended: with `N_JOBS=-1` on
an 8-core machine the full pipeline (20 seeds × NSGA-II + SOGA + SFS + noise
sweeps) runs in roughly 30–90 minutes per dataset, depending on the feature
count.

### Run the pipeline

1. Open `training_notebook.ipynb` in Jupyter (or VS Code's notebook UI).
2. In cell 2, **uncomment exactly one dataset config block**:
   - `arrhythmia` (default), `readmit`, or the synthetic dataset placeholder.
3. Optionally edit `N_JOBS` (default `-1`) and `USE_KNEE_POINT_SELECTION`
   (default `True`).
4. **Run all cells**. Outputs are written to a timestamped directory in the
   working folder, with the structure:

```
YYYY-MM-DD_HH-MM-SS/
├── multi/seed_<N>/                  # NSGA-II convergence + Pareto-front PNGs/CSVs
├── single/seed_<N>/                 # Single-objective convergence
└── evaluation/
    ├── seed_<N>/                    # Per-seed noise-robustness plots and CSVs
    ├── stability/                   # Jaccard similarity heatmap, feature frequency
    ├── vif/                         # Multicollinearity analysis (VIF distributions)
    ├── feature_comparison/          # Only-in-MORSE vs only-in-SOGA feature lists
    ├── baseline_all_features/       # Clean-AUC table for the all-features baseline
    ├── baseline_forward_stepwise/   # Same for forward stepwise selection
    ├── all_models_comparison/       # 4-model noise robustness curves
    ├── wilcoxon/                    # Paired Wilcoxon p-values per noise level
    ├── feature_counts/              # Boxplot of selected feature counts per method
    └── sign_consistency_verification/  # Fold-vs-full-train sign-consistency gap
```

### Numerical reproducibility

The pipeline is fully reproducible up to BLAS-thread-count effects:

- The notebook's first cell caps `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OMP_NUM_THREADS`, and `BLIS_NUM_THREADS` to 1 **before** `numpy` is
  imported, so the per-worker BLAS pool does not fight the `joblib` loky
  worker pool over cores.
- All `LogisticRegression` instances use `random_state=seed`; both the GA and
  SFS receive `StratifiedKFold(random_state=seed)`.
- `N_JOBS` does not affect numerical results — each seed is self-contained
  and seeds do not share mutable state across workers.

---

## What the notebook delivers

Per dataset, the end-to-end run produces the following analysis artefacts.
All are described inline in the notebook with the markdown headers above the
corresponding code cells:

- **Per-seed Pareto front** (`pareto_front.png`) with the three canonical
  selection candidates highlighted (knee point as a red star, best
  sign-consistency as an orange diamond, best AUC as a green triangle).
- **Noise-robustness sweeps** for each of the four compared methods (MORSE,
  SO-GA, no-selection, SFS) on the test set under (a) additive Gaussian
  noise on continuous features and (b) random corruption on binary dummy
  features. Plots are shown as mean ± 1 std across seeds.
- **Feature-selection stability** (pairwise Jaccard similarity across seeds)
  with a side-by-side heatmap of MORSE versus SO-GA.
- **VIF distribution** for MORSE-selected versus SO-GA-selected features,
  with a boxplot and summary statistics — empirical evidence that the
  sign-consistency objective tends to suppress multicollinear features.
- **Feature-set comparison** (only-in-MORSE / only-in-SO-GA / common) per
  seed, saved as CSV.
- **Sign-inconsistency report** for the SO-GA model — which selected
  features end up with coefficient signs opposing their marginal correlation
  with the target. This is the failure mode MORSE explicitly avoids.
- **Paired Wilcoxon signed-rank tests** between every pair of methods at
  every noise level, with raw p-values and a 0.05-significance flag.
- **Feature-count comparison** boxplot showing the parsimony of each method.
- **Fold-vs-full-train sign-consistency verification**: a sanity check that
  the per-fold sign-consistency the GA optimised translates to the final
  full-train refit (mean gap and per-seed deltas).

---

## Open science contribution

This repository is released as a **fully open contribution to the
open-science community**:

- **Code**: all source, all configuration, and all evaluation logic are
  released under the repository's open licence — no closed-source
  components, no proprietary dependencies beyond standard Python scientific
  libraries.
- **Datasets**: the two redistributable datasets used in the empirical study
  (Arrhythmia and Diabetes 130-US Hospitals) are included alongside their
  preprocessing notebooks. The third dataset (RadFusion) is access-controlled
  but the access pathway is documented in the original publication; the
  notebook is wired to consume it identically.
- **Reproducibility**: the timestamped output directory captures every plot,
  every CSV, and every per-seed model used to produce the published numbers.
  Re-running the notebook on the same dataset configuration reproduces the
  published results within floating-point precision.
- **Methodology transparency**: every methodological choice (knee-point
  selection, 3-fold inner CV, 20 random seeds for paired statistical tests,
  fixed train/test split versus seed variation) is documented in inline
  markdown cells, including the trade-offs that were considered and the
  alternatives that were rejected.

Issues, replication attempts, and extensions are very welcome. If you reuse
the sign-consistency objective or the MORSE pipeline in your own work,
please cite the corresponding paper (forthcoming) and the datasets below.

---

## AI tool disclosure

Large-language-model assistance (**Claude Opus** model family, by
Anthropic) was used during the development of this repository for code
prototyping, debugging support, plotting helpers, and the writing of the
documentation in this README. All methodological choices, hyperparameter
selections, empirical validation, and final scientific conclusions are the
work of the human authors. The pipeline produces deterministic numerical
output that is independent of any AI-generated content.

---

## Dataset citations

### Arrhythmia (UCI Machine Learning Repository, 1998)

> Guvenir, H. A., Acar, B., Demiroz, G., & Cekin, A. (1997). A supervised
> machine learning algorithm for arrhythmia analysis. In *Computers in
> Cardiology 1997* (pp. 433–436). IEEE.
> https://doi.org/10.1109/CIC.1997.647926

> Guvenir, H. A. (1998). *Arrhythmia* [Dataset]. UCI Machine Learning
> Repository. https://doi.org/10.24432/C5BS32

### Diabetes 130-US Hospitals for Years 1999-2008 — "Readmit" (UCI Machine Learning Repository, 2014)

> Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S.,
> Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on
> hospital readmission rates: analysis of 70,000 clinical database patient
> records. *BioMed Research International*, 2014, Article 781670.
> https://doi.org/10.1155/2014/781670

> Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S.,
> Cios, K. J., & Clore, J. N. (2014). *Diabetes 130-US Hospitals for Years
> 1999-2008* [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5230J
