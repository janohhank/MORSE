# MORSE

**M**ulti-objective **O**ptimization with **R**obust **S**election of **E**xplanatory variables.

A reference implementation and empirical study of a **sign-consistency-aware
feature selection framework** for logistic regression, built on the NSGA-II
multi-objective evolutionary algorithm. MORSE accepts the AUC trade-off that
classical feature selection methods optimise alone and adds a structural
robustness objective: the **sign agreement between each feature's marginal
correlation with the target and its fitted regression coefficient**. The two
objectives are co-optimised over the space of binary feature-subset
indicators, producing a Pareto front of (AUC, sign-consistency) trade-offs
from which the final model is then chosen: the **max-S end** of the front
(its most sign-consistent solution, the default) or its **knee point**.

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

MORSE's hypothesis — empirically probed on the clinical, risk-scoring and
higher-education datasets in this repository — is that **explicitly
penalising sign inconsistency during feature selection yields more
parsimonious models that are more robust to test-time perturbations and to
distribution shift**, possibly at some cost in clean-test AUC.

---

## Method, in one paragraph

For every candidate binary feature mask `m ∈ {0, 1}^p` the GA evaluates two
fitness values on stratified 3-fold cross-validation of the training set:

1. **Predictive performance** — mean AUC of an L2-penalised logistic
   regression fit on each fold's training partition and scored on the fold's
   validation partition. The inputs are standardised inside every fold, with
   the mean and SD of the fold's training partition, so no validation row
   influences the scaling its fold model is fitted with (the same for all
   three searches). The AUC variant (ROC-AUC or PR-AUC) is configurable;
   the shipped notebook uses **ROC-AUC** (`USE_ROC_AUC = True`), which does
   not depend on the class prevalence (PR-AUC suits strongly imbalanced tasks
   such as the readmission dataset).
2. **Sign consistency** — for the fitted coefficients `β_k` of the selected
   features, the fraction whose product `corr(x_k, y) · β_k` is strictly
   positive (computed per fold on the fold's training partition; the
   negative-or-near-zero fraction is the penalty).

NSGA-II (via DEAP) is run over the population of feature masks. From each
seed's final Pareto front, the final model is the **max-S end** — the
solution with the best sign consistency, which, being non-dominated, is also
the most accurate among the most consistent solutions
(`USE_KNEE_POINT_SELECTION = False`, the default) — or the **knee point**
(maximum perpendicular distance from the line connecting the two extreme
candidates of the normalised front; `USE_KNEE_POINT_SELECTION = True`). The
rule only picks a solution of the finished front and does not affect
training, so a finished run can be evaluated with the other rule without
retraining. The final logistic regression is **refit on the full training
set** (with a fresh `StandardScaler`) on the selected feature subset for
downstream evaluation.

The repository compares MORSE against three baselines: a single-objective GA
(AUC only), forward stepwise selection via scikit-learn, and the no-selection
all-features logistic regression. Every method is evaluated across **20
random seeds** on a **single fixed test set**, clean and under the
**robustness suite**: test-set stresses generated automatically from the data,
without naming any feature (re-weighted test populations — principal-component
and dependence shifts — and corrupted test sets — measurement noise, recording
noise, under-recording, values not available, numerical values masked; see
[docs/robustness.md](docs/robustness.md)). On College Scorecard the final
models are also scored on a **real out-of-domain test set**, the institution
types that TableShift holds out (see
[the natural shift](#the-natural-distribution-shift-of-college-scorecard)).
Results are reported as **mean ± 1 standard deviation across seeds**, with
paired tests over the seeds. See
[Numerical reproducibility](#numerical-reproducibility) for exactly what the
seed varies — this is a deliberate design choice.

---

## Repository layout

```
.
├── training_notebook.ipynb          # Single end-to-end pipeline notebook
│
├── multi_objective_training.py      # NSGA-II GA: AUC + sign-consistency objectives
├── single_objective_training.py     # Baseline AUC-only GA (eaMuPlusLambda)
├── forward_stepwise_training.py     # Baseline: sklearn SequentialFeatureSelector
├── all_features_training.py         # Baseline: no selection (all-ones mask)
├── training_config.py               # GA hyperparameter dataclass
├── training_utils.py                # CSV writer + Pareto-front selection helpers
├── plot_utils.py                    # All figures (convergence, Pareto, robustness, etc.)
├── evaluation_utils.py              # Marginal correlations, model build/eval, balanced
│                                    #   sens/spec threshold (+ the stress tests of older runs)
├── robustness_config.py             # Robustness-suite settings (the same for every dataset)
├── robustness_utils.py              # Automatic feature schema, re-weightings, corruption bank
├── robustness_evaluation.py         # Robustness suite: notebook entry point + stand-alone CLI
├── legacy_stress_evaluation.py      # Legacy stress grid (noise x PC1 shift, 0/1 re-draw, AURS):
│                                    #   optional notebook block + stand-alone CLI
├── docs/robustness.md               # The robustness suite in detail
├── checkpoint_utils.py              # Per-seed training checkpoints (Pareto fronts, masks)
│                                    #   and resuming a run
├── run_manifest.py                  # run_manifest.json: what a run was trained on; rebuilding its data
├── deap_types.py                    # The DEAP fitness / individual classes (one definition)
├── requirements.txt                 # Pinned dependency set
├── tests/                           # Unit tests: python -m unittest discover -s tests -v
│
├── arrhythmia/                      # UCI Arrhythmia dataset + preparation notebook
│   ├── arrhythmia.data
│   ├── arrhythmia.names
│   ├── arrhythmia_data_preparation.ipynb
│   ├── arrhythmia_preprocessed_train_data.csv
│   └── arrhythmia_preprocessed_test_data.csv
│
├── college_scorecard/               # U.S. Department of Education College Scorecard (TableShift task)
│   ├── CollegeScorecardDataDictionary.xlsx
│   ├── college_scorecard_data_preparation.ipynb
│   ├── college_scorecard_ood_evaluation.py         # natural-shift evaluation of a finished run
│   ├── college_scorecard_preprocessed_train_data.csv
│   ├── college_scorecard_preprocessed_test_data.csv
│   └── college_scorecard_preprocessed_ood_test_data.csv  # the held-out institution types
│
├── radfusion/                       # RadFusion EHR preprocessing (the data are not redistributed)
│   └── radfusion_ehr_preprocess.ipynb
│
└── readmit/                         # UCI Diabetes 130-US Hospitals dataset
    ├── diabetic_data.csv
    ├── readmit_130_hospitals_data_preparation.ipynb
    ├── readmit_130_hospitals_preprocessed_train_data.csv
    └── readmit_130_hospitals_preprocessed_test_data.csv
```

**College Scorecard** is public-domain U.S. government data. Its raw download
(`Most-Recent-Cohorts-Institution.csv`, about 100 MB) is not committed; the
preparation notebook names the file and its download link, and the three
preprocessed CSVs it produces are committed. The task definition (119
institutional features; the label is `C150_4 > 0.5`, i.e. more than half of
the first-time, full-time students complete their studies within 150% of the
normal time) and the out-of-domain split (eight Carnegie Classification types
held out) follow TableShift. The preprocessing is MORSE's own — training-only
median imputation with missing-value flags and one-hot codes instead of
TableShift's binning for tree models — so that the signs of the inputs stay
interpretable.

**RadFusion** (the electronic health records of a CT pulmonary angiography
cohort, pulmonary-embolism detection) is **not redistributed** because of its
access-controlled licence. Its preprocessing notebook is included: with the
raw EHR tables obtained through the dataset's official access pathway, it
writes the CSVs that the RadFusion block of the notebook's cell 2 reads, and
cells 3 and 4 then select the RadFusion inputs with an explicit column list.
Two ICD groups in that list may encode the outcome itself:
`DISEASES OF PULMONARY CIRCULATION:presence` contains the codes of pulmonary
embolism, and `DISEASES OF VEINS AND LYMPHATICS, AND OTHER DISEASES OF
CIRCULATORY SYSTEM:presence` those of venous thrombosis. The MORSE
experiments exclude both, so remove them from the two column lists before a
RadFusion run.

### Key modules at a glance

| Module | Responsibility |
|---|---|
| `multi_objective_training.MultiObjectiveTraining` | NSGA-II loop with the dual AUC + sign-consistency fitness. Uses DEAP's `selNSGA2` survival selection and `selTournamentDCD` mating selection; per-fold marginal correlations are computed internally on each fold's training partition. |
| `single_objective_training.SingleObjectiveTraining` | Single-objective AUC-only GA (DEAP `eaMuPlusLambda`) used as the SO-GA baseline. |
| `forward_stepwise_training.ForwardStepwiseTraining` | Forward stepwise baseline wrapping sklearn's `SequentialFeatureSelector` (a `StandardScaler` + logistic-regression pipeline, so every CV fold is standardised on its own training rows). |
| `all_features_training.AllFeaturesTraining` | No-selection baseline: returns the all-ones mask through the same `.run()` shape. |
| `training_utils` | `standardised_folds` (the CV folds of the two GAs, each standardised on its own training rows), `save_stats_csv`, `ensure_directory`, `repository_root` (the folder every result directory is created in), and the Pareto-front selection helpers (`knee_point_index`, `best_sign_consistency_index`, `best_auc_index`, `select_pareto_individual`). |
| `plot_utils` | Every figure: single/multi-objective convergence, the Pareto front (highlighting the three canonical candidates), the five robustness-suite figures, the legacy stress grid's line plots and heatmap grid, the sensitivity/specificity curve, the feature-count boxplot, and the sign-consistency boxplot. |
| `evaluation_utils` | `compute_marginal_correlations` (Matthews / point-biserial), `build_model_package` (final refit on full train), `predict_scores` / `score_predictions` / `evaluate_model` (optionally weighted metrics), `compute_model_sign_consistency` (sign consistency of a final model), `find_balanced_threshold`, and `select_deployment_model` / `out_of_fold_scores` (the best MORSE model and its threshold, chosen without the test set). The stress tests of the legacy stress grid (`apply_proportional_noise`, `apply_dummy_noise`, `fit_covariate_shift_axis` / `covariate_shift_weights`, column-type detection) and its summary `compute_aurs`. |
| `robustness_config` | `RobustnessConfig`: every setting of the robustness suite (severity levels, pair eligibility, support thresholds, corruption levels and bank size), the same for every dataset. |
| `robustness_utils` | The robustness suite's building blocks: weighted ROC-AUC / average precision and effective sample size, `infer_feature_schema` (types, one-hot groups, values with their availability flags — by name first, statistically for shared flags — and, as a diagnostic, 0/1 combinations never seen in training), `population_axes` and `dependence_pairs` / `DependenceTilt` (re-weighted test populations, calibrated by `calibrate_strengths` to a training ESS), and `CorruptionBank` (a fixed bank of corrupted test sets with common random numbers: measurement noise, recording noise, under-recording, values not available, numerical values masked). |
| `robustness_evaluation` | `build_final_models` and `run_robustness_suite` (every final model under every scenario; tables, tests, figures, report), `load_run_models` (a finished run rebuilt from its checkpoints), plus the stand-alone entry point `python robustness_evaluation.py --run <result folder>`. See [docs/robustness.md](docs/robustness.md). |
| `legacy_stress_evaluation` | `run_legacy_stress_grid`: the robustness evaluation of runs made before the suite (Gaussian noise × PC1 covariate-shift grid, 0/1 re-draw noise, AURS), unchanged — it reproduces those runs' CSVs byte for byte — as an optional notebook block and as `python legacy_stress_evaluation.py --run <result folder>`. |
| `run_manifest` | `run_manifest.json`, written when a run starts (data files and hashes, input order, objective, Pareto rule, settings), and `load_run_data`, which rebuilds a finished run's data from it — or from a notebook copy for older runs — plus `verify_training_data` against the checkpoint fingerprint. |
| `checkpoint_utils` | Checkpointing of the training stage: `TrainingCheckpointStore` (one folder per finished seed with the MORSE Pareto front as CSV, the SO-GA / SFS / all-features masks, a completion marker, and a fingerprint of the data and settings), `train_missing_seeds` (trains only the seeds without a checkpoint and saves each seed the moment it finishes), and atomic-write helpers. |
| `deap_types` | The DEAP fitness / individual classes (`FitnessMulti` / `Individual`, `FitnessSingle` / `IndividualSingle`), defined once for the trainers, the notebook and the checkpoint loader. |
| `college_scorecard/college_scorecard_ood_evaluation` | The natural-shift evaluation of a College Scorecard run ([see below](#the-natural-distribution-shift-of-college-scorecard)): every final model and every front solution, refit from the checkpoints, on the in-domain test set and on the held-out institution types, with paired tests over the seeds and bootstrap intervals over the institutions. |

---

## How to reproduce

### Prerequisites

- **Python 3.12+** for the pinned stack below (`numpy 2.5` and `scipy 1.18`
  require it; the code itself only uses Python 3.10+ syntax such as
  `X | None` unions and builtin generics).
- Install the pinned scientific stack — the exact versions the reported runs
  were produced with:

```bash
python -m pip install -r requirements.txt
```

This installs `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`,
`seaborn`, `deap`, and `joblib` — the only third-party packages the code
imports — plus `moocore`, which `deap` depends on.

A reasonably fast multi-core workstation is recommended: with `N_JOBS=-1` the
full pipeline (20 seeds × NSGA-II + SO-GA + SFS + the robustness suite) runs in
roughly 30–90 minutes per dataset, depending on the feature count.

### (Optional) regenerate the preprocessed CSVs

The preprocessed CSVs of Arrhythmia, College Scorecard and Diabetes 130-US
are committed, so you can run the training notebook directly. To regenerate
them from the raw data, run the per-dataset preparation notebook
(`arrhythmia/arrhythmia_data_preparation.ipynb`,
`college_scorecard/college_scorecard_data_preparation.ipynb` after
downloading the raw file it names, or
`readmit/readmit_130_hospitals_data_preparation.ipynb`); RadFusion's
(`radfusion/radfusion_ehr_preprocess.ipynb`) needs the access-controlled raw
tables. All of them fit the data-dependent preprocessing (e.g. median
imputation) on the **training partition only**, after the train/test split,
so no test-set information leaks into the training data.

### Run the pipeline

1. Open `training_notebook.ipynb` in Jupyter (or VS Code's notebook UI).
2. In **cell 2**, select the dataset by (un)commenting its config block:
   `arrhythmia`, `readmit`, `college_scorecard` or `radfusion` (the committed
   notebook has the RadFusion block active; see the RadFusion note above).
3. Optionally edit the run switches in the same cell:
   - `N_JOBS` (default `-1`, all cores; one worker process per seed),
   - `USE_KNEE_POINT_SELECTION` (default `False` → the max-S end of every
     front; `True` → the knee point),
   - `USE_ROC_AUC` (default `True` → ROC-AUC is the main objective/metric;
     `False` → PR-AUC),
   - `RESUME_FROM` (default `None`, which starts a **new** run; the result
     directory of an earlier run to continue or re-evaluate it without
     retraining, see [Checkpoints](#checkpoints-and-resuming-a-run)).

   The GA settings (population size 188, 650 generations, crossover and
   mutation probabilities) are the defaults of `training_config.py`; edit them
   there. A resumed run must use the settings its checkpoints record (for
   example `ngen = 1000` for a run that was trained with 1000 generations).
4. **Run all cells.** Outputs are written to a timestamped directory in the
   **repository root** — whatever working directory the kernel was started in
   (the path is printed at the start of the run). Run `training_notebook.ipynb`
   from the repository root; a copy of the notebook archived inside a result
   folder contains the code of its time and does not get later changes:

```
YYYY-MM-DD_HH-MM-SS/
├── run_manifest.json                     # data files + hashes, input order, objective, Pareto rule, settings
├── multi/seed_<N>/                       # convergence.csv, convergence.png, pareto_front.png
├── single/seed_<N>/                      # convergence.csv, convergence.png
├── checkpoints/training/                 # fingerprint.json + seed_<N>/ (morse_front.csv, selections.json, complete.json)
└── evaluation/
    ├── final_models_<rule>.json          # the fitted final models (coefficients, scalers) of the Pareto rule
    ├── robustness/                       # robustness suite: report, CSV tables, tests, five figures
    ├── all_models_comparison/            # legacy stress grid (optional block): per-seed CSVs, AURS, four figures
    ├── feature_counts/                   # feature-count boxplot + per-seed / summary CSVs
    ├── sign_consistency/                 # sign consistency of the 4 final models: boxplot + per-seed / summary CSVs
    ├── best_morse_model/                 # the model chosen without the test set: metrics CSV + curve PDF
    └── ood_test/                         # College Scorecard only: the natural-shift evaluation (separate script)
```

### Checkpoints and resuming a run

Training is the slow part (tens of minutes for 20 seeds); the evaluation after
it takes minutes. Every seed is therefore written to
`<run>/checkpoints/training/seed_<N>/` **the moment the seed is done** (by the
worker that trained it). Forward stepwise selection and the all-features model
do not depend on the seed, so they are computed once and saved as
`shared_baselines.json` in the checkpoint folder; each seed's checkpoint
contains a copy:

- `morse_front.csv` — every Pareto individual of MORSE: AUC, sign
  consistency, feature count and the feature mask as a 0/1 string (in the
  feature order of the dataset),
- `selections.json` — the SO-GA best individual (mask + fitness), the SFS
  mask and the all-features mask,
- `complete.json` — written last; a seed without it counts as not trained, so
  a crash in the middle of a write can never leave a checkpoint that only
  looks complete.

Everything is plain CSV/JSON, written atomically: the files are readable,
usable for the paper (fronts, masks) and independent of library versions.
`fingerprint.json` records the training configuration, the cross-validation
settings, how the inputs are standardised (`"standardisation": "per_fold"`:
the trainers receive the unscaled training data and standardise every fold
themselves), the feature names and hashes of the training data. The fitted
logistic-regression models are rebuilt from the masks in seconds rather than
pickled; their coefficients and scalers are recorded once in
`evaluation/final_models_<rule>.json`, and every later re-evaluation must
reproduce that record (a newer library or code that fits differently stops
with an explanation).

Before training starts, the notebook also writes **`run_manifest.json`**: the
CSV files the training and test data were read from (with SHA-256 hashes —
found by trying the file names of the configuration cells and keeping the list
that reproduces the data exactly), the target and the input order, fingerprints
of the training and test arrays, the main objective, MORSE's Pareto rule, the GA
settings, the CV and the seeds. With it, a finished run can be evaluated again
without a copy of the notebook. Resuming a run with other data is refused.

To continue after a failure (a crash in the evaluation, an interrupted
training, a lost kernel): restart the kernel, set
`RESUME_FROM = "<result directory>"` (a folder name in the repository root,
or an absolute path) in cell 2 and run all cells. Training then loads the seeds
that have a checkpoint and trains only the missing ones (reusing the saved
SFS / all-features result);
the evaluation runs again into the same directory. Resuming with different
data or settings raises `CheckpointMismatchError` (listing what differs)
instead of mixing results, while a change of the algorithm source code only
warns. Runs made before 2026-09-28 standardised the whole training set once,
before the search, so the validation rows of every fold took part in its
scaling: their checkpoints cannot be resumed with the current code (their
fingerprint has no `standardisation` entry), but they can still be evaluated
again — the scripts below recognise them, check their data the way they were
trained and re-score their fronts that way. Runs made before checkpointing
existed cannot be resumed — unless their
kernel is still alive: create the store exactly as in cell 9 and call
`import_results(training_store, seeds, training_results_multi,
training_results_single, training_results_fwd, training_results_all)` to write
the results that are in memory to disk.

### Re-running the robustness suite on a finished run

The robustness suite needs only the checkpoints, so it can be run again — with
changed settings or a newer version of the suite — without retraining:

```bash
python robustness_evaluation.py --run <result folder>
```

The run's data are rebuilt from its `run_manifest.json` (the recorded CSV
files, which must still give the recorded arrays) — or, for runs made before
the manifest existed, from the configuration and data-loading cells of the
notebook copy archived in the run folder (or of the copy given with
`--notebook`) — and checked against the run's checkpoint fingerprint; the final
models are refit from the saved masks and checked against
`evaluation/final_models_<rule>.json`. The outputs go to
`<run>/evaluation/robustness/`; an evaluation with a Pareto rule other than the
run's own (`--selection knee` / `max_s`) goes to `robustness_<rule>/`, and a
folder holding an evaluation of other data or of another rule is never
overwritten (`--out` for another folder; `--seeds` is described by `--help`).
Every stage's tables are saved as soon as the stage is done, and `config.json`
records the status, the selection rule, the data fingerprints and the hashes
of the manifest and the checkpoint fingerprint. See
[docs/robustness.md](docs/robustness.md).

The legacy stress grid works the same way:

```bash
python legacy_stress_evaluation.py --run <result folder>
```

writes `<run>/evaluation/all_models_comparison/` (`..._<rule>` / `..._roc` /
`..._pr` when the rule or the metric differs from the run's own).

### The natural distribution shift of College Scorecard

The 945 institutions of the Carnegie Classification types that TableShift
holds out are never used during training or selection. A finished College
Scorecard run is scored on them without retraining:

```bash
python college_scorecard/college_scorecard_ood_evaluation.py --run <result folder>
```

Without `--run`, the newest run folder whose checkpoints were trained on the
College Scorecard training file is used. The script refits every final model
(the three front positions of MORSE, the SO-GA, SFS and the all-features
model) and every distinct front solution from the checkpoints, checks the
recomputed in-domain scores against the run's own results, and reports the
in-domain and out-of-domain ROC-AUC, the raw and prevalence-calibrated
PR-AUC, the drop between them, paired Wilcoxon tests of MORSE against every
baseline over the seeds and bootstrap intervals over the institutions.
`--selection auto|max_s|knee` picks the MORSE model that is compared with the
baselines (`auto` = the run's own rule). The outputs go to
`<run>/evaluation/ood_test/`.

### Numerical reproducibility

The pipeline is fully deterministic up to BLAS-thread-count effects — and,
importantly, the **only** source of randomness that the per-seed `seed`
varies is the GA's own algorithmic stochasticity. This is deliberate:

- **What is held fixed across all 20 seeds.** The train/test split is produced
  once by the preparation notebook and committed to CSV. The inner
  cross-validation splitter is `StratifiedKFold(n_splits=3, shuffle=True,
  random_state=42)` — created once and reused for every seed and every method,
  so the fold partition is identical across seeds.
- **What the seed varies.** `set_seed(s)` seeds `random` and `numpy.random`,
  which drives the GA's population initialisation, tournament selection,
  crossover, and mutation. `LogisticRegression` uses `solver="lbfgs"`, which
  is deterministic, so its `random_state` has no numerical effect; and
  `SequentialFeatureSelector` is deterministic given the fixed CV. Consequently
  **only MORSE and the SO-GA vary with the seed** — the forward-stepwise and
  all-features baselines are identical across all seeds. The notebook
  therefore computes them **once** (before the seed pool starts) and copies the
  result into every seed's checkpoint; `checkpoint_utils.require_fixed_cv`
  refuses a splitter whose folds would depend on the seed. (Selecting features
  with a different CV split does change the SFS result — its selected sets
  overlap only about 30–50% between splits — so the fixed folds are what makes
  it look stable.)
- **Why this design.** The goal here is to characterise the **algorithmic
  randomness** of the evolutionary optimiser — its run-to-run stability and
  central tendency on a fixed data partition — *without* confounding it with
  **sampling randomness** from resampled train/test splits. Holding the data
  partition fixed isolates the optimiser's variance component: the across-seed
  spread (and any paired statistical test across seeds) answers "how reliably
  does the GA reach a good feature subset?", not "how well does the method
  generalise across datasets?".
- **Scope note.** Because a single fixed split is used, the reported spreads
  are **not** an estimate of generalisation error across resampled data, and
  should not be read as one. A study that additionally wants sampling-variance
  estimates would wrap the whole pipeline in repeated stratified train/test
  splits or nested cross-validation; that is intentionally out of scope for
  this repository, which targets the algorithmic-randomness question above.
- **Parallelism.** `N_JOBS` does not affect numerical results — each seed is
  self-contained and seeds share no mutable state across workers. The
  notebook's first cell caps `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OMP_NUM_THREADS`, and `BLIS_NUM_THREADS` to 1 **before** `numpy` is
  imported, so the per-worker BLAS pool does not oversubscribe cores against
  the `joblib` loky worker pool. In a kernel that has already imported
  `numpy`, the cap has no effect.
- **BLAS threads and the final-model record.** The L-BFGS fit of a large
  model depends slightly on the BLAS thread count (up to 7.7e-7 in the
  coefficients of a 293-input model), more than the 1e-8 that the check
  against `evaluation/final_models_<rule>.json` allows. Re-evaluate a run with
  the BLAS threading its record was made with; the command-line evaluations
  (`robustness_evaluation.py`, `legacy_stress_evaluation.py`) do not cap the
  threads.

---

## What the notebook delivers

Per dataset, the end-to-end run produces the following analysis artefacts.
All are described inline in the notebook with markdown headers above the
corresponding code cells:

- **Per-seed convergence plots** — a 3-panel figure for MORSE (AUC, sign
  consistency, Pareto-front size vs generation) and a fitness curve for the
  SO-GA.
- **Per-seed Pareto front** (`pareto_front.png`) with the three canonical
  selection candidates highlighted: the knee point (cyan star), best
  sign-consistency (orange diamond), and best AUC (green triangle).
- **Robustness suite** (`evaluation/robustness/`, see
  [docs/robustness.md](docs/robustness.md)) — the final models of the four
  methods for every seed on the test set, under stresses generated
  automatically from the data (no feature is named; the settings are the same
  for every dataset), with ROC-AUC as the primary metric:
  - **re-weighted test populations** (every row keeps its own (x, y), so
    P(y | x) is unchanged): population shifts along the leading principal
    components and towards typical / atypical rows (severity = the effective
    sample size left on the training rows, 90% to 60%), and **dependence
    shifts** that weaken pairs of correlated inputs (25% to 100% of the
    correlation removed, never beyond zero) or strengthen them (+12.5% to +50%)
    while each input keeps its location and spread. Every family is summarised
    over one cohort — the scenarios usable at every level — so the report, the
    tests and the figures describe the same scenarios;
  - **corrupted test sets** from a fixed bank shared by all models (common
    random numbers, independent of the GA seed): measurement noise, recording
    noise of 0/1 inputs (one-hot groups re-drawn as one input),
    under-recording, values that become "not available" (value and flag),
    and numerical values masked (continuous inputs set to their training
    median, every 0/1 input and flag unchanged — the one masking that applies
    to every dataset, with or without availability flags). The corruptions
    keep the structure of the data — one-hot groups stay valid, availability
    flags (paired with their values by name first) are never re-drawn, and a
    value whose flag says "not available" keeps its fill value — which is
    checked after every corruption; combinations of 0/1 values merely never
    seen in training are counted as a diagnostic, not prevented;
  - per family and severity: the stressed score and its change from each
    model's clean score, MORSE against every baseline (paired Wilcoxon tests
    over the seeds), an exploratory sign-consistency-vs-degradation analysis,
    five figures and `report.txt`.

- **Legacy stress grid** (`evaluation/all_models_comparison/`, optional block
  after the suite, `RUN_LEGACY_STRESS_GRID`) — the robustness evaluation of runs
  made before the suite, unchanged so that results stay comparable with them:
  Gaussian noise × a covariate shift along the first principal component (a
  re-weighting of the test rows, both directions), re-draw noise on the 0/1
  inputs, and AURS (the fraction of each method's own clean score kept over the
  grid). Its limits are the reasons for the suite: availability flags are
  re-drawn like any 0/1 input, and AURS nets gains in one shift direction
  against losses in the other.
- **Feature-count comparison** — a boxplot + stripplot of the number of
  selected features per method across seeds, with per-seed and summary CSVs,
  showing the parsimony of each method.
- **Sign consistency of the final models** — for each of the four methods and
  each seed, the fraction of the final (full-training-set refit) model's
  features whose logistic-regression coefficient has the same sign as the
  feature's marginal correlation with the target, using the same strict rule
  as the GA fitness (`evaluation_utils.compute_model_sign_consistency`).
  This measures MORSE's target quantity on the baselines too, which never see
  it during selection. Read it together with the feature counts: a model with
  a single feature is trivially 100% consistent. Boxplot + per-seed and
  summary CSVs.
- **Best-MORSE-model report, chosen without the test set** — the seed whose
  selected Pareto solution has the best cross-validated objective, and the
  threshold at which sensitivity ≈ specificity on out-of-fold predictions of
  the training data (`evaluation_utils.select_deployment_model`); the test set
  is used once, to report accuracy, ROC-AUC, PR-AUC, F1, sensitivity,
  specificity and balanced accuracy at that threshold (CSV), alongside the
  out-of-fold sensitivity/specificity-vs-threshold curve (PDF). An earlier
  version chose the seed and the threshold on the test set, which made its
  numbers optimistic.

### Scope & possible extensions

This repository deliberately focuses on the pipeline above. Analyses that a
broader study might add — and that are **not** part of this codebase — include
feature-selection **stability** (Jaccard similarity across seeds),
**multicollinearity** diagnostics (VIF distributions of the selected
features), and an explicit fold-vs-full-train sign-consistency verification.
The fixed-split design documented above means that significance tests across
seeds — including those of the robustness suite — speak to algorithmic
reproducibility rather than to cross-dataset generalisation.

---

## Open science contribution

This repository is released as an **open contribution to the open-science
community**:

- **Code**: all source, configuration, and evaluation logic are released
  under the repository's open licence — no closed-source components, no
  proprietary dependencies beyond standard Python scientific libraries.
- **Datasets**: the redistributable datasets (Arrhythmia, College Scorecard
  and Diabetes 130-US Hospitals) are included alongside their preprocessing
  notebooks and preprocessed CSVs (the raw College Scorecard download is
  regenerable and not committed). RadFusion is access-controlled: its
  preprocessing notebook is included, its data are not.
- **Reproducibility**: dependencies are pinned in `requirements.txt`, the
  preprocessing is leakage-free (all fitted steps learn from the training
  partition only), and the timestamped output directory captures every plot
  and CSV. Re-running the notebook on the same dataset configuration
  reproduces the results within floating-point precision.
- **Methodology transparency**: every methodological choice (the Pareto-front
  selection rule, 3-fold inner CV, the fixed-split / algorithmic-randomness
  design, 20 seeds) is documented in inline markdown cells and in this README.

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

### College Scorecard (U.S. Department of Education, 2026)

> U.S. Department of Education. (2026). *College Scorecard: Most Recent
> Cohorts, Institution-Level Data* (file
> `Most-Recent-Cohorts-Institution_06102026.zip`, released 10 June 2026)
> [Dataset]. https://collegescorecard.ed.gov/data/ (accessed 25 September
> 2026).

The task definition (features and label) and the out-of-domain split (the
held-out Carnegie Classification types) follow TableShift:

> Gardner, J., Popović, Z., & Schmidt, L. (2023). Benchmarking distribution
> shift in tabular data with TableShift. In *Advances in Neural Information
> Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track*.
> https://arxiv.org/abs/2312.07577

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

### RadFusion (not redistributed; access-controlled)

> Zhou, Y., Huang, S.-C., Fries, J. A., Youssef, A., Amrhein, T. J.,
> Chang, M., Banerjee, I., Rubin, D., Xing, L., Shah, N., & Lungren, M. P.
> (2021). RadFusion: Benchmarking performance and fairness for multimodal
> pulmonary embolism detection from CT and EHR. *arXiv preprint*
> arXiv:2111.11665. https://arxiv.org/abs/2111.11665
