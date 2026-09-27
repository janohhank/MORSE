# Robustness suite

The robustness suite scores the final models of a run — MORSE, the single-objective GA (SO-GA), forward
stepwise selection (SFS) and the all-features model, for every seed — on the test set under stresses
that are **generated automatically from the data**. No feature is named anywhere: every scenario comes
from the training data by fixed rules, with the same settings for every dataset
(`robustness_config.RobustnessConfig`). So the same code runs on every dataset, and no scenario can be
picked after seeing which one favours a method.

| Module | Responsibility |
|---|---|
| `robustness_config.py` | The settings (`RobustnessConfig`). |
| `robustness_utils.py` | Weighted metrics, the automatic feature schema, the re-weightings and their severity, the corruption bank. |
| `robustness_evaluation.py` | Builds the final models, runs every scenario, writes tables, figures and the report; stand-alone entry point. |
| `plot_utils.py` | The five figures. |

## Running it

* **In the notebook.** The robustness cell of `training_notebook.ipynb` runs the suite right after
  training and writes everything to `<run>/evaluation/robustness/`.
* **On a finished run, without retraining.**

  ```bash
  python robustness_evaluation.py --run 2026-09-25_14-06-20_college_scorecard_pr
  ```

  The run's data are rebuilt by executing the configuration and data-loading cells of the notebook copy
  archived in the run folder: every code cell from the first one that assigns `TARGET_COLUMN` to the
  first one that assigns `X_test`. This reproduces the run's paths, the merged validation file and any
  column whitelist. The rebuilt training data are then checked against the run's checkpoint fingerprint
  (feature names, hashes of the standardised training matrix and of the labels). The models are refit
  from the checkpointed masks. A run folder without an archived notebook can use a copy given with
  `--notebook` (the fingerprint check still applies).
  Options: `--selection auto|knee|max_s` picks MORSE's Pareto solution (`auto` uses the run's own rule;
  another rule writes to `robustness_<rule>/`, so it never overwrites the run's own evaluation),
  `--seeds 42 43 ...` evaluates a subset of seeds, and `--out` sets another output folder.
  A run takes about 0.5–1 minute per dataset.

## Two kinds of stress

**Re-weighted test populations** change which rows the test population consists of, i.e. P(x), and
never a row itself. Every row keeps its own (x, y) pair, so P(y | x) is unchanged (a covariate shift).
Only real rows are re-emphasised, so a re-weighting cannot produce an impossible combination of values
and needs no knowledge of what the features mean.

**Corrupted test sets** change recorded values: measurement noise, recording errors, lost records,
values that are not available. A corruption does not keep P(y | observed x). It also has to keep the data
valid: a one-hot group keeps exactly one level, a "not measured" flag keeps its fill value, and no
combination of 0/1 values is created that never occurs in training. The rules for that are inferred from
the training data too (next section).

The two kinds answer different questions and are reported separately.

## Automatic feature schema (`robustness_utils.infer_feature_schema`)

Everything is inferred from the **training rows** only:

* **Types.** 0/1 inputs, continuous inputs (more than two distinct values), constants, and "other" (two
  values that are not 0/1; left alone).
* **Values with an availability flag.** A numeric input that holds one value on every row of one level
  of a 0/1 flag (at least 10 rows), and that value on at most 5% of the other rows. This is an imputed
  value with its missing/measured indicator, e.g. `ADM_RATE:missing` = 1 ⇒ `ADM_RATE` = 0.7542, or
  `albumin:Binary` = 0 ⇒ `albumin:Value` = 3.1. The 5% condition matters: without it, near-empty
  columns (RadFusion's `hgb:Value` is measured in 0.2% of the exams) look constant on any subgroup.
* **One-hot groups.** 0/1 inputs named `<prefix>_<level>` (pandas' dummy encoding) with at most one of
  them equal to 1 in every training row. The name only *proposes* a group; the data must confirm it. On
  College Scorecard this finds the 8 groups (STABBR, CCSIZSET, AccredAgency, LOCALE, region, ...) and
  rejects `feature_*` (Arrhythmia) and `Outpatient_*` (RadFusion), whose members co-occur. Co-occurrence
  alone cannot find groups reliably: chains of never-together pairs merged 73 unrelated College
  Scorecard columns into one "group".
* **Forbidden combinations.** A pair of 0/1 values that never occurs together in training although
  independence predicts at least 5 rows: two levels of a one-hot group, sex-specific codes, a flag that
  is only ever set together with another one. Combinations predicted to be rarer than that cannot be
  told apart from chance and are not protected.
* **Stand-alone 0/1 inputs** are those in no group and not an availability flag. Those with a training
  prevalence below 50% (1 = the recorded event) are subject to under-recording.

`schema.json` in the output folder lists everything that was inferred.

## Re-weighted test populations

### Population shifts

The leading three principal components of the standardised training inputs (the sign of each is fixed
so that its loadings sum to a positive value), tilted in both directions. Also "extremes": the
Mahalanobis distance from the centre within the leading components that explain 80% of the variance,
tilted towards atypical rows (+) or typical rows (−).

A population family is summarised by its **worst** scenario. Its scenarios move the population in
opposite directions along each axis, so a mean would net harmful shifts against beneficial ones.

### Dependence shifts

For pairs of correlated inputs, the test population is re-weighted so that the pair becomes **less**
dependent (*decorrelate*) or **more** dependent (*strengthen*), while each input keeps its location and
spread.

* **Pairs.** A fixed rule on the training correlations, identical for every method and seed: training
  |r| in [0.3, 0.95); not two levels of one one-hot group; not a value with its own availability flag;
  for 0/1 members, at least 20 training rows and 5 test rows in every cell of the 2 × 2 table (or every
  level of a single 0/1 member). The strongest 200 pairs by |r| are used, and all of them are evaluated.
  There is no top-k choice to cherry-pick.
* **The tilt** (`DependenceTilt`). Each input enters as a score: a 0/1 input standardised, a continuous
  input as its normal score under the training distribution, clipped to ±2.5. The weights are the
  minimum-KL re-weighting of the training rows of the form

  w ∝ exp(s · z_a z_b + λ·t),  t = (z_a, z_b[, z_a², z_b²] for continuous inputs),

  with λ solved so that the weighted training rows keep the unweighted means (and, for continuous
  inputs, the second moments) of the scores (entropy balancing). Only the product moment, the
  dependence, moves. For two 0/1 inputs this changes the odds ratio of their 2 × 2 table while keeping
  both prevalences. The same function is applied to the test rows.
  A plain exp(s · z_a z_b) tilt, without the balancing terms, mostly moves the means instead: at a
  training ESS of 60% it shifted both means by up to 0.7–0.8 SD on College Scorecard and RadFusion
  pairs.
* **Summary.** Each direction is its own family, summarised by the mean over its usable pairs. A
  direction is summarised at a severity only when at least 10 pairs are usable there (`min_pairs`).
  Strengthening an already strong dependence of two 0/1 inputs is often impossible with their
  prevalences held fixed ("saturated"), so on RadFusion only a few strengthened pairs remain.

**Why this family matters for MORSE.** Re-weighting never changes P(y | x). A model whose coefficients
are the true conditional effects stays the best ranker under any of these shifts, including a genuine
suppressor whose coefficient sign differs from its marginal correlation. What re-weighting *can* expose
are coefficients that only work through the training correlation: correlated inputs with large
opposite-sign coefficients that cancel. Those break when a pair is decorrelated (rows in which the two
disagree get more weight), and matter less when the dependence is strengthened. The prediction to check
is therefore: **MORSE should lose less than the SO-GA under decorrelation, not under strengthening.**
Genuine suppressors push the other way, so the test can come out against MORSE.

### Severity: the effective sample size (ESS)

Every re-weighting is calibrated to a target **effective sample size on the training rows**:

ESS = (Σw)² / Σw².

A weighted average over n rows is as precise as a plain average over ESS rows. ESS = n for equal
weights; the more unequal the weights, the smaller the ESS. The suite uses **ESS/n = 90%, 80%, 70%,
60%** (`ess_levels`).

**Why ESS, and what the numbers mean.** Every re-weighting statistic (a PC score, a distance, a product
of two scores) is first mapped to normal scores under its training distribution. An exponential tilt
w ∝ exp(s·g) of a standard normal g then moves g by exactly s standard deviations and leaves
ESS/n = exp(−s²). So one ESS level means the same shift size for every axis and every dataset, whatever
the scale or distribution of the statistic:

| training ESS / n | shift of the tilted statistic | KL divergence to the clean population |
|---|---|---|
| 90% | 0.32 SD | 0.05 |
| 80% | 0.47 SD | 0.11 |
| 70% | 0.60 SD | 0.18 |
| 60% | 0.71 SD | 0.26 |
| 50% | 0.83 SD | 0.35 |
| 37% | 1.00 SD | 0.50 |

Equal tilt strengths do *not* mean equal severity across axes or datasets. That is why the strength is
calibrated (`calibrate_strengths`) and reported, but not used as the severity scale.

**How the choice affects the results.** A lower ESS is a larger shift, so the changes in ROC-AUC grow
(roughly in proportion to the shift in SD). It is also a noisier estimate: the standard error of a
weighted AUC grows like 1/√ESS, and the test sets are small (136 rows on Arrhythmia, 190 on RadFusion).
Below about 60% the smaller test sets run out of effective rows per class. The suite therefore evaluates
the whole grid and draws curves over it, so the conclusion can be checked for stability across the
levels. The overview figure and the sign-consistency analysis use 60% (`headline_ess`).

**Support on the test set.** The ESS is calibrated on the training rows, but the test rows are what is
scored. A scenario counts as *supported* if the re-weighted test rows keep at least 30% of the rows as
ESS and at least 20 effective rows in each class. A *saturated* scenario cannot reach its target on
the training rows (or its moments cannot be balanced there). Unsupported and saturated scenarios are
reported in `scenarios.csv` but left out of the summaries.

### Metrics under re-weighting

ROC-AUC is the primary metric: it compares positives with negatives, so it does not depend on the
class prevalence. It is identical whether the weights are normalised globally or within each class.
The weights are normalised within each class (`normalise_within_classes`), so that PR-AUC (average
precision) measures the shifted population at the clean prevalence. Otherwise it would mostly track the
prevalence change: a College Scorecard PC1 tilt at 60% ESS moves the prevalence from 0.644 to 0.457. The
re-weighted prevalence is recorded in `scenarios.csv`.

## Corrupted test sets (`CorruptionBank`)

| family | what happens at level p (0 = clean) |
|---|---|
| measurement noise | x + p · (training SD) · N(0, 1) on every continuous input; values that are not available keep their fill value |
| recording noise | every stand-alone 0/1 input, and every one-hot group as one categorical input, is re-drawn from its training distribution with probability p |
| under-recording | every recorded 1 of a stand-alone 0/1 input with training prevalence below 50% is lost (set to 0) with probability p |
| values not available | every available value with an availability flag becomes "not available" (flag set, fill value) with probability p; a flag that is never off while another one is on takes that one along |

The levels are 0.1, 0.2, …, 1.0 (`corruption_levels`). The **corruption bank** holds 10 realisations
per family (`corruption_repetitions`). Their random draws come from their own seed (`corruption_seed`),
one stream per family, independent of the GA seed. Every model is scored on the very same corrupted
data, and the same draws are reused at every level (**common random numbers**): the Gaussian noise at
0.4 is exactly twice that at 0.2, and the cells corrupted at a level are a subset of those corrupted at
any higher level. After a corruption, a change that created a forbidden combination is undone for that
row. Of two conflicting changes, only the one that enters at the higher level is undone, which keeps
the nesting. `corruption_diagnostics.csv` counts what was eligible, selected, changed and undone.

The earlier notebook sweeps seeded their noise with the GA seed, so their "SD across seeds" mixed the
search variability with the corruption draws. For SFS and the all-features model, which are identical in
every seed, that band was pure corruption noise. The bank separates the two.

## Statistics

* **Unit of replication: the GA run (seed).** For every family and severity each model gets one number:
  the worst usable scenario (population shifts), the mean over the usable pairs (dependence), or the
  mean over the corruption bank (corruption). Corruption repetitions are averaged *within* a run before
  runs are compared; they are never treated as extra runs.
* **MORSE vs SO-GA:** paired two-sided Wilcoxon signed-rank test over the seeds, on the change from each
  model's clean score and on the stressed score itself. **MORSE vs SFS / all features:** the one-sample
  test of MORSE's seeds against the baseline's single value. A positive difference means MORSE is
  better. For the re-weighting families, `tests.csv` also gives the share of usable scenarios in which
  MORSE's run-averaged change beats the SO-GA's. This is descriptive only: the scenarios share their
  test rows.
* **Uncertainty labels.** "SD across runs" is the algorithmic variability, after averaging over
  scenarios or the corruption bank. `corruption_sd` is the SD over the repetitions, averaged over the
  runs.
* **Sign consistency vs degradation** (`sign_vs_degradation.csv`): Spearman correlation between the sign
  consistency S of every GA model and its change of ROC-AUC, and the partial correlation given the model
  size K. This is exploratory: a correlation does not show that S causes the difference.
* A smaller degradation alone does not make a method better: the tables always give the stressed score
  next to the change.

## Outputs (`<run>/evaluation/robustness/`)

| file | content |
|---|---|
| `config.json` | the settings, the git commit, source hashes, the training fingerprint's hash |
| `schema.json` | the inferred feature schema |
| `models.csv` | every final model: size, sign consistency, clean ROC-AUC / PR-AUC |
| `scenarios.csv` | every re-weighting scenario: strength, training/test/per-class ESS, support, saturation, and what actually moved (statistic shift, correlation before/after, mean shifts, SD ratios, balance error) |
| `reweighting_scores.csv`, `corruption_scores.csv` | every model under every scenario / corrupted test set |
| `corruption_diagnostics.csv` | what every corrupted test set changed |
| `model_family_scores.csv` | every model per family and severity (`summarised` = False: too few usable pairs) |
| `summary.csv` | per family, severity and method: stressed score, change, SDs, worst case, clean score |
| `tests.csv` | MORSE against every baseline |
| `sign_vs_degradation.csv` (+ `_points.csv`) | the exploratory sign-consistency analysis |
| `report.txt` | the printed report |
| `robustness_population.png` | ROC-AUC along every population axis, both directions |
| `robustness_dependence.png` | mean change under decorrelation / strengthening, and MORSE − SO-GA per pair |
| `robustness_corruption.png` | ROC-AUC against the corruption level, per family |
| `robustness_overview.png` | clean vs stressed ROC-AUC per family at the headline severity |
| `robustness_sign_vs_degradation.png` | sign consistency vs change, marker size = model size |

## Limitations

* Re-weighting only re-emphasises existing test rows. It cannot create combinations the test set does
  not contain, and it cannot change P(y | x). A real change of the relationship between inputs and
  outcome (concept shift) needs other evidence: the College Scorecard out-of-domain set
  (`college_scorecard/college_scorecard_ood_evaluation.py`), or a synthetic dataset with a known outcome
  mechanism.
* On small test sets many dependence scenarios are not supported (Arrhythmia), and strengthening 0/1
  pairs is often saturated (RadFusion). The report shows how many pairs each summary rests on.
* One-hot groups are only found for pandas-style `<prefix>_<level>` names. Impossible combinations are
  only protected when at least 5 rows are expected under independence.
* Measurement noise ignores bounds and integer values: the models are frozen linear scores, so this
  changes no computation, but a noisy count can be negative.
* Under-recording applies to every stand-alone 0/1 input with a prevalence below 50%. The rule cannot
  tell a record (a diagnosis code) from an attribute (an institution type).
* The robustness evaluation of runs made before the suite — the Gaussian noise × PC1 covariate-shift
  grid, 0/1 re-draw noise and AURS — is kept unchanged as the **legacy stress grid**
  (`legacy_stress_evaluation.py`): an optional notebook block after the suite and
  `python legacy_stress_evaluation.py --run <run>`, writing `evaluation/all_models_comparison/` as before.
  It reproduces the stored CSVs of earlier runs byte for byte. Its limits are the reasons for the suite:
  it re-draws availability flags like any 0/1 input, and AURS averages the two shift directions.
