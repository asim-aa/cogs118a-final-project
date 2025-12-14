# COGS118A Final Project

This project reproduces the style of experiments from Caruana & Niculescu-Mizil (2006) by evaluating multiple machine-learning classifiers across several real-world datasets. The goal is to compare SVM, Random Forest, MLP, and KNN across different datasets and training splits using systematic cross-validation with leakage-free preprocessing and reproducible seeds.

The project includes:

- Automated experiment runner
- Hyperparameter search
- Repeated trials
- Aggregation of results
- Publication-quality figures
- A final report following ML conference format (NeurIPS/ICML-style)

## Recruiter-Friendly Overview
- Built an end-to-end ML evaluation harness: automated runs, CV-based hyperparameter tuning, and leakage-safe preprocessing via Pipelines.
- Reproducible experiments: deterministic seeds for splits, CV, and stochastic models; logged configs and scores per run.
- Benchmarks 4 classifiers across 5 UCI datasets and 3 train/test splits (180+ runs) with aggregation and reporting plots.
- Outputs include raw JSON logs, an aggregated CSV, and ready-to-use figures for reports.

## Setup Instructions

1) Clone or download the project

```bash
git clone git@github.com:asim-aa/cogs118a-final-project.git
cd cogs118a-final-project
```

2) Create and activate the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

3) Install dependencies

```bash
pip install -r requirements.txt
```

4) Get data (if CSVs are missing)

```bash
python src/download_datasets.py
```

5) Verify Python path

```bash
which python
```

It should point to:

```bash
.../cogs118a-final-project/venv/bin/python
```

## Running Experiments

### Quick test run (all datasets × 1 classifier × 1 split × 1 trial)
Confirms the environment works.

```bash
python src/run_all.py --test
```

You should see JSON printed for a single SVM experiment per dataset, and raw results saved to:

```bash
results/raw/
```

### Full experiment run (all datasets, splits, classifiers)
Runs 180+ experiments (5 datasets × 4 classifiers × 3 training splits: 20/80, 50/50, 80/20 × 3 repeated trials).

```bash
python src/run_all.py
```

Outputs:
- Hundreds of individual results in `results/raw/`
- Aggregated CSV (after running the aggregation step below) at:

```bash
results/aggregated/aggregated_results.csv
```

After the grid completes, create the aggregated table:

```bash
python src/run_all.py --aggregate
```

## Generate Figures
After experiments complete:

```bash
python src/make_figures.py
```

Plots saved to `results/figures/`:
- `test_accuracy_by_classifier_dataset.png` (bar chart)
- `accuracy_vs_split.png` (line chart)
- `test_accuracy_error_bars.png` (error bars)


## Datasets Used
Evaluated on 5 UCI datasets:
1. Bank Marketing
2. Breast Cancer Wisconsin (Diagnostic)
3. Heart Disease (Cleveland)
4. Wine Quality (quality scores; red/white combined, multiclass)
5. Digits (Pen-Based Recognition of Handwritten Digits)

Preprocessing (impute, one-hot encode categoricals, standardize numerics) happens inside per-model Pipelines to avoid train/test leakage (see `src/experiment.py`).

## Training / Tuning Protocol
- 5-fold Stratified CV on the training split only.
- Per trial, `random_state=trial_id` is applied to splits, CV folds, and stochastic models (RF/MLP/SVM).
- Hyperparameter grids:
  - SVM: `C` ∈ {1, 10}, `gamma` ∈ {0.01, "scale"}, kernel=rbf
  - RF: `n_estimators` ∈ {200}, `max_depth` ∈ {None, 20}, `min_samples_split` ∈ {2}, `max_features` ∈ {"sqrt"}
  - MLP: `hidden_layer_sizes` ∈ {(64,), (128,)}, `learning_rate_init` ∈ {0.001}, `alpha` ∈ {0.0001, 0.001}, activation=relu
  - KNN: `n_neighbors` ∈ {5, 9}, `weights` ∈ {"uniform", "distance"}, `p` ∈ {2}
- Logged per run: dataset, classifier, split, trial/seed, n_samples, n_features, train/test sizes, best_params, best CV score, train/val/test accuracy. Aggregations compute mean±std per (dataset, classifier, split).

## Reproducibility
All randomness controlled via NumPy/SKLearn seeds for repeatable results.

## Expected Results Summary
- Digits → near-perfect accuracy
- Breast Cancer → extremely high performance
- Bank & Heart → mid-range accuracy
- Wine Quality → hardest (~55–65%)
- Random Forest typically best overall; KNN competitive on some datasets with enough data
- More training data → higher test accuracy

These trends mirror Caruana & Niculescu-Mizil findings.

## Contact / Notes
If you see missing dataset errors, ensure CSVs are in `data/`:

```bash
data/bank.csv
data/breast_cancer.csv
data/heart.csv
data/wine_quality.csv
data/digits.csv
```
