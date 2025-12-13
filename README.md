# COGS118A Final Project

This project reproduces the style of experiments from Caruana & Niculescu-Mizil (2006) by evaluating multiple machine-learning classifiers across several real-world datasets. The goal is to compare SVM, Random Forest, and MLP across different datasets and training splits using systematic cross-validation.

The project includes:

- Automated experiment runner
- Hyperparameter search
- Repeated trials
- Aggregation of results
- Publication-quality figures
- A final report following ML conference format (NeurIPS/ICML-style)

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

4) Verify Python path

```bash
which python
```

It should point to:

```bash
.../cogs118a-final-project/venv/bin/python
```

## Running Experiments

### Quick test run (1 dataset × 1 classifier × 1 split × 1 trial)
Confirms the environment works.

```bash
python src/run_all.py --test
```

You should see JSON printed for a single SVM experiment, and a raw result saved to:

```bash
results/raw/
```

### Full experiment run (all datasets, splits, classifiers)
Runs 81+ experiments (5 datasets × 3 classifiers × 3 training splits: 20/80, 50/50, 80/20 × 3 repeated trials).

```bash
python src/run_all.py
```

Outputs:
- Hundreds of individual results in `results/raw/`
- Aggregated CSV at:

```bash
results/aggregated/aggregated_results.csv
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
- `performance_heatmap.png` (heatmap)

## Datasets Used
Evaluated on 5 UCI datasets:
1. Bank Marketing
2. Breast Cancer Wisconsin (Diagnostic)
3. Heart Disease (Cleveland)
4. Wine Quality (white/red combined as binary)
5. Digits (Pen-Based Recognition of Handwritten Digits)

Preprocessing (label merging, normalization) happens in `src/load_data.py`.

## Reproducibility
All randomness controlled via NumPy/SKLearn seeds for repeatable results.

## Expected Results Summary
- Digits → near-perfect accuracy
- Breast Cancer → extremely high performance
- Bank & Heart → mid-range accuracy
- Wine Quality → hardest (~55–65%)
- Random Forest typically best overall
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
