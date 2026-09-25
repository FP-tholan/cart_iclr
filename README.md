# CART: Contextual Cell Reliability for Robust Tabular Prediction

This repository contains the implementation and experimental code for:
> **CART: Contextual Cell Reliability for Robust Tabular Prediction**

The repository is provided anonymously for double-blind review.

---

## 1. Overview
CART is a robust tabular prediction framework that models **cell-level reliability** and uses it to selectively combine an observed cell representation with a repaired representation. The repair branch uses a **leave-one-out (LOO) context** to construct a genuine counterfactual representation, preventing the target cell from directly observing its own value during repair.

## 2. Repository Structure
```text
.
├── baselines/                  # Prediction and reconstruction baselines
├── checkpoints/                # Pretrained checkpoints (main_results_table4 & table5)
├── data/                       # OpenML loaders and synthetic corruption logic
├── evaluation/                 # Metrics (NR-AUC, Cell-AUROC/AUPRC)
├── experiments/                # Scripts reproducing tables/figures in the paper
│   ├── main_benchmark_table4.py
│   ├── main_benchmark_table5.py
│   ├── ablation_7way.py
│   ├── fixed_sweep_val.py
│   ├── statistical_test_n8.py
│   └── unseen_baselines.py
├── extra_experiments/          # Additional analyses (SHAP, targeted repair, sensitivity) not in the main text
├── models/                     # CART architecture
├── results/                    # Generated experimental outputs
├── requirements.txt
└── README.md

```

## 3. Reproducing Paper Results

Install dependencies via `pip install -r requirements.txt`. All scripts are executed from the root directory.

### Main Benchmark (Tables 1, 2, 4, 5 & Figures 2, 3)

Evaluates CART against prediction-oriented baselines (CatBoost, XGBoost, FT-Transformer, MLP, QuAIL) and reconstruction-oriented baselines (DAE, TabAE-E2E, VAE) under cell-level corruption.

```bash
python -m experiments.main_benchmark_table4
python -m experiments.main_benchmark_table5

```

### Seven-Way Mechanistic Ablation (Table 3 & Figure 4)

Isolates the contribution of the adaptive reliability gate using seven fusion strategies (including a permuted trust condition that shuffles learned reliability values across cells).

```bash
python -m experiments.ablation_7way

```

### Adaptive vs. Fixed-Gate Sweep (Table 6)

Compares CART's adaptive gating against the best possible fixed reliability ratio selected on a held-out validation split.

```bash
python -m experiments.fixed_sweep_val

```

### Unseen Corruption Shifts (Table 7)

Evaluates robustness under unseen-magnitude additive noise and single-feature distributional shifts.

```bash
python -m experiments.unseen_baselines

```

### Statistical Analysis

Aggregates the 7-way ablation results to perform the dataset-level two-sided Wilcoxon signed-rank test (n=8) reported in the paper.

```bash
python -m experiments.statistical_test_n8

```

## 4. Pretrained Checkpoints

Checkpoints for all models across the 8 datasets and 3 random seeds are provided in the anonymous cloud storage link (or `checkpoints/` directory) to facilitate evaluation without retraining.

```

```