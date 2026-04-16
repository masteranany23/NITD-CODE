# 🧬 Pediatric Genetic Disorder Prediction

> Multi-target classification of genetic disorder type and subclass from heterogeneous pediatric clinical data.

---

## Problem Statement

Predict two targets from structured clinical and demographic features:

| Target | Column | Classes |
|--------|--------|---------|
| **Target 1** | `Genetic Disorder` | 4 (Mitochondrial, Single-gene, Multifactorial, Unknown) |
| **Target 2** | `Disorder Subclass` | 10 (Leigh syndrome, Cystic fibrosis, Diabetes, etc.) |

Challenges: missing values, noisy data, severe class imbalance, multi-class prediction.

---

## Repository Structure

```
├── notebook880f15c302.ipynb              # Full preprocessing pipeline
├── model_target1_genetic_disorder.ipynb  # Model training — Target 1 (4 classes)
├── model_target2_disorder_subclass.ipynb # Model training — Target 2 (10 classes)
├── result.csv                            # Final predictions
└── README.md
```

---

## Pipeline

### Stage 1 — Preprocessing

| Step | Detail |
|------|--------|
| Drop irrelevant features | Patient names, IDs, location strings |
| Imputation | Median (numerical) · Mode (categorical) |
| Outlier treatment | IQR capping |
| Feature engineering | Symptom aggregation, parental risk flags, age bins |
| Encoding | OrdinalEncoder + LabelEncoder for targets |
| Class imbalance | **SMOTE** (Synthetic Minority Oversampling) per target |
| Feature selection | Mutual Information + Random Forest importance |
| **Output** | `X_train (39 474 × 53)` · `X_test (9 465 × 53)` — 0 nulls, 0 infs ✅ |

### Stage 2 — Model Training (one notebook per target)

```
Load → Baseline CV → Optuna Tuning → Stacking / Voting Ensemble → OOF Report → Predict
```

| Model | Detail |
|-------|--------|
| **XGBoost** | Gradient boosting · Optuna 50 trials |
| **LightGBM** | Leaf-wise boosting · Optuna 50 trials |
| **CatBoost** | Ordered boosting · Optuna 40 trials |
| **Random Forest** | Bagging · `class_weight=balanced` · Optuna 30 trials |
| **Stacking Ensemble** | LR meta-learner over 5-fold base predictions |
| **Soft-Voting Ensemble** | Probability averaging across all four models |

**Evaluation metric:** F1-macro (5-fold Stratified CV)

---

## Key Design Decisions

- SMOTE applied to **training split only** — prevents leakage
- Optuna **TPE sampler** for efficient Bayesian hyperparameter search
- Stacking with **inner CV** — avoids meta-learner overfitting
- Target 2 models use **larger trees / more estimators** to handle 10-class complexity
- Label encoders (`le_t1.pkl`, `le_t2.pkl`) saved and reused to decode predictions back to original string labels

---

## Output — `result.csv`

```
Patient Id | Genetic Disorder | Disorder Subclass
```

---

## How to Reproduce

```bash
pip install xgboost lightgbm catboost optuna imbalanced-learn scikit-learn pandas numpy matplotlib seaborn
```

1. Run `notebook880f15c302.ipynb` (preprocessing)
2. Run `model_target1_genetic_disorder.ipynb`
3. Run `model_target2_disorder_subclass.ipynb`
4. Merge outputs → `result.csv`

**Environment:** Python 3.10 · Kaggle free-tier CPU
