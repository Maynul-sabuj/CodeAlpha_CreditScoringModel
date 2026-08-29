# CodeAlpha_CreditScoringModel

**CodeAlpha Machine Learning Internship — Task 1: Credit Scoring Model**

Predicts whether an individual is a **good** or **bad** credit risk using
financial-history features (income, existing debt, payment history, credit
utilization, etc.), and compares three classification algorithms.

## Objective

Build a model that predicts an individual's creditworthiness from past
financial data, and evaluate it using Precision, Recall, F1-score, and
ROC-AUC — as specified in the task brief.

## Approach

- **Algorithms compared:** Logistic Regression, Decision Tree, Random Forest
- **Feature engineering:** numeric scaling (`StandardScaler`) + one-hot
  encoding of categorical fields, wrapped in a single `sklearn.Pipeline`
  so preprocessing never leaks between train/test splits
- **Evaluation:** classification report (precision/recall/F1 per class),
  ROC-AUC, confusion matrices, and a Random Forest feature-importance chart
- **Best model selection:** highest F1-score (a balanced metric, since
  credit datasets are naturally imbalanced) is saved to `models/`

## Dataset

`data/credit_data.csv` is a **synthetically generated** dataset
(`generate_dataset.py`) of 5,000 individuals with realistic underwriting
features:

| Feature | Description |
|---|---|
| `age`, `employment_length_years`, `credit_history_years` | demographic/tenure |
| `annual_income`, `existing_debt`, `debt_to_income_ratio` | financial history |
| `num_credit_lines`, `num_late_payments_2yrs`, `credit_utilization` | credit behavior |
| `savings_balance`, `checking_balance` | liquidity |
| `loan_amount`, `loan_term_months`, `purpose` | the loan being scored |
| `has_bankruptcy`, `home_ownership` | risk flags |
| `creditworthy` | **target**: 1 = good risk, 0 = bad risk |

The features are combined through a realistic underwriting formula with
added noise, so the classification task behaves like a genuine credit-risk
problem (not trivially separable). It's ~74% good / 26% bad, similar to the
real-world Statlog German Credit dataset's ~70/30 split.

**Want to use real data instead?** Swap `data/credit_data.csv` for any
dataset with the same columns and a `creditworthy` target (e.g. the
[UCI/Statlog German Credit dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data),
or a Kaggle credit-scoring dataset) — `credit_scoring.py` requires no code
changes as long as the target column is named `creditworthy`.

## Project Structure

```
CodeAlpha_CreditScoringModel/
├── data/
│   └── credit_data.csv          # dataset (generated)
├── models/
│   └── best_credit_scoring_model.joblib
├── outputs/
│   ├── model_metrics.csv
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── best_model_info.json
├── generate_dataset.py          # creates data/credit_data.csv
├── credit_scoring.py            # main training & evaluation pipeline
├── Credit_Scoring_Model.ipynb   # notebook walkthrough (same pipeline)
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
python3 generate_dataset.py   # creates data/credit_data.csv
python3 credit_scoring.py     # trains models, prints metrics, saves plots
```

Or open `Credit_Scoring_Model.ipynb` for a step-by-step walkthrough with
inline visuals.

## Results

| Model | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.793 | 0.929 | 0.856 | 0.768 |
| Decision Tree | 0.770 | 0.934 | 0.844 | 0.676 |
| Random Forest | 0.770 | 0.963 | **0.856** | 0.761 |

*(Metrics are for the class-weighted average on the held-out test set;
exact numbers may vary slightly with the random seed. See
`outputs/model_metrics.csv` for the full per-class breakdown.)*

Random Forest was selected as the best model by F1-score. Its feature
importances (see `outputs/feature_importance.png`) confirm that
**debt-to-income ratio**, **credit utilization**, and **late payment
history** are the strongest predictors of credit risk — consistent with
real-world underwriting practice.

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

---

