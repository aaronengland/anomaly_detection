# Project Overview

Anomaly Detection on financial transactions — a portfolio project for Aaron England's personal website (aaron-england.com). This project is specifically designed to demonstrate skills relevant to a **Staff Data Scientist role at Paylocity**, an HCM/HR software company that does payroll anomaly detection and unusual workforce pattern detection as part of their AI product suite.

## Why This Project Exists

Paylocity's data science team builds anomaly detection systems for payroll transactions and workforce data. This project demonstrates unsupervised anomaly detection on financial transaction data — a direct analog to detecting anomalous payroll activity.

## S3 Bucket

`anomaly-detection-demo`

All data is read from and written to this bucket. Raw data goes in `00_data_collection/`, preprocessed splits in `02_preprocessing/`.

## Dataset

PaySim Synthetic Financial Transactions (Kaggle) — ~6.3M mobile money transactions with fraud labels. Features: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud. Fraud rate is ~0.13% (extreme class imbalance).

## Project Structure

```
anomaly_detection/
├── 00_data_collection/notebook.ipynb     # Download PaySim from Kaggle, clean, upload to S3
├── 01_eda/notebook.ipynb                 # Transaction EDA: distributions, fraud by type, balance analysis
├── 02_preprocessing/notebook.ipynb       # Feature engineering, scaling, temporal train/test split
├── 03_isolation_forest/notebook.ipynb    # Isolation Forest anomaly detection
├── 04_lof/notebook.ipynb                 # Local Outlier Factor anomaly detection
├── 05_comparison/notebook.ipynb          # Side-by-side model comparison + ensemble analysis
├── requirements.txt
└── README.md
```

## Execution Order

Run notebooks sequentially: 00 → 01 → 02 → 03 → 04 → 05. Each notebook reads from S3 and writes outputs to `./output/`.

## Key Technical Details

- **Unsupervised approach**: Models train on non-fraud data only from the training period — simulates real-world deployment where labels may not be available
- **Temporal split**: First 80% of time steps = train, last 20% = test (no data leakage)
- **Feature engineering**: balance_change_orig, balance_change_dest, amount_to_balance_ratio, is_balance_zeroed, hour_of_day, one-hot transaction type
- **Both models use Bayesian hyperparameter tuning via Optuna** (20 trials each)
- **Isolation Forest tuning**: n_estimators, max_samples, contamination, max_features
- **LOF tuning**: n_neighbors, contamination
- **Evaluation**: Precision, Recall, F1 at optimal threshold; PR AUC; ROC AUC; confusion matrices
- **Comparison notebook** includes ensemble analysis (flag only when both models agree)

## Coding Conventions

- **Classes**: `AnomalyEDA`, `AnomalyPreprocessor`, `IsolationForestModel`, `LOFModel`, `ModelComparison`
- **Hungarian notation**: `str_`, `int_`, `flt_`, `cls_`, `df_`, `list_`, `dict_`, `bool_`
- **Constants section** at top of each notebook: `str_bucket`, `str_task`, `str_dirname_output`
- **Plots**: Saved to `./output/` with `dpi=150`, `bbox_inches='tight'`
- **S3 loading**: Try S3 first, fall back to local path

## What Comes After This Repo

Once all notebooks are run, the results will be integrated into Aaron's personal website as a dedicated `/anomaly-detection` page with an interactive demo (users explore transactions and the model flags anomalies) and full methodology write-up.
