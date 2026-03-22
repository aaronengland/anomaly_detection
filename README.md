# Anomaly Detection on Financial Transactions

This project builds an anomaly detection system that identifies fraudulent transactions in financial data. It uses **unsupervised machine learning** — meaning the models learn what "normal" transactions look like without being told which ones are fraud — and then flags anything that looks unusual.

This is the same approach used in the real world for payroll fraud detection, where companies need to catch suspicious activity before labeled examples of fraud are available.

## What Does This Project Do?

Imagine you work at a company that processes millions of financial transactions. Most of them are legitimate, but a tiny fraction (about 1 in 800) are fraudulent. You can't manually review every transaction, so you need an automated system that says: "Hey, this one looks suspicious — you should check it."

That's exactly what this project builds. It trains two different anomaly detection models, compares their performance, and combines them into an ensemble that achieves **96% precision** — meaning when the system flags a transaction as suspicious, it's right 96 out of 100 times.

## The Dataset

This project uses **PaySim**, a publicly available synthetic dataset that simulates mobile money transactions. It's available on [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1).

- **6.3 million transactions** across 5 types: CASH_IN, CASH_OUT, DEBIT, PAYMENT, and TRANSFER
- **11 columns** including transaction amount, sender/receiver balances before and after, and a fraud label
- **Only 0.13% of transactions are fraud** — this extreme imbalance is realistic and makes the problem challenging

## How It Works (Step by Step)

The project is organized as 6 Jupyter notebooks, meant to be run in order:

### Notebook 00 — Data Collection

Downloads the PaySim dataset from Kaggle, performs basic cleaning (checking for missing values, ensuring correct data types), and uploads the cleaned data to AWS S3 as a parquet file for fast access by the other notebooks.

### Notebook 01 — Exploratory Data Analysis (EDA)

Explores the dataset visually to understand what we're working with:

- **Class imbalance**: 99.87% of transactions are normal, only 0.13% are fraud
- **Fraud by transaction type**: Fraud only occurs in TRANSFER and CASH_OUT transactions
- **Amount patterns**: Fraudulent transactions tend to have distinctive amount and balance-change patterns
- **Temporal trends**: How transaction volume and fraud incidents change over time

This step generates 8 visualizations that help build intuition about the data before modeling.

![Class Distribution](01_eda/output/01_class_distribution.png)

![Transaction Type Distribution](01_eda/output/02_transaction_type_distribution.png)

![Amount Distribution](01_eda/output/03_amount_distribution.png)

![Amount by Type](01_eda/output/04_amount_by_type.png)

![Fraud by Transaction Type](01_eda/output/05_fraud_by_type.png)

![Balance Analysis](01_eda/output/06_balance_analysis.png)

![Correlation Heatmap](01_eda/output/07_correlation_heatmap.png)

![Temporal Distribution](01_eda/output/08_step_distribution.png)

### Notebook 02 — Preprocessing

Prepares the data for the machine learning models:

1. **Feature engineering** — Creates new columns from the raw data that help the models detect fraud:
   - `balance_change_orig` / `balance_change_dest`: How much the sender's and receiver's balances changed
   - `amount_to_balance_ratio`: How large the transaction is relative to the sender's balance (a big withdrawal from a small account is suspicious)
   - `is_balance_zeroed`: Whether the sender's balance went to zero (a strong fraud signal)
   - `hour_of_day`: What time the transaction occurred
   - One-hot encoded transaction type (CASH_IN, CASH_OUT, etc.)

2. **Temporal train/valid/test split** — Splits the data by time, not randomly:
   - **Training set (first 60% of time)**: ~600K non-fraud transactions only. The models learn what "normal" looks like from this.
   - **Validation set (next 20% of time)**: ~228K transactions (both fraud and non-fraud). Used to tune model settings and pick the best decision threshold.
   - **Test set (final 20% of time)**: ~124K transactions (both fraud and non-fraud). Used for final, unbiased performance evaluation.

   Splitting by time simulates real deployment: the model trains on historical data and must detect fraud in future transactions it hasn't seen before.

3. **Feature scaling** — Standardizes all numeric features to have mean 0 and standard deviation 1, which helps both models perform better.

### Notebook 03 — Isolation Forest

The first anomaly detection model. **Isolation Forest** works by randomly partitioning data with decision trees. The key insight: anomalies are rare and different from normal data, so they can be "isolated" in fewer random splits. Normal data points, being similar to each other, require many more splits to separate.

- **Hyperparameter tuning**: Uses Optuna (a Bayesian optimization library) to search for the best model settings across 5 trials, optimizing for PR-AUC on the validation set
- **Best parameters found**: 260 trees, 8 features per tree, ~1.3% contamination rate

**Results on test set:**

| Metric | Score |
|--------|-------|
| PR-AUC | 0.5182 |
| ROC-AUC | 0.9140 |
| Precision | 0.7912 |
| Recall | 0.3918 |
| F1-Score | 0.5241 |

This means: when Isolation Forest flags a transaction as fraud, it's correct **79% of the time** (precision), and it catches **39% of all fraud** (recall).

![Isolation Forest Anomaly Scores](03_isolation_forest/output/01_anomaly_scores.png)

![Isolation Forest Precision-Recall Curve](03_isolation_forest/output/02_precision_recall_curve.png)

![Isolation Forest ROC Curve](03_isolation_forest/output/03_roc_curve.png)

![Isolation Forest Confusion Matrix](03_isolation_forest/output/04_confusion_matrix.png)

### Notebook 04 — Local Outlier Factor (LOF)

The second anomaly detection model. **LOF** works differently — it compares each transaction's "local density" (how close its nearest neighbors are) to its neighbors' local densities. If a point is in a much sparser region than its neighbors, it's likely an anomaly.

- **Hyperparameter tuning**: 5 Optuna trials optimizing PR-AUC, tuning the number of neighbors and contamination rate
- **Best parameters found**: 21 neighbors, ~9.8% contamination rate

**Results on test set:**

| Metric | Score |
|--------|-------|
| PR-AUC | 0.3320 |
| ROC-AUC | 0.9330 |
| Precision | 0.3083 |
| Recall | 0.5756 |
| F1-Score | 0.4015 |

LOF catches **more fraud** (58% recall vs. 39%), but at the cost of **more false alarms** (31% precision vs. 79%). This is the classic precision-recall tradeoff.

![LOF Anomaly Scores](04_lof/output/01_anomaly_scores.png)

![LOF Precision-Recall Curve](04_lof/output/02_precision_recall_curve.png)

![LOF ROC Curve](04_lof/output/03_roc_curve.png)

![LOF Confusion Matrix](04_lof/output/04_confusion_matrix.png)

### Notebook 05 — Model Comparison and Ensemble

Compares both models side-by-side and explores what happens when we combine them.

**Head-to-head comparison:**

| Metric | Isolation Forest | LOF |
|--------|-----------------|-----|
| PR-AUC | 0.5182 | 0.3320 |
| ROC-AUC | 0.9140 | 0.9330 |
| Precision | 0.7912 | 0.3083 |
| Recall | 0.3918 | 0.5756 |
| F1-Score | 0.5241 | 0.4015 |

**Ensemble (both models must agree):**

When we only flag a transaction as fraud if **both** models independently agree it's suspicious:

| Metric | Score |
|--------|-------|
| Precision | **0.9604** |
| Recall | 0.3374 |
| F1-Score | 0.4993 |
| Transactions flagged | 581 (0.47% of test set) |
| True positives | 558 |

The ensemble achieves **96% precision** — when it flags something, it's almost certainly fraud. The tradeoff is lower recall (it catches 34% of all fraud), but in a real-world setting where each flag triggers a manual review, minimizing false alarms is often more important than catching every single case.

![ROC Comparison](05_comparison/output/01_roc_comparison.png)

![Precision-Recall Comparison](05_comparison/output/02_pr_comparison.png)

![Metrics Comparison](05_comparison/output/03_metrics_comparison.png)

![Score Distributions](05_comparison/output/04_score_distributions.png)

## Why These Two Models?

Both are standard **unsupervised anomaly detection** algorithms, meaning they don't need labeled fraud examples to learn:

- **Isolation Forest** looks at the data globally — it asks "how easy is it to isolate this point from everything else?"
- **LOF** looks locally — it asks "is this point in a sparser region than its neighbors?"

They catch different kinds of anomalies, which is why combining them works well. This is also the standard starting point for anomaly detection in industry.

## Why Unsupervised?

In real fraud detection, you often don't have labels. New types of fraud emerge before anyone identifies them. An unsupervised approach learns what "normal" looks like and flags anything that deviates — it can catch novel fraud patterns that a supervised model (trained only on known fraud types) would miss.

In this project, we do have fraud labels (`isFraud`), but we only use them for **evaluation** — to measure how well the models perform. The models themselves never see these labels during training.

## Why PR-AUC Instead of ROC-AUC?

With only 0.13% fraud, the dataset is extremely imbalanced. ROC-AUC can be misleading here because the false positive rate denominator (number of non-fraud transactions) is so large that even thousands of false positives barely move the metric. A model could look great on ROC-AUC while being useless in practice.

**PR-AUC (Precision-Recall Area Under Curve)** focuses directly on the fraud class: "of the things we flagged, how many were actually fraud?" (precision) and "of all the fraud, how much did we find?" (recall). This is the metric that actually matters for deployment.

## Project Structure

```
anomaly_detection/
├── 00_data_collection/notebook.ipynb   # Download from Kaggle, clean, upload to S3
├── 01_eda/notebook.ipynb               # Exploratory data analysis and visualizations
├── 02_preprocessing/notebook.ipynb     # Feature engineering, splitting, scaling
├── 03_isolation_forest/notebook.ipynb  # Isolation Forest model
├── 04_lof/notebook.ipynb              # Local Outlier Factor model
├── 05_comparison/notebook.ipynb       # Side-by-side comparison and ensemble
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## How to Run

### Prerequisites

- Python 3.8+
- AWS credentials configured (`aws configure`) with S3 read/write access
- Kaggle API credentials (`~/.kaggle/kaggle.json`) for dataset download

### Setup

```bash
cd anomaly_detection
pip install -r requirements.txt
```

### Execution

Run the notebooks in order: `00` -> `01` -> `02` -> `03` -> `04` -> `05`. Each notebook reads its input from S3 and saves its output back to S3. Plot images are saved to each notebook's `./output/` directory.

### Running on AWS SageMaker

1. Create a SageMaker notebook instance with an IAM role that has S3 read/write access
2. Upload this project to the instance
3. Open and run each notebook sequentially
