# Anomaly Detection in Payroll Transactions

A comprehensive machine learning project for detecting anomalous transactions in payroll systems, designed to support HCM (Human Capital Management) companies in identifying unusual patterns and potential fraud.

## Overview

This project implements unsupervised anomaly detection on the PaySim synthetic financial transactions dataset, reframed as payroll anomaly detection—a direct use case for companies like Paylocity that handle payroll processing and must detect unusual workforce payment patterns.

The project compares two state-of-the-art unsupervised algorithms:
- **Isolation Forest**: Tree-based method that isolates anomalies by random feature selection
- **Local Outlier Factor (LOF)**: Density-based method that identifies points with significantly lower local density than their neighbors

Both models are trained on historical transaction data and evaluated on held-out test sets using standard ML metrics adapted for imbalanced anomaly detection.

## Project Structure

| Notebook | Purpose |
|----------|---------|
| `00_data_collection/notebook.ipynb` | Download PaySim dataset from Kaggle, clean, and upload to S3 |
| `01_eda/notebook.ipynb` | Exploratory data analysis: distributions, correlations, fraud patterns |
| `02_preprocessing/notebook.ipynb` | Feature engineering, scaling, and temporal train/test split |
| `03_isolation_forest/notebook.ipynb` | Build, tune, and evaluate Isolation Forest model |
| `04_lof/notebook.ipynb` | Build, tune, and evaluate Local Outlier Factor model |
| `05_comparison/notebook.ipynb` | Compare model performance and create ensemble insights |

## Dataset

| Aspect | Details |
|--------|---------|
| **Name** | PaySim Synthetic Financial Transactions |
| **Source** | Kaggle: https://www.kaggle.com/datasets/ealaxi/paysim1 |
| **Records** | 6,362,620 transactions |
| **Fraud Rate** | ~0.13% (highly imbalanced) |
| **Features** | step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud |
| **S3 Location** | `s3://anomaly-detection-demo/` |

## Methodology

### 00 Data Collection
- Download PaySim dataset from Kaggle using `kagglehub`
- Perform basic data cleaning (null handling, type conversions)
- Preserve the class distribution in any sampling
- Upload cleaned data to S3 for reproducibility

### 01 Exploratory Data Analysis
- Analyze class imbalance: 99.87% normal vs. 0.13% fraudulent
- Examine fraud patterns by transaction type (TRANSFER and CASH_OUT most common)
- Visualize amount distributions, balance changes, and temporal trends
- Compute correlation matrix to understand feature relationships

### 02 Preprocessing
- **Feature Engineering**:
  - Balance change calculations (orig and dest)
  - Amount-to-balance ratios (proxy for risk)
  - Balance zeroing indicator (strong fraud signal)
  - Hour-of-day from transaction step
  - One-hot encoding of transaction type
- **Scaling**: StandardScaler on numeric features
- **Temporal Split**: 80% of steps for training (non-fraud only), 20% for testing (all data)
  - This simulates deployment where models are trained on known-clean data

### 03 Isolation Forest
- **Algorithm**: Tree-based isolation via random subspace partitioning
- **Hyperparameter Tuning**: Optuna optimization of contamination, n_estimators, max_samples, max_features
- **Evaluation**: Precision, Recall, F1 at multiple thresholds; ROC-AUC and PR-AUC
- **Why This Model**: Effective on high-dimensional data, scales well, robust to feature scaling

### 04 Local Outlier Factor
- **Algorithm**: Density-based anomaly detection using k-nearest neighbors local density
- **Hyperparameter Tuning**: Optuna optimization of n_neighbors, contamination
- **Evaluation**: Same metrics as Isolation Forest for fair comparison
- **Why This Model**: Captures local density variations; good at finding contextual anomalies

### 05 Model Comparison
- Side-by-side metrics comparison at optimal thresholds
- ROC curve overlay and precision-recall curve overlay
- Ensemble analysis: flag anomalies when both models agree (high precision, lower recall)
- Visualization of anomaly score distributions and agreement patterns

## Results

*Results will be populated after model training. Expected performance:*

- **Isolation Forest**: ROC-AUC ~0.85–0.92, PR-AUC ~0.20–0.35 (accounts for class imbalance)
- **LOF**: ROC-AUC ~0.80–0.90, PR-AUC ~0.15–0.30 (density-sensitive)
- **Ensemble (both agree)**: Higher precision (~40–60%), lower recall (~20–35%)

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Unsupervised Learning** | Fraud labels are for evaluation only; in production, new fraud types emerge before labeling |
| **Temporal Train/Test Split** | Simulates real-world deployment where model is trained on historical clean data |
| **Train on Non-Fraud Only** | Mirrors actual operational pipelines where model is fitted to baseline behavior |
| **Isolation Forest + LOF** | Complementary algorithms: isolation (axis-aligned partitions) vs. density (local neighborhoods) |
| **Threshold Tuning** | Precision-recall curve analysis determines operating point based on business cost of false positives/negatives |
| **Feature Engineering** | Balance changes and ratios are domain-relevant for payroll: zero-balance, large rapid changes signal risk |
| **Class Imbalance Handling** | Evaluation via PR-AUC and F1; no synthetic oversampling to preserve natural distribution |

## Getting Started

### Prerequisites
- Python 3.8+
- AWS credentials configured (for S3 access)
- Kaggle API credentials (for dataset download)

### Setup

1. Clone the repository and install dependencies:
   ```bash
   cd anomaly_detection
   pip install -r requirements.txt
   ```

2. Configure AWS credentials:
   ```bash
   aws configure
   ```

3. Set up Kaggle API credentials:
   - Download `kaggle.json` from https://www.kaggle.com/settings/account
   - Place in `~/.kaggle/kaggle.json`
   - `chmod 600 ~/.kaggle/kaggle.json`

4. Run notebooks in order:
   - Start with `00_data_collection/notebook.ipynb` to download and upload data
   - Then run `01_eda` through `05_comparison` sequentially

### Running on AWS SageMaker

1. Create a SageMaker notebook instance with appropriate IAM role (S3 read/write, Kaggle API access)
2. Upload this project to the instance
3. Open notebooks and execute cells sequentially
4. All output plots are saved to the `output/` subdirectories
5. Models are serialized for deployment or further analysis

## Files

- `requirements.txt`: Python package dependencies
- `00_data_collection/notebook.ipynb`: Data ingestion pipeline
- `01_eda/notebook.ipynb`: Exploratory analysis and visualization
- `02_preprocessing/notebook.ipynb`: Feature engineering and data preparation
- `03_isolation_forest/notebook.ipynb`: Isolation Forest implementation
- `04_lof/notebook.ipynb`: LOF implementation
- `05_comparison/notebook.ipynb`: Model comparison and ensemble analysis
- `README.md`: This file

## Contact & Attribution

This project demonstrates advanced anomaly detection techniques for payroll transaction monitoring. Designed for portfolio review.
