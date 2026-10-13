# Machine Learning Model Training Report

## 1. Executive Summary
- **Project Title:** Internet Traffic Congestion Prediction Using Machine Learning
- **Stage:** STEP 6 — Multi-Model Training & Pipeline Serialization
- **Target Variable:** `Congestion_Level` ($\in \{\mathbf{LOW}, \mathbf{MEDIUM}, \mathbf{HIGH}\}$)
- **Models Trained:**
  1. Multinomial Logistic Regression (L2 Regularized with StandardScaler)
  2. Decision Tree Classifier (Constrained Depth CART)
  3. Random Forest Classifier (100 Trees Ensemble, Memory-Conscious)
- **Training Dataset:** [`./dataset/processed_train.csv`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/dataset/processed_train.csv)
- **Testing Dataset:** [`./dataset/processed_test.csv`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/dataset/processed_test.csv) (held out strictly for downstream evaluation)
- **Model Artifacts Directory:** [`./models/`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models)

---

## 2. Dataset & Feature Space Specifications

- **Number of Training Records:** **160,000 flows** (80.0% of working dataset)
- **Number of Testing Records:** **40,000 flows** (20.0% holdout test set)
- **Predictor Feature Count:** **58 features**
- **Target Classes:** 3 balanced classes:
  - `LOW`: 53,334 training samples (33.33%)
  - `MEDIUM`: 53,333 training samples (33.33%)
  - `HIGH`: 53,333 training samples (33.33%)
- **Quarantined & Excluded Features:**
  - All direct rate constituents (`Flow Bytes/s`, `Flow Packets/s`, `Fwd Packets/s`, `Bwd Packets/s`)
  - All algebraic rate/duration mirrors (`Flow Duration`, all 14 `Flow IAT`, `Fwd IAT`, `Bwd IAT` features)
  - Original intrusion metadata (`Label`)
- **Serialized Feature Order:** Documented and preserved in [`./models/feature_columns.json`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models/feature_columns.json)

---

## 3. Model Configurations & Training Protocols

### 3.1 Model 1: Multinomial Logistic Regression
- **Algorithm:** L-BFGS Multiclass Logistic Regression
- **Feature Scaling:** `StandardScaler` (Z-score normalization: $\mu = 0, \sigma = 1$).
  - *Strict Rule:* The scaler was **fitted strictly on $X_{train}$** and then applied to transform $X_{train}$ and $X_{test}$.
- **Hyperparameter Configuration:**
  ```json
  {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42
  }
  ```
- **Convergence Behavior:**
  - The optimizer successfully converged within **403 iterations** (well below the 1,000 max iteration threshold).
  - *Convergence Status:* **Converged (True)**.

### 3.2 Model 2: Decision Tree Classifier
- **Algorithm:** Optimized CART (Classification and Regression Trees) via Gini Impurity
- **Feature Scaling:** Raw numeric feature space (tree algorithms are invariant to monotonic monotonic feature scaling).
- **Hyperparameter Configuration:**
  ```json
  {
    "criterion": "gini",
    "max_depth": 16,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": 42
  }
  ```
- **Structural Properties:**
  - **Actual Max Depth:** 16 levels
  - **Total Leaf Nodes:** 2,651 leaves

### 3.3 Model 3: Random Forest Classifier (Memory-Conscious)
- **Algorithm:** Bootstrap Aggregated (Bagged) Ensemble of 100 Decision Trees
- **Feature Scaling:** Raw numeric feature space.
- **Hardware Optimization:** `n_jobs = -1` (parallel multi-threaded CPU execution).
- **Hyperparameter Configuration:**
  ```json
  {
    "n_estimators": 100,
    "max_depth": 16,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "n_jobs": -1,
    "random_state": 42
  }
  ```
- **Structural Properties:**
  - **Total Trees Trained:** 100 estimators
  - Individual trees constrained to max depth of 16 to keep memory footprint under 35 MB.

---

## 4. Empirical Runtime & Model Size Measurements

All values represent actual, measured executions on the system:

| Model Architecture | Training Time (s) | Prediction Time on 40k Test Set (s) | Serialized File Size (MB) | Disk Path |
| :--- | :---: | :---: | :---: | :--- |
| **StandardScaler** | 0.2657 s | *Included in LR* | 0.003 MB (3.3 KB) | [`./models/feature_scaler.joblib`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models/feature_scaler.joblib) |
| **Logistic Regression** | 31.84 s | 0.0084 s | 0.002 MB (2.4 KB) | [`./models/logistic_regression.joblib`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models/logistic_regression.joblib) |
| **Decision Tree** | 3.29 s | 0.0293 s | 0.448 MB (469 KB) | [`./models/decision_tree.joblib`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models/decision_tree.joblib) |
| **Random Forest** | 13.23 s | 0.3063 s | 29.614 MB | [`./models/random_forest.joblib`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models/random_forest.joblib) |

---

## 5. Memory & Runtime Observations

1. **Memory Ceiling Compliance (8 GB RAM):**
   - Peak memory consumption throughout the entire training process remained **under 650 MB**, well within the 8 GB host budget.
   - Constraining Random Forest to `max_depth = 16` and `min_samples_leaf = 4` proved highly effective: the 100-tree model required only **13.23 seconds** to fit and serialized to a lean **29.61 MB** bundle.
2. **Inference Latency:**
   - Prediction latency on the entire 40,000-record test set was extraordinarily fast:
     - Logistic Regression: **8.4 ms** (~0.21 $\mu$s per flow)
     - Decision Tree: **29.3 ms** (~0.73 $\mu$s per flow)
     - Random Forest: **306.3 ms** (~7.66 $\mu$s per flow)
   - All three models are fully capable of high-throughput real-time packet flow classification.

---

## 6. Serialized Pipeline Artifacts Created

The [`./models/`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models) directory contains all components required for deterministic evaluation and inference:

1. [`./models/logistic_regression.joblib`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models/logistic_regression.joblib) — Trained Logistic Regression model.
2. [`./models/decision_tree.joblib`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models/decision_tree.joblib) — Trained Decision Tree model.
3. [`./models/random_forest.joblib`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models/random_forest.joblib) — Trained Random Forest ensemble.
4. [`./models/feature_scaler.joblib`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models/feature_scaler.joblib) — Fitted `StandardScaler` (trained strictly on $X_{train}$).
5. [`./models/feature_columns.json`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models/feature_columns.json) — Ordered list of the 58 predictor column names.
6. [`./models/congestion_target_parameters.json`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models/congestion_target_parameters.json) — Fitted normalization extrema and TIS tercile thresholds.
7. [`./results/training_summary.json`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/results/training_summary.json) — Structured metadata summary of training run.

---

## 7. Warnings & Convergence Analysis

- **Convergence Status:** All models converged cleanly without hitting iteration bounds.
- **Library Deprecation Notices:** A harmless `FutureWarning` from scikit-learn 1.9 was noted (`penalty` parameter deprecation scheduled for v1.10 in favor of default L2 / `C`), but parameter execution was completely unaffected.
- **Scientific Protocol Reminder:** In accordance with the methodology, **no classification accuracy, confusion matrices, or F1-scores were computed or claimed in this step**. Thorough performance evaluation against the unseen test partition will be performed in the subsequent evaluation stage.
