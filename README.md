# Internet Traffic Congestion Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIC--IDS2017-green.svg)](https://www.unb.ca/cic/datasets/ids-2017.html)

A machine learning application that predicts network traffic congestion states (`LOW`, `MEDIUM`, `HIGH`) from network flow telemetry derived from the benchmark **CIC-IDS2017** dataset. Built with an 8 GB RAM memory-safe engineering architecture and paired with a modern Streamlit web dashboard.

---

## 🌟 Key Features

- **Production-Ready Random Forest Model:** Achieved **85.39% Test Accuracy** and **85.53% Macro F1-Score** on 40,000 unseen network flows.
- **Strict Target Leakage Prevention:** All rate metrics (`Flow Bytes/s`, `Flow Packets/s`, `Flow Duration`, and all 14 `IAT` features) are isolated in a quarantine zone, ensuring the model learns genuine packet morphology, TCP window buffer headroom, and duty-cycle patterns.
- **Interactive Streamlit Dashboard (`app.py`):**
  - **Dashboard Overview:** Project background, taxonomy, and system telemetry stats.
  - **Congestion Prediction:**
    - **Option A (Batch CSV Upload):** Upload flow records, execute batch inference, view summary statistics, and download annotated CSVs.
    - **Option B (Single-Flow Prediction):** Form with 7 categorized sections (58 network features) and one-click test presets.
  - **Model Performance:** Side-by-side comparative evaluations (Random Forest vs. Decision Tree vs. Logistic Regression) with confusion matrices.
  - **Feature Importance:** Top 10 feature breakdown using Gini impurity reduction.
  - **About the Project:** Architectural documentation and methodology summaries.

---

## 🚦 Congestion Level Taxonomy

The ground-truth target is formulated as a mathematically defined **Traffic Intensity Score (TIS)** combining normalized logarithmic bandwidth demand and packet processing load:

$$\text{TIS} = 0.5 \cdot \text{norm}(\log_{10}(1 + \text{Flow Bytes/s})) + 0.5 \cdot \text{norm}(\log_{10}(1 + \text{Flow Packets/s}))$$

| Level | TIS Range | Network Characteristics |
| :--- | :--- | :--- |
| 🟢 **LOW** | $\le 0.2894$ | Baseline quiet traffic, prolonged idle pauses (median ~5.86s), high client window buffer headroom, large MTU payloads. |
| 🟡 **MEDIUM** | $0.2894 < \text{TIS} \le 0.5255$ | Moderate transmission load, intermittent bursts, transitional buffer strain. |
| 🔴 **HIGH** | $> 0.5255$ | Saturated links, zero-idle continuous transmission, small-packet flood dynamics (median ~16 bytes). |

---

## 📊 Model Evaluation Summary

Evaluated on 40,000 holdout flows from `dataset/processed_test.csv`:

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Test Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Production)** | **85.39%** | **85.73%** | **85.45%** | **85.53%** | 0.82 s |
| **Decision Tree** | 84.85% | 84.97% | 84.87% | 84.91% | 0.09 s |
| **Logistic Regression** | 70.38% | 70.92% | 70.43% | 70.59% | 0.03 s |

---

## 🛠️ Project Structure

```text
├── app.py                             # Streamlit Web Application
├── requirements.txt                   # Application dependencies
├── .gitignore                         # Git exclusion rules
├── src/
│   ├── predict.py                     # Production inference engine & schema validation
│   ├── train_models.py                # Model training pipeline
│   ├── preprocess.py                  # Chunked preprocessor & feature quarantine
│   ├── evaluate_models.py             # Model evaluation suite
│   ├── feature_importance.py          # Random Forest feature importance analysis
│   └── create_working_dataset.py      # Stratified sampling from raw CIC-IDS2017
├── models/
│   ├── random_forest.joblib           # Trained production model (100 trees, depth 16)
│   ├── decision_tree.joblib           # Trained decision tree model
│   ├── logistic_regression.joblib     # Trained baseline logistic regression
│   ├── feature_columns.json           # Exact 58 predictor column schema
│   ├── congestion_target_parameters.json # TIS normalization & tercile thresholds
│   └── demonstration_presets.json     # Pre-extracted test flows for single prediction
├── results/
│   ├── model_comparison.csv           # Evaluation results across all 3 models
│   ├── evaluation_metrics.json        # Detailed metrics and inference times
│   ├── feature_importance.csv         # 58-feature importance ranking
│   ├── graphs/                        # Visualizations & EDA distributions
│   └── confusion_matrices/            # Evaluation confusion matrix plots
├── dataset/
│   └── processed_test.csv             # Unseen test flows (available for batch upload testing)
└── docs / reports:
    ├── CONGESTION_LABEL_METHODOLOGY.md
    ├── PREPROCESSING_REPORT.md
    ├── EDA_REPORT.md
    ├── MODEL_TRAINING_REPORT.md
    ├── MODEL_EVALUATION.md
    ├── FEATURE_IMPORTANCE_REPORT.md
    └── PREDICTION_SYSTEM_REPORT.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Sreeman-1007/ML-casestudy.git
cd ML-casestudy
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📜 Academic Disclaimer
This software is a machine learning capstone demonstration platform that evaluates statistical flow-level traffic intensity proxies. It does not interface directly with physical router queue counters or execute active network packet modification.
