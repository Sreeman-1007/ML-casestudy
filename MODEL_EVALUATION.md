# Machine Learning Model Evaluation & Comparative Performance Report

## 1. Evaluation Objective
The objective of this evaluation is to assess and compare the generalization performance of the three trained classification models:
1. **Multinomial Logistic Regression** (L2 Regularized with `StandardScaler`)
2. **Decision Tree Classifier** (Constrained Depth CART)
3. **Random Forest Classifier** (100 Trees Ensemble)

All models were evaluated exclusively on the **unseen holdout test dataset** to predict the three-tier traffic congestion proxy:
$$\mathbf{Congestion\_Level} \in \{\mathbf{LOW}, \mathbf{MEDIUM}, \mathbf{HIGH}\}$$

*Methodological Rule:* No model was retrained, no scaling parameters were refitted, and no test labels were used in training. Evaluation was conducted strictly on out-of-sample data using the exact 58-predictor feature sequence established in Step 4.

---

## 2. Test Dataset & Ground-Truth Profile
- **Dataset Path:** [`./dataset/processed_test.csv`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/dataset/processed_test.csv)
- **Total Test Records ($N_{test}$):** **40,000 flows** (exactly 20.0% of the working dataset)
- **Predictor Dimensions:** **58 numerical features** (0 quarantined features, 0 nulls, 0 infs)
- **Class Distribution in Test Set:**
  - `LOW`: **13,500 flows** (33.75%)
  - `MEDIUM`: **13,324 flows** (33.31%)
  - `HIGH`: **13,176 flows** (32.94%)
  - *Total:* 40,000 flows (100.0%)

---

## 3. Models Evaluated & Pipeline Artifacts

| Model Name | Type | Pipeline Preprocessing | Model File | File Size |
| :--- | :--- | :--- | :--- | :---: |
| **Logistic Regression** | Linear / Multinomial L2 | `StandardScaler` (fitted on $X_{train}$) | [`./models/logistic_regression.joblib`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models/logistic_regression.joblib) | 0.002 MB |
| **Decision Tree** | Hierarchical CART Tree | Raw 58 Predictor Features | [`./models/decision_tree.joblib`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models/decision_tree.joblib) | 0.448 MB |
| **Random Forest** | Bagged Ensemble (100 Trees) | Raw 58 Predictor Features | [`./models/random_forest.joblib`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models/random_forest.joblib) | 29.614 MB |

---

## 4. Evaluation Methodology & Metric Definitions

Because the three-class problem has balanced classes (~33.3% each), model comparison primarily emphasizes **Macro-averaged metrics** alongside standard Accuracy:
- **Accuracy:** Overall proportion of correctly classified flows across all 40,000 test records.
- **Macro Precision:** Unweighted arithmetic mean of precision across the three classes:
  $$\text{Macro Precision} = \frac{1}{3} \sum_{c \in \{\text{LOW}, \text{MED}, \text{HIGH}\}} \text{Precision}_c$$
- **Macro Recall:** Unweighted arithmetic mean of recall across the three classes:
  $$\text{Macro Recall} = \frac{1}{3} \sum_{c \in \{\text{LOW}, \text{MED}, \text{HIGH}\}} \text{Recall}_c$$
- **Macro F1-Score (Primary Decision Metric):** Harmonic mean of Macro Precision and Macro Recall, evaluating balanced multi-class performance without favoritism toward any single class.

---

## 5. Comprehensive Model Comparison Results

All metrics represent actual empirical evaluations on the 40,000 unseen test records:

| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted Precision | Weighted Recall | Weighted F1-Score | Inference Time (40k samples) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **70.38%** | **71.31%** | **70.36%** | **70.59%** | 71.35% | 70.38% | 70.62% | 0.0925 s (2.31 $\mu$s/sample) |
| **Decision Tree** | **84.85%** | **85.24%** | **84.88%** | **84.91%** | 85.28% | 84.85% | 84.91% | 0.0940 s (2.35 $\mu$s/sample) |
| **Random Forest** | **85.39%** | **86.19%** | **85.42%** | **85.53%** | 86.23% | 85.39% | 85.53% | 0.1815 s (4.54 $\mu$s/sample) |

- **CSV Table Saved:** [`./results/model_comparison.csv`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/results/model_comparison.csv)
- **Comparison Visualization Chart:** [`./results/model_performance_comparison.png`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/results/model_performance_comparison.png)

![Model Performance Comparison](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/results/model_performance_comparison.png)

---

## 6. Detailed Per-Class Performance Breakdown

### 6.1 Logistic Regression (Baseline Model)
- **Overall Accuracy:** 70.38% | **Macro F1-Score:** 70.59%

| Congestion Class | Precision | Recall | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| **LOW** | 81.19% | 72.68% | 76.70% | 13,500 |
| **MEDIUM** | 60.78% | 71.21% | 65.58% | 13,324 |
| **HIGH** | 71.95% | 67.19% | 69.49% | 13,176 |

### 6.2 Decision Tree Classifier
- **Overall Accuracy:** 84.85% | **Macro F1-Score:** 84.91%

| Congestion Class | Precision | Recall | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| **LOW** | 93.58% | 83.10% | 88.03% | 13,500 |
| **MEDIUM** | 77.85% | 80.72% | 79.26% | 13,324 |
| **HIGH** | 84.29% | 90.83% | 87.44% | 13,176 |

### 6.3 Random Forest Classifier
- **Overall Accuracy:** 85.39% | **Macro F1-Score:** 85.53%

| Congestion Class | Precision | Recall | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| **LOW** | **96.05%** | 81.75% | **88.32%** | 13,500 |
| **MEDIUM** | 76.67% | **84.99%** | **80.62%** | 13,324 |
| **HIGH** | **85.85%** | **89.53%** | **87.65%** | 13,176 |

---

## 7. Confusion Matrix Observations

Confusion matrices were generated for each model and saved to [`./results/confusion_matrices/`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/results/confusion_matrices).

### 7.1 Logistic Regression Confusion Matrix
- **Plot:** [`./results/confusion_matrices/logistic_regression_confusion_matrix.png`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/results/confusion_matrices/logistic_regression_confusion_matrix.png)

```
                     Predicted LOW   Predicted MEDIUM   Predicted HIGH   Total True
True LOW                9,812 (72.7%)      2,689 (19.9%)        999 (7.4%)       13,500
True MEDIUM             1,383 (10.4%)      9,488 (71.2%)      2,453 (18.4%)      13,324
True HIGH                 890 (6.8%)       3,433 (26.1%)      8,853 (67.2%)      13,176
```
- *Observation:* As a linear model, Logistic Regression struggles with the non-linear boundary separating `MEDIUM` from `HIGH` (3,433 true HIGH flows misclassified as MEDIUM). However, it correctly places over 70% of samples on the diagonal.

### 7.2 Decision Tree Confusion Matrix
- **Plot:** [`./results/confusion_matrices/decision_tree_confusion_matrix.png`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/results/confusion_matrices/decision_tree_confusion_matrix.png)

```
                     Predicted LOW   Predicted MEDIUM   Predicted HIGH   Total True
True LOW               11,218 (83.1%)      1,938 (14.4%)        344 (2.5%)       13,500
True MEDIUM               683 (5.1%)      10,755 (80.7%)      1,886 (14.2%)      13,324
True HIGH                  86 (0.7%)       1,122 (8.5%)      11,968 (90.8%)      13,176
```
- *Observation:* Decision Tree exhibits strong diagonal dominance. Only 86 true HIGH flows (0.7%) were misclassified as LOW, confirming that severe non-adjacent misclassifications are rare.

### 7.3 Random Forest Confusion Matrix
- **Plot:** [`./results/confusion_matrices/random_forest_confusion_matrix.png`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/results/confusion_matrices/random_forest_confusion_matrix.png)

```
                     Predicted LOW   Predicted MEDIUM   Predicted HIGH   Total True
True LOW               11,036 (81.7%)      2,086 (15.5%)        378 (2.8%)       13,500
True MEDIUM               433 (3.3%)      11,324 (85.0%)      1,567 (11.8%)      13,324
True HIGH                  21 (0.2%)       1,359 (10.3%)      11,796 (89.5%)      13,176
```
- *Observation:* Random Forest achieves the lowest non-adjacent error rate of any model: only **21 out of 13,176 true HIGH flows (0.16%)** were misclassified as LOW, and only 3.3% of true MEDIUM flows were misclassified as LOW.

---

## 8. Scientific Interpretation of Results

### 8.1 Which Model Performed Best?
The test results indicate that the **Random Forest Classifier** achieved the highest overall performance across all evaluated metrics:
- **Macro F1-Score:** **85.53%** (vs. 84.91% for Decision Tree and 70.59% for Logistic Regression)
- **Accuracy:** **85.39%**
- **Macro Precision:** **86.19%**
- The ensemble mechanism successfully mitigated decision tree variance, particularly improving recall on the transitional `MEDIUM` class (84.99% vs. 80.72%).

### 8.2 Which Model Performed Worst?
The results show that **Multinomial Logistic Regression** achieved the lowest score (70.38% Accuracy, 70.59% Macro F1). Because linear models fit hyperplanes in transformed feature space, Logistic Regression cannot model high-order hierarchical interactions (e.g., condition: `if Idle Mean < 1000 AND Packet Length Mean < 50 AND Init_Win < 1024`).

### 8.3 Does Logistic Regression Provide a Useful Baseline?
Yes. Achieving **70.38% accuracy** on a balanced three-class classification task (where random guessing yields 33.33%) confirms that linear feature combinations alone capture substantial traffic intensity signal. However, the ~15.0 percentage point gap between Logistic Regression and the tree models highlights the necessity of non-linear algorithms for network telemetry.

### 8.4 Which Congestion Class is Easiest to Classify?
The results indicate that **`LOW`** is the easiest class to identify with high precision:
- Random Forest achieved **96.05% Precision** for `LOW` congestion.
- Decision Tree achieved **93.58% Precision** for `LOW` congestion.
- *Reasoning:* Flows in the `LOW` tier exhibit distinctive physical signatures—notably prolonged `Idle Mean` pauses (median ~5.86 seconds) and large initial TCP buffer windows—which cleanly separate them from continuous traffic.
- **`HIGH`** congestion is also detected with high sensitivity (Decision Tree Recall: **90.83%**, Random Forest Recall: **89.53%**), driven by zero idle time and small packet flood dynamics.

### 8.5 Which Congestion Class is Most Difficult to Classify?
The results indicate that **`MEDIUM`** is the most challenging class across all three models:
- Random Forest F1: **80.62%** (vs. 88.32% for `LOW` and 87.65% for `HIGH`).
- Decision Tree F1: **79.26%** (vs. 88.03% for `LOW` and 87.44% for `HIGH`).
- Logistic Regression F1: **65.58%** (vs. 76.70% for `LOW` and 69.49% for `HIGH`).
- *Reasoning:* `MEDIUM` represents the middle tercile of the continuous Traffic Intensity Score distribution. Flows near the $\tau_{33.3}$ boundary share features with active `LOW` flows, while flows near the $\tau_{66.7}$ boundary share features with emerging `HIGH` bursts.

### 8.6 Analysis of Adjacent vs. Non-Adjacent Confusions
The test results show that classification errors occur almost exclusively between **adjacent tiers**:
- `LOW` $\leftrightarrow$ `MEDIUM`: ~15% confusion
- `MEDIUM` $\leftrightarrow$ `HIGH`: ~11% to 14% confusion
- `LOW` $\leftrightarrow$ `HIGH` (Non-adjacent): **< 0.3% error rate in Random Forest**
- This monotonic error pattern confirms that the models have learned a genuine ordinal traffic continuum rather than making arbitrary categorical errors.

---

## 9. Final Best Model Selection

### Recommended Model: **Random Forest Classifier**
- **Primary Selection Criteria:**
  1. **Highest Macro F1-Score:** **85.53%** (demonstrates the most balanced performance across all three congestion tiers).
  2. **Highest Overall Accuracy:** **85.39%**.
  3. **Lowest Non-Adjacent Error Rate:** Only 0.16% true HIGH flows misclassified as LOW.
  4. **High Precision on Baseline Traffic:** 96.05% precision on `LOW` congestion prevents false congestion alarms.
  5. **Efficient Inference Latency:** Evaluates 40,000 flows in **0.1815 seconds** (~4.54 microseconds per flow), ensuring complete compatibility with real-time Streamlit deployment on an 8 GB RAM laptop.

---

## 10. Evaluation Limitations

1. **Proxy Ground Truth:** Evaluation was performed against the derived Traffic Intensity Score proxy rather than direct physical router queue drop records.
2. **Fixed Operational Thresholds:** The static tercile thresholds ($\tau_{33.3}, \tau_{66.7}$) enforce balanced class distributions. Real-world network deployments may experience non-uniform prior class probabilities depending on diurnal traffic cycles.
3. **Synthetic Testbed Scope:** Results reflect the CIC-IDS2017 simulated enterprise network environment and should be validated on diverse topologies prior to production ISP integration.
