# Preprocessing & Congestion Target Creation Report

## 1. Executive Summary
- **Project Title:** Internet Traffic Congestion Prediction Using Machine Learning
- **Source Working Dataset:** [`./dataset/working_dataset.csv`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/dataset/working_dataset.csv)
- **Output Processed Training Dataset:** [`./dataset/processed_train.csv`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/dataset/processed_train.csv)
- **Output Processed Testing Dataset:** [`./dataset/processed_test.csv`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/dataset/processed_test.csv)
- **Preprocessing Configuration Artifact:** [`./results/preprocessing_config.json`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/results/preprocessing_config.json)
- **Fitted Target Parameters Artifact:** [`./models/congestion_target_parameters.json`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models/congestion_target_parameters.json)
- **Class Distribution Plot:** [`./results/congestion_distribution.png`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/results/congestion_distribution.png)
- **Split Configuration:** 80% Train (160,000 records) / 20% Test (40,000 records) with `random_state = 42`

---

## 2. Dataset Dimensions & Volume Audit

| Stage | Record Count | Column Count | Disk Size |
| :--- | :---: | :---: | :---: |
| **Starting Working Dataset** | 200,000 | 78 | 63.78 MB |
| **Processed Training Set (`processed_train.csv`)** | 160,000 | 59 (58 predictors + 1 target) | 31.41 MB |
| **Processed Test Set (`processed_test.csv`)** | 40,000 | 59 (58 predictors + 1 target) | 7.86 MB |
| **Total Processed Records** | **200,000** | **59** | **39.27 MB** |

- **Rows Removed:** **0 rows removed**. Every individual network flow is preserved.
- **Columns Before Preprocessing:** 78 columns
- **Columns After Preprocessing:** 59 columns (58 numerical predictors + `Congestion_Level`)
- **Columns Removed (Quarantine / Target Exclusion):** 20 columns

---

## 3. Data Cleaning & Anomaly Handling

### 3.1 Missing Value (`NaN`) Handling
- **Observation:** In the training split, `Flow Bytes/s` contained 48 null values (0.03%), while `Flow Packets/s` and all 58 predictor features had **0 null values**.
- **Root Cause:** Zero-microsecond flow durations (`Flow Duration = 0`) where 0 total bytes produced an undefined $0/0$ floating-point division in the CICFlowMeter engine.
- **Handling:** For target score calculation, missing `Flow Bytes/s` values were imputed with $0.0$ (representing negligible transmission rate). The target column calculation handles this gracefully, and `Flow Bytes/s` was subsequently excluded from the predictor matrix. The resulting predictor matrices have **0 null values**.

### 3.2 Infinity (`+Inf`, `-Inf`) and Negative Value Handling
- **Observation:** In the training split, `Flow Bytes/s` had 89 `+Inf` values and 5 negative values (due to 32-bit signed integer counter overflow). `Flow Packets/s` had 137 `+Inf` values and 6 negative values.
- **Root Cause:** Flows transmitting packets with sub-microsecond timestamps produced $N/0 = +\text{Inf}$ in CICFlowMeter. Negative numbers represent integer overflow wrappers.
- **Handling:**
  - Negative values were clipped to $0.0$ via $\max(0, x)$.
  - Positive infinities (`+Inf`) were capped at the **maximum finite value observed in the training split**:
    - Maximum finite `Flow Bytes/s`: **2,070,000,000.00 bytes/sec** (~2.07 GB/s)
    - Maximum finite `Flow Packets/s`: **3,000,000.00 packets/sec** (~3.0 Mpps)
  - This preserves the physical reality that instantaneous packet bursts represent the highest observed bandwidth saturation without inducing floating-point errors.

### 3.3 Duplicate Records Handling
- **Observation:** 7,470 exact duplicate rows exist across the 200,000 records.
- **Justification for Retention:** In network packet inspection, duplicate records are legitimate and expected phenomena. Port scans, SYN floods, keepalive pings, and repetitive DNS probes generate thousands of identical flow summaries across discrete timestamps. Deduplication would artificially penalize burst and attack frequencies. Therefore, all 200,000 records are retained.

---

## 4. Congestion Target Formulation: Step-by-Step Programmatic Fitting

Target parameters were fitted **strictly on the 80% training partition (160,000 records)** and applied out-of-sample to the 20% test partition (40,000 records).

### 4.1 Mathematical Formulation
1. **Logarithmic Compression:**
   $$\tilde{B}_{train} = \log_{10}(1 + \max(0, B_{train}))$$
   $$\tilde{P}_{train} = \log_{10}(1 + \max(0, P_{train}))$$
2. **Fitted Training Range Extrema:**
   - $\tilde{B}_{min}^{train} = 0.0000, \quad \tilde{B}_{max}^{train} = 9.315970$
   - $\tilde{P}_{min}^{train} = 0.0000, \quad \tilde{P}_{max}^{train} = 6.477121$
3. **Normalized Sub-Scores:**
   $$S_{B, train} = \frac{\tilde{B}_{train} - 0.0}{9.315970 - 0.0}, \quad S_{P, train} = \frac{\tilde{P}_{train} - 0.0}{6.477121 - 0.0}$$
4. **Traffic Intensity Score (TIS):**
   $$\text{TIS}_{train} = 0.5 \cdot S_{B, train} + 0.5 \cdot S_{P, train} \quad \in [0, 1]$$
5. **Programmatically Derived Empirical Terciles on Training Set:**
   - **Lower Tercile ($\tau_{33.3}$):** **`0.28941254`**
   - **Upper Tercile ($\tau_{66.7}$):** **`0.52553162`**

### 4.2 Class Mapping Rule
$$\text{Congestion\_Level} = \begin{cases}
\mathbf{LOW}, & \text{if } \text{TIS} \le 0.28941254 \\
\mathbf{MEDIUM}, & \text{if } 0.28941254 < \text{TIS} \le 0.52553162 \\
\mathbf{HIGH}, & \text{if } \text{TIS} > 0.52553162
\end{cases}$$

### 4.3 Test Set Out-of-Sample Transformation
The test set was transformed using the exact training extrema ($\tilde{B}_{max}^{train} = 9.315970, \tilde{P}_{max}^{train} = 6.477121$) and training thresholds ($\tau_{33.3} = 0.28941254, \tau_{66.7} = 0.52553162$). Zero information from the test set influenced normalization bounds or classification thresholds.

---

## 5. Final Class Distribution

The resulting class balance is stable across both training and holdout test partitions:

| Congestion Level | Training Set (160,000 rows) | Test Set (40,000 rows) | Combined (200,000 rows) |
| :--- | :---: | :---: | :---: |
| **LOW** | 53,334 (33.33%) | 13,500 (33.75%) | 66,834 (33.42%) |
| **MEDIUM** | 53,333 (33.33%) | 13,324 (33.31%) | 66,657 (33.33%) |
| **HIGH** | 53,333 (33.33%) | 13,176 (32.94%) | 66,509 (33.25%) |
| **Total** | **160,000 (100.0%)** | **40,000 (100.0%)** | **200,000 (100.0%)** |

![Congestion Class Distribution](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/results/congestion_distribution.png)

---

## 6. Target Leakage Prevention & Feature Separation

To prevent models from simply memorizing algebraic identities, 20 features were quarantined and eliminated from the predictor matrix:

### 6.1 Quarantined Features (20 Excluded Columns)
1. `Flow Bytes/s` *(Direct constituent of TIS formula)*
2. `Flow Packets/s` *(Direct constituent of TIS formula)*
3. `Fwd Packets/s` *(Direct directional arrival rate component)*
4. `Bwd Packets/s` *(Direct directional arrival rate component)*
5. `Flow Duration` *(Direct denominator of rate equations, Corr = -0.984 with rate)*
6. `Flow IAT Mean` *(Mathematical inverse of packet arrival rate: $1 / \text{Rate}$, Corr = -0.989)*
7. `Flow IAT Std` *(Coupled jitter dynamics)*
8. `Flow IAT Max` *(Coupled inter-arrival timing)*
9. `Flow IAT Min` *(Coupled inter-arrival timing)*
10. `Fwd IAT Total` *(Directional duration equivalent)*
11. `Fwd IAT Mean` *(Directional packet arrival rate inverse)*
12. `Fwd IAT Std` *(Directional timing jitter)*
13. `Fwd IAT Max` *(Directional timing bounds)*
14. `Fwd IAT Min` *(Directional timing bounds)*
15. `Bwd IAT Total` *(Directional duration equivalent)*
16. `Bwd IAT Mean` *(Directional packet arrival rate inverse)*
17. `Bwd IAT Std` *(Directional timing jitter)*
18. `Bwd IAT Max` *(Directional timing bounds)*
19. `Bwd IAT Min` *(Directional timing bounds)*
20. `Label` *(Original intrusion classification, not a predictor and not the target)*

### 6.2 Allowed Predictor Features (58 Features in $X$)
The models are trained strictly on physical packet sizing, protocol flags, buffer dynamics, and duty cycle patterns:
- **Packet Length & Morphology (17):** `Total Length of Fwd Packets`, `Total Length of Bwd Packets`, `Fwd Packet Length Max/Min/Mean/Std`, `Bwd Packet Length Max/Min/Mean/Std`, `Min/Max Packet Length`, `Packet Length Mean/Std/Variance`, `Average Packet Size`, `Avg Fwd/Bwd Segment Size`
- **Packet Counts & Totals (5):** `Total Fwd Packets`, `Total Backward Packets`, `Subflow Fwd Packets`, `Subflow Bwd Packets`, `act_data_pkt_fwd`
- **Subflow Volumes (2):** `Subflow Fwd Bytes`, `Subflow Bwd Bytes`
- **TCP Control Flags (12):** `FIN Flag Count`, `SYN Flag Count`, `RST Flag Count`, `PSH Flag Count`, `ACK Flag Count`, `URG Flag Count`, `CWE Flag Count`, `ECE Flag Count`, `Fwd PSH Flags`, `Bwd PSH Flags`, `Fwd URG Flags`, `Bwd URG Flags`
- **TCP Window & Segment Overhead (4):** `Init_Win_bytes_forward`, `Init_Win_bytes_backward`, `min_seg_size_forward`, `Down/Up Ratio`
- **Header Lengths (2):** `Fwd Header Length`, `Bwd Header Length`
- **Duty Cycle & Bursts (8):** `Active Mean/Std/Max/Min`, `Idle Mean/Std/Max/Min`
- **Bulk Transfer Metrics (6):** `Fwd/Bwd Avg Bytes/Bulk`, `Avg Packets/Bulk`, `Avg Bulk Rate`
- **Network Port (1):** `Destination Port`

---

## 7. Limitations & Preprocessing Constraints

1. **Proxy Formulation:** The target represents flow-level volumetric intensity rather than router queue drops (which are not available in passive PCAP datasets).
2. **Fixed Quantile Assumption:** Terciles enforce equal class proportions to ensure balanced training; in production ISP environments, severe congestion is an episodic tail-event.
3. **RAM Optimization:** Datasets were processed with in-place assignments and memory-safe transformations, keeping peak RAM consumption during preprocessing below 250 MB.
