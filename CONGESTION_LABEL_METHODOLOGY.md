# Congestion Label Methodology & Target Leakage Architecture

## 1. Executive Summary & Problem Formulation

### 1.1 The Fundamental Problem with CIC-IDS2017's Original Labels
The CIC-IDS2017 dataset was originally curated by the Canadian Institute for Cybersecurity to benchmark **Network Intrusion Detection Systems (NIDS)**. 

Its ground-truth `Label` column categorizes flows into:
- Baseline benign network activity (`BENIGN`)
- 14 distinct cyber-attack vectors (`DDoS`, `PortScan`, `DoS Hulk`, `FTP-Patator`, `SSH-Patator`, `Bot`, `Web Attacks`, etc.)

**Why the original `Label` cannot be used for congestion prediction:**
1. **Semantic Mismatch:** Intrusion events denote malicious intent or protocol violations, whereas network congestion represents a physical/logical queuing state where transmission demands exceed channel capacity or buffer availability.
2. **Attacks $\ne$ Congestion:** 
   - A low-rate attack (e.g., `Infiltration`, slow application brute-forcing, or `DoS Slowhttptest`) occupies negligible bandwidth and creates minimal queue pressure, yet it is labeled as an attack.
   - Legitimate high-bandwidth transfers (e.g., large file backups, 4K video streaming, bulk scientific data transfers) can severely congest a router interface while remaining 100% `BENIGN`.
3. **Absence of Native Congestion Ground Truth:** CIC-IDS2017 does **not** provide router queue lengths, drop counts, or explicit Low/Medium/High congestion classes.

Therefore, for this machine learning project, a **theoretically grounded and defensible Congestion-Level Proxy** must be formulated from empirical flow telemetry.

---

## 2. Telemetry Inventory: Available Congestion-Related Features

An empirical audit of [`./dataset/working_dataset.csv`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/dataset/working_dataset.csv) (200,000 records, 78 columns) reveals five primary functional dimensions:

| Dimension | Telemetry Features Present in Dataset | Physical / Networking Relevance |
| :--- | :--- | :--- |
| **Volumetric Rates** | `Flow Bytes/s`, `Flow Packets/s`, `Fwd Packets/s`, `Bwd Packets/s` | Direct throughput and packet processing strain on routers/switches. |
| **Duration & Timing** | `Flow Duration`, `Flow IAT Mean/Std/Max/Min`, `Fwd IAT...`, `Bwd IAT...` | Flow lifetime and inter-packet gap compression during burst transmissions. |
| **Packet Morphology** | `Total Length of Fwd/Bwd Packets`, `Packet Length Mean/Std/Var/Min/Max`, `Average Packet Size` | Sizing characteristics: large MTU-saturating transfers vs. micro-packet storms. |
| **Protocol Dynamics** | `SYN`, `ACK`, `PSH`, `FIN`, `RST`, `URG`, `ECE` Flag Counts; `Init_Win_bytes_forward/backward` | Flow control behavior, window exhaustion, TCP handshake states, and retransmission signals. |
| **Duty Cycle & Bursts** | `Active Mean/Std/Max/Min`, `Idle Mean/Std/Max/Min`, `act_data_pkt_fwd`, `Subflow Fwd/Bwd Packets/Bytes` | Duty cycle: continuous link occupancy vs. sparse interactive polling. |

---

## 3. Analysis of Target Leakage: The Critical Scientific Risk

### 3.1 What is Target Leakage in Congestion Proxy Modeling?
Target leakage occurs when information from the target variable is inadvertently introduced into the training features. In derived-proxy projects, target leakage often takes the form of **mathematical tautology**:

$$\text{If } \text{Target} = f(X_1, X_2) \quad \text{and} \quad \{X_1, X_2\} \subseteq X_{train}$$

A machine learning model (such as a Decision Tree or Random Forest) will simply discover the inverse decision boundary of $f(X_1, X_2)$. This yields **artificially inflated metrics (e.g., 99.8% accuracy)** while learning zero genuine networking behavior.

### 3.2 The Subtle Algebraic Coupling in Network Telemetry
In network flow monitoring, rate and timing features are algebraically linked:
1. **Packet Arrival Rate:** $\text{Flow Packets/s} \approx \frac{\text{Total Packets}}{\text{Flow Duration}}$
2. **Mean Inter-Arrival Time:** $\text{Flow IAT Mean} \approx \frac{\text{Flow Duration}}{\text{Total Packets} - 1}$
3. **The Reciprocal Coupling:**
   $$\text{Flow Packets/s} \approx \frac{1}{\text{Flow IAT Mean}}$$

**Empirical Evidence:** In our working dataset, the Pearson correlation between $\log(\text{Flow Packets/s})$ and $\log(\text{Flow IAT Mean})$ is **$-0.989$**, and with $\log(\text{Flow Duration})$ is **$-0.984$**.

> [!WARNING]
> If `Flow Packets/s` or `Flow Bytes/s` is used to define the Congestion Target, and features like `Flow Duration` or `Flow IAT Mean` are left in the training set, the model will trivially invert the inter-arrival time to reconstruct the rate and achieve near-perfect target leakage.

### 3.3 Leakage Mitigation Protocol: Feature Quarantine
To guarantee scientific validity, all features that directly constitute or algebraically mirror the proxy formula must be placed into a **Strict Feature Quarantine** and excluded from ML training.

---

## 4. Evaluation of Candidate Methodologies

### Candidate 1: Multi-Criteria Volumetric Traffic Rate Index (Recommended)
- **Concept:** Congestion in queuing theory (Kleinrock / Little's Law) is governed by two simultaneous resource strains: **link bandwidth saturation** (Bytes/sec) and **switching/queuing overhead** (Packets/sec).
- **Formulation:** A composite index combining log-transformed, normalized `Flow Bytes/s` and `Flow Packets/s`.
- **Thresholding:** Balanced quantiles (terciles at 33.3% and 66.7%).
- **Pros:** Transparent, grounded in network performance standards (RFC 2544), mathematically robust, easy to decouple from predictors.
- **Cons:** Excludes rate and IAT features from ML input, requiring the model to rely on packet sizing, TCP flags, window buffer sizes, and duty cycles.

### Candidate 2: Multi-Factor Resource Strain Index (Volume + Window + Duty Cycle)
- **Concept:** Blends throughput rate with TCP initial window sizes (`Init_Win_bytes_forward`) and active duty cycle ($\frac{\text{Active}}{\text{Active} + \text{Idle}}$).
- **Pros:** Considers host transport aggressiveness alongside raw link rate.
- **Cons:** Encroaches upon protocol dynamics, requiring `Init_Win` and `Active/Idle` to be quarantined as well. This leaves too few remaining features for ML training and introduces arbitrary subjective weightings ($w_1, w_2, w_3$).

### Candidate 3: Unsupervised Clustering Proxy (K-Means on Telemetry)
- **Concept:** Cluster flows into 3 clusters using K-Means or GMM.
- **Pros:** Does not require an explicit algebraic formula.
- **Cons:** Highly sensitive to severe data skew; typically creates heavily imbalanced clusters (e.g., 85% Low, 12% Medium, 3% High), obscuring physical interpretability and creating acute class imbalance.

---

## 5. Recommended Methodology: Strict Train-Split-First Target Parameter Fitting

To uphold strict machine learning standards and prevent both **target leakage** and **train-to-test data leakage**, target-definition parameters will NOT be fitted across the whole dataset globally. Instead, the working dataset will be split into training and testing sets **before** fitting any normalization parameters or thresholds, and all parameters will be computed programmatically rather than hard-coded.

### 5.1 Step 1: Pre-Split Train/Test Partitioning
Before computing any normalization parameters, percentiles, or thresholds:
1. Partition the 200,000-row working dataset into:
   - **Training Set (80%):** 160,000 records
   - **Test Set (20%):** 40,000 records
2. Set a fixed, reproducible seed: `random_state = 42`.
3. The test set is set aside untouched to serve as genuine, out-of-sample data.

### 5.2 Step 2: Input Metric Cleaning (Training Portion)
Let $B_{train}$ be `Flow Bytes/s` and $P_{train}$ be `Flow Packets/s` in the training split.
1. Impute missing/infinite values (caused by 0-duration flows in CICFlowMeter) by clipping to zero or dataset bounds:
   $$B_{train} = \max(0, B_{train}), \quad P_{train} = \max(0, P_{train})$$

### 5.3 Step 3: Non-Linear Logarithmic Compression (Training Portion)
Network flow rates span over 8 orders of magnitude (from single bytes to gigabits). A log transformation stabilizes variance and prevents extreme outliers from compressing the scale:
$$\tilde{B}_{train} = \log_{10}(1 + B_{train})$$
$$\tilde{P}_{train} = \log_{10}(1 + P_{train})$$

### 5.4 Step 4: Fit Normalization Parameters Programmatically on Training Set
Calculate and save the empirical extrema from the **training set only**:
- $B_{min}^{train} = \min(\tilde{B}_{train}), \quad B_{max}^{train} = \max(\tilde{B}_{train})$
- $P_{min}^{train} = \min(\tilde{P}_{train}), \quad P_{max}^{train} = \max(\tilde{P}_{train})$

Normalize training rates to the $[0, 1]$ interval:
$$S_{B, train} = \frac{\tilde{B}_{train} - B_{min}^{train}}{B_{max}^{train} - B_{min}^{train}}$$
$$S_{P, train} = \frac{\tilde{P}_{train} - P_{min}^{train}}{P_{max}^{train} - P_{min}^{train}}$$

### 5.5 Step 5: Traffic Intensity Score (TIS)
Combine bandwidth strain and packet processing overhead with equal weighting ($w = 0.5$):
$$\text{TIS}_{train} = 0.5 \cdot S_{B, train} + 0.5 \cdot S_{P, train} \quad \in [0, 1]$$

### 5.6 Step 6: Programmatic Quantile Threshold Calculation on Training Set
> [!IMPORTANT]
> **No Hard-Coded Thresholds:** Threshold values must NEVER be hard-coded. They are derived programmatically using empirical percentiles from $\text{TIS}_{train}$ alone:
- **Lower Tercile Threshold:** $\tau_{33.3} = \text{Percentile}(\text{TIS}_{train}, 33.333)$
- **Upper Tercile Threshold:** $\tau_{66.7} = \text{Percentile}(\text{TIS}_{train}, 66.667)$

Assign Training Congestion Classes:
$$\text{Congestion Level}_{train} = \begin{cases}
\mathbf{LOW}, & \text{if } \text{TIS}_{train} \le \tau_{33.3} \\
\mathbf{MEDIUM}, & \text{if } \tau_{33.3} < \text{TIS}_{train} \le \tau_{66.7} \\
\mathbf{HIGH}, & \text{if } \text{TIS}_{train} > \tau_{66.7}
\end{cases}$$

This guarantees perfectly balanced training classes (~33.33% LOW, ~33.33% MEDIUM, ~33.34% HIGH) on the training partition without synthetic oversampling.

### 5.7 Step 7: Persistence of Fitted Target-Definition Parameters
All fitted target-definition parameters must be saved programmatically to a structured metadata file (e.g., [`./models/congestion_target_parameters.json`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models/congestion_target_parameters.json)) containing:
```json
{
  "train_random_state": 42,
  "train_ratio": 0.8,
  "log_bytes_min": "<calculated programmatically>",
  "log_bytes_max": "<calculated programmatically>",
  "log_pkts_min": "<calculated programmatically>",
  "log_pkts_max": "<calculated programmatically>",
  "tau_33_3": "<calculated programmatically from training TIS>",
  "tau_66_7": "<calculated programmatically from training TIS>",
  "class_mapping": {"0": "LOW", "1": "MEDIUM", "2": "HIGH"}
}
```
This enables exact, deterministic reproducibility for downstream modeling, validation, and real-time inference in Streamlit.

### 5.8 Step 8: Out-of-Sample Test Set Transformation
To evaluate the machine learning models under strict zero-leakage conditions, the test set is transformed **using the exact fitted parameters saved from the training set**:
1. Clean and log-transform test flow rates:
   $$\tilde{B}_{test} = \log_{10}(1 + \max(0, B_{test})), \quad \tilde{P}_{test} = \log_{10}(1 + \max(0, P_{test}))$$
2. Scale test rates using the **training extrema**:
   $$S_{B, test} = \frac{\tilde{B}_{test} - B_{min}^{train}}{B_{max}^{train} - B_{min}^{train}}, \quad S_{P, test} = \frac{\tilde{P}_{test} - P_{min}^{train}}{P_{max}^{train} - P_{min}^{train}}$$
3. Compute test TIS:
   $$\text{TIS}_{test} = 0.5 \cdot S_{B, test} + 0.5 \cdot S_{P, test}$$
4. Categorize test records using the **training thresholds** ($\tau_{33.3}$ and $\tau_{66.7}$):
   $$\text{Congestion Level}_{test} = \begin{cases}
   \mathbf{LOW}, & \text{if } \text{TIS}_{test} \le \tau_{33.3} \\
   \mathbf{MEDIUM}, & \text{if } \tau_{33.3} < \text{TIS}_{test} \le \tau_{66.7} \\
   \mathbf{HIGH}, & \text{if } \text{TIS}_{test} > \tau_{66.7}
   \end{cases}$$
This ensures zero information from the test set is ever used to influence normalization bounds or classification cutoffs.


---

## 6. Target Leakage & Feature Separation Architecture

To ensure models learn realistic structural relationships, features are divided into **Allowed Predictors** and **Quarantined Exclusions**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ALL 78 FEATURES IN DATASET                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌─────────────────────────────────┐   ┌──────────────────────────────────┐
│      QUARANTINED / EXCLUDED     │   │        ALLOWED ML FEATURES       │
│        (Target Leakage)         │   │        (Predictor Set X)         │
├─────────────────────────────────┤   ├──────────────────────────────────┤
│ • Flow Bytes/s (constituent)    │   │ • Packet Sizing & Statistics     │
│ • Flow Packets/s (constituent)  │   │ • TCP Control Flags              │
│ • Fwd Packets/s, Bwd Packets/s  │   │ • TCP Window & Buffer Metrics    │
│ • Flow Duration (algebraic)     │   │ • Active / Idle Dynamics         │
│ • All Flow & Fwd/Bwd IAT feats  │   │ • Header & Segment Overheads     │
│ • Original Label (not target)   │   │ • Destination Port               │
└─────────────────────────────────┘   └──────────────────────────────────┘
```

### 6.1 Quarantined Features (EXCLUDED from ML Training)
These features are prohibited from entering the feature matrix $X$:
1. `Flow Bytes/s` *(Direct formula component)*
2. `Flow Packets/s` *(Direct formula component)*
3. `Fwd Packets/s`, `Bwd Packets/s` *(Direct rate components)*
4. `Flow Duration` *(Direct denominator of rate)*
5. `Flow IAT Mean`, `Flow IAT Std`, `Flow IAT Max`, `Flow IAT Min` *(Algebraic inverse of packet rate, Corr = -0.989)*
6. `Fwd IAT Total`, `Fwd IAT Mean`, `Fwd IAT Std`, `Fwd IAT Max`, `Fwd IAT Min` *(Directional IAT rates)*
7. `Bwd IAT Total`, `Bwd IAT Mean`, `Bwd IAT Std`, `Bwd IAT Max`, `Bwd IAT Min` *(Directional IAT rates)*
8. `Label` *(Original intrusion classification)*
9. `Fwd Header Length.1` *(Already pruned duplicate)*

### 6.2 Allowed Features for ML Models (The Predictor Matrix $X$)
The remaining **60 features** represent legitimate structural indicators of network flows:
- **Packet Length & Morphology (17 features):**
  `Total Length of Fwd Packets`, `Total Length of Bwd Packets`, `Fwd Packet Length Max/Min/Mean/Std`, `Bwd Packet Length Max/Min/Mean/Std`, `Min/Max Packet Length`, `Packet Length Mean/Std/Variance`, `Average Packet Size`, `Avg Fwd/Bwd Segment Size`.
- **Packet Counts & Totals (5 features):**
  `Total Fwd Packets`, `Total Backward Packets`, `Subflow Fwd Packets`, `Subflow Bwd Packets`, `act_data_pkt_fwd`.
- **Subflow Volumes (2 features):**
  `Subflow Fwd Bytes`, `Subflow Bwd Bytes`.
- **TCP Control Flags (12 features):**
  `FIN Flag Count`, `SYN Flag Count`, `RST Flag Count`, `PSH Flag Count`, `ACK Flag Count`, `URG Flag Count`, `CWE Flag Count`, `ECE Flag Count`, `Fwd PSH Flags`, `Bwd PSH Flags`, `Fwd URG Flags`, `Bwd URG Flags`.
- **TCP Buffer & Window Dynamics (4 features):**
  `Init_Win_bytes_forward`, `Init_Win_bytes_backward`, `min_seg_size_forward`, `Down/Up Ratio`.
- **Header Lengths (2 features):**
  `Fwd Header Length`, `Bwd Header Length`.
- **Active & Idle Duty Cycles (8 features):**
  `Active Mean/Std/Max/Min`, `Idle Mean/Std/Max/Min`.
- **Bulk Transfer Metrics (6 features):**
  `Fwd Avg Bytes/Bulk`, `Fwd Avg Packets/Bulk`, `Fwd Avg Bulk Rate`, `Bwd Avg Bytes/Bulk`, `Bwd Avg Packets/Bulk`, `Bwd Avg Bulk Rate`.
- **Network Port / Protocol (1 feature):**
  `Destination Port` (distinguishes web, database, interactive, and ephemeral ports).

---

## 7. Model Comparison Roadmap (For Next Stage)

With the balanced, leakage-safe dataset, the next stage will compare three diverse classification paradigms:
1. **Logistic Regression (Multinomial with L2 regularization):** Serves as the transparent, linear decision-boundary baseline.
2. **Decision Tree Classifier (CART):** Evaluates non-linear, hierarchical feature splits and rule interpretability.
3. **Random Forest Classifier (Ensemble of bagged trees):** Leverages feature bagging to minimize variance, model complex multi-feature interactions, and generate reliable Gini feature importance rankings.

---

## 8. Limitations & Scientific Assumptions

1. **Passive Flow Approximation:** Because CIC-IDS2017 was captured passively at network tap points, true internal router metrics (queue occupancy, buffer drops, packet discards) are unavailable. The proxy assumes that sustained high flow-level volumetric rates are strong indicators of potential queue congestion.
2. **Tercile Uniformity Assumption:** The 33.3% / 66.7% thresholding strategy enforces equal class sizes. In a real ISP network, extreme congestion is typically a rare event (<5% of time). The balanced design is chosen intentionally to benchmark classification algorithms fairly without class collapse.
3. **Flow Scope:** Each record represents an individual flow summary rather than a multi-flow aggregate link snapshot.

---

## 9. Recommended Project Terminology

### Comparison of Project Descriptions:
- **A. Internet Traffic Congestion Prediction:**
  - *Assessment:* Catchy and standard for college project syllabus titles. However, "Prediction" implies forecasting future states rather than classifying an ongoing flow.
- **B. Network Traffic Congestion-Level Estimation:**
  - *Assessment:* Scientifically accurate for determining the operating state (Low/Medium/High) of an observed network flow.
- **C. Network Traffic Congestion Proxy Prediction:**
  - *Assessment:* Methodologically exact because it explicitly acknowledges the use of a derived proxy, but slightly redundant in presentation contexts.

### Official Project Recommendation:
- **Formal Academic Title:**
  **"Internet Traffic Congestion Prediction Using Machine Learning"** *(Retains the approved project title for academic submission, reports, and UI)*.
- **Scientific Subtitle & Technical Framing:**
  **"A Supervised Estimation Framework Using Network Traffic Intensity Proxies"** *(Defends the methodology rigorously during project presentations and technical evaluation)*.
