# Exploratory Data Analysis (EDA) Report

## 1. EDA Objective
The objective of this exploratory data analysis is to investigate the structural patterns, distributions, correlations, and variabilities within the preprocessed network flow features of the training dataset. Specifically, this analysis examines whether the **58 retained predictor features** (from which all direct rate and inter-arrival time target-defining features have been strictly quarantined) exhibit discernible, non-trivial associations with the newly defined three-tier target:
$$\mathbf{Congestion\_Level} \in \{\text{LOW}, \text{MEDIUM}, \text{HIGH}\}$$

*Note: In accordance with scientific methodology, observed relationships are treated as statistical associations/patterns and not direct causal mechanisms.*

---

## 2. Dataset & Partition Specifications
- **Source Files Analyzed:**
  - Training Partition: [`./dataset/processed_train.csv`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/dataset/processed_train.csv)
  - Testing Partition: [`./dataset/processed_test.csv`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/dataset/processed_test.csv)
  - Pipeline Configuration: [`./results/preprocessing_config.json`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/results/preprocessing_config.json)
- **Sample Sizes:**
  - **Training Set:** **160,000 records** (80.0%)
  - **Holdout Test Set:** **40,000 records** (20.0%)
  - **Total:** 200,000 records
- **Feature Dimensions:**
  - 58 Numerical Predictor Features
  - 1 Categorical Target Variable (`Congestion_Level`)
  - Total: 59 columns per record (0 missing values, 0 infinite values)

---

## 3. Target Class Distribution
The target variable was created using empirical tercile thresholds fitted on the training split's Traffic Intensity Score ($\tau_{33.3} = 0.289413, \tau_{66.7} = 0.525532$).

| Congestion Level | Training Count | Training Percentage | Test Count | Test Percentage |
| :--- | :---: | :---: | :---: | :---: |
| **LOW** | 53,334 | 33.33% | 13,500 | 33.75% |
| **MEDIUM** | 53,333 | 33.33% | 13,324 | 33.31% |
| **HIGH** | 53,333 | 33.33% | 13,176 | 32.94% |
| **Total** | **160,000** | **100.00%** | **40,000** | **100.00%** |

- **Graph Generated:** [`./results/graphs/congestion_class_distribution.png`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/results/graphs/congestion_class_distribution.png)
- **Observation:** The training set is balanced across all three tiers, preventing majority-class bias and eliminating the need for synthetic oversampling techniques like SMOTE.

---

## 4. Feature Selection Rationale for In-Depth Analysis
From the 58 available predictors, **12 representative features** were selected across key network telemetry categories to investigate distributions and relationships:

| # | Selected Feature | Network Category | Domain Rationale for Inclusion |
| :-: | :--- | :--- | :--- |
| 1 | `Packet Length Mean` | Packet Morphology | Measures central tendency of payload sizing; separates large MTU transfers from small-packet flooding. |
| 2 | `Average Packet Size` | Packet Morphology | Global bidirectional segment scale; reflects overall protocol payload density. |
| 3 | `Packet Length Std` | Packet Morphology | Variability of packet sizing; high variance indicates rich interactive/media streams, low variance indicates automated probing. |
| 4 | `Total Fwd Packets` | Volume Intensity | Primary measure of client-to-server transmission activity and request volume. |
| 5 | `Total Backward Packets` | Volume Intensity | Server-to-client response packet volume; identifies asymmetric traffic flows. |
| 6 | `Total Length of Fwd Packets` | Payload Volume | Cumulative client payload in bytes; directly measures aggregate data push. |
| 7 | `Init_Win_bytes_forward` | TCP Buffering | Client TCP receive window advertisement; signals host buffering headroom and TCP scaling dynamics. |
| 8 | `Init_Win_bytes_backward` | TCP Buffering | Server TCP window buffer advertisement; reflects receiver-side backpressure. |
| 9 | `Active Mean` | Duty Cycle | Average active burst duration before entering idle pauses; identifies sustained transmission sessions. |
| 10 | `Idle Mean` | Duty Cycle | Average silence duration between active bursts; separates persistent streams from intermittent polling. |
| 11 | `ACK Flag Count` | Control Flags | Transport acknowledgment frequency; signals session completion and handshake behavior. |
| 12 | `Destination Port` | Transport Service | Distinguishes well-known server ports (80 HTTP, 443 HTTPS, 53 DNS) from ephemeral service ports. |

---

## 5. Feature Distribution Observations
Individual distribution plots were generated and saved in [`./results/graphs/feature_distributions/`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/results/graphs/feature_distributions):

1. **Extreme Positive Skewness:**
   - Features like `Total Fwd Packets` (skewness: 86.8), `Total Backward Packets` (skewness: 88.3), and `Total Length of Fwd Packets` (skewness: 72.1) exhibit extreme positive skew. Over 75% of flows contain $\le 4$ packets, while a sparse tail of volumetric flows contains up to 1,600+ packets. Logarithmic scaling ($\log_{10}(1 + x)$) resolves these multi-modal distributions for visualization.
2. **Bimodal Morphology:**
   - `Packet Length Mean` and `Average Packet Size` show bimodal spikes around small control headers (~0 to 60 bytes) and standard Ethernet MTU sizes (~1,400 to 1,500 bytes).
3. **Discrete Windows & Ephemeral Ports:**
   - `Init_Win_bytes_forward` concentrates around discrete power-of-two window sizes (e.g., 256, 1024, 8192, 29200, 65535), typical of standard OS TCP stack configurations.
   - `Destination Port` clusters heavily around port 80 (HTTP), 443 (HTTPS), 53 (DNS), and high ephemeral ports ($> 49152$).

---

## 6. Congestion vs. Feature Observations (Empirical Summary)
Comparative box plots across `LOW`, `MEDIUM`, and `HIGH` tiers were saved in [`./results/graphs/congestion_relationships/`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/results/graphs/congestion_relationships).

Numerical comparisons from [`./results/eda_feature_summary.csv`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/results/eda_feature_summary.csv):

| Feature | LOW Mean | LOW Median | MEDIUM Mean | MEDIUM Median | HIGH Mean | HIGH Median | Observed Pattern |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Packet Length Mean` | 265.69 | 87.45 | 192.37 | 65.33 | 45.19 | 12.25 | Declines monotonically as congestion tier rises. |
| `Average Packet Size` | 287.45 | 111.00 | 220.72 | 86.00 | 56.56 | 16.33 | Flows in HIGH tier are dominated by short control segments. |
| `Packet Length Std` | 523.00 | 150.85 | 308.04 | 33.41 | 32.42 | 3.46 | Variance collapses sharply in HIGH tier (uniform packet sizes). |
| `Total Fwd Packets` | 9.83 | 5.00 | 6.87 | 2.00 | 18.06 | 2.00 | HIGH tier exhibits substantial mean tail volume (18.1 pkts). |
| `Total Backward Packets` | 9.34 | 5.00 | 8.08 | 2.00 | 21.96 | 1.00 | HIGH tier shows asymmetric server bursts (mean 22.0 pkts). |
| `Total Length of Fwd Packets` | 1,055.58 | 123.00 | 379.09 | 57.00 | 246.85 | 31.00 | Forward payload length is highest in LOW tier. |
| `Init_Win_bytes_forward` | 9,804.54 | 406.50 | 6,109.37 | -1.00 | 4,847.25 | 122.00 | LOW tier maintains significantly higher client TCP buffer allocations. |
| `Init_Win_bytes_backward` | 2,927.87 | 136.00 | 1,903.61 | -1.00 | 836.71 | -1.00 | Server window sizes shrink progressively toward HIGH tier. |
| `Active Mean` ($\mu$s) | 242,074.89 | 985.00 | 10,179.07 | 0.00 | 1,683.89 | 0.00 | LOW tier exhibits prolonged active session bursts. |
| `Idle Mean` ($\mu$s) | 18,744,198.16 | 5,862,258.00 | 99,721.90 | 0.00 | 577.26 | 0.00 | **Strongest separator:** LOW tier has massive inter-burst pauses (~18.7s mean). |
| `ACK Flag Count` | 0.37 | 0.00 | 0.26 | 0.00 | 0.27 | 0.00 | ACK frequency is higher in LOW tier sessions. |
| `Destination Port` | 3,901.79 | 80.00 | 8,788.91 | 53.00 | 11,882.74 | 443.00 | LOW centers on HTTP (80); MEDIUM on DNS (53); HIGH on HTTPS/ephemeral. |

---

## 7. Correlation Analysis
A full 50×50 correlation matrix of all non-constant predictors was rendered to [`./results/graphs/correlation_heatmap.png`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/results/graphs/correlation_heatmap.png).

### 7.1 Exact Collinear Pairs ($r = +1.0000$)
Several feature pairs in CIC-IDS2017 exhibit perfect mathematical redundancy:
1. `Total Fwd Packets` $\leftrightarrow$ `Subflow Fwd Packets` ($r = 1.0000$)
2. `Total Backward Packets` $\leftrightarrow$ `Subflow Bwd Packets` ($r = 1.0000$)
3. `Total Length of Fwd Packets` $\leftrightarrow$ `Subflow Fwd Bytes` ($r = 1.0000$)
4. `Total Length of Bwd Packets` $\leftrightarrow$ `Subflow Bwd Bytes` ($r = 1.0000$)
5. `Fwd Packet Length Mean` $\leftrightarrow$ `Avg Fwd Segment Size` ($r = 1.0000$)
6. `Bwd Packet Length Mean` $\leftrightarrow$ `Avg Bwd Segment Size` ($r = 1.0000$)
7. `Fwd PSH Flags` $\leftrightarrow$ `SYN Flag Count` ($r = 1.0000$)
8. `Fwd URG Flags` $\leftrightarrow$ `CWE Flag Count` ($r = 1.0000$)
9. `RST Flag Count` $\leftrightarrow$ `ECE Flag Count` ($r = 1.0000$)

*Explanation:* CICFlowMeter calculates subflow statistics identical to flow totals when flows are brief. Furthermore, "Segment Size" is an exact duplicate alias of "Packet Length Mean". These collinear features provide zero net new information and could be pruned in future feature selection to improve linear model stability.

### 7.2 Strongest Negative Correlations
- `Min Packet Length` $\leftrightarrow$ `PSH Flag Count` ($r = -0.4666$)
- `PSH Flag Count` $\leftrightarrow$ `ACK Flag Count` ($r = -0.4494$)
- `Bwd Packet Length Min` $\leftrightarrow$ `PSH Flag Count` ($r = -0.3948$)
- `Min Packet Length` $\leftrightarrow$ `ACK Flag Count` ($r = -0.3735$)

---

## 8. Feature Variability & Constant Feature Detection
Basic descriptive statistics were audited across all 58 predictors:

### 8.1 Zero-Variance (Constant) Features
**8 features** exhibit exactly zero variance ($\sigma = 0, \text{nunique} = 1$) across all 160,000 training records:
1. `Bwd PSH Flags` (constant 0)
2. `Bwd URG Flags` (constant 0)
3. `Fwd Avg Bytes/Bulk` (constant 0)
4. `Fwd Avg Packets/Bulk` (constant 0)
5. `Fwd Avg Bulk Rate` (constant 0)
6. `Bwd Avg Bytes/Bulk` (constant 0)
7. `Bwd Avg Packets/Bulk` (constant 0)
8. `Bwd Avg Bulk Rate` (constant 0)

*Context:* Bulk transfer features and backward PSH/URG flags were unpopulated by the CICFlowMeter logging tool during the 2017 capture. They carry zero predictive power and can safely be omitted during model training.

### 8.2 Near-Zero Variance Features
- `Fwd URG Flags` / `CWE Flag Count`: Only 15 non-zero entries across 160,000 rows (99.99% zeros).
- `SYN Flag Count` / `Fwd PSH Flags`: 4.5% non-zero entries.

---

## 9. Outlier Observations
Box plots analyzing heavy tails are saved in [`./results/graphs/outliers/`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/results/graphs/outliers):

1. **Volume Features:**
   - `Total Fwd Packets`: 18,220 records (11.39%) exceed the upper $1.5 \times \text{IQR}$ threshold. Maximum is 1,611 packets.
   - `Total Backward Packets`: 17,940 records (11.21%) exceed $1.5 \times \text{IQR}$. Maximum is 1,935 packets.
   - `Total Length of Bwd Packets`: 19,410 records (12.13%) exceed $1.5 \times \text{IQR}$. Maximum is 3.43 MB.
2. **Duty Cycle Features:**
   - `Active Mean`: 14,880 records (9.30%) exceed $1.5 \times \text{IQR}$. Maximum is 100 seconds.
   - `Idle Mean`: 31,520 records (19.70%) exceed $1.5 \times \text{IQR}$. Maximum is 120 seconds.
3. **Engineering Assessment:**
   In network traffic engineering, extreme packet counts and byte transfers represent **valid macroscopic events** (e.g., bulk file transfers, volumetric floods, long-running active database sessions). Deleting these outliers would strip the dataset of real high-intensity traffic phenomena. Non-linear tree algorithms (Decision Tree, Random Forest) are inherently robust to monotonic scale variations and will handle these heavy tails without artificial clipping.

---

## 10. Key Patterns Discovered
1. **The Idle-Time Divergence:**
   The `Idle Mean` metric acts as a primary discriminator between LOW congestion (median ~5.86 seconds of idle quietness) and MEDIUM/HIGH congestion (median 0 seconds). High-congestion traffic involves continuous, uninterrupted channel usage.
2. **Payload Size Inversion:**
   High traffic intensity is strongly associated with small average packet sizes (mean 45.2 bytes, median 12.3 bytes), reflecting rapid streams of small TCP control probes, SYN packets, and high-frequency short transactions. In contrast, LOW congestion flows exhibit larger, more variable payloads (mean 265.7 bytes).
3. **TCP Buffer Pressure:**
   Client and server TCP initial window sizes (`Init_Win_bytes_forward/backward`) diminish significantly from LOW to HIGH tiers, capturing the effect of receiver buffer exhaustion during sustained traffic bursts.

---

## 11. Candidate Feature Recommendations for Machine Learning
Based on variability, distinct medians across tiers, and low redundancy:
1. **Strong Non-Linear Predictors:**
   - `Idle Mean` (separates idle baseline from continuous traffic)
   - `Active Mean` (captures active burst duration)
   - `Packet Length Mean` / `Average Packet Size` (differentiates small floods from bulk payload)
   - `Packet Length Std` (captures packet uniformity)
   - `Total Fwd Packets` / `Total Backward Packets` (captures flow volume and asymmetry)
   - `Init_Win_bytes_forward` / `Init_Win_bytes_backward` (reflects TCP flow control state)
   - `Destination Port` (distinguishes service protocols)
   - `PSH Flag Count` / `ACK Flag Count` (captures push-data urgency and acknowledgment pace)
2. **Features Recommended for Pruning Prior to Linear Modeling:**
   - The 8 zero-variance constant columns.
   - Exact duplicate collinear pairs (keep `Total Fwd Packets`, drop `Subflow Fwd Packets`; keep `Packet Length Mean`, drop `Avg Fwd Segment Size`).

---

## 12. Limitations of the EDA
1. **Flow-Level Aggregation:** The data reflects summarized flow records rather than continuous packet-by-packet time-series queues. Time-of-day inter-flow queuing cannot be observed directly.
2. **Dataset Specificity:** The patterns reflect the Canadian Institute for Cybersecurity testbed environment (simulated benign enterprise traffic mixed with scripted attack bursts) and may differ in topologies with distinct MTU and link bandwidth configurations.
3. **No Direct Causal Claims:** Observed patterns indicate statistical correlation with our derived Traffic Intensity Score proxy rather than proving physical link packet drops.
