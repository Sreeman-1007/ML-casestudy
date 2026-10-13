# Working Dataset Creation Report

## 1. Executive Summary & Overview
- **Project Title:** Internet Traffic Congestion Prediction Using Machine Learning
- **Dataset Source:** CIC-IDS2017 (Canadian Institute for Cybersecurity)
- **Target Record Count:** 200,000 records
- **Actual Records Selected:** **200,000 records**
- **Output File Path:** [`./dataset/working_dataset.csv`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/dataset/working_dataset.csv)
- **Output File Size on Disk:** **63.78 MB** (66,878,884 bytes)
- **Number of Columns:** **78 columns**
- **Reproducibility Seed (`random_state`):** `42`

---

## 2. Source CSV Files & Sampling Methodology

### Source Files Used
The full dataset comprises **8 CSV files** (~843.68 MB on disk, 2,830,743 total rows) located in `./dataset/MachineLearningCVE/`. None of the original CSV files were altered or modified.

| # | Source CSV File Name | Total Original Records | Sampled Records | Stratum Share |
| :-: | :--- | :---: | :---: | :---: |
| 1 | `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 225,745 | 25,000 | 12.5% |
| 2 | `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 286,467 | 25,000 | 12.5% |
| 3 | `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 191,033 | 25,000 | 12.5% |
| 4 | `Monday-WorkingHours.pcap_ISCX.csv` | 529,918 | 25,000 | 12.5% |
| 5 | `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 288,602 | 25,000 | 12.5% |
| 6 | `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 170,366 | 25,000 | 12.5% |
| 7 | `Tuesday-WorkingHours.pcap_ISCX.csv` | 445,909 | 25,000 | 12.5% |
| 8 | `Wednesday-workingHours.pcap_ISCX.csv` | 692,703 | 25,000 | 12.5% |
| **Total** | **8 Source CSV Files** | **2,830,743** | **200,000** | **100.0%** |

### Sampling Method & Algorithm
To prevent sampling bias toward days with large captures (e.g., Wednesday with 692k rows vs. Thursday Morning with 170k rows) while avoiding Out-Of-Memory (OOM) conditions:
1. **Stratified Session Quota:** Exactly 25,000 records were allocated to each of the 8 capture scenarios, ensuring equal representation across every weekday and traffic condition (benign baseline, brute-force, web attacks, botnets, port scans, and volumetric floods).
2. **Deterministic Random Indexing:** For each source file, indices in range $[0, N_{file}-1]$ were sampled uniformly without replacement using `np.random.default_rng(seed=42)`.
3. **Memory-Safe Chunked Filtering (`chunksize=50000`):** Each CSV was processed strictly in 50,000-row chunks via `pd.read_csv(..., chunksize=50000)`. Only records whose global index matched the preselected random index set were extracted into memory.
4. **Incremental Streaming Write:** Collected records for each file were appended directly to `./dataset/working_dataset.csv`, and in-memory DataFrames were garbage-collected immediately. At no point was the entire dataset loaded into RAM.

---

## 3. Column Selection & Schema Description

The raw CIC-IDS2017 files contain 79 columns with inconsistent whitespace padding in headers and an artifact duplication (`Fwd Header Length.1`). 

### Transformations Applied:
1. **Header Normalization:** Leading/trailing whitespace was stripped (`str.strip()`) across all columns.
2. **Deduplication:** The column `Fwd Header Length.1` (column index 55) was excluded because it is an exact, redundant copy of `Fwd Header Length` (column index 34) caused by a known CICFlowMeter logging duplicate.
3. **Retained Features:** Exactly **78 columns** are preserved in the working dataset (77 numerical network features + 1 original label).

### Complete List of Columns (78)
1. `Destination Port`
2. `Flow Duration`
3. `Total Fwd Packets`
4. `Total Backward Packets`
5. `Total Length of Fwd Packets`
6. `Total Length of Bwd Packets`
7. `Fwd Packet Length Max`
8. `Fwd Packet Length Min`
9. `Fwd Packet Length Mean`
10. `Fwd Packet Length Std`
11. `Bwd Packet Length Max`
12. `Bwd Packet Length Min`
13. `Bwd Packet Length Mean`
14. `Bwd Packet Length Std`
15. `Flow Bytes/s`
16. `Flow Packets/s`
17. `Flow IAT Mean`
18. `Flow IAT Std`
19. `Flow IAT Max`
20. `Flow IAT Min`
21. `Fwd IAT Total`
22. `Fwd IAT Mean`
23. `Fwd IAT Std`
24. `Fwd IAT Max`
25. `Fwd IAT Min`
26. `Bwd IAT Total`
27. `Bwd IAT Mean`
28. `Bwd IAT Std`
29. `Bwd IAT Max`
30. `Bwd IAT Min`
31. `Fwd PSH Flags`
32. `Bwd PSH Flags`
33. `Fwd URG Flags`
34. `Bwd URG Flags`
35. `Fwd Header Length`
36. `Bwd Header Length`
37. `Fwd Packets/s`
38. `Bwd Packets/s`
39. `Min Packet Length`
40. `Max Packet Length`
41. `Packet Length Mean`
42. `Packet Length Std`
43. `Packet Length Variance`
44. `FIN Flag Count`
45. `SYN Flag Count`
46. `RST Flag Count`
47. `PSH Flag Count`
48. `ACK Flag Count`
49. `URG Flag Count`
50. `CWE Flag Count`
51. `ECE Flag Count`
52. `Down/Up Ratio`
53. `Average Packet Size`
54. `Avg Fwd Segment Size`
55. `Avg Bwd Segment Size`
56. `Fwd Avg Bytes/Bulk`
57. `Fwd Avg Packets/Bulk`
58. `Fwd Avg Bulk Rate`
59. `Bwd Avg Bytes/Bulk`
60. `Bwd Avg Packets/Bulk`
61. `Bwd Avg Bulk Rate`
62. `Subflow Fwd Packets`
63. `Subflow Fwd Bytes`
64. `Subflow Bwd Packets`
65. `Subflow Bwd Bytes`
66. `Init_Win_bytes_forward`
67. `Init_Win_bytes_backward`
68. `act_data_pkt_fwd`
69. `min_seg_size_forward`
70. `Active Mean`
71. `Active Std`
72. `Active Max`
73. `Active Min`
74. `Idle Mean`
75. `Idle Std`
76. `Idle Max`
77. `Idle Min`
78. `Label`

---

## 4. Original Label Distribution in Working Dataset

> [!NOTE]
> The original `Label` represents intrusion detection events (attacks vs. benign baseline). **It is retained exclusively for context and analysis, and is NOT treated as a ground-truth congestion label.**

| Original Label | Count | Percentage | Traffic Nature |
| :--- | :---: | :---: | :--- |
| `BENIGN` | 161,589 | 80.79% | Normal baseline background traffic |
| `DDoS` | 14,234 | 7.12% | Volumetric distributed attack bursts |
| `PortScan` | 13,817 | 6.91% | Rapid connection/probing packets |
| `DoS Hulk` | 8,151 | 4.08% | Heavy HTTP request flooding |
| `FTP-Patator` | 445 | 0.22% | Brute-force authentication bursts |
| `DoS GoldenEye` | 402 | 0.20% | Layer-7 denial of service bursts |
| `SSH-Patator` | 338 | 0.17% | SSH protocol connection bursts |
| `Bot` | 263 | 0.13% | Botnet C&C and attack communication |
| `Web Attack - Brute Force` | 232 | 0.12% | Application login probe bursts |
| `DoS Slowhttptest` | 222 | 0.11% | Slow-rate HTTP connection exhaustion |
| `DoS slowloris` | 198 | 0.10% | Slow headers resource holding |
| `Web Attack - XSS` | 103 | 0.05% | Malicious payload web traffic |
| `Infiltration` | 4 | 0.002% | Post-exploitation internal traffic |
| `Web Attack - Sql Injection` | 2 | 0.001% | Database injection web traffic |
| **Total** | **200,000** | **100.00%** | Comprehensive cross-scenario sample |

---

## 5. Missing & Infinite Values Audit

In accordance with the instruction not to silently alter or remove rows prematurely:
- **`Flow Bytes/s`:** **61 null (`NaN`) values** (0.0305%) and **110 infinite (`Inf`) values** (0.055%).
- **`Flow Packets/s`:** **0 null values**, but **171 infinite (`Inf`) values** (0.0855%).
- **All other 76 columns:** 0 null values (100% complete).

**Root Cause:** These occurrences are an artifact of the CICFlowMeter calculation where a network flow has a `Flow Duration = 0` microseconds, resulting in a mathematical division by zero. All 200,000 rows were retained intact and will be handled during the planned preprocessing stage.

---

## 6. Memory Profile & 8 GB RAM Feasibility

### Measured Footprint
- **File size on disk:** 63.78 MB
- **In-memory size (64-bit float/int, uncompressed):** ~120.19 MB
- **In-memory size with downcasting (`float32`, `int32`):** **~58.5 MB**
- **Available RAM on system:** ~3.22 GB

### Why 200,000 Records is Optimal for 8 GB RAM:
1. **Training Efficiency:** Standard tree models (Random Forest, Gradient Boosting, XGBoost/LightGBM) on 200,000 samples and ~20–30 selected features require only **300 MB to 600 MB of peak memory** during cross-validation, easily fitting into available RAM without triggering paging/swapping.
2. **Statistical Robustness:** 200,000 rows provides substantial power to capture tail distributions, bursty traffic events, and rare congestion indicators without requiring full 2.8M row memory penalties.
3. **Fast Iteration:** Training cycles, hyperparameter searches, and evaluation metrics complete in seconds to a few minutes rather than hours.

---

## 7. Limitations of the Sampling Strategy

1. **Rare Attack Class Sparsity:** Because sampling was performed uniformly per file rather than oversampling rare minority attacks, micro-classes like `Heartbleed` (which only had 11 records across 692,000 rows in Wednesday's file) were not captured, and `Infiltration` has only 4 instances. (Note: Since the objective of this project is **traffic congestion prediction** rather than cyber-attack classification, this does not hinder congestion modeling).
2. **Uniform Day Weighting:** Each file contributed an equal 25,000 records regardless of raw capture duration or raw packet volume. While this balances cross-day behavior, it slightly over-represents smaller captures (e.g., Thursday Morning) relative to large ones (e.g., Wednesday).
