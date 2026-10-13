# Project Setup & Dataset Inspection Report

## 1. Project Title
**Internet Traffic Congestion Prediction Using Machine Learning**

---

## 2. Environment & System Information

- **Python Version:** `3.13.5` (Anaconda distribution, 64-bit AMD64)
- **Virtual Environment Status:** **Active & Functional**
  - **Environment Path:** `./.venv/` (`C:\Users\Sreeman\Downloads\intenret traffic congestion prediction\.venv`)
  - **Interpreter:** `./.venv/Scripts/python.exe`
  - **Core Libraries Installed:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `streamlit`
- **RAM Information:**
  - **Target System Specification:** 8.00 GB RAM
  - **OS Total Physical Memory:** 15.69 GB
  - **Currently Free / Available Memory:** ~3.22 GB
  - **Implication:** Memory-safe chunking and data type optimization are mandatory to avoid Out-Of-Memory (OOM) crashes.

---

## 3. Project Directory Structure

```
intenret traffic congestion prediction/
├── .venv/                          # Virtual environment with installed dependencies
├── dataset/
│   └── MachineLearningCVE/         # Official CIC-IDS2017 CSV files (8 files)
├── models/                         # Storage for trained ML models, scalers, and metadata
├── results/                        # Evaluation outputs, metrics, and logs
│   ├── confusion_matrices/         # Confusion matrix plots
│   └── graphs/                     # Feature importance, ROC curves, loss curves
├── src/                            # Modular Python source code (data pipeline, models, utils)
└── PROJECT_SETUP.md                # Initial project inspection report
```

---

## 4. Dataset Inspection (CIC-IDS2017)

- **Dataset Location:** `./dataset/MachineLearningCVE/`
- **Total Files:** 8 CSV files
- **Total On-Disk Size:** ~843.68 MB (~884,645,859 bytes)
- **Number of Columns:** 79 columns across all files

### CSV Files & Sizes

| File Name | Size (MB) | Raw Byte Count | Sample Missing Values | Sample Duplicate Rows | Original Labels Detected |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 73.55 MB | 77,123,859 | None in sample | 0 / 1000 | `BENIGN`, `DDoS` |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 73.34 MB | 76,906,168 | None in sample | 11 / 1000 | `BENIGN`, `PortScan` |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 55.62 MB | 58,316,725 | None in sample | 11 / 1000 | `BENIGN`, `Bot` |
| `Monday-WorkingHours.pcap_ISCX.csv` | 168.73 MB | 176,927,918 | None in sample | 51 / 1000 | `BENIGN` |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 79.25 MB | 83,102,436 | None in sample | 2 / 1000 | `BENIGN`, `Infiltration` |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 49.61 MB | 52,023,263 | None in sample | 14 / 1000 | `BENIGN`, `Web Attack - Brute Force`, `Web Attack - Sql Injection`, `Web Attack - XSS` |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 128.82 MB | 135,078,995 | `Flow Bytes/s` (4 nulls in sample) | 4 / 1000 | `BENIGN`, `FTP-Patator`, `SSH-Patator` |
| `Wednesday-workingHours.pcap_ISCX.csv` | 214.74 MB | 225,166,395 | None in sample | 4 / 1000 | `BENIGN`, `DoS GoldenEye`, `DoS Hulk`, `DoS Slowhttptest`, `DoS slowloris`, `Heartbleed` |

---

## 5. Column & Feature Categorization

All 8 files contain **79 columns**.

> [!NOTE]
> Column names in raw CSV files contain leading and trailing whitespace (e.g., `' Destination Port'`, `' Flow Duration'`, `' Label'`). These will be systematically stripped (`df.columns = df.columns.str.strip()`) during preprocessing.

### Breakdown of Available Network Features

1. **Flow Duration:**
   - `Flow Duration`
2. **Packet Counts & Segment Counts:**
   - `Total Fwd Packets`, `Total Backward Packets`, `Subflow Fwd Packets`, `Subflow Bwd Packets`, `act_data_pkt_fwd`
3. **Packet Lengths (Volume & Variability):**
   - `Total Length of Fwd Packets`, `Total Length of Bwd Packets`, `Fwd Packet Length Max`, `Fwd Packet Length Min`, `Fwd Packet Length Mean`, `Fwd Packet Length Std`, `Bwd Packet Length Max`, `Bwd Packet Length Min`, `Bwd Packet Length Mean`, `Bwd Packet Length Std`, `Min Packet Length`, `Max Packet Length`, `Packet Length Mean`, `Packet Length Std`, `Packet Length Variance`, `Average Packet Size`, `Avg Fwd Segment Size`, `Avg Bwd Segment Size`
4. **Flow Rates & Traffic Intensity:**
   - `Flow Bytes/s`, `Flow Packets/s`, `Fwd Packets/s`, `Bwd Packets/s`, `Down/Up Ratio`
5. **Inter-Arrival Times (IAT Dynamics):**
   - `Flow IAT Mean`, `Flow IAT Std`, `Flow IAT Max`, `Flow IAT Min`, `Fwd IAT Total`, `Fwd IAT Mean`, `Fwd IAT Std`, `Fwd IAT Max`, `Fwd IAT Min`, `Bwd IAT Total`, `Bwd IAT Mean`, `Bwd IAT Std`, `Bwd IAT Max`, `Bwd IAT Min`
6. **Active & Idle Time Behavior:**
   - `Active Mean`, `Active Std`, `Active Max`, `Active Min`, `Idle Mean`, `Idle Std`, `Idle Max`, `Idle Min`
7. **TCP Flags & Control Dynamics:**
   - `FIN Flag Count`, `SYN Flag Count`, `RST Flag Count`, `PSH Flag Count`, `ACK Flag Count`, `URG Flag Count`, `CWE Flag Count`, `ECE Flag Count`, `Fwd PSH Flags`, `Bwd PSH Flags`, `Fwd URG Flags`, `Bwd URG Flags`
8. **TCP Window & Header Overhead:**
   - `Init_Win_bytes_forward`, `Init_Win_bytes_backward`, `min_seg_size_forward`, `Fwd Header Length`, `Bwd Header Length`, `Fwd Header Length.1`
9. **Bulk Transfer Metrics:**
   - `Fwd Avg Bytes/Bulk`, `Fwd Avg Packets/Bulk`, `Fwd Avg Bulk Rate`, `Bwd Avg Bytes/Bulk`, `Bwd Avg Packets/Bulk`, `Bwd Avg Bulk Rate`
10. **Target & Metadata:**
    - `Destination Port`, `Label`

---

## 6. Original Label Information & Critical Methodological Rule

### Original Labels Present in Dataset
The dataset's `Label` column categorizes network flows into baseline non-malicious traffic and intrusion attack classes:
1. `BENIGN`
2. `Bot`
3. `DDoS`
4. `DoS GoldenEye`
5. `DoS Hulk`
6. `DoS Slowhttptest`
7. `DoS slowloris`
8. `FTP-Patator`
9. `Heartbleed`
10. `Infiltration`
11. `PortScan`
12. `SSH-Patator`
13. `Web Attack - Brute Force`
14. `Web Attack - Sql Injection`
15. `Web Attack - XSS`

### Methodological Rule for Congestion Prediction
> [!IMPORTANT]
> **CIC-IDS2017 does NOT directly contain network congestion labels (Low / Medium / High).**
> - The original labels represent **intrusion / security events**, not queueing or congestion states.
> - An attack does not automatically equal congestion, and high-volume legitimate traffic can cause congestion without being an attack.
> - Congestion labels will **not** be derived from the intrusion labels.
> - In subsequent phases, congestion ground truth will be constructed using empirical network traffic intensity features (e.g., flow rate thresholds, packet inter-arrival dynamics, queue/window indicators).
> - **No congestion labels or ML models have been created in this initial setup step.**

---

## 7. Memory-Safe Recommendations for an 8 GB RAM Computer

1. **Never Load All CSVs Simultaneously:** The raw CSV files total ~844 MB on disk and expand to 5–8 GB in memory when parsed into standard 64-bit Python objects.
2. **Chunked Processing (`chunksize`):** When reading large CSV files, always process in chunks (e.g., `chunksize=25000` to `50000`).
3. **Data Type Downcasting:**
   - Cast `int64` to `int32` or `int16` / `uint16` (reduces memory by 50–75%).
   - Cast `float64` to `float32` (reduces memory by 50%).
4. **Column Filtering (`usecols`):** Only load the subset of features needed for analysis and modeling.
5. **Stratified / Representative Sampling:** Create an aggregated, balanced, memory-safe sample dataset (e.g., 100,000–200,000 rows, taking ~50–100 MB of RAM) for exploratory analysis and model training.
6. **Explicit Garbage Collection:** Delete intermediate variables (`del df`) and call `gc.collect()` after resource-heavy operations.
7. **Clean Infinite and NaN Values Early:** Columns such as `Flow Bytes/s` and `Flow Packets/s` have known divisions by zero (infinity values) and `NaN` in raw PCAP conversions that must be handled chunk-wise.
