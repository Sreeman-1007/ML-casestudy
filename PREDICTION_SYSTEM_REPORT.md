# Production Prediction System Documentation Report

## 1. Executive Summary
- **Project Title:** Internet Traffic Congestion Prediction Using Machine Learning
- **Stage:** STEP 9 — Standalone Inference Engine & Pipeline Integration
- **Module Path:** [`./src/predict.py`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/src/predict.py)
- **Selected Model:** **Random Forest Classifier** (Selected in Step 7 for highest Macro F1-score: 85.53%)
- **Model File Path:** [`./models/random_forest.joblib`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models/random_forest.joblib) (29.614 MB)
- **Feature Ordering Artifact:** [`./models/feature_columns.json`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models/feature_columns.json) (58 features)
- **Target Metadata Artifact:** [`./models/congestion_target_parameters.json`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/models/congestion_target_parameters.json)

---

## 2. Production Model Architecture

The prediction system is powered by the production **Random Forest ensemble**:
- **Estimator Count:** 100 decision trees
- **Maximum Tree Depth:** 16
- **Splitting Criterion:** Gini Impurity
- **Ensemble Bagging:** Bootstrap aggregated sub-sampling with `max_features='sqrt'`
- **Inference Latency:** ~4.54 microseconds per flow (~0.18s for 40,000 flows)
- **Hardware Footprint:** Model size is ~29.6 MB, allowing in-memory residency with < 50 MB RAM consumption.

---

## 3. Required Input Features & Sequence Ordering

The prediction module accepts flow summaries comprising **exactly 58 numerical predictor features**. The exact sequence strictly mirrors the training matrix:

| Index | Feature Name | Index | Feature Name | Index | Feature Name |
| :-: | :--- | :-: | :--- | :-: | :--- |
| **0** | `Destination Port` | **20** | `Init_Win_bytes_backward` | **40** | `URG Flag Count` |
| **1** | `Total Fwd Packets` | **21** | `act_data_pkt_fwd` | **41** | `CWE Flag Count` |
| **2** | `Total Backward Packets` | **22** | `min_seg_size_forward` | **42** | `ECE Flag Count` |
| **3** | `Total Length of Fwd Packets` | **23** | `Active Mean` | **43** | `Down/Up Ratio` |
| **4** | `Total Length of Bwd Packets` | **24** | `Active Std` | **44** | `Average Packet Size` |
| **5** | `Fwd Packet Length Max` | **25** | `Active Max` | **45** | `Avg Fwd Segment Size` |
| **6** | `Fwd Packet Length Min` | **26** | `Active Min` | **46** | `Avg Bwd Segment Size` |
| **7** | `Fwd Packet Length Mean` | **27** | `Idle Mean` | **47** | `Fwd Avg Bytes/Bulk` |
| **8** | `Fwd Packet Length Std` | **28** | `Idle Std` | **48** | `Fwd Avg Packets/Bulk` |
| **9** | `Bwd Packet Length Max` | **29** | `Idle Max` | **49** | `Fwd Avg Bulk Rate` |
| **10** | `Bwd Packet Length Min` | **30** | `Idle Min` | **50** | `Bwd Avg Bytes/Bulk` |
| **11** | `Bwd Packet Length Mean` | **31** | `Fwd PSH Flags` | **51** | `Bwd Avg Packets/Bulk` |
| **12** | `Bwd Packet Length Std` | **32** | `Bwd PSH Flags` | **52** | `Bwd Avg Bulk Rate` |
| **13** | `Fwd Header Length` | **33** | `Fwd URG Flags` | **53** | `Subflow Fwd Packets` |
| **14** | `Bwd Header Length` | **34** | `Bwd URG Flags` | **54** | `Subflow Fwd Bytes` |
| **15** | `Min Packet Length` | **35** | `FIN Flag Count` | **55** | `Subflow Bwd Packets` |
| **16** | `Max Packet Length` | **36** | `SYN Flag Count` | **56** | `Subflow Bwd Bytes` |
| **17** | `Packet Length Mean` | **37** | `RST Flag Count` | **57** | `Init_Win_bytes_forward` |
| **18** | `Packet Length Std` | **38** | `PSH Flag Count` | — | — |
| **19** | `Packet Length Variance` | **39** | `ACK Flag Count` | — | — |

*Quarantine Verification:* All direct rate constituents (`Flow Bytes/s`, `Flow Packets/s`), algebraic duration/IAT equivalents, and intrusion labels are completely absent from the inference interface.

---

## 4. Preprocessing & Input Validation Pipeline

The module implements strict input validation through `CongestionPredictor.validate_input()`:
1. **Format Flexibility:** Supports a single `dict`, a list of `dicts`, or a `pandas.DataFrame`.
2. **Missing Feature Detection:** Confirms all 58 features are present. Raises a detailed `ValueError` specifying every missing feature name if any are omitted.
3. **Type & Finite Value Validation:** Coerces all inputs to numeric values. Rejects non-numeric strings, `None`, `NaN`, and `+/-Infinity` with clear error messages.
4. **Column Alignment:** Reorders dictionary keys to match `models/feature_columns.json` exactly, preventing silent feature permutation bugs.
5. **Preprocessing Applied:** For the Random Forest ensemble, raw numeric features are passed directly without scaling, preserving the decision thresholds formed during training.

---

## 5. Congestion Class Mapping

The prediction module maps inputs to the three-tier congestion ground truth established in Step 4:
- **`LOW`:** Minimal network traffic intensity; prolonged inter-burst idle pauses (median ~5.86s), high client window headroom, large payload-to-control ratios.
- **`MEDIUM`:** Moderate network load; intermediate burst activity, emerging queuing delays, transitional buffer occupancy.
- **`HIGH`:** Severe network traffic congestion; continuous zero-idle transmission, small-packet flood dynamics (median ~16 bytes), reduced TCP window capacity.

---

## 6. Prediction Function Interface

### Primary Reusable Function:
```python
from src.predict import predict_congestion

# 1. Discrete Prediction
result = predict_congestion(input_flow_dict)
# Returns: 'LOW', 'MEDIUM', or 'HIGH'

# 2. Probabilistic Prediction with Confidence Scoring
result_with_probs = predict_congestion(input_flow_dict, return_probabilities=True)
# Returns:
# {
#   "predicted_congestion_level": "HIGH",
#   "confidence": 0.9914,
#   "probabilities": {
#     "LOW": 0.002,
#     "MEDIUM": 0.0066,
#     "HIGH": 0.9914
#   }
# }
```

---

## 7. Real Test Dataset Validation & Verification Results

The prediction module was tested using **unseen real flow records** extracted directly from [`./dataset/processed_test.csv`](file:///c:/Users/Sreeman/Downloads/intenret%20traffic%20congestion%20prediction/dataset/processed_test.csv):

| Record Index | Ground-Truth Class | Predicted Class | Confidence | Class Probabilities | Sample Telemetry | Outcome |
| :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **#0** | `LOW` | **`LOW`** | **93.70%** | LOW: 93.7%, MED: 5.9%, HIGH: 0.4% | AvgPktSize=116.2, IdleMean=0, FwdPkts=2 | **MATCH** |
| **#4** | `LOW` | **`LOW`** | **99.79%** | LOW: 99.8%, MED: 0.2%, HIGH: 0.0% | AvgPktSize=3.0, IdleMean=0, FwdPkts=3 | **MATCH** |
| **#2** | `MEDIUM` | **`MEDIUM`** | **60.49%** | LOW: 39.5%, MED: 60.5%, HIGH: 0.0% | AvgPktSize=0.0, IdleMean=0, FwdPkts=2 | **MATCH** |
| **#3** | `MEDIUM` | **`HIGH`** | **54.45%** | LOW: 5.5%, MED: 40.0%, HIGH: 54.4% | AvgPktSize=50.5, IdleMean=0, FwdPkts=2 | *Boundary Mismatch* |
| **#1** | `HIGH` | **`HIGH`** | **63.06%** | LOW: 17.4%, MED: 19.6%, HIGH: 63.1% | AvgPktSize=9.0, IdleMean=0, FwdPkts=2 | **MATCH** |
| **#14** | `HIGH` | **`HIGH`** | **99.14%** | LOW: 0.2%, MED: 0.6%, HIGH: 99.1% | AvgPktSize=5.5, IdleMean=0, FwdPkts=2 | **MATCH** |

### Robustness & Validation Tests:
1. **Missing Feature Test:** A dictionary missing 56 required fields was submitted. **Result:** Correctly intercepted by `validate_input()` with a descriptive `ValueError` listing missing features.
2. **Invalid Data Type Test:** A record containing a string in `Destination Port` (`"INVALID_PORT"`) was submitted. **Result:** Correctly intercepted with a descriptive conversion error.
3. **Batch Prediction Test:** A `pd.DataFrame` containing multiple records was evaluated simultaneously. **Result:** Generated sequential batch predictions `['LOW', 'LOW', 'MEDIUM', 'HIGH', 'HIGH', 'HIGH']` in 2.1 milliseconds.

---

## 8. Limitations & Production Considerations

1. **Feature Availability:** In a live network monitoring setup (e.g., using `libpcap` or Wireshark), a flow aggregation tool (such as CICFlowMeter or Zeek) must compute bidirectional flow statistics before calling `predict_congestion()`.
2. **Episodic Boundary Ambiguity:** Transitional flows between `MEDIUM` and `HIGH` near the 66.7th percentile threshold exhibit split probabilities (~40% vs. ~54%), which reflects the continuous nature of network traffic intensity.
3. **Model Re-Training Separation:** The inference module operates strictly in read-only inference mode; model weights cannot be modified through the prediction interface.
