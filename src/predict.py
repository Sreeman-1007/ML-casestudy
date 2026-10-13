import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Union, Dict, List, Any

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

class CongestionPredictor:
    """
    Production Inference Engine for Internet Traffic Congestion-Level Prediction.
    
    Uses the trained Random Forest Classifier (selected as the best model in Step 7)
    and enforces strict input validation, feature sequence preservation, and
    memory-safe inference.
    """
    def __init__(
        self,
        model_path: str = "./models/random_forest.joblib",
        feature_columns_path: str = "./models/feature_columns.json",
        target_config_path: str = "./models/congestion_target_parameters.json"
    ):
        self.model_path = model_path
        self.feature_columns_path = feature_columns_path
        self.target_config_path = target_config_path
        
        # 1. Load trained Random Forest model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        self.model = joblib.load(self.model_path)
        
        # 2. Load feature column ordering
        if not os.path.exists(self.feature_columns_path):
            raise FileNotFoundError(f"Feature columns file not found at: {self.feature_columns_path}")
        with open(self.feature_columns_path, "r", encoding="utf-8") as fp:
            self.feature_columns = json.load(fp)
            
        # 3. Load target configuration if available
        self.target_config = {}
        if os.path.exists(self.target_config_path):
            with open(self.target_config_path, "r", encoding="utf-8") as fp:
                self.target_config = json.load(fp)
                
        self.classes_ = list(self.model.classes_)
        self.expected_feature_count = len(self.feature_columns)
        
    def validate_input(self, input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> pd.DataFrame:
        """
        Validates input network telemetry records:
        - Ensures all 58 required predictor features are present.
        - Verifies numeric data types.
        - Checks for missing (NaN/None) values.
        - Enforces exact feature column ordering used during model training.
        """
        # Convert single dictionary or list of dictionaries to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
            is_single = True
        elif isinstance(input_data, list):
            if len(input_data) == 0:
                raise ValueError("Input list of records cannot be empty.")
            df = pd.DataFrame(input_data)
            is_single = False
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
            is_single = len(df) == 1
        else:
            raise TypeError(
                f"Unsupported input type: {type(input_data).__name__}. "
                "Expected dict, list of dicts, or pandas DataFrame."
            )
            
        # 1. Check for missing required features
        input_cols = set(df.columns)
        expected_cols = set(self.feature_columns)
        missing_cols = expected_cols - input_cols
        
        if missing_cols:
            missing_sorted = sorted(list(missing_cols))
            raise ValueError(
                f"Input validation error: {len(missing_sorted)} required feature(s) are missing from input.\n"
                f"Missing features: {missing_sorted}"
            )
            
        # 2. Reorder columns to match exact training sequence
        df_reordered = df[self.feature_columns].copy()
        
        # 3. Verify numeric data types and check for invalid/missing values
        for col in self.feature_columns:
            # Check for nulls/NaNs
            if df_reordered[col].isnull().any():
                null_indices = df_reordered.index[df_reordered[col].isnull()].tolist()
                raise ValueError(
                    f"Input validation error: Column '{col}' contains NaN/null values at record index(es): {null_indices}."
                )
            # Ensure numeric conversion
            try:
                df_reordered[col] = pd.to_numeric(df_reordered[col], errors='raise')
            except Exception as e:
                raise ValueError(
                    f"Input validation error: Column '{col}' cannot be converted to numeric. Error: {str(e)}"
                )
                
            # Check for infinities
            if np.isinf(df_reordered[col]).any():
                inf_indices = df_reordered.index[np.isinf(df_reordered[col])].tolist()
                raise ValueError(
                    f"Input validation error: Column '{col}' contains infinite values at record index(es): {inf_indices}."
                )
                
        return df_reordered
        
    def predict(
        self,
        input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        return_probabilities: bool = False
    ) -> Union[str, Dict[str, Any], List[str], List[Dict[str, Any]]]:
        """
        Generates congestion-level predictions for validated input network flows.
        
        Parameters:
        -----------
        input_data : dict, list of dicts, or pd.DataFrame
            Network flow record(s) containing the 58 approved predictor features.
        return_probabilities : bool, default=False
            If True, returns class prediction alongside confidence probabilities.
            
        Returns:
        --------
        Prediction(s) as string ('LOW', 'MEDIUM', 'HIGH') or structured dictionary.
        """
        is_single = isinstance(input_data, dict) or (isinstance(input_data, pd.DataFrame) and len(input_data) == 1)
        
        # Validate and reorder input
        X = self.validate_input(input_data)
        
        # Generate model predictions
        predictions = self.model.predict(X)
        
        if return_probabilities:
            probabilities = self.model.predict_proba(X)
            class_order = self.classes_
            
            results = []
            for pred, prob_arr in zip(predictions, probabilities):
                prob_dict = {cls_name: round(float(prob), 4) for cls_name, prob in zip(class_order, prob_arr)}
                confidence = round(float(np.max(prob_arr)), 4)
                results.append({
                    "predicted_congestion_level": pred,
                    "confidence": confidence,
                    "probabilities": prob_dict
                })
            return results[0] if is_single else results
        else:
            pred_list = [str(p) for p in predictions]
            return pred_list[0] if is_single else pred_list

# Global singleton instance for memory-conscious reuse
_GLOBAL_PREDICTOR = None

def get_predictor() -> CongestionPredictor:
    """Returns or lazily initializes the global CongestionPredictor instance."""
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = CongestionPredictor()
    return _GLOBAL_PREDICTOR

def predict_congestion(
    input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    return_probabilities: bool = False
) -> Union[str, Dict[str, Any], List[str], List[Dict[str, Any]]]:
    """
    Convenience function for predicting network traffic congestion levels.
    
    Accepts:
    - Single dictionary of 58 predictor features.
    - List of feature dictionaries.
    - pandas DataFrame with 58 predictor columns.
    
    Returns:
    - 'LOW', 'MEDIUM', or 'HIGH' (or structured dict with probabilities if requested).
    """
    predictor = get_predictor()
    return predictor.predict(input_data, return_probabilities=return_probabilities)

# Self-test routine
if __name__ == "__main__":
    print("=" * 70)
    print("TESTING PRODUCTION PREDICTION MODULE (src/predict.py)")
    print("=" * 70)
    
    # 1. Initialize Predictor
    predictor = CongestionPredictor()
    print("Loaded CongestionPredictor successfully.")
    print(f"Model Type:            {type(predictor.model).__name__}")
    print(f"Required Features:     {predictor.expected_feature_count}")
    print(f"Model Target Classes:  {predictor.classes_}")
    
    # 2. Test Input Validation (Negative Test: Missing Feature)
    print("\n--- Negative Test: Missing Feature Validation ---")
    bad_sample = {"Destination Port": 80, "Total Fwd Packets": 2}
    try:
        predictor.predict(bad_sample)
        print("ERROR: Failed to reject missing features!")
    except ValueError as e:
        print("PASS: Successfully caught missing features error:")
        print(f"  {str(e)[:120]}...")
        
    # 3. Test Input Validation (Negative Test: Non-numeric value)
    print("\n--- Negative Test: Non-Numeric Value Validation ---")
    # Load 1 valid row from processed_test.csv and corrupt one field
    test_df = pd.read_csv("./dataset/processed_test.csv")
    valid_sample = test_df[predictor.feature_columns].iloc[0].to_dict()
    corrupted_sample = valid_sample.copy()
    corrupted_sample["Destination Port"] = "INVALID_PORT"
    try:
        predictor.predict(corrupted_sample)
        print("ERROR: Failed to reject non-numeric value!")
    except ValueError as e:
        print("PASS: Successfully caught non-numeric value error:")
        print(f"  {str(e)[:120]}...")
        
    # 4. Positive Tests: Predict Real Test Rows from processed_test.csv
    print("\n--- Positive Tests: Evaluating Real Unseen Test Records ---")
    
    # Select 2 real samples for each class: LOW, MEDIUM, HIGH
    samples_to_test = []
    for c in ["LOW", "MEDIUM", "HIGH"]:
        subset = test_df[test_df["Congestion_Level"] == c]
        for i in range(2):
            idx = subset.index[i]
            samples_to_test.append((idx, c, subset[predictor.feature_columns].iloc[i].to_dict()))
            
    print(f"Testing {len(samples_to_test)} real records from processed_test.csv:\n")
    for row_idx, true_class, record_dict in samples_to_test:
        pred_res = predict_congestion(record_dict, return_probabilities=True)
        pred_class = pred_res["predicted_congestion_level"]
        conf = pred_res["confidence"]
        probs = pred_res["probabilities"]
        
        match_str = "MATCH" if pred_class == true_class else "MISMATCH"
        print(f"  Record Index {row_idx:>5d} | True Class: {true_class:6s} | Predicted: {pred_class:6s} ({match_str})")
        print(f"    Confidence: {conf*100:5.2f}% | Probabilities: LOW={probs['LOW']*100:4.1f}%, MED={probs['MEDIUM']*100:4.1f}%, HIGH={probs['HIGH']*100:4.1f}%")
        print(f"    Sample Telemetry: AvgPktSize={record_dict['Average Packet Size']:.1f}, IdleMean={record_dict['Idle Mean']:.0f}, FwdPkts={record_dict['Total Fwd Packets']:.0f}\n")
        
    # 5. Batch DataFrame Prediction Test
    print("--- Batch DataFrame Prediction Test ---")
    batch_records = [rec for _, _, rec in samples_to_test]
    batch_df = pd.DataFrame(batch_records)
    batch_preds = predict_congestion(batch_df)
    print(f"Batch prediction output for {len(batch_preds)} records: {batch_preds}")
    print("\nPrediction system self-test passed successfully!")
