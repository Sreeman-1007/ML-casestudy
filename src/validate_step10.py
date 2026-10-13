import os
import sys
import json
import io
import pandas as pd
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath("."))
from src.predict import CongestionPredictor

def run_tests():
    print("=" * 60)
    print("STEP 10 AUTOMATED VERIFICATION SUITE")
    print("=" * 60)
    
    # 1. Verify Imports and Model Loading
    print("\n[Test 1] Testing Model & Predictor Initialization...")
    predictor = CongestionPredictor(
        model_path="./models/random_forest.joblib",
        feature_columns_path="./models/feature_columns.json",
        target_config_path="./models/congestion_target_parameters.json"
    )
    assert predictor.model is not None
    assert len(predictor.feature_columns) == 58
    print("  -> Predictor successfully initialized with 58 feature columns.")
    
    # 2. Test Single-Record Prediction from Demonstration Presets
    print("\n[Test 2] Testing Single-Record Demonstration Presets...")
    with open("./models/demonstration_presets.json", "r", encoding="utf-8") as f:
        presets = json.load(f)
    
    for name, sample in presets.items():
        res = predictor.predict(sample, return_probabilities=True)
        print(f"  -> Preset '{name}': Predicted={res['predicted_congestion_level']} (Confidence={res['confidence']:.2%})")
        assert res["predicted_congestion_level"] in ["LOW", "MEDIUM", "HIGH"]
        assert 0.0 <= res["confidence"] <= 1.0
        assert len(res["probabilities"]) == 3

    # 3. Test Batch Prediction on CSV Slice
    print("\n[Test 3] Testing Batch CSV Prediction & Annotated CSV Generation...")
    # Load 50 rows from processed_test.csv without the target column
    test_df = pd.read_csv("./dataset/processed_test.csv", nrows=50)
    if "Congestion_Level" in test_df.columns:
        test_df = test_df.drop(columns=["Congestion_Level"])
    
    batch_res = predictor.predict(test_df, return_probabilities=True)
    assert len(batch_res) == 50
    preds = [r["predicted_congestion_level"] for r in batch_res]
    confs = [r["confidence"] for r in batch_res]
    
    # Emulate app CSV export
    df_out = test_df.copy()
    df_out.insert(0, "Predicted_Congestion_Level", preds)
    df_out.insert(1, "Prediction_Confidence", confs)
    
    buf = io.StringIO()
    df_out.to_csv(buf, index=False)
    csv_str = buf.getvalue()
    assert len(csv_str) > 0
    print(f"  -> Successfully generated annotated batch CSV ({len(df_out)} rows, {len(csv_str)} bytes).")
    
    # 4. Test Missing Column Validation
    print("\n[Test 4] Testing Missing Column Error Handling...")
    invalid_df = test_df.drop(columns=["Destination Port"])
    try:
        predictor.predict(invalid_df)
        raise AssertionError("Failed to catch missing column!")
    except ValueError as e:
        print(f"  -> Caught expected missing column error: {e}")
        
    # 5. Test Invalid/Non-numeric Value Handling
    print("\n[Test 5] Testing Non-numeric / Invalid Value Handling...")
    corrupt_df = test_df.copy()
    corrupt_df["Total Fwd Packets"] = corrupt_df["Total Fwd Packets"].astype(object)
    corrupt_df.loc[0, "Total Fwd Packets"] = "INVALID_STRING"
    try:
        predictor.predict(corrupt_df)
        raise AssertionError("Failed to catch non-numeric value!")
    except ValueError as e:
        print(f"  -> Caught expected invalid numeric value error: {e}")
        
    # 6. Verify Results Files Needed by App
    print("\n[Test 6] Verifying Results Artifacts Existence & Schema...")
    assert os.path.exists("./results/model_comparison.csv"), "Missing model_comparison.csv"
    assert os.path.exists("./results/evaluation_metrics.json"), "Missing evaluation_metrics.json"
    assert os.path.exists("./results/feature_importance.csv"), "Missing feature_importance.csv"
    
    fi_df = pd.read_csv("./results/feature_importance.csv")
    assert "Feature" in fi_df.columns and "Importance" in fi_df.columns and "Rank" in fi_df.columns
    assert len(fi_df) == 58
    print("  -> Results files verified successfully.")
    
    # 7. Check App Import Syntax
    print("\n[Test 7] Verifying app.py syntax and import...")
    import app
    print("  -> app.py syntax and imports verified.")
    
    print("\n" + "=" * 60)
    print("ALL 7 STEP 10 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
