import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def train_and_save_models(
    train_path="./dataset/processed_train.csv",
    test_path="./dataset/processed_test.csv",
    models_dir="./models",
    results_dir="./results"
):
    print("=" * 70)
    print("STEP 6: MACHINE LEARNING MODEL TRAINING")
    print("=" * 70)
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load Datasets
    print(f"\n[1/5] Loading datasets...")
    print(f"  Training file: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"  Testing file:  {test_path}")
    test_df = pd.read_csv(test_path)
    
    target_col = "Congestion_Level"
    predictor_cols = [c for c in train_df.columns if c != target_col]
    
    print(f"  Training records:  {len(train_df):,}")
    print(f"  Testing records:   {len(test_df):,}")
    print(f"  Predictor columns: {len(predictor_cols)}")
    print(f"  Target classes:    {sorted(train_df[target_col].unique().tolist())}")
    
    # Save exact feature column ordering
    feature_cols_path = os.path.join(models_dir, "feature_columns.json")
    with open(feature_cols_path, "w") as fp:
        json.dump(predictor_cols, fp, indent=2)
    print(f"  Saved predictor column order to: {feature_cols_path}")
    
    # Extract feature matrices and target vectors
    X_train = train_df[predictor_cols]
    y_train = train_df[target_col]
    X_test = test_df[predictor_cols]
    y_test = test_df[target_col]
    
    training_summary = {
        "dataset_info": {
            "train_records": len(train_df),
            "test_records": len(test_df),
            "predictor_count": len(predictor_cols),
            "target_classes": sorted(train_df[target_col].unique().tolist())
        },
        "models": {}
    }
    
    # =========================================================================
    # MODEL 1: LOGISTIC REGRESSION (WITH STANDARD SCALER)
    # =========================================================================
    print("\n[2/5] Training Model 1: Multinomial Logistic Regression...")
    # Fit scaler ONLY on training data
    print("  Fitting StandardScaler on X_train...")
    scaler = StandardScaler()
    scaler_fit_start = time.perf_counter()
    X_train_scaled = scaler.fit_transform(X_train)
    scaler_fit_time = time.perf_counter() - scaler_fit_start
    
    # Transform test data using the fitted scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Save fitted scaler
    scaler_path = os.path.join(models_dir, "feature_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  Saved fitted scaler to: {scaler_path} (fit time: {scaler_fit_time:.4f}s)")
    
    lr_config = {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42
    }
    lr_model = LogisticRegression(**lr_config)
    
    print(f"  Training LogisticRegression (max_iter={lr_config['max_iter']}, solver='{lr_config['solver']}')...")
    lr_train_start = time.perf_counter()
    lr_model.fit(X_train_scaled, y_train)
    lr_train_time = time.perf_counter() - lr_train_start
    
    # Measure prediction time on test set
    lr_pred_start = time.perf_counter()
    _ = lr_model.predict(X_test_scaled)
    lr_pred_time = time.perf_counter() - lr_pred_start
    
    lr_converged = bool(lr_model.n_iter_[0] < lr_config["max_iter"])
    print(f"  -> Training Time:   {lr_train_time:.2f}s")
    print(f"  -> Prediction Time: {lr_pred_time:.4f}s")
    print(f"  -> Iterations:      {lr_model.n_iter_[0]}/{lr_config['max_iter']} (Converged: {lr_converged})")
    
    lr_model_path = os.path.join(models_dir, "logistic_regression.joblib")
    joblib.dump(lr_model, lr_model_path)
    lr_file_size_mb = os.path.getsize(lr_model_path) / (1024 * 1024)
    print(f"  -> Model saved to:  {lr_model_path} ({lr_file_size_mb:.3f} MB)")
    
    training_summary["models"]["logistic_regression"] = {
        "model_file": lr_model_path,
        "file_size_mb": round(lr_file_size_mb, 4),
        "config": lr_config,
        "train_time_seconds": round(lr_train_time, 4),
        "prediction_time_seconds": round(lr_pred_time, 4),
        "iterations": int(lr_model.n_iter_[0]),
        "converged": lr_converged,
        "scaler_file": scaler_path
    }
    
    # =========================================================================
    # MODEL 2: DECISION TREE CLASSIFIER
    # =========================================================================
    print("\n[3/5] Training Model 2: Decision Tree Classifier...")
    dt_config = {
        "criterion": "gini",
        "max_depth": 16,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "random_state": 42
    }
    dt_model = DecisionTreeClassifier(**dt_config)
    
    print(f"  Training DecisionTreeClassifier (max_depth={dt_config['max_depth']}, min_samples_leaf={dt_config['min_samples_leaf']})...")
    dt_train_start = time.perf_counter()
    dt_model.fit(X_train, y_train)
    dt_train_time = time.perf_counter() - dt_train_start
    
    dt_pred_start = time.perf_counter()
    _ = dt_model.predict(X_test)
    dt_pred_time = time.perf_counter() - dt_pred_start
    
    print(f"  -> Training Time:   {dt_train_time:.2f}s")
    print(f"  -> Prediction Time: {dt_pred_time:.4f}s")
    print(f"  -> Tree Depth:      {dt_model.get_depth()}")
    print(f"  -> Number of Leaves:{dt_model.get_n_leaves():,}")
    
    dt_model_path = os.path.join(models_dir, "decision_tree.joblib")
    joblib.dump(dt_model, dt_model_path)
    dt_file_size_mb = os.path.getsize(dt_model_path) / (1024 * 1024)
    print(f"  -> Model saved to:  {dt_model_path} ({dt_file_size_mb:.3f} MB)")
    
    training_summary["models"]["decision_tree"] = {
        "model_file": dt_model_path,
        "file_size_mb": round(dt_file_size_mb, 4),
        "config": dt_config,
        "train_time_seconds": round(dt_train_time, 4),
        "prediction_time_seconds": round(dt_pred_time, 4),
        "actual_depth": int(dt_model.get_depth()),
        "number_of_leaves": int(dt_model.get_n_leaves())
    }
    
    # =========================================================================
    # MODEL 3: RANDOM FOREST CLASSIFIER (MEMORY-CONSCIOUS FOR 8 GB RAM)
    # =========================================================================
    print("\n[4/5] Training Model 3: Random Forest Classifier (Memory-Conscious)...")
    rf_config = {
        "n_estimators": 100,
        "max_depth": 16,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "n_jobs": -1,
        "random_state": 42
    }
    rf_model = RandomForestClassifier(**rf_config)
    
    print(f"  Training RandomForestClassifier (trees={rf_config['n_estimators']}, max_depth={rf_config['max_depth']}, n_jobs={rf_config['n_jobs']})...")
    rf_train_start = time.perf_counter()
    rf_model.fit(X_train, y_train)
    rf_train_time = time.perf_counter() - rf_train_start
    
    rf_pred_start = time.perf_counter()
    _ = rf_model.predict(X_test)
    rf_pred_time = time.perf_counter() - rf_pred_start
    
    print(f"  -> Training Time:   {rf_train_time:.2f}s")
    print(f"  -> Prediction Time: {rf_pred_time:.4f}s")
    print(f"  -> Total Estimators:{len(rf_model.estimators_)}")
    
    rf_model_path = os.path.join(models_dir, "random_forest.joblib")
    joblib.dump(rf_model, rf_model_path)
    rf_file_size_mb = os.path.getsize(rf_model_path) / (1024 * 1024)
    print(f"  -> Model saved to:  {rf_model_path} ({rf_file_size_mb:.3f} MB)")
    
    training_summary["models"]["random_forest"] = {
        "model_file": rf_model_path,
        "file_size_mb": round(rf_file_size_mb, 4),
        "config": rf_config,
        "train_time_seconds": round(rf_train_time, 4),
        "prediction_time_seconds": round(rf_pred_time, 4),
        "trees_trained": len(rf_model.estimators_)
    }
    
    # =========================================================================
    # SAVE SUMMARY METADATA
    # =========================================================================
    summary_path = os.path.join(results_dir, "training_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(training_summary, fp, indent=2)
    print(f"\n[5/5] Saved complete training summary to: {summary_path}")
    
    print("\n" + "=" * 70)
    print("STEP 6 MODEL TRAINING COMPLETE!")
    print("=" * 70)
    return training_summary

if __name__ == "__main__":
    train_and_save_models()
