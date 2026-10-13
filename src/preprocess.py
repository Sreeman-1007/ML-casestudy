import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def preprocess_and_label_dataset(
    input_path="./dataset/working_dataset.csv",
    output_train_path="./dataset/processed_train.csv",
    output_test_path="./dataset/processed_test.csv",
    config_output_path="./results/preprocessing_config.json",
    plot_output_path="./results/congestion_distribution.png",
    random_state=42,
    test_size=0.20
):
    print("=" * 70)
    print("STEP 4: PREPROCESSING & CONGESTION LABEL CREATION")
    print("=" * 70)
    
    # 1. Load working dataset
    print(f"Loading working dataset: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    initial_rows, initial_cols = df.shape
    print(f"Loaded: {initial_rows:,} records, {initial_cols} columns")
    
    # 2. Train/Test Split BEFORE fitting any target parameters
    print(f"\nPerforming {int((1-test_size)*100)}/{int(test_size*100)} Train/Test split (random_state={random_state})...")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_count = len(train_df)
    test_count = len(test_df)
    print(f"  Training records: {train_count:,} ({train_count/initial_rows*100:.1f}%)")
    print(f"  Testing records:  {test_count:,} ({test_count/initial_rows*100:.1f}%)")
    
    # 3. Fit target parameters on Training portion ONLY
    print("\nFitting Traffic Intensity Score (TIS) parameters on Training set...")
    b_train_raw = train_df['Flow Bytes/s']
    p_train_raw = train_df['Flow Packets/s']
    
    # Audit invalid values on train
    b_train_nulls = int(b_train_raw.isnull().sum())
    b_train_infs = int(np.isinf(b_train_raw).sum())
    p_train_nulls = int(p_train_raw.isnull().sum())
    p_train_infs = int(np.isinf(p_train_raw).sum())
    b_train_neg = int((b_train_raw < 0).sum())
    p_train_neg = int((p_train_raw < 0).sum())
    
    print(f"  Train Flow Bytes/s:   {b_train_nulls} nulls, {b_train_infs} infs, {b_train_neg} negative values")
    print(f"  Train Flow Packets/s: {p_train_nulls} nulls, {p_train_infs} infs, {p_train_neg} negative values")
    
    # Find maximum finite values on training set
    b_max_finite = float(b_train_raw.replace([np.inf, -np.inf], np.nan).dropna().max())
    p_max_finite = float(p_train_raw.replace([np.inf, -np.inf], np.nan).dropna().max())
    print(f"  Train max finite Flow Bytes/s:   {b_max_finite:,.2f}")
    print(f"  Train max finite Flow Packets/s: {p_max_finite:,.2f}")
    
    # Clean train values
    b_train_clean = b_train_raw.replace([np.inf, -np.inf], b_max_finite).fillna(0).clip(lower=0)
    p_train_clean = p_train_raw.replace([np.inf, -np.inf], p_max_finite).fillna(0).clip(lower=0)
    
    # Non-linear log transformation
    log_b_train = np.log10(1.0 + b_train_clean)
    log_p_train = np.log10(1.0 + p_train_clean)
    
    # Range parameters on train
    b_min_val = float(log_b_train.min())
    b_max_val = float(log_b_train.max())
    p_min_val = float(log_p_train.min())
    p_max_val = float(log_p_train.max())
    
    print(f"  Fitted log10(1 + Bytes/s) range:   [{b_min_val:.4f}, {b_max_val:.4f}]")
    print(f"  Fitted log10(1 + Packets/s) range: [{p_min_val:.4f}, {p_max_val:.4f}]")
    
    # Compute normalized score on train
    sb_train = (log_b_train - b_min_val) / (b_max_val - b_min_val + 1e-9)
    sp_train = (log_p_train - p_min_val) / (p_max_val - p_min_val + 1e-9)
    tis_train = 0.5 * sb_train + 0.5 * sp_train
    
    # Compute empirical tercile thresholds on training TIS
    tau_33 = float(np.percentile(tis_train, 33.333333333333336))
    tau_66 = float(np.percentile(tis_train, 66.66666666666667))
    print(f"  Fitted Lower Tercile Threshold (tau_33.3): {tau_33:.6f}")
    print(f"  Fitted Upper Tercile Threshold (tau_66.7): {tau_66:.6f}")
    
    # Assign training Congestion_Level
    train_labels = pd.cut(
        tis_train,
        bins=[-np.inf, tau_33, tau_66, np.inf],
        labels=['LOW', 'MEDIUM', 'HIGH']
    )
    train_df['Congestion_Level'] = train_labels
    
    # 4. Apply training-fitted parameters to Test set
    print("\nApplying training parameters to Test set...")
    b_test_raw = test_df['Flow Bytes/s']
    p_test_raw = test_df['Flow Packets/s']
    
    b_test_clean = b_test_raw.replace([np.inf, -np.inf], b_max_finite).fillna(0).clip(lower=0)
    p_test_clean = p_test_raw.replace([np.inf, -np.inf], p_max_finite).fillna(0).clip(lower=0)
    
    log_b_test = np.log10(1.0 + b_test_clean)
    log_p_test = np.log10(1.0 + p_test_clean)
    
    sb_test = ((log_b_test - b_min_val) / (b_max_val - b_min_val + 1e-9)).clip(0.0, 1.0)
    sp_test = ((log_p_test - p_min_val) / (p_max_val - p_min_val + 1e-9)).clip(0.0, 1.0)
    tis_test = 0.5 * sb_test + 0.5 * sp_test
    
    test_labels = pd.cut(
        tis_test,
        bins=[-np.inf, tau_33, tau_66, np.inf],
        labels=['LOW', 'MEDIUM', 'HIGH']
    )
    test_df['Congestion_Level'] = test_labels
    
    # Print class counts
    train_class_counts = train_df['Congestion_Level'].value_counts()[['LOW', 'MEDIUM', 'HIGH']].to_dict()
    test_class_counts = test_df['Congestion_Level'].value_counts()[['LOW', 'MEDIUM', 'HIGH']].to_dict()
    
    print("\nTraining Class Distribution:")
    for k, v in train_class_counts.items():
        print(f"  {k:6s}: {v:>6,} ({v/train_count*100:6.2f}%)")
        
    print("\nTesting Class Distribution (using training thresholds):")
    for k, v in test_class_counts.items():
        print(f"  {k:6s}: {v:>6,} ({v/test_count*100:6.2f}%)")
        
    # 5. Quarantine & Target Leakage Prevention
    quarantined_features = [
        'Flow Bytes/s',
        'Flow Packets/s',
        'Fwd Packets/s',
        'Bwd Packets/s',
        'Flow Duration',
        'Flow IAT Mean',
        'Flow IAT Std',
        'Flow IAT Max',
        'Flow IAT Min',
        'Fwd IAT Total',
        'Fwd IAT Mean',
        'Fwd IAT Std',
        'Fwd IAT Max',
        'Fwd IAT Min',
        'Bwd IAT Total',
        'Bwd IAT Mean',
        'Bwd IAT Std',
        'Bwd IAT Max',
        'Bwd IAT Min',
        'Label'
    ]
    
    print(f"\nApplying Feature Quarantine (Removing {len(quarantined_features)} target-defining & algebraic leakage features)...")
    train_processed = train_df.drop(columns=quarantined_features, errors='ignore')
    test_processed = test_df.drop(columns=quarantined_features, errors='ignore')
    
    predictor_features = [c for c in train_processed.columns if c != 'Congestion_Level']
    print(f"  Predictor Features Retained: {len(predictor_features)}")
    print(f"  Target Feature: Congestion_Level")
    print(f"  Total Processed Columns: {train_processed.shape[1]}")
    
    # 6. Save Processed Datasets
    print(f"\nSaving processed datasets...")
    train_processed.to_csv(output_train_path, index=False)
    print(f"  Saved: {output_train_path} ({os.path.getsize(output_train_path)/(1024*1024):.2f} MB)")
    
    test_processed.to_csv(output_test_path, index=False)
    print(f"  Saved: {output_test_path} ({os.path.getsize(output_test_path)/(1024*1024):.2f} MB)")
    
    # 7. Save Preprocessing Configuration
    config = {
        "random_state": random_state,
        "split_ratio": {"train": 1.0 - test_size, "test": test_size},
        "records": {
            "total_initial": initial_rows,
            "train": train_count,
            "test": test_count
        },
        "target_column": "Congestion_Level",
        "target_formula": "TIS = 0.5 * norm(log10(1 + Flow Bytes/s)) + 0.5 * norm(log10(1 + Flow Packets/s))",
        "fitted_target_parameters": {
            "b_max_finite": b_max_finite,
            "p_max_finite": p_max_finite,
            "log_bytes_min": b_min_val,
            "log_bytes_max": b_max_val,
            "log_pkts_min": p_min_val,
            "log_pkts_max": p_max_val,
            "tau_33_3": tau_33,
            "tau_66_7": tau_66
        },
        "class_distributions": {
            "train": train_class_counts,
            "train_percentages": {k: round(v/train_count*100, 2) for k, v in train_class_counts.items()},
            "test": test_class_counts,
            "test_percentages": {k: round(v/test_count*100, 2) for k, v in test_class_counts.items()}
        },
        "features": {
            "total_processed_columns": train_processed.shape[1],
            "predictor_count": len(predictor_features),
            "predictor_features": predictor_features,
            "quarantined_count": len(quarantined_features),
            "quarantined_features": quarantined_features
        }
    }
    
    os.makedirs(os.path.dirname(config_output_path), exist_ok=True)
    with open(config_output_path, "w") as fp:
        json.dump(config, fp, indent=2)
    print(f"  Saved config: {config_output_path}")
    
    # Also save target parameters copy in models/
    models_config_path = "./models/congestion_target_parameters.json"
    with open(models_config_path, "w") as fp:
        json.dump(config["fitted_target_parameters"], fp, indent=2)
    print(f"  Saved target params: {models_config_path}")
    
    # 8. Create Distribution Visualization
    print("\nGenerating class distribution graph...")
    plt.figure(figsize=(9, 5.5), dpi=150)
    sns.set_theme(style="whitegrid")
    
    classes = ['LOW', 'MEDIUM', 'HIGH']
    train_vals = [train_class_counts[c] for c in classes]
    test_vals = [test_class_counts[c] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    rects1 = ax.bar(x - width/2, train_vals, width, label='Train Set (80%)', color='#2563eb', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, test_vals, width, label='Test Set (20%)', color='#10b981', edgecolor='black', linewidth=0.8)
    
    ax.set_ylabel('Number of Flow Records', fontsize=12, fontweight='bold')
    ax.set_title('Congestion Level Class Distribution (Train vs. Test)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    
    # Add count and percentage labels above bars
    def autolabel(rects, total):
        for rect in rects:
            height = rect.get_height()
            pct = (height / total) * 100
            ax.annotate(f'{height:,}\n({pct:.1f}%)',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='semibold')
                        
    autolabel(rects1, train_count)
    autolabel(rects2, test_count)
    
    ax.set_ylim(0, max(train_vals) * 1.22)
    plt.tight_layout()
    plt.savefig(plot_output_path)
    plt.close()
    print(f"  Saved plot: {plot_output_path}")
    
    print("\n" + "=" * 70)
    print("STEP 4 COMPLETE!")
    print("=" * 70)
    return config

if __name__ == "__main__":
    preprocess_and_label_dataset()
