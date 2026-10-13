import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
sns.set_theme(style="whitegrid")

def analyze_feature_importance(
    model_path="./models/random_forest.joblib",
    features_path="./models/feature_columns.json",
    csv_output_path="./results/feature_importance.csv",
    plot_output_path="./results/graphs/random_forest_feature_importance.png"
):
    print("=" * 70)
    print("STEP 8: RANDOM FOREST FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    # 1. Load Pretrained Random Forest Model
    print(f"\n[1/4] Loading pretrained Random Forest model: {model_path}")
    rf_model = joblib.load(model_path)
    print(f"  Loaded model successfully: {type(rf_model).__name__}")
    print(f"  Estimators in forest:      {len(rf_model.estimators_)}")
    
    # 2. Load Feature Column Names
    print(f"\n[2/4] Loading feature column names: {features_path}")
    with open(features_path, "r") as fp:
        feature_names = json.load(fp)
    print(f"  Predictor features count:  {len(feature_names)}")
    
    importances = rf_model.feature_importances_
    print(f"  Importance values count:   {len(importances)}")
    
    # Strict verification
    if len(feature_names) != len(importances):
        raise ValueError(
            f"Dimension mismatch: {len(feature_names)} feature names vs {len(importances)} importance values"
        )
        
    # 3. Check Against Quarantined Leakage Features
    quarantine_forbidden = [
        "flow bytes/s", "flow packets/s", "fwd packets/s", "bwd packets/s",
        "flow duration", "flow iat", "fwd iat", "bwd iat", "label"
    ]
    quarantine_violations = [
        f for f in feature_names if any(q in f.lower() for q in quarantine_forbidden)
    ]
    if quarantine_violations:
        print(f"CRITICAL ERROR: Quarantined features detected in predictor set: {quarantine_violations}")
        raise RuntimeError(f"Leakage violation detected: {quarantine_violations}")
    print("  Leakage verification passed: ZERO quarantined features detected.")
    
    # 4. Create Feature Importance Table
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    df_imp["Rank"] = df_imp.index + 1
    
    # Reorder columns: Feature, Importance, Rank
    df_imp = df_imp[["Feature", "Importance", "Rank"]]
    
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    df_imp.to_csv(csv_output_path, index=False)
    print(f"\n[3/4] Saved complete feature importance table to: {csv_output_path}")
    print(f"  Total ranked features: {len(df_imp)}")
    
    # Print Top 10
    top10 = df_imp.head(10).copy()
    print("\n  Top 10 Most Important Features:")
    for _, row in top10.iterrows():
        print(f"    Rank {int(row['Rank']):2d}: {row['Feature']:25s} -> {row['Importance']:.6f} ({row['Importance']*100:.2f}%)")
        
    # 5. Generate Horizontal Bar Chart for Top 10 Features
    print(f"\n[4/4] Generating Top 10 Feature Importance Visualization...")
    os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    
    # Reverse order so highest rank is on top
    plot_df = top10.sort_values(by="Importance", ascending=True)
    
    bars = ax.barh(
        plot_df["Feature"],
        plot_df["Importance"],
        color="#2563eb",
        edgecolor="black",
        linewidth=0.8,
        height=0.65
    )
    
    # Annotate bar values
    max_val = plot_df["Importance"].max()
    for bar in bars:
        w = bar.get_width()
        pct = w * 100
        ax.annotate(
            f" {w:.4f} ({pct:.2f}%)",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold"
        )
        
    ax.set_xlim(0, max_val * 1.25)
    ax.set_xlabel("Gini Feature Importance Score", fontsize=11, fontweight="bold")
    ax.set_ylabel("Network Traffic Feature", fontsize=11, fontweight="bold")
    ax.set_title(
        "Top 10 Most Important Network Features in Random Forest\n(Internet Traffic Congestion-Level Prediction)",
        fontsize=13,
        fontweight="bold",
        pad=15
    )
    plt.tight_layout()
    plt.savefig(plot_output_path)
    plt.close()
    print(f"  Saved plot to: {plot_output_path}")
    
    print("\n" + "=" * 70)
    print("STEP 8 FEATURE IMPORTANCE ANALYSIS COMPLETE!")
    print("=" * 70)
    return df_imp

if __name__ == "__main__":
    analyze_feature_importance()
