import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
sns.set_theme(style="whitegrid")

def evaluate_models(
    test_path="./dataset/processed_test.csv",
    models_dir="./models",
    results_dir="./results",
    cm_dir="./results/confusion_matrices"
):
    print("=" * 70)
    print("STEP 7: MACHINE LEARNING MODEL EVALUATION ON UNSEEN TEST DATA")
    print("=" * 70)
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)
    
    # 1. Load Unseen Test Data
    print(f"\n[1/5] Loading unseen test dataset: {test_path}")
    test_df = pd.read_csv(test_path)
    n_test = len(test_df)
    target_col = "Congestion_Level"
    
    # Load exact predictor feature sequence
    feature_cols_path = os.path.join(models_dir, "feature_columns.json")
    with open(feature_cols_path, "r") as fp:
        feature_cols = json.load(fp)
        
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    class_order = ["LOW", "MEDIUM", "HIGH"]
    print(f"  Test Records:      {n_test:,}")
    print(f"  Predictor Count:   {len(feature_cols)}")
    print(f"  Target Classes:    {class_order}")
    print(f"  Class Distribution in Test Set:")
    for c in class_order:
        cnt = (y_test == c).sum()
        print(f"    {c:6s}: {cnt:>6,} ({cnt/n_test*100:5.2f}%)")
        
    # 2. Load Pretrained Models & Scaler
    print(f"\n[2/5] Loading pretrained models and scaler from: {models_dir}")
    scaler_path = os.path.join(models_dir, "feature_scaler.joblib")
    scaler = joblib.load(scaler_path)
    
    lr_model = joblib.load(os.path.join(models_dir, "logistic_regression.joblib"))
    dt_model = joblib.load(os.path.join(models_dir, "decision_tree.joblib"))
    rf_model = joblib.load(os.path.join(models_dir, "random_forest.joblib"))
    print("  Loaded: StandardScaler, LogisticRegression, DecisionTree, RandomForest")
    
    # Scale test features for Logistic Regression using the pre-fitted scaler
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": (lr_model, X_test_scaled, "logistic_regression"),
        "Decision Tree": (dt_model, X_test, "decision_tree"),
        "Random Forest": (rf_model, X_test, "random_forest")
    }
    
    metrics_summary = []
    classification_reports = {}
    confusion_matrices = {}
    
    print("\n[3/5] Running model predictions & computing evaluation metrics...")
    for model_name, (model, feat_data, file_prefix) in models.items():
        # Measure prediction time
        t_start = time.perf_counter()
        y_pred = model.predict(feat_data)
        t_pred = time.perf_counter() - t_start
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, labels=class_order, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, labels=class_order, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, labels=class_order, average="macro", zero_division=0)
        
        prec_weighted = precision_score(y_test, y_pred, labels=class_order, average="weighted", zero_division=0)
        rec_weighted = recall_score(y_test, y_pred, labels=class_order, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, labels=class_order, average="weighted", zero_division=0)
        
        # Classification report dict
        cr_dict = classification_report(y_test, y_pred, labels=class_order, output_dict=True, zero_division=0)
        classification_reports[model_name] = cr_dict
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=class_order)
        confusion_matrices[model_name] = cm.tolist()
        
        metrics_summary.append({
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "Macro Precision": round(prec_macro, 4),
            "Macro Recall": round(rec_macro, 4),
            "Macro F1": round(f1_macro, 4),
            "Weighted Precision": round(prec_weighted, 4),
            "Weighted Recall": round(rec_weighted, 4),
            "Weighted F1": round(f1_weighted, 4),
            "Testing Time": round(t_pred, 4)
        })
        
        print(f"\n  === {model_name} ===")
        print(f"  Accuracy:         {acc*100:.2f}%")
        print(f"  Macro Precision:  {prec_macro*100:.2f}%")
        print(f"  Macro Recall:     {rec_macro*100:.2f}%")
        print(f"  Macro F1-score:   {f1_macro*100:.2f}%")
        print(f"  Weighted F1:      {f1_weighted*100:.2f}%")
        print(f"  Inference Time:   {t_pred:.4f}s ({t_pred/n_test*1e6:.2f} \u03bcs/sample)")
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
        # Compute percentage matrix for annotations
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0) * 100
        
        annot_matrix = np.empty_like(cm, dtype=object)
        for i in range(len(class_order)):
            for j in range(len(class_order)):
                annot_matrix[i, j] = f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)"
                
        sns.heatmap(
            cm,
            annot=annot_matrix,
            fmt="",
            cmap="Blues",
            xticklabels=class_order,
            yticklabels=class_order,
            cbar=True,
            linewidths=1.0,
            linecolor="white",
            ax=ax,
            annot_kws={"fontsize": 11, "fontweight": "bold"}
        )
        ax.set_title(f"Confusion Matrix: {model_name}\n(Unseen Test Data, N = {n_test:,})", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Congestion Level", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Congestion Level", fontsize=11, fontweight="bold")
        
        plt.tight_layout()
        cm_path = os.path.join(cm_dir, f"{file_prefix}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"  Saved confusion matrix: {cm_path}")
        
    # Save Model Comparison CSV
    comparison_df = pd.DataFrame(metrics_summary)
    comparison_csv_path = os.path.join(results_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\n[4/5] Saved model comparison table to: {comparison_csv_path}")
    
    # 4. Create Performance Visualization Chart
    print("\n[5/5] Generating Model Performance Comparison Bar Chart...")
    metrics_to_plot = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    colors = ["#64748b", "#3b82f6", "#10b981"]
    
    for idx, row in comparison_df.iterrows():
        model_name = row["Model"]
        vals = [row[m] for m in metrics_to_plot]
        offset = (idx - 1) * width
        rects = ax.bar(x + offset, vals, width, label=model_name, color=colors[idx], edgecolor="black", linewidth=0.8)
        
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h*100:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")
                        
    ax.set_ylabel("Metric Score (0.0 to 1.0)", fontsize=12, fontweight="bold")
    ax.set_title("Model Performance Comparison on Unseen Test Set (N = 40,000)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", fontsize=11)
    
    plt.tight_layout()
    chart_path = os.path.join(results_dir, "model_performance_comparison.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"  Saved comparison chart to: {chart_path}")
    
    # Save complete metrics details to JSON
    eval_details = {
        "model_comparison": metrics_summary,
        "classification_reports": classification_reports,
        "confusion_matrices": confusion_matrices
    }
    details_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(details_path, "w") as fp:
        json.dump(eval_details, fp, indent=2)
    print(f"  Saved structured evaluation metrics: {details_path}")
    
    print("\n" + "=" * 70)
    print("STEP 7 MODEL EVALUATION COMPLETE!")
    print("=" * 70)
    return eval_details

if __name__ == "__main__":
    evaluate_models()
