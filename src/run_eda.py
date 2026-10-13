import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Set professional aesthetics
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
sns.set_theme(style="whitegrid")

def run_eda(
    train_path="./dataset/processed_train.csv",
    output_dir="./results",
    graphs_dir="./results/graphs"
):
    print("=" * 70)
    print("STEP 5: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    
    # 1. Load Training Data
    print(f"Loading training dataset from: {train_path}")
    train_df = pd.read_csv(train_path)
    n_rows, n_cols = train_df.shape
    print(f"Loaded: {n_rows:,} records, {n_cols} columns")
    
    predictors = [c for c in train_df.columns if c != 'Congestion_Level']
    target_col = 'Congestion_Level'
    
    # Ensure directories exist
    feat_dist_dir = os.path.join(graphs_dir, "feature_distributions")
    rel_dir = os.path.join(graphs_dir, "congestion_relationships")
    outlier_dir = os.path.join(graphs_dir, "outliers")
    os.makedirs(feat_dist_dir, exist_ok=True)
    os.makedirs(rel_dir, exist_ok=True)
    os.makedirs(outlier_dir, exist_ok=True)
    
    palette = {"LOW": "#3b82f6", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}
    
    # =========================================================================
    # 2. TARGET DISTRIBUTION PLOT
    # =========================================================================
    print("\n[1/6] Generating Congestion Class Distribution Plot...")
    class_counts = train_df[target_col].value_counts()[['LOW', 'MEDIUM', 'HIGH']]
    class_pcts = (class_counts / n_rows) * 100
    
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    bars = ax.bar(['LOW', 'MEDIUM', 'HIGH'], class_counts, color=['#3b82f6', '#f59e0b', '#ef4444'], edgecolor='black', linewidth=1.0)
    ax.set_title('Training Target Distribution: Congestion Level', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Flow Records', fontsize=12, fontweight='bold')
    ax.set_xlabel('Congestion Class', fontsize=12, fontweight='bold')
    ax.set_ylim(0, class_counts.max() * 1.2)
    
    for bar in bars:
        h = bar.get_height()
        pct = (h / n_rows) * 100
        ax.annotate(f"{h:,}\n({pct:.2f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    plt.tight_layout()
    target_dist_path = os.path.join(graphs_dir, "congestion_class_distribution.png")
    plt.savefig(target_dist_path)
    plt.close()
    print(f"  Saved: {target_dist_path}")

    # =========================================================================
    # 3. SELECT 10 IMPORTANT FEATURES FOR IN-DEPTH ANALYSIS
    # =========================================================================
    selected_features = [
        "Packet Length Mean",
        "Average Packet Size",
        "Packet Length Std",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
        "Active Mean",
        "Idle Mean",
        "ACK Flag Count",
        "Destination Port"
    ]
    
    # Feature explanations for report
    feature_meta = {
        "Packet Length Mean": "Measures average payload sizing; distinguishes bulky data packets from small control floods.",
        "Average Packet Size": "Overall mean segment size per flow across forward and backward directions.",
        "Packet Length Std": "Reflects packet heterogeneity; video/interactive bursts show high variance vs uniform probing.",
        "Total Fwd Packets": "Primary volume indicator of client-side transmission load.",
        "Total Backward Packets": "Server-side response volume; critical for detecting request/response asymmetry.",
        "Total Length of Fwd Packets": "Aggregate client byte volume indicating payload pressure.",
        "Init_Win_bytes_forward": "Client TCP receive window advertisement, showing buffering capacity and flow control limits.",
        "Init_Win_bytes_backward": "Server TCP receive window size, revealing server buffer pressure.",
        "Active Mean": "Mean duration of active transmission bursts before entering idle state.",
        "Idle Mean": "Mean duration of channel silence between bursts; directly differentiates continuous from sparse traffic.",
        "ACK Flag Count": "Transport-layer acknowledgment frequency, signaling bidirectional session health.",
        "Destination Port": "Service-layer identifier distinguishing common services (80, 443, 53) from ephemeral traffic."
    }

    # =========================================================================
    # 4. FEATURE DISTRIBUTIONS (HISTOGRAMS / LOG TRANSFORMS)
    # =========================================================================
    print("\n[2/6] Generating Individual Feature Distribution Plots...")
    for feat in selected_features:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
        s = train_df[feat].copy()
        
        # Apply log1p for heavily skewed features for clear visualization
        skew_val = s.skew()
        is_skewed = abs(skew_val) > 2.0 and s.min() >= 0
        
        if is_skewed:
            plot_data = np.log1p(s)
            xlabel = f"log10(1 + {feat})"
            plot_data = np.log10(1.0 + s)
        else:
            plot_data = s
            xlabel = feat
            
        sns.histplot(plot_data, bins=40, kde=True, ax=ax, color="#2563eb", edgecolor="black", alpha=0.6)
        ax.set_title(f"Distribution of {feat}\n(Skewness: {skew_val:.2f})", fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
        ax.set_ylabel("Frequency", fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        safe_name = feat.replace(" ", "_").replace("/", "_").lower()
        out_f = os.path.join(feat_dist_dir, f"dist_{safe_name}.png")
        plt.savefig(out_f)
        plt.close()
        
    print(f"  Saved 12 individual distribution plots to: {feat_dist_dir}")

    # =========================================================================
    # 5. CONGESTION LEVEL VS IMPORTANT FEATURES (RELATIONSHIPS)
    # =========================================================================
    print("\n[3/6] Generating Congestion Level vs. Feature Relationship Plots...")
    for feat in selected_features:
        fig, ax = plt.subplots(figsize=(7, 4.8), dpi=130)
        s = train_df[feat].copy()
        
        # Use log10 scale on y-axis if heavily skewed and positive
        if s.min() >= 0 and s.max() > 100 and s.skew() > 2.0:
            plot_df = train_df[[feat, target_col]].copy()
            plot_df['log_val'] = np.log10(1.0 + plot_df[feat])
            sns.boxplot(x=target_col, y='log_val', data=plot_df, order=['LOW', 'MEDIUM', 'HIGH'],
                        palette=palette, ax=ax, showmeans=True,
                        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "7"})
            ax.set_ylabel(f"log10(1 + {feat})", fontsize=11, fontweight='bold')
        else:
            sns.boxplot(x=target_col, y=feat, data=train_df, order=['LOW', 'MEDIUM', 'HIGH'],
                        palette=palette, ax=ax, showmeans=True,
                        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "7"})
            ax.set_ylabel(feat, fontsize=11, fontweight='bold')
            
        ax.set_title(f"{feat} by Congestion Level\n(White dot represents mean)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Congestion Level", fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        safe_name = feat.replace(" ", "_").replace("/", "_").lower()
        out_f = os.path.join(rel_dir, f"rel_{safe_name}.png")
        plt.savefig(out_f)
        plt.close()
        
    print(f"  Saved 12 relationship plots to: {rel_dir}")

    # =========================================================================
    # 6. CORRELATION ANALYSIS & HEATMAP
    # =========================================================================
    print("\n[4/6] Generating Predictor Correlation Heatmap...")
    # Filter out constant columns for correlation
    non_const_cols = [c for c in predictors if train_df[c].nunique() > 1]
    const_cols = [c for c in predictors if train_df[c].nunique() <= 1]
    
    corr_matrix = train_df[non_const_cols].corr()
    
    # Save large, readable heatmap
    fig, ax = plt.subplots(figsize=(24, 20), dpi=120)
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,
        linewidths=0.2,
        linecolor="#cbd5e1",
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        ax=ax
    )
    ax.set_title(f"Pearson Correlation Matrix of Retained Predictor Features ({len(non_const_cols)} Features)", fontsize=18, fontweight='bold', pad=20)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    corr_path = os.path.join(graphs_dir, "correlation_heatmap.png")
    plt.savefig(corr_path)
    plt.close()
    print(f"  Saved: {corr_path}")
    
    # Find top correlations
    pairs = []
    for i in range(len(non_const_cols)):
        for j in range(i + 1, len(non_const_cols)):
            f1, f2 = non_const_cols[i], non_const_cols[j]
            val = corr_matrix.loc[f1, f2]
            pairs.append((f1, f2, val))
            
    top_pos = sorted([p for p in pairs if p[2] > 0], key=lambda x: -x[2])[:12]
    top_neg = sorted([p for p in pairs if p[2] < 0], key=lambda x: x[2])[:10]

    # =========================================================================
    # 7. OUTLIER ANALYSIS
    # =========================================================================
    print("\n[5/6] Generating Outlier Boxplots...")
    outlier_features = [
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Active Mean",
        "Idle Mean"
    ]
    
    for feat in outlier_features:
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
        s = train_df[feat]
        
        # Calculate IQR outlier metrics
        q25 = float(s.quantile(0.25))
        q75 = float(s.quantile(0.75))
        iqr = q75 - q25
        upper_bound = q75 + 1.5 * iqr
        outliers_count = int((s > upper_bound).sum())
        outlier_pct = (outliers_count / n_rows) * 100
        
        # Plot with log scale
        log_s = np.log10(1.0 + s.clip(lower=0))
        sns.boxplot(x=log_s, ax=ax, color="#60a5fa", fliersize=2.5, flierprops={"marker": "d", "color": "#dc2626", "alpha": 0.4})
        ax.set_title(f"Outlier Distribution: {feat}\n({outliers_count:,} records > 1.5*IQR, {outlier_pct:.2f}%)", fontsize=11, fontweight='bold')
        ax.set_xlabel(f"log10(1 + {feat})", fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        safe_name = feat.replace(" ", "_").replace("/", "_").lower()
        out_f = os.path.join(outlier_dir, f"outlier_{safe_name}.png")
        plt.savefig(out_f)
        plt.close()
        
    print(f"  Saved 6 outlier plots to: {outlier_dir}")

    # =========================================================================
    # 8. CONGESTION-LEVEL FEATURE SUMMARY TABLE
    # =========================================================================
    print("\n[6/6] Computing Congestion Level Feature Summary Table...")
    summary_rows = []
    
    grouped_mean = train_df.groupby(target_col, observed=False)[selected_features].mean()
    grouped_median = train_df.groupby(target_col, observed=False)[selected_features].median()
    grouped_std = train_df.groupby(target_col, observed=False)[selected_features].std()
    
    for feat in selected_features:
        summary_rows.append({
            "Feature": feat,
            "LOW_Mean": round(grouped_mean.loc['LOW', feat], 2),
            "LOW_Median": round(grouped_median.loc['LOW', feat], 2),
            "LOW_Std": round(grouped_std.loc['LOW', feat], 2),
            "MEDIUM_Mean": round(grouped_mean.loc['MEDIUM', feat], 2),
            "MEDIUM_Median": round(grouped_median.loc['MEDIUM', feat], 2),
            "MEDIUM_Std": round(grouped_std.loc['MEDIUM', feat], 2),
            "HIGH_Mean": round(grouped_mean.loc['HIGH', feat], 2),
            "HIGH_Median": round(grouped_median.loc['HIGH', feat], 2),
            "HIGH_Std": round(grouped_std.loc['HIGH', feat], 2),
            "Description": feature_meta.get(feat, "")
        })
        
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "eda_feature_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    print("\n" + "=" * 70)
    print("EDA PROCESSING COMPLETE!")
    print("=" * 70)
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "predictors_count": len(predictors),
        "constant_features": const_cols,
        "top_positive_corr": top_pos,
        "top_negative_corr": top_neg,
        "summary_df": summary_df
    }

if __name__ == "__main__":
    run_eda()
