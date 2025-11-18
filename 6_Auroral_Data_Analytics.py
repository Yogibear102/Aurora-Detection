#!/usr/bin/env python3
"""
Enhanced Auroral Activity Data Analytics & Visualization
=========================================================
Comprehensive visual exploration of Aditya-L1 MAG data for space weather forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

# Professional styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
FIGSIZE_LARGE = (14, 8)
FIGSIZE_MEDIUM = (10, 6)
FIGSIZE_SMALL = (8, 5)

# ===========================
# CONFIG
# ===========================
TRAIN_PATH = "/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/data/Aurora_Train.csv"
TEST_PATH = "/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/data/Aurora_Test.csv"
MODEL_PATH = "/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/models/classification/random_forest.joblib"
OUTPUT_DIR = "/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/plots/data_analytics"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# LOAD & PREPARE DATA
# ===========================
print("📊 Loading Aditya-L1 MAG Data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"✓ Train: {len(train_df):,} samples | Test: {len(test_df):,} samples")
print(f"✓ Features: {train_df.columns.tolist()}")

# Identify feature columns
exclude_cols = ['aurora_label', 'datetime', 'source_file', 'time']
feature_cols = [c for c in train_df.columns if c not in exclude_cols]
FEATURES_TO_PLOT = feature_cols[:8] if len(feature_cols) > 8 else feature_cols

# Convert datetime if available
if 'datetime' in train_df.columns:
    train_df['datetime'] = pd.to_datetime(train_df['datetime'], errors='coerce')
    test_df['datetime'] = pd.to_datetime(test_df['datetime'], errors='coerce')

# ===========================
# 1. DATASET OVERVIEW
# ===========================
print("\n📈 Creating Dataset Overview...")
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)
fig.suptitle('Aditya-L1 Auroral Dataset Overview', fontsize=16, fontweight='bold')

# Class distribution
ax = axes[0, 0]
train_counts = train_df['aurora_label'].value_counts()
colors = ['#2ecc71', '#e74c3c']
ax.bar(['Quiet (0)', 'Active (1)'], train_counts.values, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Aurora Activity Distribution (Training)', fontweight='bold')
for i, v in enumerate(train_counts.values):
    ax.text(i, v + 100, f'{v:,}\n({v/len(train_df)*100:.1f}%)', ha='center', fontweight='bold')

# Missing values heatmap
ax = axes[0, 1]
missing = train_df[feature_cols].isnull().sum().sort_values(ascending=False).head(15)
if missing.sum() > 0:
    ax.barh(range(len(missing)), missing.values, color='coral', alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(missing)))
    ax.set_yticklabels(missing.index, fontsize=9)
    ax.set_xlabel('Missing Count', fontsize=11)
    ax.set_title('Missing Values by Feature', fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No Missing Values ✓', ha='center', va='center', 
            fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.axis('off')

# Feature correlation with target
ax = axes[1, 0]
correlations = []
for col in feature_cols[:20]:
    if train_df[col].dtype in ['float64', 'int64']:
        corr = train_df[col].corr(train_df['aurora_label'])
        correlations.append((col, corr))
correlations = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)[:10]
cols, corrs = zip(*correlations)
colors_corr = ['#e74c3c' if c > 0 else '#3498db' for c in corrs]
ax.barh(range(len(cols)), corrs, color=colors_corr, alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(cols)))
ax.set_yticklabels(cols, fontsize=9)
ax.set_xlabel('Correlation with Aurora Activity', fontsize=11)
ax.set_title('Top 10 Feature Correlations', fontweight='bold')
ax.axvline(0, color='black', linewidth=0.8)

# Data statistics
ax = axes[1, 1]
stats_text = f"""
DATASET STATISTICS
{'='*35}

Training Samples: {len(train_df):,}
Test Samples: {len(test_df):,}
Total Features: {len(feature_cols)}

Active Aurora Events: {train_counts.get(1, 0):,}
Quiet Periods: {train_counts.get(0, 0):,}
Activity Rate: {train_counts.get(1, 0)/len(train_df)*100:.2f}%

Class Imbalance Ratio: {train_counts.get(0, 0)/train_counts.get(1, 1):.2f}:1
"""
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax.axis('off')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/00_dataset_overview.png", dpi=300, bbox_inches='tight')
plt.close()

# ===========================
# 2. ENHANCED FEATURE DISTRIBUTIONS
# ===========================
print("📊 Creating Feature Distributions...")
n_features = len(FEATURES_TO_PLOT)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
axes = axes.flatten() if n_rows > 1 else [axes]
fig.suptitle('Magnetic Field Feature Distributions (Aditya-L1 MAG)', fontsize=16, fontweight='bold')

for idx, feature in enumerate(FEATURES_TO_PLOT):
    ax = axes[idx]
    
    # KDE plots with train/test comparison
    sns.kdeplot(data=train_df[feature].dropna(), label='Train', ax=ax, 
                fill=True, alpha=0.5, linewidth=2)
    sns.kdeplot(data=test_df[feature].dropna(), label='Test', ax=ax, 
                fill=True, alpha=0.5, linewidth=2)
    
    # Add statistical annotations
    train_mean = train_df[feature].mean()
    ax.axvline(train_mean, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Mean: {train_mean:.2f}')
    
    ax.set_title(f'{feature}', fontweight='bold', fontsize=11)
    ax.set_xlabel('')
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(n_features, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_feature_distributions.png", dpi=300, bbox_inches='tight')
plt.close()

# ===========================
# 3. CLASS-WISE COMPARISONS (VIOLIN + BOX)
# ===========================
print("🎻 Creating Class-wise Comparisons...")
n_cols = 2
n_rows = (len(FEATURES_TO_PLOT) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
axes = axes.flatten() if n_rows > 1 else [axes]
fig.suptitle('Magnetic Features by Aurora Activity State', fontsize=16, fontweight='bold')

for idx, feature in enumerate(FEATURES_TO_PLOT):
    ax = axes[idx]
    
    # Violin plot with box plot overlay
    parts = ax.violinplot([train_df[train_df['aurora_label']==0][feature].dropna(),
                           train_df[train_df['aurora_label']==1][feature].dropna()],
                          positions=[0, 1], showmeans=True, showmedians=True)
    
    # Color the violins
    for pc, color in zip(parts['bodies'], ['#2ecc71', '#e74c3c']):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Quiet (0)', 'Active (1)'])
    ax.set_ylabel(feature, fontsize=11, fontweight='bold')
    ax.set_title(f'{feature} Distribution', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistical test
    quiet_vals = train_df[train_df['aurora_label']==0][feature].dropna()
    active_vals = train_df[train_df['aurora_label']==1][feature].dropna()
    if len(quiet_vals) > 0 and len(active_vals) > 0:
        t_stat, p_val = stats.ttest_ind(quiet_vals, active_vals)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax.text(0.5, 0.95, f'p-value: {p_val:.2e} {sig}', 
                transform=ax.transAxes, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

for idx in range(len(FEATURES_TO_PLOT), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_classwise_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# ===========================
# 4. TIME-SERIES ANALYSIS
# ===========================
print("⏰ Creating Time-series Analysis...")
for feature in FEATURES_TO_PLOT[:4]:  # Top 4 features
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_LARGE, height_ratios=[3, 1])
    fig.suptitle(f'Temporal Analysis: {feature} (Aditya-L1)', fontsize=14, fontweight='bold')
    
    # Main time series
    ax = axes[0]
    if 'datetime' in train_df.columns and train_df['datetime'].notna().any():
        x_axis = train_df['datetime']
        xlabel = 'Date/Time'
    else:
        x_axis = train_df.index
        xlabel = 'Sample Index'
    
    # Rolling mean
    window = min(100, len(train_df)//10)
    rolling_mean = train_df[feature].rolling(window=window, center=True).mean()
    rolling_std = train_df[feature].rolling(window=window, center=True).std()
    
    ax.plot(x_axis, train_df[feature], alpha=0.3, color='gray', linewidth=0.5, label='Raw')
    ax.plot(x_axis, rolling_mean, color='blue', linewidth=2, label=f'Rolling Mean (w={window})')
    ax.fill_between(x_axis, rolling_mean - rolling_std, rolling_mean + rolling_std, 
                     alpha=0.2, color='blue', label='±1 Std Dev')
    
    # Highlight active aurora periods
    active_mask = train_df['aurora_label'] == 1
    ax.scatter(x_axis[active_mask], train_df[feature][active_mask], 
              color='red', s=20, alpha=0.6, label='Active Aurora', zorder=5)
    
    ax.set_ylabel(f'{feature}', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Aurora activity timeline
    ax = axes[1]
    ax.fill_between(x_axis, 0, train_df['aurora_label'], 
                    color='red', alpha=0.5, step='mid')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Aurora\nActivity', fontsize=9)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Quiet', 'Active'])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_timeseries_{feature}.png", dpi=300, bbox_inches='tight')
    plt.close()

# ===========================
# 5. CORRELATION MATRIX
# ===========================
print("🔗 Creating Correlation Matrix...")
fig, ax = plt.subplots(figsize=(12, 10))
feature_subset = feature_cols[:20]  # Top 20 features
corr_matrix = train_df[feature_subset].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot_kws={'size': 8}, ax=ax)
ax.set_title('Feature Correlation Matrix (Aditya-L1 MAG)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# ===========================
# 6. FEATURE IMPORTANCE
# ===========================
print("🎯 Creating Feature Importance Analysis...")
try:
    rf_model = joblib.load(MODEL_PATH)
    importances = rf_model.feature_importances_
    
    # Get feature names from model
    model_features = train_df.drop(columns=exclude_cols, errors='ignore').columns
    feat_importances = pd.DataFrame({
        'feature': model_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)
    fig.suptitle('Feature Importance Analysis (Random Forest)', fontsize=14, fontweight='bold')
    
    # Top features bar plot
    ax = axes[0]
    top_n = 15
    top_features = feat_importances.head(top_n)
    colors_imp = plt.cm.RdYlGn(np.linspace(0.4, 0.9, len(top_features)))
    ax.barh(range(len(top_features)), top_features['importance'], color=colors_imp, edgecolor='black')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'], fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title(f'Top {top_n} Most Important Features', fontweight='bold')
    ax.invert_yaxis()
    
    # Cumulative importance
    ax = axes[1]
    cumsum = feat_importances['importance'].cumsum()
    ax.plot(range(1, len(cumsum)+1), cumsum, marker='o', linewidth=2, markersize=4)
    ax.axhline(0.8, color='red', linestyle='--', label='80% Threshold')
    ax.axhline(0.95, color='orange', linestyle='--', label='95% Threshold')
    ax.set_xlabel('Number of Features', fontsize=11)
    ax.set_ylabel('Cumulative Importance', fontsize=11)
    ax.set_title('Cumulative Feature Importance', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Top 5 features: {', '.join(feat_importances.head(5)['feature'].tolist())}")
    
except Exception as e:
    print(f"⚠️  Could not load RF model for feature importance: {e}")

# ===========================
# 7. PCA & T-SNE VISUALIZATION
# ===========================
print("🔬 Creating Dimensionality Reduction Visualizations...")
X = train_df[feature_cols].fillna(0).values
y = train_df['aurora_label'].values

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)
fig.suptitle('Feature Space Visualization (Aditya-L1 MAG)', fontsize=14, fontweight='bold')

# PCA
ax = axes[0]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', 
                     alpha=0.6, s=30, edgecolors='k', linewidth=0.3)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11)
ax.set_title('PCA Projection', fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Aurora Label')
ax.grid(True, alpha=0.3)

# t-SNE
ax = axes[1]
print("  Computing t-SNE (this may take a moment)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X[:5000])  # Limit for speed
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y[:5000], cmap='coolwarm',
                     alpha=0.6, s=30, edgecolors='k', linewidth=0.3)
ax.set_xlabel('t-SNE 1', fontsize=11)
ax.set_ylabel('t-SNE 2', fontsize=11)
ax.set_title('t-SNE Projection', fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Aurora Label')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_dimensionality_reduction.png", dpi=300, bbox_inches='tight')
plt.close()

# ===========================
# 8. ADVANCED: PAIR PLOT
# ===========================
print("📊 Creating Pair Plot...")
top_features_for_pair = FEATURES_TO_PLOT[:4] + ['aurora_label']
pair_df = train_df[top_features_for_pair].sample(n=min(2000, len(train_df)), random_state=42)

g = sns.pairplot(pair_df, hue='aurora_label', palette={0: '#2ecc71', 1: '#e74c3c'},
                 diag_kind='kde', plot_kws={'alpha': 0.6, 's': 20, 'edgecolor': 'k', 'linewidth': 0.3},
                 diag_kws={'alpha': 0.7})
g.fig.suptitle('Feature Pair Plot (Top 4 Features)', y=1.01, fontsize=14, fontweight='bold')
plt.savefig(f"{OUTPUT_DIR}/07_pairplot.png", dpi=300, bbox_inches='tight')
plt.close()

# ===========================
# 9. SUMMARY REPORT
# ===========================
print("\n" + "="*60)
print("📊 AURORAL ACTIVITY PREDICTION - DATA ANALYTICS SUMMARY")
print("="*60)
print(f"\n✅ All visualizations saved to: {OUTPUT_DIR}")
print("\nGenerated Visualizations:")
print("  1. Dataset Overview (class distribution, missing values, correlations)")
print("  2. Feature Distributions (train/test comparison)")
print("  3. Class-wise Comparisons (violin plots with statistical tests)")
print("  4. Time-series Analysis (with rolling statistics)")
print("  5. Correlation Matrix")
print("  6. Feature Importance (Random Forest)")
print("  7. Dimensionality Reduction (PCA & t-SNE)")
print("  8. Feature Pair Plot")
print("\n" + "="*60)
print("🚀 Ready for Space Weather Forecasting!")
print("="*60)