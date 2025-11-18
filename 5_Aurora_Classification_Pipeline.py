#!/usr/bin/env python3
"""
Enhanced Auroral Activity Classification Pipeline
================================================
Multi-model ensemble approach with comprehensive evaluation:
- Uses regression model predictions as meta-features
- Trains multiple classifiers (RF, XGBoost, LightGBM)
- Handles missing data and feature engineering
- Extensive evaluation metrics, plots, and reports
- Class imbalance handling
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, matthews_corrcoef
)
import joblib
import tensorflow as tf
from tensorflow import keras
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available, skipping LightGBM classifier")
    
import os
from datetime import datetime

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ===========================
# CONFIGURATION
# ===========================
class Config:
    TRAIN_PATH = "/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/data/Aurora_Train.csv"
    TEST_PATH  = "/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/data/Aurora_Test.csv"
    OUTPUT_DIR = "/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/models/classification"
    PLOTS_DIR  = "/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/plots/classification"
    
    RF_MODEL_PATH   = "/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/models/regression/random_forest_tuned.joblib"
    XGB_MODEL_PATH  = "/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/models/regression/xgboost_model.joblib"
    LSTM_MODEL_PATH = "/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/models/regression/lstm_model.keras"
    
    SEQ_LEN = 10
    RANDOM_STATE = 42
    EXCLUDE_COLS = ['datetime', 'aurora_label', 'source_file', 'time', 'AE']

# Create directories
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.PLOTS_DIR, exist_ok=True)

# ===========================
# UTILITY FUNCTIONS
# ===========================
def load_data():
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    train_df = pd.read_csv(Config.TRAIN_PATH)
    test_df  = pd.read_csv(Config.TEST_PATH)
    
    print(f"✓ Train samples: {len(train_df):,}")
    print(f"✓ Test samples:  {len(test_df):,}")
    print(f"✓ Train features: {train_df.shape[1]}")
    
    train_missing = train_df.isnull().sum().sum()
    test_missing = test_df.isnull().sum().sum()
    print(f"✓ Missing values - Train: {train_missing}, Test: {test_missing}")
    
    print("\nClass Distribution (Train):")
    print(train_df['aurora_label'].value_counts())
    print(f"Active ratio: {train_df['aurora_label'].mean():.2%}")
    
    return train_df, test_df


def create_sequences(X, seq_len):
    X_seq = []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i:i+seq_len])
    return np.array(X_seq)


def load_regression_models():
    print("\n" + "="*60)
    print("LOADING REGRESSION MODELS")
    print("="*60)
    
    models = {}
    try:
        models['rf'] = joblib.load(Config.RF_MODEL_PATH)
        print("✓ Random Forest model loaded")
    except Exception as e:
        print(f"⚠️ RF model load failed: {e}")
        models['rf'] = None
    try:
        models['xgb'] = joblib.load(Config.XGB_MODEL_PATH)
        print("✓ XGBoost model loaded")
    except Exception as e:
        print(f"⚠️ XGBoost model load failed: {e}")
        models['xgb'] = None
    try:
        models['lstm'] = keras.models.load_model(Config.LSTM_MODEL_PATH)
        print("✓ LSTM model loaded")
    except Exception as e:
        print(f"⚠️ LSTM model load failed: {e}")
        models['lstm'] = None
    
    return models


def generate_meta_features(X_train, X_test, y_train, y_test, models):
    print("\n" + "="*60)
    print("GENERATING META-FEATURES")
    print("="*60)
    
    X_train_seq = create_sequences(X_train.values, Config.SEQ_LEN)
    X_test_seq  = create_sequences(X_test.values, Config.SEQ_LEN)
    
    X_train_aligned = X_train.iloc[Config.SEQ_LEN-1:].copy().reset_index(drop=True)
    X_test_aligned  = X_test.iloc[Config.SEQ_LEN-1:].copy().reset_index(drop=True)
    y_train_aligned = y_train.iloc[Config.SEQ_LEN-1:].reset_index(drop=True)
    y_test_aligned  = y_test.iloc[Config.SEQ_LEN-1:].reset_index(drop=True)
    
    # === FIX: Use only original features for regression predictions ===
    X_train_orig = X_train_aligned.drop(columns=[c for c in X_train_aligned.columns if 'pred' in c], errors='ignore')
    X_test_orig  = X_test_aligned.drop(columns=[c for c in X_test_aligned.columns if 'pred' in c], errors='ignore')
    
    if models['rf'] is not None:
        X_train_aligned['rf_pred'] = models['rf'].predict(X_train_orig)
        X_test_aligned['rf_pred']  = models['rf'].predict(X_test_orig)
    if models['xgb'] is not None:
        X_train_aligned['xgb_pred'] = models['xgb'].predict(X_train_orig)
        X_test_aligned['xgb_pred']  = models['xgb'].predict(X_test_orig)
    if models['lstm'] is not None:
        X_train_aligned['lstm_pred'] = models['lstm'].predict(X_train_seq, verbose=0).flatten()
        X_test_aligned['lstm_pred']  = models['lstm'].predict(X_test_seq, verbose=0).flatten()
    
    print(f"✓ Final features: {X_train_aligned.shape[1]}")
    print(f"✓ Train samples: {len(X_train_aligned)}, Test samples: {len(X_test_aligned)}")
    
    return X_train_aligned, X_test_aligned, y_train_aligned, y_test_aligned


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'avg_precision': average_precision_score(y_test, y_proba),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    print(f"✓ {model_name}: Accuracy {metrics['accuracy']:.4f}, F1 {metrics['f1_score']:.4f}, ROC-AUC {metrics['roc_auc']:.4f}")
    return metrics


def train_classifiers(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    classifiers, results = {}, {}
    
    # Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=300, max_depth=25, min_samples_split=5, min_samples_leaf=2,
        max_features='sqrt', class_weight='balanced', random_state=Config.RANDOM_STATE,
        n_jobs=-1
    )
    rf_clf.fit(X_train, y_train)
    classifiers['Random Forest'] = rf_clf
    results['Random Forest'] = evaluate_model(rf_clf, X_test, y_test, "Random Forest")
    
    # XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=10, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
        random_state=Config.RANDOM_STATE, n_jobs=-1, verbosity=0
    )
    xgb_clf.fit(X_train, y_train)
    classifiers['XGBoost'] = xgb_clf
    results['XGBoost'] = evaluate_model(xgb_clf, X_test, y_test, "XGBoost")
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=200, max_depth=15, learning_rate=0.1, num_leaves=50,
            subsample=0.8, colsample_bytree=0.8, class_weight='balanced',
            random_state=Config.RANDOM_STATE, n_jobs=-1
        )
        lgb_clf.fit(X_train, y_train)
        classifiers['LightGBM'] = lgb_clf
        results['LightGBM'] = evaluate_model(lgb_clf, X_test, y_test, "LightGBM")
    
    # Logistic Regression
    lr_clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=Config.RANDOM_STATE, n_jobs=-1)
    lr_clf.fit(X_train_scaled, y_train)
    classifiers['Logistic Regression'] = (lr_clf, scaler)
    results['Logistic Regression'] = evaluate_model(lr_clf, X_test_scaled, y_test, "Logistic Regression")
    
    return classifiers, results


def create_visualizations(results, X_test, y_test, feature_names, classifiers):
    print("\nCreating visualizations...")
    models = list(results.keys())
    metrics_to_plot = ['accuracy', 'f1_score', 'roc_auc', 'mcc']
    
    # Metric bar plots
    plt.figure(figsize=(12,6))
    for metric in metrics_to_plot:
        plt.bar(models, [results[m][metric] for m in models])
    plt.tight_layout()
    
    # ROC and PR curves
    plt.figure(figsize=(10,8))
    for model_name in models:
        fpr, tpr, _ = roc_curve(y_test, results[model_name]['y_proba'])
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={results[model_name]["roc_auc"]:.3f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curves'); plt.legend(); plt.grid(alpha=0.3)
    
    plt.figure(figsize=(10,8))
    for model_name in models:
        precision, recall, _ = precision_recall_curve(y_test, results[model_name]['y_proba'])
        plt.plot(recall, precision, label=f'{model_name} (AP={results[model_name]["avg_precision"]:.3f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curves'); plt.legend(); plt.grid(alpha=0.3)
    
    # Confusion matrices
    fig, axes = plt.subplots(1,len(models), figsize=(6*len(models),5))
    if len(models) == 1: axes = [axes]
    for idx, model_name in enumerate(models):
        cm = confusion_matrix(y_test, results[model_name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues',
                    xticklabels=['Quiet','Active'], yticklabels=['Quiet','Active'])
        axes[idx].set_title(model_name)
    plt.tight_layout()


def save_models_and_results(classifiers, results):
    print("Saving models and results...")
    for name, model in classifiers.items():
        fname = name.replace(' ','_').lower()
        if isinstance(model, tuple):  # Logistic Regression + scaler
            joblib.dump(model[0], f"{Config.OUTPUT_DIR}/{fname}_model.joblib")
            joblib.dump(model[1], f"{Config.OUTPUT_DIR}/{fname}_scaler.joblib")
        else:
            joblib.dump(model, f"{Config.OUTPUT_DIR}/{fname}.joblib")
    summary = pd.DataFrame({m:{k:v for k,v in r.items() if k not in ['y_pred','y_proba']} for m,r in results.items()}).T
    summary.to_csv(f"{Config.OUTPUT_DIR}/model_comparison_results.csv")
    best_model = summary['roc_auc'].idxmax()
    with open(f"{Config.OUTPUT_DIR}/best_model_info.txt",'w') as f:
        f.write(f"Best Model: {best_model}\nROC-AUC: {summary.loc[best_model,'roc_auc']:.4f}\n")


# ===========================
# MAIN
# ===========================
def main():
    print("\nAurora Activity Classification Pipeline")
    train_df, test_df = load_data()
    
    feature_cols = [c for c in train_df.columns if c not in Config.EXCLUDE_COLS]
    X_train_raw = train_df[feature_cols].fillna(0)
    X_test_raw  = test_df[feature_cols].fillna(0)
    y_train = train_df['aurora_label']
    y_test  = test_df['aurora_label']
    
    regression_models = load_regression_models()
    
    X_train, X_test, y_train, y_test = generate_meta_features(
        X_train_raw, X_test_raw, y_train, y_test, regression_models
    )
    
    classifiers, results = train_classifiers(X_train, y_train, X_test, y_test)
    
    create_visualizations(results, X_test, y_test, X_train.columns, classifiers)
    save_models_and_results(classifiers, results)
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
