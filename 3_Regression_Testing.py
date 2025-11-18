#!/usr/bin/env python3
"""
MAG L2 Regression Pipeline - Predicting |B| at t+1
=====================================================
Predicts continuous magnetic field magnitude values using:
- Random Forest Regressor
- XGBoost Regressor
- LSTM Neural Network

Includes hyperparameter tuning, feature importance analysis,
and comprehensive evaluation metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================
# CONFIGURATION
# ============================================
DATA_PATH = "/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/data/MAG_L2_model_dataset.csv"
OUTPUT_DIR = "/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/models/regression"
PLOTS_DIR = "/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/plots/regression"

# Create output directories
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# LSTM sequence parameters
SEQUENCE_LENGTH = 10  # Use 10 timesteps to predict next value
TEST_SIZE = 0.2

print("=" * 70)
print("MAG L2 REGRESSION PIPELINE")
print("=" * 70)

# ============================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================
print("\n[1/8] Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# Define target: |B| at t+1 (next timestep)
df['B_mag_target'] = df['B_mag'].shift(-1)
df = df.dropna()

print(f"Shape after creating target: {df.shape}")

# Define features (exclude time and target columns)
exclude_cols = ['time', 'B_mag_target', 'source_file'] if 'time' in df.columns else ['B_mag_target', 'source_file']
feature_cols = [col for col in df.columns if col not in exclude_cols and col in df.select_dtypes(include=[np.number]).columns]

X = df[feature_cols].values
y = df['B_mag_target'].values

print(f"Features: {len(feature_cols)}")
print(f"Samples: {len(X)}")

# Time-series split (80/20)
split_idx = int(len(X) * (1 - TEST_SIZE))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ============================================
# STEP 2: BASELINE - RANDOM FOREST
# ============================================
print("\n[2/8] Training Random Forest Regressor...")

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print(f"Random Forest - RMSE: {rf_rmse:.4f}")
print(f"Random Forest - MAE: {rf_mae:.4f}")
print(f"Random Forest - R²: {rf_r2:.4f}")

# Save model
joblib.dump(rf_model, f"{OUTPUT_DIR}/random_forest_model.joblib")
print(f"Saved: {OUTPUT_DIR}/random_forest_model.joblib")

# ============================================
# STEP 3: XGBOOST REGRESSOR
# ============================================
print("\n[3/8] Training XGBoost Regressor...")

xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

print(f"XGBoost - RMSE: {xgb_rmse:.4f}")
print(f"XGBoost - MAE: {xgb_mae:.4f}")
print(f"XGBoost - R²: {xgb_r2:.4f}")

# Save model
joblib.dump(xgb_model, f"{OUTPUT_DIR}/xgboost_model.joblib")
print(f"Saved: {OUTPUT_DIR}/xgboost_model.joblib")

# ============================================
# STEP 4: PREPARE SEQUENCES FOR LSTM
# ============================================
print(f"\n[4/8] Creating sequences for LSTM (length={SEQUENCE_LENGTH})...")

def create_sequences(X, y, seq_length):
    """Create sequences for LSTM input"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

# Create sequences from training data
X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQUENCE_LENGTH)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQUENCE_LENGTH)

print(f"LSTM Train sequences: {X_train_seq.shape}")
print(f"LSTM Test sequences: {X_test_seq.shape}")

# ============================================
# STEP 5: BUILD AND TRAIN LSTM
# ============================================
print("\n[5/8] Building and training LSTM model...")

lstm_model = keras.Sequential([
    layers.LSTM(64, activation='relu', return_sequences=True, 
                input_shape=(SEQUENCE_LENGTH, X_train.shape[1])),
    layers.Dropout(0.2),
    layers.LSTM(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

lstm_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print(lstm_model.summary())

# Early stopping callback
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train LSTM
history = lstm_model.fit(
    X_train_seq, y_train_seq,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Predict with LSTM
lstm_pred = lstm_model.predict(X_test_seq).flatten()

lstm_rmse = np.sqrt(mean_squared_error(y_test_seq, lstm_pred))
lstm_mae = mean_absolute_error(y_test_seq, lstm_pred)
lstm_r2 = r2_score(y_test_seq, lstm_pred)

print(f"\nLSTM - RMSE: {lstm_rmse:.4f}")
print(f"LSTM - MAE: {lstm_mae:.4f}")
print(f"LSTM - R²: {lstm_r2:.4f}")

# Save LSTM model
lstm_model.save(f"{OUTPUT_DIR}/lstm_model.keras")
print(f"Saved: {OUTPUT_DIR}/lstm_model.keras")

# ============================================
# STEP 6: HYPERPARAMETER TUNING (Random Forest)
# ============================================
print("\n[6/8] Hyperparameter tuning for Random Forest...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [15, 20, 25],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}

tscv = TimeSeriesSplit(n_splits=3)

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

print(f"Best parameters: {grid_search.best_params_}")

best_rf_pred = best_rf.predict(X_test)
best_rf_rmse = np.sqrt(mean_squared_error(y_test, best_rf_pred))
best_rf_r2 = r2_score(y_test, best_rf_pred)

print(f"Tuned RF - RMSE: {best_rf_rmse:.4f}")
print(f"Tuned RF - R²: {best_rf_r2:.4f}")

joblib.dump(best_rf, f"{OUTPUT_DIR}/random_forest_tuned.joblib")

# ============================================
# STEP 7: EVALUATION VISUALIZATIONS
# ============================================
print("\n[7/8] Creating evaluation plots...")

# 1. Model Comparison
results = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'LSTM', 'RF (Tuned)'],
    'RMSE': [rf_rmse, xgb_rmse, lstm_rmse, best_rf_rmse],
    'MAE': [rf_mae, xgb_mae, lstm_mae, mean_absolute_error(y_test, best_rf_pred)],
    'R²': [rf_r2, xgb_r2, lstm_r2, best_rf_r2]
})

print("\nFinal Model Comparison:")
print(results.to_string(index=False))
results.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, metric in enumerate(['RMSE', 'MAE', 'R²']):
    axes[i].bar(results['Model'], results[metric], color='steelblue')
    axes[i].set_title(f'{metric} Comparison')
    axes[i].set_ylabel(metric)
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/model_comparison.png", dpi=150)
plt.close()

# 2. Prediction vs Actual (XGBoost - best traditional model)
plt.figure(figsize=(12, 5))
plt.plot(y_test[:500], label='Actual', alpha=0.7)
plt.plot(xgb_pred[:500], label='Predicted (XGBoost)', alpha=0.7)
plt.title('XGBoost: Predicted vs Actual |B|')
plt.xlabel('Time Steps')
plt.ylabel('|B| (scaled)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"{PLOTS_DIR}/xgboost_predictions.png", dpi=150)
plt.close()

# 3. Residual Plot
residuals = y_test - xgb_pred
plt.figure(figsize=(12, 5))
plt.scatter(xgb_pred, residuals, alpha=0.3, s=1)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('XGBoost Residual Plot')
plt.xlabel('Predicted |B|')
plt.ylabel('Residuals')
plt.grid(alpha=0.3)
plt.savefig(f"{PLOTS_DIR}/residual_plot.png", dpi=150)
plt.close()

# 4. Feature Importance (XGBoost)
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances (XGBoost)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/feature_importance.png", dpi=150)
plt.close()

print(f"\nTop 10 Important Features:")
print(importance_df.head(10).to_string(index=False))

# 5. LSTM Training History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('LSTM Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('LSTM Training MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/lstm_training_history.png", dpi=150)
plt.close()

# ============================================
# STEP 8: ERROR ANALYSIS
# ============================================
print("\n[8/8] Running error analysis...")

# Calculate error quantiles
errors = np.abs(y_test - xgb_pred)
error_quantiles = np.percentile(errors, [50, 75, 90, 95, 99])

print("\nError Distribution (MAE quantiles):")
print(f"50th percentile: {error_quantiles[0]:.4f}")
print(f"75th percentile: {error_quantiles[1]:.4f}")
print(f"90th percentile: {error_quantiles[2]:.4f}")
print(f"95th percentile: {error_quantiles[3]:.4f}")
print(f"99th percentile: {error_quantiles[4]:.4f}")

# Error distribution plot
plt.figure(figsize=(10, 5))
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(error_quantiles[2], color='r', linestyle='--', label='90th percentile')
plt.title('Prediction Error Distribution (XGBoost)')
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"{PLOTS_DIR}/error_distribution.png", dpi=150)
plt.close()

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 70)
print("REGRESSION PIPELINE COMPLETE!")
print("=" * 70)
print(f"\nBest Model: {'XGBoost' if xgb_r2 > lstm_r2 else 'LSTM'}")
print(f"Best R² Score: {max(xgb_r2, lstm_r2):.4f}")
print(f"\nAll models saved to: {OUTPUT_DIR}")
print(f"All plots saved to: {PLOTS_DIR}")
print("\nNext steps:")
print("1. Use the best model for real-time prediction")
print("2. Deploy model with inference pipeline")
print("3. Monitor model performance on new data")
print("=" * 70)