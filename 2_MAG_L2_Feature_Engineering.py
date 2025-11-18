#!/usr/bin/env python3
"""
MAG L2 Feature Engineering Pipeline
Complete preprocessing for Aditya-L1 MAG data
"""

import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
from scipy.signal import medfilt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

# ============================================
# STEP 1: LOAD DATA
# ============================================
# Load the preprocessed MAG_L2_final_features.csv file.
# Convert timestamps to datetime and sort them.
# Ensures the pipeline works on a strictly time-ordered dataset,
# which is essential for time-series modeling.
print("=" * 60)
print("MAG L2 FEATURE ENGINEERING PIPELINE")
print("=" * 60)

df = pd.read_csv("/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/data/MAG_L2_final_features.csv")
df["time"] = pd.to_datetime(df["time"], errors='coerce')
df = df.sort_values("time").reset_index(drop=True)

print("Raw shape:", df.shape)
print(df.head())

# ============================================
# STEP 2: SPIKE CLEANING
# ============================================
# This step removes sharp spikes in Bx, By, Bz using:
# - A rolling median/MAD threshold detector
# - Replacing outlier points with the local median
# - Final smoothing using a 5-point median filter
#
# Logic: Magnetometer sensors often produce high-frequency spikes
# due to spacecraft motion, instrument noise, or solar particle hits.
# These spikes corrupt dB/dt and rolling statistics. Cleaning them
# stabilizes the signal without distorting true magnetic variations.
print("\nStep 2: Cleaning spikes from magnetometer components...")

def clean_series(series):
    series = series.copy()
    window_size = 15                # length of sliding window
    n_sigmas = 3                    # MAD threshold multiplier
    k = 1.4826                      # converts MAD → std equivalent
    L = window_size//2              # half-window padding
    
    cleaned = series.copy()
    
    # Loop through all data points and detect spikes
    for i in range(L, len(series)-L):
        window = series[i-L:i+L+1]
        med = np.median(window)
        mad = median_abs_deviation(window)
        
        # Avoid division by zero in perfectly flat regions
        if mad == 0:
            mad = 1e-6
        
        threshold = n_sigmas * k * mad
        
        # If deviates too much from local median → spike
        if abs(series[i] - med) > threshold:
            cleaned[i] = med
    
    # Apply a median filter to smooth small local jumps
    cleaned = medfilt(cleaned, kernel_size=5)
    
    return cleaned

df["Bx_clean"] = clean_series(df["Bx"])
df["By_clean"] = clean_series(df["By"])
df["Bz_clean"] = clean_series(df["Bz"])

print("Spikes cleaned from raw magnetometer components.")

# ============================================
# STEP 3: RECOMPUTE MAGNITUDE AND DERIVATIVE
# ============================================
# Compute |B| = sqrt(Bx² + By² + Bz²) from cleaned components.
# Then compute dB/dt, the first-order temporal derivative.
#
# dB/dt is highly sensitive to spikes, so the recomputation AFTER
# cleaning ensures physically meaningful gradients.
#
# Interpolate any tiny gaps to maintain a continuous time-series.
print("\nStep 3: Recomputing B_mag and dB_dt from cleaned components...")

df["B_mag"] = np.sqrt(
    df["Bx_clean"]**2 +
    df["By_clean"]**2 +
    df["Bz_clean"]**2
)

df["dB_dt"] = df["B_mag"].diff()

# Safe interpolation for all numeric fields
df = df.set_index("time")
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].interpolate(method="linear")
df = df.reset_index()
df = df.dropna()

print(df.head())

# ============================================
# STEP 4: FEATURE ENGINEERING
# ============================================
# Create physically meaningful features for ML models:
# - Rolling means and stds (capture short-term trends)
# - Lag features (allow model to learn temporal dependencies)
# - Log transforms (stabilize variance for spiky geophysical data)
#
# Rolling windows: 5s & 10s typical for solar wind fluctuations.
print("\nStep 4: Creating advanced features...")

# Rolling windows
for w in [5, 10]:
    df[f"B_mag_mean_{w}"] = df["B_mag"].rolling(w).mean()
    df[f"B_mag_std_{w}"] = df["B_mag"].rolling(w).std()

# Lag features — essential for time-series prediction models
for lag in [1, 2, 3]:
    df[f"B_mag_lag_{lag}"] = df["B_mag"].shift(lag)
    df[f"dB_dt_lag_{lag}"] = df["dB_dt"].shift(lag)

# Log transforms — stabilizes long-tail distributions
df["B_mag_log10"] = np.log10(np.abs(df["B_mag"]) + 1e-12)
df["dB_dt_log10"] = np.log10(np.abs(df["dB_dt"]) + 1e-12)

df = df.dropna()
print("Feature engineering done. Shape:", df.shape)

# ============================================
# STEP 5: QUALITY CHECKS (VISUALIZATION)
# ============================================
# Quick sanity checks to validate:
# - Signal smoothness after cleaning
# - Noise levels (rolling std)
# - Autocorrelation structure (time-series dependency)
# - SNR to measure signal vs noise power
# - Outlier count after full pipeline
#
# These plots help ensure the dataset quality before modeling.
print("\nStep 5: Running quality checks...")

# 1. Magnetic signal plot
plt.figure(figsize=(14,4))
plt.plot(df["B_mag"])
plt.title("Cleaned |B| Signal")
plt.savefig("/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/MAG_L2_Feature_Engineering_1_cleaned_signal.png")
plt.close()

# 2. Rolling noise test
df["roll_std"] = df["B_mag"].rolling(50).std()
plt.figure(figsize=(14,4))
plt.plot(df["roll_std"])
plt.title("Rolling Std - Noise Test")
plt.savefig("/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/MAG_L2_Feature_Engineering_2_noise_test.png")
plt.close()

# 3. Autocorrelation plot — checks periodicity and correlation
plt.figure(figsize=(12,3))
autocorrelation_plot(df["B_mag"])
plt.savefig("/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/MAG_L2_Feature_Engineering_3_autocorrelation.png")
plt.close()

# 4. Compute Signal-to-Noise Ratio (SNR)
signal_power = np.mean(df["B_mag"]**2)
noise_power = np.var(df["B_mag"] - df["B_mag"].rolling(5).mean())
snr = 10 * np.log10(signal_power / (noise_power + 1e-12))
print(f"SNR (dB): {snr}")

# 5. Outlier check — should be minimal after cleaning
p99 = df["B_mag"].quantile(0.99)
outliers = df[df["B_mag"] > p99]
print(f"Remaining outliers: {len(outliers)}")

# ============================================
# STEP 6: SCALING
# ============================================
# StandardScaler gives zero-mean, unit-variance features.
# Essential for:
# - Gradient-based ML models
# - Neural networks
# - Distance-based models (SVM, KNN)
#
# Only scale raw physical components and derivatives,
# NOT rolling means, logs, etc., unless explicitly intended.
print("\nStep 6: Scaling features...")

scaler = StandardScaler()
scale_cols = ["Bx_clean","By_clean","Bz_clean","B_mag","dB_dt"]
df[scale_cols] = scaler.fit_transform(df[scale_cols])

print("Scaling complete.")

# ============================================
# STEP 7: TRAIN-TEST SPLIT
# ============================================
# Time-series SPLIT (no shuffle!) to prevent data leakage.
#
# Features = everything except time + B_mag (because B_mag is target)
# Target   = scaled B_mag (magnetic field magnitude)
#
# 80/20 split is commonly used for sequential models.
print("\nStep 7: Creating train-test split...")

# drop time column
X = df.drop(columns=["time", "B_mag"])  # features
y = df["B_mag"]                         # target

# time-series split (NO SHUFFLE!)
split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:",  X_test.shape, y_test.shape)

# ============================================
# STEP 8: SAVE FINAL DATASET
# ============================================
# Save the complete ML-ready dataset after all cleaning,
# feature creation, scaling, and splitting.
# This file will be used for model training.
print("\nStep 8: Saving final dataset...")

df.to_csv("/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/data/MAG_L2_model_dataset.csv", index=False)
print("Saved final ML-ready dataset: MAG_L2_FINAL_READY_FOR_MODEL.csv")

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE!")
print("=" * 60)
print(f"\nFinal dataset shape: {df.shape}")
print(f"Features: {X.columns.tolist()}")
print("\nReady for modeling!")
