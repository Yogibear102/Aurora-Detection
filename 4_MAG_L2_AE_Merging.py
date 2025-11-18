import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. LOAD MAG DATA
# -----------------------------
# Read magnetometer dataset (MAG) which contains magnetic field measurements
mag_data = pd.read_csv('/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/data/MAG_L2_model_dataset.csv')

# Convert 'time' column to datetime format
mag_data['datetime'] = pd.to_datetime(mag_data['time'])

# Sort data by datetime and reset index
mag_data = mag_data.sort_values('datetime').reset_index(drop=True)

print(f"MAG data loaded: {len(mag_data)} records")

# -----------------------------
# 2. LOAD AE DATA
# -----------------------------
# AE dataset represents auroral electrojet index (target variable)
ae_data = pd.read_csv(
    '/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/data/AE dataset/AE dataset.txt',
    sep='\s+',  # whitespace-separated
    header=None  # no header row in the file
)

# Assign column names
ae_data.columns = ['year','month','day','HHMM','AE1','AE2','AE3','AE4']

# Convert HHMM to hour and minute
ae_data['hour'] = ae_data['HHMM'] // 100
ae_data['minute'] = ae_data['HHMM'] % 100

# Average multiple AE indices into one
ae_data['AE'] = ae_data[['AE1','AE2','AE3','AE4']].mean(axis=1)

# Create datetime column
ae_data['datetime'] = pd.to_datetime(ae_data[['year','month','day','hour','minute']])

# Sort AE data
ae_data = ae_data.sort_values('datetime').reset_index(drop=True)

print(f"AE data loaded: {len(ae_data)} records")

# -----------------------------
# 3. ALIGN MAG AND AE DATA
# -----------------------------
# Align MAG and AE datasets by nearest timestamp
merged = pd.merge_asof(
    mag_data.sort_values('datetime'),
    ae_data[['datetime','AE']].sort_values('datetime'),
    on='datetime'
)

# Drop rows where AE could not be matched
merged = merged.dropna(subset=['AE'])
print(f"Aligned dataset: {len(merged)} records")

# -----------------------------
# 4. CREATE BINARY AURORA LABEL
# -----------------------------
# Manually pick a threshold to guarantee some Active labels
# AE values in your dataset range roughly 10.75 - 48.5
threshold = 20  # You can adjust this to control the Active/Quiet ratio

# Create binary label: 1 = Active aurora, 0 = Quiet aurora
merged['aurora_label'] = (merged['AE'] > threshold).astype(int)

# Print distribution
print("\nAurora class distribution:")
print(merged['aurora_label'].value_counts())

# -----------------------------
# 5. SELECT FEATURES
# -----------------------------
# Use all MAG dataset features except 'time' and 'datetime' for classification
feature_cols = [col for col in merged.columns if col not in ['time','datetime','aurora_label','AE']]

X = merged[feature_cols]
y = merged['aurora_label']

print(f"\nUsing {len(feature_cols)} features for classification.")

# -----------------------------
# 6. TRAIN/TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Train Active/Quiet ratio: {y_train.mean():.3f}")
print(f"Test Active/Quiet ratio: {y_test.mean():.3f}")

# -----------------------------
# 7. SAVE DATASETS
# -----------------------------
merged.to_csv('/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/data/Aurora_Classification_dataset.csv', index=False)

train_data = merged.loc[X_train.index]
test_data = merged.loc[X_test.index]

train_data.to_csv('/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/data/Aurora_Train.csv', index=False)
test_data.to_csv('/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/data/Aurora_Test.csv', index=False)

print("\n✓ Data preparation complete!")
print("Files created:")
print("  - Aurora_Classification_dataset.csv (full dataset)")
print("  - Aurora_Train.csv (training set)")
print("  - Aurora_Test.csv (test set)")
