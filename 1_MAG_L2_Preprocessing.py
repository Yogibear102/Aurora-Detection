
# This script is a complete 3-stage processing pipeline for 
# Aditya-L1 MAG Level-2 magnetometer data stored in .nc (NetCDF/HDF5) files.
#
# It performs:
#
# ---------------------------------------------------------------
# (1) RAW EXTRACTION  
#     - Reads all ".nc" files inside the input folder.
#     - Filters only files containing "L2" in their name.
#     - Recursively extracts every dataset inside each .nc file 
#       (all nested HDF5 groups).
#     - Flattens array datasets into 1-D vectors.
#     - Pads variables with different lengths using NaN so all rows align.
#     - Combines all files into a single table.
#     - Saves the merged output as:
#
#         MAG_L2_raw_combined.csv
#
# ---------------------------------------------------------------
# (2) FEATURE ENGINEERING  
#     - Detects the time column (tries multiple common names).
#     - Converts time to a proper pandas datetime index.
#     - Identifies Bx, By, Bz magnetic field components 
#       (supports multiple naming conventions: *_gse, *_gsm, etc.).
#
#     - Converts all numeric columns from strings → floats.
#
#     - Uses PCHIP interpolation (shape-preserving cubic spline) 
#       to fill gaps in key MAG columns:
#         Bx, By, Bz, and coordinate variants
#
#     - Reconstructs canonical Bx, By, Bz even if names differ in input.
#
#     Computes derived MAG features:
#       • B_mag           = |B| = sqrt(Bx² + By² + Bz²)
#       • dB/dt           = rate of change of |B|
#       • Rolling features:
#             mean/std over 5min and 10min windows
#       • Lag features:
#             B_mag_lag_1,2,3
#             dB_dt_lag_1,2,3
#
#     (Optional)
#     - Computes solar wind dynamic pressure if density & velocity exist.
#
#     - Saves feature table as:
#
#         MAG_L2_features.csv
#
#
#     **Stage-1 scaling:**
#       - StandardScaler applied only to numeric columns.
#       - Saves scaled file:
#
#           MAG_L2_features_scaled.csv
#
#       - Saves the scaler object:
#           mag_scaler_stage1.joblib
#
# ---------------------------------------------------------------
# (3) FINAL CLEANUP + ML-READY DATA  
#     - Log-transforms selected MAG features:
#           B_mag
#           dB_dt
#           rolling means / rolling stds
#       using log10(abs(x) + epsilon)
#
#     - Removes extreme outliers above the 99.9th percentile.
#       (Keeps NaN values untouched)
#
#     - Applies a *second* StandardScaler to all numeric columns.
#       Saves scaler:
#           mag_scaler_final.joblib
#
#     - Produces final ML-ready dataset:
#
#         MAG_L2_final_features.csv
#
# ---------------------------------------------------------------
# TECHNICAL NOTES:
#   • Uses PCHIP interpolation (preferred for magnetometer signals)
#   • Automatically detects time units (ns/ms/s Unix epoch)
#   • Handles different coordinate systems (GSE/GSM)
#   • Keeps source_file column for provenance
#   • Gracefully handles missing / inconsistent datasets
#
# ---------------------------------------------------------------
# FINAL OUTPUT FILES CREATED:
#
#   1. MAG_L2_raw_combined.csv        → Raw extracted .nc files merged
#   2. MAG_L2_features.csv            → Feature engineering results
#   3. MAG_L2_features_scaled.csv     → Stage-1 normalized version
#   4. MAG_L2_final_features.csv      → Final cleaned + log-scaled ML dataset
#
#   Scalers:
#       • mag_scaler_stage1.joblib
#       • mag_scaler_final.joblib
#
# ---------------------------------------------------------------
# MAIN PURPOSE:
#   To generate a clean, consistent, ready-to-train dataset for
#   machine learning on geomagnetic disturbances / auroral prediction.
# ================================================================


import os
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy.interpolate import PchipInterpolator
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# USER CONFIGURATION
# -------------------------
INPUT_FOLDER = "/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/data/MAG  dataset (01-11 November)"
OUTPUT_DIR = "/Users/dhrithikiran/Downloads/Auroral_Prediction_Project/final/data"

# Output filenames you requested
RAW_COMBINED = os.path.join(OUTPUT_DIR, "MAG_L2_raw_combined.csv")
FEATURES_CSV = os.path.join(OUTPUT_DIR, "MAG_L2_features.csv")
SCALED_CSV = os.path.join(OUTPUT_DIR, "MAG_L2_features_scaled.csv")
FINAL_CSV = os.path.join(OUTPUT_DIR, "MAG_L2_final_features.csv")

# Scalers saved for reproducibility
SCALER_STAGE1 = os.path.join(OUTPUT_DIR, "mag_scaler_stage1.joblib")
SCALER_FINAL  = os.path.join(OUTPUT_DIR, "mag_scaler_final.joblib")

# Processing settings
ROLL_WINDOWS = ["5min", "10min"]
LAG_STEPS = [1, 2, 3]
MIN_POINTS_FOR_INTERP = 3
EPS = 1e-6  # epsilon for log transforms

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# HELPERS: extract .nc contents
# -------------------------
def extract_from_group(group, prefix=""):
    """
    Recursively extract all datasets in an HDF5 group.
    Keys are full paths like "group/subgroup/dataset".
    Values are 1D numpy arrays (flattened).
    """
    data = {}
    for key in group.keys():
        item = group[key]
        full_name = f"{prefix}/{key}".strip("/")
        if isinstance(item, h5py.Dataset):
            try:
                arr = item[()]
                arr = np.array(arr).flatten()
                data[full_name] = arr
            except Exception:
                # unreadable dataset; skip
                pass
        elif isinstance(item, h5py.Group):
            nested = extract_from_group(item, full_name)
            data.update(nested)
    return data

def combine_l2_nc_files(input_folder, output_csv):
    """
    Read all .nc files in input_folder with 'L2' in their name,
    extract datasets recursively, pad to equal lengths and combine
    into a single CSV saved to output_csv.
    """
    all_frames = []
    processed = 0
    print("Scanning folder:", input_folder)
    for fname in sorted(os.listdir(input_folder)):
        if not fname.endswith(".nc"):
            continue
        if "L2" not in fname:
            continue
        path = os.path.join(input_folder, fname)
        print("Processing:", fname)
        try:
            with h5py.File(path, "r") as nc:
                extracted = extract_from_group(nc)
            if not extracted:
                print("  -> No readable datasets found, skipping.")
                continue
            max_len = max(len(v) for v in extracted.values())
            # pad arrays with NaN to max_len
            for k in list(extracted.keys()):
                arr = extracted[k]
                if len(arr) < max_len:
                    extracted[k] = np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
            df = pd.DataFrame(extracted)
            df["source_file"] = fname
            all_frames.append(df)
            processed += 1
        except Exception as e:
            print(f"  Error reading {fname}: {e}")
    if not all_frames:
        raise RuntimeError("No L2 .nc files found or readable in folder.")
    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(output_csv, index=False)
    print(f"Saved combined CSV ({processed} files) -> {output_csv}")
    return output_csv

# -------------------------
# HELPERS: feature engineering pipeline
# -------------------------
def infer_time_column(df):
    """
    Find a plausible time column from common names or numeric epoch columns.
    Returns column name or raises ValueError.
    """
    candidates = ['time', 'utc_time', 'TIME', 'Time', 'timestamp', 'epoch', 'epoch_for_cdf_mod', 'time_sec']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: choose numeric column with large values
    for c in df.columns:
        try:
            sample = df[c].dropna().astype(str).head(20).tolist()
            # check if sample looks numeric
            if all(s.replace('.', '', 1).replace('-', '', 1).isdigit() for s in sample if s):
                # choose if values look large enough to be epoch
                vals = pd.to_numeric(df[c], errors="coerce").dropna()
                if not vals.empty and vals.abs().max() > 1e6:
                    return c
        except Exception:
            continue
    raise ValueError("Could not find a time column automatically. Provide one of: " + ", ".join(candidates))

def parse_time_column(df, time_col):
    """
    Convert time column to pandas datetime using heuristics for units.
    """
    s = pd.to_numeric(df[time_col], errors="coerce")
    if s.dropna().empty:
        # try direct parse
        return pd.to_datetime(df[time_col], errors="coerce")
    max_val = s.dropna().abs().max()
    # decide unit
    if max_val > 1e16:
        unit = "ns"
    elif max_val > 1e12:
        unit = "ns"
    elif max_val > 1e9:
        unit = "ms"
    elif max_val > 1e6:
        unit = "s"
    else:
        # fallback to pandas auto-parse
        try:
            return pd.to_datetime(df[time_col], errors="coerce")
        except Exception:
            unit = "s"
    return pd.to_datetime(s, unit=unit, origin='unix', errors='coerce')

def ensure_B_columns(df):
    """
    Detect likely Bx, By, Bz column names among common variants.
    Returns dict with keys 'Bx','By','Bz' -> column name or None.
    """
    candidates_map = {
        'Bx': ['Bx', 'Bx_gse', 'Bx_gsm', 'B_x', 'bx', 'BX'],
        'By': ['By', 'By_gse', 'By_gsm', 'B_y', 'by', 'BY'],
        'Bz': ['Bz', 'Bz_gse', 'Bz_gsm', 'B_z', 'bz', 'BZ'],
    }
    found = {}
    for k, cand in candidates_map.items():
        hit = None
        for c in cand:
            if c in df.columns:
                hit = c
                break
        found[k] = hit
    return found

def pchip_interpolate_series(time_numeric, y, target_time_numeric):
    """
    PCHIP interpolation with basic safeguards.
    Returns interpolated array aligned with target_time_numeric.
    """
    mask = np.isfinite(y) & np.isfinite(time_numeric)
    if mask.sum() < MIN_POINTS_FOR_INTERP:
        return np.full_like(target_time_numeric, np.nan, dtype=float)
    try:
        f = PchipInterpolator(time_numeric[mask], y[mask], extrapolate=False)
        return f(target_time_numeric)
    except Exception:
        # fallback to numpy.interp (linear)
        try:
            return np.interp(target_time_numeric, time_numeric[mask], y[mask], left=np.nan, right=np.nan)
        except Exception:
            return np.full_like(target_time_numeric, np.nan, dtype=float)

def run_feature_pipeline(input_csv, output_features_csv, scaler_path=None):
    """
    Load combined CSV, parse time, interpolate core columns with PCHIP,
    compute features (B_mag, dB_dt, rolling stats, lag features),
    save intermediate features CSV and return dataframe.
    """
    print("Loading combined CSV:", input_csv)
    raw = pd.read_csv(input_csv, dtype=str)  # read as string avoid surprises

    # detect and parse time
    time_col = infer_time_column(raw)
    print("Detected time column:", time_col)
    raw[time_col] = raw[time_col].replace('', np.nan)
    dt = parse_time_column(raw, time_col)
    raw['__datetime'] = dt
    raw = raw.dropna(subset=['__datetime']).copy()
    raw = raw.sort_values('__datetime').reset_index(drop=True)
    raw.index = pd.DatetimeIndex(raw['__datetime'])

    # find B columns
    b_map = ensure_B_columns(raw)
    print("Detected B columns mapping:", b_map)

    # coerce numeric columns
    to_num = [c for c in raw.columns if c != '__datetime']
    for c in to_num:
        try:
            raw[c] = pd.to_numeric(raw[c], errors='coerce')
        except Exception:
            raw[c] = raw[c]

    # prepare interpolation
    df_work = raw.copy()
    time_numeric = df_work.index.view(np.int64) / 1e9
    target_time_numeric = time_numeric

    primary_cols = [v for v in b_map.values() if v] + [c for c in ['x_gse','y_gse','z_gse','x_gsm','y_gsm','z_gsm'] if c in df_work.columns]
    primary_cols = list(dict.fromkeys(primary_cols))

    print("Columns considered for interpolation:", primary_cols)

    out = pd.DataFrame(index=df_work.index)
    out['time'] = df_work['__datetime']

    for col in primary_cols:
        if col in df_work.columns:
            y = df_work[col].to_numpy(dtype=float)
            out[col] = pchip_interpolate_series(time_numeric, y, target_time_numeric)

    # if canonical names missing, copy from raw if present
    for col in ['Bx','By','Bz','Bx_gse','By_gse','Bz_gse','Bx_gsm','By_gsm','Bz_gsm']:
        if col not in out.columns and col in df_work.columns:
            out[col] = df_work[col].astype(float)

    # pick canonical columns (prefer *_gse then *_gsm then plain)
    def select_first_available(choices):
        for c in choices:
            if c in out.columns and out[c].notna().any():
                return out[c].astype(float)
        # if none found return NaN series
        return pd.Series(np.nan, index=out.index)

    out['Bx'] = select_first_available(['Bx_gse','Bx_gsm','Bx', b_map.get('Bx')])
    out['By'] = select_first_available(['By_gse','By_gsm','By', b_map.get('By')])
    out['Bz'] = select_first_available(['Bz_gse','Bz_gsm','Bz', b_map.get('Bz')])

    # compute |B|
    out['B_mag'] = np.sqrt(out['Bx']**2 + out['By']**2 + out['Bz']**2)

    # compute dB/dt (use time delta in seconds)
    dt_seconds = out.index.to_series().diff().dt.total_seconds().fillna(0).to_numpy()
    dB = out['B_mag'].diff().to_numpy()
    dt_nonzero = dt_seconds.copy()
    dt_nonzero[dt_nonzero == 0] = np.nan
    out['dB_dt'] = dB / dt_nonzero

    # rolling statistics
    for rw in ROLL_WINDOWS:
        out[f'B_mag_roll_mean_{rw}'] = out['B_mag'].rolling(rw, min_periods=1).mean()
        out[f'B_mag_roll_std_{rw}'] = out['B_mag'].rolling(rw, min_periods=1).std().fillna(0)

    # lag features (by rows)
    for lag in LAG_STEPS:
        out[f'B_mag_lag_{lag}'] = out['B_mag'].shift(lag)
        out[f'dB_dt_lag_{lag}'] = out['dB_dt'].shift(lag)

    # try to compute solar wind pressure if density & velocity exist (optional)
    density_col = None
    velocity_col = None
    for candidate in ['proton_density','numden_p','proton_number_density','n']:
        if candidate in df_work.columns:
            density_col = candidate
            break
    for candidate in ['proton_velocity','proton_bulk_speed','proton_xvelocity','V']:
        if candidate in df_work.columns:
            velocity_col = candidate
            break
    if density_col and velocity_col:
        out['proton_density'] = df_work[density_col].astype(float)
        out['proton_velocity'] = df_work[velocity_col].astype(float)
        out['solar_wind_pressure'] = out['proton_density'] * (out['proton_velocity'] ** 2)
    else:
        if density_col:
            out['proton_density'] = df_work[density_col].astype(float)
        if velocity_col:
            out['proton_velocity'] = df_work[velocity_col].astype(float)

    if 'source_file' in df_work.columns:
        out['source_file'] = df_work['source_file']

    # re-order columns: time, Bx, By, Bz, B_mag, dB_dt, rest...
    core = ['time','Bx','By','Bz','B_mag','dB_dt']
    rest = [c for c in out.columns if c not in core]
    out = out[core + rest]

    # drop rows where core measurements entirely missing (optional keep if any present)
    core_present = out[['Bx','By','Bz','B_mag']].notna().any(axis=1)
    if core_present.sum() == 0:
        print("Warning: no B-vector measurements found after interpolation. File will still be saved but features may be NaN.")

    # save unscaled features
    out.to_csv(output_features_csv, index=False)
    print("Saved features CSV ->", output_features_csv)

    # Stage 1 scaling: standardize numeric columns (save scaler)
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    scaler = None
    stage1_output = SCALED_CSV
    if numeric_cols:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(out[numeric_cols].fillna(0.0))
        scaled_df = pd.DataFrame(scaled, columns=numeric_cols, index=out.index)
        final_stage1 = pd.concat([out.drop(columns=numeric_cols), scaled_df], axis=1)
        final_stage1.to_csv(stage1_output, index=False)
        if scaler_path:
            joblib.dump(scaler, scaler_path)
            print("Saved stage-1 scaler ->", scaler_path)
        print("Saved stage-1 scaled CSV ->", stage1_output)
        return out, final_stage1, scaler
    else:
        out.to_csv(stage1_output, index=False)
        print("No numeric columns detected; saved features only.")
        return out, out.copy(), None

# -------------------------
# FINAL CLEANING: log + outlier removal + final scaling
# -------------------------
def final_cleanup(stage1_df, final_csv, final_scaler_path):
    """
    Apply log10 transforms to selected columns, remove extreme outliers,
    re-scale numeric columns and save final CSV + scaler.
    stage1_df should be the dataframe produced after stage 1 (unscaled).
    """
    df = stage1_df.copy()

    # columns we will log-transform if present
    log_features = [
        "B_mag",
        "dB_dt",
        "B_mag_roll_mean_5min",
        "B_mag_roll_std_5min",
        "B_mag_roll_mean_10min",
        "B_mag_roll_std_10min",
    ]

    # add log10 columns (abs + eps to avoid negative/zero issues)
    for col in log_features:
        if col in df.columns:
            df[col + "_log10"] = np.log10(df[col].abs().fillna(0.0) + EPS)

    # remove extreme outliers (>99.9 percentile) for numeric columns that exist
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        for col in numeric_cols:
            upper = df[col].quantile(0.999)
            # keep rows <= upper or NaN (NaNs are preserved)
            df = df[(df[col].isna()) | (df[col] <= upper)]

    # final scaling
    numeric_cols_after = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler_final = None
    if numeric_cols_after:
        scaler_final = StandardScaler()
        df[numeric_cols_after] = scaler_final.fit_transform(df[numeric_cols_after].fillna(0.0))
        joblib.dump(scaler_final, final_scaler_path)
        print("Saved final scaler ->", final_scaler_path)

    # save final ML-ready CSV
    df.to_csv(final_csv, index=False)
    print("Saved final ML-ready CSV ->", final_csv)
    return df, scaler_final

# -------------------------
# MAIN ENTRY
# -------------------------
def main():
    # 1) Combine L2 .nc files -> RAW_COMBINED
    try:
        combine_l2_nc_files(INPUT_FOLDER, RAW_COMBINED)
    except Exception as e:
        print("Error during L2 extraction:", e)
        return

    # 2) Run feature engineering -> FEATURES_CSV and SCALED_CSV (stage1)
    try:
        features_unscaled, features_stage1_df, scaler_stage1 = run_feature_pipeline(RAW_COMBINED, FEATURES_CSV, SCALER_STAGE1)
    except Exception as e:
        print("Error during feature engineering:", e)
        return

    # 3) Final cleanup: log transforms, outlier removal, final scaling -> FINAL_CSV
    try:
        # final_cleanup expects unscaled features (features_unscaled)
        final_df, scaler_final = final_cleanup(features_unscaled, FINAL_CSV, SCALER_FINAL)
    except Exception as e:
        print("Error during final cleanup:", e)
        return

    # Print short summary
    print("\nPipeline finished successfully.")
    print("Files produced:")
    print(" - Combined raw CSV:    ", RAW_COMBINED)
    print(" - Feature CSV:         ", FEATURES_CSV)
    print(" - Stage-1 scaled CSV:  ", SCALED_CSV)
    print(" - Final ML-ready CSV:  ", FINAL_CSV)
    if scaler_stage1:
        print(" - Stage-1 scaler:      ", SCALER_STAGE1)
    if scaler_final:
        print(" - Final scaler:        ", SCALER_FINAL)
    print("\nFinal dataset preview (first 5 rows):")
    print(final_df.head())

if __name__ == "__main__":
    main()
