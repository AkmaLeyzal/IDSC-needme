# ============================================================
# Brugada Syndrome ECG Dataset - Data Loading Script
# ============================================================
# Dataset: 363 subjects, 12-lead ECG recordings at 100Hz
# Format: WFDB (.dat/.hea files)
# ============================================================

import os
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt

# ============================================================
# 1. SET BASE PATH
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "brugada-syndrome-dataset")

# ============================================================
# 2. LOAD METADATA
# ============================================================
print("=" * 60)
print("LOADING METADATA")
print("=" * 60)

metadata = pd.read_csv(os.path.join(DATASET_DIR, "metadata.csv"))
data_dict = pd.read_csv(os.path.join(DATASET_DIR, "metadata_dictionary.csv"))

print(f"\nTotal subjects: {len(metadata)}")
print(f"\nMetadata columns: {list(metadata.columns)}")
print(f"\nData Dictionary:\n{data_dict}")

# Subject distribution
print(f"\n--- Subject Distribution ---")
print(f"Healthy (brugada=0):   {(metadata['brugada'] == 0).sum()}")
print(f"Brugada (brugada=1):   {(metadata['brugada'] == 1).sum()}")
print(f"Other   (brugada=2):   {(metadata['brugada'] == 2).sum()}")
print(f"Sudden death cases:    {(metadata['sudden_death'] == 1).sum()}")
print(f"Basal pattern cases:   {(metadata['basal_pattern'] == 1).sum()}")

print(f"\nMetadata head:\n{metadata.head(10)}")

# ============================================================
# 3. LOAD ALL ECG SIGNALS
# ============================================================
print("\n" + "=" * 60)
print("LOADING ALL ECG SIGNALS")
print("=" * 60)

files_dir = os.path.join(DATASET_DIR, "files")
patient_ids = metadata["patient_id"].values

all_signals = []  # Will store all ECG signals
all_patient_ids = []  # Corresponding patient IDs
failed_patients = []  # Track any loading failures
lead_names = None  # Will be populated from first record

for i, pid in enumerate(patient_ids):
    record_path = os.path.join(files_dir, str(pid), str(pid))
    try:
        record = wfdb.rdrecord(record_path)
        signals = record.p_signal  # Shape: (1200, 12) for 12s at 100Hz
        all_signals.append(signals)
        all_patient_ids.append(pid)
        
        if lead_names is None:
            lead_names = record.sig_name
            sampling_freq = record.fs
            
        if (i + 1) % 50 == 0 or (i + 1) == len(patient_ids):
            print(f"  Loaded {i + 1}/{len(patient_ids)} records...")
            
    except Exception as e:
        failed_patients.append((pid, str(e)))
        print(f"  WARNING: Failed to load patient {pid}: {e}")

# Convert to numpy array
all_signals = np.array(all_signals)  # Shape: (n_patients, 1200, 12)

print(f"\n--- ECG Data Summary ---")
print(f"Total records loaded: {len(all_signals)}")
print(f"Signal shape: {all_signals.shape}")
print(f"Lead names: {lead_names}")
print(f"Sampling frequency: {sampling_freq} Hz")
print(f"Recording duration: {all_signals.shape[1] / sampling_freq} seconds")
print(f"Failed to load: {len(failed_patients)}")

if failed_patients:
    print(f"\nFailed patients:")
    for pid, err in failed_patients:
        print(f"  - {pid}: {err}")

# ============================================================
# 4. CREATE COMBINED DATAFRAME (metadata + signal stats)
# ============================================================
print("\n" + "=" * 60)
print("CREATING COMBINED DATASET")
print("=" * 60)

# Calculate signal statistics for each patient
signal_stats = []
for i, pid in enumerate(all_patient_ids):
    stats = {
        "patient_id": pid,
        "signal_mean": np.mean(all_signals[i]),
        "signal_std": np.std(all_signals[i]),
        "signal_min": np.min(all_signals[i]),
        "signal_max": np.max(all_signals[i]),
    }
    # Per-lead statistics
    for j, lead in enumerate(lead_names):
        stats[f"{lead}_mean"] = np.mean(all_signals[i, :, j])
        stats[f"{lead}_std"] = np.std(all_signals[i, :, j])
    signal_stats.append(stats)

signal_stats_df = pd.DataFrame(signal_stats)
combined_df = metadata.merge(signal_stats_df, on="patient_id", how="inner")

print(f"Combined dataset shape: {combined_df.shape}")
print(f"\nCombined dataset head:\n{combined_df.head()}")

# ============================================================
# 5. VISUALIZE SAMPLE ECG
# ============================================================
print("\n" + "=" * 60)
print("VISUALIZING SAMPLE ECG")
print("=" * 60)

# Plot one sample from each category
categories = {0: "Healthy", 1: "Brugada", 2: "Other"}

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
time_axis = np.arange(all_signals.shape[1]) / sampling_freq

for idx, (label, name) in enumerate(categories.items()):
    # Find first patient in this category
    cat_patients = metadata[metadata["brugada"] == label]["patient_id"].values
    if len(cat_patients) > 0:
        pid = cat_patients[0]
        pid_idx = all_patient_ids.index(pid)
        
        # Plot Lead II (commonly used for ECG analysis)
        lead_ii_idx = lead_names.index("II") if "II" in lead_names else 1
        axes[idx].plot(time_axis, all_signals[pid_idx, :, lead_ii_idx], linewidth=0.8)
        axes[idx].set_title(f"{name} (Patient {pid}) - Lead II", fontsize=12)
        axes[idx].set_ylabel("Amplitude (mV)")
        axes[idx].grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "brugada_sample_ecg.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Sample ECG plot saved to brugada_sample_ecg.png")

# ============================================================
# 6. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("DATA LOADING COMPLETE")
print("=" * 60)
print(f"""
Available variables:
  - metadata        : DataFrame with clinical labels ({metadata.shape})
  - data_dict       : DataFrame with variable descriptions
  - all_signals     : NumPy array of ECG signals ({all_signals.shape})
  - all_patient_ids : List of patient IDs (length {len(all_patient_ids)})
  - lead_names      : List of lead names ({lead_names})
  - sampling_freq   : Sampling frequency ({sampling_freq} Hz)
  - combined_df     : Merged metadata + signal stats ({combined_df.shape})
""")
