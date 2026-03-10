# ============================================================
# Myocardial Perfusion Scintigraphy - Data Loading Script
# ============================================================
# Dataset: 83 patients, SPECT cardiac images
# Format: DICOM (103 files) + NIfTI masks (101 files)
# Resolution: 70x70 pixels, 50 slices, 4mm spacing
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import medical imaging libraries
try:
    import pydicom
    HAS_PYDICOM = True
    print("✓ pydicom loaded successfully")
except ImportError:
    HAS_PYDICOM = False
    print("✗ pydicom not installed. Install with: pip install pydicom")

try:
    import nibabel as nib
    HAS_NIBABEL = True
    print("✓ nibabel loaded successfully")
except ImportError:
    HAS_NIBABEL = False
    print("✗ nibabel not installed. Install with: pip install nibabel")

# ============================================================
# 1. SET BASE PATH
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "myocardial-perfusion-dataset")
DICOM_DIR = os.path.join(DATASET_DIR, "DICOM")
NIFTI_DIR = os.path.join(DATASET_DIR, "NIfTI")

# ============================================================
# 2. LOAD DEMOGRAPHICS / METADATA
# ============================================================
print("\n" + "=" * 60)
print("LOADING DEMOGRAPHICS & METADATA")
print("=" * 60)

demographics_path = os.path.join(DATASET_DIR, "demographics.csv")
demographics = pd.read_csv(demographics_path, sep=",", on_bad_lines="skip")

print(f"\nDemographics shape: {demographics.shape}")
print(f"Demographics columns: {list(demographics.columns)}")
print(f"\nDemographics head:\n{demographics.head(10)}")

# ============================================================
# 3. LOAD ALL DICOM FILES
# ============================================================
if HAS_PYDICOM:
    print("\n" + "=" * 60)
    print("LOADING ALL DICOM FILES")
    print("=" * 60)
    
    dicom_files = sorted(glob.glob(os.path.join(DICOM_DIR, "*.dcm")))
    print(f"Found {len(dicom_files)} DICOM files")
    
    all_dicom_data = []  # Will store pixel arrays
    all_dicom_metadata = []  # Will store key metadata
    failed_dicoms = []
    
    for i, dcm_path in enumerate(dicom_files):
        try:
            ds = pydicom.dcmread(dcm_path)
            pixel_data = ds.pixel_array  # Shape: (50, 70, 70) - 50 slices
            
            # Extract key metadata
            meta = {
                "filename": os.path.basename(dcm_path),
                "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "N/A")),
                "patient_id": str(getattr(ds, "PatientID", "N/A")),
                "patient_sex": str(getattr(ds, "PatientSex", "N/A")),
                "patient_birth_date": str(getattr(ds, "PatientBirthDate", "N/A")),
                "study_date": str(getattr(ds, "StudyDate", "N/A")),
                "modality": str(getattr(ds, "Modality", "N/A")),
                "rows": int(getattr(ds, "Rows", 0)),
                "columns": int(getattr(ds, "Columns", 0)),
                "num_frames": int(getattr(ds, "NumberOfFrames", 0)),
                "pixel_spacing": str(getattr(ds, "PixelSpacing", "N/A")),
                "slice_thickness": str(getattr(ds, "SliceThickness", "N/A")),
                "pixel_min": int(np.min(pixel_data)),
                "pixel_max": int(np.max(pixel_data)),
                "pixel_mean": float(np.mean(pixel_data)),
            }
            
            all_dicom_data.append(pixel_data)
            all_dicom_metadata.append(meta)
            
            if (i + 1) % 20 == 0 or (i + 1) == len(dicom_files):
                print(f"  Loaded {i + 1}/{len(dicom_files)} DICOM files...")
                
        except Exception as e:
            failed_dicoms.append((os.path.basename(dcm_path), str(e)))
            print(f"  WARNING: Failed to load {os.path.basename(dcm_path)}: {e}")
    
    # Convert to numpy array
    all_dicom_data = np.array(all_dicom_data)  # Shape: (103, 50, 70, 70)
    dicom_metadata_df = pd.DataFrame(all_dicom_metadata)
    
    print(f"\n--- DICOM Data Summary ---")
    print(f"Total DICOM files loaded: {len(all_dicom_data)}")
    print(f"Data shape: {all_dicom_data.shape}")
    print(f"Data dtype: {all_dicom_data.dtype}")
    print(f"Value range: [{np.min(all_dicom_data)}, {np.max(all_dicom_data)}]")
    print(f"Failed to load: {len(failed_dicoms)}")
    
    if failed_dicoms:
        print(f"\nFailed files:")
        for fname, err in failed_dicoms:
            print(f"  - {fname}: {err}")

else:
    print("\n⚠ Skipping DICOM loading (pydicom not installed)")
    all_dicom_data = None
    dicom_metadata_df = None

# ============================================================
# 4. LOAD ALL NIfTI SEGMENTATION MASKS
# ============================================================
if HAS_NIBABEL:
    print("\n" + "=" * 60)
    print("LOADING ALL NIfTI SEGMENTATION MASKS")
    print("=" * 60)
    
    nifti_files = sorted(glob.glob(os.path.join(NIFTI_DIR, "*_mask.nii.gz")) + 
                         glob.glob(os.path.join(NIFTI_DIR, "*_mask.gz")))
    print(f"Found {len(nifti_files)} NIfTI mask files")
    
    all_masks = []
    all_mask_metadata = []
    failed_masks = []
    
    for i, nii_path in enumerate(nifti_files):
        try:
            nii_img = nib.load(nii_path)
            mask_data = nii_img.get_fdata()
            
            # Extract the SOP Instance UID from filename
            basename = os.path.basename(nii_path)
            uid = basename.replace("_mask.nii.gz", "").replace("_mask.gz", "")
            
            meta = {
                "filename": basename,
                "uid": uid,
                "shape": mask_data.shape,
                "voxel_dims": nii_img.header.get_zooms(),
                "unique_labels": np.unique(mask_data).tolist(),
                "lv_wall_voxels": int(np.sum(mask_data == 1)),
                "total_voxels": int(np.prod(mask_data.shape)),
            }
            
            all_masks.append(mask_data)
            all_mask_metadata.append(meta)
            
            if (i + 1) % 20 == 0 or (i + 1) == len(nifti_files):
                print(f"  Loaded {i + 1}/{len(nifti_files)} NIfTI masks...")
                
        except Exception as e:
            failed_masks.append((os.path.basename(nii_path), str(e)))
            print(f"  WARNING: Failed to load {os.path.basename(nii_path)}: {e}")
    
    mask_metadata_df = pd.DataFrame(all_mask_metadata)
    
    print(f"\n--- NIfTI Mask Summary ---")
    print(f"Total masks loaded: {len(all_masks)}")
    if len(all_masks) > 0:
        print(f"Sample mask shape: {all_masks[0].shape}")
        print(f"Labels used: 0 (background), 1 (left ventricular wall)")
    print(f"Failed to load: {len(failed_masks)}")

else:
    print("\n⚠ Skipping NIfTI loading (nibabel not installed)")
    all_masks = None
    mask_metadata_df = None

# ============================================================
# 5. MATCH DICOM AND NIfTI FILES
# ============================================================
if HAS_PYDICOM and HAS_NIBABEL and dicom_metadata_df is not None and mask_metadata_df is not None:
    print("\n" + "=" * 60)
    print("MATCHING DICOM AND NIfTI FILES")
    print("=" * 60)
    
    # Extract UIDs from filenames for matching
    dicom_uids = set(dicom_metadata_df["sop_instance_uid"].values)
    mask_uids = set(mask_metadata_df["uid"].values)
    
    matched = dicom_uids & mask_uids
    dicom_only = dicom_uids - mask_uids
    mask_only = mask_uids - dicom_uids
    
    print(f"DICOM files: {len(dicom_uids)}")
    print(f"NIfTI masks: {len(mask_uids)}")
    print(f"Matched pairs: {len(matched)}")
    print(f"DICOM without mask: {len(dicom_only)}")
    print(f"Masks without DICOM: {len(mask_only)}")

# ============================================================
# 6. VISUALIZE SAMPLE IMAGES
# ============================================================
if HAS_PYDICOM and all_dicom_data is not None and len(all_dicom_data) > 0:
    print("\n" + "=" * 60)
    print("VISUALIZING SAMPLE IMAGES")
    print("=" * 60)
    
    # Show middle slices from first 4 patients
    n_samples = min(4, len(all_dicom_data))
    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))
    
    for i in range(n_samples):
        mid_slice = all_dicom_data.shape[1] // 2  # Middle slice
        
        # DICOM image
        axes[0, i].imshow(all_dicom_data[i, mid_slice], cmap="hot")
        axes[0, i].set_title(f"DICOM #{i+1}\n(slice {mid_slice})", fontsize=10)
        axes[0, i].axis("off")
        
        # NIfTI mask (if available)
        if HAS_NIBABEL and all_masks is not None and i < len(all_masks):
            mask_mid = all_masks[i].shape[2] // 2 if len(all_masks[i].shape) == 3 else 0
            if len(all_masks[i].shape) == 3:
                axes[1, i].imshow(all_masks[i][:, :, mask_mid], cmap="gray")
            else:
                axes[1, i].imshow(all_masks[i], cmap="gray")
            axes[1, i].set_title(f"Mask #{i+1}", fontsize=10)
            axes[1, i].axis("off")
        else:
            axes[1, i].text(0.5, 0.5, "No mask", ha="center", va="center", transform=axes[1, i].transAxes)
            axes[1, i].axis("off")
    
    axes[0, 0].set_ylabel("DICOM Image", fontsize=12)
    axes[1, 0].set_ylabel("Segmentation Mask", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "myocardial_sample_images.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("Sample images saved to myocardial_sample_images.png")

# ============================================================
# 7. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("DATA LOADING COMPLETE")
print("=" * 60)

summary_parts = [
    f"  - demographics     : DataFrame with DICOM metadata ({demographics.shape})",
]

if all_dicom_data is not None:
    summary_parts.extend([
        f"  - all_dicom_data   : NumPy array of SPECT images ({all_dicom_data.shape})",
        f"  - dicom_metadata_df: DataFrame with per-file metadata ({dicom_metadata_df.shape})",
    ])

if all_masks is not None:
    summary_parts.extend([
        f"  - all_masks        : List of NIfTI segmentation masks ({len(all_masks)} files)",
        f"  - mask_metadata_df : DataFrame with mask metadata ({mask_metadata_df.shape})",
    ])

print(f"""
Available variables:
{chr(10).join(summary_parts)}
""")
