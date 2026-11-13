#!/bin/bash

# Exit immediately if any command fails
set -e

# Project root directory (directory containing this shell script)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

printf "Project directory: %s\n" "$PROJECT_DIR"
PYTHON=$(which python3)

echo "======================================"
echo "Starting 47mT Reconstruction Pipeline"
echo "======================================"

DATA_FOLDER="data_r21/BT_0002_47mT_s1"
RECON_OUTPUTS="47mT_recon"

# Step 1: Convert raw axial orientation data to NPY
SUBFOLDER="3DTSE/1"
NPY_NAME="axial_circshift_1zy.npy"
NIFTI_NAME="axial_circshift_1zy.nii.gz"

echo "[1/3] Running raw data preparation..."

python3 "$PROJECT_DIR/read_raw_roll2npy_axial.py" \
    --data_folder "$DATA_FOLDER" \
    --recon_outputs "$RECON_OUTPUTS" \
    --subfolder "$SUBFOLDER" \
    --save_npy_name "$NPY_NAME" \
    --base_name "$NIFTI_NAME"

# # Step 2: Convert raw coronal orientation data to NPY

SUBFOLDER="3DTSE/2"
NPY_NAME="sagittal_circshift_1yx.npy"
NIFTI_NAME="sagittal_circshift_1yx.nii.gz"

# echo "[2/3] Running raw data preparation..."
python3 "$PROJECT_DIR/read_raw_roll2npy_axial.py" \
    --data_folder "$DATA_FOLDER" \
    --recon_outputs "$RECON_OUTPUTS" \
    --subfolder "$SUBFOLDER" \
    --save_npy_name "$NPY_NAME" \
    --base_name "$NIFTI_NAME"

# # Step 3: Convert raw saggital orientation data to NPY
SUBFOLDER="3DTSE/3"
NPY_NAME="coronal_circshift_1zx.npy"
NIFTI_NAME="coronal_circshift_1zx.nii.gz"

# echo "[3/3] Running raw data preparation..."
python3 "$PROJECT_DIR/read_raw_roll2npy_axial.py" \
    --data_folder "$DATA_FOLDER" \
    --recon_outputs "$RECON_OUTPUTS" \
    --subfolder "$SUBFOLDER" \
    --save_npy_name "$NPY_NAME" \
    --base_name "$NIFTI_NAME"

# Step 4,5,6: NiftyMIC, ZSSR, and SuperSynth
RECON_BASE="$DATA_FOLDER/$RECON_OUTPUTS/npy"
NIFTYMIC_DIR="$RECON_BASE/native_niftymic_inputs_thr40"
OUTPUT_BASE="$RECON_BASE/super_synth"

# Step 4: Prepare data for NiftyMIC
python3 "$PROJECT_DIR/prep4niftymic_no_upsamp_iterator_xxx.py" \
    --base_dir "$RECON_BASE"

# # Step 5: Run ZSSR reconstruction
File_NAME="srr_native_thr40_1cycle_no_intensity_2mm.nii.gz"
python3 "$PROJECT_DIR/zssr.py" \
    --base_dir "$NIFTYMIC_DIR" \
    --file_name "$File_NAME"

# Step 6: Brain parcellation using SuperSynth
bash "$PROJECT_DIR/scripts/mri_super_synth" \
    "$NIFTYMIC_DIR" \
    "$OUTPUT_BASE"