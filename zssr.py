# This file reproduces the figure in the R21 application

# Import necessary modules
import os.path
import sys
sys.path.insert(0, './')  # Adjust the path as necessary to import from src_niv
sys.path.append('./src')
sys.path.append('./local')
sys.path.append('./Niftymic_related_r21')
# sys.path.append('/Users/sairamgeethanath/Documents/Projects/Tools/Low_field/Propsa/Recon/Code2PP/')
sys.path.insert(0, '/Users/sairamgeethanath/Documents/Projects/Tools/Low_field/Propsa/Recon/Code2PP/')

# import cProcessPipeline as cPP
# from sim_input_SR import create_object, down_res

from display_vlf_ni_data import plot_anatomy_raw, plot_anatomy_nifti
# import niftyreg as nreg
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from preprocess4srr import non_local_means_denoising
from prep4srr_2step_v2 import do_resize, make_nifti, create_nifti_header, norm_data, pad_zeros
from src.ZSSR_master import configs, configs_2, ZSSR
from src.ZSSR_master.ZSSR import *
import subprocess

niftymic = False
zssr = True
visualize = False

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--base_dir",
    type=str,
    default="data_r21/BT_0002_47mT_s1/47mT_recon/npy/native_niftymic_inputs_thr40",
    help="Base directory for SRR outputs"
)

parser.add_argument(
    "--file_name",
    type=str,
    default="srr_alpha001_out1_ic0_cycle2_iso2_HuberL2.nii.gz",
    help="Input NIfTI filename"
)

args = parser.parse_args()

input_folder = args.base_dir
file_name = args.file_name

print("Base directory:", input_folder)

niftymic_output = os.path.join(input_folder, file_name)

# improve this remove nii.gz from file_name
file_name_no_ext = os.path.basename(file_name).removesuffix(".nii.gz")

zssr_fname = os.path.join(input_folder, f"{file_name_no_ext}_zssr.nii.gz")

# Output: SRR
if visualize:
    print('Loading NiftyMIC SRR output ..........')
    im_srr = nib.load(niftymic_output).get_fdata()
    print("Input shape:", im_srr.shape)
    
    # Rotate the volume by 20 degree

    plot_anatomy_raw(im_srr, clim=[0, 2048])

# -------------------------------------------------------------
# Load input LF/HF-SRR volume
# -------------------------------------------------------------

if zssr:

    print('Passing through ZSSR ..........')

    img = nib.load(niftymic_output)

    im_srr = img.get_fdata()
    print("Input shape:", im_srr.shape)

    ## Not required yet

    # img_new = im_srr
    # img_new = np.zeros((im_srr.shape[0] + 2, im_srr.shape[1] , im_srr.shape[2]))
    # img_new[:-2, :,:] = im_srr
    # plot_anatomy_raw(img_new, clim=[0, 2048])

    # # Pad only the second axis (Y axis)
    # img_pad = np.pad(
    #     im_srr,
    #     pad_width=((0, 0),   # axis 0: no padding
    #                (0, 1),   # axis 1: pad 1 voxel at the end
    #                (0, 0)),  # axis 2: no padding
    #     mode='constant',
    #     constant_values=0
    # )
    # print(img_pad.shape)
    # X, Y, Z = img_new.shape
    # print("Input shape:", img_new.shape)

    # affine = im_srr.affine

    # ============================================================
    # 0. Slice-based ZSSR Runner
    # ============================================================
    def run_zssr_slice(slice2d, recon_config):
        """Run ZSSR on a single 2D slice with normalization."""
        # minv, maxv = slice2d.min(), slice2d.max()
        # s = (slice2d - minv) / (maxv - minv + 1e-8)
        s = slice2d
        s3 = np.stack([s, s, s], axis=-1)

        out = ZSSR(
            input_img=s3,
            conf=recon_config,
            ground_truth=None,
            kernels=None
        ).run()
        # Undo the normalization before passing it back so that brain intensities are maintained
        return out


    # ============================================================
    # 1. PASS x — Upscale z only
    # ============================================================
    def upscale_x(volume, recon_config):
        X, Y, Z = volume.shape
        recon_config.scale_factors = [[1, 2]] 

        out = np.zeros((X, Y, Z*2), dtype=float)

        for x in range(X):
            print(f"[Upscale X] Slice {x+1}/{X}")
            slice_yz = volume[x, :, :]        # shape (Y, Z)
            if np.sum(slice_yz) == 0:
                out[x, :, :] = 0
            else:
                sr_slice = run_zssr_slice(slice_yz, recon_config)
                # replicate along the X dimension
                out[x, :, :] = sr_slice[:, :, 0]

        return out

    # ============================================================
    # 2. PASS y — Upscale x only
    # ============================================================
    def upscale_y(volume, recon_config):
        X, Y, Z = volume.shape
        recon_config.scale_factors = [[2, 1]]

        out = np.zeros((X*2, Y, Z), dtype=float)

        for y in range(Y):
            print(f"[Upscale Y] Slice {y+1}/{Y}")
            slice_xz = np.squeeze(volume[:, y, :])      # shape (X, Z)
            sr_slice = run_zssr_slice(slice_xz, recon_config)

            out[:, y, :] = sr_slice[:, :, 0]

        return out

    # ============================================================
    # 3. PASS z — Upscale y only
    # ============================================================
    def upscale_z(volume, recon_config):
        X, Y, Z = volume.shape
        recon_config.scale_factors = [[1, 2]]  

        out = np.zeros((X, Y*2, Z), dtype=float)

        for z in range(Z):
            print(f"[Upscale Z] Slice {z+1}/{Z}")
            slice_xy = volume[:, :, z]      # shape (X, Y)
            sr_slice = run_zssr_slice(slice_xy, recon_config)

            out[:, :, z] = sr_slice[:, :, 0]

        return out

    # ============================================================
    # 4. FINAL PIPELINE  — X → Y → Z
    # ============================================================
    def run_xyz_progressive_zssr(im_srr, recon_config):
        print("\n==============================")
        print("Step 1: Upscaling X (×2)")
        print("==============================")
        vol_x2 = upscale_x(im_srr, recon_config)

        print("\n==============================")
        print("Step 2: Upscaling Y (*2)")
        print("==============================")
        vol_xy2 = upscale_y(vol_x2, recon_config)

        print("\n==============================")
        print("Step 3: Upscaling Z (*2)")
        print("==============================")
        vol_xyz2 = upscale_z(vol_xy2, recon_config)

        print("\nFinal SR Shape:", vol_xyz2.shape)
        return vol_xyz2

    # ====================
    # 5. RUN FULL PIPELINE
    # ====================

    # change of recon.config
    recon_config = configs.Config()
    # recon_config.scale_factors = [[np.sqrt(target_resolution_fact[0]), 1]]
    recon_config.scale_factors = [[1, 2]]
    recon_config.max_iters = 3000
    recon_config.min_iters = 500
    recon_config.width = 32
    recon_config.depth = 12
    recon_config.noise_std = 0.0
    recon_config.crop_size = 32
    num_rows = 16
    num_cols = 14

    # Normalize im_srr with min-max normalization
    min_val, max_val = im_srr.min(), im_srr.max()
    im_srr = (im_srr - min_val) / (max_val - min_val + 1e-8)

    # Run the full progressive ZSSR pipeline

    vol_xz = run_xyz_progressive_zssr(im_srr, recon_config)

    # -------------------------------------------------------------
    # Final 3D SR output
    # -------------------------------------------------------------
    
    im_srr_3d_zssr = vol_xz
    print("\nFinal SR shape:", im_srr_3d_zssr.shape)
    #  Visualize the image and mask to confirm correctness
    # plot_anatomy_raw(im_srr_3d_zssr, clim=[0, 512])
    # -------------------------------------------------------------
    # Save NIfTI with corrected voxel spacing
    # (halving voxel spacing because resolution doubled)
    # -------------------------------------------------------------
    # new_affine = affine.copy()
    # new_affine[:3, :3] /= 2   # voxel spacing becomes half = ×2 SR
    
    make_nifti(data=im_srr_3d_zssr, fname=zssr_fname, mask=False,
                   res=[1, 1, 1], dim_info=[0, 1, 2])  # phase, freq, slice

    # out_img = nib.Nifti1Image(im_srr_3d_zssr.astype(np.float32))
    # nib.save(out_img, "srr_zssr_3D_full_1.nii.gz")

    print("\nSaved:", zssr_fname)

if visualize:
    print('Loading ZSSR SRR output ..........')
    img = nib.load(zssr_fname)

    im_srr = img.get_fdata()
    print("Input shape:", im_srr.shape)
    plot_anatomy_raw(im_srr, clim=[0, 2048])