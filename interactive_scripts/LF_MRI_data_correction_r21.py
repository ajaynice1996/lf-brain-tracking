import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.ndimage import rotate
from pathlib import Path
import os
from nibabel.viewers import OrthoSlicer3D
import matplotlib.gridspec as gridspec

# -----------------------------
# Load image
# -----------------------------
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog


# -----------------------------
# Helper: in-plane rotation
# -----------------------------
def rotate_inplane(volume, angle_deg):
    out = np.zeros_like(volume)
    for z in range(volume.shape[2]):
        out[:, :, z] = rotate(
            volume[:, :, z],
            angle=angle_deg,
            reshape=False,
            order=1,
            mode='constant',
            cval=0
        )
    return out

# -----------------------------
# File dialog
# -----------------------------
for i in range(70):
    root = tk.Tk()
    root.withdraw()  # hide the main window

    initial_dir = "DataSRR"
    filename = filedialog.askopenfilename(
        title="Select a NIfTI file",
        filetypes=[("NIfTI files", "*.nii *.nii.gz")],
        initialdir=initial_dir
    )

    if not filename:
        raise ValueError("No file selected!")

    print("Selected file:", filename)

    # -----------------------------
    # Extract folder name from selected file path
    # -----------------------------
    # Example: filename = "Data/Nipah_IRF_data/LFMRI_DATA_IRF_best/35528/IRF_071E_2_C1_20240710&35528_D_minus27&V01&3DTSE&4.nii.gz"
    folder_name = Path(filename).parent.name  # gets '35528'
    print("Folder name extracted:", folder_name)

    # -------------------------------
    # Set output directory and output filename
    # -------------------------------
    out_dir = f"DataSRR/volunteer_xxx/corrected"
    os.makedirs(out_dir, exist_ok=True)

    out_name = Path(filename).name  # just the file name
    out_path = os.path.join(out_dir, out_name)

    print("Output directory:", out_dir)
    print("Output file path:", out_path)


    os.makedirs(out_dir, exist_ok=True)
    out_name = Path(filename).name
    out_path = os.path.join(out_dir, out_name)

    nifti = nib.load(filename)
    im_orig = nifti.get_fdata()
    ny, nx, nz = im_orig.shape

    print(f"Input shape: {im_orig.shape}")
    print(f"Voxel size: {nifti.header.get_zooms()} mm")

    # -----------------------------
    # Determine 10 middle slices
    # -----------------------------
    n_display = 10

    # Axial slices: along axis 2
    nz = im_orig.shape[2]
    ax_start = max(0, (nz - n_display) // 2)
    ax_slices = np.arange(ax_start, min(ax_start + n_display, nz))

    # Coronal slices: along axis 1
    nx = im_orig.shape[1]
    cor_start = max(0, (nx - n_display) // 2)
    cor_slices = np.arange(cor_start, min(cor_start + n_display, nx))

    # Sagittal slices: along axis 0
    ny = im_orig.shape[0]
    sag_start = max(0, (ny - n_display) // 2)
    sag_slices = np.arange(sag_start, min(sag_start + n_display, ny))

    print("Axial slices:", ax_slices)
    print("Coronal slices:", cor_slices)
    print("Sagittal slices:", sag_slices)

    # -----------------------------
    # Figure with GridSpec
    # -----------------------------
    fig = plt.figure(figsize=(20, 10))

    gs = gridspec.GridSpec(
        3, n_display,
        figure=fig,
        hspace=0.2,
        wspace=0.05
    )

    # -----------------------------
    # Axial slices: top row
    # -----------------------------
    ax_axial = [
        fig.add_subplot(gs[0, i])
        for i in range(len(ax_slices))
    ]

    im_axial = [
        ax.imshow(im_orig[:, :, z], cmap='gray')
        for ax, z in zip(ax_axial, ax_slices)
    ]

    for ax, z in zip(ax_axial, ax_slices):
        ax.axis('off')
        ax.set_title(f"Axial {z}", fontsize=8)


    # -----------------------------
    # Coronal slices: middle row
    # -----------------------------
    ax_coronal = [
        fig.add_subplot(gs[1, i])
        for i in range(len(cor_slices))
    ]

    im_coronal = [
        ax.imshow(im_orig[:, x, :], cmap='gray')
        for ax, x in zip(ax_coronal, cor_slices)
    ]

    for ax, x in zip(ax_coronal, cor_slices):
        ax.axis('off')
        ax.set_title(f"Coronal {x}", fontsize=8)


    # -----------------------------
    # Sagittal slices: bottom row
    # -----------------------------
    ax_sagittal = [
        fig.add_subplot(gs[2, i])
        for i in range(len(sag_slices))
    ]

    im_sagittal = [
        ax.imshow(im_orig[y, :, :].T, cmap='gray')
        for ax, y in zip(ax_sagittal, sag_slices)
    ]

    for ax, y in zip(ax_sagittal, sag_slices):
        ax.axis('off')
        ax.set_title(f"Sagittal {y}", fontsize=8)

    # -----------------------------
    # Sliders
    # -----------------------------
    axcolor = 'lightgoldenrodyellow'
    slider_height = 0.025
    ax_x = plt.axes([0.15, 0.01, 0.7, slider_height], facecolor=axcolor)
    ax_y = plt.axes([0.15, 0.04, 0.7, slider_height], facecolor=axcolor)
    ax_z = plt.axes([0.15, 0.07, 0.7, slider_height], facecolor=axcolor)
    ax_r = plt.axes([0.15, 0.10, 0.7, slider_height], facecolor=axcolor)

    sx = Slider(ax_x, 'X shift', -40, 40, valinit=0, valstep=1)
    sy = Slider(ax_y, 'Y shift', -40, 40, valinit=0, valstep=1)
    sz = Slider(ax_z, 'Z shift', -nz//2, nz//2, valinit=0, valstep=1)
    sr = Slider(ax_r, 'Rotate (deg)', -45, 45, valinit=0)

    # -----------------------------
    # Update function
    # -----------------------------
    def update(val):

        # Apply shifts to the entire 3D volume
        vol = np.roll(
            im_orig.copy(),
            int(sx.val),
            axis=1
        )

        vol = np.roll(
            vol,
            int(sy.val),
            axis=0
        )

        vol = np.roll(
            vol,
            int(sz.val),
            axis=2
        )

        # Rotate the entire volume in the axial plane
        vol = rotate_inplane(vol, sr.val)

        # Update the 10 axial slices
        for img, z in zip(im_axial, ax_slices):
            img.set_data(vol[:, :, z])

        # Update the 10 coronal slices
        for img, x in zip(im_coronal, cor_slices):
            img.set_data(vol[:, x, :])

        # Update the 10 sagittal slices
        for img, y in zip(im_sagittal, sag_slices):
            img.set_data(vol[y, :, :].T)

        fig.canvas.draw_idle()


    # Connect sliders
    sx.on_changed(update)
    sy.on_changed(update)
    sz.on_changed(update)
    sr.on_changed(update)

    # -----------------------------
    # Save button
    # -----------------------------
    ax_save = plt.axes([0.88, 0.01, 0.1, 0.05])
    btn_save = Button(ax_save, 'SAVE', color='lightgreen')

    def save_callback(event):
        vol = np.roll(im_orig.copy(), int(sx.val), axis=1)
        vol = np.roll(vol, int(sy.val), axis=0)
        vol = np.roll(vol, int(sz.val), axis=2)
        vol = rotate_inplane(vol, sr.val)

        nib.save(nib.Nifti1Image(vol, nifti.affine, nifti.header), out_path)
        print(f"✅ Saved: {out_path}")
        nifti_saved = nib.load(out_path)
        print(f"Saved shape: {nifti_saved.shape}")
        print(f"Saved voxel size: {nifti_saved.header.get_zooms()} mm")

        s = OrthoSlicer3D(nifti_saved.get_fdata())
        s.show()

    btn_save.on_clicked(save_callback)

    plt.show()

    nifti_img_saved = nib.load(out_path)
    print("Saved NIfTI image shape:", nifti_img_saved.shape)
    print("Saved NIfTI image header:", nifti_img_saved.header)
    # get resolution
    resolution = nifti_img_saved.header.get_zooms()
    print("Saved NIfTI image resolution:", resolution)
    visible = False
    if visible:
        s = OrthoSlicer3D(np.abs(nifti_img_saved.get_fdata()))
        s.clim = [0, np.abs(1.5 * np.max(np.abs(nifti_img_saved.get_fdata())))]
        s.cmap = 'gray'
        s.show()
