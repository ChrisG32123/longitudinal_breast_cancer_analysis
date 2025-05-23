#!/usr/bin/env python3
"""
By Chris Gerlach, May 22, 2025

Creates subplot visuals of the tumor masks overlaid evenly distributed slices
across each 3D frame across time.
"""
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from itertools import groupby

import numpy as np
import SimpleITK as sitk
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ——— CONFIG ——————————————————————————————————————————————————————————————
VALID_PATIENT_INFO = Path("/mnt/home/gerlac37/ISPY2/data/valid_patient_information.json")
SLICE_STEP         = 10
OUT_DIR            = Path("/mnt/home/gerlac37/ISPY2/visuals/overlays")
# —————————————————————————————————————————————————————————————————————————

def read_seg_series_from_patient_info(series_id: str, series_path: Path) -> sitk.Image:
    """Unzip → read entire 3D segmentation → clean up."""
    temp_dir = Path(tempfile.gettempdir()) / f"seg_{series_id}"
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(series_path,'r') as zf:
        zf.extractall(temp_dir)

    reader = sitk.ImageSeriesReader()
    files  = reader.GetGDCMSeriesFileNames(str(temp_dir))
    reader.SetFileNames(files)
    seg = reader.Execute()

    shutil.rmtree(temp_dir, ignore_errors=True)
    return seg

# ————— DCE Extraction (temporal + spatial sorting) —————————————————————————————
def read_dce_series_from_patient_info(series_id: str, series_zip: Path) -> list[sitk.Image]:
    """
    Unzip the DCE series, sort the slices temporally then spatially,
    and return a list of correctly ordered 3D images.
    """
    tags = {
        "NumTemps":             "0020|0105",  # for sanity‑check
        "TempPosID":            "0020|0100",
        "AcqTime":              "0008|0032",
        "InstanceNumber":       "0020|0013",
        "SliceLocation":        "0020|1041",
        "ImagePositionPatient": "0020|0032",
    }

    # 1) Unzip into temp dir
    tmp = Path(tempfile.gettempdir()) / f"dce_{series_id}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(series_zip, "r") as zf:
        zf.extractall(tmp)

    # 2) Gather files
    reader = sitk.ImageSeriesReader()
    all_files = reader.GetGDCMSeriesFileNames(str(tmp))
    if not all_files:
        raise FileNotFoundError(f"No DICOMs in {tmp}")

    # 3) Read meta from first file
    first = sitk.ImageFileReader()
    first.SetFileName(all_files[0]); first.LoadPrivateTagsOn(); first.ReadImageInformation()
    keys0 = set(first.GetMetaDataKeys())
    num_meta = None
    if tags["NumTemps"] in keys0:
        num_meta = int(first.GetMetaData(tags["NumTemps"])) + 1

    # 4) Build index of (time_idx, z, path)
    entries = []
    for fp in all_files:
        rdr = sitk.ImageFileReader()
        rdr.SetFileName(fp); rdr.LoadPrivateTagsOn(); rdr.ReadImageInformation()
        keys = set(rdr.GetMetaDataKeys())

        # temporal grouping key
        if tags["TempPosID"] in keys:
            t_idx = int(rdr.GetMetaData(tags["TempPosID"]))
        elif tags["AcqTime"] in keys:
            t_idx = int(float(rdr.GetMetaData(tags["AcqTime"])) // 1)
        else:
            raise KeyError(f"No temporal tag in {fp}")

        # spatial sorting key
        if tags["InstanceNumber"] in keys:
            z = int(rdr.GetMetaData(tags["InstanceNumber"]))
        elif tags["SliceLocation"] in keys:
            z = float(rdr.GetMetaData(tags["SliceLocation"]))
        elif tags["ImagePositionPatient"] in keys:
            ipp = rdr.GetMetaData(tags["ImagePositionPatient"])
            z = float(ipp.split("\\")[-1])
        else:
            raise KeyError(f"No spatial tag in {fp}")

        entries.append((t_idx, z, fp))

    # 5) Global sort
    entries.sort(key=lambda e: (e[0], e[1]))

    # 6) Infer & check number of timepoints
    unique_t = sorted({e[0] for e in entries})
    num_inferred = len(unique_t)
    if num_meta is not None and num_meta != num_inferred:
        print(f"Warning: meta num={num_meta} vs inferred={num_inferred}")

    # 7) Group & load volumes
    volumes: list[sitk.Image] = []
    for t_idx, group in groupby(entries, key=lambda e: e[0]):
        fps = [e[2] for e in group]
        vr = sitk.ImageSeriesReader()
        vr.SetFileNames(fps)
        volumes.append(vr.Execute())

    shutil.rmtree(tmp, ignore_errors=True)
    return volumes

def extract_tumour_mask(seg_img: sitk.Image) -> sitk.Image:
    """Squeeze + threshold → recreate 3D mask with original spacing/origin/direction."""
    arr = np.squeeze(sitk.GetArrayFromImage(seg_img))
    mask = (arr % 2 == 0).astype(np.uint8)
    out = sitk.GetImageFromArray(mask)
    out.SetSpacing(seg_img.GetSpacing())
    out.SetOrigin(seg_img.GetOrigin())
    out.SetDirection(np.array(seg_img.GetDirection()).reshape(seg_img.GetDimension(), seg_img.GetDimension())[:3,:3].reshape(-1))
    return out

def reorient_and_resample(tumour_img: sitk.Image, dce_img: sitk.Image):
    """Bring both into RAS and same grid (nearest‐neighbor for mask)."""
    orient = sitk.DICOMOrientImageFilter()
    orient.SetDesiredCoordinateOrientation("RAS")
    dce_r = orient.Execute(dce_img)
    tum_r = orient.Execute(tumour_img)

    tum_rs = sitk.Resample(
        tum_r, dce_r, sitk.Transform(),
        sitk.sitkNearestNeighbor, 0,
        tum_r.GetPixelID()
    )
    return tum_rs, dce_r

def visualize_timepoint(patient_id: str,
                        time_label: str,
                        dce_frames: list[sitk.Image],
                        tumour_img: sitk.Image):
    """
    One figure per time point:
      • cols = dynamic frames
      • rows = spatial slices
    """
    # convert to numpy arrays
    dce_arrs = [sitk.GetArrayFromImage(im) for im in dce_frames]
    mask_arr = sitk.GetArrayFromImage(tumour_img)
    depth    = tumour_img.GetDepth()
    slices   = list(range(0, depth, SLICE_STEP))

    n_rows = len(slices)
    n_cols = len(dce_arrs)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3*n_cols, 3*n_rows),
                             squeeze=False)
    fig.suptitle(f"Patient {patient_id} — {time_label}", fontsize=16)

    for r, sl in enumerate(slices):
        for c, arr in enumerate(dce_arrs):
            ax = axes[r][c]
            ax.imshow(arr[sl], cmap='gray', interpolation='nearest')
            ax.imshow(mask_arr[sl], cmap='jet', alpha=0.5, interpolation='nearest')
            if r == 0:
                ax.set_title(f"Frame {c+1}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"slice {sl}", rotation=0,
                              labelpad=40, va='center', fontsize=10)
            ax.axis('off')

    fig.legend(["Mask"], loc='upper right', bbox_to_anchor=(0.98, 0.98))
    plt.tight_layout(rect=(0,0,1,0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUT_DIR / f"overlay_{patient_id}_{time_label}.png"
    plt.savefig(save_path)
    plt.close(fig)


def process_patient(patient_id: str, patient_info: dict):
    """
    For each time‐label:
      1) read all DCE frames as 3D volumes
      2) read & extract tumour mask (3D)
      3) reorient & resample every DCE frame to the mask grid
      4) visualize that timepoint (cols=frames, rows=slices)
    """
    print(f"\n=== Processing patient {patient_id} ===")
    for tp in patient_info["Time Labels"]:
        label     = tp["Label"]
        dce_path  = Path(tp["DCE Zip Path"])
        mask_path = Path(tp["Mask Zip Path"])

        print(f"  Time Label: {label}")

        # 1) Read all frames using improved DCE extraction
        dce_frames = read_dce_series_from_patient_info(patient_id, dce_path)

        # 2) Read & extract mask
        seg_img   = read_seg_series_from_patient_info(patient_id, mask_path)
        tum_mask  = extract_tumour_mask(seg_img)

        # 3) Reorient & resample mask + each frame
        reoriented_resampled_frames = []
        for i, f in enumerate(dce_frames, start=1):
            tum_rs, dce_rs = reorient_and_resample(tum_mask, f)
            reoriented_resampled_frames.append(dce_rs)
            print(f"    → Frame {i} reoriented & resampled")

        # 4) Visualize that timepoint
        visualize_timepoint(patient_id,
                            label,
                            reoriented_resampled_frames,
                            tum_rs)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(VALID_PATIENT_INFO,'r') as f:
        info = json.load(f)

    for pid, pinfo in info.items():
        process_patient(pid, pinfo)

if __name__ == "__main__":
    main()