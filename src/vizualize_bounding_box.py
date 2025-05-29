#!/usr/bin/env python3
"""
By Chris Gerlach, May 24, 2025

Creates subplot visuals of the tumor masks overlaid evenly distributed slices
across each 3D frame across time—and separately draws the ROI bounding box.
"""
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from itertools import groupby

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

# ——— CONFIG ——————————————————————————————————————————————————————————————
VALID_PATIENT_INFO = Path("/mnt/home/gerlac37/ISPY2/data/valid_patient_information.json")
SLICE_STEP         = 10
VIS_DIR            = Path("/mnt/home/gerlac37/ISPY2/visuals")
OVERLAY_DIR        = VIS_DIR / "overlays"
OUTLINE_DIR      = VIS_DIR / "outlines"
CROPPED_DIR           = VIS_DIR / "bboxes"
# —————————————————————————————————————————————————————————————————————————

def read_seg_series_from_patient_info(series_id: str, series_path: Path) -> sitk.Image:
    """Unzip → read entire 3D segmentation → clean up."""
    tmp = Path(tempfile.gettempdir()) / f"seg_{series_id}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(series_path, 'r') as zf:
        zf.extractall(tmp)
    reader = sitk.ImageSeriesReader()
    files  = reader.GetGDCMSeriesFileNames(str(tmp))
    reader.SetFileNames(files)
    seg = reader.Execute()
    shutil.rmtree(tmp, ignore_errors=True)
    return seg

def read_dce_series_from_patient_info(series_id: str, series_zip: Path) -> list[sitk.Image]:
    """
    Unzip the DCE series, sort the slices temporally then spatially,
    and return a list of correctly ordered 3D images.
    """
    tags = {
        "NumTemps":             "0020|0105",
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

def extract_segmentation_masks(seg_img: sitk.Image) -> tuple[sitk.Image, sitk.Image]:
    """Return (tumor_mask, full_roi_mask) as binary volumes."""
    arr = np.squeeze(sitk.GetArrayFromImage(seg_img))
    tumor_mask       = (arr == 2).astype(np.uint8)  # TUMOR MASK
    full_roi_mask    = ((arr == 1) | (arr == 17)).astype(np.uint8)  # REGION OF INTEREST
    def toSitk(m):
        out = sitk.GetImageFromArray(m)
        out.SetSpacing(seg_img.GetSpacing())
        out.SetOrigin(seg_img.GetOrigin())
        out.SetDirection(np.array(seg_img.GetDirection())
                          .reshape(seg_img.GetDimension(),
                                   seg_img.GetDimension())[:3,:3]
                          .reshape(-1))
        return out
    return toSitk(tumor_mask), toSitk(full_roi_mask)

def reorient_and_resample(bbox_img, tumour_img, dce_img):
    """Bring all into RAS & same grid (nearest‐neighbor for masks)."""
    orient = sitk.DICOMOrientImageFilter()
    orient.SetDesiredCoordinateOrientation("RAS")
    dce_r  = orient.Execute(dce_img)
    tum_r  = orient.Execute(tumour_img)
    bbox_r = orient.Execute(bbox_img)
    tum_rs  = sitk.Resample(tum_r,  dce_r, sitk.Transform(),
                            sitk.sitkNearestNeighbor, 0, tum_r.GetPixelID())
    bbox_rs = sitk.Resample(bbox_r, dce_r, sitk.Transform(),
                            sitk.sitkNearestNeighbor, 0, bbox_r.GetPixelID())
    return bbox_rs, tum_rs, dce_r

def visualize_timepoint(patient_id, time_label, frames, mask_vol):
    """Save mask‐overlay only."""
    ARRs = [sitk.GetArrayFromImage(f) for f in frames]
    MARR = sitk.GetArrayFromImage(mask_vol)
    depth = mask_vol.GetDepth()
    slices = list(range(0, depth, SLICE_STEP))
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(slices), len(ARRs),
                             figsize=(3*len(ARRs),3*len(slices)),
                             squeeze=False)
    fig.suptitle(f"{patient_id} — {time_label}", fontsize=16)
    for r, z in enumerate(slices):
        for c, arr in enumerate(ARRs):
            ax = axes[r][c]
            ax.imshow(arr[z], cmap="gray", interpolation="nearest")
            ax.imshow(MARR[z], cmap="jet", alpha=0.5, interpolation="nearest")
            ax.axis("off")
            if r==0: ax.set_title(f"Frame {c+1}", fontsize=10)
            if c==0: ax.set_ylabel(f"slice {z}", rotation=0,
                                   labelpad=40, va="center", fontsize=10)
    fig.legend(["Mask"], loc="upper right", bbox_to_anchor=(0.98,0.98))
    plt.tight_layout(rect=(0,0,1,0.96))
    fig.savefig(OVERLAY_DIR / f"{patient_id}_{time_label}_overlay.png")
    plt.close(fig)

def visualize_timepoint_with_bbox(patient_id, time_label, frames, mask_vol, bbox_vol):
    """Save mask + bounding‐box overlay."""
    ARRs = [sitk.GetArrayFromImage(f) for f in frames]
    MARR = sitk.GetArrayFromImage(mask_vol)
    BARR = sitk.GetArrayFromImage(bbox_vol)
    depth = bbox_vol.GetDepth()
    slices = list(range(0, depth, SLICE_STEP))
    OUTLINE_DIR.mkdir(parents=True, exist_ok=True)

    # compute 3D bounding box coords once
    nz = np.nonzero(BARR)
    if len(nz[0]) == 0:
        print("Bounding box image is empty, cannot calculate bounding box coordinates.")
        return
    z0,y0,x0 = np.min(nz, axis=1)
    z1,y1,x1 = np.max(nz, axis=1)
    width  = x1 - x0
    height = y1 - y0

    fig, axes = plt.subplots(len(slices), len(ARRs),
                             figsize=(3*len(ARRs),3*len(slices)),
                             squeeze=False)
    fig.suptitle(f"{patient_id} — {time_label} (with ROI box)", fontsize=16)
    for r, z in enumerate(slices):
        for c, arr in enumerate(ARRs):
            ax = axes[r][c]
            ax.imshow(arr[z], cmap="gray", interpolation="nearest")
            ax.imshow(MARR[z], cmap="jet", alpha=0.5, interpolation="nearest")
            # draw rectangle in pixel coords: x0,y0 is upper-left corner
            rect = Rectangle((x0, y0), width, height,
                             fill=False, linewidth=1,
                             edgecolor="yellow")
            ax.add_patch(rect)
            ax.axis("off")
            if r==0: ax.set_title(f"Frame {c+1}", fontsize=10)
            if c==0: ax.set_ylabel(f"slice {z}", rotation=0,
                                   labelpad=40, va="center", fontsize=10)
    fig.legend(["Mask","ROI box"], loc="upper right", bbox_to_anchor=(0.98,0.98))
    plt.tight_layout(rect=(0,0,1,0.96))
    fig.savefig(OUTLINE_DIR / f"{patient_id}_{time_label}_bbox.png")
    plt.close(fig)

def visualize_timepoint_only_bbox(patient_id, time_label, frames, mask_vol, bbox_vol):
    """Save mask + bounding‐box overlay."""
    ARRs = [sitk.GetArrayFromImage(f) for f in frames]
    MARR = sitk.GetArrayFromImage(mask_vol)
    BARR = sitk.GetArrayFromImage(bbox_vol)
    depth = bbox_vol.GetDepth()
    slices = list(range(0, depth, SLICE_STEP))
    CROPPED_DIR.mkdir(parents=True, exist_ok=True)

    # compute 3D bounding box coords once
    nz = np.nonzero(BARR)
    if len(nz[0]) == 0:
        print("Bounding box image is empty, cannot calculate bounding box coordinates.")
        return
    z0,y0,x0 = np.min(nz, axis=1)
    z1,y1,x1 = np.max(nz, axis=1)
    width  = x1 - x0
    height = y1 - y0

    fig, axes = plt.subplots(len(slices), len(ARRs),
                             figsize=(3*len(ARRs),3*len(slices)),
                             squeeze=False)
    fig.suptitle(f"{patient_id} — {time_label} (with ROI box)", fontsize=16)
    for r, z in enumerate(slices):
        for c, arr in enumerate(ARRs):
            ax = axes[r][c]
            ax.imshow(arr[z][y0:y1, x0:x1], cmap="gray", interpolation="nearest")
            ax.imshow(MARR[z][y0:y1, x0:x1], cmap="jet", alpha=0.5, interpolation="nearest")
            ax.axis("off")
            if r==0: ax.set_title(f"Frame {c+1}", fontsize=10)
            if c==0: ax.set_ylabel(f"slice {z}", rotation=0,
                                   labelpad=40, va="center", fontsize=10)
    fig.legend(["Mask","ROI box"], loc="upper right", bbox_to_anchor=(0.98,0.98))
    plt.tight_layout(rect=(0,0,1,0.96))
    fig.savefig(CROPPED_DIR / f"{patient_id}_{time_label}_bbox.png")
    plt.close(fig)

def process_patient(patient_id, info):
    print(f"\n=== Processing {patient_id} ===")
    for tp in info["Time Labels"]:
        lbl = tp["Label"]
        dce_path  = Path(tp["DCE Zip Path"])
        mask_path = Path(tp["Mask Zip Path"])
        print("→", lbl)

        # 1) read & split DCE
        dce_frames = read_dce_series_from_patient_info(patient_id, dce_path)

        # 2) read & mask extraction
        seg_img       = read_seg_series_from_patient_info(patient_id, mask_path)
        tum_mask, bbox_mask = extract_segmentation_masks(seg_img)

        # 3) reorient, resample, collect
        dces_rrs = []
        tum_rrs = []
        bbx_rrs = []
        for f in dce_frames:
            b_rs, t_rs, d_rs = reorient_and_resample(bbox_mask, tum_mask, f)
            dces_rrs.append(d_rs)
            tum_rrs.append(t_rs)
            bbx_rrs.append(b_rs)

        # visualize
        # visualize_timepoint(patient_id, lbl, dces_rrs, tum_rrs[-1])
        # visualize_timepoint_with_bbox(patient_id, lbl, dces_rrs, tum_rrs[-1], bbx_rrs[-1])
        visualize_timepoint_only_bbox(patient_id, lbl, dces_rrs, tum_rrs[-1], bbx_rrs[-1])

def main():
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    OUTLINE_DIR.mkdir(parents=True, exist_ok=True)
    CROPPED_DIR.mkdir(parents=True, exist_ok=True)
    with open(VALID_PATIENT_INFO, 'r') as f:
        info = json.load(f)
    for pid, pinfo in info.items():
        process_patient(pid, pinfo)

if __name__ == "__main__":
    main()
