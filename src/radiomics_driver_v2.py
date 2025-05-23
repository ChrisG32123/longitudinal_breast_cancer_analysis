#!/usr/bin/env python3
"""
Extract PyRadiomics shape features for exactly one patient,
selected by --patient-index (0-based) from the sorted patient list.
Adds steps to isolate the tumor mask (label 0) and crop the image + mask
to the tumor bounding box with a 5-pixel margin.
"""

import argparse
import json
import zipfile
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from itertools import groupby

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

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

def extract_segmentation_masks(seg_img: sitk.Image) -> list[sitk.Image]:
    """Squeeze + threshold → recreate 3D tumor mask with original spacing/origin/direction."""
    arr = np.squeeze(sitk.GetArrayFromImage(seg_img))

    tumor_mask = (arr % 2 == 0).astype(np.uint8)
    bounding_box_mask = ((arr == 1) | (arr == 17)).astype(np.uint8)  # pixel values for tumorous or non-tumorous tissue

    masks = [tumor_mask, bounding_box_mask]
    extracted_images = []
    for mask in masks:
        out = sitk.GetImageFromArray(mask)
        out.SetSpacing(seg_img.GetSpacing())
        out.SetOrigin(seg_img.GetOrigin())
        out.SetDirection(np.array(seg_img.GetDirection()).reshape(seg_img.GetDimension(), seg_img.GetDimension())[:3,:3].reshape(-1))
        extracted_images.append(out)
    return extracted_images

def roi_crop(dce_img: sitk.Image, bbox_img: sitk.Image, tumor_img:sitk.Image, margin: int = 5) -> list[sitk.Image]:
    """
    Crop the dce and tumor image according to the bounding box image's coordinates.
    """
    bounding_box_arr = sitk.GetArrayFromImage(bbox_img)
    nz = np.nonzero(bounding_box_arr)
    if len(nz[0]) == 0:
        raise ValueError("Bounding box image is empty, cannot calculate bounding box coordinates.")
    
    # min and max indices along each axis
    z0, y0, x0 = np.min(nz, axis=1)
    z1, y1, x1 = np.max(nz, axis=1)

    # add m pixel margin, pad and clamp to each image bounds
    x0 = int(max(0, x0 - margin))
    y0 = int(max(0, y0 - margin))
    z0 = int(max(0, z0 - margin))
    x1 = int(min(dce_img.GetSize()[0] - 1, x1 + margin))
    y1 = int(min(dce_img.GetSize()[1] - 1, y1 + margin))
    z1 = int(min(dce_img.GetSize()[2] - 1, z1 + margin))

    # crop both image and mask
    roi = sitk.RegionOfInterestImageFilter()
    roi.SetIndex([x0, y0, z0])
    roi.SetSize([x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1])
    cropped_frame = []
    for img in [dce_img, tumor_img]:
        cropped_img = roi.Execute(img)
        # cropped_img.CopyInformation(img)  # TODO: Fix!! Cannot copy information as sizes no longer match
        cropped_frame.append(cropped_img)

    return cropped_frame
    
def reorient_and_resample(bbox_img: sitk.Image, tumor_img: sitk.Image, dce_img: sitk.Image):
    """Bring all images into RAS and same grid (nearest‐neighbor for mask)."""
    orient = sitk.DICOMOrientImageFilter()
    orient.SetDesiredCoordinateOrientation("RAS")
    dce_r = orient.Execute(dce_img)
    tum_r = orient.Execute(tumor_img)
    bbox_r = orient.Execute(bbox_img)

    tum_rs = sitk.Resample(
        tum_r, dce_r, sitk.Transform(),
        sitk.sitkNearestNeighbor, 0,
        tum_r.GetPixelID()
    )

    bbox_rs = sitk.Resample(
        bbox_r, dce_r, sitk.Transform(),
        sitk.sitkNearestNeighbor, 0,
        bbox_r.GetPixelID()
    )

    return bbox_rs, tum_rs, dce_r

# ——— Paths —————————————————————————————————————————————————————
HOME_DATA     = Path("/mnt/home/gerlac37/ISPY2/data")
VALID_PATIENT_INFO    = HOME_DATA / "valid_patient_information.json"
REF_RADIOMICS = HOME_DATA / "Multi-feature-MRI-NACT-Data.xlsx"
OUT_DIR       = HOME_DATA / "patient_radiomics_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ——— Argument parsing —————————————————————————————————————————
p = argparse.ArgumentParser()
p.add_argument("--patient-index", type=int, required=True,
               help="0-based index into the sorted list of valid patients")
args = p.parse_args()

# ——— Build the patient list —————————————————————————————————————
ref_df   = pd.read_excel(REF_RADIOMICS)
ref_ids  = set(ref_df["CLINICAL-TRIAL-SUBJECT-ID"].astype(int))

valid_patient_info = json.loads(VALID_PATIENT_INFO.read_text())
valid_ids  = set(map(int, valid_patient_info.keys()))

# Common patient IDs in reference list and have valid patient info
patient_ids = sorted(ref_ids & valid_ids)
if args.patient_index >= len(patient_ids):
    raise IndexError(f"Index {args.patient_index} out of range (0–{len(patient_ids)-1})")

# Get the patient ID for the specified index (script handles single patient)
patient_id = patient_ids[args.patient_index]
print(f"Processing patient #{args.patient_index} → ID {patient_id}")

# ——— Prepare the extractor —————————————————————————————————————
EXTRACTOR = featureextractor.RadiomicsFeatureExtractor()
EXTRACTOR.enableFeatureClassByName("shape")

# ——— Worker for one patient ————————————————————————————————————
def process_patient(pid):
    rows = []
    reader = sitk.ImageSeriesReader()
    info = valid_patient_info[str(pid)]

    for tp in info["Time Labels"]:
        try:
            # — get the timepoint info
            dce_path  = Path(tp["DCE Zip Path"])
            mask_path = Path(tp["Mask Zip Path"])

            # Read all frames using improved DCE extraction
            dce_frames = read_dce_series_from_patient_info(patient_id, dce_path)

            # Read & extract mask
            seg_img   = read_seg_series_from_patient_info(patient_id, mask_path)
            tum_mask_img, bbox_mask_img = extract_segmentation_masks(seg_img)

            # Process images
            processed_frames = []
            for i, f in enumerate(dce_frames, start=1):
                # Reorient & resample mask + each frame
                bbox_rs, tum_rs, dce_r = reorient_and_resample(bbox_mask_img, tum_mask_img, f)
                # Crop to region of interest using bounding box from the segmentation image
                dce_img_crop, tum_img_crop = roi_crop(dce_r, bbox_rs, tum_rs)
                processed_frames.append((dce_img_crop, tum_img_crop))

            # Temporarily pull out first post-contrast frame  # TODO: Study all DCE frames later
            dce, tum = processed_frames[1] if len(processed_frames) > 1 else processed_frames[0]

            # extract radiomics features
            R = EXTRACTOR.execute(dce, tum)

            # Add features to output
            rows.append({
                "patient_id":       pid,
                "date_str":         tp["Date"],
                "date_ordinal":     datetime.strptime(tp["Date"], "%m-%d-%Y").toordinal(),
                "volume":           R.get("original_shape_VoxelVolume", np.nan),
                "sphericity":       R.get("original_shape_Sphericity", np.nan),
                "longest_diameter": R.get("original_shape_Maximum3DDiameter", np.nan),
                "time_label":       tp["Label"],
            })

        except Exception as e:
            print(f"[WARN] pid {pid} {tp['Label']}: {e}")

    return rows

# ——— Run & write out —————————————————————————————————————————
results = process_patient(patient_id)
outfile = OUT_DIR / f"radiomics_{patient_id}.csv"
pd.DataFrame(results).to_csv(outfile, index=False)
print(f"Wrote {len(results)} rows to {outfile}")
