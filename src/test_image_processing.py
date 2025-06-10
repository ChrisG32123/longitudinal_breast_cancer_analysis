#!/usr/bin/env python3
"""
test_orientation_nifti_generalized.py

By Chris Gerlach, June 1, 2025 (Revised June 6, 2025)

For the first 10 patients in VALID_PATIENT_INFO:
 - Read one DCE volume (the first time point)
 - Read the full 3D segmentation
 - For each unique integer label in the segmentation (including 0), build a binary mask
 - Reorient each mask to RAS and resample onto the first DCE volume’s grid
 - Save DCE plus one NIfTI per unique label under:
     test_output_nifti/<patient_id>/<time_label>/:
       DCE.nii.gz
       0Mask.nii.gz
       1Mask.nii.gz
       17Mask.nii.gz
       ... etc.
"""

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from itertools import groupby

import numpy as np
import SimpleITK as sitk

# ——— CONFIG ——————————————————————————————————————————————————————————————
VALID_PATIENT_INFO = Path("/mnt/home/gerlac37/ISPY2/data/valid_patient_information.json")
TEST_OUTPUT        = Path("/mnt/home/gerlac37/ISPY2/test_output_nifti")
MAX_PATIENTS       = 10
# —————————————————————————————————————————————————————————————————————————


def read_seg_series_from_patient_info(series_id: str, series_path: Path) -> sitk.Image:
    """
    Unzip → read entire 3D segmentation → clean up.
    Returns a single 3D SITK volume whose voxel values are the label map.
    """
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
    and return a list of 3D SITK volumes—one volume per time point.
    """
    tags = {
        "NumTemps":             "0020|0105",
        "TempPosID":            "0020|0100",
        "AcqTime":              "0008|0032",
        "InstanceNumber":       "0020|0013",
        "SliceLocation":        "0020|1041",
        "ImagePositionPatient": "0020|0032",
    }

    tmp = Path(tempfile.gettempdir()) / f"dce_{series_id}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(series_zip, "r") as zf:
        zf.extractall(tmp)

    reader = sitk.ImageSeriesReader()
    all_files = reader.GetGDCMSeriesFileNames(str(tmp))
    if not all_files:
        raise FileNotFoundError(f"No DICOMs found under {tmp}")

    first = sitk.ImageFileReader()
    first.SetFileName(all_files[0])
    first.LoadPrivateTagsOn()
    first.ReadImageInformation()
    keys0 = set(first.GetMetaDataKeys())
    num_meta = None
    if tags["NumTemps"] in keys0:
        num_meta = int(first.GetMetaData(tags["NumTemps"])) + 1

    entries = []
    for fp in all_files:
        rdr = sitk.ImageFileReader()
        rdr.SetFileName(fp)
        rdr.LoadPrivateTagsOn()
        rdr.ReadImageInformation()
        keys = set(rdr.GetMetaDataKeys())

        if tags["TempPosID"] in keys:
            t_idx = int(rdr.GetMetaData(tags["TempPosID"]))
        elif tags["AcqTime"] in keys:
            t_idx = int(float(rdr.GetMetaData(tags["AcqTime"])) // 1)
        else:
            raise KeyError(f"No temporal tag in {fp}")

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

    entries.sort(key=lambda e: (e[0], e[1]))

    unique_t = sorted({e[0] for e in entries})
    if num_meta is not None and num_meta != len(unique_t):
        print(f"WARNING: metadata says {num_meta} timepoints, found {len(unique_t)}")

    volumes: list[sitk.Image] = []
    for t_idx, group in groupby(entries, key=lambda e: e[0]):
        fps = [e[2] for e in group]
        vr = sitk.ImageSeriesReader()
        vr.SetFileNames(fps)
        volumes.append(vr.Execute())

    shutil.rmtree(tmp, ignore_errors=True)
    return volumes


def extract_all_label_masks(seg_img: sitk.Image) -> dict[int, sitk.Image]:
    """
    Given a multi-label SITK segmentation (any integer per voxel),
    build a binary SITK mask for each unique integer value (including 0).

    Returns a dict: { label_value: binary_mask_SITK_image }
    """
    arr = np.squeeze(sitk.GetArrayFromImage(seg_img))
    unique_vals = np.unique(arr)

    masks: dict[int, sitk.Image] = {}
    for val in unique_vals:
        binary_arr = (arr == val).astype(np.uint8)
        m_img = sitk.GetImageFromArray(binary_arr)
        m_img.SetSpacing(seg_img.GetSpacing())
        m_img.SetOrigin(seg_img.GetOrigin())
        dir3 = (
            np.array(seg_img.GetDirection())
            .reshape(seg_img.GetDimension(), seg_img.GetDimension())[:3, :3]
        )
        m_img.SetDirection(tuple(dir3.flatten()))
        masks[int(val)] = m_img

    return masks


def reorient_and_resample(mask_img: sitk.Image, reference_img: sitk.Image) -> sitk.Image:
    """
    Reorient mask to RAS and resample onto reference_img’s grid (nearest-neighbor).
    Returns the resampled mask.
    """
    orient = sitk.DICOMOrientImageFilter()
    orient.SetDesiredCoordinateOrientation("RAS")
    m_ras = orient.Execute(mask_img)
    ref_ras = orient.Execute(reference_img)

    m_rs = sitk.Resample(
        m_ras, ref_ras, sitk.Transform(),
        sitk.sitkNearestNeighbor, 0, m_ras.GetPixelID()
    )
    return m_rs


def write_volume_as_nifti(volume: sitk.Image, out_file: Path):
    """
    Write a 3D SITK volume as a compressed NIfTI file (.nii.gz).
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(volume, str(out_file), True)  # True → compressed (.nii.gz)


def process_one_patient(patient_id: str, pinfo: dict):
    """
    For this patient:
      - Load first DCE volume
      - Load segmentation
      - Build one binary mask per unique label in seg
      - Reorient+resample each mask onto the DCE grid
      - Save DCE + one NIfTI per label under test_output_nifti/<pid>/<label>/
    """
    for tp in pinfo["Time Labels"]:
        lbl = tp["Label"]
        dce_path = Path(tp["DCE Zip Path"])
        mask_path = Path(tp["Mask Zip Path"])

        # Read DCE volumes
        dce_volumes = read_dce_series_from_patient_info(patient_id, dce_path)
        if not dce_volumes:
            print(f"  → {patient_id} {lbl}: no DCE volumes, skipping")
            continue
        first_dce = dce_volumes[0]

        # Read segmentation
        seg_img = read_seg_series_from_patient_info(patient_id, mask_path)

        # Extract masks for every unique label
        label_masks = extract_all_label_masks(seg_img)
        # label_masks: { integer_label: SITK_binary_mask }

        # Reorient and resample DCE itself to RAS (so masks align)
        orient = sitk.DICOMOrientImageFilter()
        orient.SetDesiredCoordinateOrientation("RAS")
        dce_ras = orient.Execute(first_dce)

        # Set up output directory
        out_dir = TEST_OUTPUT / patient_id / lbl
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save DCE.nii.gz
        write_volume_as_nifti(dce_ras, out_dir / "DCE.nii.gz")

        # For each unique label, reorient+resample and save as "<label>Mask.nii.gz"
        for label_val, mask_img in label_masks.items():
            resampled = reorient_and_resample(mask_img, first_dce)
            out_filename = out_dir / f"{label_val}Mask.nii.gz"
            write_volume_as_nifti(resampled, out_filename)

        print(f"  → wrote {patient_id}/{lbl}: DCE + {len(label_masks)} label masks")


def main():
    with open(VALID_PATIENT_INFO, 'r') as f:
        info = json.load(f)

    if TEST_OUTPUT.exists():
        shutil.rmtree(TEST_OUTPUT)
    TEST_OUTPUT.mkdir(parents=True, exist_ok=True)

    for idx, (pid, pinfo) in enumerate(info.items()):
        if idx >= MAX_PATIENTS:
            break
        print(f"Processing patient {idx+1}/{MAX_PATIENTS}: {pid}")
        try:
            process_one_patient(pid, pinfo)
        except Exception as e:
            print(f"  ERROR for {pid}: {e}")

    print("Finished. Inspect test_output_nifti/ with MITK or any NIfTI viewer.")


if __name__ == "__main__":
    main()
