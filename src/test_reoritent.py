#!/usr/bin/env python3
"""
test_orientation_nifti.py

By Chris Gerlach, June 1, 2025

For the first 10 patients in VALID_PATIENT_INFO:
 - Read one DCE volume (the first time point)
 - Read the full 3D segmentation, extract tumor and bounding-box masks
 - Reorient each to RAS and resample masks onto the DCE grid
 - Save each of the three resulting 3D SITK volumes as NIfTI files
   under test_output/<patient_id>/<time_label>/:
     DCE.nii.gz
     TumorMask.nii.gz
     BBoxMask.nii.gz
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
MAX_PATIENTS      = 10
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


def extract_segmentation_masks(seg_img: sitk.Image) -> tuple[sitk.Image, sitk.Image]:
    """
    From a multi-label seg_img, build:
      - tumor_mask  (all odd labels := tumor)
      - full_roi_mask (labels == 1 or == 17 denotes bounding-box region)
    Returns both as fresh SITK volumes with identical spacing/origin/direction.
    """
    arr = np.squeeze(sitk.GetArrayFromImage(seg_img))
    tumor_mask = (arr % 2 == 0).astype(np.uint8)
    full_roi_mask = ((arr == 1) | (arr == 17)).astype(np.uint8)

    def to_sitk(m: np.ndarray) -> sitk.Image:
        out = sitk.GetImageFromArray(m)
        out.SetSpacing(seg_img.GetSpacing())
        out.SetOrigin(seg_img.GetOrigin())
        dir3 = (
            np.array(seg_img.GetDirection())
            .reshape(seg_img.GetDimension(), seg_img.GetDimension())[:3, :3]
        )
        out.SetDirection(tuple(dir3.flatten()))
        return out

    return to_sitk(tumor_mask), to_sitk(full_roi_mask)


def reorient_and_resample(bbox_img: sitk.Image, tumour_img: sitk.Image, dce_img: sitk.Image):
    """
    Take 3D volumes in unknown orientation → reorient to RAS using DICOM tags → resample both masks
    onto the DCE’s grid. Return (bbox_rs, tum_rs, dce_r).
    """
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


def write_volume_as_nifti(volume: sitk.Image, out_file: Path):
    """
    Write a 3D SITK volume as a compressed NIfTI file (.nii.gz).
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(volume, str(out_file), True)  # True → write as compressed NIfTI (.nii.gz)


def process_one_patient(patient_id: str, pinfo: dict):
    """
    For this patient, we will:
      1) Load DCE list-of-3D volumes
      2) Load segmentation volume → extract (tumor_mask, bbox_mask)
      3) Run reorient_and_resample(...) on the FIRST DCE volume only
      4) Write out the three results as NIfTI under test_output
    """
    for tp in pinfo["Time Labels"]:
        lbl       = tp["Label"]
        dce_path  = Path(tp["DCE Zip Path"])
        mask_path = Path(tp["Mask Zip Path"])

        dce_volumes = read_dce_series_from_patient_info(patient_id, dce_path)
        if not dce_volumes:
            print(f"  → {patient_id} {lbl}: no DCE volumes, skipping")
            continue

        seg_img = read_seg_series_from_patient_info(patient_id, mask_path)
        tum_mask, bbox_mask = extract_segmentation_masks(seg_img)

        first_dce = dce_volumes[0]
        bbox_rs, tum_rs, dce_r = reorient_and_resample(bbox_mask, tum_mask, first_dce)

        out_dir = TEST_OUTPUT / patient_id / lbl
        write_volume_as_nifti(dce_r,   out_dir / "DCE.nii.gz")
        write_volume_as_nifti(tum_rs,  out_dir / "TumorMask.nii.gz")
        write_volume_as_nifti(bbox_rs, out_dir / "BBoxMask.nii.gz")

        print(f"  → wrote {patient_id}/{lbl} → DCE.nii.gz, TumorMask.nii.gz, BBoxMask.nii.gz")


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
