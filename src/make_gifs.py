"""
By Chris Gerlach, May 22, 2025

Creates gifs showing the gadolinium contrast moving through the breast
during the DCE series capture.
"""
import os
from pathlib import Path
import json
import shutil
import tempfile
import zipfile
from itertools import groupby

import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm

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

# ── Example usage ─────────────────────────────────────────────────────────────

VALID_PATIENT_INFO = Path("/mnt/home/gerlac37/ISPY2/data/valid_patient_information.json")

with open(VALID_PATIENT_INFO, 'r') as f:
    info = json.load(f)

for patient_id in tqdm(info.keys()):
    try:
        patient = info[patient_id]
        zip_path = Path(patient["Time Labels"][0]["DCE Zip Path"])

        # call the updated function
        dce_imgs = read_dce_series_from_patient_info(patient_id, zip_path)

        # convert to NumPy array
        dce_arr = np.stack([sitk.GetArrayFromImage(img) for img in dce_imgs])
        print("DCE array shape:", dce_arr.shape)

        ### Make gifs of central slice across frames
        test_dce_arr = dce_arr
        center_slice = test_dce_arr.shape[1] // 2
        print("Test shape", test_dce_arr.shape)
        print("Test shape, swap axes", np.swapaxes(test_dce_arr, 0,1).shape)
        print("Test shape, swap axes and take center slices", np.swapaxes(test_dce_arr, 0,1)[center_slice-5:center_slice+5].shape)

        # Loop through each slice in the range of 10 slices
        skip_slices = 5
        num_slices = 3
        for slice_idx, slice in enumerate(np.swapaxes(test_dce_arr, 0,1)[::skip_slices][(center_slice//skip_slices-num_slices):(center_slice//skip_slices+num_slices)]):
            print("Slice Shape", slice.shape)

            # Convert each slice to a PIL image for each frame
            images = []
            for img in slice:
                images.append(Image.fromarray(img))

            # Save the images as a GIF
            out_path = f'/mnt/home/gerlac37/ISPY2/visuals/gifs_overlay/{patient_id}_T0_slice_{center_slice//skip_slices-num_slices+slice_idx*skip_slices}.gif'
            images[0].save(out_path, save_all=True, append_images=images[1:], duration=1000, loop=0)
            print('Saved gif. Number of images: ', len(images))
    except Exception as e:
        print(f"Error processing patient {patient_id}: {e}")
        continue