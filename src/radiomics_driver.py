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

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# ——— Paths —————————————————————————————————————————————————————
HOME_DATA     = Path("/mnt/home/gerlac37/ISPY2/data")
VALID_INFO    = HOME_DATA / "valid_patient_information.json"
REF_RADIOMICS = HOME_DATA / "Multi-feature-MRI-NACT-Data.xlsx"
OUT_DIR       = HOME_DATA / "patient_radiomics_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ——— Argument parsing —————————————————————————————————————————
p = argparse.ArgumentParser()
p.add_argument("--patient-index", type=int, required=True,
               help="0-based index into the sorted list of valid patients")
args = p.parse_args()

# ——— Build the patient list —————————————————————————————————————
ref_df   = pd.read_excel(REF_RADIOMICS)
ref_ids  = set(ref_df["CLINICAL-TRIAL-SUBJECT-ID"].astype(int))

valid_info = json.loads(VALID_INFO.read_text())
valid_ids  = set(map(int, valid_info.keys()))

patient_ids = sorted(ref_ids & valid_ids)
if args.patient_index >= len(patient_ids):
    raise IndexError(f"Index {args.patient_index} out of range (0–{len(patient_ids)-1})")

patient_id = patient_ids[args.patient_index]
print(f"Processing patient #{args.patient_index} → ID {patient_id}")

# ——— Prepare the extractor —————————————————————————————————————
EXTRACTOR = featureextractor.RadiomicsFeatureExtractor()
EXTRACTOR.enableFeatureClassByName("shape")

# ——— Worker for one patient ————————————————————————————————————
def process_patient(pid):
    rows = []
    reader = sitk.ImageSeriesReader()
    info = valid_info[str(pid)]

    for di in info["Time Labels"]:
        try:
            # — unzip DCE & SEG into a temp folder
            tmp = Path(tempfile.mkdtemp(prefix=f"{pid}_{di['Label']}_"))
            for sid_key, key in (("DCE Series ID","DCE Zip Path"),
                                 ("Mask Series ID","Mask Zip Path")):
                sid = di[sid_key]
                zp  = di[key]
                with zipfile.ZipFile(zp) as zf:
                    zf.extractall(tmp / str(sid))

            # — read the DCE series
            reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(tmp/di["DCE Series ID"])))
            dce = reader.Execute()
            # — read the segmentation series
            reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(tmp/di["Mask Series ID"])))
            seg = reader.Execute()

            # — clean up the extracted files
            shutil.rmtree(tmp, ignore_errors=True)

            # — if 4D, drop to 3D (first volume)
            for img in (dce, seg):
                if img.GetDimension() == 4:
                    size4 = list(img.GetSize()); size4[3] = 0
                    img = sitk.Extract(img, size4, [0,0,0,0])
            dce = dce if dce.GetDimension()==3 else img
            seg = seg if seg.GetDimension()==3 else img

            # — ensure mask and image geometries match
            if dce.GetSize() != seg.GetSize():
                seg = sitk.Resample(
                    seg, dce, sitk.Transform(),
                    sitk.sitkNearestNeighbor, 0, seg.GetPixelID()
                )

            # — isolate the tumor label (0) and rebuild a binary mask
            arr   = sitk.GetArrayFromImage(seg)            # shape: [z,y,x]
            mask0 = (arr == 0).astype(np.uint8)             # 1 where tumor is, 0 elsewhere
            seg   = sitk.GetImageFromArray(mask0)           # back to SimpleITK
            seg.CopyInformation(dce)                        # preserve origin/spacing/direction

            # — compute bounding box of the tumor mask (with 5-pixel margin)
            nz = np.nonzero(mask0)
            if len(nz[0]) == 0:
                print(f"[WARN] pid {pid} {di['Label']}: empty mask, skipping")
                continue

            # min and max indices along each axis
            z0, y0, x0 = np.min(nz, axis=1)
            z1, y1, x1 = np.max(nz, axis=1)
            m = 5
            # pad and clamp to image bounds
            x0 = int(max(0, x0 - m))
            y0 = int(max(0, y0 - m))
            z0 = int(max(0, z0 - m))
            x1 = int(min(dce.GetSize()[0] - 1, x1 + m))
            y1 = int(min(dce.GetSize()[1] - 1, y1 + m))
            z1 = int(min(dce.GetSize()[2] - 1, z1 + m))

            # crop both image and mask
            roi = sitk.RegionOfInterestImageFilter()
            roi.SetIndex([x0, y0, z0])
            roi.SetSize([x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1])
            dce = roi.Execute(dce)
            seg = roi.Execute(seg)

            # — now extract shape radiomics on the cropped volume
            R = EXTRACTOR.execute(dce, seg)

            rows.append({
                "patient_id":       pid,
                "date_str":         di["Date"],
                "date_ordinal":     datetime.strptime(di["Date"], "%m-%d-%Y").toordinal(),
                "volume":           R.get("original_shape_VoxelVolume", np.nan),
                "sphericity":       R.get("original_shape_Sphericity", np.nan),
                "longest_diameter": R.get("original_shape_Maximum3DDiameter", np.nan),
                "time_label":       di["Label"],
            })

        except Exception as e:
            print(f"[WARN] pid {pid} {di['Label']}: {e}")

    return rows

# ——— Run & write out —————————————————————————————————————————
results = process_patient(patient_id)
outfile = OUT_DIR / f"radiomics_{patient_id}.csv"
pd.DataFrame(results).to_csv(outfile, index=False)
print(f"Wrote {len(results)} rows to {outfile}")
