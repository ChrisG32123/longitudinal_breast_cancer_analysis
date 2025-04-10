#!/usr/bin/env python3

import os
import shutil
import zipfile
from datetime import datetime
import argparse
import multiprocessing
from tqdm import tqdm
import SimpleITK as sitk

# Valid file types to keep *after* organization in the output directory
VALID_EXTENSIONS = {".dcm", ".json", ".nii.gz"}
VALID_FILENAMES = {"LICENSE"}

def filter_series(date_path):
    """
    Return the set of series IDs for which we have both a .zip and a .json
    in `date_path`. We do NOT check ZIP validity here – that is deferred to the unzip step.
    """
    zip_ids  = set()
    json_ids = set()

    for f in os.listdir(date_path):
        fpath = os.path.join(date_path, f)
        if f.endswith('.zip') and os.path.isfile(fpath):
            zip_ids.add(os.path.splitext(f)[0])
        elif f.endswith('.json') and os.path.isfile(fpath):
            json_ids.add(os.path.splitext(f)[0])

    return zip_ids.intersection(json_ids)

def process_series(task):
    """
    Process a single (patient_folder, date_folder, series_id) tuple:
      1) Copy .zip and .json from input_dir to output_dir
      2) Attempt to unzip the .zip
      3) Remove local .zip
      4) Keep .json in the subfolder
    If the ZIP is corrupt, skip it (and remove partially created folder).
    """
    patient_folder, date_folder, series_id, input_dir, output_dir = task

    # Input paths
    in_date_path = os.path.join(input_dir, patient_folder, date_folder)
    zip_file_path = os.path.join(in_date_path, series_id + ".zip")
    json_file_path = os.path.join(in_date_path, series_id + ".json")
    
    # Output structure
    out_date_path = os.path.join(output_dir, patient_folder, date_folder)
    os.makedirs(out_date_path, exist_ok=True)

    # Subfolder for this series
    series_out_dir = os.path.join(out_date_path, series_id)
    os.makedirs(series_out_dir, exist_ok=True)

    ### Unzip Series ###

    # Copy the ZIP into the subfolder
    local_zip_path = os.path.join(series_out_dir, series_id + ".zip")
    try:
        shutil.copy2(zip_file_path, local_zip_path)
    except Exception as e:
        print(f"[WARNING] Failed to copy ZIP {zip_file_path}: {e}", flush=True)
        return

    # Unzip in place, catching any corruption errors
    try:
        with zipfile.ZipFile(local_zip_path, 'r') as zf:
            zf.extractall(series_out_dir)
    except zipfile.BadZipFile:
        print(f"[WARNING] Bad/corrupt ZIP for series {series_id}, skipping.", flush=True)
        os.remove(local_zip_path)
        # Optionally remove the partial output folder:
        shutil.rmtree(series_out_dir, ignore_errors=True)
        return
    except Exception as e:
        print(f"[WARNING] Failed to unzip {local_zip_path}: {e}", flush=True)
        os.remove(local_zip_path)
        shutil.rmtree(series_out_dir, ignore_errors=True)
        return

    # Remove the local ZIP
    os.remove(local_zip_path)

    ### NIfTI Conversion ###

    # Collect DICOM series files
    dcm_files = []
    for root, dirs, files in os.walk(series_out_dir):
        for fname in files:
            if fname.lower().endswith(".dcm"):
                dcm_files.append(os.path.join(root, fname))

    # Check that we have DICOM files to convert
    if len(dcm_files) == 0:
        print(f"[WARNING] No DICOM files found after unzipping for series {series_id}. Skipping.")
        shutil.rmtree(series_out_dir, ignore_errors=True)  # remove partial data
        return

    # Convert DICOM to NIfTI
    try:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dcm_files)
        image = reader.Execute()
    except Exception as e:
        print(f"[WARNING] SITK failed to read DICOM for series {series_id}: {e}", flush=True)
        shutil.rmtree(series_out_dir, ignore_errors=True)
        return

    # Write out NIfTI file
    nifti_path = os.path.join(series_out_dir, series_id + ".nii.gz")
    try:
        sitk.WriteImage(image, nifti_path)
    except Exception as e:
        print(f"[WARNING] Failed to write NIfTI for {series_id}: {e}", flush=True)
        shutil.rmtree(series_out_dir, ignore_errors=True)
        return
    
    # Compress DICOM files
    for dcm_file in dcm_files:
        try:
            os.remove(dcm_file)
        except Exception as e:
            print(f"[WARNING] Failed to remove DICOM file {dcm_file}: {e}", flush=True)

    ### Copy JSON Sidecar ###

    # Copy JSON sidecar
    local_json_path = os.path.join(series_out_dir, series_id + ".json")
    try:
        shutil.copy2(json_file_path, local_json_path)
    except Exception as e:
        print(f"[WARNING] Failed to copy JSON for {series_id}: {e}", flush=True)

    print(f"Finished series {series_id} for patient {patient_folder}, date {date_folder}.", flush=True)

    return (patient_folder, date_folder, series_id)

def remove_invalid_files_and_empty_dirs(root_dir):
    """
    In-place cleaning of `root_dir`:
      1) Remove any file not .dcm/.json or named LICENSE
      2) Remove empty folders (bottom-up)
    """
    # 1) Remove invalid files
    # We'll do a bottom-up pass so we can safely remove subfolders if they become empty
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # Remove invalid files in this directory
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            ext = os.path.splitext(fname)[1].lower()
            if (ext not in VALID_EXTENSIONS) and (fname not in VALID_FILENAMES):
                try:
                    os.remove(fpath)
                    print(f"[INFO] Removed invalid file: {fpath}")
                except Exception as e:
                    print(f"[WARNING] Could not remove {fpath}: {e}")

    # 2) Remove empty directories (again bottom-up)
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # If the directory is empty after removing invalid files/subfolders, remove it
        try:
            if not os.listdir(dirpath):  # no files, no subdirs
                os.rmdir(dirpath)
                print(f"[INFO] Removed empty directory: {dirpath}")
        except Exception as e:
            print(f"[WARNING] Failed to remove {dirpath}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Organize I-SPY2 data in parallel, then clean the output directory in-place."
    )
    parser.add_argument("--input_dir", required=True, help="Path to unorganized input data")
    parser.add_argument("--output_dir", required=True, help="Where to store organized + cleaned output")
    parser.add_argument("--nprocs", type=int, default=1, help="Number of parallel processes")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    nprocs = args.nprocs

    # Build a list of tasks for each valid (zip + json) pair
    tasks = []
    print(f"Scanning input directory: {input_dir}", flush=True)
    patients = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for patient_folder in patients:
        patient_path = os.path.join(input_dir, patient_folder)

        # Quick checks
        if not (patient_folder.startswith("ACRIN-6698") or patient_folder.startswith("ISPY2")):
            print(f"Skipping {patient_folder}: not recognized ID format.", flush=True)
            continue

        print(f"  Found patient folder: {patient_folder}", flush=True)

        date_folders = [
            d for d in os.listdir(patient_path)
            if os.path.isdir(os.path.join(patient_path, d))
        ]
        for date_folder in date_folders:
            # Validate date format
            try:
                datetime.strptime(date_folder, "%m-%d-%Y")
            except ValueError:
                print(f"    Skipping {date_folder}: invalid mm-dd-YYYY format.", flush=True)
                continue

            date_path = os.path.join(patient_path, date_folder)
            series_ids = filter_series(date_path)

            # Build tasks for each valid series
            for sid in series_ids:
                tasks.append((patient_folder, date_folder, sid, input_dir, output_dir))

    print(f"Found {len(tasks)} valid (zip+json) pairs. Starting parallel extraction...", flush=True)

    # Organize: Unzipping from input_dir to output_dir
    if nprocs > 1:
        print(f"Using {nprocs} processes for organization...", flush=True)
        with multiprocessing.Pool(processes=nprocs) as pool:
            for _ in tqdm(pool.imap_unordered(process_series, tasks), total=len(tasks)):
                pass
    else:
        # Serial fallback
        for _ in tqdm(map(process_series, tasks), total=len(tasks)):
            pass

    print("[INFO] Organization complete. Now cleaning the output directory in-place...", flush=True)

    # Clean: Remove invalid files and empty dirs from output_dir
    remove_invalid_files_and_empty_dirs(output_dir)

    print("[INFO] All done! Organized + cleaned data is in:", output_dir)

if __name__ == "__main__":
    main()
