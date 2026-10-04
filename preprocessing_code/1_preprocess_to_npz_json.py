import io
import json
import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import cfgrib
import numpy as np
import xarray as xr
from tqdm import tqdm

from utils_era5 import (
    preprocess_grib_pressure_levels,
    preprocess_grib_single_levels,
    resize_latlon_grid,
    xarray_to_tvhw,
)

warnings.filterwarnings(
    "ignore",
    message=r".*default value for compat will change from compat='no_conflicts' to compat='override'.*",
    category=FutureWarning,
    module=r"cfgrib\.xarray_store",
)

# ============================================================
# INPUT / OUTPUT PARAMETERS
# ============================================================
DIR_DATA_JSON_SRC = "../data/record_json" # See README.md for how to download this folder.
DIR_DATA_ERA5_SRC = "<ORIGINAL_ERA5_DIR>" # The directory should contain subfolders for each ERA5 sample (ERA5_FOLDER_NAME)
                                          # each with two GRIB files: single_levels.grib and pressure_levels.grib
DIR_PROCESSED_FOLDER = "<PROCESSED_ERA5_DIR>" # The directory where the processed NPZ and JSON files will be saved
DATASET_NAMES = sorted([
    filename.split('.')[0].split('_')[-1]
    for filename in os.listdir(DIR_DATA_JSON_SRC)
    if filename.startswith("data_") and filename.endswith(".json")
])

TARGET_GRID_SIZE = 56
COMPRESS_NPZ_INSIDE_SAMPLE = True
OVERWRITE_TEXT = True

NUM_WORKERS = 4
# ============================================================

def payload_to_npz_bytes(payload: dict, compress: bool = True) -> bytes:
    buffer = io.BytesIO()
    if compress:
        np.savez_compressed(buffer, **payload)
    else:
        np.savez(buffer, **payload)
    return buffer.getvalue()


def save_intermediate_sample(sample):
    folder_name, dict_text_prompt, event_years = sample

    npz_path = os.path.join(DIR_PROCESSED_FOLDER, "npz", f"{folder_name}.npz")
    json_path = os.path.join(DIR_PROCESSED_FOLDER, "json", f"{folder_name}.json")

    # If both final files already exist, optionally overwrite the text JSON
    if os.path.isfile(npz_path) and os.path.isfile(json_path):
        if OVERWRITE_TEXT == False:
            return folder_name
        else:
            tmp_json_path = json_path + ".tmp"
            with open(tmp_json_path, "w", encoding="utf-8") as f:
                json.dump(dict_text_prompt, f, ensure_ascii=False, indent=2)
            os.replace(tmp_json_path, json_path)
            return folder_name

    result = process_one_sample(sample)

    tmp_npz_path = npz_path + ".tmp"
    tmp_json_path = json_path + ".tmp"

    with open(tmp_npz_path, "wb") as f:
        f.write(result["npz"])
    os.replace(tmp_npz_path, npz_path)

    with open(tmp_json_path, "w", encoding="utf-8") as f:
        json.dump(result["json"], f, ensure_ascii=False, indent=2)
    os.replace(tmp_json_path, json_path)

    return folder_name

def build_info_json(all_samples):
    return {
        "num_samples": len(all_samples),
        "samples": [
            {
                "ERA5_FOLDER_NAME": folder_name,
                "EVENT_YEARS": event_years,
            }
            for folder_name, _, event_years in all_samples
        ],
    }

def process_one_sample(sample):
    try:
        folder_name, dict_text_prompt, event_years = sample

        folder_path = os.path.join(DIR_DATA_ERA5_SRC, folder_name)

        files = sorted(os.listdir(folder_path))
        num_files = len(files)
        num_gribs = sum(f.endswith(".grib") for f in files)

        path_single = os.path.join(folder_path, "single_levels.grib")
        path_pressure = os.path.join(folder_path, "pressure_levels.grib")

        assert num_files == 2, f"{folder_name}: expected 2 files, got {num_files}"
        assert num_gribs == 2, f"{folder_name}: expected 2 grib files, got {num_gribs}"
        assert os.path.isfile(path_single), f"Missing file: {path_single}"
        assert os.path.isfile(path_pressure), f"Missing file: {path_pressure}"

        # Each worker opens its own files
        pressure_levels = cfgrib.open_datasets(
            path_pressure,
            backend_kwargs={"indexpath": ""}
        )
        single_levels = cfgrib.open_datasets(
            path_single,
            backend_kwargs={"indexpath": ""}
        )

        ds_pressure_levels = preprocess_grib_pressure_levels(pressure_levels)
        ds_single_levels = preprocess_grib_single_levels(
            single_levels,
            ref_ds=ds_pressure_levels,
            nan_fill_value=0,
        )

        ds_all = xr.merge([ds_pressure_levels, ds_single_levels], compat="no_conflicts")

        for var in ds_all.data_vars:
            assert not ds_all[var].isnull().any().item(), f"NaNs found in variable '{var}'"

        ds_all_resized = resize_latlon_grid(
            ds_all,
            n_lat=TARGET_GRID_SIZE,
            n_lon=TARGET_GRID_SIZE,
        )

        payload_resized = xarray_to_tvhw(ds_all_resized)
        payload_resized["folder_name"] = np.array(folder_name)

        result = {
            "npz": payload_to_npz_bytes(
                payload_resized,
                compress=COMPRESS_NPZ_INSIDE_SAMPLE,
            ),
            "json": dict_text_prompt,
        }
    except Exception as e:
        raise ValueError(f"folder_name: {sample[0]}\n\n{e}")
    return result

def build_samples(dataset_name):
    samples = []
    with open(os.path.join(DIR_DATA_JSON_SRC, f"data_{dataset_name}.json"), "r", encoding="utf-8") as f:
        samples_json = json.load(f)
    
    for sample in samples_json:
        north, west, south, east = sample["ERA5_SNAPSHOT_BBOX"]

        assert north > south
        assert east > west
        assert all(event_year >= 2000 for event_year in sample["EVENT_YEARS"])
        assert "CAPTIONS" in sample
        assert sample["CAPTIONS"] is not None

        samples.append([
            sample["ERA5_FOLDER_NAME"],
            sample["CAPTIONS"],
            sample["EVENT_YEARS"],
        ])

    return samples

if __name__ == '__main__':
    all_samples = []
    print("dataset_name | len(samples)")
    for dataset_name in DATASET_NAMES:
        samples = build_samples(dataset_name)
        all_samples.extend(samples)
        print(f"{dataset_name} | {len(samples)}")
    
    all_samples = all_samples[:20]

    print("\nTotal sample size:")
    print(f"{len(all_samples)}")

    os.makedirs(DIR_PROCESSED_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(DIR_PROCESSED_FOLDER, "npz"), exist_ok=True)
    os.makedirs(os.path.join(DIR_PROCESSED_FOLDER, "json"), exist_ok=True)
    
    info = build_info_json(all_samples)
    with open(os.path.join(DIR_PROCESSED_FOLDER, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        for _ in tqdm(
            executor.map(save_intermediate_sample, all_samples, chunksize=16),
            total=len(all_samples),
            desc="Saving intermediate files",
        ):
            pass