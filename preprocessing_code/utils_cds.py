import os
import shutil
from datetime import datetime, timedelta

import cdsapi
import cfgrib
import numpy as np
import pygrib
import xarray as xr

from utils_era5_vars import *

DATASET_SINGLE_LEVELS = "reanalysis-era5-single-levels"
DATASET_PRESSURE_LEVELS = "reanalysis-era5-pressure-levels"

def download_single_levels(
    cds_url,
    cds_key,
    north,
    west,
    south,
    east,
    year,
    month,
    day,
    time,
    output_file="single_levels_atmos.grib",
):
    client = cdsapi.Client(url=cds_url, key=cds_key)

    request = {
        "product_type": ["reanalysis"],
        "variable": SINGLE_LEVEL_ATMOS_VARS + SINGLE_LEVEL_WAVE_VARS,
        "year": year,
        "month": month,
        "day": day,
        "time": time,
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [north, west, south, east],
    }

    client.retrieve(DATASET_SINGLE_LEVELS, request).download(output_file)

def download_pressure_levels(
    cds_url,
    cds_key,
    north,
    west,
    south,
    east,
    year,
    month,
    day,
    time,
    output_file="pressure_levels.grib",
):
    client = cdsapi.Client(url=cds_url, key=cds_key)

    request = {
        "product_type": ["reanalysis"],
        "variable": PRESSURE_LEVEL_VARS,
        "pressure_level": PRESSURE_LEVELS,
        "year": year,
        "month": month,
        "day": day,
        "time": time,
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [north, west, south, east],
    }

    client.retrieve(DATASET_PRESSURE_LEVELS, request).download(output_file)

def download_era5_snapshot(
    cds_url, cds_key, dir_save, folder_name,
    era5_snapshot_time, era5_snapshot_bbox,
):
    target_dir = os.path.join(dir_save, folder_name)
    os.makedirs(target_dir, exist_ok=True)

    north, west, south, east = era5_snapshot_bbox

    assert -90 <= south < north <= 90

    valid_lon_180 = -180 <= west < east <= 180
    valid_lon_360 = 0 <= west < east <= 360
    assert (valid_lon_180 or valid_lon_360)

    year = era5_snapshot_time["year"]
    month = era5_snapshot_time["month"]
    day = era5_snapshot_time["day"]
    time = era5_snapshot_time["time"]

    output_file = os.path.join(target_dir, "single_levels.grib")
    if not os.path.exists(output_file):
        download_single_levels(
            cds_url, cds_key,
            north, west, south, east,
            year, month, day, time,
            output_file=output_file,
        )

    output_file = os.path.join(target_dir, "pressure_levels.grib")
    if not os.path.exists(output_file):
        download_pressure_levels(
            cds_url, cds_key,
            north, west, south, east,
            year, month, day, time,
            output_file=output_file,
        )

def check_all_bucket_files_complete(dir_save, samples):
    for sample in samples:
        folder_name = sample['folder_name']
        target_dir = os.path.join(dir_save, folder_name)
        if not os.path.exists(os.path.join(target_dir, "single_levels.grib")):
            return False
        if not os.path.exists(os.path.join(target_dir, "pressure_levels.grib")):
            return False
    return True

def download_era5_bucket(
    cds_url, cds_key, dir_save,
    dataset_name, bucket, samples,
):
    if check_all_bucket_files_complete(dir_save, samples):
        return

    dir_save_bucket = os.path.join(dir_save, "buckets")
    folder_name = f"{dataset_name}_bucket_{bucket['bucket_id']}"
    era5_snapshot_time = {
        key: bucket["cds_request"][key]
        for key in ["year", "month", "day", "time"]
    }
    era5_snapshot_bbox = bucket["cds_request"]["area"]  # [north, west, south, east]

    os.makedirs(dir_save_bucket, exist_ok=True)

    download_era5_snapshot(
        cds_url, cds_key, dir_save_bucket, folder_name,
        era5_snapshot_time, era5_snapshot_bbox,
    )

    for sample in samples:
        extract_era5_sample_from_bucket(
            os.path.join(dir_save_bucket, folder_name),
            sample, dir_save,
        )
    
    if os.path.exists(os.path.join(dir_save_bucket, folder_name)):
        shutil.rmtree(os.path.join(dir_save_bucket, folder_name))

def assert_all_samples_in_bucket(samples, bucket):
    bucket_cds_request = bucket["cds_request"]
    bucket_area = bucket_cds_request["area"]  # [north, west, south, east]
    for sample in samples:
        sample_snapshot_bbox = sample["ERA5_SNAPSHOT_BBOX"]  # [north, west, south, east]
        sample_snapshot_time = sample["ERA5_SNAPSHOT_TIME"]
        assert sample_snapshot_bbox[0] <= bucket_area[0]
        assert sample_snapshot_bbox[1] >= bucket_area[1]
        assert sample_snapshot_bbox[2] >= bucket_area[2]
        assert sample_snapshot_bbox[3] <= bucket_area[3]
        
        assert sample_snapshot_time["year"] in bucket_cds_request["year"]
        assert sample_snapshot_time["month"] in bucket_cds_request["month"]
        assert sample_snapshot_time["day"] in bucket_cds_request["day"]
        assert sample_snapshot_time["time"] in bucket_cds_request["time"]



def extract_era5_sample_from_bucket(
    bucket_dir,
    sample,
    output_dir,
):
    north, west, south, east = sample["ERA5_SNAPSHOT_BBOX"]

    assert -90 <= south < north <= 90
    
    valid_lon_180 = -180 <= west < east <= 180
    valid_lon_360 = 0 <= west < east <= 360
    assert (valid_lon_180 or valid_lon_360)

    single_levels_file = os.path.join(bucket_dir, "single_levels.grib")
    pressure_levels_file = os.path.join(bucket_dir, "pressure_levels.grib")

    if not os.path.exists(single_levels_file):
        raise FileNotFoundError(f"Missing bucket file: {single_levels_file}")
    if not os.path.exists(pressure_levels_file):
        raise FileNotFoundError(f"Missing bucket file: {pressure_levels_file}")

    sample_dir = os.path.join(output_dir, sample["folder_name"])
    os.makedirs(sample_dir, exist_ok=True)

    single_out = os.path.join(sample_dir, "single_levels.grib")
    pressure_out = os.path.join(sample_dir, "pressure_levels.grib")

    tmp_single = single_out + ".tmp"
    tmp_pressure = pressure_out + ".tmp"

    # if not os.path.exists(single_out):
    _subset_and_save(
        single_levels_file, tmp_single,
        sample["ERA5_SNAPSHOT_BBOX"],
        sample["ERA5_SNAPSHOT_TIME"],
    )
    shutil.move(tmp_single, single_out)

    # if not os.path.exists(pressure_out):
    _subset_and_save(
        pressure_levels_file, tmp_pressure,
        sample["ERA5_SNAPSHOT_BBOX"],
        sample["ERA5_SNAPSHOT_TIME"],
    )
    shutil.move(tmp_pressure, pressure_out)

    for tmp_file in [tmp_single, tmp_pressure]:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

def safe_grib_get(grb, key, default=None):
    try:
        return grb[key] if grb.valid_key(key) else default
    except Exception:
        return default

def _message_valid_time(grb):
    data_date = safe_grib_get(grb, "dataDate")
    data_time = safe_grib_get(grb, "dataTime")
    step_range = safe_grib_get(grb, "stepRange", "0")

    if data_date is None or data_time is None:
        return None

    base_dt = datetime.strptime(f"{data_date}{int(data_time):04d}", "%Y%m%d%H%M")

    # stepRange can be '0', '6', '0-6', etc.
    if isinstance(step_range, str) and "-" in step_range:
        step_hours = int(step_range.split("-")[-1])
    else:
        step_hours = int(step_range)

    return base_dt + timedelta(hours=step_hours)

def _subset_and_save(grib_path, out_path, sample_bbox, sample_time):
    north, west, south, east = sample_bbox

    assert -90 <= south < north <= 90
    
    valid_lon_180 = -180 <= west < east <= 180
    valid_lon_360 = 0 <= west < east <= 360
    assert (valid_lon_180 or valid_lon_360)

    hh, mm = map(int, sample_time["time"].split(":"))
    target_dt = datetime(
        int(sample_time["year"]),
        int(sample_time["month"]),
        int(sample_time["day"]),
        hh,
        mm,
    )

    with open(out_path, "wb") as f:
        grbs = pygrib.open(grib_path)
        try:
            wrote_any = False

            for grb in grbs:
                if _message_valid_time(grb) != target_dt:
                    continue

                lats_full, lons_full = grb.latlons()
                if lats_full.ndim != 2 or lons_full.ndim != 2:
                    raise ValueError(
                        f"Expected 2D lat/lon grid for message {grb.messagenumber} in {grib_path}"
                    )

                lat_values = lats_full[:, 0]
                lon_values = lons_full[0, :]

                lat_min, lat_max = float(lat_values.min()), float(lat_values.max())
                lon_min, lon_max = float(lon_values.min()), float(lon_values.max())

                # keep your tolerance logic
                try:
                    assert lat_min <= south < north <= lat_max, (
                        f"Sample latitudes {south}, {north} out of bounds "
                        f"({lat_min}, {lat_max}) in {grib_path}"
                    )
                except AssertionError:
                    assert lat_min - 0.25 <= south < north <= lat_max + 0.25, (
                        f"Sample latitudes {south}, {north} out of bounds "
                        f"({lat_min}, {lat_max}) in {grib_path}"
                    )

                try:
                    assert lon_min <= west < east <= lon_max, (
                        f"Sample longitudes {west}, {east} out of bounds "
                        f"({lon_min}, {lon_max}) in {grib_path}"
                    )
                except AssertionError:
                    assert lon_min - 0.25 <= west < east <= lon_max + 0.25, (
                        f"Sample longitudes {west}, {east} out of bounds "
                        f"({lon_min}, {lon_max}) in {grib_path}"
                    )

                # subset on this message's own grid
                sub_values, sub_lats, sub_lons = grb.data(
                    lat1=south,
                    lat2=north,
                    lon1=west,
                    lon2=east,
                )

                if sub_values.size == 0:
                    raise ValueError(
                        f"Sample bbox {sample_bbox} produced an empty selection "
                        f"for message {grb.messagenumber} in {grib_path}"
                    )

                if sub_values.ndim != 2:
                    raise ValueError(
                        f"Expected 2D subset for message {grb.messagenumber}, "
                        f"got shape {sub_values.shape} in {grib_path}"
                    )

                ny, nx = sub_values.shape

                grb["values"] = sub_values
                grb["Ni"] = nx
                grb["Nj"] = ny
                grb["latitudeOfFirstGridPointInDegrees"] = float(sub_lats[0, 0])
                grb["longitudeOfFirstGridPointInDegrees"] = float(sub_lons[0, 0])
                grb["latitudeOfLastGridPointInDegrees"] = float(sub_lats[-1, -1])
                grb["longitudeOfLastGridPointInDegrees"] = float(sub_lons[-1, -1])

                if nx > 1:
                    dx = ((float(sub_lons[0, 1]) - float(sub_lons[0, 0]) + 180) % 360) - 180
                    grb["iDirectionIncrementInDegrees"] = dx
                if ny > 1:
                    dy = abs(float(sub_lats[1, 0]) - float(sub_lats[0, 0]))
                    grb["jDirectionIncrementInDegrees"] = dy

                f.write(grb.tostring())
                wrote_any = True

        finally:
            grbs.close()

    if not wrote_any:
        raise ValueError(
            f"No GRIB messages in {grib_path} match valid_time={target_dt}"
        )