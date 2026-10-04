import copy
import hashlib
import math
import random
import unicodedata
import pandas as pd
from datetime import timedelta

TEST_YEARS = [2011, 2012, 2013]

def split_train_test(samples):
    train_samples = []
    test_samples = []

    for sample in samples:
        event_years = sample["EVENT_YEARS"]

        assert all(year >= 2000 for year in event_years), (
            f"Invalid EVENT_YEARS for {sample['ERA5_FOLDER_NAME']}: {event_years}"
        )

        if any(year in TEST_YEARS for year in event_years):
            test_samples.append(sample)
        else:
            train_samples.append(sample)

    return train_samples, test_samples

def filter_by_year_range(data, year_range):
    start_year, end_year = year_range
    return [
        d for d in data
        if start_year <= d["EVENT_YEAR"] <= end_year
    ]

def sample_random_era5_snapshot_time(utc_start, utc_end):
    start_hour = utc_start.replace(minute=0, second=0, microsecond=0)
    end_hour = utc_end.replace(minute=0, second=0, microsecond=0)

    total_hours = int((end_hour - start_hour).total_seconds() // 3600)
    assert total_hours >= 0

    sampled_hour_offset = random.randint(0, total_hours)
    sampled_dt = start_hour + timedelta(hours=sampled_hour_offset)

    snapshot = {
        "year": str(sampled_dt.year),
        "month": f"{sampled_dt.month:02d}",
        "day": f"{sampled_dt.day:02d}",
        "time": f"{sampled_dt.hour:02d}:00",
    }

    return snapshot, sampled_dt

def snap_to_era5_grid(value, resolution=0.25):
    return round(value / resolution) * resolution

def normalize_longitude(lon):
    return ((lon + 180.0) % 360.0) - 180.0

def get_era5_area_from_center(latitude, longitude, bounding_box_grid_size, resolution=0.25):
    assert bounding_box_grid_size >= 1

    center_lat = snap_to_era5_grid(latitude, resolution)
    center_lon = snap_to_era5_grid(longitude, resolution)

    n = bounding_box_grid_size
    total_span = (n - 1) * resolution

    left_count = (n - 1) // 2
    right_count = (n - 1) - left_count

    north = center_lat + left_count * resolution
    south = center_lat - right_count * resolution

    # Shift box to preserve full size in latitude
    if north > 90.0:
        shift = north - 90.0
        north -= shift
        south -= shift

    if south < -90.0:
        shift = -90.0 - south
        north += shift
        south += shift

    # Safety check: preserve exact span
    actual_span = round((north - south) / resolution)
    expected_span = n - 1
    assert actual_span == expected_span, (north, south, actual_span, expected_span)

    west = center_lon - left_count * resolution
    east = center_lon + right_count * resolution

    west = normalize_longitude(west)
    east = normalize_longitude(east)

    return [north, west, south, east]

def longitude_span_in_degrees(west, east):
    span = normalize_longitude(east) - normalize_longitude(west)
    if span < 0:
        span += 360.0
    return span

def is_longitude_in_box(longitude, west, east):
    longitude = normalize_longitude(longitude)
    west = normalize_longitude(west)
    east = normalize_longitude(east)

    if west <= east:
        return west <= longitude <= east
    else:
        return longitude >= west or longitude <= east

def ceil_to_hour(dt):
    if dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
        return dt
    return dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

def floor_to_hour(dt):
    return dt.replace(minute=0, second=0, microsecond=0)

def floor_to_res(x: float, res: float) -> float:
    return math.floor(x / res) * res

def ceil_to_res(x: float, res: float) -> float:
    return math.ceil(x / res) * res

def get_target_size_and_center(bbox, bounding_box_grid_size, resolution=0.25):
    lat_min, lat_max, lon_min, lon_max = bbox

    lon_min = normalize_longitude(lon_min)
    lon_max = normalize_longitude(lon_max)

    assert -90 <= lat_min <= lat_max <= 90
    assert -180 <= lon_min <= 180
    assert -180 <= lon_max <= 180
    assert bounding_box_grid_size >= 1
    assert resolution > 0

    # snap outward
    lat_min = floor_to_res(lat_min, resolution)
    lat_max = ceil_to_res(lat_max, resolution)
    lon_min = floor_to_res(lon_min, resolution)
    lon_max = ceil_to_res(lon_max, resolution)

    lat_grids = int(round((lat_max - lat_min) / resolution)) + 1
    lon_span = longitude_span_in_degrees(lon_min, lon_max)
    lon_grids = int(round(lon_span / resolution)) + 1

    assert lon_span >= 0.

    target_grid_size = max(lat_grids, lon_grids, bounding_box_grid_size)

    center_lat = (lat_min + lat_max) / 2.0
    center_lon = normalize_longitude(lon_min + lon_span / 2.0)

    return center_lat, center_lon, target_grid_size

def filter_non_wrapping_samples(samples):
    valid_samples = []
    for sample in samples:
        north, west, south, east = sample["ERA5_SNAPSHOT_BBOX"]
        assert west != east
        assert north > south
        if west < east:
            valid_samples.append(sample)
    return valid_samples

def normalize_antimeridian_bboxes(samples):
    normalized_samples = []

    for sample in samples:
        north, west, south, east = sample["ERA5_SNAPSHOT_BBOX"]

        assert west != east
        assert north > south

        normalized_sample = copy.deepcopy(sample)

        if west < east:
            normalized_samples.append(normalized_sample)
        else:
            assert west >= 0
            assert east <= 0

            normalized_sample["ERA5_SNAPSHOT_BBOX"] = [north, west, south, east + 360]
            normalized_samples.append(normalized_sample)

    return normalized_samples

def narrative_to_int(episode_narrative):
    if pd.isna(episode_narrative):
        text = "<NULL>"
    else:
        text = unicodedata.normalize("NFC", str(episode_narrative)).strip()

    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)