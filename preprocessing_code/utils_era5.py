import copy
import xarray as xr
import numpy as np

from utils_era5_vars import SPECIAL_NEAREST_VARIABLES, ALL_VARS

def preprocess_grib_pressure_levels(grib_pressure_levels, level_dim="isobaricInhPa", time_coord="valid_time"):
    assert len(grib_pressure_levels) == 1, "Expected only one dataset"

    new_vars = {}

    ds = grib_pressure_levels[0]
    levels = np.sort(ds[level_dim].values)

    for var_name, da in ds.data_vars.items():
        assert level_dim in da.dims, f"Expected variable {var_name} to depend on {level_dim}"
        for lev in levels:
            lev_int = int(lev)
            assert lev_int == lev, f"Expected integer pressure levels, got {lev}"

            new_name = f"{var_name}_{lev_int}"

            # Select one pressure level and drop that dimension
            new_vars[new_name] = (
                da.sel({level_dim: lev})
                .drop_vars(level_dim)
                .reset_coords(drop=True)
            )

    assert time_coord in ds.coords, f"Expected time coordinate '{time_coord}' not found in dataset"

    coords={
        "latitude": ds["latitude"].values,
        "longitude": ds["longitude"].values,
        time_coord: ds[time_coord].values,
    }

    return xr.Dataset(new_vars, coords=coords, attrs=ds.attrs)


def preprocess_grib_single_levels(grib_single_levels, ref_ds, time_coord="valid_time", nan_fill_value=None):
    processed = []

    target_lat = ref_ds["latitude"].values
    target_lon = ref_ds["longitude"].values

    for ds in grib_single_levels:
        ds2 = copy.deepcopy(ds)

        assert "latitude" in ds2.coords
        assert "longitude" in ds2.coords

        same_grid = (
            np.array_equal(ds2["latitude"].values, target_lat) and
            np.array_equal(ds2["longitude"].values, target_lon)
        )

        if not same_grid:
            vars_with_linear = [v for v in ds2.data_vars if v not in SPECIAL_NEAREST_VARIABLES]
            vars_with_nearest = [v for v in ds2.data_vars if v in SPECIAL_NEAREST_VARIABLES]

            parts = []

            if vars_with_linear:
                parts.append(
                    ds2[vars_with_linear].interp(
                        latitude=target_lat,
                        longitude=target_lon,
                        method="linear"
                    )
                )

            if vars_with_nearest:
                parts.append(
                    ds2[vars_with_nearest].interp(
                        latitude=target_lat,
                        longitude=target_lon,
                        method="nearest"
                    )
                )

            ds2 = xr.merge(parts, compat="no_conflicts")

        assert time_coord in ds2.coords, f"Expected coordinate '{time_coord}' not found in dataset"

        assert np.array_equal(ds2["latitude"].values, target_lat), "Latitude does not match reference grid"
        assert np.array_equal(ds2["longitude"].values, target_lon), "Longitude does not match reference grid"

        if nan_fill_value is not None:
            ds2 = ds2.fillna(nan_fill_value)
        
        ds2 = xr.Dataset(
            data_vars={k: ds2[k].reset_coords(drop=True) for k in ds2.data_vars},
            coords={
                "latitude": ds2["latitude"].values,
                "longitude": ds2["longitude"].values,
                time_coord: ds2[time_coord].values,
            },
            attrs=ds2.attrs,
        )

        processed.append(ds2)

    merged = xr.merge(processed, compat="no_conflicts")

    return merged




def resize_latlon_grid(ds, n_lat, n_lon, method="linear", time_coord="valid_time"):
    """
    Resize an xarray Dataset onto a new regular lat/lon grid.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with coordinates latitude, longitude, and optionally valid_time.
    n_lat : int
        Number of latitude points in the output grid.
    n_lon : int
        Number of longitude points in the output grid.
    method : str
        Default interpolation method for normal variables.
    time_coord : str
        Name of time coordinate to preserve as scalar coordinate.

    Returns
    -------
    xr.Dataset
    """
    assert "latitude" in ds.coords, "Dataset must contain latitude coordinate"
    assert "longitude" in ds.coords, "Dataset must contain longitude coordinate"

    old_lat = ds["latitude"].values
    old_lon = ds["longitude"].values

    if len(old_lat) == n_lat and len(old_lon) == n_lon:
        return ds

    # Preserve the original latitude direction
    new_lat = np.linspace(old_lat[0], old_lat[-1], n_lat)
    new_lon = np.linspace(old_lon[0], old_lon[-1], n_lon)

    vars_with_linear = [v for v in ds.data_vars if v not in SPECIAL_NEAREST_VARIABLES]
    vars_with_nearest = [v for v in ds.data_vars if v in SPECIAL_NEAREST_VARIABLES]

    parts = []

    if vars_with_linear:
        parts.append(
            ds[vars_with_linear].interp(
                latitude=new_lat,
                longitude=new_lon,
                method=method
            )
        )

    if vars_with_nearest:
        parts.append(
            ds[vars_with_nearest].interp(
                latitude=new_lat,
                longitude=new_lon,
                method="nearest"
            )
        )

    ds_resized = xr.merge(parts, compat="no_conflicts")

    for var in ds_resized.data_vars:
        if np.issubdtype(ds_resized[var].dtype, np.floating):
            ds_resized[var] = ds_resized[var].astype("float32")

    coords = {
        "latitude": ds_resized["latitude"].values,
        "longitude": ds_resized["longitude"].values,
        time_coord: ds[time_coord].values,
    }

    ds_resized = xr.Dataset(
        data_vars={k: ds_resized[k].reset_coords(drop=True) for k in ds_resized.data_vars},
        coords=coords,
        attrs=ds.attrs,
    )

    return ds_resized

def xarray_to_tvhw(ds, time_coord="valid_time", dtype=np.float32):
    """
    Save an xarray.Dataset to a compressed .npz file with shape (T, V, H, W).

    Stored arrays:
      - x: (T, V, H, W)
      - latitude: (H,)
      - longitude: (W,)
      - valid_time: (T,)
    """
    assert "latitude" in ds.coords, "Dataset must contain 'latitude'"
    assert "longitude" in ds.coords, "Dataset must contain 'longitude'"
    assert time_coord in ds.coords, f"Dataset must contain '{time_coord}'"

    ds2 = ds
    
    # If valid_time is a scalar coord, promote it to a dimension of length 1
    if time_coord not in ds2.dims:
        ds2 = ds2.expand_dims({time_coord: [ds2[time_coord].values]})

    assert len(ds2.data_vars) == len(ALL_VARS), f"Expected {len(ALL_VARS)} variables, found {len(ds2.data_vars)}"

    arrays = []
    for var in ALL_VARS:
        da = ds2[var]

        # Ensure dimension order is (T, H, W)
        da = da.transpose(time_coord, "latitude", "longitude")
        arrays.append(da.values.astype(dtype, copy=False))

    # Stack variables on axis=1 -> (T, V, H, W)
    x = np.stack(arrays, axis=1)

    payload = {
        "x": x,
        "latitude": ds2["latitude"].values,
        "longitude": ds2["longitude"].values,
        time_coord: ds2[time_coord].values,
    }

    # np.savez_compressed(out_path, **payload)
    assert np.all(np.diff(payload["latitude"]) < 0), 'payload["latitude"] must be strictly decreasing'
    dlon = (np.diff(payload["longitude"]) + 180) % 360 - 180
    assert np.all(dlon > 0), 'longitude must be strictly increasing (accounting for wraparound)'

    return payload