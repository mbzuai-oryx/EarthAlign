SPECIAL_NEAREST_VARIABLES = ["mwd", "lsm"] # Variables to interpolate with nearest instead of the default method.

SINGLE_LEVEL_ATMOS_VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "instantaneous_10m_wind_gust",
    "mean_sea_level_pressure",
    "total_precipitation",
    "convective_precipitation",
    "snowfall",
    "snow_depth",
    "snowmelt",
    "runoff",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "convective_available_potential_energy",
    "convective_inhibition",
    "boundary_layer_height",
    "cloud_base_height",
    "total_cloud_cover",
    "zero_degree_level",
    "total_column_water_vapour",
    "vertical_integral_of_eastward_water_vapour_flux",
    "vertical_integral_of_northward_water_vapour_flux",
]

SINGLE_LEVEL_WAVE_VARS = [
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_direction",
    "peak_wave_period",
]

PRESSURE_LEVEL_VARS = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "vertical_velocity",
    "vorticity",
]

PRESSURE_LEVELS = ["925", "850", "700", "500", "300"]

SINGLE_LEVEL_ATMOS_MAP = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "instantaneous_10m_wind_gust": "i10fg",
    "mean_sea_level_pressure": "msl",
    "total_precipitation": "tp",
    "convective_precipitation": "cp",
    "snowfall": "sf",
    "snow_depth": "sd",
    "snowmelt": "smlt",
    "runoff": "ro",
    "volumetric_soil_water_layer_1": "swvl1",
    "volumetric_soil_water_layer_2": "swvl2",
    "volumetric_soil_water_layer_3": "swvl3",
    "volumetric_soil_water_layer_4": "swvl4",
    "convective_available_potential_energy": "cape",
    "convective_inhibition": "cin",
    "boundary_layer_height": "blh",
    "cloud_base_height": "cbh",
    "total_cloud_cover": "tcc",
    "zero_degree_level": "deg0l",
    "total_column_water_vapour": "tcwv",
    "vertical_integral_of_eastward_water_vapour_flux": "viwve",
    "vertical_integral_of_northward_water_vapour_flux": "viwvn",
}

SINGLE_LEVEL_WAVE_MAP = {
    "significant_height_of_combined_wind_waves_and_swell": "swh",
    "mean_wave_direction": "mwd",
    "peak_wave_period": "pp1d",
}

PRESSURE_LEVEL_MAP = {
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
    "vertical_velocity": "w",
    "vorticity": "vo",
}

ALL_VARS = (
    [SINGLE_LEVEL_ATMOS_MAP[var_name] for var_name in SINGLE_LEVEL_ATMOS_VARS]
    + [SINGLE_LEVEL_WAVE_MAP[var_name] for var_name in SINGLE_LEVEL_WAVE_VARS]
    + [
        f"{PRESSURE_LEVEL_MAP[var_name]}_{pressure_level}"
        for var_name in PRESSURE_LEVEL_VARS
        for pressure_level in sorted(PRESSURE_LEVELS, key=lambda x: int(x))
    ]
)