CLIMATE_USE_METEOROLOGY_DATA = True
CLIMATE_INJECT_TIME_HOUR = True
CLIMATE_INJECT_TIME_DOY = True
CLIMATE_INJECT_TIME_YEAR = True
CLIMATE_INJECT_POSITION = True
USE_METEOROLOGY_TOKENIZER = True # False means using a Conv layer
USE_ROPE_2D = False

CLIMATE_INPUT_NUM_VARIABLES = 63
CLIMATE_METADATA_POS_NUM_VARIABLES  = 4
CLIMATE_METADATA_TIME_HOUR_NUM_VARIABLES = 2
CLIMATE_METADATA_TIME_DOY_NUM_VARIABLES = 2
CLIMATE_METADATA_TIME_YEAR_NUM_VARIABLES = 1

import os
import numpy as np
_stats_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "era5_crop_stats.npz")
with np.load(_stats_path) as f:
    ERA5_CROP_STAT = {k: f[k] for k in f.files}