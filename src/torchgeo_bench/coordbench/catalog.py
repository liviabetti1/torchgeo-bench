"""Coordinate benchmark names shared by lightweight validation and table loaders."""

DEEPMIND_EVAL_CONFIGS = (
    "africa_crop_mask",
    "aster_ged",
    "canada_crops_coarse",
    "canada_crops_fine",
    "descals",
    "ethiopia_crops",
    "glance",
    "lcmap_lc",
    "lcmap_lcc",
    "lcmap_lu",
    "lcmap_luc",
    "lucas_lc",
    "lucas_lu",
    "openet_ensemble",
    "us_trees",
)

USAVARS_LABELS = (
    "treecover",
    "elevation",
    "population",
    "nightlights",
    "income",
    "roads",
    "housing",
)

SUSTAINBENCH_TASKS = {
    "asset": "asset_index",
    "water": "water_index",
    "sanitation": "sanitation_index",
    "child_mortality": "under5_mort",
    "women_edu": "women_edu",
    "women_bmi": "women_bmi",
}

ERA5_ECMWF_LABELS = (
    "d2m",
    "t2m",
    "stl1",
    "stl2",
    "stl3",
    "stl4",
    "swvl1",
    "swvl2",
    "swvl3",
    "swvl4",
    "ssrd",
    "strd",
    "sde",
    "snowc",
    "u10",
    "v10",
    "sp",
    "tp",
    "skt",
)

CHELSA_LABELS = (
    "hurs",
    "pr",
    "ps",
    "rsds",
    "sfcWind",
    "tas",
    "tasmin",
    "tasmax",
    "tz",
    "we",
)

ELECTRIC_LOAD_VARIABLES = (
    "mean",
    "min",
    "max",
    "median",
)

CDC_PLACES_MEASURES = {
    "phys_health": "PHLTH",
    "diabetes": "DIABETES",
    "copd": "COPD",
    "cancer": "CANCER",
    "chd": "CHD",
    "mental_health": "MHLTH",
    "checkup": "CHECKUP",
    "sleep_lt7": "SLEEP",
    "asthma": "CASTHMA",
    "obesity": "OBESITY",
    "smoking": "CSMOKING",
    "high_chol": "HIGHCHOL",
}

SUSTAINBENCH_TASKS = {
    "asset": "asset_index",
    "water": "water_index",
    "sanitation": "sanitation_index",
    "child_mortality": "under5_mort",
    "women_edu": "women_edu",
    "women_bmi": "women_bmi",
}

FAMILY_BENCHMARKS: dict[str, tuple[str, ...]] = {
    "pdfm": ("pdfm-conus27",),
    "air_temp": ("satclip-air-temp",),
    "california_housing": ("california-housing",),
    "satclip": (
        "satclip-country",
        "satclip-ecoregion",
        "satclip-biome",
        "satclip-population",
        "satclip-elevation",
    ),
    "sustainbench": tuple(f"sustainbench-{k}" for k in SUSTAINBENCH_TASKS),
    "better_together": (
        "bt-cropharvest",
        "bt-biomass",
        "bt-landcover",
        "bt-bioclim",
        "bt-population",
        "bt-distroad",
    ),
    "cdc_places": tuple(f"places-{k}" for k in CDC_PLACES_MEASURES),
    "usavars": tuple(f"mosaiks-{label}" for label in USAVARS_LABELS),
    "country": ("country",),
    "ecoregions": ("ecoregions",),
    "worldclim": ("worldclim-bio1", "worldclim-bio12"),
    "soilgrids": ("soilgrids-soc", "soilgrids-phh2o"),
    "deepmind": tuple(f"dm-{stem}" for stem in DEEPMIND_EVAL_CONFIGS),
    "era5_ecmwf": ("era5_ecmwf",),
    "chelsa_daily": ("chelsa_daily",),
    "usa_electric_load_daily": [f"usa_daily_electric_load_{year}" for year in range(2016, 2024)],
    "usa_electric_load_weekly" : ("usa_electric_load_weekly",),
    "chelsa_monthly": ("chelsa_monthly",),
    "global_population": [f"global_population_{landmass}" for
                          landmass in ["africa", "europe", "asia", "oceania", "north_america", "south_america", "global"]],
    "viirs_night_lights": ("viirs_night_lights",),
    "usa_census": ("usa_census",),
}
