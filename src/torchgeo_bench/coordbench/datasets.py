"""CoordBench dataset loaders — coordinate (lon/lat) -> label benchmark tables.

Every benchmark is read directly from the unified HuggingFace dataset repo
``taylor-geospatial/coordbench`` (one normalized parquet per config). This is the
location-encoder evaluation suite ported from the MIND codebase: point (lon, lat)
in, a downstream label out, probed with a frozen encoder.

A loader returns one or more :class:`CoordBenchmark`. ``task_type`` selects the
probe (ridge R^2 for regression, accuracy for classification). Families are
registered in :data:`FAMILY_LOADERS`; :func:`load_benchmarks` expands a selection
(``"all"``, family names, or individual benchmark names) into a flat list.
"""

import logging
import os
from collections.abc import Callable

import numpy as np
import pandas as pd
import geopandas as gpd

from torchgeo_bench.coordbench.catalog import (
    ERA5_ECMWF_LABELS,
    CDC_PLACES_MEASURES,
    CHELSA_LABELS,
    CHELSA_MONTHLY_LABELS,
    DEEPMIND_EVAL_CONFIGS,
    ELECTRIC_LOAD_VARIABLES,
    FAMILY_BENCHMARKS,
    SUSTAINBENCH_TASKS,
    USAVARS_LABELS,
    PDFM_NON_TASK,
    USAVARS_NODATA,
    USAVARS_LOG_LABELS
)

from torchgeo_bench.coordbench.benchmark import CoordBenchmark
from torchgeo_bench.coordbench.config_enums import TemporalResolution

logger = logging.getLogger(__name__)

# https://huggingface.co/datasets/taylor-geospatial/coordbench
COORDBENCH_REPO = os.environ.get("COORDBENCH_REPO", "taylor-geospatial/coordbench")
# Integrate with coordbench (right now, this is my personal repo)
COORDBENCH_EXTENSION_REPO = os.environ.get("COORDBENCH_EXTENSION_REPO", "Temporal-Coordbench") # REMOVE 

# canonical (non-task) schema columns, excluded when scanning for task columns
_CANONICAL_EXTRA = frozenset({"timestamp", "timestamp_end", "split", "id"})

def load_config(config: str) -> pd.DataFrame:
    """Read one CoordBench config's normalized parquet table from HuggingFace."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(COORDBENCH_REPO, f"data/{config}/data.parquet", repo_type="dataset")
    return pd.read_parquet(path)

def load_config_gdf(config: str, filename="data") -> gpd.GeoDataFrame:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(f"Temporal-Coordbench/{config}", filename=f"{filename}.parquet", repo_type="dataset")
    return gpd.read_parquet(path)

def load_config_temporal(config: str, columns: list[str] | None = None, filename="data") -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(f"Temporal-Coordbench/{config}", f"{filename}.parquet", repo_type="dataset")
    return pd.read_parquet(path, columns=columns)

def _lonlat_cols(df: pd.DataFrame) -> tuple[str, str]:
    lon_col = next(c for c in ("lon", "longitude", "Lon", "x") if c in df.columns)
    lat_col = next(c for c in ("lat", "latitude", "Lat", "y") if c in df.columns)
    return lon_col, lat_col

def _polygon_helper(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    df.drop(["lon", "lat"], axis=1, inplace=True)
    df["sub_sampled"] = df[col_name].map(
        lambda items: items[:10])  # hard-coded, can go up to 1k

    df.drop(col_name, axis=1, inplace=True)

    combined_df = df.explode("sub_sampled")  # results in a column of [lon, lat]
    combined_df['lon'] = combined_df["sub_sampled"].str[0].astype("float32")  # isolate lon/lat columns
    combined_df['lat'] = combined_df["sub_sampled"].str[1].astype("float32")
    combined_df.drop("sub_sampled", axis=1, inplace=True)

    return combined_df


def _first_col(df: pd.DataFrame, *names: str) -> str | None:
    return next((n for n in names if n in df.columns), None)


def load_pdfm() -> list[CoordBenchmark]:
    """PDFM conus27 — 27 US ZIP-level health/socio/environment regression tasks."""
    df = load_config("pdfm_conus27")
    lon_col, lat_col = _lonlat_cols(df)
    non_task = PDFM_NON_TASK | {lon_col, lat_col} | _CANONICAL_EXTRA
    task_cols = [c for c in df.columns if c not in non_task]
    return [
        CoordBenchmark(
            name="pdfm-conus27",
            lat=df[lat_col].to_numpy(np.float64),
            lon=df[lon_col].to_numpy(np.float64),
            tasks={c: df[c].to_numpy(np.float64) for c in task_cols},
        )
    ]


def load_air_temp() -> list[CoordBenchmark]:
    """SatCLIP air temperature — ~3000 global stations (Hooker et al. 2018)."""
    df = load_config("air_temp")
    lon_col, lat_col = _lonlat_cols(df)
    cols = {c.lower(): c for c in df.columns}
    temp_keys = [k for k in cols if "temp" in k or k == "meant"] or [
        k for k in cols if k.startswith("mean")
    ]
    temp_col = cols[temp_keys[0]]
    return [
        CoordBenchmark(
            name="satclip-air-temp",
            lat=df[lat_col].to_numpy(np.float64),
            lon=df[lon_col].to_numpy(np.float64),
            tasks={"air_temperature": df[temp_col].to_numpy(np.float64)},
        )
    ]


def load_california_housing() -> list[CoordBenchmark]:
    """California housing (sklearn) — regional median house value regression."""
    df = load_config("california_housing")
    lon_col, lat_col = _lonlat_cols(df)
    return [
        CoordBenchmark(
            name="california-housing",
            lat=df[lat_col].to_numpy(np.float64),
            lon=df[lon_col].to_numpy(np.float64),
            tasks={"median_house_value": df["MedHouseVal"].to_numpy(np.float64)},
        )
    ]


def load_satclip() -> list[CoordBenchmark]:
    """SatCLIP's 5 released downstream datasets (Klemmer et al.).

    country/ecoregion/biome (classification) + population/elevation (regression).
    Ocean nodata (country index 0, ecoregion/biome index -1) is dropped.
    """
    out: list[CoordBenchmark] = []

    c = load_config("satclip_country")
    lon_c, lat_c = _lonlat_cols(c)
    c = c[c["country"] != 0]  # index 0 = ocean/nodata
    out.append(
        CoordBenchmark(
            name="satclip-country",
            lat=c[lat_c].to_numpy(np.float64),
            lon=c[lon_c].to_numpy(np.float64),
            tasks={"country": c["country"].to_numpy()},
            task_type="classification",
        )
    )

    e = load_config("satclip_ecoregion")
    lon_e, lat_e = _lonlat_cols(e)
    e = e[e["ecoregion_class_index"] != -1]  # -1 = ocean/nodata
    for col, name in [
        ("ecoregion_class_index", "satclip-ecoregion"),
        ("biome_class_index", "satclip-biome"),
    ]:
        out.append(
            CoordBenchmark(
                name=name,
                lat=e[lat_e].to_numpy(np.float64),
                lon=e[lon_e].to_numpy(np.float64),
                tasks={col: e[col].to_numpy()},
                task_type="classification",
            )
        )

    p = load_config("satclip_population")  # target is already log_population
    lon_p, lat_p = _lonlat_cols(p)
    out.append(
        CoordBenchmark(
            name="satclip-population",
            lat=p[lat_p].to_numpy(np.float64),
            lon=p[lon_p].to_numpy(np.float64),
            tasks={"log_population": p["log_population"].to_numpy(np.float64)},
        )
    )

    el = load_config("satclip_elevation")
    lon_el, lat_el = _lonlat_cols(el)
    out.append(
        CoordBenchmark(
            name="satclip-elevation",
            lat=el[lat_el].to_numpy(np.float64),
            lon=el[lon_el].to_numpy(np.float64),
            tasks={"elevation": el["elevation"].to_numpy(np.float64)},
        )
    )
    return out


def load_sustainbench() -> list[CoordBenchmark]:
    """SustainBench DHS SDG indices — 6 socioeconomic regressions at cluster centroids.

    Uses the dataset's official trainval/test split (``test_mask``), matching the
    SustainBench protocol rather than k-fold CV.
    """
    df = load_config("sustainbench")  # merged table; split == "trainval" | "test"
    is_test = (df["split"] == "test").to_numpy()
    out: list[CoordBenchmark] = []
    for key, col in SUSTAINBENCH_TASKS.items():
        m = df[col].notna() & df["lat"].notna() & df["lon"].notna()
        sub = df[m]
        out.append(
            CoordBenchmark(
                name=f"sustainbench-{key}",
                lat=sub["lat"].to_numpy(np.float64),
                lon=sub["lon"].to_numpy(np.float64),
                tasks={col: sub[col].to_numpy(np.float64)},
                test_mask=is_test[m.to_numpy()],
            )
        )
    return out


def load_daily_usa_electric_usage(subsample_locations = False, from_polygon=False) -> list[CoordBenchmark]:

    counties = load_config_gdf("usa_electric_load","usa_counties")

    benchmarks = []
    # generates 8 datasets, each corresponding to one year with 1M daily datapoints
    for year in range(2016, 2024):
        electrical_load = load_config_temporal("usa_electric_load", filename=f"usa_electrical_load_{year}")
        combined_df = electrical_load.merge(counties, on="county", how="left")

        if subsample_locations:
            pool = combined_df["county"].drop_duplicates().sample(n=1_500, random_state=0)  # total counties is 3117
            combined_df = combined_df.merge(pool, on=["county"])

        if from_polygon:
            combined_df = _polygon_helper(combined_df, "samples")
            name = "_spatial_agg"
        else:
            name = "_from_centroid"

        tasks = {}
        # do log_norm
        for task_name in ELECTRIC_LOAD_VARIABLES:
            if task_name not in combined_df.columns:
                continue
            tasks[task_name] = combined_df[task_name].to_numpy(np.float64)
            combined_df[task_name] += 1 # to avoid div/0 for log
            tasks[f"{task_name}_log"] = np.log(combined_df[task_name].to_numpy(np.float64))


        ds = CoordBenchmark(
            name=f"usa_daily_electric_load_{year}{name}",
            lat=combined_df["lat"].to_numpy(np.float64),
            lon=combined_df["lon"].to_numpy(np.float64),
            tasks=tasks,
            # kept float64 for posix since it will lose second-level precision in float32
            posix_timestamp=combined_df["timestamp"].to_numpy(np.float64),
            temporal_resolution=TemporalResolution.DAILY,
            spatial_aggregation_key=("county", combined_df["county"])
        )

        benchmarks.append(ds)

    return benchmarks


def load_weekly_usa_electric_usage(subsample_locations = False, from_polygon=False) -> list[CoordBenchmark]:
    counties = load_config_gdf("usa_electric_load", filename="usa_counties")
    electrical_load = load_config_temporal("usa_electric_load", filename=f"weekly_usa_electrical_load_2016_2023")
    combined_df = electrical_load.merge(counties, on="county", how="left")

    if subsample_locations:
        pool = combined_df["county"].drop_duplicates().sample(n=1_500, random_state=0) #total counties is 3117
        combined_df = combined_df.merge(pool, on=["county"])

    if from_polygon:
        combined_df = _polygon_helper(combined_df, "samples")
        name = "_spatial_agg"
    else:
        name = "_from_centroid"

    tasks = {}
    # do log_norm
    for task_name in ELECTRIC_LOAD_VARIABLES:
        if task_name not in combined_df.columns:
            continue
        tasks[task_name] = combined_df[task_name].to_numpy(np.float64) #remove if log norm is the better task
        combined_df[task_name] += 1
        tasks[f"{task_name}_log"] = np.log(combined_df[task_name].to_numpy(np.float64))

    ds = CoordBenchmark(
        name=f"usa_weekly_electric_load{name}",
        lat=combined_df["lat"].to_numpy(np.float64),
        lon=combined_df["lon"].to_numpy(np.float64),
        tasks=tasks,
        # kept float64 for posix since it will lose second-level precision in float32
        posix_timestamp=combined_df["start_timestamp"].to_numpy(np.float64),
        end_timestamp=combined_df["end_timestamp"].to_numpy(np.float64),
        temporal_resolution=TemporalResolution.WEEKLY,
        spatial_aggregation_key=("county", combined_df["county"])
    )

    return [ds]


def load_usa_census_data(subsample_locations: bool = False, from_polygon=False) -> list[CoordBenchmark]:
    dfs = []

    for dates in ["2010_2019", "2020_2025"]:
        counties = load_config_temporal("usa_census_data_2010_2025", filename=f"counties_{dates}")
        data = load_config_temporal("usa_census_data_2010_2025", filename=f"usa_census_{dates}")
        combined = data.merge(counties, on=["NAMELSAD", "STATEFP"], how="left")
        dfs.append(combined)

    combined_df = pd.concat(dfs, ignore_index=True)

    if subsample_locations:
        pool = combined_df["county"].drop_duplicates().sample(n=1_500, random_state=0)  # total counties is 3117
        combined_df = combined_df.merge(pool, on=["county"])

    #if we want to do spatial sampling/pooling
    if from_polygon:
        combined_df = _polygon_helper(combined_df, "sampled")
        name = "_spatial_agg"
    else:
        name = "_from_centroid"

    #otherwise, default to lat/lon in file that refer to centroid

    combined_df["log_population"] = np.log(combined_df["population"].to_numpy(np.float64))

    bench = CoordBenchmark(
        name=f"usa_yearly_census{name}",
        lat=combined_df["lat"].to_numpy(np.float64),
        lon=combined_df["lon"].to_numpy(np.float64),
        tasks={"log_population": combined_df["log_population"].to_numpy(np.float64)},
        # kept float64 for posix since it will lose second-level precision in float32
        posix_timestamp=combined_df["start_timestamp"].to_numpy(np.float64),
        end_timestamp=combined_df["end_timestamp"].to_numpy(np.float64),
        temporal_resolution=TemporalResolution.YEARLY,
        spatial_aggregation_key=("county", combined_df["county"].to_numpy())
    )

    return [bench]


def load_global_population(subsample_locations: bool = False) -> list[CoordBenchmark]:

    benchmarks = []
    for land_mass in ["africa", "europe", "asia", "oceania", "north_america", "south_america", "global"]:
        df = load_config_temporal("world_population", filename=f"{land_mass}_2015_to_2025")
        if subsample_locations:
            df = df.sample(n=10_000, random_state=42)

        df["population"] += 1 # to avoid div/0 issues when taking log

        result = CoordBenchmark(
            name=f"global_population_{land_mass}",
            lat=df["lat"].to_numpy(np.float64),
            lon=df["lon"].to_numpy(np.float64),
            tasks={"log_population": np.log(df["population"].to_numpy(np.float64))},
            posix_timestamp=df["start_timestamp"].to_numpy(np.float64),
            end_timestamp=df["end_timestamp"].to_numpy(np.float64),
            temporal_resolution=TemporalResolution.YEARLY,
        )
        benchmarks.append(result)

    return benchmarks


def load_chelsa_monthly(subsample_locations: bool = False) -> list[CoordBenchmark]:
    """Monthly CHELSA 1km resolution climate data from 2011 to 2018"""
    df = load_config_temporal("chelsa_monthly")

    if subsample_locations:
        pool = df[["lat", "lon"]].drop_duplicates().sample(n=10_000, random_state=0)
        df = df.merge(pool, on=["lat", "lon"])

    #TODO: vpd has a sensor error for a timestep somewhere in 2013,
    # and everything jumps out in huge way against that
    tasks = {}
    for task_name in CHELSA_MONTHLY_LABELS:
        if task_name not in df.columns:
            continue

        if task_name in ["pet", "cmi"]:
            df[task_name] = df[task_name] + 1 # for log norm
            tasks[f"{task_name}_log"] = np.log(df[task_name].to_numpy(np.float64))
        else:
            tasks[task_name] = df[task_name].to_numpy(np.float64)


    monthly = CoordBenchmark(
        name="chelsa",
        lat=df["lat"].to_numpy(np.float32),
        lon=df["lon"].to_numpy(np.float32),
        tasks=tasks,
        # kept float64 for posix since it will lose second-level precision in float32
        posix_timestamp=df["start_timestamp"].to_numpy(np.float64),
        end_timestamp=df["end_timestamp"].to_numpy(np.float64),
        temporal_resolution=TemporalResolution.MONTHLY,
    )
    return [monthly]


def load_better_together() -> list[CoordBenchmark]:
    """Better Together (van der Plas et al.) — 6 coordinate -> label tasks.

    dw/bioclim/footprint share one 10k random-sample pool joined by ``id``; this
    restricts bioclim/human_footprint to that pool, matching the original protocol.
    """
    out: list[CoordBenchmark] = []

    ch = load_config("bt_cropharvest")
    out.append(
        CoordBenchmark(
            name="bt-cropharvest",
            lat=ch["lat"].to_numpy(np.float64),
            lon=ch["lon"].to_numpy(np.float64),
            tasks={"crop_type": ch["label_name"].to_numpy()},
            task_type="classification",
        )
    )

    bm = load_config("bt_biomass")  # biomass_mean is clean; biomass_center has -9999 sentinels
    m = bm["biomass_mean"].notna()
    out.append(
        CoordBenchmark(
            name="bt-biomass",
            lat=bm.loc[m, "lat"].to_numpy(np.float64),
            lon=bm.loc[m, "lon"].to_numpy(np.float64),
            tasks={"biomass": bm.loc[m, "biomass_mean"].to_numpy(np.float64)},
        )
    )

    dw = load_config("bt_landcover")
    dw = dw[dw["random_sample"] == 1]  # the 10k pool the paper probes
    lc_cols = [
        "water",
        "trees",
        "grass",
        "flooded_vegetation",
        "crops",
        "shrub_and_scrub",
        "built",
        "bare",
        "snow_and_ice",
    ]
    out.append(
        CoordBenchmark(
            name="bt-landcover",
            lat=dw["lat"].to_numpy(np.float64),
            lon=dw["lon"].to_numpy(np.float64),
            tasks={c: dw[c].to_numpy(np.float64) for c in lc_cols},  # fractional cover
        )
    )

    pool_ids = set(dw["id"])
    bio = load_config("bt_bioclim")
    bio = bio[bio["id"].isin(pool_ids)]
    bio_cols = [f"bioclim_{i:02d}" for i in range(1, 20)]
    out.append(
        CoordBenchmark(
            name="bt-bioclim",
            lat=bio["lat"].to_numpy(np.float64),
            lon=bio["lon"].to_numpy(np.float64),
            tasks={c: bio[c].to_numpy(np.float64) for c in bio_cols},
        )
    )

    hf = load_config("bt_human_footprint")
    hf = hf[hf["id"].isin(pool_ids)]
    out.append(
        CoordBenchmark(
            name="bt-population",
            lat=hf["lat"].to_numpy(np.float64),
            lon=hf["lon"].to_numpy(np.float64),
            tasks={"pop_density": hf["pop_density"].to_numpy(np.float64)},
        )
    )
    out.append(
        CoordBenchmark(
            name="bt-distroad",
            lat=hf["lat"].to_numpy(np.float64),
            lon=hf["lon"].to_numpy(np.float64),
            tasks={"maxdist_road": hf["maxdist_road"].to_numpy(np.float64)},  # clipped at 50km
        )
    )
    return out


def load_cdc_places() -> list[CoordBenchmark]:
    """CDC PLACES 2023 ZCTA health/behavior measures — one regression per measure."""
    df = load_config("cdc_places")  # already lowercased + lon/lat parsed at build time
    out: list[CoordBenchmark] = []
    for key, pref in CDC_PLACES_MEASURES.items():
        col = f"{pref.lower()}_crudeprev"
        if col not in df.columns:
            logger.warning("[cdc_places] missing column %s; skipping %s", col, key)
            continue
        m = df[col].notna() & df["lat"].notna() & df["lon"].notna()
        sub = df[m]
        out.append(
            CoordBenchmark(
                name=f"places-{key}",
                lat=sub["lat"].to_numpy(np.float64),
                lon=sub["lon"].to_numpy(np.float64),
                tasks={col: sub[col].to_numpy(np.float64)},
            )
        )
    return out


def load_usavars() -> list[CoordBenchmark]:
    """MOSAIKS/USAVars — one regression benchmark per label (7 total)."""
    out: list[CoordBenchmark] = []
    skip = {"id", "lat", "lon"} | _CANONICAL_EXTRA
    for label in USAVARS_LABELS:
        df = load_config(f"usavars_{label}")
        df.columns = [str(c).strip() for c in df.columns]
        cl = {c.lower(): c for c in df.columns}
        cands = [
            c
            for c in df.columns
            if c.lower() not in skip
            and not c.startswith("Unnamed")
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not cands or not ({"lat", "lon"} <= set(cl)):
            continue
        value_col = next((c for c in cands if label in c.lower()), None)
        value_col = value_col or next((c for c in cands if c.lower() == "price"), None) or cands[0]
        lat_a = df[cl["lat"]].to_numpy(np.float64)
        lon_a = df[cl["lon"]].to_numpy(np.float64)
        val_a = df[value_col].to_numpy(np.float64)
        keep = val_a != USAVARS_NODATA
        lat_a, lon_a, val_a = lat_a[keep], lon_a[keep], val_a[keep]
        if label in USAVARS_LOG_LABELS:
            val_a = np.log1p(val_a)
        out.append(
            CoordBenchmark(name=f"mosaiks-{label}", lat=lat_a, lon=lon_a, tasks={label: val_a})
        )
    return out


def load_country() -> list[CoordBenchmark]:
    """Global country classification — 30k random land points, Natural Earth admin-0."""
    df = load_config("country")
    return [
        CoordBenchmark(
            name="country",
            lat=df["lat"].to_numpy(np.float64),
            lon=df["lon"].to_numpy(np.float64),
            tasks={"country": df["country"].to_numpy()},
            task_type="classification",
        )
    ]


def load_ecoregions() -> list[CoordBenchmark]:
    """Global biome + ecoregion classification — RESOLVE Ecoregions2017, 30k points."""
    df = load_config("ecoregions")
    return [
        CoordBenchmark(
            name="ecoregions",
            lat=df["lat"].to_numpy(np.float64),
            lon=df["lon"].to_numpy(np.float64),
            tasks={"biome": df["biome"].to_numpy(), "ecoregion": df["ecoregion"].to_numpy()},
            task_type="classification",
        )
    ]


def load_worldclim() -> list[CoordBenchmark]:
    """WorldClim v2.1 bioclim — annual mean temperature (bio1) + precipitation (bio12)."""
    df = load_config("worldclim_bio")
    return [
        CoordBenchmark(
            name=f"worldclim-bio{v}",
            lat=df["lat"].to_numpy(np.float64),
            lon=df["lon"].to_numpy(np.float64),
            tasks={f"bio{v}": df[f"bio{v}"].to_numpy(np.float64)},
        )
        for v in (1, 12)
        if f"bio{v}" in df.columns
    ]


def load_soilgrids() -> list[CoordBenchmark]:
    """ISRIC SoilGrids v2 — soil organic carbon (soc) + pH (phh2o) regression."""
    df = load_config("soilgrids")
    return [
        CoordBenchmark(
            name=f"soilgrids-{p}",
            lat=df["lat"].to_numpy(np.float64),
            lon=df["lon"].to_numpy(np.float64),
            tasks={p: df[p].to_numpy(np.float64)},
        )
        for p in ("soc", "phh2o")
        if p in df.columns
    ]

def load_viirs_night_lights(subsample_locations:bool = False) -> list[CoordBenchmark]:
    df = load_config_temporal("viirs_night_lights")

    if subsample_locations:
        pool = df[["lat", "lon"]].drop_duplicates().sample(n=10_000, random_state=0)
        df = df.merge(pool, on=["lat", "lon"])

    task = CoordBenchmark(
        name="viirs_night_lights",
        lat=df["lat"].to_numpy(np.float32),
        lon=df["lon"].to_numpy(np.float32),
        posix_timestamp=df["start_timestamp"].to_numpy(np.float64),
        end_timestamp=df["end_timestamp"].to_numpy(np.float64),
        tasks={"night_lights": df["night_lights"].to_numpy(np.float64)},
        temporal_resolution=TemporalResolution.YEARLY
    )

    return [task]

def load_era5_ecmwf(subsample_locations: bool = False) -> list[CoordBenchmark]:
    """"""
    df = load_config_temporal("era5_ecmwf_2017_daily")
    task_cols = [v for v in ERA5_ECMWF_LABELS if v in df.columns]
    df = df[["lat", "lon", "posix_timestamp", *task_cols]]
    if subsample_locations:
        pool = df[["lat", "lon"]].drop_duplicates().sample(n=10_000, random_state=0)
        df = df.merge(pool, on=["lat", "lon"])

    daily = CoordBenchmark(
        name="era5_ecmwf",
        lat=df["lat"].to_numpy(np.float32),
        lon=df["lon"].to_numpy(np.float32),
        tasks={v: df[v].to_numpy(np.float32) for v in task_cols},
        # kept float64 for posix since it will lose second-level precision in float32
        posix_timestamp=df["posix_timestamp"].to_numpy(np.float64),
        temporal_resolution=TemporalResolution.DAILY,
        year=2017
    )
    return [daily]

def load_chelsa_daily(subsample_locations: bool = False) -> list[CoordBenchmark]:
    """"""
    df = load_config_temporal("chelsa_2017_daily")
    task_cols = [v for v in CHELSA_LABELS if v in df.columns]
    df = df[["lat", "lon", "posix_timestamp", *task_cols]]
    if subsample_locations:
        pool = df[["lat", "lon"]].drop_duplicates().sample(n=10_000, random_state=0)
        df = df.merge(pool, on=["lat", "lon"])

    daily = CoordBenchmark(
        name="chelsa",
        lat=df["lat"].to_numpy(np.float32),
        lon=df["lon"].to_numpy(np.float32),
        tasks={v: df[v].to_numpy(np.float32) for v in task_cols},
        # kept float64 for posix since it will lose second-level precision in float32
        posix_timestamp=df["posix_timestamp"].to_numpy(np.float64),
        temporal_resolution=TemporalResolution.DAILY,
        year=2017
    )
    return [daily]


def load_deepmind() -> list[CoordBenchmark]:
    """DeepMind/AlphaEarth geospatial eval suite — one benchmark per eval config.

    Classification vs regression is inferred (integer label + few classes ->
    classification). Carries per-point POSIX timestamp (used to be year) from the 
    valid-time-start timestamp.
    """
    out: list[CoordBenchmark] = []
    for stem in DEEPMIND_EVAL_CONFIGS:
        df = load_config(f"dm_{stem}")
        if "label" not in df.columns:
            continue
        lon_col, lat_col = _lonlat_cols(df)
        ts_col = _first_col(df, "timestamp", "valid_time_start_ms")
        label = df["label"].to_numpy()
        integral = np.all(np.isfinite(label)) and np.allclose(label, np.round(label))
        is_clf = bool(integral and np.unique(label).size <= 100)
        # CoordBench always carries a `timestamp` column, all-null when the source has no
        # per-point time; keep posix_timestamp None in that case rather than an all-NaN array.
        # (used to be year)
        posix_timestamp = None
        if ts_col is not None:
            pts = pd.to_datetime(df[ts_col], unit="ms")

            if pts.notna().any():
                posix_timestamp = pts.to_numpy().astype("datetime64[s]").astype("int64")
        out.append(
            CoordBenchmark(
                name=f"dm-{stem}",
                lat=df[lat_col].to_numpy(np.float64),
                lon=df[lon_col].to_numpy(np.float64),
                tasks={stem: label.astype(np.int64) if is_clf else label.astype(np.float64)},
                task_type="classification" if is_clf else "regression",
                posix_timestamp=posix_timestamp,
            )
        )
    return out

def load_ghcn_daily(subsample_locations: bool = True) -> list[CoordBenchmark]:
    """GHCN-Daily — 26k global stations, daily temperature/precipitation from 1992 to 2024"""

    df = load_config_temporal("ghcn_daily", columns=["lat", "lon", "posix_timestamp", "tmax", "tmin", "tmean"])

    if subsample_locations:
        # Drops locations
        pool = df[["lat", "lon"]].drop_duplicates().sample(n=1_000, random_state=0)
        df = df.merge(pool, on=["lat", "lon"])

        # Further subsamples the time series to 1k per location, silly pandas groupby + sample syntax
        df = df.sample(frac=1, random_state=42).groupby(["lat", "lon"]).head(1_000)

    # from huggingface_hub import hf_hub_download
    # path = hf_hub_download(COORDBENCH_EXTENSION_REPO, f"ghcn_daily/data.parquet", repo_type="dataset")
    # df = pd.read_parquet(path, columns=["lat", "lon", "posix_timestamp", "tmax", "tmin", "tmean"])

    # df = load_config_extended("ghcn_daily")
    lon_col, lat_col = _lonlat_cols(df)
    task_cols = ["tmax", "tmin", "tmean"]
    return [
        CoordBenchmark(
            name="ghcn-daily",
            lat=df[lat_col].to_numpy(np.float64),
            lon=df[lon_col].to_numpy(np.float64),
            tasks={c: df[c].to_numpy(np.float64) for c in task_cols},
            posix_timestamp=df["posix_timestamp"].to_numpy(np.float64),
        )
    ]

# Family name -> loader. A family groups benchmarks that share a source/protocol.
FAMILY_LOADERS: dict[str, Callable[[], list[CoordBenchmark]]] = {
    "pdfm": load_pdfm,
    "air_temp": load_air_temp,
    "california_housing": load_california_housing,
    "satclip": load_satclip,
    "sustainbench": load_sustainbench,
    "better_together": load_better_together,
    "cdc_places": load_cdc_places,
    "usavars": load_usavars,
    "country": load_country,
    "ecoregions": load_ecoregions,
    "worldclim": load_worldclim,
    "soilgrids": load_soilgrids,
    "deepmind": load_deepmind,
    "era5_ecmwf": load_era5_ecmwf,
    "chelsa_daily": load_chelsa_daily,
    "usa_electric_load_daily": load_daily_usa_electric_usage,
    "usa_electric_load_weekly": load_weekly_usa_electric_usage,
    "chelsa_monthly": load_chelsa_monthly,
    "usa_census": load_usa_census_data,
    "global_population": load_global_population,
    "viirs_night_lights": load_viirs_night_lights,
    "ghcn_daily": load_ghcn_daily,
}

POLYGON_BENCH = [
    "usa_census",
    "usa_electric_load_daily",
    "usa_electric_load_weekly"
]

CAN_SUBSAMPLE = [
    "era5_ecmwf",
    "usa_electric_load_daily",
    "usa_electric_load_weekly",
    "chelsa_monthly",
    "usa_census",
    "global_population",
    "viirs_night_lights"
]
# Benchmark names each family emits; lets a selection load only the needed family,
# and lets callers enumerate the suite without a download.
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
    "chelsa": ("chelsa",),
    "ghcn_daily": ("ghcn-daily",),
}

_BENCHMARK_TO_FAMILY: dict[str, str] = {
    name: family for family, names in FAMILY_BENCHMARKS.items() for name in names
}


def list_families() -> list[str]:
    """Return the registered benchmark family names."""
    return list(FAMILY_LOADERS)


def list_benchmarks() -> list[str]:
    """Return every individual benchmark name, in family/registration order."""
    return [name for family in FAMILY_LOADERS for name in FAMILY_BENCHMARKS[family]]


def load_benchmarks(names: str | list[str] = "all", from_polygon=False, should_subsample=False) -> list[CoordBenchmark]:
    """Expand a selection into a flat list of :class:`CoordBenchmark`.

    Args:
        names: ``"all"``, a comma-separated string, or a list. Each entry is a
            family name (loads every benchmark in it) or an individual benchmark
            name (loads only its family, then keeps just that benchmark). Only
            the families needed to satisfy the selection are fetched.
        from_polygon: False, denotes whether this dataset can/should perform spatial pooling
        should_subsample: False, denotes whether this dataset should be subsampled down; only available for some spatiotemporal datasets

    Returns:
        Flat list of benchmarks, deduplicated by name, in family order.
    """
    if isinstance(names, str):
        selection = (
            list(FAMILY_LOADERS)
            if names == "all"
            else [n.strip() for n in names.split(",") if n.strip()]
        )
    else:
        selection = list(names)

    families_to_load: list[str] = []
    name_filter: dict[str, set[str]] = {}
    for entry in selection:
        if entry in FAMILY_LOADERS:
            family = entry
            name_filter[family] = set()  # empty filter -> keep all of the family
        elif entry in _BENCHMARK_TO_FAMILY:
            family = _BENCHMARK_TO_FAMILY[entry]
            if family not in name_filter or name_filter[family]:
                name_filter.setdefault(family, set()).add(entry)
        else:
            raise KeyError(
                f"unknown coordbench selection {entry!r}; not a family "
                f"({sorted(FAMILY_LOADERS)}) or benchmark name"
            )
        if family not in families_to_load:
            families_to_load.append(family)

    out: list[CoordBenchmark] = []
    seen: set[str] = set()
    for family in families_to_load:
        wanted = name_filter.get(family, set())
        if family in POLYGON_BENCH and family in CAN_SUBSAMPLE:
            loaded = FAMILY_LOADERS[family](from_polygon  = from_polygon, subsample_locations = should_subsample)
        elif family in POLYGON_BENCH:
            loaded = FAMILY_LOADERS[family](from_polygon = from_polygon)
        elif family in CAN_SUBSAMPLE:
            loaded = FAMILY_LOADERS[family](subsample_locations = should_subsample)
        else:
            loaded = FAMILY_LOADERS[family]()
        for bench in loaded:
            if bench.name in seen or (wanted and bench.name not in wanted):
                continue
            seen.add(bench.name)
            out.append(bench)
    return out