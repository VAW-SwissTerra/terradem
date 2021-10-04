"""Tools to calculate mass balance and convert appropriately from volume."""
from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from tqdm import tqdm

import terradem.dem_tools
import terradem.files
import terradem.metadata


def read_mb_index() -> pd.DataFrame:

    data = pd.read_csv(
        terradem.files.INPUT_FILE_PATHS["massbalance_index"],
        delim_whitespace=True,
        skiprows=2,
        index_col=0,
    )
    data.index.name = "year"

    return data


def match_zones() -> Callable[[float, float, float, float], tuple[float, str]]:
    standard_start_year = 1930
    standard_end_year = 2020
    mb = read_mb_index().cumsum()

    standard_mb = pd.Series(
        index=mb.columns,
        data=np.diff(mb.T[[standard_start_year, standard_end_year]], axis=1).ravel(),
    )

    zones = sorted(mb.columns, key=lambda x: len(x), reverse=True)

    lk50_outlines = gpd.read_file(terradem.files.INPUT_FILE_PATHS["lk50_outlines"])

    for zone in zones:
        matches = []
        for i, character in enumerate(zone):
            matches.append(lk50_outlines[f"RivLevel{i}"] == str(character))

        all_matches = np.all(matches, axis=0)

        lk50_outlines.loc[all_matches, "zone"] = zone

    # Zone A55 is not covered by the zones, so hardcode this to be A54 instead.
    lk50_outlines.loc[
        (lk50_outlines["RivLevel0"] == "A") & (lk50_outlines["RivLevel1"] == "5") & (lk50_outlines["RivLevel2"] == "5"),
        "zone",
    ] = "A54"

    lk50_outlines["easting"] = lk50_outlines.geometry.centroid.x
    lk50_outlines["northing"] = lk50_outlines.geometry.centroid.y

    def get_mb_factor(easting: float, northing: float, start_year: float, end_year: float) -> tuple[float, str]:

        # Calculate the distance between the point and each lk50_outline centroid
        distance = np.linalg.norm(
            [lk50_outlines["easting"] - easting, lk50_outlines["northing"] - northing],
            axis=0,
        )

        # Find the closest lk50 outline
        min_distance_idx = np.argwhere(distance == distance.min()).ravel()[0]

        # Extract the representative zone for the closest lk50 outline.
        mb_zone = lk50_outlines.iloc[min_distance_idx]["zone"]

        # Calculate the mass balance of that zone for the given start and end year
        actual_mb = mb.loc[int(end_year), mb_zone] - mb.loc[int(start_year), mb_zone]

        # Calculate the conversion factor to the standard_start_year--standard_end_year
        factor = standard_mb[mb_zone] / actual_mb

        return factor, zone

    return get_mb_factor


def get_volume_change() -> None:

    glacier_indices_ds = rio.open(terradem.files.TEMP_FILES["lk50_rasterized"])

    ddem_versions = {
        "non_interp": terradem.files.TEMP_FILES["ddem_coreg_tcorr"],
        "norm-regional-national": terradem.files.TEMP_FILES["ddem_coreg_tcorr_national-interp-extrap"],
        "norm-regional-sgi1-subregion": terradem.files.TEMP_FILES["ddem_coreg_tcorr_subregion1-interp-extrap"],
        "norm-regional-sgi0-subregion": terradem.files.TEMP_FILES["ddem_coreg_tcorr_subregion0-interp-extrap"],
    }

    output = pd.DataFrame(
        index=ddem_versions.keys(), columns=["mean", "median", "std", "area", "volume_change", "coverage"]
    )

    print("Reading glacier mask")
    glacier_mask = glacier_indices_ds.read(1, masked=True).filled(0) > 0
    total_area = np.count_nonzero(glacier_mask) * (glacier_indices_ds.res[0] * glacier_indices_ds.res[1])

    for key in tqdm(ddem_versions):
        ddem_ds = rio.open(ddem_versions[key])
        ddem_values = ddem_ds.read(1, masked=True).filled(np.nan)[glacier_mask]

        output.loc[key] = {
            "mean": np.nanmean(ddem_values),
            "median": np.nanmedian(ddem_values),
            "std": np.nanstd(ddem_values),
            "area": total_area,
            "volume_change": np.nanmean(ddem_values) * total_area,
            "coverage": np.count_nonzero(np.isfinite(ddem_values)) / np.count_nonzero(glacier_mask),
        }

    print(output)

    output.to_csv("temp/volume_change.csv")


def get_corrections():
    standard_start_year = 1930
    standard_end_year = 2020
    mb_index = read_mb_index().cumsum()

    dirpath = pathlib.Path(terradem.files.TEMP_SUBDIRS["tcorr_meta_coreg"])

    data_list: list[dict[str, Any]] = []
    for filepath in dirpath.iterdir():
        with open(filepath) as infile:
            data = json.load(infile)

        data["station"] = filepath.stem

        data_list.append(data)

    corrections = pd.DataFrame(data_list).set_index("station")
    corrections["start_date"] = pd.to_datetime(corrections["start_date"])

    for zone, data in corrections.groupby("sgi_zone", as_index=False):
        corrections.loc[data.index, "masschange_standard"] = (
            mb_index.loc[standard_start_year, zone] - mb_index.loc[standard_end_year, zone]
        )

        corrections.loc[data.index, "masschange_actual"] = (
            mb_index.loc[data["start_date"].dt.year.values, zone].values
            - mb_index.loc[data["end_year"].astype(int), zone].values
        )

    def get_masschanges(easting: float, northing: float) -> tuple[float, float]:
        distances = np.argmin(
            np.linalg.norm([corrections["easting"] - easting, corrections["northing"] - northing], axis=0)
        )
        return corrections.iloc[distances]["masschange_standard"], corrections.iloc[distances]["masschange_actual"]

    return get_masschanges


def temporal_corr_error_model():
    stochastic_yearly_error = 0.2  # m/a w.e.

    masschange_model = get_corrections()

    def error_model(easting: float, northing: float):

        standard, actual = masschange_model(easting, northing)

        return np.sqrt(
            (((2 * stochastic_yearly_error ** 2) / standard ** 2) + ((2 * stochastic_yearly_error ** 2) / actual ** 2))
            * (standard / actual) ** 2
        )
    return error_model
