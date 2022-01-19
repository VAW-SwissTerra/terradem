"""Tools to calculate mass balance and convert appropriately from volume."""
from __future__ import annotations

import json
import os
import pathlib
import warnings
from typing import Any, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import shapely
from tqdm import tqdm

import terradem.dem_tools
import terradem.files
import terradem.metadata

ICE_DENSITY_CONVERSION = 0.85
ICE_DENSITY_ERROR = 0.06

STANDARD_START_YEAR = 1931
STANDARD_END_YEAR = 2016


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
    mb = read_mb_index().cumsum()

    standard_mb = pd.Series(
        index=mb.columns,
        data=np.diff(mb.T[[STANDARD_START_YEAR, STANDARD_END_YEAR]], axis=1).ravel(),
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

        # Calculate the conversion factor to the STANDARD_START_YEAR--STANDARD_END_YEAR
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
            mb_index.loc[STANDARD_START_YEAR, zone] - mb_index.loc[STANDARD_END_YEAR, zone]
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


def get_start_and_end_years():
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

    def get_start_and_end_year(easting: float, northing: float) -> tuple[float, float]:
        distances = np.argmin(
            np.linalg.norm([corrections["easting"] - easting, corrections["northing"] - northing], axis=0)
        )
        return (
            corrections.iloc[distances]["start_date"].year
            + corrections.iloc[distances]["start_date"].month / 12
            + corrections.iloc[distances]["start_date"].day / 364.75,
            corrections.iloc[distances]["end_year"],
        )

    return get_start_and_end_year


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


def match_sgi_ids():

    sgi_2016 = gpd.read_file(terradem.files.INPUT_FILE_PATHS["sgi_2016"])
    sgi_2016["name_lower"] = sgi_2016["name"].str.lower().fillna("")
    data_dir = pathlib.Path("data/external/mass_balance")

    warnings.filterwarnings("ignore", category=shapely.errors.ShapelyDeprecationWarning)
    result_data = []
    ids = {
        "seewijnen": "B52-22",
        "corbassiere": "B83-03",
        "murtel": "E23-16",
        "gietro": "B82-14",
        "findelen": "B56-03",
    }

    results = pd.DataFrame(columns=["sgi-id", "year", "dh", "dm"])

    for filepath in filter(lambda s: "longterm" in str(s), data_dir.iterdir()):
        name = filepath.stem.replace("_longterm", "")
        if name in ids:
            match = sgi_2016.loc[sgi_2016["sgi-id"] == ids[name]].iloc[0]
        else:
            name = {
                "ugrindelwald": "unterer grindelwald",
            }.get(name, None) or name
            try:
                match = (
                    sgi_2016[sgi_2016["name_lower"].str.findall(f".*{name}.*").apply(len) > 0]
                    .sort_values("area_km2")
                    .iloc[-1]
                )
            except IndexError:
                warnings.warn(f"Cannot find {name}")
                continue

        data = (
            pd.read_csv(filepath, skiprows=1, delim_whitespace=True, na_values=[-99.0])
            .rename(columns={"Year": "year", "B_a(mw.e.)": "dh"})
            .ffill()
        )
        data["dm"] = (data["Area(km2)"] * 1e6) * data["dh"]
        data["sgi-id"] = match["sgi-id"]

        result_data.append(data[["sgi-id", "year", "dh", "dm"]])
        continue

    results = pd.concat(result_data).set_index(["sgi-id", "year"]).squeeze()

    glacier_wise_dh = pd.read_csv(terradem.files.TEMP_FILES["glacier_wise_dh"])

    matthias_dh: pd.DataFrame = (
        results.loc[
            (results.index.get_level_values(1) >= STANDARD_START_YEAR)
            & (results.index.get_level_values(1) <= STANDARD_END_YEAR)
        ]
        .groupby(level=0)
        .cumsum()
        .groupby(level=0)
        .last()
    )

    glacier_wise_dh.index = glacier_wise_dh["sgi_id"].apply(terradem.utilities.sgi_1973_to_2016)
    glacier_wise_dh["dm_err_tonswe"] = (glacier_wise_dh["dm_tons_we"] / glacier_wise_dh["dh_m_we"]) * glacier_wise_dh[
        "dh_err_mwe"
    ]
    glacier_wise_dh.loc[glacier_wise_dh.index == "B36-26", ["dh_m_we", "dm_tons_we"]] *= 0.86   # A correction factor for including Mittelaletschgletscher

    matthias_dh = matthias_dh.merge(
        glacier_wise_dh[["dh_m_we", "dm_tons_we", "dh_err_mwe", "dm_err_tonswe"]], left_index=True, right_index=True
    ).rename(
        columns={
            "dh_m_we": "geodetic_dh",
            "dh_err_mwe": "geodetic_dh_err",
            "dm_tons_we": "geodetic_dm",
            "dm_err_tonswe": "geodetic_dm_err",
            "dh": "glaciological_dh",
            "dm": "glaciological_dm",
        }
    )

    matthias_dh[["glaciological_dh", "glaciological_dm"]] /= STANDARD_END_YEAR - STANDARD_START_YEAR

    # matthias_dh["geodetic_dh"] = glacier_wise_dh.loc[matthias_dh.index, "dh_m_we"].values * (STANDARD_END_YEAR - STANDARD_START_YEAR)

    #import matplotlib.pyplot as plt

    #print(matthias_dh.corr())
    #matthias_dh["diff_dh"] = matthias_dh["geodetic_dh"] - matthias_dh["glaciological_dh"]
    #print(abs(matthias_dh["diff_dh"].median()) / matthias_dh[["geodetic_dh", "glaciological_dh"]].abs().mean().mean())

    return matthias_dh

    for i, col in enumerate(["dh", "dm"]):
        plt.subplot(1,2, i + 1)
        sign = -1 if col == "dm" else 1
        plt.errorbar(matthias_dh[f"geodetic_{col}"] * sign, matthias_dh[f"glaciological_{col}"] * sign, xerr=matthias_dh[f"geodetic_{col}_err"] * 2, marker="s", lw=0, elinewidth=2, ecolor="black")
    
        coeffs = np.polyfit(matthias_dh[f"geodetic_{col}"], matthias_dh[f"glaciological_{col}"], deg=1, w=matthias_dh[f"geodetic_{col}_err"])
        minval = matthias_dh[[f"geodetic_{col}", f"glaciological_{col}"]].min().min()

        plt.plot([minval * sign, 0], [minval * sign, 0])
        plt.plot(xs * sign, np.poly1d(coeffs)(xs) * sign)
        plt.title(f"{STANDARD_START_YEAR}$-${STANDARD_END_YEAR}")

        if col == "dm m":
            plt.xscale("log")
            plt.yscale("log")

        plt.xlabel(f"Geodetic MB ({'m' if col == 'dh' else 'tons'} w.e. a⁻¹)")
        plt.ylabel(f"Matthias's MB ({'m' if col == 'dh' else 'tons'} w.e. a⁻¹)")

    plt.show()
    return
    plt.subplot(122)
    plt.scatter(-matthias_dh["geodetic_dm"], -matthias_dh["glaciological_dm"])
    minval = matthias_dh[["geodetic_dm", "glaciological_dm"]].min().min()
    plt.plot([-minval, 0], [-minval, 0])
    plt.title(f"{STANDARD_START_YEAR}$-${STANDARD_END_YEAR}")
    plt.xlabel(r"Geodetic MB (tons w.e. a$^{-1}$)")
    plt.ylabel(r"Matthias's MB (tons w.e. a$^{-1}$)")
    plt.show()

    print(glacier_wise_dh)

    print(matthias_dh)
