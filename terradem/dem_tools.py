"""Tools to handle DEMs."""
from __future__ import annotations

import json
import os
import warnings
from numbers import Number
from typing import Any, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import scipy.ndimage
import skimage.morphology
import xdem.spatial_tools
from tqdm import tqdm

import terradem.files
import terradem.massbalance
import terradem.metadata
import terradem.utilities


def merge_rasters(directory_or_filepaths: Union[str, list[str]], output_path: str):
    """
    Merge all rasters in a directory using the nanmean of each pixel.

    The bounds and resolution will be mirrored to the base_dem.

    :param directory: The directory to search for DEMs.
    """
    if isinstance(directory_or_filepaths, str):
        filepaths = [os.path.join(directory_or_filepaths, fn)
                     for fn in os.listdir(directory_or_filepaths) if fn.endswith(".tif")]
    else:
        filepaths = directory_or_filepaths

    with rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"]) as base_dem_ds:
        bounds = base_dem_ds.bounds
        transform = base_dem_ds.transform
        meta = base_dem_ds.meta

    # Allocate an empty output raster array
    merged_raster = np.empty(
        (int((bounds.top - bounds.bottom) / 5),
         int((bounds.right - bounds.left) / 5)),
        dtype="float32") + np.nan

    nodata_values: list[Number] = []

    for filepath in tqdm(filepaths, desc="Merging DEMs"):
        dem_ds = rio.open(filepath)

        intersecting_bounds = rio.coords.BoundingBox(
            left=max(bounds.left, dem_ds.bounds.left),
            right=min(bounds.right, dem_ds.bounds.right),
            bottom=max(bounds.bottom, dem_ds.bounds.bottom),
            top=min(bounds.top, dem_ds.bounds.top)
        )
        # If any of these assertions are true, the small raster doesn't overlap with the big raster.
        if (intersecting_bounds.left > intersecting_bounds.right) or (intersecting_bounds.bottom > intersecting_bounds.top):
            continue

        upper, left = rio.transform.rowcol(
            transform, intersecting_bounds.left, intersecting_bounds.top)
        merged_s = np.s_[upper: upper +
                         dem_ds.height, left: left + dem_ds.width]

        dem_window = rio.windows.from_bounds(
            *intersecting_bounds,
            transform=dem_ds.transform,
            width=dem_ds.width,
            height=dem_ds.height
        )

        dem_data = dem_ds.read(1, window=dem_window,
                               masked=True).filled(np.nan)
        finites = np.isfinite(dem_data)
        reasonable_max = dem_data < 5000
        reasonable_min = dem_data > -1000
        inliers = finites & reasonable_max & reasonable_min

        if dem_ds.nodata is not None:
            nodata_values.append(dem_ds.nodata)

        raster_slice = merged_raster[merged_s]

        raster_slice[inliers] = np.nanmean(
            [raster_slice[inliers], dem_data[inliers]], axis=0)

    nodata = np.median(nodata_values) if len(nodata_values) > 0 else -9999

    meta.update(dict(
        nodata=nodata,
        compress="DEFLATE",
        tiled=True
    ))
    with rio.open(
            output_path,
            mode="w",
            **meta) as raster:
        raster.write(np.where(np.isfinite(merged_raster),
                              merged_raster, nodata), 1)


def generate_ddems(dem_filepaths: Optional[list[str]] = None,
                   output_directory: str = terradem.files.TEMP_SUBDIRS["ddems_coreg"], overwrite: bool = False):
    """
    Generate yearly dDEMs.
    """
    if dem_filepaths is None:
        dem_filepaths = [os.path.join(terradem.files.TEMP_SUBDIRS["dems_coreg"], fn)
                         for fn in os.listdir(terradem.files.TEMP_SUBDIRS["dems_coreg"])]

    base_dem_ds = rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"])
    base_dem_years_ds = rio.open(
        terradem.files.INPUT_FILE_PATHS["base_dem_years"])

    dates = terradem.metadata.get_stereo_pair_dates()

    for dem_path in tqdm(dem_filepaths, desc="Generating dDEMs"):
        dem_ds = rio.open(dem_path)

        stereo_pair = os.path.basename(dem_path).replace("_dense_DEM.tif", "")

        date = dates[stereo_pair]
        date_decimal_years = date.year + date.month / 12 + date.day / 365

        dem = dem_ds.read(1, masked=True).filled(np.nan)
        # Read the base DEM within the bounds of the input DEM.
        base_dem = base_dem_ds.read(
            1,
            window=rio.windows.from_bounds(
                *dem_ds.bounds,
                transform=base_dem_ds.transform,
                height=dem_ds.height,
                width=dem_ds.width
            ),
            boundless=True,
            masked=True
        ).filled(np.nan)
        # Read the base DEM years product. If it is out of bounds, just assume that the date is 1 Septermber 2018.
        try:
            base_dem_years = base_dem_years_ds.read(
                1,
                window=rio.windows.from_bounds(
                    *dem_ds.bounds,
                    transform=base_dem_years_ds.transform,
                    height=dem_ds.height,
                    width=dem_ds.width
                ),
                boundless=True,
                masked=True
            ).filled(np.nan)
        except rio.errors.WindowError:
            base_dem_years = np.ones_like(dem) * (2018 + 9/12)

        # Calculate the dDEM
        ddem = base_dem - dem

        # If all dDEM values are nan, skip it.
        if np.all(~np.isfinite(ddem)):
            continue

        # Calculate the time difference and divide by the amount of years.
        time_diff = base_dem_years - date_decimal_years
        yearly_ddem = ddem / time_diff

        # Generate an output path and description of the dDEM.
        output_path = os.path.join(output_directory, f"{stereo_pair}_ddem.tif")
        description = (f"Yearly dDEM ({stereo_pair}). {date.date()} to ~{np.nanmean(base_dem_years):.2f}."
                       f" Average time: {np.nanmean(time_diff):.2f} years.")

        # Save the dDEM.
        with rio.open(output_path, "w", compress="deflate", tiled=True, **dem_ds.meta) as raster:
            raster.set_band_description(1, description)
            raster.set_band_unit(1, "m")
            raster.write(np.where(np.isfinite(yearly_ddem),
                                  yearly_ddem, dem_ds.nodata), 1)


def get_ddem_statistics():

    coregistered_ddem_paths = {os.path.basename(fp).replace("_ddem.tif", ""): fp for fp in
                               terradem.utilities.list_files(terradem.files.TEMP_SUBDIRS["ddems_coreg"], r".*\.tif")}
    original_ddem_paths = {os.path.basename(fp).replace("_ddem.tif", ""): fp for fp in
                           terradem.utilities.list_files(terradem.files.TEMP_SUBDIRS["ddems_non_coreg"], r".*\.tif")}

    stable_ground_ds = rio.open(
        terradem.files.INPUT_FILE_PATHS["stable_ground_mask"])

    stat_indices = []
    for station in np.unique(np.r_[list(coregistered_ddem_paths.keys()), list(original_ddem_paths.keys())]):
        for product in ["coreg", "noncoreg"]:
            for surface in ["stable", "glacial"]:
                stat_indices.append((station, product, surface))
    stats = pd.DataFrame(index=pd.MultiIndex.from_tuples(
        stat_indices, names=["station", "product", "surface"]))

    for station in tqdm(coregistered_ddem_paths, desc="Calculating dDEM statistics.", smoothing=0):

        if original_ddem_paths.get(station) is None:
            continue

        ddem_coreg_ds = rio.open(coregistered_ddem_paths[station])
        ddem_non_coreg_ds = rio.open(original_ddem_paths[station])

        ddem_coreg = ddem_coreg_ds.read(1, masked=True).filled(np.nan)
        ddem_non_coreg = ddem_non_coreg_ds.read(1, window=ddem_non_coreg_ds.window(*ddem_coreg_ds.bounds),
                                                boundless=True, masked=True).filled(np.nan)
        stable_ground = stable_ground_ds.read(1, window=stable_ground_ds.window(*ddem_coreg_ds.bounds),
                                              boundless=True, masked=True).filled(0) == 1

        for abbrev, product in zip(["coreg", "noncoreg"], [ddem_coreg, ddem_non_coreg]):
            for surface, values in zip(["stable", "glacial"], [product[stable_ground], product[~stable_ground]]):
                index = (station, abbrev, surface)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="Mean of empty slice")
                    warnings.filterwarnings("ignore", message="All-NaN slice")
                    warnings.filterwarnings(
                        "ignore", message="Degrees of freedom <= 0")
                    stats.loc[index, "mean"] = np.nanmean(values)
                    stats.loc[index, "median"] = np.nanmedian(values)
                    stats.loc[index, "std"] = np.nanstd(values)
                    stats.loc[index, "nmad"] = xdem.spatial_tools.nmad(values)
                    stats.loc[index, "count"] = np.count_nonzero(
                        np.isfinite(values))

    stats.to_csv(terradem.files.TEMP_FILES["ddem_stats"])


def filter_ddems():
    """
    This function should be moved somewhere else.
    """
    stats = pd.read_csv(
        terradem.files.TEMP_FILES["ddem_stats"], index_col=[0, 1, 2])

    products_to_use: dict[str, str] = {}
    ddem_paths: list[str] = []

    for station_name in stats.index.unique("station"):
        station_data = stats.loc[station_name]

        good_product = "coreg" if station_data.loc[pd.IndexSlice["coreg",
                                                                 "stable"], "count"] > (1000) else "noncoreg"

        if good_product != "coreg":
            continue

        station_data = station_data.loc[good_product]

        bad_criteria = [
            station_data.loc["glacial"]["mean"] > (20 / 80),
            station_data.loc["glacial"]["mean"] < (-300 / 80)
        ]

        if any(bad_criteria):
            continue

        products_to_use[station_name] = good_product

        ddem_path = os.path.join(
            terradem.files.TEMP_SUBDIRS["ddems_coreg" if good_product ==
                                        "coreg" else "ddems_non_coreg"],
            f"{station_name}_ddem.tif")
        ddem_paths.append(ddem_path)

    return ddem_paths


def gaussian_nan_filter(array: np.ndarray, sigma: float = 2.0, truncate: float = 4.0) -> np.ndarray:
    """
    Taken from: https://stackoverflow.com/a/36307291
    """
    sigma = 2.0                  # standard deviation for Gaussian kernel
    truncate = 4.0               # truncate filter at this many sigmas
    U = array
    V = U.copy()
    V[np.isnan(U)] = 0
    VV = scipy.ndimage.gaussian_filter(V, sigma=sigma, truncate=truncate)

    W = 0*U.copy()+1
    W[np.isnan(U)] = 0
    WW = scipy.ndimage.gaussian_filter(W, sigma=sigma, truncate=truncate)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered")
        Z = VV/WW

    return Z


def ddem_minmax_filter(ddem: np.ndarray, radius: float = 100., outlier_threshold: float = 2.) -> np.ndarray:

    blurred = gaussian_nan_filter(ddem, sigma=20)

    outliers = (np.abs(ddem - blurred) / blurred) > 1.5

    ddem[outliers] = np.nan

    return ddem


def filter_raster(input_path: str, output_path: str, radius: float = 100., outlier_threshold: float = 2.):

    with rio.open(input_path) as raster:
        data = raster.read(1, masked=True).filled(np.nan)
        meta = raster.meta

    filtered_data = ddem_minmax_filter(
        data, radius=radius, outlier_threshold=outlier_threshold)

    meta.update(dict(compress="deflate"))
    with rio.open(output_path, "w", **meta) as raster:
        raster.write(np.where(np.isfinite(filtered_data),
                              filtered_data, meta["nodata"]), 1)


def filter_all_ddems(input_directory: str = terradem.files.TEMP_SUBDIRS["ddems_coreg"], output_directory: str = terradem.files.TEMP_SUBDIRS["ddems_coreg_filtered"]):

    for filepath in tqdm(terradem.utilities.list_files(input_directory, r".*\.tif"), smoothing=0):
        new_filepath = os.path.join(
            output_directory, os.path.basename(filepath))

        filter_raster(filepath, new_filepath)


def ddem_temporal_correction(input_directory: str = terradem.files.TEMP_SUBDIRS["ddems_coreg"], output_directory: str = terradem.files.TEMP_SUBDIRS["ddems_coreg_tcorr"], output_meta_dir: str = terradem.files.TEMP_SUBDIRS["tcorr_meta_coreg"]):

    get_mb_factor = terradem.massbalance.match_zones()

    start_dates = terradem.metadata.get_stereo_pair_dates()

    base_dem_years_ds = rio.open(
        terradem.files.INPUT_FILE_PATHS["base_dem_years"])
    glacier_mask_ds = rio.open(terradem.files.TEMP_FILES["lk50_rasterized"])

    for filepath in tqdm(terradem.utilities.list_files(input_directory, r".*\.tif"), smoothing=0):

        station = terradem.utilities.station_from_filepath(filepath)

        new_filepath = os.path.join(
            output_directory, os.path.basename(filepath))

        with rio.open(filepath) as raster:
            ddem = raster.read(1, masked=True).filled(np.nan)
            meta = raster.meta
            bounds = raster.bounds

        try:
            upper, left = base_dem_years_ds.index(bounds.left, bounds.top)
            window = rio.windows.Window(
                col_off=left, row_off=upper, width=ddem.shape[1], height=ddem.shape[1])
            base_dem_years = base_dem_years_ds.read(
                1,
                window=window,
                boundless=True,
                masked=True
            ).filled(2018)

            mean_end_year = np.mean(base_dem_years)
            exact_end_year = True
        except rio.windows.WindowError as e:
            mean_end_year = 2018
            exact_end_year = False
            print(base_dem_years_ds.transform, bounds)
            raise e

        glacier_mask = glacier_mask_ds.read(
            1,
            window=glacier_mask_ds.window(*bounds),
            boundless=True,
            masked=True
        ).filled(0) != 0

        easting = np.mean([bounds.right, bounds.left])
        northing = np.mean([bounds.top, bounds.bottom])
        factor, zone = get_mb_factor(
            easting, northing, start_dates[station].year, mean_end_year)

        ddem[glacier_mask] *= factor

        meta.update(dict(
            compress="deflate",
        ))
        with rio.open(new_filepath, "w", **meta) as raster:
            raster.write(np.where(np.isfinite(ddem), ddem, meta["nodata"]), 1)

        out_meta = {"start_date": str(start_dates[station].date()),
                    "end_year": float(mean_end_year), "factor": float(factor), "n_glacier_values": int(np.count_nonzero(glacier_mask)), "easting": float(easting), "northing": float(northing), "sgi_zone": zone, "exact_end_year": exact_end_year}

        with open(os.path.join(output_meta_dir, f"{station}.json"), "w") as outfile:
            json.dump(out_meta, outfile)
