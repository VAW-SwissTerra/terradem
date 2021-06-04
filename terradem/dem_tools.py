"""Tools to handle DEMs."""
from __future__ import annotations

import json
import os
import warnings
from numbers import Number

import numpy as np
import pandas as pd
import rasterio as rio
import scipy.ndimage
import xdem.spatial_tools
from tqdm import tqdm

import terradem.files
import terradem.massbalance
import terradem.metadata
import terradem.utilities


def merge_rasters(directory_or_filepaths: str | list[str], output_path: str) -> None:
    """
    Merge all rasters in a directory using the nanmean of each pixel.

    The bounds and resolution will be mirrored to the base_dem.

    :param directory_or_filepaths: The directory to search for rasters or a list of filepaths.
    :param output_path: The path to write the merged raster.
    """
    if isinstance(directory_or_filepaths, str):
        filepaths = [
            os.path.join(directory_or_filepaths, fn) for fn in os.listdir(directory_or_filepaths) if fn.endswith(".tif")
        ]
    else:
        filepaths = directory_or_filepaths

    # Open the base_dem and read its metadata. The bounds, res, and meta will be used for the merged raster.
    with rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"]) as base_dem_ds:
        bounds = base_dem_ds.bounds
        transform = base_dem_ds.transform
        meta = base_dem_ds.meta

    # Allocate an empty output raster array
    merged_raster = (
        np.empty(
            (
                int((bounds.top - bounds.bottom) / 5),
                int((bounds.right - bounds.left) / 5),
            ),
            dtype="float32",
        )
        + np.nan
    )

    # There may be multiple nodata values, so the most common will be used for the output.
    nodata_values: list[Number] = []

    # Loop over all rasters and merge them into the merged_raster array.
    for filepath in tqdm(filepaths, desc="Merging rasters"):
        raster_ds = rio.open(filepath)

        # Find the bounds of intersection between the merged raster bounds and the current raster.
        intersecting_bounds = rio.coords.BoundingBox(
            left=max(bounds.left, raster_ds.bounds.left),
            right=min(bounds.right, raster_ds.bounds.right),
            bottom=max(bounds.bottom, raster_ds.bounds.bottom),
            top=min(bounds.top, raster_ds.bounds.top),
        )
        # If any of these assertions are true, the small raster doesn't overlap with the big raster.
        if any(
            (
                (intersecting_bounds.left > intersecting_bounds.right),
                (intersecting_bounds.bottom > intersecting_bounds.top),
            )
        ):
            continue

        # Find the upper left pixel location
        upper, left = rio.transform.rowcol(transform, intersecting_bounds.left, intersecting_bounds.top)

        # Read the data from within the intersecting bounds.
        raster_data = raster_ds.read(
            1,
            window=rio.windows.from_bounds(
                *intersecting_bounds,
                transform=raster_ds.transform,
                width=raster_ds.width,
                height=raster_ds.height,
            ),
            masked=True,
        ).filled(np.nan)

        # Create a slice for the small raster's values into the merged raster.
        merged_s = np.s_[upper : upper + raster_data.shape[0], left : left + raster_data.shape[1]]

        # Make an inlier mask to exclude odd data (and ignore nans in the calculaton)
        finites = np.isfinite(raster_data)
        reasonable_max = raster_data < 5000
        reasonable_min = raster_data > -1000
        inliers = finites & reasonable_max & reasonable_min

        # Append the nodata value (if any) to the nodata_values list (the most common will be used)
        if raster_ds.nodata is not None:
            nodata_values.append(raster_ds.nodata)

        # Extract a view of the merged raster to modify with the new raster.
        raster_view = merged_raster[merged_s]

        # Assign the average of the old and the new values.
        raster_view[inliers] = np.nanmean([raster_view[inliers], raster_data[inliers]], axis=0)

    # Find the most common nodata value, or default to one if no nodata value was provided.
    nodata = np.median(nodata_values) if len(nodata_values) > 0 else -9999

    meta.update({"nodata": nodata, "compress": "DEFLATE", "tiled": True})
    # Write the merged raster.
    with rio.open(output_path, mode="w", **meta) as raster:
        raster.write(np.where(np.isfinite(merged_raster), merged_raster, nodata), 1)


def generate_ddems(
    dem_filepaths: list[str] | None = None,
    output_directory: str = terradem.files.TEMP_SUBDIRS["ddems_coreg"],
    overwrite: bool = False,
    default_end_year: float = 2018.0,
) -> None:
    """
    Generate yearly dDEMs to the reference DEM.

    :param dem_filepaths: Filepaths to the DEMs to subtract.
    :param output_directory: The directory to export the dDEMs.
    :param overwrite: Overwrite files that already exist?
    :param default_end_year: The year to fall back to if the base_dem_years does not cover the dDEM.
    """
    # If no DEM filepaths were given, default to the coregistered DEM directory.
    if dem_filepaths is None:
        dem_filepaths = [
            os.path.join(terradem.files.TEMP_SUBDIRS["dems_coreg"], fn)
            for fn in os.listdir(terradem.files.TEMP_SUBDIRS["dems_coreg"])
        ]

    # Open the base DEM and base DEM years rasters.
    base_dem_ds = rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"])
    base_dem_years_ds = rio.open(terradem.files.INPUT_FILE_PATHS["base_dem_years"])

    # Get the associated dates for each stereo pair.
    dates = terradem.metadata.get_stereo_pair_dates()

    for dem_path in tqdm(dem_filepaths, desc="Generating dDEMs"):
        dem_ds = rio.open(dem_path)

        stereo_pair = terradem.utilities.station_from_filepath(dem_path)
        output_path = os.path.join(output_directory, f"{stereo_pair}_ddem.tif")

        if not overwrite and os.path.isfile(output_path):
            continue

        # Extract the associated date for the DEM and convert it to decimal years.
        date = dates[stereo_pair]
        date_decimal_years = date.year + date.month / 12 + date.day / 365

        dem = dem_ds.read(1, masked=True).filled(np.nan)
        # Read the base DEM within the bounds of the input DEM.
        base_dem = base_dem_ds.read(1, window=base_dem_ds.window(*dem_ds.bounds), boundless=True, masked=True).filled(
            np.nan
        )
        # Read the base DEM years raster (to get the end date)
        base_dem_years = base_dem_years_ds.read(
            1,
            window=rio.windows.Window(
                *(base_dem_years_ds.index(dem_ds.bounds.left, dem_ds.bounds.top)[::-1]),
                width=dem_ds.width,
                height=dem_ds.height,
            ),
            boundless=True,
            masked=True,
        ).filled(np.nan)

        # If the base_dem_years is entirely empty, assign it to the default year.
        if np.all(~np.isfinite(base_dem_years)):
            base_dem_years[:] = default_end_year
        # Otherwise, fill potential gaps with the mean.
        else:
            base_dem_years[~np.isfinite(base_dem_years)] = np.nanmean(base_dem_years)

        # Calculate the dDEM
        ddem = base_dem - dem

        # If all dDEM values are nan, skip it.
        if np.all(~np.isfinite(ddem)):
            continue

        # Calculate the time difference and divide by the amount of years.
        time_diff = base_dem_years - date_decimal_years
        yearly_ddem = ddem / time_diff

        # Generate an output description of the dDEM.
        description = (
            f"Yearly dDEM ({stereo_pair}). {date.date()} to ~{np.nanmean(base_dem_years):.2f}. "
            f"Average time: {np.nanmean(time_diff):.2f} years."
        )

        # Save the dDEM.
        with rio.open(output_path, "w", compress="deflate", tiled=True, **dem_ds.meta) as raster:
            raster.set_band_description(1, description)
            raster.set_band_unit(1, "m")
            raster.write(np.where(np.isfinite(yearly_ddem), yearly_ddem, dem_ds.nodata), 1)


def get_ddem_statistics() -> None:
    """
    Calculate statistics for each dDEM in the coreg and noncoreg directories.

    For each dDEM (coregistered and not coregistered), the following are calculated on glacial/stable surfaces:
        * mean
        * median
        * standard deviation
        * nmad
        * valid pixel count
    """
    # Read the filepaths of each stereo pair in stereo_pair:filepath key-value pairs.
    coregistered_ddem_paths = {
        terradem.utilities.station_from_filepath(fp): fp
        for fp in terradem.utilities.list_files(terradem.files.TEMP_SUBDIRS["ddems_coreg"], r".*\.tif")
    }
    original_ddem_paths = {
        terradem.utilities.station_from_filepath(fp): fp
        for fp in terradem.utilities.list_files(terradem.files.TEMP_SUBDIRS["ddems_non_coreg"], r".*\.tif")
    }

    # Open the stable ground mask to differentiate between glacial and stable surfaces.
    stable_ground_ds = rio.open(terradem.files.INPUT_FILE_PATHS["stable_ground_mask"])
    glacier_mask_ds = rio.open(terradem.files.TEMP_FILES["lk50_rasterized"])

    # First, create the multiindex hierarchy of
    # [station (stereo_pair), product (coreg/noncoreg), surface (stable/glacial)] pairs.
    # This is made first to make sure all dDEMs (coregistered or not) get statistics.
    stat_indices = []
    for station in np.unique(np.r_[list(coregistered_ddem_paths.keys()), list(original_ddem_paths.keys())]):
        for product in ["coreg", "noncoreg"]:
            for surface in ["stable", "glacial"]:
                stat_indices.append((station, product, surface))

    # Create the (so far) empty statistics output file.
    stats = pd.DataFrame(index=pd.MultiIndex.from_tuples(stat_indices, names=["station", "product", "surface"]))

    for station in tqdm(coregistered_ddem_paths, desc="Calculating dDEM statistics.", smoothing=0):
        if original_ddem_paths.get(station) is None:
            continue

        for abbrev, filepaths in zip(["coreg", "noncoreg"], [coregistered_ddem_paths, original_ddem_paths]):

            filepath = filepaths.get(station)
            if filepath is None:
                continue

            # Read the dDEM and extract a stable ground and glacier mask for it.
            ddem_ds = rio.open(filepath)
            ddem = ddem_ds.read(1, masked=True).filled(np.nan)
            stable_ground = (
                stable_ground_ds.read(
                    1,
                    window=stable_ground_ds.window(*ddem_ds.bounds),
                    boundless=True,
                    masked=True,
                ).filled(0)
                == 1
            )
            glacier_mask = (
                glacier_mask_ds.read(
                    1,
                    window=glacier_mask_ds.window(*ddem_ds.bounds),
                    boundless=True,
                    masked=True,
                ).filled(0)
                != 0
            )

            # Loop over stable and glacial values.
            for surface, values in zip(["stable", "glacial"], [ddem[stable_ground], ddem[glacier_mask]]):
                index = (station, abbrev, surface)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Mean of empty slice")
                    warnings.filterwarnings("ignore", message="All-NaN slice")
                    warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
                    stats.loc[index, "mean"] = np.nanmean(values)
                    stats.loc[index, "median"] = np.nanmedian(values)
                    stats.loc[index, "std"] = np.nanstd(values)
                    stats.loc[index, "nmad"] = xdem.spatial_tools.nmad(values)
                    stats.loc[index, "count"] = np.count_nonzero(np.isfinite(values))

    stats.to_csv(terradem.files.TEMP_FILES["ddem_stats"])


def filter_ddems() -> list[str]:
    """
    This function should be moved somewhere else.
    """
    stats = pd.read_csv(terradem.files.TEMP_FILES["ddem_stats"], index_col=[0, 1, 2])

    products_to_use: dict[str, str] = {}
    ddem_paths: list[str] = []

    for station_name in stats.index.unique("station"):
        station_data = stats.loc[station_name]

        good_product = "coreg" if station_data.loc[pd.IndexSlice["coreg", "stable"], "count"] > (1000) else "noncoreg"

        if good_product != "coreg":
            continue

        station_data = station_data.loc[good_product]

        bad_criteria = [
            station_data.loc["glacial"]["mean"] > (20 / 80),
            station_data.loc["glacial"]["mean"] < (-300 / 80),
        ]

        if any(bad_criteria):
            continue

        products_to_use[station_name] = good_product

        ddem_path = os.path.join(
            terradem.files.TEMP_SUBDIRS["ddems_coreg" if good_product == "coreg" else "ddems_non_coreg"],
            f"{station_name}_ddem.tif",
        )
        ddem_paths.append(ddem_path)

    return ddem_paths


def gaussian_nan_filter(array: np.ndarray, sigma: float = 2.0, truncate: float = 4.0) -> np.ndarray:
    """
    Taken from: https://stackoverflow.com/a/36307291
    """
    sigma = 2.0  # standard deviation for Gaussian kernel
    truncate = 4.0  # truncate filter at this many sigmas
    U = array
    V = U.copy()
    V[np.isnan(U)] = 0
    VV = scipy.ndimage.gaussian_filter(V, sigma=sigma, truncate=truncate)

    W = 0 * U.copy() + 1
    W[np.isnan(U)] = 0
    WW = scipy.ndimage.gaussian_filter(W, sigma=sigma, truncate=truncate)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered")
        Z = VV / WW

    return Z


def ddem_minmax_filter(ddem: np.ndarray, radius: float = 100.0, outlier_threshold: float = 2.0) -> np.ndarray:

    blurred = gaussian_nan_filter(ddem, sigma=20)

    outliers = (np.abs(ddem - blurred) / blurred) > 1.5

    ddem[outliers] = np.nan

    return ddem


def filter_raster(
    input_path: str,
    output_path: str,
    radius: float = 100.0,
    outlier_threshold: float = 2.0,
) -> None:

    with rio.open(input_path) as raster:
        data = raster.read(1, masked=True).filled(np.nan)
        meta = raster.meta

    filtered_data = ddem_minmax_filter(data, radius=radius, outlier_threshold=outlier_threshold)

    meta.update({"compress": "deflate"})
    with rio.open(output_path, "w", **meta) as raster:
        raster.write(np.where(np.isfinite(filtered_data), filtered_data, meta["nodata"]), 1)


def filter_all_ddems(
    input_directory: str = terradem.files.TEMP_SUBDIRS["ddems_coreg"],
    output_directory: str = terradem.files.TEMP_SUBDIRS["ddems_coreg_filtered"],
) -> None:

    for filepath in tqdm(terradem.utilities.list_files(input_directory, r".*\.tif"), smoothing=0):
        new_filepath = os.path.join(output_directory, os.path.basename(filepath))

        filter_raster(filepath, new_filepath)


def ddem_temporal_correction(
    input_directory: str = terradem.files.TEMP_SUBDIRS["ddems_coreg"],
    output_directory: str = terradem.files.TEMP_SUBDIRS["ddems_coreg_tcorr"],
    output_meta_dir: str = terradem.files.TEMP_SUBDIRS["tcorr_meta_coreg"],
    overwrite: bool = False,
    default_end_year: float = 2018.0,
) -> None:
    """
    Temporally correct the dDEMs using a representative mass balance series.

    :param input_directory: The directory of the dDEMs to correct.
    :param output_directory: The output directory for the corrected dDEMs.
    :param output_meta_dir: The output directory for associated metadata.
    :param overwrite: Overwrite files that already exist?
    :param default_end_year: The year to fall back to if the base_dem_years does not cover the dDEM.
    """
    # Return create the function to pass an appropriate factor based on location and dates.
    get_mb_factor = terradem.massbalance.match_zones()

    # Get the associated date for each dDEM
    start_dates = terradem.metadata.get_stereo_pair_dates()

    # Open the base (modern) DEM years dataset and the glacier mask.
    base_dem_years_ds = rio.open(terradem.files.INPUT_FILE_PATHS["base_dem_years"])
    glacier_mask_ds = rio.open(terradem.files.TEMP_FILES["lk50_rasterized"])

    # Loop over each dDEM and correct it.
    for filepath in tqdm(
        terradem.utilities.list_files(input_directory, r".*\.tif"),
        smoothing=0,
        desc="Performing temporal correction.",
    ):
        # Extract the station name to fetch metadata from it.
        station = terradem.utilities.station_from_filepath(filepath)

        # Create the output filepath.
        new_filepath = os.path.join(output_directory, os.path.basename(filepath))

        # Skip if it already exists and the overwrite flag is not enabled.
        if not overwrite and os.path.isfile(new_filepath):
            continue

        # Read the dDEM and its metadata
        with rio.open(filepath) as raster:
            ddem = raster.read(1, masked=True).filled(np.nan)
            meta = raster.meta
            bounds = raster.bounds

        # Read the base DEM years raster (to get the end date)
        base_dem_years = base_dem_years_ds.read(
            1,
            window=rio.windows.Window(
                *(base_dem_years_ds.index(bounds.left, bounds.top)[::-1]),
                width=ddem.shape[1],
                height=ddem.shape[0],
            ),
            boundless=True,
            masked=True,
        ).filled(np.nan)

        # Check if the base_dem_years covers the dDEM (or if all are nans)
        exact_end_year = np.count_nonzero(np.isfinite(base_dem_years)) > 0
        # If there is coverage, take the mean year, and if not, take the default year.
        mean_end_year = np.nanmean(base_dem_years) if exact_end_year else default_end_year

        # Read the glacier mask
        glacier_mask = (
            glacier_mask_ds.read(1, window=glacier_mask_ds.window(*bounds), boundless=True, masked=True).filled(0) != 0
        )

        # Take the centre of the raster as an approximation of its centroid.
        easting = np.mean([bounds.right, bounds.left])
        northing = np.mean([bounds.top, bounds.bottom])
        # Find the associated correction factor and mass balance zone at the "centroid"
        factor, zone = get_mb_factor(easting, northing, start_dates[station].year, mean_end_year)

        # Multiply the glacier values by the factor to standardize the years.
        ddem[glacier_mask] *= factor

        meta.update({"compress": "deflate"})
        # Write the corrected dDEM
        with rio.open(new_filepath, "w", **meta) as raster:
            raster.write(np.where(np.isfinite(ddem), ddem, meta["nodata"]), 1)

        # Bundle the associated metadata and write it to disk.
        out_meta = {
            "start_date": str(start_dates[station].date()),
            "end_year": float(mean_end_year),
            "factor": float(factor),
            "n_glacier_values": int(np.count_nonzero(glacier_mask)),
            "easting": float(easting),
            "northing": float(northing),
            "sgi_zone": zone,
            "exact_end_year": exact_end_year,
        }

        with open(os.path.join(output_meta_dir, f"{station}.json"), "w") as outfile:
            json.dump(out_meta, outfile)
