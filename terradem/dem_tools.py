"""Tools to handle DEMs."""
from __future__ import annotations

import os
from numbers import Number
from typing import Optional

import numpy as np
import rasterio as rio
from tqdm import tqdm

import terradem.files
import terradem.metadata


def merge_rasters(directory: str, output_path: str):
    """
    Merge all rasters in a directory using the nanmean of each pixel.

    The bounds and resolution will be mirrored to the base_dem.

    :param directory: The directory to search for DEMs.
    """
    filepaths = [os.path.join(directory, fn) for fn in os.listdir(directory) if fn.endswith(".tif")]

    with rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"]) as base_dem_ds:
        bounds = base_dem_ds.bounds
        transform = base_dem_ds.transform
        meta = base_dem_ds.meta

    # Allocate an empty output raster array
    merged_raster = np.empty(
        (int((bounds.top - bounds.bottom) / 5), int((bounds.right - bounds.left) / 5)),
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

        upper, left = rio.transform.rowcol(transform, intersecting_bounds.left, intersecting_bounds.top)
        merged_s = np.s_[upper: upper + dem_ds.height, left: left + dem_ds.width]

        dem_window = rio.windows.from_bounds(
            *intersecting_bounds,
            transform=dem_ds.transform,
            width=dem_ds.width,
            height=dem_ds.height
        )

        dem_data = dem_ds.read(1, window=dem_window, masked=True).filled(np.nan)
        finites = np.isfinite(dem_data)
        reasonable_max = dem_data < 5000
        reasonable_min = dem_data > -1000
        inliers = finites & reasonable_max & reasonable_min

        if dem_ds.nodata is not None:
            nodata_values.append(dem_ds.nodata)

        raster_slice = merged_raster[merged_s]

        raster_slice[inliers] = np.nanmean([raster_slice[inliers], dem_data[inliers]], axis=0)

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
        raster.write(np.where(np.isfinite(merged_raster), merged_raster, nodata), 1)


def generate_ddems(dem_filepaths: Optional[list[str]] = None, overwrite: bool = False):
    """
    Generate yearly dDEMs.
    """

    if dem_filepaths is None:
        dem_filepaths = [os.path.join(terradem.files.TEMP_SUBDIRS["dems_coreg"], fn)
                         for fn in os.listdir(terradem.files.TEMP_SUBDIRS["dems_coreg"])]

    base_dem_ds = rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"])
    base_dem_years_ds = rio.open(terradem.files.INPUT_FILE_PATHS["base_dem_years"])

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
        output_path = os.path.join(terradem.files.TEMP_SUBDIRS["ddems"], f"{stereo_pair}_ddem.tif")
        description = (f"Yearly dDEM ({stereo_pair}). {date.date()} to ~{np.nanmean(base_dem_years):.2f}."
                       f" Average time: {np.nanmean(time_diff):.2f} years.")

        # Save the dDEM.
        with rio.open(output_path, "w", compress="deflate", tiled=True, **dem_ds.meta) as raster:
            raster.set_band_description(1, description)
            raster.set_band_unit(1, "m")
            raster.write(np.where(np.isfinite(yearly_ddem), yearly_ddem, dem_ds.nodata), 1)
