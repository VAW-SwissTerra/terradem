"""Tools to handle DEMs."""
from __future__ import annotations

import os

import numpy as np
import rasterio as rio
from tqdm import tqdm

import terradem.files


def merge_dems(directory: str, output_path: str):
    """
    Merge all DEMs in a directory using the nanmean of each pixel.

    The bounds and resolution will be mirrored to the base_dem.

    :param directory: The directory to search for DEMs.
    """
    filepaths = [os.path.join(directory, fn) for fn in os.listdir(directory) if fn.endswith(".tif")]

    with rio.open(terradem.files.EXTERNAL_DATA_PATHS["base_dem"]) as base_dem_ds:
        bounds = base_dem_ds.bounds
        transform = base_dem_ds.transform
        meta = base_dem_ds.meta

    # Allocate an empty output DEM array
    merged_dem = np.empty(
        (int((bounds.top - bounds.bottom) / 5), int((bounds.right - bounds.left) / 5)),
        dtype="float32") + np.nan

    for filepath in tqdm(filepaths, desc="Merging DEMs"):
        dem_ds = rio.open(filepath)

        upper, left = rio.transform.rowcol(transform, dem_ds.bounds.left, dem_ds.bounds.top)
        dem_slice = np.s_[upper:upper + dem_ds.height, left: left+dem_ds.width]

        dem_data = dem_ds.read(1, masked=True).filled(np.nan)
        finites = np.isfinite(dem_data)
        reasonable_max = dem_data < 4700
        reasonable_min = dem_data > 200
        inliers = finites & reasonable_max & reasonable_min

        merged_dem[dem_slice][inliers] = np.nanmean([merged_dem[dem_slice][inliers], dem_data[inliers]], axis=0)

    meta.update(dict(
        nodata=-9999,
        compress="DEFLATE",
        tiled=True
    ))
    with rio.open(
            output_path,
            mode="w",
            **meta) as raster:
        raster.write(np.where(np.isfinite(merged_dem), merged_dem, -9999), 1)
