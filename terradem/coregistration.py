"""DEM coregistration functions."""
from __future__ import annotations

import concurrent.futures
import json
import os
import random
import threading
import warnings
from contextlib import contextmanager
from typing import Optional, Union

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import scipy.interpolate
import shapely
import xdem
from tqdm import tqdm

import terradem.files


def apply_matrix(raster: Union[np.ndarray, np.ma.masked_array], transform: rio.warp.Affine, matrix: np.ndarray,
                 centroid: Optional[tuple[float, float, float]] = None) -> np.ndarray:

    mask = (~np.isfinite(raster)) | (raster.mask if isinstance(raster, np.ma.masked_array) else False)
    raster_arr = np.array(raster)
    raster_arr[mask] = np.nan

    if len(raster_arr.shape) == 2:
        raster_arr = raster_arr[np.newaxis, :, :]

    height_mask = mask.reshape(raster_arr.shape)[0, :, :]

    x_coords, y_coords = xdem.coreg._get_x_and_y_coords(raster_arr.shape[1:], transform)
    bounds, resolution = xdem.coreg._transform_to_bounds_and_res(raster_arr.shape[1:], transform)

    if centroid is None:
        centroid = (x_coords.mean(), y_coords.mean(), 0)

    x_coords -= centroid[0]
    y_coords -= centroid[1]

    shifted_transform = rio.transform.from_origin(
        bounds.left - centroid[0], bounds.top - centroid[1], resolution, resolution)

    point_cloud = np.vstack((x_coords.reshape((1, -1)), y_coords.reshape((1, -1)),
                             raster_arr.reshape((raster_arr.shape[0], -1)))).T[~height_mask.ravel()]

    transformed_cloud = cv2.perspectiveTransform(point_cloud[np.newaxis, :, :3], matrix).squeeze()

    if point_cloud.shape[1] > 3:
        transformed_cloud = np.vstack((transformed_cloud.T, point_cloud[:, 3:].T)).T

    points = shapely.geometry.MultiPoint(transformed_cloud[:, :2])

    value_mask = rio.features.rasterize(
        points,
        out_shape=x_coords.shape,
        transform=shifted_transform,
        merge_alg=rio.features.MergeAlg.add,
    ) == 1

    bands: list[np.ndarray] = []
    for i in range(2, transformed_cloud.shape[1]):
        finites = np.isfinite(transformed_cloud[:, i])
        band = scipy.interpolate.griddata(
            points=transformed_cloud[:, :2][finites],
            values=transformed_cloud[:, i][finites],
            xi=(x_coords, y_coords),
            method="cubic",
        )
        band[~value_mask] = np.nan
        bands.append(band)

    transformed_dem = np.array(bands)

    return transformed_dem.reshape(raster.shape)


def make_pipeline() -> xdem.coreg.CoregPipeline:

    def new_apply_func(self: xdem.coreg.Coreg, dem: np.ndarray, transform: rio.warp.Affine):

        matrix = self.to_matrix()
        centroid = self._meta.get("centroid")

        return apply_matrix(raster=dem, transform=transform, matrix=matrix, centroid=centroid)

    biascorr = xdem.coreg.BiasCorr
    icp = xdem.coreg.ICP
    nuth_kaab = xdem.coreg.NuthKaab

    biascorr._apply_func = lambda self, dem, transform: dem + self._meta["bias"]

    icp._apply_func = new_apply_func
    nuth_kaab._apply_func = new_apply_func

    pipeline = xdem.coreg.CoregPipeline([biascorr(np.median), icp()])  # , nuth_kaab()])

    return pipeline


def run_coregistration(reference_dem: np.ndarray, dem: np.ndarray, transform: rio.warp.Affine):

    bias = np.nanmedian(reference_dem - dem)

    dem -= bias

    x_coords, y_coords = xdem.coreg._get_x_and_y_coords(dem.shape, transform)
    bounds, resolution = xdem.coreg._transform_to_bounds_and_res(dem.shape, transform)

    centroid = np.array([np.mean([bounds.left, bounds.right]), np.mean([bounds.bottom, bounds.top]), 0.0])
    # Subtract by the bounding coordinates to avoid float32 rounding errors.
    x_coords -= centroid[0]
    y_coords -= centroid[1]

    ref_pc = np.dstack((x_coords.ravel(), y_coords.ravel(), reference_dem.ravel())
                       ).squeeze()[np.isfinite(reference_dem).ravel()]
    tba_pc = np.dstack((x_coords.ravel(), y_coords.ravel(), dem.ravel())).squeeze()[np.isfinite(dem).ravel()]

    print(ref_pc.shape)

    icp = cv2.ppf_match_3d_ICP()
    _, residual, pose = icp.registerModelToScene(ref_pc, tba_pc)


def coregister_dem(filepath: str,
                   base_dem_lock: Optional[threading.Lock] = None, stable_ground_lock: Optional[threading.Lock] = None,
                   pixel_buffer: int = 10) -> bool:
    """
    Coregister a DEM to the "base_dem" using a pipeline of BiasCorr + ICP + NuthKaab.

    :param filepath: The path to the DEM to coregister.
    :param base_dem_lock: Optional. A threading lock to ensure only one thread reads from the base_dem dataset.
    :param stable_ground_lock: Optional. A threading lock to ensure only one thread reads from the stable_ground data.
    :param pixel_buffer: The amount of pixels to buffer the read window with (allows for larger horizontal shifts).

    :returns: True if the coregistration was successful or False if it wasn't.
    """
    # Open the datasets.
    dem_ds = rio.open(filepath)
    base_dem_ds = rio.open(terradem.files.EXTERNAL_DATA_PATHS["base_dem"])
    stable_ground_ds = rio.open(terradem.files.EXTERNAL_DATA_PATHS["stable_ground_mask"])

    resolution = dem_ds.res[0]

    # Create a window to read parts of the larger base_dem and stable_ground datasets.
    buffered_left = dem_ds.bounds.left - pixel_buffer * resolution
    buffered_top = dem_ds.bounds.top + pixel_buffer * resolution
    large_window = rio.windows.Window(
        *base_dem_ds.index(buffered_left, buffered_top)[::-1],
        height=dem_ds.height + pixel_buffer,
        width=dem_ds.width + pixel_buffer
    )
    small_window = rio.windows.Window(
        *dem_ds.index(buffered_left, buffered_top),
        height=dem_ds.height + pixel_buffer,
        width=dem_ds.width + pixel_buffer
    )

    big_transform = rio.transform.from_origin(buffered_left, buffered_top, resolution, resolution)
    big_bounds, _ = xdem.coreg._transform_to_bounds_and_res(dem_ds.shape, big_transform)

    shifted_transform = rio.transform.from_origin(
        west=(big_bounds.right - big_bounds.left) / 2,
        north=(big_bounds.top - big_bounds.bottom) / 2, xsize=resolution, ysize=resolution)

    # Read the DEM to be coregistered.
    dem = dem_ds.read(1, masked=True, window=small_window, boundless=True).filled(np.nan)

    # Read the base DEM with a threading lock (if given, otherwise make an empty context)
    with (base_dem_lock or contextmanager(lambda: iter([None]))()):
        base_dem = base_dem_ds.read(1, window=large_window, masked=True, boundless=True).filled(np.nan)
    # Read the stable ground mask with a threading lock (if given, otherwise make an empty context)
    with (stable_ground_lock or contextmanager(lambda: iter([None]))()):
        stable_ground = stable_ground_ds.read(1, window=large_window, boundless=True, fill_value=0).astype(bool)

    # Make sure they ended up in the same shape.
    assert dem.shape == base_dem.shape == stable_ground.shape

    # Create a coregistration pipeline.
    pipeline = make_pipeline()

    # Try to fit a matrix.
    try:
        with warnings.catch_warnings():
            # There should be no warnings, so if there is one, raise an error.
            warnings.simplefilter("error")
            # Fit the pipeline
            pipeline.fit(
                reference_dem=base_dem,
                dem_to_be_aligned=dem,
                inlier_mask=stable_ground,
                transform=big_transform,
            )
    except (ValueError, NotImplementedError, AssertionError, RuntimeWarning) as exception:
        allowed_exceptions = [
            "Less than 10 different cells exist",  # The nuth and kaab approach complains
            "Mean of empty slice",  # A warning turned-error of the bias correction
        ]
        for e_string in allowed_exceptions:
            if e_string in str(exception):
                return False

        print(exception)
        return False
        #raise exception

    # Extract the matrix.
    matrix = pipeline.to_matrix()
    centroid = pipeline.pipeline[1]._meta["centroid"]

    # If the total displacement is larger than 2000m, something is probably wrong.
    if np.linalg.norm(matrix[[0, 1, 2], 3]) > 2000:
        return False

    dem_coreg = apply_matrix(dem, transform=big_transform, matrix=matrix, centroid=centroid)

    meta = dem_ds.meta
    meta.update(dict(transform=big_transform))
    # Write the coregistered DEM.
    with rio.open(os.path.join(terradem.files.TEMP_SUBDIRS["dems_coreg"], os.path.basename(filepath)),
                  mode="w",
                  **meta) as raster:
        raster.write(dem_coreg.astype(dem.dtype), 1)

    # Save the resultant transformation matrix.
    with open(os.path.join(
            terradem.files.TEMP_SUBDIRS["coreg_matrices"],
            os.path.splitext(os.path.basename(filepath))[0] + ".json"
    ), "w") as outfile:
        json.dump({"centroid": centroid.tolist(), "matrix": matrix.tolist()}, outfile)

    return True


def coregister_all_dems(overwrite: bool = False, n_threads=1):
    """
    Coregister all DEMs in the data/results/dems/ directory.

    :param overwrite: Redo already existing coregistrations?
    """
    # Create threading locks for files/variables that will be read in multiple threads.
    base_dem_lock = threading.Lock()
    stable_ground_lock = threading.Lock()
    progress_bar_lock = threading.Lock()

    # List the filenames of the DEMs
    filenames = os.listdir(terradem.files.DIRECTORY_PATHS["dems"])
    #
    if not overwrite:
        existing = os.listdir(terradem.files.TEMP_SUBDIRS["dems_coreg"])
        filenames = [fp for fp in filenames if fp not in existing]

    filepaths = [os.path.join(terradem.files.DIRECTORY_PATHS["dems"], fp) for fp in filenames]
    random.shuffle(filepaths)

    #filepaths = filepaths[:100]

    progress_bar = tqdm(total=len(filepaths), desc="Coregistering DEMs", smoothing=0)

    def coregister(filepath):
        """Coregister the DEM in one thread."""
        with warnings.catch_warnings():
            # There should be no warnings, so if there is one, raise an error.
            warnings.simplefilter("error")
            coregister_dem(
                filepath=filepath,
                base_dem_lock=base_dem_lock,
                stable_ground_lock=stable_ground_lock
            )

        with progress_bar_lock:
            progress_bar.update()

    if n_threads == 1:
        for filepath in filepaths:
            coregister(filepath)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            list(executor.map(coregister, filepaths))
