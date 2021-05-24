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
import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import rasterio.warp
import scipy.interpolate
import scipy.ndimage
import shapely
import skimage.transform
import xdem
from tqdm import tqdm

import terradem.files


def apply_matrix(
    raster: Union[np.ndarray, np.ma.masked_array],
    transform: rio.warp.Affine,
    matrix: np.ndarray,
    centroid: Optional[tuple[float, float, float]] = None,
) -> np.ndarray:

    mask = (~np.isfinite(raster)) | (
        raster.mask if isinstance(raster, np.ma.masked_array) else False
    )
    raster_arr = np.array(raster)
    raster_arr[mask] = np.nan

    if len(raster_arr.shape) == 2:
        raster_arr = raster_arr[np.newaxis, :, :]

    height_mask = mask.reshape(raster_arr.shape)[0, :, :]

    x_coords, y_coords = xdem.coreg._get_x_and_y_coords(raster_arr.shape[1:], transform)
    bounds, resolution = xdem.coreg._transform_to_bounds_and_res(
        raster_arr.shape[1:], transform
    )

    if centroid is None:
        centroid = (x_coords.mean(), y_coords.mean(), 0)

    x_coords -= centroid[0]
    y_coords -= centroid[1]

    shifted_transform = rio.transform.from_origin(
        bounds.left - centroid[0], bounds.top - centroid[1], resolution, resolution
    )

    point_cloud = np.vstack(
        (
            x_coords.reshape((1, -1)),
            y_coords.reshape((1, -1)),
            raster_arr.reshape((raster_arr.shape[0], -1)),
        )
    ).T[~height_mask.ravel()]

    transformed_cloud = cv2.perspectiveTransform(
        point_cloud[np.newaxis, :, :3], matrix
    ).squeeze()

    if point_cloud.shape[1] > 3:
        transformed_cloud = np.vstack((transformed_cloud.T, point_cloud[:, 3:].T)).T

    points = shapely.geometry.MultiPoint(transformed_cloud[:, :2])

    value_mask = (
        rio.features.rasterize(
            points,
            out_shape=x_coords.shape,
            transform=shifted_transform,
            merge_alg=rio.features.MergeAlg.add,
        )
        == 1
    )

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
    def new_apply_func(
        self: xdem.coreg.Coreg, dem: np.ndarray, transform: rio.warp.Affine
    ):

        matrix = self.to_matrix()
        centroid = self._meta.get("centroid")

        return apply_matrix(
            raster=dem, transform=transform, matrix=matrix, centroid=centroid
        )

    biascorr = xdem.coreg.BiasCorr
    icp = xdem.coreg.ICP
    nuth_kaab = xdem.coreg.NuthKaab

    biascorr._apply_func = lambda self, dem, transform: dem + self._meta["bias"]

    icp._apply_func = new_apply_func
    nuth_kaab._apply_func = new_apply_func

    pipeline = xdem.coreg.CoregPipeline([biascorr(np.median), icp()])  # , nuth_kaab()])

    return pipeline


def run_coregistration(
    reference_dem: np.ndarray, dem: np.ndarray, transform: rio.warp.Affine
):

    bias = np.nanmedian(reference_dem - dem)

    dem -= bias

    x_coords, y_coords = xdem.coreg._get_x_and_y_coords(dem.shape, transform)
    bounds, resolution = xdem.coreg._transform_to_bounds_and_res(dem.shape, transform)

    centroid = np.array(
        [
            np.mean([bounds.left, bounds.right]),
            np.mean([bounds.bottom, bounds.top]),
            0.0,
        ]
    )
    # Subtract by the bounding coordinates to avoid float32 rounding errors.
    x_coords -= centroid[0]
    y_coords -= centroid[1]

    ref_pc = np.dstack(
        (x_coords.ravel(), y_coords.ravel(), reference_dem.ravel())
    ).squeeze()[np.isfinite(reference_dem).ravel()]
    tba_pc = np.dstack((x_coords.ravel(), y_coords.ravel(), dem.ravel())).squeeze()[
        np.isfinite(dem).ravel()
    ]

    print(ref_pc.shape)

    icp = cv2.ppf_match_3d_ICP()
    _, residual, pose = icp.registerModelToScene(ref_pc, tba_pc)


def coregister_dem(
    filepath: str,
    base_dem_lock: Optional[threading.Lock] = None,
    stable_ground_lock: Optional[threading.Lock] = None,
    pixel_buffer: int = 10,
    plot: bool = False,
) -> bool:
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
    base_dem_ds = rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"])
    stable_ground_ds = rio.open(terradem.files.INPUT_FILE_PATHS["stable_ground_mask"])

    resolution = dem_ds.res[0]

    # Create a window to read parts of the larger base_dem and stable_ground datasets.
    buffered_left = dem_ds.bounds.left - pixel_buffer * resolution
    buffered_top = dem_ds.bounds.top + pixel_buffer * resolution
    large_window = rio.windows.Window(
        *(base_dem_ds.index(buffered_left, buffered_top)[::-1]),
        height=dem_ds.height + pixel_buffer,
        width=dem_ds.width + pixel_buffer,
    )
    small_window = rio.windows.Window(
        *(dem_ds.index(buffered_left, buffered_top)[::-1]),
        height=dem_ds.height + pixel_buffer,
        width=dem_ds.width + pixel_buffer,
    )

    big_transform = rio.transform.from_origin(
        buffered_left, buffered_top, resolution, resolution
    )

    # Read the DEM to be coregistered.
    dem = dem_ds.read(1, masked=True, window=small_window, boundless=True).filled(
        np.nan
    )

    # Read the base DEM with a threading lock (if given, otherwise make an empty context)
    with (base_dem_lock or contextmanager(lambda: iter([None]))()):
        base_dem = base_dem_ds.read(
            1, window=large_window, masked=True, boundless=True
        ).filled(np.nan)
    # Read the stable ground mask with a threading lock (if given, otherwise make an empty context)
    with (stable_ground_lock or contextmanager(lambda: iter([None]))()):
        stable_ground = stable_ground_ds.read(
            1, window=large_window, boundless=True, fill_value=0
        ).astype(bool)

    # Make sure they ended up in the same shape.
    assert dem.shape == base_dem.shape == stable_ground.shape

    # Calculate the overlap between the DEMs
    overlapping_area = (
        np.count_nonzero(
            np.logical_and(
                *[
                    scipy.ndimage.maximum_filter(
                        mask, size=pixel_buffer, mode="constant"
                    )
                    for mask in (
                        np.isfinite(dem) & stable_ground,
                        np.isfinite(base_dem) & stable_ground,
                    )
                ]
            )
        )
        * (resolution ** 2)
    )
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
    except Exception as exception:
        allowed_exceptions = [
            "Less than 10 different cells exist",  # The nuth and kaab approach complains
            "Mean of empty slice",  # A warning turned-error of the bias correction
            "registerModelToScene",  # ICP may fail because of lacking data.
            "ICP coregistration failed",  # This happens if the point clouds were not even close to converging.
            "axis 1 is out of bounds",  # I honestly don't know what this comes from. Too few points?
        ]
        for e_string in allowed_exceptions:
            if e_string in str(exception):
                return False

        raise exception

    # Extract the matrix.
    matrix = pipeline.to_matrix()
    centroid = pipeline.pipeline[1]._meta[
        "centroid"
    ]  # pylint: disable=protected-access

    # If the total displacement is larger than 2000m, something is probably wrong.
    if np.linalg.norm(matrix[[0, 1, 2], 3]) > 2000:
        return False

    dem_coreg = apply_matrix(
        dem, transform=big_transform, matrix=matrix, centroid=centroid
    )

    if False:
        ddem_pre = np.where(stable_ground, base_dem - dem, np.nan)
        ddem_pos = np.where(stable_ground, base_dem - dem_coreg, np.nan)

        plt.figure(dpi=200)
        plt.subplot(221)
        plt.imshow(ddem_pre, cmap="coolwarm_r", vmin=-10, vmax=10)
        plt.subplot(222)
        plt.imshow(ddem_pos, cmap="coolwarm_r", vmin=-10, vmax=10)
        plt.subplot(223)
        plt.hist(ddem_pre.ravel(), bins=np.linspace(-50, 50, 100))
        plt.subplot(224)
        plt.hist(ddem_pos.ravel(), bins=np.linspace(-50, 50, 100))

        plt.show()

    meta = dem_ds.meta
    meta.update(dict(transform=big_transform))
    meta = dict(
        width=dem_coreg.shape[1],
        height=dem_coreg.shape[0],
        transform=big_transform,
        count=1,
        nodata=dem_ds.nodata,
        crs=dem_ds.crs,
        dtype=dem.dtype,
        compress="deflate",
    )

    # Write the coregistered DEM.
    with rio.open(
        os.path.join(
            terradem.files.TEMP_SUBDIRS["dems_coreg"], os.path.basename(filepath)
        ),
        mode="w",
        **meta,
    ) as raster:
        raster.write(dem_coreg.astype(dem.dtype), 1)

    # Save the resultant transformation matrix.
    with open(
        os.path.join(
            terradem.files.TEMP_SUBDIRS["coreg_matrices"],
            os.path.splitext(os.path.basename(filepath))[0] + ".json",
        ),
        "w",
    ) as outfile:
        json.dump(
            {
                "centroid": centroid.tolist(),
                "matrix": matrix.tolist(),
                "overlapping_area": overlapping_area,
            },
            outfile,
        )

    return True


def coregister_all_dems(
    overwrite: bool = False, n_threads=1, subset: Optional[int] = None
):
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

    filepaths = [
        os.path.join(terradem.files.DIRECTORY_PATHS["dems"], fp) for fp in filenames
    ]
    if subset is not None:
        random.shuffle(filepaths)
        filepaths = filepaths[:subset]

    progress_bar = tqdm(total=len(filepaths), desc="Coregistering DEMs", smoothing=0)

    def coregister(filepath):
        """Coregister the DEM in one thread."""
        with warnings.catch_warnings():
            # There should be no warnings, so if there is one, raise an error.
            warnings.simplefilter("error")
            coregister_dem(
                filepath=filepath,
                base_dem_lock=base_dem_lock,
                stable_ground_lock=stable_ground_lock,
            )

        with progress_bar_lock:
            progress_bar.update()

    if n_threads == 1:
        for filepath in filepaths:
            coregister(filepath)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            list(executor.map(coregister, filepaths))


def transform_orthomosaic(transform_path: str, overwrite: bool = False) -> bool:
    """
    Transform an orthomosaic using its corresponding transform object.

    A DEM and orthomosaic with the same station name as the transform will be matched.

    :param transform_path: Path to the json transform object.
    :param overwrite: If False, the transformation will be skipped if a file already exists.

    :returns: A boolean flag of whether it failed or not.
    """
    station_name = os.path.basename(transform_path).replace("_dense_DEM.json", "")

    ortho_path = os.path.join(
        terradem.files.DIRECTORY_PATHS["orthos"], f"{station_name}_orthomosaic.tif"
    )
    dem_path = os.path.join(
        terradem.files.DIRECTORY_PATHS["dems"], f"{station_name}_dense_DEM.tif"
    )

    output_path = os.path.join(
        terradem.files.TEMP_SUBDIRS["orthos_coreg"], os.path.basename(ortho_path)
    )

    if not overwrite and os.path.isfile(output_path):
        return False

    with open(transform_path) as infile:
        transform = json.load(infile)

    transform["matrix"] = np.array(transform["matrix"])
    transform["centroid"] = np.array(transform["centroid"])
    if abs(transform["matrix"][2, 3]) > 50:
        return False

    ortho_ds = rio.open(ortho_path)
    dem_ds = rio.open(dem_path)

    ortho = ortho_ds.read(1, masked=True).filled(0)

    dem = np.zeros(ortho.shape, dtype="float32")

    rasterio.warp.reproject(
        dem_ds.read(1, masked=True).filled(np.nan),
        destination=dem,
        src_transform=dem_ds.transform,
        dst_transform=ortho_ds.transform,
        dst_resolution=ortho_ds.res,
        src_crs=dem_ds.crs,
        dst_crs=ortho_ds.crs,
    )

    # ortho = gu.Raster(ortho_path)
    # dem = gu.Raster(dem_path).reproject(ortho)

    point_cloud = np.dstack(
        xdem.coreg._get_x_and_y_coords(dem.shape, ortho_ds.transform) + (dem,)
    ).reshape((-1, 3))
    outlier_mask = ~np.isfinite(point_cloud[:, 2])
    point_cloud[outlier_mask, 2] = np.nanmedian(point_cloud[:, 2])

    point_cloud -= transform["centroid"].T

    transformed_cloud = cv2.perspectiveTransform(
        point_cloud.reshape((1, -1, 3)), transform["matrix"]
    ).reshape((-1, 3))

    difference = ((transformed_cloud[:, :2] - point_cloud[:, :2]) / ortho_ds.res)[
        :, ::-1
    ]
    difference[:, 0] *= -1

    pixel_coordinates = np.mgrid[
        0 : dem.shape[0], 0 : dem.shape[1]
    ] - difference.T.reshape((2,) + dem.shape)

    new_ortho_arr = skimage.transform.warp(
        image=ortho, inverse_map=pixel_coordinates, preserve_range=True, order=1
    ).astype("uint8")

    new_mask = skimage.transform.warp(
        image=outlier_mask.reshape(ortho.shape).astype("uint8"),
        inverse_map=pixel_coordinates,
        order=0,
        cval=1,
    ).astype(bool)

    meta = ortho_ds.meta
    meta.update(dict(nodata=0, count=1, compress="lzw"))

    with rio.open(output_path, "w", **meta) as raster:
        raster.write(np.where(new_mask, 0, new_ortho_arr), 1)

    return

    data = np.vstack(
        (dem.data.filled(np.nan), ortho.data[:1, :, :].astype("float32").filled(np.nan))
    )

    transformed_data = apply_matrix(
        data,
        transform=ortho.transform,
        matrix=np.array(transform["matrix"]),
        centroid=transform["centroid"],
    )

    new_ortho_data = np.ma.masked_array(
        transformed_data[1, :, :].astype(ortho.data.dtype),
        mask=np.isnan(transformed_data[1, :, :]),
    )

    meta = ortho.ds.meta
    meta["count"] = 2
    with rio.open(output_path, mode="w", compress="lzw", **meta) as raster:
        raster.write(new_ortho_data, 1)
        raster.write(((~new_ortho_data.mask).astype(int) * 255).astype("uint8"), 2)
        raster.colorinterp = (rio.enums.ColorInterp.gray, rio.enums.ColorInterp.alpha)

    return True


def transform_all_orthomosaics(overwrite: bool = False, n_threads: int = 1):

    transforms = [
        os.path.join(terradem.files.TEMP_SUBDIRS["coreg_matrices"], fn)
        for fn in os.listdir(terradem.files.TEMP_SUBDIRS["coreg_matrices"])
    ]

    if not overwrite:
        finished_stations = [
            s[: s.index("_ortho")]
            for s in os.listdir(terradem.files.TEMP_SUBDIRS["orthos_coreg"])
            if s.endswith(".tif")
        ]

        transforms = [
            path
            for path in transforms
            if os.path.basename(path).replace("_dense_DEM.json", "")
            not in finished_stations
        ]

    progress_bar = tqdm(
        total=len(transforms), desc="Transforming orthomosaics", smoothing=0
    )

    # If only one thread should be used, loop in a normal python loop (easier to debug)
    if n_threads == 1:
        for transform_path in transforms:
            transform_orthomosaic(transform_path, overwrite=overwrite)
            progress_bar.update()
    # Otherwise, deploy multiple asynchronous threads.
    else:

        def transform_ortho(transform_path):
            transform_orthomosaic(transform_path, overwrite=overwrite)
            progress_bar.update()

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            list(executor.map(transform_ortho, transforms))

    progress_bar.close()
