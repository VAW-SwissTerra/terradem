from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import cv2
import geopandas as gpd
import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.features
import scipy.ndimage
import shapely.geometry
from tqdm import tqdm
import yaml

import simple_ortho.command_line
import terradem.files
import terradem.utilities


def get_all_viewsheds(overwrite: bool = False):

    raise NotImplementedError("This doesn't work currently.")
    stations = pd.read_csv(terradem.files.INPUT_FILE_PATHS["swisstopo_metadata"])["station_name"].str.replace("_L", "").str.replace("_R", "").unique()

    with rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"]) as raster:
        crs = raster.crs

    viewsheds = gpd.GeoSeries(crs=crs)
    for station in tqdm(stations, desc="Calculating viewsheds...", smoothing=0.1):
        viewshed = get_viewshed(station, overwrite=overwrite)
        viewsheds.loc[station] = viewshed.dissolve()

        print(viewsheds)

    return viewsheds


def get_viewshed(station_name: str = "station_1536", overwrite: bool = False):

    cache_path = Path(terradem.files.TEMP_SUBDIRS["viewsheds"]).joinpath(station_name + "_viewshed.geojson")

    if not overwrite and cache_path.is_file():
        return gpd.read_file(cache_path)

    image_meta = pd.read_csv(terradem.files.INPUT_FILE_PATHS["swisstopo_metadata"])
    image_meta = image_meta[image_meta["station_name"].str.contains(station_name)]


    buffer = 6000
    bounds = rio.coords.BoundingBox(
        image_meta["easting"].min() - buffer,
        image_meta["northing"].min() - buffer,
        image_meta["easting"].max() + buffer,
        image_meta["northing"].max() + buffer,
    )

    with rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"]) as raster:
        window = rio.windows.from_bounds(*bounds, transform=raster.transform)
        # bounds = rio.coords.BoundingBox(
        #    *(tuple(raster.xy(int(window.col_off + window.width), int(window.row_off)))) + tuple(raster.xy(int(window.col_off), int(window.row_off + window.height))))
        # window = rio.windows.from_bounds(*bounds, transform=raster.transform)
        dem = raster.read(1, window=window)
        crs = raster.crs

    # with rio.open(terradem.files.TEMP_FILES["ddem_coreg_tcorr_national-interp-extrap"]) as raster:
    #     ddem = scipy.ndimage.gaussian_filter(raster.read(1, window=window, masked=True).filled(0) * 90, 6)
    #     mask = rasterio.features.rasterize(lk50_outlines.geometry, out_shape=ddem.shape, fill=0, transform=rio.transform.from_bounds(*bounds, *ddem.shape[::-1])) == 1
    #     ddem[~mask] = 0
    #     dem -=ddem

    dem[:, 0] = 1e4
    dem[:, -1] = 1e4
    dem[0, :] = 1e4
    dem[-1, :] = 1e4

    with open("data/results/metadata/photogrammetry_camera_metadata.json") as infile:
        camera_metadata = {
            key: value for key, value in json.load(infile).items() if key in image_meta["Image file"].values
        }

    ori_content = pd.DataFrame()
    for key in camera_metadata:
        ori_content.loc[
            ori_content.shape[0], ["filename", "easting", "northing", "altitude", "omega", "phi", "kappa"]
        ] = (
            [os.path.splitext(key)[0]]
            + list(camera_metadata[key]["estimated_location"].values())
            + list(camera_metadata[key]["reference_rotation"]["opk"].values())
        )
    key = list(camera_metadata)[0]

    with open("data/results/metadata/photogrammetry_sensor_metadata.json") as infile:
        sensor_metadata = json.load(infile)[camera_metadata[key]["sensor"]]

    config_content = {
        "camera": {
            "name": "wild",
            "focal_len": sensor_metadata["estimated"]["F"],
            "sensor_size": [
                float(
                    np.diff([camera_metadata[key]["fiducials"][s]["location_mm"]["x"] for s in ["left", "right"]])[0]
                ),
                float(
                    np.diff([camera_metadata[key]["fiducials"][s]["location_mm"]["y"] for s in ["bottom", "top"]])[0]
                ),
            ],
        },
        "ortho": {
            "dem_interp": "cubic_spline",
            "dem_band": 1,
            "interp": "bilinear",
            "per_band": False,
            "build_ovw": False,
            "overwrite": False,
            "driver": "GTiff",
            "dtype": "uint8",
            "write_mask": False,
            "resolution": [5.0, 5.0],
            "tile_size": [256, 256],
            "compress": "deflate",
            "interleave": "pixel",
            "photometric": "MINISBLACK",
            "nodata": 0,
        },
    }

    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir)
        #path = Path("removeme")

        with rio.open(
            path.joinpath("dem.tif"),
            "w",
            "GTiff",
            width=dem.shape[1],
            height=dem.shape[0],
            count=1,
            crs=crs,
            transform=rio.transform.from_bounds(*bounds, dem.shape[1], dem.shape[0]),
            dtype="float32",
        ) as raster:
            raster.write(dem, 1)

        viewshed_paths: list[Path] = []
        for position, data in image_meta.groupby("Position"):
            easting, northing = data.iloc[0][["easting", "northing"]].astype(str)
            subprocess.run(
                [
                    "gdal_viewshed",
                    "-ox",
                    easting,
                    "-oy",
                    northing,
                    "-oz",
                    "2",
                    path.joinpath("dem.tif"),
                    path.joinpath(f"{position}_viewshed.tif"),
                ],
                check=True,
                stdout=subprocess.PIPE,
            )
            viewshed_paths.append(path.joinpath(f"{position}_viewshed.tif"))

        with open(path.joinpath("config.yaml"), "w") as outfile:
            yaml.dump(config_content, outfile)

        ori_content.to_csv(path.joinpath("camera_pos_ori.txt"), index=False, header=False, sep=" ")

        os.makedirs(path.joinpath("imgs/"), exist_ok=True)
        filepaths: list[Path] = []
        width = 500
        height = int(width * config_content["camera"]["sensor_size"][1] / config_content["camera"]["sensor_size"][0])
        for filename in image_meta["Image file"].values:
            new_filepath = path.joinpath("imgs/").joinpath(filename)
            # shutil.copy(Path("../SwissTerra/input/images/").joinpath(filename), new_filepath)

            with rio.open(
                new_filepath,
                "w",
                driver="GTiff",
                width=width,
                height=height,
                count=1,
                crs=crs,
                transform=rio.transform.from_bounds(*bounds, width, height),
                dtype="uint8",
            ) as raster:
                raster.write(np.zeros((height, width), dtype="uint8") + 255, 1)

            filepaths.append(new_filepath)

        os.makedirs(path.joinpath("orthos/"), exist_ok=True)
        with terradem.utilities.no_stdout():
            simple_ortho.command_line.main(
                src_im_file=filepaths,
                pos_ori_file=path.joinpath("camera_pos_ori.txt"),
                dem_file=path.joinpath("dem.tif"),
                read_conf=path.joinpath("config.yaml"),
                ortho_dir=path.joinpath("orthos/"),
                verbosity=3,
            )

        orthos = {}
        for ortho_path in path.joinpath("orthos/").iterdir():
            ortho = gu.Raster(str(ortho_path), load_data=False, masked=False)
            if ortho.width > 10000:
                continue
            ortho.load()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="nodata")
                orthos[ortho_path.stem.replace("_ORTHO", "")] = (
                    ortho.reproject(dst_bounds=bounds, dst_res=5).data == 255
                )

        viewsheds = []
        for viewshed_path in viewshed_paths:
            viewshed = gu.Raster(str(viewshed_path), masked=False)
            viewshed.set_ndv(128)

            viewsheds.append(viewshed.reproject(dst_bounds=bounds, dst_res=5).data == 255)

    viewshed = np.logical_and(*viewsheds)

    del viewsheds

    masks = []
    for p, data in image_meta.groupby("Position"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*regex.*")
            mask = np.logical_or.reduce(
                [orthos[key] for key in data["Image file"].str.replace(".tif", "") if key in orthos], axis=0
            )

        masks.append(mask)

    mask = np.logical_and.reduce([viewshed] + masks, axis=0).squeeze()

    polygons = (
        gpd.GeoDataFrame(
            geometry=[
                shapely.geometry.shape(t[0])
                for t in rasterio.features.shapes(
                    mask.astype("uint8"),
                    mask,
                    transform=rio.transform.from_bounds(*bounds, mask.shape[1], mask.shape[0]),
                )
            ],
            crs=crs,
        )
        .dissolve()
        .buffer(5)
        .explode()
    )

    polygons = gpd.GeoSeries(polygons.geometry.exterior.apply(shapely.geometry.Polygon).values.ravel(), crs=crs)

    polygons.to_file(cache_path, driver="GeoJSON")

    return polygons

