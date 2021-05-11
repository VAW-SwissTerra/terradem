"""Scripts for handling glacier outlines."""


import geopandas as gpd
import numpy as np
import rasterio as rio
import rasterio.features

import terradem.files
import xdem


def rasterize_outlines():

    lk50 = gpd.read_file("data/results/outlines/lk50_outlines.shp")

    values = [(geom, i)
              for i, geom in enumerate(lk50.geometry.values, start=1)]

    print(values[:10])

    base_dem = rio.open(
        terradem.files.INPUT_FILE_PATHS["base_dem"], load_data=False)

    rasterized = rasterio.features.rasterize(
        values, out_shape=base_dem.shape, default_value=0, dtype="uint16", transform=base_dem.transform)

    assert rasterized.max() > 0

    meta = base_dem.meta
    meta.update(dict(compress="lzw", tiled=True, dtype="uint16", nodata=None))
    with rio.open("temp/lk50_rasterized.tif", "w", **meta) as raster:
        raster.write(rasterized, 1)

    print(rasterized)


def hypsometric():

    glacier_indices_ds = rio.open("temp/lk50_rasterized.tif")
    ref_dem_ds = rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"])
    ddem_ds = rio.open("temp/merged_ddem.tif")

    bounds = rio.coords.BoundingBox(left=626790, top=176570, right=684890, bottom=135150)
    window = ddem_ds.window(*bounds)

    assert glacier_indices_ds.shape == ref_dem_ds.shape == ddem_ds.shape

    print("Reading data")
    ddem = ddem_ds.read(1, masked=True, window=window).filled(np.nan)
    ref_dem = ref_dem_ds.read(1, masked=True, window=window).filled(1000)
    glacier_indices = glacier_indices_ds.read(1, masked=True, window=window).filled(0)

    print("Extracting signal")
    interpolated_ddem = xdem.volume.norm_regional_hypsometric_interpolation(
        voided_ddem=ddem,
        ref_dem=ref_dem,
        glacier_index_map=glacier_indices,
        verbose=True)

    meta = ddem_ds.meta
    meta.update(dict(
        compress="deflate",
        tiled=True,
        height=ddem.shape[0],
        width=ddem.shape[1],
        transform=rio.transform.from_bounds(*bounds, width=ddem.shape[1], height=ddem.shape[0])
    ))
    with rio.open("temp/interpolated_ddem.tif", "w", **meta) as raster:
        raster.write(interpolated_ddem, 1)
