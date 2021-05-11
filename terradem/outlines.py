"""Scripts for handling glacier outlines."""


import geopandas as gpd
import numpy as np
import rasterio as rio
import rasterio.features

import terradem.files


def rasterize_outlines():

    lk50 = gpd.read_file("data/results/outlines/lk50_outlines.shp")

    values = [(geom, i)
              for i, geom in enumerate(lk50.geometry.values, start=1)]

    base_dem = rio.open(
        terradem.files.INPUT_FILE_PATHS["base_dem"], load_data=False)

    rasterized = rasterio.features.rasterize(
        values, out_shape=base_dem.shape, default_value=0, dtype="uint16")

    print(rasterized)
