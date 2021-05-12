"""Scripts for handling glacier outlines."""
import os

import geopandas as gpd
import numpy as np
import rasterio as rio
import rasterio.features

import terradem.files


def rasterize_outlines(overwrite: bool = False):
    """
    Create a glacier index map from the LK50 outlines, giving the same index +1 as they occur in the shapefile.

    :param overwrite: Overwrite if it already exists?
    """

    if not overwrite and os.path.isfile(terradem.files.TEMP_FILES["lk50_rasterized"]):
        return

    # Read the lk50 glacier outlines
    lk50 = gpd.read_file(terradem.files.INPUT_FILE_PATHS["lk50_outlines"])

    # Read the base DEM (to mirror its metadata)
    base_dem = rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"], load_data=False)

    # Rasterize the lk50 outlines, associating the same index of the shapefile (plus 1) as the glacier. Periglacial = 0
    rasterized = rasterio.features.rasterize(
        [(geom, i) for i, geom in enumerate(lk50.geometry.values, start=1)],
        out_shape=base_dem.shape, default_value=0, dtype="uint16", transform=base_dem.transform)

    # Check that at least one glacier was covered.
    assert rasterized.max() > 0

    # Write the mask
    meta = base_dem.meta
    meta.update(dict(compress="lzw", tiled=True, dtype="uint16", nodata=None))
    with rio.open(terradem.files.TEMP_FILES["lk50_rasterized"], "w", **meta) as raster:
        raster.write(rasterized, 1)
