"""Scripts for handling glacier outlines."""
import os
import itertools

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio as rio
import rasterio.features
import shapely
import xdem.spatial_tools

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


def validate_outlines(n_points_per_line=100) -> pd.DataFrame:
    """
    Validate the LK50 outlines using sparse outlines from orthomosaics.

    Points are sampled along the outlines, and the distance to the closest point along a corresponding lk50
    outline is measured.

    :param n_points_per_line: How many points to sample for each digitized outline.
    :returns: A DataFrame of distances between the products.
    """
    digitized_outlines = gpd.read_file(terradem.files.INPUT_FILE_PATHS["digitized_outlines"])

    lk50_outlines = gpd.read_file(terradem.files.INPUT_FILE_PATHS["lk50_outlines"])

    errors = pd.DataFrame()
    for sgi_id, lines in digitized_outlines.groupby("sgi-id"):
        # Extract the corresponding LK50 outlines and convert their exteriors to one MultiLineString collection
        lk50_geom = shapely.geometry.MultiLineString(
            lk50_outlines.loc[lk50_outlines["SGI"] == sgi_id].geometry.exterior.values
        )

        # Interpolate points along the digitized outlines to compare with the lk50 outlines.
        points = itertools.chain.from_iterable(
            (
                (line.interpolate(distance) for distance in np.linspace(0, line.length, num=n_points_per_line))
                for line in lines.geometry
            )
        )

        # Find the closest point (along the line, not vertex) for each digitized point to the lk50 outlines
        nearest_points = [shapely.ops.nearest_points(point, lk50_geom) for point in points]

        # Measure the x and y distances between the points.
        distances = np.array([[points[1].x - points[0].x, points[1].y - points[0].y] for points in nearest_points])

        errors.loc[sgi_id, "median"] = np.linalg.norm(np.median(distances, axis=0))
        errors.loc[sgi_id, "mean"] = np.linalg.norm(np.mean(distances, axis=0))
        errors.loc[sgi_id, "std"] = np.linalg.norm(np.std(distances, axis=0))
        errors.loc[sgi_id, "nmad"] = xdem.spatial_tools.nmad(np.diff(distances, axis=1))
        errors.loc[sgi_id, "count"] = distances.shape[0]

    return errors

