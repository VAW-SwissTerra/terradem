"""Scripts for handling glacier outlines."""
import itertools
import os
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.features
import shapely
import xdem.spatial_tools
from tqdm import tqdm

import terradem.files


def rasterize_outlines(
    outlines: Optional[gpd.GeoDataFrame] = None,
    output_filepath: str = terradem.files.TEMP_FILES["lk50_rasterized"],
    overwrite: bool = False,
):
    """
    Create a glacier index map from the LK50 outlines, giving the same index +1 as they occur in the shapefile.

    :param overwrite: Overwrite if it already exists?
    """
    if not overwrite and os.path.isfile(output_filepath):
        return

    # Read the lk50 glacier outlines if they weren't provided.
    if outlines is None:
        lk50 = gpd.read_file(
            terradem.files.INPUT_FILE_PATHS["lk50_outlines"]
        ).sort_values("SGI")
        # Index 0 should be stable ground, so start at one.
        lk50.index += 1
    else:
        lk50 = outlines

    # Read the base DEM (to mirror its metadata)
    base_dem = rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"], load_data=False)

    # Rasterize the lk50 outlines, associating the same index of the shapefile (plus 1) as the glacier. Periglacial = 0
    rasterized = rasterio.features.rasterize(
        [(geom, i) for i, geom in zip(lk50.index, lk50.geometry.values)],
        out_shape=base_dem.shape,
        default_value=0,
        dtype="uint16",
        transform=base_dem.transform,
    )

    # Check that at least one glacier was covered.
    assert rasterized.max() > 0

    # Write the mask
    meta = base_dem.meta
    meta.update(dict(compress="lzw", tiled=True, dtype="uint16", nodata=None))
    with rio.open(output_filepath, "w", **meta) as raster:
        raster.write(rasterized, 1)


def get_sgi_regions(level: int) -> dict[str, gpd.GeoDataFrame]:
    if level > 2:
        raise ValueError("The max SGI region level is 2")

    # Read the LK50 outlines (with the same metadata as the SGI1973, hence containing SGI info)
    lk50_outlines = gpd.read_file(
        terradem.files.INPUT_FILE_PATHS["lk50_outlines"]
    ).sort_values("SGI")

    lk50_outlines["level"] = ""
    for i in range(level + 1):
        lk50_outlines["level"] += lk50_outlines[f"RivLevel{i}"]

    output = {region: data for region, data in lk50_outlines.groupby("level")}

    return output


def rasterize_sgi_zones(level=1, overwrite: bool = False):
    """
    Rasterize the LK50 outlines by SGI subregion level.

    :param level: The SGI subregion level to subdivide: 0, 1 or 2.
    :param overwrite: Skip file if it already exists?
    """
    for region, data in tqdm(get_sgi_regions(level=level).items()):
        output_path = os.path.join(
            terradem.files.TEMP_SUBDIRS["rasterized_sgi_zones"], f"SGI_{region}.tif"
        )
        rasterize_outlines(
            outlines=data, output_filepath=output_path, overwrite=overwrite
        )


def validate_outlines(n_points_per_line=100) -> pd.DataFrame:
    """
    Validate the LK50 outlines using sparse outlines from orthomosaics.

    Points are sampled along the outlines, and the distance to the closest point along a corresponding lk50
    outline is measured.

    :param n_points_per_line: How many points to sample for each digitized outline.
    :returns: A DataFrame of distances between the products.
    """
    digitized_outlines = gpd.read_file(
        terradem.files.INPUT_FILE_PATHS["digitized_outlines"]
    )

    lk50_outlines = gpd.read_file(terradem.files.INPUT_FILE_PATHS["lk50_outlines"])

    errors = pd.DataFrame()
    for sgi_id, lines in digitized_outlines.groupby("sgi-id"):
        # Extract the corresponding LK50 outlines and convert their exteriors to one MultiLineString collection
        matching_outlines = lk50_outlines.loc[lk50_outlines["SGI"] == sgi_id]

        # Extract the exteriors of each polygon (possibly inside a multipolygon collection)
        exteriors: list[shapely.geometry.base.BaseGeometry] = []
        for geometry in matching_outlines.geometry:
            if geometry.geom_type == "Polygon":
                exteriors.append(geometry.exterior)
            elif geometry.geom_type == "MultiPolygon":
                exteriors += [geom.exterior for geom in geometry]

        lk50_geom = shapely.geometry.MultiLineString(exteriors)

        # Interpolate points along the digitized outlines to compare with the lk50 outlines.
        points = itertools.chain.from_iterable(
            (
                (
                    line.interpolate(distance)
                    for distance in np.linspace(0, line.length, num=n_points_per_line)
                )
                for line in lines.geometry
            )
        )

        # Find the closest point (along the line, not vertex) for each digitized point to the lk50 outlines
        nearest_points = [
            shapely.ops.nearest_points(point, lk50_geom) for point in points
        ]

        # Measure the x and y distances between the points.
        distances = np.array(
            [
                [points[1].x - points[0].x, points[1].y - points[0].y]
                for points in nearest_points
            ]
        )

        errors.loc[sgi_id, "median"] = np.linalg.norm(np.median(distances, axis=0))
        errors.loc[sgi_id, "mean"] = np.linalg.norm(np.mean(distances, axis=0))
        errors.loc[sgi_id, "std"] = np.linalg.norm(np.std(distances, axis=0))
        errors.loc[sgi_id, "nmad"] = xdem.spatial_tools.nmad(np.diff(distances, axis=1))
        errors.loc[sgi_id, "count"] = distances.shape[0]

    return errors
