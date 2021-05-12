"""Tools to calculate mass balance and convert appropriately from volume."""


import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import shapely

import terradem.dem_tools
import terradem.files
import terradem.metadata


def read_mb_index():

    data = pd.read_csv(terradem.files.INPUT_FILE_PATHS["massbalance_index"],
                       delim_whitespace=True, skiprows=2, index_col=0)
    data.index.name = "year"

    return data


def match_zones():
    standard_start_year = 1930
    standard_end_year = 2020
    mb = read_mb_index().cumsum()

    standard_mb = pd.Series(
        index=mb.columns,
        data=np.diff(
            mb.T[[standard_start_year, standard_end_year]], axis=1).ravel()
    )

    zones = sorted(mb.columns, key=lambda x: len(x), reverse=True)

    lk50_outlines = gpd.read_file(
        terradem.files.INPUT_FILE_PATHS["lk50_outlines"])

    for zone in zones:
        matches = []
        for i, character in enumerate(zone):
            matches.append(lk50_outlines[f"RivLevel{i}"] == str(character))

        all_matches = np.all(matches, axis=0)

        lk50_outlines.loc[all_matches, "zone"] = zone

    # Zone A55 is not covered by the zones, so hardcode this to be A54 instead.
    lk50_outlines.loc[(
        lk50_outlines["RivLevel0"] == "A") &
        (lk50_outlines["RivLevel1"] == "5") &
        (lk50_outlines["RivLevel2"] == "5"),
        "zone"] = "A54"

    lk50_outlines["easting"] = lk50_outlines.geometry.centroid.x
    lk50_outlines["northing"] = lk50_outlines.geometry.centroid.y

    def get_mb_factor(easting: float, northing: float, start_year: float, end_year: float) -> tuple[float, str]:

        # Calculate the distance between the point and each lk50_outline centroid
        distance = np.linalg.norm([
            lk50_outlines["easting"] - easting,
            lk50_outlines["northing"] - northing
        ], axis=0)

        # Find the closest lk50 outline
        min_distance_idx = np.argwhere(distance == distance.min()).ravel()[0]

        # Extract the representative zone for the closest lk50 outline.
        mb_zone = lk50_outlines.iloc[min_distance_idx]["zone"]

        # Calculate the mass balance of that zone for the given start and end year
        actual_mb = mb.loc[int(end_year), mb_zone] - \
            mb.loc[int(start_year), mb_zone]

        # Calculate the conversion factor to the standard_start_year--standard_end_year
        factor = standard_mb[mb_zone] / actual_mb

        return factor, zone

    return get_mb_factor