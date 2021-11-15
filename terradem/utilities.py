"""Utility functions for Python."""
from __future__ import annotations

from typing import overload, Sequence
import os
import re
import numpy as np
import pyproj
#import pyproj.transformer
#import pyproj.crs

_LV03_TO_LV95 = pyproj.transformer.Transformer.from_crs(pyproj.crs.CRS.from_epsg(21781), pyproj.crs.CRS.from_epsg(2056))

def list_files(directory: str, pattern: str = ".*") -> list[str]:
    """
    List all files in a directory and return their absolute paths.

    :param directory: The directory to list files within.
    :param pattern: A regex pattern to match (for example to filter certain extensions).
    """
    files: list[str] = []

    for filename in os.listdir(directory):

        if re.match(pattern, filename) is None:
            continue

        filepath = os.path.abspath(os.path.join(directory, filename))

        if not os.path.isfile(filepath):
            continue

        files.append(filepath)

    return files


def station_from_filepath(filepath: str) -> str:
    """Parse the station_XXXX or station_XXXX_Y part from a filepath."""
    basename = os.path.basename(filepath)

    max_index = 14 if "_A" in basename or "_B" in basename else 12

    station = basename[:max_index]

    assert station.startswith("station_")

    return station


@overload
def lv03_to_lv95(easting: np.ndarray, northing: np.ndarray) -> np.ndarray: ...
@overload
def lv03_to_lv95(easting: float, northing: float) -> tuple[float, float]: ...

def lv03_to_lv95(easting: np.ndarray | float, northing: np.ndarray | float) -> np.ndarray | tuple[float, float]:

    trans = _LV03_TO_LV95.transform(easting, northing)

    return trans if not isinstance(easting, Sequence) else np.array(trans).T