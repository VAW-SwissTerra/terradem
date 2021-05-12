"""Utility functions for Python."""
from __future__ import annotations

import os
import re


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
    """Parse the station_XXXX part from a filepath."""
    basename = os.path.basename(filepath)

    station = basename[:12]

    assert station.startswith("station_")

    return station
