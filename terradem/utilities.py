"""Utility functions for Python."""
from __future__ import annotations
import ctypes
import io
import os
import sys
import tempfile
from contextlib import contextmanager
from typing import Any, Optional

from typing import overload, Sequence
import os
import re
import numpy as np
import pyproj
#import pyproj.transformer
#import pyproj.crs
libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')


@contextmanager
def no_stdout(stream=None, disable=False):
    """
    Redirect the stdout to a stream file.

    Source: https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/

    param: stream: a BytesIO object to write to.
    param: disable: whether to temporarily disable the feature.

    """
    if disable:
        yield
        return
    if stream is None:
        stream = io.BytesIO()
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)

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

def sgi_1973_to_2016(sgi_id: str) -> str:
    """
    Convert the slightly different SGI1973 to the SGI2016 id format.
    
    :examples:
        >>> sgi_1973_to_2016("B55")
        'B55'
        >>> sgi_1973_to_2016("B55-19")
        'B55-19'
        >>> sgi_1973_to_2016("E73-2")
        'E73-02'
    """
    if "-" not in sgi_id:
        return sgi_id

    start, end = sgi_id.split("-")
    return start + "-" + end.zfill(2)

def sgi_2016_to_1973(sgi_id: str) -> str:
    """
    Convert the slightly different SGI2016 to the SGI1973 id format.
    
    :examples:
        >>> sgi_2016_to_1973("B55")
        'B55'
        >>> sgi_2016_to_1973("B55-19")
        'B55-19'
        >>> sgi_2016_to_1973("E73-02")
        'E73-2'
    """
    if "-" not in sgi_id:
        return sgi_id

    start, end = sgi_id.split("-")
    return start + "-" + str(int(end))


@overload
def lv03_to_lv95(easting: np.ndarray, northing: np.ndarray) -> np.ndarray: ...
@overload
def lv03_to_lv95(easting: float, northing: float) -> tuple[float, float]: ...

def lv03_to_lv95(easting: np.ndarray | float, northing: np.ndarray | float) -> np.ndarray | tuple[float, float]:

    trans = _LV03_TO_LV95.transform(easting, northing)

    return trans if not isinstance(easting, Sequence) else np.array(trans).T
