"""Functions for dDEM / DEM interpolation."""
from typing import Optional

import numpy as np
import pandas as pd
import rasterio as rio

import terradem.files
import xdem


def normalized_regional_hypsometric(ddem_filepath: str, output_filepath: str, output_signal_filepath: str,
                                    min_coverage: float = 0.1, signal: Optional[pd.DataFrame] = None):
    """
    Interpolate gaps in a dDEM using normalized regional hypsometric interpolation.

    :param ddem_filepath: The voided dDEM filepath.
    :param output_filepath: The path to the output interpolated dDEM.
    :param output_signal_filepath: The path to the output hypsometric signal.
    :param min_coverage: The minimum fractional coverage of a glacier to interpolate. Defaults to 10%.
    """
    glacier_indices_ds = rio.open(terradem.files.TEMP_FILES["lk50_rasterized"])
    ref_dem_ds = rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"])
    ddem_ds = rio.open(ddem_filepath)

    bounds = rio.coords.BoundingBox(left=626790, top=176570, right=644890, bottom=135150)
    bounds = ddem_ds.bounds  # Remove this to validate on a testing subset.

    print("Reading data")
    ddem = ddem_ds.read(1, masked=True, window=ddem_ds.window(*bounds)).filled(np.nan)
    ref_dem = ref_dem_ds.read(1, masked=True, window=ref_dem_ds.window(*bounds)).filled(1000)
    glacier_indices = glacier_indices_ds.read(1, masked=True, window=glacier_indices_ds.window(*bounds),
                                              ).filled(0)

    if signal is None:
        print("Extracting signal")
        signal = xdem.volume.get_regional_hypsometric_signal(
            ddem=ddem,
            ref_dem=ref_dem,
            glacier_index_map=glacier_indices,
            verbose=True
        )
        signal.to_csv(output_signal_filepath)
    interpolated_ddem = xdem.volume.norm_regional_hypsometric_interpolation(
        voided_ddem=ddem,
        ref_dem=ref_dem,
        glacier_index_map=glacier_indices,
        regional_signal=signal,
        min_coverage=min_coverage,
        verbose=True)

    meta = ddem_ds.meta
    meta.update(dict(
        compress="deflate",
        tiled=True,
        height=ddem.shape[0],
        width=ddem.shape[1],
        transform=rio.transform.from_bounds(*bounds, width=ddem.shape[1], height=ddem.shape[0])
    ))
    with rio.open(output_filepath, "w", **meta) as raster:
        raster.write(np.where(np.isfinite(interpolated_ddem), interpolated_ddem, ddem_ds.nodata), 1)

def regional_hypsometric(ddem_filepath: str, output_filepath: str):
    pass
    
