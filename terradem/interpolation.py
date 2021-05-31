"""Functions for dDEM / DEM interpolation."""
import os
from typing import Optional

import numpy as np
import pandas as pd
import rasterio as rio
import xdem
from tqdm import tqdm

import terradem.files
import terradem.outlines


def normalized_regional_hypsometric(ddem_filepath: str, output_filepath: str, output_signal_filepath: str,
                                    min_coverage: float = 0.1, signal: Optional[pd.DataFrame] = None,
                                    glacier_indices_filepath: str = terradem.files.TEMP_FILES["lk50_rasterized"],
                                    verbose: bool = True):
    """
    Interpolate gaps in a dDEM using normalized regional hypsometric interpolation.

    :param ddem_filepath: The voided dDEM filepath.
    :param output_filepath: The path to the output interpolated dDEM.
    :param output_signal_filepath: The path to the output hypsometric signal.
    :param min_coverage: The minimum fractional coverage of a glacier to interpolate. Defaults to 10%.
    """
    glacier_indices_ds = rio.open(glacier_indices_filepath)
    ref_dem_ds = rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"])
    ddem_ds = rio.open(ddem_filepath)

    bounds = rio.coords.BoundingBox(
        left=626790, top=176570, right=644890, bottom=135150
    )
    bounds = ddem_ds.bounds  # Remove this to validate on a testing subset.

    if verbose:
        print("Reading data")
    ddem = ddem_ds.read(1, masked=True, window=ddem_ds.window(*bounds)).filled(np.nan)
    ref_dem = ref_dem_ds.read(1, masked=True, window=ref_dem_ds.window(*bounds)).filled(
        1000
    )
    glacier_indices = glacier_indices_ds.read(
        1,
        masked=True,
        window=glacier_indices_ds.window(*bounds),
    ).filled(0)

    if signal is None:
        if verbose:
            print("Extracting signal")
        signal = xdem.volume.get_regional_hypsometric_signal(
            ddem=ddem,
            ref_dem=ref_dem,
            glacier_index_map=glacier_indices,
            verbose=verbose
        )
        signal.to_csv(output_signal_filepath)

    interpolated_ddem = xdem.volume.norm_regional_hypsometric_interpolation(
        voided_ddem=ddem,
        ref_dem=ref_dem,
        glacier_index_map=glacier_indices,
        regional_signal=signal,
        min_coverage=min_coverage,
        verbose=verbose)

    meta = ddem_ds.meta
    meta.update(
        dict(
            compress="deflate",
            tiled=True,
            height=ddem.shape[0],
            width=ddem.shape[1],
            transform=rio.transform.from_bounds(
                *bounds, width=ddem.shape[1], height=ddem.shape[0]
            ),
        )
    )
    with rio.open(output_filepath, "w", **meta) as raster:
        raster.write(
            np.where(np.isfinite(interpolated_ddem), interpolated_ddem, ddem_ds.nodata),
            1,
        )


def get_regional_signals(level: int = 1):
    """
    Get the regional signals of all SGI regions.

    :param level: The SGI region level to subdivide in.
    """
    ref_dem_ds = rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"])
    ddem_ds = rio.open(terradem.files.TEMP_FILES["ddem_coreg_tcorr"])

    print("Reading data")
    ref_dem = ref_dem_ds.read(1, masked=True).filled(-9999)
    ddem = ddem_ds.read(1, masked=True).filled(np.nan)

    assert ref_dem.shape == ddem.shape

    for region in tqdm(terradem.outlines.get_sgi_regions(level=level)):

        glacier_indices_ds = rio.open(
            os.path.join(
                terradem.files.TEMP_SUBDIRS["rasterized_sgi_zones"], f"SGI_{region}.tif"
            )
        )

        glacier_indices = glacier_indices_ds.read(1)

        signal = xdem.volume.get_regional_hypsometric_signal(
            ref_dem=ref_dem, ddem=ddem, glacier_index_map=glacier_indices
        )
        signal.to_csv(
            os.path.join(
                terradem.files.TEMP_SUBDIRS["hypsometric_signals"],
                f"SGI_{region}_normalized.csv",
            )
        )


def read_hypsometric_signal(filepath: str) -> pd.DataFrame:
    """
    Read a hypsometric signal dataframe.

    Basically just pd.read_csv with some tweaks.
    """
    signal = pd.read_csv(filepath)
    signal.index = pd.IntervalIndex.from_tuples(signal.iloc[:, 0].apply(
        lambda s: tuple(map(float, s.replace("(", "").replace("]", "").split(",")))))

    signal.drop(columns=signal.columns[0], inplace=True)

    return signal


def subregion_normalized_hypsometric(ddem_filepath: str, output_filepath: str, level: int = 1, min_coverage: float = 0.1):

    regions = terradem.outlines.get_sgi_regions(level=level).items()

    for i, (region, outlines) in tqdm(enumerate(regions), total=len(regions), desc=f"Running norm. hypso. on region sublevel {level}"):

        # Read the already produced hypsometric signal
        signal = read_hypsometric_signal(os.path.join(
            terradem.files.TEMP_SUBDIRS["hypsometric_signals"],
            f"SGI_{region}_normalized.csv"
        ))

        total_area = outlines.geometry.area.sum()
        covered_area = signal["count"].sum() * (5 ** 2)
        if (covered_area / total_area) < min_coverage:
            continue

        glacier_indices_filepath = os.path.join(
            terradem.files.TEMP_SUBDIRS["rasterized_sgi_zones"],
            f"SGI_{region}.tif"
        )

        input_filepath = ddem_filepath if i == 0 else output_filepath

        normalized_regional_hypsometric(
            ddem_filepath=input_filepath,
            output_filepath=output_filepath,
            output_signal_filepath="",
            min_coverage=min_coverage,
            signal=signal,
            glacier_indices_filepath=glacier_indices_filepath,
            verbose=False
        )


def regional_hypsometric(ddem_filepath: str, output_filepath: str):
    pass
