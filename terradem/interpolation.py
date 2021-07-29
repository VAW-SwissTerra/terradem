"""Functions for dDEM / DEM interpolation."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import rasterio as rio
import scipy.interpolate
import xdem
from tqdm import tqdm

import terradem.files
import terradem.outlines


def normalized_regional_hypsometric(
    ddem_filepath: str,
    output_filepath: str,
    output_ideal_filepath: str,
    output_signal_filepath: str,
    min_coverage: float = 0.1,
    signal: pd.DataFrame | None = None,
    glacier_indices_filepath: str = terradem.files.TEMP_FILES["lk50_rasterized"],
    verbose: bool = True,
) -> None:
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

    bounds = rio.coords.BoundingBox(left=626790, top=176570, right=644890, bottom=135150)
    bounds = ddem_ds.bounds  # Remove this to validate on a testing subset.

    if verbose:
        print("Reading data")
    ddem = ddem_ds.read(1, masked=True, window=ddem_ds.window(*bounds)).filled(np.nan)
    ref_dem = ref_dem_ds.read(1, masked=True, window=ref_dem_ds.window(*bounds)).filled(1000)
    glacier_indices = glacier_indices_ds.read(
        1,
        masked=True,
        window=glacier_indices_ds.window(*bounds),
    ).filled(0)

    if signal is None:
        if verbose:
            print("Extracting signal")
        signal = xdem.volume.get_regional_hypsometric_signal(
            ddem=ddem, ref_dem=ref_dem, glacier_index_map=glacier_indices, verbose=verbose
        )
        signal.to_csv(output_signal_filepath)

    ideal_ddem = xdem.volume.norm_regional_hypsometric_interpolation(
        voided_ddem=ddem,
        ref_dem=ref_dem,
        glacier_index_map=glacier_indices,
        regional_signal=signal,
        min_coverage=min_coverage,
        idealized_ddem=True,
        verbose=verbose,
    )
    interpolated_ddem = np.where(np.isfinite(ddem), ddem, ideal_ddem)

    meta = ddem_ds.meta
    meta.update(
        {
            "compress": "deflate",
            "tiled": True,
            "height": ddem.shape[0],
            "width": ddem.shape[1],
            "transform": rio.transform.from_bounds(*bounds, width=ddem.shape[1], height=ddem.shape[0]),
        }
    )
    with rio.open(output_filepath, "w", **meta) as raster:
        raster.write(
            np.where(np.isfinite(interpolated_ddem), interpolated_ddem, ddem_ds.nodata),
            1,
        )
    with rio.open(output_ideal_filepath, "w", **meta) as raster:
        raster.write(
            np.where(np.isfinite(ideal_ddem), ideal_ddem, ddem_ds.nodata),
            1,
        )


def get_regional_signals(level: int = 1) -> None:
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

    regions = ["national"] if level == -1 else terradem.outlines.get_sgi_regions(level=level)

    for region in tqdm(regions, disable=(len(regions) < 2)):

        filepath = (
            terradem.files.TEMP_FILES["lk50_rasterized"]
            if region == "national"
            else os.path.join(terradem.files.TEMP_SUBDIRS["rasterized_sgi_zones"], f"SGI_{region}.tif")
        )
        glacier_indices_ds = rio.open(filepath)

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
    signal.index = pd.IntervalIndex.from_tuples(
        signal.iloc[:, 0].apply(lambda s: tuple(map(float, s.replace("(", "").replace("]", "").split(","))))
    )

    signal.drop(columns=signal.columns[0], inplace=True)

    return signal


def subregion_normalized_hypsometric(
    ddem_filepath: str, output_filepath: str, output_filepath_ideal: str, level: int = 1, min_coverage: float = 0.1
) -> None:

    # If level == -1, read it as "national scale" (no subdivision)
    # In that case, the second argument here can be None as the associated check is unnecessary
    if level == -1:
        regions: list[tuple[str, Any | None]] = [("national", None)]
    else:
        regions = list(terradem.outlines.get_sgi_regions(level=level).items())

    ddem_ds = rio.open(ddem_filepath)
    base_dem_ds = rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"])

    window = rio.windows.from_bounds(634079, 136532, 652019, 145879, transform=ddem_ds.transform)
    window = rio.windows.from_bounds(*ddem_ds.bounds, transform=ddem_ds.transform)

    ddem = ddem_ds.read(1, masked=True, window=window).filled(np.nan)
    base_dem = base_dem_ds.read(1, masked=True, window=window).filled(-9999)

    ideal_ddem = np.empty_like(ddem) + np.nan

    # All signals need to exist already. If they don't, generate them.
    if not all(
        os.path.isfile(os.path.join(terradem.files.TEMP_SUBDIRS["hypsometric_signals"], f"SGI_{region}_normalized.csv"))
        for region, _ in regions
    ):
        print(f"Acquiring new hypsometric signals for subregion level {level}")
        get_regional_signals(level=level)

    # Loop over all subregions (or just one loop if it's national) and generate an ideal dDEM
    for _, (region, outlines) in tqdm(
        enumerate(regions),
        total=len(regions),
        desc=f"Running norm. hypso. on region sublevel {level}",
        disable=(level == -1),
    ):

        # Read the already produced hypsometric signal
        signal = read_hypsometric_signal(
            os.path.join(terradem.files.TEMP_SUBDIRS["hypsometric_signals"], f"SGI_{region}_normalized.csv")
        )

        # For small subregions with few glaciers, validate that there is actually any cover.
        if outlines is not None:
            total_area = outlines.geometry.area.sum()
            covered_area = signal["count"].sum() * (5 ** 2)
            if (covered_area / total_area) < min_coverage:
                continue

        glacier_indices_filepath = (
            terradem.files.TEMP_FILES["lk50_rasterized"]
            if level == -1
            else os.path.join(terradem.files.TEMP_SUBDIRS["rasterized_sgi_zones"], f"SGI_{region}.tif")
        )

        glacier_indices = rio.open(glacier_indices_filepath).read(1, masked=True, window=window).filled(0)

        # Generate an ideal dDEM for the subregion
        new_ideal_ddem = xdem.volume.norm_regional_hypsometric_interpolation(
            voided_ddem=ddem,
            ref_dem=base_dem,
            glacier_index_map=glacier_indices,
            min_coverage=min_coverage,
            regional_signal=signal,
            verbose=False,
            idealized_ddem=True,
        )

        # Fill the final ideal dDEM with the new ideal dDEM (within its subregion only)
        finites = np.isfinite(new_ideal_ddem)
        ideal_ddem[finites] = new_ideal_ddem[finites]

    # Once the ideal dDEM is finished, replace nans with the ideal wherever possible.
    # Places where both products are NaN will also be NaN
    nonfinites = ~np.isfinite(ddem)
    ddem[nonfinites] = ideal_ddem[nonfinites]

    meta = ddem_ds.meta
    meta.update(
        {
            "compress": "deflate",
            "tiled": True,
        }
    )

    # Save the output normal and ideal dDEM
    with rio.open(output_filepath, "w", **meta) as raster:
        raster.write(
            np.where(np.isfinite(ddem), ddem, ddem_ds.nodata),
            1,
        )
    with rio.open(output_filepath_ideal, "w", **meta) as raster:
        raster.write(
            np.where(np.isfinite(ideal_ddem), ideal_ddem, ddem_ds.nodata),
            1,
        )


def regional_hypsometric(ddem_filepath: str, output_filepath: str, output_filepath_ideal: str) -> None:

    ddem_ds = rio.open(ddem_filepath)
    glacier_indices_ds = rio.open(terradem.files.TEMP_FILES["lk50_rasterized"])
    base_dem_ds = rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"])

    ddem = ddem_ds.read(1, masked=True).filled(np.nan)
    base_dem = base_dem_ds.read(1, masked=True).filled(np.nan)
    glacier_mask = glacier_indices_ds.read(1, masked=True).filled(0) > 0

    inlier_mask = np.isfinite(ddem) & np.isfinite(base_dem) & glacier_mask

    # Estimate the elevation dependent gradient.
    gradient = xdem.volume.hypsometric_binning(
        ddem[inlier_mask],
        base_dem[inlier_mask],
        bins=50,
        kind="quantile",
    )

    # Interpolate possible missing elevation bins in 1D - no extrapolation done here
    interpolated_gradient = xdem.volume.interpolate_hypsometric_bins(gradient)

    gradient_model = scipy.interpolate.interp1d(
        interpolated_gradient.index.mid, interpolated_gradient["value"].values, fill_value="extrapolate"
    )

    # Create an idealized dDEM using the relationship between elevation and dDEM
    idealized_ddem = np.empty_like(ddem) + np.nan
    idealized_ddem[glacier_mask] = gradient_model(base_dem[glacier_mask])

    # Replace ddem gaps with idealized hypsometric ddem, but only within mask
    corrected_ddem = np.where(glacier_mask & ~np.isfinite(ddem), idealized_ddem, ddem)

    meta = ddem_ds.meta
    meta.update({"compress": "deflate", "tiled": True})

    with rio.open(output_filepath, "w", **meta) as out_raster:
        out_raster.write(np.where(np.isfinite(corrected_ddem), corrected_ddem, meta["nodata"]), 1)

    with rio.open(output_filepath_ideal, "w", **meta) as out_raster:
        out_raster.write(np.where(np.isfinite(idealized_ddem), idealized_ddem, meta["nodata"]), 1)
