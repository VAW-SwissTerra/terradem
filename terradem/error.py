"""Functions to calculate DEM/dDEM error."""
from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import xdem
from tqdm import tqdm

import terradem.files


def compare_idealized_interpolation(
    idealized_ddem_path: str = terradem.files.TEMP_FILES["ddem_coreg_tcorr_interp-ideal"],
    ddem_path: str = terradem.files.TEMP_FILES["ddem_coreg_tcorr"],
    max_residuals: float | int = 1e7,
) -> None:
    """
    Compare an "idealized" interpolated dDEM with actual glacier values.

    :param idealized_ddem_path: The path to the idealized interpolated dDEM.
    :param ddem_path: The path to the dDEM without interpolation.
    """
    glacier_mask_ds = rio.open(terradem.files.TEMP_FILES["lk50_rasterized"])
    idealized_ddem_ds = rio.open(idealized_ddem_path)
    ddem_ds = rio.open(ddem_path)

    windows = [window for ij, window in glacier_mask_ds.block_windows()]
    random.shuffle(windows)

    differences = np.zeros(shape=(0,), dtype="float32")

    for window in tqdm(windows):
        glacier_mask = glacier_mask_ds.read(1, window=window, masked=True).filled(0) > 0
        idealized_ddem = idealized_ddem_ds.read(1, window=window, masked=True).filled(np.nan)
        ddem = ddem_ds.read(1, window=window, masked=True).filled(np.nan)

        ddem[~glacier_mask] = np.nan

        if np.all(np.isnan(ddem)):
            continue

        difference = idealized_ddem - ddem

        differences = np.append(differences, difference[np.isfinite(difference)])

        if max_residuals != 0 and differences.shape[0] > int(max_residuals):
            break

    print(differences)
    print(np.median(differences), xdem.spatial_tools.nmad(differences))


def slope_vs_error(num_values: int | float = 5e7) -> None:
    glacier_mask_ds = rio.open(terradem.files.TEMP_FILES["lk50_rasterized"])
    ddem_ds = rio.open(terradem.files.TEMP_FILES["ddem_coreg_tcorr"])
    slope_ds = rio.open(terradem.files.TEMP_FILES["base_dem_slope"])
    aspect_ds = rio.open(terradem.files.TEMP_FILES["base_dem_aspect"])
    dem_ds = rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"])
    curvature_ds = rio.open(terradem.files.TEMP_FILES["base_dem_curvature"])

    windows = [window for ij, window in glacier_mask_ds.block_windows()]
    random.shuffle(windows)

    values = np.zeros(shape=(0, 5), dtype="float32")

    progress_bar = tqdm(total=num_values)

    for window in windows:
        glacier_mask = glacier_mask_ds.read(1, window=window, masked=True).filled(0) > 0

        ddem = ddem_ds.read(1, window=window, masked=True).filled(np.nan)
        slope = slope_ds.read(1, window=window, masked=True).filled(np.nan)
        aspect = aspect_ds.read(1, window=window, masked=True).filled(np.nan)
        dem = dem_ds.read(1, window=window, masked=True).filled(np.nan)
        curvature = curvature_ds.read(1, window=window, masked=True).filled(np.nan)

        mask = (
            ~glacier_mask
            & np.isfinite(ddem)
            & np.isfinite(slope)
            & np.isfinite(dem)
            & np.isfinite(aspect)
            & np.isfinite(curvature)
        )

        n_valid = np.count_nonzero(mask)

        if n_valid == 0:
            continue

        values = np.append(
            values, np.vstack((ddem[mask], slope[mask], aspect[mask], dem[mask], curvature[mask])).T, axis=0
        )

        progress_bar.update(np.count_nonzero(mask))

        if values.shape[0] > num_values:
            progress_bar.close()
            break

    values = values[(values[:, 0] > -0.5) & (values[:, 0] < 0.5)]

    dims = {1: "Slope (degrees)", 2: "Aspect (degrees)", 3: "Elevation (m a.s.l.)", 4: "Curvature"}

    for dim, xlabel in dims.items():
        plt.subplot(2, 2, dim)
        bins = np.linspace(values[:, dim].min(), values[:, dim].max(), 20)
        indices = np.digitize(values[:, dim], bins=bins)
        for i in np.unique(indices):
            vals = values[:, 0][indices == i]
            xs = values[:, dim][indices == i].mean()
            plt.boxplot(x=vals, positions=[xs], vert=True, manage_ticks=False, sym="", widths=(bins[1] - bins[0]) * 0.9)

        plt.xlabel(xlabel)
        plt.ylabel("Temporally corrected elev. change (m/a)")
        plt.ylim(values[:, 0].min(), values[:, 0].max())

    plt.tight_layout()
    plt.show()
