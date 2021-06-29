"""Functions to calculate DEM/dDEM error."""
from __future__ import annotations

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import xdem
from tqdm import tqdm

import terradem.files


def compare_idealized_interpolation(
    ddem_path: str = terradem.files.TEMP_FILES["ddem_coreg_tcorr"],
    max_residuals: float | int = 1e8,
) -> None:
    """
    Compare an "idealized" interpolated dDEM with actual glacier values.

    :param idealized_ddem_path: The path to the idealized interpolated dDEM.
    :param ddem_path: The path to the dDEM without interpolation.
    """
    glacier_mask_ds = rio.open(terradem.files.TEMP_FILES["lk50_rasterized"])
    ddem_ds = rio.open(ddem_path)

    idealized_ddems = {key: value for key, value in terradem.files.TEMP_FILES.items() if "-ideal" in key}

    idealized_ddem_dss = {key: rio.open(value) for key, value in idealized_ddems.items() if os.path.isfile(value)}

    windows = [window for ij, window in glacier_mask_ds.block_windows()]
    random.shuffle(windows)

    differences = {key: np.zeros(shape=(0,)) for key in idealized_ddem_dss}

    progress_bar = tqdm(
        total=max_residuals if max_residuals != 0 else len(windows), desc="Comparing dDEM with idealized dDEMs"
    )

    for window in windows:
        glacier_mask = glacier_mask_ds.read(1, window=window, masked=True).filled(0) > 0
        ddem = ddem_ds.read(1, window=window, masked=True).filled(np.nan)
        ddem[~glacier_mask] = np.nan

        if np.all(np.isnan(ddem)):
            continue

        new_value_count: list[int] = []
        for key in idealized_ddem_dss:
            idealized_ddem = idealized_ddem_dss[key].read(1, window=window, masked=True).filled(np.nan)

            difference = idealized_ddem - ddem
            difference = difference[np.isfinite(difference)]

            new_value_count.append(difference.shape[0])

            differences[key] = np.append(differences[key], difference)

        if max_residuals != 0:
            progress_bar.update(min(new_value_count))
        else:
            progress_bar.update()

        if max_residuals != 0 and all(diff.shape[0] > int(max_residuals) for diff in differences.values()):
            break

    output = pd.DataFrame(
        {
            key: {"median": np.median(values), "nmad": xdem.spatial_tools.nmad(values)}
            for key, values in differences.items()
        }
    )

    output.to_csv(terradem.files.TEMP_FILES["ddem_vs_ideal_error"])

    print(output.to_string())


def slope_vs_error(num_values: int | float = 5e6) -> None:
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
            if i == bins.shape[0]:
                continue
            vals = values[:, 0][indices == i]
            xs = bins[i]
            plt.boxplot(x=vals, positions=[xs], vert=True, manage_ticks=False, sym="", widths=(bins[1] - bins[0]) * 0.9)

        plt.xlabel(xlabel)
        plt.ylabel("Temporally corrected elev. change (m/a)")
        plt.ylim(values[:, 0].min(), values[:, 0].max())

    plt.tight_layout()
    plt.show()
