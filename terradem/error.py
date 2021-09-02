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


def get_error(n_values: int = int(5e6), overwrite: bool = False) -> None:  # noqa

    cache_path = "temp/error_vals.pkl"

    # test_bounds = rio.coords.BoundingBox(659940, 164390, 667690,172840)

    datasets = {
        ds: rio.open(key)
        for ds, key in [
            ("glacier_mask", terradem.files.TEMP_FILES["lk50_rasterized"]),
            ("ddem", terradem.files.TEMP_FILES["ddem_coreg_tcorr"]),
            ("base_dem", terradem.files.INPUT_FILE_PATHS["base_dem"]),
        ]
        + [(attr, terradem.files.TEMP_FILES[f"base_dem_{attr}"]) for attr in terradem.files.TERRAIN_ATTRIBUTES]
    }

    if not overwrite and os.path.isfile(cache_path):
        values = pd.read_pickle(cache_path)
        print("Read values")
    else:
        windows = [window for ij, window in datasets["ddem"].block_windows()]
        random.shuffle(windows)
        # windows = [datasets["ddem"].window(*test_bounds)]

        values = pd.DataFrame(columns=[k for k in datasets.keys() if k != "glacier_mask"])

        progress_bar = tqdm(total=n_values, smoothing=0)

        for window in windows:
            data = {
                key: datasets[key].read(1, window=window, masked=True).filled(np.nan if key != "glacier_mask" else 0)
                for key in datasets
            }

            window_values = pd.DataFrame(np.vstack([value.ravel() for value in data.values()]).T, columns=data.keys())[
                values.columns
            ]
            stable_ground_mask = (data["glacier_mask"] == 0).ravel() & np.all(np.isfinite(window_values), axis=1)

            if np.count_nonzero(stable_ground_mask) == 0:
                continue

            values = values.append(window_values[stable_ground_mask])
            progress_bar.update(np.count_nonzero(stable_ground_mask))
            if values.shape[0] > n_values:
                break

        values.to_pickle(cache_path)

        progress_bar.close()

    values = values.sample(100000)
    print(values)

    predictors = ["slope", "aspect", "curvature", "profile_curvature", "planform_curvature"]

    bins = 30

    df = xdem.spatialstats.nd_binning(
        values["ddem"],
        values[predictors].values.T,
        list_var_names=predictors,
        list_var_bins=[np.nanquantile(values[p], np.linspace(0, 1, bins)) for p in predictors],
    )

    error_model = xdem.spatialstats.interp_nd_binning(df, list_var_names=predictors, statistic="nmad", min_count=10)

    plotting_props = {
        "slope": {"label": "Slope (°)"},
        "aspect": {"label": "Aspect (°)"},
        "planform_curvature": {"label": "Planform curvature (1/100 m)"},
        "profile_curvature": {"label": "Profile curvature (1/100 m)"},
        "curvature": {"label": "Curvature (1/100 m)"},
        "nmad": {"label": "NMAD (m)"},
    }

    if False:
        plt.figure(figsize=(8, 5), dpi=200)
        for i, attr in enumerate(predictors, start=1):
            axis: plt.Axes = plt.subplot(2, 3, i)
            data = df[
                (~df[attr].isna()) & (df["count"] > 0) & (df[[p for p in predictors if p != attr]].isna().all(axis=1))
            ][["count", "nmad", attr]].set_index(attr)

            if "curvature" in attr:
                data = data[np.abs(data.index.mid) < 10]

            plt.plot(data.index.mid, data["nmad"], color="black")
            plt.scatter(data.index.mid, data["nmad"], marker="x")
            plt.ylim(0, 0.25)
            if i < 4:
                plt.gca().xaxis.tick_top()
                axis.xaxis.set_label_position("top")
            plt.xlabel(plotting_props[attr]["label"])
            if i in [1, 4]:
                plt.ylabel(plotting_props["nmad"]["label"])
            else:
                axis.set_yticks(axis.get_yticks())  # If this is not done, the next command will complain.
                axis.set_yticklabels([""] * len(axis.get_yticks()))
            plt.grid()

        plt.tight_layout(w_pad=0.1)

        plt.close()
    # df = df[np.abs(pd.IntervalIndex(df["profile_curvature"]).mid.values) < 10]

    print(df)
    del values
    del df

    """
    return
    values["ddem_abs"] = values["ddem"].abs()
    bin_counts, bin_edges = np.histogram(values["ddem_abs"], bins=np.linspace(0, 4))

    values["indices"] = np.digitize(values["ddem_abs"], bin_edges)



    model = sklearn.linear_model.RANSACRegressor(
        base_estimator=sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.Normalizer(),
            sklearn.preprocessing.PolynomialFeatures(2), sklearn.linear_model.LinearRegression()
        )
    ).fit(values[predictors], values["ddem_abs"])

    score = model.score(values[predictors][model.inlier_mask_], values["ddem_abs"][model.inlier_mask_])
    print(score)
    """

    import threading

    meta = datasets["ddem"].meta.copy()
    meta.update({"compress": "deflate", "tiled": True, "BIGTIFF": "YES"})

    # small_height = int((test_bounds.top - test_bounds.bottom) / datasets["ddem"].res[0])
    # small_width = int((test_bounds.right - test_bounds.left) / datasets["ddem"].res[1])

    # meta.update({"transform": rio.transform.from_bounds(*test_bounds, small_width, small_height), "height": \
    # small_height, "width": small_width})

    with rio.open("temp/error.tif", "w", **meta) as raster:

        read_lock = threading.Lock()
        write_lock = threading.Lock()

        windows = [w for ij, w in raster.block_windows()]

        progress_bar = tqdm(total=len(windows), smoothing=0.1)

        def process(window: rio.windows.Window) -> None:
            with read_lock:
                big_window = datasets["ddem"].window(*rio.windows.bounds(window, meta["transform"]))
                data = np.vstack(
                    [datasets[key].read(window=big_window, masked=True).filled(np.nan) for key in predictors]
                )

            assert data.shape[0] == len(predictors)
            finites = np.all(np.isfinite(data), axis=0)

            output = np.zeros(data.shape[1:], dtype=data.dtype) + np.nan
            if np.count_nonzero(finites) > 0:
                modelled = error_model(data.reshape((len(predictors), -1)).T[finites.ravel()])
                output[finites] = modelled

            with write_lock:
                raster.write(output, 1, window=window)
            progress_bar.update()

        for window in windows:
            process(window)
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #    list(executor.map(process, windows))

        progress_bar.close()

    for dataset in datasets.values():
        dataset.close()


def terrain_error() -> None:

    slope_ds = rio.open(terradem.files.TEMP_FILES["base_dem_slope"])
    curvature_ds = rio.open(terradem.files.TEMP_FILES["base_dem_curvature"])
    stable_ground_ds = rio.open(terradem.files.INPUT_FILE_PATHS["stable_ground_mask"])
    ddem_ds = rio.open(terradem.files.TEMP_FILES["ddem_coreg_tcorr"])

    windows = [w for ij, w in ddem_ds.block_windows()]
    random.shuffle(windows)

    data = {"stable_ground": np.zeros((0,), dtype=bool)}
    data.update({key: np.zeros((0,), dtype="float32") for key in ["ddem", "curvature", "slope"]})

    for window in windows[:20]:

        stable_ground = (stable_ground_ds.read(window=window, masked=True).filled(0) == 1).ravel()

        for key, dataset in [
            ("ddem", ddem_ds),
            ("curvature", curvature_ds),
            ("slope", slope_ds),
        ]:
            data[key] = np.append(
                data[key],
                np.where(
                    stable_ground,
                    dataset.read(window=window, masked=True).filled(np.nan).ravel(),
                    np.nan,
                ).ravel(),
            )
        data["stable_ground"] = np.append(data["stable_ground"], stable_ground).astype(bool)

    if np.all(~data["stable_ground"]) or any(np.all(~np.isfinite(data[key])) for key in data):
        raise ValueError("No single finite periglacial value found.")

    custom_bins = [
        np.unique(
            np.concatenate(
                [
                    np.nanquantile(arr, np.linspace(start, stop, num))
                    for start, stop, num in [(0, 0.95, 20), (0.96, 0.99, 5), (0.991, 1, 10)]
                ]
            )
        )
        for arr in [data["slope"], data["curvature"]]
    ]
    error_df = xdem.spatialstats.nd_binning(
        values=data["ddem"],
        list_var=[data[key] for key in ["curvature", "slope"]],
        list_var_names=["curvature", "slope"],
        statistics=["count", xdem.spatial_tools.nmad],
        list_var_bins=custom_bins,
    )
    print(error_df[error_df["nd"] == 1])
    error_model = xdem.spatialstats.interp_nd_binning(
        df=error_df,
        list_var_names=["curvature", "slope"],
        min_count=30,
    )

    row_min, col_min = ddem_ds.index(626897, 108592, precision=0)

    window = rio.windows.Window(col_min, row_min, 5000, 5000)

    print(window)


    stable_ground = stable_ground_ds.read(window, masked=True).filled(0) == 1
    slope = np.where(stable_ground, slope_ds.read(window, masked=True).filled(np.nan), np.nan)
    curvature = np.where(stable_ground, curvature_ds.read(window, masked=True).filled(np.nan), np.nan)
    ddem = np.where(stable_ground, ddem_ds.read(window, masked=True).filled(np.nan), np.nan)

    error = error_model((slope, curvature))

    for arr in [error] + list(data.values()):
        print(arr.shape)
    # Standardize by the error, remove snow/ice values, and remove large outliers.
    standardized_dh = np.where(~stable_ground, np.nan, ddem / error)
    standardized_dh[np.abs(standardized_dh) > (4 * xdem.spatial_tools.nmad(standardized_dh))] = np.nan

    standardized_std = np.nanstd(standardized_dh)

    norm_dh = standardized_dh / standardized_std

    # This may fail due to the randomness of the analysis, so try to run this five times
    for i in range(5):
        try:
            variogram = xdem.spatialstats.sample_empirical_variogram(
                values=norm_dh.squeeze(),
                gsd=ddem_ds.res[0],
                subsample=50,
                n_variograms=10,
                runs=30,
            )
        except ValueError as exception:
            if i == 4:
                raise exception
            continue
        break

    vgm_model, params = xdem.spatialstats.fit_sum_model_variogram(["Sph", "Sph"], variogram)

    print(np.nanmean(error))
