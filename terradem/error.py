from __future__ import annotations

import concurrent.futures
import datetime
import os
import pathlib
import pickle
import random
import threading
from collections.abc import Iterable
from typing import overload

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import scipy.interpolate
import shapely
from shapely.geometry.base import BaseGeometry
import skimage.measure
from tqdm import tqdm, trange

import terradem.files
import terradem.massbalance
import xdem


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
            key: {"median": np.median(values), "nmad": xdem.spatialstats.nmad(values)}
            for key, values in differences.items()
        }
    ).T

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

    # Open the datasets to use
    slope_ds = rio.open(terradem.files.TEMP_FILES["base_dem_slope"])
    curvature_ds = rio.open(terradem.files.TEMP_FILES["base_dem_curvature"])
    stable_ground_ds = rio.open(terradem.files.INPUT_FILE_PATHS["stable_ground_mask"])
    ddem_ds = rio.open(terradem.files.TEMP_FILES["ddem_coreg_tcorr"])

    progress_bar = tqdm(total=4, desc="Reading data")

    # Read the stable ground mask, where True will be stable and False will be unstable
    data = {"stable_ground": stable_ground_ds.read(1, masked=True).filled(0) == 1}
    progress_bar.update()
    # data.update({key: np.zeros((0,), dtype="float32") for key in ["ddem", "curvature", "slope"]})

    # Read all of the other data and replace masked values with nan
    for key, dataset in [("ddem", ddem_ds), ("curvature", curvature_ds), ("slope", slope_ds)]:
        data[key] = dataset.read(1, masked=True).filled(np.nan)
        progress_bar.update()

    progress_bar.close()

    # Extract only the valid pixels (and later on subsample it)
    print(f"{datetime.datetime.now()} Subsetting data")
    data["valid_mask"] = data["stable_ground"] & np.logical_and.reduce(
        [np.isfinite(data[key]) for key in data if key != "stable_ground"]
    )
    subset = {key: data[key][data["valid_mask"]] for key in data if key not in ["stable_ground", "valid_mask"]}
    del data["valid_mask"]

    # Subsample the data if its size is higher than the subsampling max
    subsampling = int(1e6)
    for key in subset:
        if subset[key].shape[0] < subsampling:
            continue
        subset[key] = np.random.permutation(subset[key])[:subsampling]

    variable_names = ["curvature", "slope"]
    custom_bins = [
        np.unique(
            np.concatenate(
                [
                    np.nanquantile(subset[name], np.linspace(start, stop, num))
                    for start, stop, num in [(0, 0.95, 20), (0.96, 0.99, 5), (0.991, 1, 10)]
                ]
            )
        )
        for name in variable_names
    ]
    print(f"{datetime.datetime.now()} Performing ND-binning")
    error_df = xdem.spatialstats.nd_binning(
        values=subset["ddem"],
        list_var=[subset[key] for key in variable_names],
        list_var_names=variable_names,
        statistics=["count", xdem.spatialstats.nmad],
        list_var_bins=custom_bins,
    )
    error_df.to_csv(terradem.files.TEMP_FILES["topographic_error_df"])

    del subset
    print(error_df)
    error_model = xdem.spatialstats.interp_nd_binning(
        df=error_df,
        list_var_names=variable_names,
        min_count=30,
    )

    # Make a smaller test-area around Grosser Aletschgletscher
    # left = 663000
    # top = 173000
    # window = rio.windows.Window(*ddem_ds.index(left, top, precision=0), 5000, 5000)
    # transform = rio.transform.from_origin(left, top, *ddem_ds.res)
    transform = ddem_ds.transform
    window = rio.windows.from_bounds(*ddem_ds.bounds, transform=transform)

    # stable_ground = stable_ground_ds.read(window=window, masked=True).filled(0) == 1
    # slope = slope_ds.read(window=window, masked=True).filled(np.nan)
    # curvature = curvature_ds.read(window=window, masked=True).filled(np.nan)
    # ddem = ddem_ds.read(window=window, masked=True).filled(np.nan)

    print(f"{datetime.datetime.now()} Generating error field.")
    error = np.empty_like(data["slope"], dtype="float32")
    for row in tqdm(np.arange(data["slope"].shape[0]), desc="Applying model"):
        error[row, :] = error_model((data["curvature"][row, :], data["slope"][row, :]))

    # error = error_model((data["curvature"], data["slope"])).reshape(data["slope"].shape)

    # There's been a problem with the lower part of the raster being nodata.
    assert np.isfinite(error[34180, 31822])

    del data["curvature"]
    del data["slope"]

    meta = ddem_ds.meta.copy()
    meta.update(
        {
            "transform": transform,
            "count": 1,
            "compress": "LZW",
            "width": window.width,
            "height": window.height,
            "BIGTIFF": "YES",
            "nodata": -9999,
        }
    )
    print(f"{datetime.datetime.now()} Writing dDEM error raster")
    with rio.open(terradem.files.TEMP_FILES["ddem_error"], "w", **meta) as raster:
        raster.write(error.squeeze(), 1)

    # Validate that the nodata issue does not exist (that particular pixel should have a value)
    with rio.open(terradem.files.TEMP_FILES["ddem_error"]) as raster:
        sample = raster.sample([(617523, 113297)]).__next__()[0]
        assert sample != -9999

    # Standardize by the error, remove snow/ice values, and remove large outliers.
    standardized_dh = np.where(~data["stable_ground"], np.nan, data["ddem"] / error)
    standardized_dh[np.abs(standardized_dh) > (4 * xdem.spatial_tools.nmad(standardized_dh))] = np.nan

    del data
    standardized_std = np.nanstd(standardized_dh)

    norm_dh = standardized_dh / standardized_std

    del standardized_dh

    norm_dh = skimage.measure.block_reduce(norm_dh, (2, 2), func=np.nanmean)

    # xcoords, ycoords = xdem.coreg._get_x_and_y_coords(norm_dh.shape, ddem_ds.transform)

    # This may fail due to the randomness of the analysis, so try to run this five times
    print(f"{datetime.datetime.now()} Sampling empirical variogram")
    for i in range(5):
        try:
            variogram = xdem.spatialstats.sample_empirical_variogram(
                values=norm_dh,
                gsd=ddem_ds.res[0] * 2,
                subsample=30,
                n_variograms=10,
                runs=30,
            )
        except ValueError as exception:
            if i == 4:
                raise exception
            continue
        break
    variogram[["exp", "err_exp"]] *= standardized_std

    vgm_model, params = xdem.spatialstats.fit_sum_model_variogram(["Sph", "Sph"], variogram)
    xdem.spatialstats.plot_vgm(
        variogram,
        xscale_range_split=[100, 1000, 10000],
        list_fit_fun=[vgm_model],
        list_fit_fun_label=["Standardized double-range variogram"],
    )
    plt.savefig("temp_variogram.jpg", dpi=600)

    neffs = pd.Series(dtype=float)
    for area in np.linspace(0.01e6, 100e6, num=100):
        neff = xdem.spatialstats.neff_circ(area, [(params[0], "Sph", params[1]), (params[2], "Sph", params[3])])

        neffs[area] = neff

    variogram.to_csv(terradem.files.TEMP_FILES["topographic_error_variogram"])
    neffs.to_csv("temp/n_effective_samples.csv")

    print(np.nanmean(error))


def _extrapolate_point(point_1: tuple[float, float], point_2: tuple[float, float]) -> tuple[float, float]:
    """Create a point extrapoled in p1->p2 direction."""
    # p1 = [p1.x, p1.y]
    # p2 = [p2.x, p2.y]
    extrap_ratio = 10
    return (
        point_1[0] + extrap_ratio * (point_2[0] - point_1[0]),
        point_1[1] + extrap_ratio * (point_2[1] - point_1[1]),
    )


def _iter_geom(geometry: BaseGeometry) -> Iterable[BaseGeometry]:
    """
    Return an iterable of the geometry.

    Use case: If 'geometry' is either a LineString or a MultiLineString. \
            Only MultiLineString can be iterated over normally.
    """
    if "Multi" in getattr(geometry, "geom_type", "") or getattr(geometry, "geom_type", "") == "GeometryCollection":
        return geometry

    return [geometry]


@overload
def glacier_outline_error(plot: None) -> list[float]: ...

@overload
def glacier_outline_error(plot: int) -> None: ...

def glacier_outline_error(plot: int | None = None) -> list[float]:
    """
    Calculate the error of the glacier outlines by comparing with the sparse "reference" outlines.

    There are two products in use:
        - Digitized LK50 glacier polygons: (presumably) from the same source data, but their quality is unknown.
        - Orthomosaic-drawn glacier outlines: Accurate in relation to the DEMs, but sparse (about 60/3000 glaciers)

    First, 50+ lines are drawn between the centroid to points along the ortho-outlines.
    Then, a duplicate line is cut from the farthest intersecting point of the LK50 polygon.

    The length difference between the two versions of the lines are the output data.
    """
    # Load the data
    lk50 = gpd.read_file(terradem.files.INPUT_FILE_PATHS["lk50_outlines"])
    ortho_digitized = gpd.read_file(terradem.files.INPUT_FILE_PATHS["digitized_outlines"]).to_crs(lk50.crs)

    # Filter the data so only the ones that exist in each respective dataset are kept.
    ortho_digitized = ortho_digitized[ortho_digitized["sgi-id"].isin(lk50["SGI"])]
    lk50 = lk50[lk50["SGI"].isin(ortho_digitized["sgi-id"])]

    # Initialize the output residuals list, aka the length difference for each line.
    residuals: list[float] = []
    j = 0
    for _, ortho_subset in ortho_digitized.groupby("sgi-id", as_index=False):
        lk50_subset = lk50.loc[lk50["SGI"] == ortho_subset["sgi-id"].iloc[0]]

        # Extract the single or multiple polygon(s) from the LK50 data.
        lk50_glacier = (
            lk50_subset.geometry.values[0]
            if lk50_subset.shape[0] == 1
            else shapely.geometry.MultiPolygon(lk50_subset.geometry.values)
        )
        # Extract the single or multiple line(s) from the ortho-drawn data.
        ortho_glacier = (
            ortho_subset.geometry.values[0]
            if ortho_subset.shape[0] == 1
            else shapely.geometry.MultiLineString(ortho_subset.geometry.values)
        )

        centroid = lk50_glacier.centroid

        # Draw points along the ortho-drawn line(s) to measure distances with.
        points: list[shapely.geometry.Point] = []
        for line in _iter_geom(ortho_glacier):
            for i in np.linspace(0, line.length):
                points.append(line.interpolate(i))

        if plot is not None and plot == j:
            for poly in _iter_geom(lk50_glacier):
                plt.plot(*poly.exterior.xy, color="blue", linestyle="--")
            for line in _iter_geom(ortho_glacier):
                plt.plot(*line.xy, color="orange")

        # Loop over all points, draw a long line, then cut the line to get and compare lengths
        diffs = []
        for point in points:
            # This line will start at the centroid and end somewhere outside of the glacier
            line = shapely.geometry.LineString([centroid, _extrapolate_point(centroid.coords[0], point.coords[0])])

            intersections: list[shapely.geometry.Point] = []
            # For each glacier polygon, find all point-intersections to the long line
            for polyg in _iter_geom(lk50_glacier):
                for p in _iter_geom(line.intersection(shapely.geometry.LineString(list(polyg.exterior.coords)))):
                    if p.geom_type != "Point":
                        continue
                    intersections.append(p)

            # Extract the intersecting point that is farthest away from the centroid.
            farthest_lk50_intersection = sorted(
                intersections,
                key=lambda p: float(np.linalg.norm([p.x - centroid.x, p.y - centroid.y])),
                reverse=True,
            )[0]

            # The ortho-drawn line is simply the original point to the centroid
            ortho_line = shapely.geometry.LineString([centroid, point])
            # The lk50 line is the farthest intersection point to the centroid.
            lk50_line = shapely.geometry.LineString([centroid, farthest_lk50_intersection])

            # The difference in length between this line will be the error.
            diffs.append(ortho_line.length - lk50_line.length)

            if plot is not None and plot == j:
                plt.plot(*ortho_line.xy, color="red", linewidth=2)
                plt.plot(*lk50_line.xy, color="blue", linewidth=2)
                plt.plot(*min([ortho_line, lk50_line], key=lambda l: l.length).xy, color="black", linewidth=3, solid_capstyle="round")

        if plot is not None and plot == j:
            return
        residuals += diffs
        j += 1

    # Print and plot the results.
    nmad = xdem.spatialstats.nmad(residuals)
    #print(np.median(residuals), nmad)

    return residuals

    # plt.hist(residuals, bins=np.linspace(np.median(residuals) - 4 * nmad, np.median(residuals) + 4 * nmad, 50))
    # plt.show()


def _dissolve_exterior(geom_or_geoms: BaseGeometry | Iterable[BaseGeometry]) -> shapely.geometry.MultiPolygon | shapely.geometry.Polygon:
    geoms = []

    if isinstance(geom_or_geoms, Iterable):
        for geom in geom_or_geoms:
            geoms.append(_dissolve_exterior(geom))
    else:
        if geom_or_geoms.interiors:
            geoms.append(shapely.geometry.Polygon(list(geom_or_geoms.exterior.coords)))
        else:
            geoms.append(geom_or_geoms)

    return shapely.ops.unary_union(geoms)


def get_measurement_error() -> None:
    ddem_key = "ddem_coreg_tcorr_national-interp-extrap"

    lk50_outlines = gpd.read_file(terradem.files.INPUT_FILE_PATHS["lk50_outlines"])
    sgi2016 = gpd.read_file(terradem.files.INPUT_FILE_PATHS["sgi_2016"]).to_crs(lk50_outlines.crs)

    error_ds = rio.open("temp/stable_ground_error.tif")
    ddem_ds = rio.open(terradem.files.TEMP_FILES["ddem_coreg_tcorr_national-interp-extrap"])
    dem_ds = rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"])
    ddem_nointerp_ds = rio.open(terradem.files.TEMP_FILES["ddem_coreg_tcorr"])
    ddem_ideal_ds = rio.open(terradem.files.TEMP_FILES["ddem_coreg_tcorr_national-interp-extrap-ideal"])
    n_effective_samples = pd.read_csv(terradem.files.TEMP_FILES["n_effective_samples"], index_col=0, squeeze=True)

    temporal_error_model = terradem.massbalance.temporal_corr_error_model()
    start_and_end_year_model = terradem.massbalance.get_start_and_end_years()

    # TODO: Refer to the external analysis in a better way than hardcoding like this.
    interpolation_variogram = [(2.82930058e02, "Sph", 8.80780720e-02), (2.11817045e03, "Sph", 3.20393346e-01)]
    ddem_vs_ideal_error = pd.read_csv(terradem.files.TEMP_FILES["ddem_vs_ideal_error"], index_col=0).T[
        ddem_key + "-ideal"
    ]

    neff_model = scipy.interpolate.interp1d(n_effective_samples.index, n_effective_samples, fill_value="extrapolate")
    def interp_neff_model(a): return xdem.spatialstats.neff_circ(a, interpolation_variogram)

    median_outline_uncertainty = abs(np.median(glacier_outline_error()))

    # Temporary. Shuffle the outlines to have nicely representative subsets
    lk50_outlines = lk50_outlines.iloc[np.random.permutation(lk50_outlines.shape[0])]

    # result = pd.DataFrame(dtype=float)
    result_list: list[dict[str, object]] = []
    for sgi_id, outlines in tqdm(lk50_outlines.groupby("SGI", sort=False), total=lk50_outlines.shape[0], smoothing=0):
        bounds = outlines.bounds.iloc[0]
        bounds[["maxx", "maxy"]] += error_ds.res[0]
        bounds -= bounds % error_ds.res[0]

        window = rio.windows.from_bounds(*rio.coords.BoundingBox(*bounds), transform=error_ds.transform)
        transform = rio.transform.from_origin(*bounds[["minx", "maxy"]], *error_ds.res)

        error = error_ds.read(1, window=window, boundless=True, masked=True).filled(np.nan)
        # if np.count_nonzero(np.isfinite(error[mask])) == 0:
        #    continue
        ddem = ddem_ds.read(1, window=window, boundless=True, masked=True).filled(np.nan)
        ddem_nointerp = ddem_nointerp_ds.read(1, window=window, boundless=True, masked=True).filled(np.nan)
        # ddem_ideal = ddem_ideal_ds.read(1, window=window, boundless=True, masked=True).filled(np.nan)

        mask = rio.features.rasterize(outlines.geometry, out_shape=error.shape, fill=0, transform=transform) == 1

        larger_outline = outlines.geometry.buffer(median_outline_uncertainty)
        smaller_outline = outlines.geometry.buffer(-median_outline_uncertainty)

        larger_mask = rio.features.rasterize(larger_outline, out_shape=error.shape, fill=0, transform=transform) == 1
        smaller_mask = rio.features.rasterize(smaller_outline, out_shape=error.shape, fill=0, transform=transform) == 1

        area = outlines.geometry.area.sum()
        area_m2_error = abs(larger_outline.area.sum() - smaller_outline.area.sum()) / 2

        sgi_1973_ids = list(np.ravel([[outline["SGI1973"].split(",")] for _, outline in outlines.iterrows()]))
        sgi_ids = sgi_1973_ids.copy()
        for string in filter(lambda s: "-" in s, np.unique(sgi_ids)):
            start, end = string.split("-")
            sgi_ids.append(start + "-" + end.zfill(2))
        modern_outlines = sgi2016[sgi2016.geometry.centroid.intersects(
            outlines.geometry.values[0]) | sgi2016["sgi-id"].isin(sgi_ids)]
        modern_area = modern_outlines.area.sum()

        """
        polygons = shapely.ops.unary_union([shapely.geometry.Polygon(list(p.exterior.coords)) for p in (outlines.geometry.iloc[0] if "Multi" in outlines.geometry.iloc[0].geom_type else [outlines.geometry.iloc[0]])])
        if polygons.area > 1e7 and False:
            plt.title(sgi_id)
            for p in (polygons if "Multi" in polygons.geom_type else [polygons]):
                plt.plot(*p.exterior.xy)

            for geom in modern_outlines.geometry:
                for p in (geom if "Multi" in geom.geom_type else [geom]):
                
                    plt.plot(*p.exterior.xy, color="orange")
            plt.show()
            #outlines.plot()
        """

        gaps_percent = 100 * (
            1 - np.count_nonzero(np.isfinite(ddem_nointerp[mask])) / np.count_nonzero(np.isfinite(ddem[mask]))
        )

        interp_err = (gaps_percent / 100) * (
            np.abs(ddem_vs_ideal_error["median"]) + ddem_vs_ideal_error["nmad"] / np.sqrt(interp_neff_model(area))
        )

        area_error = abs((np.nanmean(ddem[larger_mask]) - np.nanmean(ddem[smaller_mask])) / 2)

        topographic_error = 0.0 if gaps_percent == 100 else np.nanmean(error[mask]) / np.sqrt(neff_model(area))

        diff = np.nanmean(ddem[mask])

        easting = np.mean([bounds["minx"], bounds["maxx"]])
        northing = np.mean([bounds["miny"], bounds["maxy"]])

        dem = dem_ds.read(1, window=window, boundless=True, masked=True).filled(np.nan)[mask]

        temporal_error = np.abs(diff) * temporal_error_model(easting, northing)

        start_year, end_year = start_and_end_year_model(easting, northing)

        result_list.append(
            {
                "sgi_id": sgi_id,
                "easting": easting,
                "northing": northing,
                "dh": diff,
                "start_area": area,
                "start_area_m2_err": area_m2_error,
                "end_area": modern_area,
                "start_year": start_year,
                "end_year": end_year,
                "gaps_percent": gaps_percent,
                "interp_err": interp_err,
                "area_err": area_error,
                "topo_err": topographic_error,
                "time_err": temporal_error,
                "max_elev": np.nanmax(dem),
                "min_elev": np.nanmin(dem),
                "med_elev": np.nanmedian(dem),
                "sgi_1973_ids": ",".join(sgi_1973_ids),
                "sgi_2016_ids": ",".join(modern_outlines["sgi-id"].unique()),
            }
        )

    result = pd.DataFrame(result_list).set_index("sgi_id")

    result["density_err"] = result["dh"].abs() * terradem.massbalance.ICE_DENSITY_ERROR

    result["dm_tons_we"] = result["dh"] * result["start_area"] * terradem.massbalance.ICE_DENSITY_CONVERSION
    result["dh_m_we"] = result["dm_tons_we"] / (result[["start_area", "end_area"]].mean(axis=1))

    result["dh_err"] = np.sqrt((result[["interp_err", "topo_err", "area_err", "time_err"]] ** 2).sum(axis=1))
    result["dh_err_mwe"] = np.sqrt(((result[["interp_err", "topo_err", "area_err", "time_err"]] *
                                     terradem.massbalance.ICE_DENSITY_CONVERSION) ** 2).sum(axis=1) + result["density_err"] ** 2)
    # result["dv"] = result["dh"] * result["area"]
    # result["dv_err"] = result["dh_err"] * result["area"]

    weighted_avg = pd.Series(np.average(result.select_dtypes(include=np.number), axis=0,
                                        weights=result["start_area"]), result.select_dtypes(include=np.number).columns)

    regional_err = pd.Series(
        {
            "area_err": weighted_avg["area_err"],
            "time_err": weighted_avg["time_err"],
            "topo_err": (weighted_avg["topo_err"] * np.sqrt(neff_model(weighted_avg["start_area"])))
            / np.sqrt(neff_model(result["start_area"].sum())),
            "interp_err": (weighted_avg["gaps_percent"] / 100)
            * (+ddem_vs_ideal_error["nmad"] / np.sqrt(interp_neff_model(result["start_area"].sum()))),
        }
    )
    regional_err["dh_err"] = np.sqrt((regional_err ** 2).sum())
    regional_err["density_err"] = weighted_avg["density_err"]
    regional_err["dh_err_mwe"] = np.sqrt((regional_err ** 2).sum())
    regional_err[["dh_err", "interp_err", "dh_err_mwe"]
                 ] += (weighted_avg["gaps_percent"] / 100) * abs(ddem_vs_ideal_error["median"])

    pd.set_option("display.float_format", lambda x: "%.4f" % x)
    print("Weighted average")
    print(weighted_avg)
    print("\n\nRegional error")
    print(regional_err)

    result.sort_values("start_area", inplace=True)

    midpoint = result.shape[0] // 2

    print("\n\nSmallest")
    print(result.iloc[:5].to_string())
    print("\n\nMiddle")
    print(result.iloc[midpoint: midpoint + 5].to_string())
    print("\n\nLargest")
    print(result.iloc[-5:].to_string())

    # print(np.average(result[], weights=result["area"]), np.average(result["dh_err"], weights=result["area"]))

    print(
        1e-6 * (result["end_area"].sum() - result["start_area"].sum())
        / (weighted_avg["end_year"] - weighted_avg["start_year"]),
        "+-",
        1e-6 * result["start_area_m2_err"].sum() / (weighted_avg["end_year"] - weighted_avg["start_year"]),
    )

    print(result[["start_area", "start_area_m2_err", "end_area"]].sum(axis=0))

    result.to_csv(terradem.files.TEMP_FILES["glacier_wise_dh"])


def interpolation_error() -> None:
    ddem_key = "ddem_coreg_tcorr_national-interp-extrap"

    cache_filepath = "temp/interpolation_error_cache.pkl"

    ddem_ds = rio.open(terradem.files.TEMP_FILES[ddem_key])
    if not os.path.isfile(cache_filepath):
        ddem_nointerp_ds = rio.open(terradem.files.TEMP_FILES["ddem_coreg_tcorr"])

        print("Reading LK50 outlines")
        lk50_rasterized = rio.open(terradem.files.TEMP_FILES["lk50_rasterized"]).read(1, masked=True).filled(0)

        print("Reading and subtracting dDEMs")
        ddem_diff = ddem_ds.read(1, masked=True).filled(np.nan) - ddem_nointerp_ds.read(1, masked=True).filled(np.nan)

        print("Reading base DEM")
        dem = rio.open(terradem.files.INPUT_FILE_PATHS["base_dem"]).read(1, masked=True).filled(np.nan)
        # Set all periglacial values to nan
        dem[lk50_rasterized == 0] = np.nan

        for i in tqdm(np.unique(lk50_rasterized)):
            if i == 0:
                continue
            mask = lk50_rasterized == i
            glacier_values = dem[mask]

            minh = np.nanmin(glacier_values)
            maxh = np.nanmax(glacier_values)

            dem[mask] = (glacier_values - minh) / (maxh - minh)

        del glacier_values, mask, lk50_rasterized

        with open(cache_filepath, "wb") as outfile:
            pickle.dump((ddem_diff, dem), outfile)
    else:
        with open(cache_filepath, "rb") as infile:
            ddem_diff, dem = pickle.load(infile)

    valid_mask = np.isfinite(ddem_diff) & np.isfinite(dem)

    bins = np.r_[[-0.1], np.linspace(0, 1, 10).round(2), [1.1]]

    categories = np.digitize(dem[valid_mask], bins)

    values = pd.DataFrame()
    for i in np.unique(categories):
        diffs = ddem_diff[valid_mask][categories == i]
        values.loc[pd.Interval(bins[i - 1] if i != 0 else 0, bins[i]), ["median", "mean", "nmad", "count"]] = (
            np.median(diffs),
            np.mean(diffs),
            xdem.spatialstats.nmad(diffs),
            diffs.size,
        )

    del ddem_diff

    """
    plt.errorbar(values.index.mid, values["mean"], values["nmad"])
    plt.scatter(values.index.mid, values["mean"])
    xlim = plt.gca().get_xlim()
    plt.hlines(0, *xlim, colors="k", linestyles="--")
    plt.xlim(*xlim)

    plt.ylabel("dH - dH_ideal (m/a)")
    plt.xlabel("Normalized glacier elevation")
    plt.savefig("temp_fig.jpg", dpi=300)
    """

    error_model = scipy.interpolate.interp1d(
        x=values.index.mid,
        y=values["nmad"],
        fill_value="extrapolate",
    )

    error_field = np.empty_like(dem)

    for row in trange(error_field.shape[0], desc="Applying model", smoothing=0):
        error_field[row, :] = error_model(dem[row, :])

    meta = ddem_ds.meta.copy()
    meta.update({"compress": "deflate", "tiled": True})
    with rio.open("temp/interpolation_error.tif", "w", **meta) as raster:
        raster.write(error_field, 1)


def total_error() -> None:
    """Derive the total integrated error when averaging over the entire region."""
    # Load the empirical variogram and (re-)estimate a double spherical model
    variogram = pd.read_csv("temp/variogram.csv", index_col=0).rename(columns={"bins.1": "bins"})
    _, params = xdem.spatialstats.fit_sum_model_variogram(["Sph", "Sph"], empirical_variogram=variogram)
    ranges = params[np.arange(params.size) % 2 == 0]
    sills = params[np.arange(params.size) % 2 != 0]

    # Load a pre-calculated (circular) area-vs-neff relationship
    n_effective_samples = pd.read_csv("temp/n_effective_samples.csv", index_col=0, squeeze=True)
    # Construct a continuous model that takes areas as arguments.
    neff_model = scipy.interpolate.interp1d(n_effective_samples.index, n_effective_samples, fill_value="extrapolate")

    # Read glacier outlines (for the glacier location data)
    outlines = gpd.read_file(terradem.files.INPUT_FILE_PATHS["lk50_outlines"]).set_index("SGI")

    # Load the glacier-wise (dH and) dH error table
    glacier_wise_dh = pd.read_csv("temp/glacier_wise_dh.csv", index_col=0)

    dv = (glacier_wise_dh["dh"] * glacier_wise_dh["area"]).sum()
    dv_err = np.sqrt(np.sum(glacier_wise_dh["dh_err"] ** 2 * glacier_wise_dh["area"] ** 2))

    specific_dh = dv / glacier_wise_dh["area"].sum()
    dh_err = dv_err / glacier_wise_dh["area"].sum()

    print(specific_dh, dh_err)

    return

    # Get the latitudes and longitudes for each glacier with values
    latlons = outlines.loc[glacier_wise_dh.index].geometry.centroid.to_crs(4326)

    errors = []
    # For each glacier, integrate its error at different correlation ranges from the vgm model
    for _, dh_df in glacier_wise_dh.iterrows():

        # Back-track the individual pixel's error by multiplying with its effective sample nr.
        pixel_wise_err = dh_df["dh_err"] * np.sqrt(neff_model(dh_df["area"]))

        glacier_error = []
        # For each range, calculate the respective integrated error for the glacier
        for i in range(ranges.size):
            neff = xdem.spatialstats.neff_circ(dh_df["area"], [(ranges[i], "Sph", sills[i])])
            glacier_error.append(pixel_wise_err / np.sqrt(neff))

        errors.append(glacier_error)

    total = xdem.spatialstats.double_sum_covar(
        errors,
        ranges,
        glacier_wise_dh["area"].values,
        latlons.y,
        latlons.x,
        nproc=10,
    )

    print(total)


def ddem_error():
    """
    Generate an error field by assigning the NMAD of the stable ground offsets for each pixel.

    For overlapping dDEMs, the squared sum of all overlapping dDEM errors is assigned.
    """
    # Extract the target metadata and bounds from the dDEM mosaic
    with rio.open(terradem.files.TEMP_FILES["ddem_coreg_tcorr"]) as raster:
        target_bounds = raster.bounds
        target_meta = raster.meta

    # Get a list of all paths to the individual dDEMs
    ddem_filepaths = [
        fp
        for fp in pathlib.Path(terradem.files.TEMP_SUBDIRS["ddems_coreg_tcorr"]).iterdir()
        if str(fp).endswith(".tif")
    ]
    # Load the precalculated dDEM statistics and extract the stable ground NMAD for each dDEM
    ddem_metadata = pd.read_csv(
        terradem.files.TEMP_FILES["ddem_stats"], index_col=["station", "product", "surface"]
    ).loc[(slice(None), "coreg", "stable"), "nmad"]

    # Instantiate the output error array and the dDEMs-per-pixel count raster
    error_arr = np.zeros((target_meta["height"], target_meta["width"]), "float32")
    counts = np.zeros(error_arr.shape, "uint8")

    write_lock = threading.Lock()
    progress_bar = tqdm(total=len(ddem_filepaths))

    def process_ddem(ddem_path):
        """Load a dDEM and assign its error where finite pixels exist in a thread-safe approach."""
        with rio.open(ddem_path) as raster:
            # The dDEM may extend farther than the mosaic, so find the intersecting part only.
            intersecting_bounds = rio.coords.BoundingBox(
                max(target_bounds.left, raster.bounds.left),
                max(target_bounds.bottom, raster.bounds.bottom),
                min(target_bounds.right, raster.bounds.right),
                min(target_bounds.top, raster.bounds.top),
            )

            # Read the dDEM (within the intersection only) and find where finite pixels exist.
            valid_mask = np.isfinite(
                raster.read(
                    1, masked=True, window=rio.windows.from_bounds(*intersecting_bounds, transform=raster.transform)
                ).filled(np.nan)
            )

        # Create slice objects for the dDEM to assign the mosaic
        window = rio.windows.from_bounds(*intersecting_bounds, transform=target_meta["transform"])
        row_slice = slice(int(window.row_off), int(window.row_off + window.height))
        col_slice = slice(int(window.col_off), int(window.col_off + window.width))

        with write_lock:
            # Add the NMAD (going toward the squared sum) of the finite pixels
            error_arr[row_slice, col_slice][valid_mask] += (
                float(ddem_metadata[ddem_path.stem.replace("_ddem", "")]) ** 2
            )
            # Increment the counter at the valid pixels
            counts[row_slice, col_slice][valid_mask] += 1
        progress_bar.update(1)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(process_ddem, ddem_filepaths))
    progress_bar.close()

    # Derive the squared sum average for all pixels.
    with np.errstate(divide="ignore", invalid="ignore"):
        error_arr = np.sqrt(error_arr) / counts

    target_meta.update({"compress": "deflate", "tiled": True, "co_opts": {"NUM_THREADS": "ALL_CPUS"}})
    # Write the squared sum average raster
    with rio.open("temp/stable_ground_error.tif", "w", **target_meta) as raster:
        raster.write(error_arr, 1)
