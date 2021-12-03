from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.patches
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import rasterio as rio
import shapely.geometry
import shapely.ops
import skimage.exposure
import sklearn
import sklearn.pipeline
from tqdm import tqdm

import terradem.climate
import terradem.error
import terradem.files
import terradem.massbalance
import terradem.orthorectification
import terradem.utilities
import xdem

DH_VLIM = 4
DH_COLORS = [
    (-DH_VLIM, "#400912"),
    (-3, "#590C19"),
    (-2, "#95142A"),  # crimson
    (-0.6, "#E56B1A"),  # rust red
    (-0.3, "#E5BD1A"),  # butterscotch
    (-0.15, "#F4D780"),  # Sand
    (0, "lightgray"),
    (0.5, "royalblue"),
    (DH_VLIM, "royalblue"),
]


DH_NORMALIZER = matplotlib.colors.Normalize(vmin=-DH_VLIM, vmax=DH_VLIM, clip=True)
DH_CMAP = matplotlib.cm.ScalarMappable(
    norm=DH_NORMALIZER,
    cmap=matplotlib.colors.LinearSegmentedColormap.from_list("dh", [(DH_NORMALIZER(a), b) for a, b in DH_COLORS]),
)

MB_COLORS = [(v, DH_COLORS[i][1]) for i, v in enumerate([-1.3, -0.9, -0.7, -0.5, -0.2, -0.1, 0, 0.5, 1])]
MB_NORMALIZER = matplotlib.colors.Normalize(vmin=MB_COLORS[0][0], vmax=MB_COLORS[-1][0], clip=True)
MB_CMAP = matplotlib.cm.ScalarMappable(
    norm=MB_NORMALIZER,
    cmap=matplotlib.colors.LinearSegmentedColormap.from_list("mb", [(MB_NORMALIZER(a), b) for a, b in MB_COLORS]),
)
DH_UNIT = "ma⁻¹"
DH_MWE_UNIT = "m w.e. a⁻¹"
DH_MWE_LABEL = r"dHdt$^{-1}$ (ma$^{-1}$ w.e.)"


IMAGE_EXAMPLE_BOUNDS = rio.coords.BoundingBox(left=6.70e5, bottom=1.567e5, right=6.76e5, top=1.68e5)

INTERPOLATION_BEFORE_AFTER_BOUNDS = rio.coords.BoundingBox(632760, 132950, 665800, 174560)
LK50_EXAMPLE_BOUNDS = rio.coords.BoundingBox(left=628150, right=633160, bottom=100800, top=105000)
OUTLINE_ERROR_BOUNDS = rio.coords.BoundingBox(left=659300, right=661000, bottom=160200, top=162000)


def colorbar(
    cmap: matplotlib.cm.ScalarMappable = DH_CMAP,
    axis: plt.Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    loc: tuple[float, float] = (0.1, 0.6),
    width: float = 0.05,
    height: float = 0.2,
    label: str = "",
    labelpad: float | int = 14,
    tick_right: bool = False,
    rotate_ticks: bool = True,
) -> plt.Axes:

    vmin = vmin or cmap.norm.vmin
    vmax = vmax or cmap.norm.vmax
    axis = axis or plt.gca()

    a: plt.Axes = plt.gca()
    inset = a.inset_axes([*loc, width, height], transform=a.transAxes)

    steps = np.linspace(vmin, vmax, 255)
    width = 0.05
    for i in steps:
        inset.add_patch(plt.Rectangle((0, i), width, steps[1] - steps[0], edgecolor="none", facecolor=cmap.to_rgba(i)))

    # axis.add_patch(plt.Rectangle((0, DH_NORMALIZER.vmax), width, steps[-1] - steps[0], edgecolor="k", facecolor="none"))
    inset.set_ylim(vmin, vmax)
    inset.set_xlim(0, width)
    inset.set_xticks([])

    ylabel_props = {"fontsize": 10, "bbox": dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.8), "labelpad": labelpad}

    if tick_right:
        inset.yaxis.set_label_position("right")
        # inset.set_ylabel("dHdt$^{-1}$ (ma$^{-1}$ w.e.)", rotation=270, labelpad=14, **ylabel_props)
        inset.set_ylabel(label, rotation=90, **ylabel_props)
        inset.yaxis.tick_right()
    else:
        inset.set_ylabel(label,  **ylabel_props)

    if rotate_ticks:
        inset.tick_params(axis="y", labelrotation=90)
        for tick in inset.get_yticklabels():
            tick.set_verticalalignment("center")

    return inset


def topographic_error():

    error = pd.read_csv(terradem.files.TEMP_FILES["topographic_error_df"], index_col=0)
    # error = error[(error["nd"] == 1) & (error["count"] > 200)]
    error = error.applymap(
        lambda cell: pd.Interval(*map(float, cell.replace("[", "").replace(")", "").split(", ")))
        if "[" in str(cell)
        else cell
    )

    error = error[
        error["curvature"].isna() | (np.abs(error["curvature"].apply(lambda i: i if pd.isna(i) else i.mid)) < 15)
    ]
    error = error[error["slope"].isna() | (np.abs(error["slope"].apply(lambda i: i if pd.isna(i) else i.mid)) < 60)]

    slopes = error[error["slope"].notna()].set_index("slope")
    slopes.index = pd.IntervalIndex.from_arrays(np.clip(slopes.index.left, 0, 90), np.clip(slopes.index.right, 0, 90))

    curvatures = error[error["curvature"].notna()].set_index("curvature")

    xdem.spatialstats.plot_2d_binning(
        error, "slope", "curvature", "nmad", "Slope (degrees)", "Total curvature (100 m$^{-1}$)", "NMAD of dh (m)"
    )

    plt.savefig("temp/figures/topographic_error.jpg")
    return

    plt.figure(figsize=(8, 5), dpi=300)
    plt.subplot(121)
    plt.plot(curvatures.index.mid, curvatures["nmad"])
    plt.scatter(curvatures.index.mid, curvatures["nmad"], marker="x")
    plt.xlabel(r"Total curvature ($100 \times m^{-1}$)")
    plt.xlim(-20, 80)
    plt.ylabel("NMAD (m)")

    plt.subplot(122)
    plt.plot(slopes.index.mid, slopes["nmad"], color="red")
    plt.scatter(slopes.index.mid, slopes["nmad"], marker="x", color="red")
    plt.xlabel(r"Slope ($\degree$)")
    plt.ylabel("NMAD (m)")

    plt.tight_layout()

    print(100 * curvatures["count"] / curvatures["count"].sum())


def error_histograms():

    data = pd.read_csv(terradem.files.TEMP_FILES["glacier_wise_dh"])

    fig = plt.figure(figsize=(8, 5), dpi=200)

    # for i, col in enumerate(filter(lambda s: "err" in s, data.columns), start=1):
    # plt.subplot(3, 2, i)
    # plt.title(col)
    # plt.hist(data[col], bins=np.linspace(0, 0.5), )

    columns = [col for col in filter(lambda s: "err" in s and s != "dh_err", data.columns)]
    names = {
        "topo_err": "Stable-ground error",
        "time_err": "Temporal error",
        "area_err": "Area error",
        "interp_err": "Interpolation error",
    }
    plt.hist(data[columns], bins=np.linspace(0, 0.4, 20), label=[names[col] for col in columns])
    plt.ylabel("Count")
    plt.xlabel(r"dH dt$^{-1}$ (m a$^{-1})$")
    plt.legend()

    plt.tight_layout()
    # plt.savefig("temp/figures/error_histograms.jpg")
    plt.show()


def topographic_error_variogram():
    vgm = pd.read_csv(terradem.files.TEMP_FILES["topographic_error_variogram"]).drop(columns=["bins.1"])
    vgm = vgm[vgm["bins"] < 2e5]

    vgm_model, params = xdem.spatialstats.fit_sum_model_variogram(["Sph"] * 2, vgm)

    xdem.spatialstats.plot_vgm(
        vgm,
        xscale_range_split=[100, 1000, 10000],
        list_fit_fun=[vgm_model],
        list_fit_fun_label=["Standardized double-range variogram"],
    )
    fig: plt.Figure = plt.gcf()

    fig.set_size_inches(8, 5)

    for axis in plt.gcf().get_axes():
        if axis.get_legend():
            axis.legend().remove()

    print(params)

    plt.savefig("temp/figures/topographic_error_variogram.jpg", dpi=300)


def temporal_correction():

    tcorr_meta = pd.DataFrame(
        map(
            lambda p: json.load(open(p)),
            filter(lambda p: "json" in str(p), Path(terradem.files.TEMP_SUBDIRS["tcorr_meta_coreg"]).iterdir()),
        )
    )

    tcorr_meta["start_date"] = pd.to_datetime(tcorr_meta["start_date"])
    tcorr_meta = tcorr_meta[tcorr_meta["n_glacier_values"] > 0]
    tcorr_meta["start_year"] = (
        tcorr_meta["start_date"].dt.year
        + tcorr_meta["start_date"].dt.month / 12
        + tcorr_meta["start_date"].dt.day / 365.2425
    )

    years = np.arange(int(tcorr_meta["start_year"].min()), int(np.ceil(tcorr_meta["end_year"].max())) + 1)

    plt.subplot(211)

    corrections = terradem.massbalance.read_mb_index().cumsum(axis=0)

    plt.plot(corrections, label=corrections.columns)
    plt.xlim(years.min() - 1, years.max() + 2)
    plt.ylabel("Cumulative SMB (m w.e.)")
    plt.xticks(plt.gca().get_xticks(), labels=[""] * len(plt.gca().get_xticks()))
    plt.legend(ncol=4)

    plt.subplot(212)
    plt.hist(tcorr_meta["start_year"], bins=years, label="Start year")
    plt.hist(tcorr_meta["end_year"], bins=years, label="End year")
    plt.xlim(years.min() - 1, years.max() + 2)
    plt.ylabel("Count")
    plt.xlabel("Year")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(tcorr_meta)


def outline_error(show: bool = True):

    fig = plt.figure(figsize=(8.3, 3.8))

    bounds = LK50_EXAMPLE_BOUNDS
    axis = plt.subplot(1, 2, 1)
    lk50_url = "/remotes/haakon/Gammalt/maud/maud/Data/SwissTerra/basedata/LK50_first_edition_compilation/LK50_first_edition_compilation.vrt"
    with rio.open(lk50_url) as raster:
        lk50_raster = np.moveaxis(
            raster.read(window=rio.windows.from_bounds(*bounds, transform=raster.transform)), 0, -1
        )

    digitized_outlines = gpd.read_file(terradem.files.INPUT_FILE_PATHS["digitized_outlines"])

    def imshow(axis):
        axis.imshow(lk50_raster, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
        digitized_outlines.plot(edgecolor="red", facecolor="none", linestyle="--", ax=axis)

    imshow(axis)
    plt.text(
        0.008,
        0.99,
        "A)",
        transform=plt.gca().transAxes,
        fontsize=12,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.2),
    )

    plt.xlim(bounds.left, bounds.right)
    plt.ylim(bounds.bottom, bounds.top)
    yticks = plt.gca().get_yticks()[[2, -2]]
    plt.yticks(yticks, (yticks + 1e6).astype(int), rotation=90, va="center")
    xticks = plt.gca().get_xticks()[[2, -3]]
    plt.xticks(xticks, (xticks + 2e6).astype(int))
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position("top")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")

    axis_xmin = -0.03
    axis_size = 0.4
    inset = axis.inset_axes([axis_xmin, 0.0, axis_size, axis_size], transform=axis.transAxes)
    imshow(inset)
    size = 400
    ymin = 103550
    xmin = 629200
    inset.set_xlim(xmin, xmin + size)
    inset.set_ylim(ymin, ymin + size)
    inset.set_xticks([])
    inset.set_yticks([])

    ax_to_coord = lambda coord: axis.transData.inverted().transform(axis.transAxes.transform(coord))

    plot_line = lambda c_from, c_to: axis.plot(
        [ax_to_coord(c_from)[0], c_to[0]], [ax_to_coord(c_from)[1], c_to[1]], color="k", linestyle=":"
    )

    plot_line((axis_xmin, axis_size), (xmin, ymin + size))
    plot_line((axis_size + axis_xmin * 2, axis_size), (xmin + size, ymin + size))
    plot_line((axis_size + axis_xmin * 2, 0), (xmin + size, ymin))
    plot_line((axis_xmin, 0), (xmin, ymin))
    axis.add_patch(plt.Rectangle((xmin, ymin), size, size, facecolor="none", edgecolor="k"))

    plt.subplot(1, 2, 2)
    bounds = OUTLINE_ERROR_BOUNDS
    with rio.open(lk50_url) as raster:
        lk50_raster = np.mean(raster.read(window=rio.windows.from_bounds(*bounds, transform=raster.transform)), axis=0)
    plt.imshow(lk50_raster, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], cmap="Greys_r")
    terradem.error.glacier_outline_error(plot=14)
    plt.xlim(OUTLINE_ERROR_BOUNDS.left, OUTLINE_ERROR_BOUNDS.right)
    plt.ylim(OUTLINE_ERROR_BOUNDS.bottom, OUTLINE_ERROR_BOUNDS.top)
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)", rotation=270)
    plt.yticks(
        plt.gca().get_yticks()[[1, -1]],
        labels=(plt.gca().get_yticks()[[1, -1]] + 1e6).astype(int),
        rotation=270,
        va="center",
    )
    xticks = plt.gca().get_xticks()[[1, -2]]
    plt.xticks(xticks, labels=(xticks + 2e6).astype(int))
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position("top")
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.text(
        0.008,
        0.99,
        "B)",
        transform=plt.gca().transAxes,
        fontsize=12,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.2),
    )

    plt.tight_layout()
    plt.savefig("temp/figures/outline_error.jpg", dpi=600)

    if show:
        plt.show()


def elevation_change_histograms():
    data = pd.read_csv(terradem.files.TEMP_FILES["glacier_wise_dh"])

    fig = plt.figure(figsize=(8, 3.5), dpi=200)

    plt.subplot(121)
    plt.hist(data["dh"], bins=np.linspace(-3, 3, 100), color="black")
    plt.ylabel("Count")
    plt.xlabel(r"dH dt$^{-1}$ (m a$^{-1})$")

    # data["alpha"] = np.clip(np.log10(data["area"]) / np.log10(data["area"].max()) * 0.1, 0, 1)
    plt.subplot(122)
    data["area_log10"] = np.log10(data["area"])
    areas = np.linspace(0, np.ceil(data["area_log10"].max()), 10)
    values: list[np.ndarray] = []
    for i in range(areas.size - 1):
        values.append(data.loc[(areas[i] <= data["area_log10"]) & (data["area_log10"] < areas[i + 1]), "dh"].values)

    plt.boxplot(
        values, positions=areas[:-1] + np.diff(areas), widths=np.diff(areas), manage_ticks=False, showfliers=False
    )

    plt.xticks(ticks=plt.gca().get_xticks(), labels=[f"$10^{int(area)}$" for area in plt.gca().get_xticks()])
    plt.ylabel(DH_MWE_LABEL)
    plt.xlabel("Area (m²)")
    plt.tight_layout()

    plt.savefig("temp/figures/dh_histograms.jpg", dpi=300)
    plt.show()


def plot_base_dem_slope(axis: plt.Axes | None = None):

    axis = axis or plt.gca()

    with rio.open(terradem.files.TEMP_FILES["base_dem_slope"], overview_level=3) as raster:
        slope = raster.read(1, masked=True)
        bounds = raster.bounds
        outline = gpd.GeoSeries(
            [
                shapely.geometry.shape(l[0])
                for l in rio.features.shapes(slope.mask.astype("uint8"), transform=raster.transform)
                if l[1] == 0
            ],
            crs=raster.crs,
        )

    axis.imshow(
        slope, cmap="Greys", extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], interpolation="bilinear"
    )


def plot_lk50_glaciers(axis: plt.Axes | None = None, lk50_outlines: gpd.GeoDataFrame | None = None) -> None:
    axis = axis or plt.gca()
    lk50_outlines = (
        lk50_outlines if lk50_outlines is not None else gpd.read_file(terradem.files.INPUT_FILE_PATHS["lk50_outlines"])
    )

    lk50_outlines["subregion"] = lk50_outlines["SGI"].str.slice(stop=2)

    lk50_outlines.plot(ax=axis, edgecolor="black", lw=0.1)  # , column="subregion")


def overview(show: bool = True):

    climate = terradem.climate.mean_climate_deviation(slice(1961, 1990))
    fig = plt.figure(figsize=(8, 4.3))

    image_meta = pd.read_csv(terradem.files.INPUT_FILE_PATHS["swisstopo_metadata"])
    image_meta["year"] = pd.to_datetime(image_meta["date"]).dt.year

    # bins = np.r_[[image_meta["year"].min()], np.percentile(image_meta["year"], [25, 50, 75]), [image_meta["year"].max()]].astype(int)
    bins = np.linspace(image_meta["year"].min(), image_meta["year"].max(), 5).astype(int)

    image_meta["year_bin"] = np.digitize(image_meta["year"], bins)
    # Make it so that the last bin includes the max year
    image_meta.loc[image_meta["year"] == bins.max(), "year_bin"] = bins.size - 1

    norm = matplotlib.colors.Normalize(vmin=1, vmax=bins.size - 1)
    image_year_cmap = matplotlib.cm.ScalarMappable(
        norm=norm,
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "years",
            [
                (norm(a), b)
                for a, b in [
                    (1, "green"),
                    (2, "lightseagreen"),
                    (3, "orange"),
                    (4, "gold"),
                ]
            ],
        ),
    )

    # with rio.open(terradem.files.TEMP_FILES["base_dem_slope"], overview_level=4) as raster:
    #    outline = gpd.GeoSeries(
    #        [
    #            shapely.geometry.shape(l[0])
    #            for l in rio.features.shapes((raster.read(1) == -9999).astype("uint8"), transform=raster.transform)
    #            if l[1] == 0
    #        ],
    #        crs=raster.crs,
    #    )

    with rio.open("temp/base_dem/alos_slope.tif") as raster:
        slope = raster.read(1, masked=True)
        bounds = raster.bounds

    lk50_outlines = gpd.read_file(terradem.files.INPUT_FILE_PATHS["lk50_outlines"])
    outline = gpd.read_file("ch_bnd/ch.shp").to_crs(lk50_outlines.crs)

    colors = {
        "precipitation": "royalblue",
        "s_temperature": "red",
        "temperature": "black",
        "w_temperature": "royalblue",
    }

    plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=3)
    plt.imshow(
        slope,
        cmap="Greys",
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        interpolation="bilinear",
        zorder=1,
    )
    # total lk50_outlines.plot(ax=plt.gca(), edgecolor="black", lw=0.1)
    plot_lk50_glaciers(lk50_outlines=lk50_outlines)
    outline.plot(ax=plt.gca(), facecolor="none", edgecolor="black")

    plt.scatter(
        image_meta["easting"],
        image_meta["northing"],
        s=2,
        alpha=0.2,
        edgecolors="none",
        cmap=image_year_cmap.get_cmap(),
        norm=image_year_cmap.norm,
        c=image_meta["year_bin"],
    )

    for i, box_bounds in enumerate(
        [IMAGE_EXAMPLE_BOUNDS, INTERPOLATION_BEFORE_AFTER_BOUNDS, LK50_EXAMPLE_BOUNDS, OUTLINE_ERROR_BOUNDS]
    ):
        plt.plot(
            *shapely.geometry.box(*box_bounds).exterior.xy,
            zorder=3,
            color="red",
            lw=1,
            path_effects=[matplotlib.patheffects.Stroke(foreground="k", linewidth=1), matplotlib.patheffects.Normal()],
        )
        plt.annotate(
            "abcdef"[i],
            (box_bounds.left, box_bounds.top),
            va="bottom",
            ha="center",
            color="white",
            path_effects=[matplotlib.patheffects.Stroke(foreground="k", linewidth=2), matplotlib.patheffects.Normal()],
        )

    weather_stations = terradem.climate.read_all_data().groupby("station").mean()
    handle = plt.scatter(
        weather_stations["easting"],
        weather_stations["northing"],
        marker="^",
        facecolor="none",
        edgecolors="k",
        s=40,
        label="Weather station",
    )

    plt.xlim(lk50_outlines.total_bounds[[0, 2]])
    plt.ylim(lk50_outlines.total_bounds[[1, 3]] * np.array([1, 1.2]) - np.diff(lk50_outlines.total_bounds[[1, 3]]) / 5)
    plt.xticks(plt.gca().get_xticks()[[1, -2]], (plt.gca().get_xticks()[[1, -2]] + 2e6).astype(int))
    plt.yticks(
        plt.gca().get_yticks()[[1, -3]],
        (plt.gca().get_yticks()[[1, -3]] + 1e6).astype(int),
        rotation=90,
        ha="right",
        va="center",
    )
    plt.text(0.5, -0.08, "Easting (m)", ha="center", va="bottom", transform=plt.gca().transAxes)
    plt.text(-0.05, 0.5, "Northing (m)", ha="right", va="center", transform=plt.gca().transAxes, rotation=90)
    plt.text(
        0.01,
        0.99,
        "A)",
        transform=plt.gca().transAxes,
        fontsize=12,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.2),
    )
    handles = [handle]
    for i in range(1, bins.size):
        handles.append(plt.scatter(0, 0, s=20, color=image_year_cmap.to_rgba(i), label=f"{bins[i - 1] + 1}–{bins[i]}"))
    plt.legend(handles=handles, loc="lower right")

    xlim = (1900, 2021)
    xticks = np.arange(1900, 2040, 40)
    plt.subplot2grid((2, 4), (0, 3))
    for key in ["temperature", "s_temperature"]:
        plt.plot(climate[key].rolling(10, min_periods=1).mean(), color=colors[key])
        plt.scatter(climate.index, climate[key], c=colors[key], s=0.5)
    plt.ylabel(r"$\Delta$T ($\degree$C)", rotation=270, labelpad=10)
    plt.xlim(xlim)
    plt.xticks(xticks, [""] * xticks.shape[0])
    plt.text(0.02, 0.97, "B)", transform=plt.gca().transAxes, fontsize=12, ha="left", va="top")
    plt.grid(zorder=0, alpha=0.4)
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")

    plt.subplot2grid((2, 4), (1, 3))
    plt.plot(climate["precipitation"].rolling(10, min_periods=1).mean(), color=colors["precipitation"])
    plt.scatter(climate.index, climate["precipitation"], c=colors["precipitation"], s=0.5)
    plt.xlim(xlim)
    plt.xticks(xticks)
    plt.text(0.02, 0.97, "C)", transform=plt.gca().transAxes, fontsize=12, ha="left", va="top")
    plt.grid(zorder=0, alpha=0.4)
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.yticks(rotation=270, va="center")
    plt.ylabel(r"$\Delta$P (mm)", rotation=270, labelpad=20)

    plt.subplots_adjust(top=0.991, bottom=0.082, left=0.05, right=0.938, hspace=0.034, wspace=0.03)

    plt.savefig("temp/figures/overview.jpg", dpi=600)

    if show:
        plt.show()


def error_ensemble(show: bool = True):
    fig = plt.figure(figsize=(8, 5), dpi=200)
    glacier_wise_dh = pd.read_csv(terradem.files.TEMP_FILES["glacier_wise_dh"])

    grid = (19, 19)
    names = {
        "topo_err": "Stable-ground error",
        "time_err": "Temporal error",
        "area_err": "Area error",
        "interp_err": "Interpolation error",
    }
    bounds = rio.coords.BoundingBox(left=639450, bottom=139000, right=644500, top=144000 + 800)

    station_name: str = "station_1536"
    image_meta = pd.read_csv(terradem.files.INPUT_FILE_PATHS["swisstopo_metadata"])
    image_meta = image_meta[image_meta["station_name"].str.contains(station_name)]
    lk50_outlines = gpd.read_file(terradem.files.INPUT_FILE_PATHS["lk50_outlines"])

    viewshed = terradem.orthorectification.get_viewshed(station_name=station_name).values.ravel()

    ddem_coreg_ds = rio.open(Path(terradem.files.TEMP_SUBDIRS["ddems_coreg"]).joinpath(f"{station_name}_ddem.tif"))
    ddem_non_coreg_ds = rio.open(
        Path(terradem.files.TEMP_SUBDIRS["ddems_non_coreg"]).joinpath(f"{station_name}_ddem.tif")
    )

    window = rio.windows.from_bounds(*bounds, transform=ddem_coreg_ds.transform)

    for i, ddem_ds in enumerate([ddem_non_coreg_ds, ddem_coreg_ds], start=1):
        axis = plt.subplot2grid(grid, (0, (i - 1) * (grid[0] // 4 + 0)), rowspan=grid[0] // 2, colspan=grid[0] // 4)
        # axis = plt.subplot(1, 2, i)
        ddem = ddem_ds.read(1, window=window, boundless=True, masked=True).filled(np.nan)

        lk50_outlines.plot(color="#ADB7D2", edgecolor="k", alpha=0.5, linewidth=0.5, ax=axis)

        plt.imshow(
            ddem,
            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
            cmap=DH_CMAP.get_cmap(),
            norm=DH_CMAP.norm,
            zorder=1,
        )
        for polygon in viewshed:
            plt.plot(*polygon.exterior.xy, linestyle="--", color="black", zorder=2, linewidth=1)
        plt.quiver(
            image_meta["easting"],
            image_meta["northing"],
            1,
            1,
            angles=90 - image_meta["yaw"],
            headwidth=3,
            headlength=2,
            width=4e-3,
        )
        plt.ylim(bounds.bottom, bounds.top)
        plt.xlim(bounds.left, bounds.right)

        axis.xaxis.tick_top()
        axis.xaxis.set_label_position("top")

        xticks = axis.get_xticks()[[1, -2]]
        yticks = axis.get_yticks()[[1, -2]]
        if i == 1:
            plt.xticks(xticks, labels=(xticks + 2e6).astype(int))
            plt.yticks(yticks, labels=(yticks + 1e6).astype(int), rotation=90, va="center")
            plt.ylabel("Northing (m)")
            plt.xlabel("Easting (m)")
        else:
            inset = colorbar(axis=axis, loc=(1.01, 0.5), vmin=-1, vmax=1, height=0.5, tick_right=True, width=0.07, rotate_ticks=False, labelpad=5)
            inset.set_ylabel(DH_MWE_LABEL, fontsize=10)
            plt.xticks(xticks, labels=[""] * len(xticks))
            plt.yticks(yticks, labels=[""] * len(yticks))
        axis.text(
            0.01,
            0.988,
            "A)" if i == 1 else "B)",
            ha="left",
            va="top",
            transform=axis.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.8),
        )

    # terradem.error.glacier_outline_error(plot=14)
    # plt.xlim(659300, 661000)
    # plt.ylim(160200, 162000)
    # plt.xlabel("Easting (m)")
    # plt.ylabel("Northing (m)")
    # plt.yticks(
    #    plt.gca().get_yticks()[[1, -1]],
    #    labels=(plt.gca().get_yticks()[[1, -1]] + 1e6).astype(int),
    #    rotation=90,
    #    ha="right",
    #    va="center",
    # )
    # plt.xticks(plt.gca().get_xticks()[[1, -1]], labels=(plt.gca().get_xticks()[[1, -1]] + 2e6).astype(int))
    # plt.gca().xaxis.tick_top()
    # plt.gca().xaxis.set_label_position("top")
    # plt.text(0.02, 0.97, "A)", transform=plt.gca().transAxes, fontsize=12, ha="left", va="top")

    plt.subplot2grid(grid, (0, 1 + grid[1] // 2), rowspan=grid[0] // 2, colspan=grid[0] // 2)
    plt.hist(glacier_wise_dh[list(names.keys())], bins=np.linspace(0, 0.4, 20), label=[names[col] for col in names])
    plt.ylabel("Count")
    plt.xlabel(r"dHdt$^{-1}$ (ma$^{-1})$")
    plt.yscale("log")
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position("top")
    plt.text(0.02, 0.97, "C)", transform=plt.gca().transAxes, fontsize=12, ha="left", va="top")
    plt.ylim(1, plt.gca().get_ylim()[1] * 1.1)
    plt.legend()

    dh_vgm = pd.read_csv(terradem.files.TEMP_FILES["topographic_error_variogram"]).drop(columns=["bins.1"])
    dh_vgm = dh_vgm[dh_vgm["bins"] < 2e5]

    interp_vgm = pd.read_csv("temp/interpolation_vgm.csv")
    interp_vgm = interp_vgm[(interp_vgm["bins"] < 2e6) & (interp_vgm["exp"].notna())]

    for i, vgm in enumerate([interp_vgm, dh_vgm]):
        vgm["bins_interval"] = pd.IntervalIndex.from_breaks(np.r_[[0], vgm["bins"]])
        vgm_model, _ = xdem.spatialstats.fit_sum_model_variogram(["Sph"] * 2, vgm)

        limits = [0, 100, 1e4, vgm["bins"].max() * 1.1]
        xticks = [0, 5e1, 1e2, 5e3, 1e4, 1e5 if i == 1 else 5e5]
        for j in range(3):
            for k in range(2):
                colspan = (grid[1] - 1) // (3 * 2)
                plt.subplot2grid(
                    grid,
                    (1 + grid[0] // 2 + 2 * k, (i * (1 + grid[1] // 2)) + j * colspan),
                    rowspan=2 if k == 0 else grid[0] // 2 - 2,
                    colspan=colspan,
                )
                if k == 0:
                    plt.bar(
                        vgm["bins_interval"].apply(lambda i: i.mid),
                        height=vgm["count"],
                        width=vgm["bins_interval"].apply(lambda i: i.length),
                        edgecolor="lightgray",
                        zorder=1,
                        color="darkslategrey",
                        linewidth=0.2,
                    )
                    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                elif k == 1:
                    plt.errorbar(
                        vgm["bins_interval"].apply(lambda i: i.mid),
                        vgm["exp"],
                        yerr=vgm["err_exp"],
                        linestyle="",
                        marker="x",
                    )
                    plt.plot(np.arange(limits[-1]), vgm_model(np.arange(limits[-1])))

                plt.ylim(0, vgm["count"].max() if k == 0 else vgm[["exp", "err_exp"]].sum(axis=1).max())
                plt.xticks(xticks, None if k == 1 else [""] * len(xticks))
                plt.xlim(limits[j], limits[j + 1])
                yticks = plt.gca().get_yticks()[:-1]
                plt.grid(zorder=0)

                if (j == 0 and i == 0) or (j == 2 and i == 1):
                    plt.ylabel(r"Variance ($m a^{-1}$)" if k == 1 else "Count")
                    plt.yticks(yticks)
                else:
                    plt.yticks(yticks, [""] * len(yticks))

                if j == 1 and k == 1:
                    plt.xlabel("Lag (m)")
                if j == 0 and k == 0:
                    plt.text(
                        0.05,
                        0.91,
                        "D)" if i == 0 else "E)",
                        transform=plt.gca().transAxes,
                        fontsize=12,
                        ha="left",
                        va="top",
                    )

                if i == 1:
                    plt.gca().yaxis.tick_right()
                    plt.gca().yaxis.set_label_position("right")
    plt.subplots_adjust(left=0.065, bottom=0.09, right=0.92, top=0.905, wspace=0, hspace=0.065)
    plt.savefig("temp/figures/error_approach_ensemble.jpg", dpi=600)
    if show:
        plt.show()


def historic_images(show: bool = True):
    sgi_2016 = gpd.read_file(terradem.files.INPUT_FILE_PATHS["sgi_2016"])
    lk50_outlines = gpd.read_file(terradem.files.INPUT_FILE_PATHS["lk50_outlines"]).to_crs(sgi_2016.crs)
    image_meta = pd.read_csv(terradem.files.INPUT_FILE_PATHS["swisstopo_metadata"])
    image_meta = gpd.GeoDataFrame(
        image_meta, geometry=gpd.points_from_xy(image_meta["easting"], image_meta["northing"], crs="epsg:21781")
    )

    image_dir = Path("../SwissTerra/input/images/")

    rhone = image_meta[(image_meta["Base number"] == "1666") & (image_meta["Position"] == "R")].copy()

    # rhone["x"] = np.deg2rad(image_meta["yaw"] + 30)
    # rhone["y"] = np.deg2rad(90 - image_meta["pitch"])
    rhone["x"] = np.where(rhone["yaw"] > 180, rhone["yaw"] - 360, rhone["yaw"])
    rhone["y"] = rhone["pitch"] - 90

    # The height (in degrees) is derived from an approximate Wild camera in a web interface
    image_height = 44
    image_width = (18 / 13) * image_height

    fig: plt.Figure = plt.figure(figsize=(8.3, 4.8), dpi=150)
    grid = (2, 3)

    crs = ccrs.LambertConformal()
    ax0: plt.Axes = plt.subplot2grid(grid, (0, 1), colspan=2, projection=crs, fig=fig)

    def add_letter(axis: plt.Axes, letter: str, loc: tuple[float, float]):
        axis.text(
            *loc,
            letter,
            ha="left",
            va="top",
            transform=axis.transAxes,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.8),
        )

    # extents: dict[int, tuple[float, float, float, float]] = {}
    for _, image in rhone.iloc[::-1].iterrows():
        img = plt.imread(image_dir.joinpath(image["Image file"]))
        if len(img.shape) > 2:
            img = img[:, :, 0]

        vmin = np.percentile(img, 1)
        vmax = np.percentile(img, 99)

        img = np.clip((img - vmin) / (vmax - vmin), 0, 255)

        ax0.imshow(
            img,
            cmap="Greys_r",
            extent=[
                image["x"] - image_width / 2,
                image["x"] + image_width / 2,
                image["y"] - image_height / 2,
                image["y"] + image_height / 2,
            ],
            transform=crs,
        )

    for _, image in rhone.iloc[::-1].iterrows():
        ax0.add_patch(
            plt.Rectangle(
                (image["x"] - image_width / 2, image["y"] - image_height / 2),
                image_width,
                image_height,
                alpha=0.08,
                transform=crs,
                edgecolor="red",
                facecolor="red",
                linestyle=":",
            )
        )

    rhone_outline = [
        (-19.1, -1.81),
        (-18.85, -3.85),
        (-19.97, -5),
        (-21, -7),
        (-21, -10),
        (-22.9, -12),
        (-23, -16),
        (-21, -16.5),
        (-22.34, -19),
        (-21.42, -21.64),
        (-21.69, -25),
        (-22.49, -26.94),
        (-22.61, -27.66),
        (-22.69, -28.68),
        (-22.05, -29.03),
        (-19.97, -28.78),
        (-18.13, -28.06),
        (-16.84, -27.34),
        (-16.29, -27.84),
        (-16.24, -28.93),
        (-15, -29),
        (-13, -27.73),
        (-11, -26.87),
        (-9.19, -26.21),
        (-7.53, -24.84),
        (-6.617, -23.17),
        (-6.24, -20.93),
        (-7, -19.58),
        (-7.38, -16.99),
        (-8.3, -14),
        (-6, -11.73),
        (-4, -11.8),
        (-3.87, -10),
        (-2.33, -9.1),
        (-1.12, -6.27),
        (1.34, -5),
        (2.17, -2.25),
        (4, 1),
        (7.1, 3.50),
    ]

    ax0.plot(*np.array(rhone_outline).T, color="royalblue", linestyle="--")

    # ax0.set_extent([-100, 50, -40, 25])
    ax0.set_xlim(-120, 50)
    ax0.set_ylim(-40, 25)
    add_letter(ax0, "B)", (0.09, 0.95))
    ax0.set_axis_off()

    colors = {"LK50": "royalblue", "SGI2016": "lightgray"}
    ax1: plt.Axes = plt.subplot2grid(grid, (0, 0), colspan=1, rowspan=2, fig=fig)
    lk50_outlines.plot(ax=ax1, color=colors["LK50"])
    sgi_2016.plot(ax=ax1, color=colors["SGI2016"])

    handles = [matplotlib.patches.Patch(facecolor=c) for c in colors.values()]
    labels = list(colors.keys())

    xlim = IMAGE_EXAMPLE_BOUNDS.left + 2e6, IMAGE_EXAMPLE_BOUNDS.right + 2e6
    ylim = IMAGE_EXAMPLE_BOUNDS.bottom + 1e6, IMAGE_EXAMPLE_BOUNDS.top + 1e6
    image_meta = image_meta.to_crs(lk50_outlines.crs)
    image_meta = image_meta[
        (image_meta.geometry.x.values > xlim[0])
        & (image_meta.geometry.x <= xlim[1])
        & (image_meta.geometry.y > ylim[0])
        & (image_meta.geometry.y <= ylim[1])
    ]

    for _, camera in image_meta.reset_index().iterrows():
        marker = matplotlib.markers.MarkerStyle(r"$\frac{|}{\cdot}$")
        marker._transform = marker.get_transform().rotate_deg(camera["yaw"])
        p = ax1.scatter(
            camera.geometry.x, camera.geometry.y, marker=marker, color="black", facecolor="none", linewidths=0.3, s=60
        )

    handles.append(p)
    labels.append("Photograph")

    # ax1.legend(handles, labels)
    ax1.set_ylim(ylim)
    ax1.set_xlim(xlim)
    add_letter(ax1, "A)", (0.05, 0.975))
    ax1.ticklabel_format(style="plain")
    xticks = ax1.get_xticks()
    yticks = ax1.get_yticks()
    ax1.set_xticks(xticks[[1, -1]])
    plt.yticks(yticks[[1, -2]], rotation=90, va="center")
    ax1.set_ylabel("Northing (m)")
    ax1.set_xlabel("Easting (m)")

    data = {
        "Wild": {
            "filename": "000-164-603.tif",
            "loc": (1, 1),
            "boxes": [[3.22e3, 0e3], [3.20e3, 4.13e3], [6.48e3, 2.12e3], [0e3, 2.10e3]],
            "letter": "C)",
        },
        "Zeiss": {
            "filename": "000-360-488.tif",
            "loc": (1, 2),
            "boxes": [[3.95e3, 0.2e3], [3.95e3, 5.5e3], [7.94e3, 2.85e3], [0e3, 1.38e3]],
            "letter": "D)",
        },
    }

    for key in data:
        axis = plt.subplot2grid(grid, data[key]["loc"], fig=fig)
        img = plt.imread(image_dir.joinpath(data[key]["filename"]))

        vmin = np.percentile(img, 1)
        vmax = np.percentile(img, 99)
        img = (img - vmin) / (vmax - vmin)

        for box in data[key]["boxes"]:
            axis.add_patch(plt.Rectangle(box[:2], 500, 500, facecolor="none", linestyle="--", edgecolor="red"))

        axis.set_title(f'“{key}"', fontsize=10)
        axis.imshow(img, cmap="Greys_r")
        axis.set_axis_off()
        add_letter(axis, data[key]["letter"], (0.05, 0.95))

    # plt.xlim(-1, 4)
    # plt.ylim(-1, 2)
    # plt.xlim(-1, 2)
    # plt.ylim(-1, 2)
    plt.subplots_adjust(top=0.995, bottom=0.087, left=0.04, right=0.995, hspace=0.0, wspace=0.05)
    plt.savefig("temp/figures/image_examples.jpg", dpi=600)
    if show:
        plt.show()


def dh_histogram():
    data = pd.read_csv(terradem.files.TEMP_FILES["glacier_wise_dh"])

    data = data[(data["dh_m_we"] - data["dh_m_we"].median()).abs() < (xdem.spatialstats.nmad(data["dh_m_we"]) * 4)]

    data["area_km2"] = data[["start_area", "end_area"]].mean(axis=1) * 1e-6

    plt.hist2d(data["med_elev"], data["dh_m_we"], bins=50, cmin=1, cmap="magma_r")

    xticks = plt.gca().get_xticks()

    # plt.xticks(xticks, labels=[r"$10^{" + str(int(n)) + "}$" for n in xticks])
    plt.ylabel("dHdt$^{-1}$ (ma$^{-1}$ w.e.)")
    # plt.xlabel("Area (km²)")
    plt.xlabel("Elevation (m a.s.l.")

    plt.show()


def west_east_transect(show: bool = True):
    warnings.simplefilter("error")
    data = pd.read_csv(terradem.files.TEMP_FILES["glacier_wise_dh"])

    data = data.sort_values("start_area", ascending=False).sort_values("start_area", ascending=False)

    data["subregion"] = data["sgi_id"].str.slice(stop=1)

    subregion = data.groupby("subregion")

    starts = subregion["easting"].min()
    ends = subregion["easting"].max()

    ranges = ends - starts

    distance = 13000
    breaks = (ranges[starts.sort_values().index].shift(1) + distance).fillna(0).cumsum()

    data["easting"] -= (starts - breaks)[data["subregion"]].values

    # data = data[data["start_area"] > 2e5].copy().sort_values("easting")
    data.sort_values("easting", inplace=True)

    distances = starts.sort_values().copy()
    distances[:] = distance
    distances.iloc[0] = 0
    distances = distances.cumsum()

    plt.figure(figsize=(8.3, 4))
    data["width"] = data["start_area"] / 1e4
    data["left"] = distances[data["subregion"]].values + data["width"].cumsum()

    plt.bar(
        data["left"] - data["width"] / 2,
        height=data["max_elev"] - data["min_elev"],
        width=data["width"],
        bottom=data["min_elev"],
        color=MB_CMAP.to_rgba(data["dh_m_we"]),
        zorder=2,
    )
    # plt.bar(data["easting"], height=data["max_elev"] - data["min_elev"], width=data["start_area"] / 1e4 + 400, bottom=data["min_elev"], color=DH_CMAP.to_rgba(data["dh_m_we"]))#, edgecolor="k", linewidth=0.3)

    for i, subset in data.groupby("subregion"):
        # bins = np.digitize(subset["left"], np.linspace(subset["left"].min(), subset["left"].max(), num=25))

        # easting_normalized = subset.groupby(bins).mean().rolling(2, min_periods=1).mean()
        easting_normalized = subset.rolling(120, min_periods=1).mean()

        plt.plot(easting_normalized["left"], easting_normalized["med_elev"], c="k")

    inset = colorbar(
        cmap=MB_CMAP,
        loc=(0.98, 0.6),
        label=DH_MWE_LABEL,
        height=0.4,
        vmax=0.5,
        vmin=-1.3,
        tick_right=True,
        labelpad=7,
        rotate_ticks=True,
    )
    #plt.setp(inset.yaxis.get_majorticklabels(), rotation=270, va="center")

    subregion = data.groupby("subregion")
    ylim = plt.gca().get_ylim()
    plt.vlines(data.groupby("subregion")["left"].max() + distance / 2, ylim[0], ylim[1], linestyles="--", color="k")

    plt.xlim(0, data["left"].max() + distance / 2)

    plt.hlines([2000, 3000, 4000], 0, data["left"].max() + distance / 2, linestyles="--", color="gray", zorder=1)

    subregion_names = {"A": "Rhine", "B": "Rhone", "C": "Po", "E": "Dan."}

    for i in ranges.index:
        subregion_data = data[data["subregion"] == i]

        mean_dh = str(round(subregion_data["dh_m_we"].mean(), 2))
        mean_dh += "0" * (5 - len(mean_dh))

        newline = "\n" if i not in ["A", "B"] else " "

        plt.annotate(
            (
                f"{subregion_names[i]}\n{round(subregion_data['start_area'].sum() * 1e-6)} km²\n{mean_dh}{newline}{DH_MWE_UNIT}"
            ),
            (
                np.mean(
                    [data.loc[data["subregion"] == i, "left"].min(), data.loc[data["subregion"] == i, "left"].max()],
                    axis=0,
                ),
                ylim[0] + 100,
            ),
            ha="center",
            path_effects=[matplotlib.patheffects.Stroke(linewidth=1, foreground="w"), matplotlib.patheffects.Normal()],
        )
    plt.ylim(ylim)

    # xticks = plt.gca().get_xticks()
    # plt.xticks(xticks, [""] * len(xticks))
    plt.xticks([])
    plt.ylabel("Elevation (m a.s.l.)")
    plt.subplots_adjust(left=0.08, bottom=0.04, right=0.914, top=0.963)

    plt.savefig("temp/figures/west_east_transect.jpg", dpi=600)
    if show:
        plt.show()


def regional_dh(show: bool = True):
    data = pd.read_csv(terradem.files.TEMP_FILES["glacier_wise_dh"])

    # Take the first two letters of the sgi-id and hash them (creating a unique int; good for grouping)
    data["subregion"] = data["sgi_id"].str.slice(stop=2).apply(hash)

    data = data.select_dtypes(np.number)

    gridsize_x = 60000
    gridsize_y = gridsize_x

    xmin = (data["easting"].min() - data["easting"].min() % gridsize_x) - gridsize_x * 2
    xmax = (data["easting"].max() - data["easting"].max() % gridsize_x) + gridsize_x * 2
    ymin = (data["northing"].min() - data["northing"].min() % gridsize_y) - gridsize_y * 2
    ymax = (data["northing"].max() - data["northing"].max() % gridsize_y) + gridsize_y * 2

    grid_x = np.arange(xmin, xmax, gridsize_x)
    grid_y = np.arange(ymin, ymax, gridsize_y)[::-1]

    data["col"] = np.digitize(data["easting"], grid_x) - 1
    data["row"] = np.digitize(data["northing"], grid_y) - 1
    data["i"] = grid_x.size * data["row"] + data["col"]

    data["grid_east"] = grid_x[data["col"].values] + gridsize_x / 2
    data["grid_north"] = grid_y[data["row"].values] + gridsize_y / 2

    data["weight"] = data["start_area"] / data["start_area"].max()

    data_times_weight = data * data["weight"].values.reshape((-1, 1))

    data_weight_where_notnull = pd.notnull(data) * data["weight"].values.reshape((-1, 1))

    group_column = "i"

    data_times_weight[group_column] = data[group_column]
    data_weight_where_notnull[group_column] = data[group_column]

    grouped = data_times_weight.groupby(group_column).sum() / data_weight_where_notnull.groupby(group_column).sum()
    grouped["area_km2"] = data.groupby(group_column).sum()["start_area"] * 1e-6
    # grouped["area_km2"] = grouped["start_area"] * grouped["count"] * 1e-6

    grouped.sort_values("easting", inplace=True)

    # grouped = grouped[grouped["dh_m_we"] > -2]

    # grouped["size"] = np.clip(grouped["dh_m_we"] * -60 + 1, 0, 200)
    size_func = lambda area_km2: np.sqrt(area_km2) * 30
    grouped["size"] = size_func(grouped["area_km2"])

    # cmap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-1, vmax=1), cmap="RdBu")
    grouped["color"] = grouped["dh_m_we"].apply(MB_CMAP.to_rgba)

    plt.figure(figsize=(8, 5), dpi=200)
    plot_base_dem_slope()
    plot_lk50_glaciers()
    plt.scatter(
        grouped["easting"],
        grouped["northing"],
        s=grouped["size"],
        c=grouped["color"],
        edgecolors="k",
        alpha=0.9,
        cmap=MB_CMAP,
    )
    for _, row in grouped.iterrows():
        text = str(round(row["dh_m_we"], 2))
        text += "0" * (5 - len(text))
        plt.annotate(
            text,
            xy=(row["easting"], row["northing"]),
            ha="center",
            va="center",
            fontsize=8,
            path_effects=[matplotlib.patheffects.Stroke(foreground="w", linewidth=2), matplotlib.patheffects.Normal()],
        )
    # print(DH_CMAP.set_clim(-1, 0.2))
    # cbar = plt.colorbar(DH_CMAP)
    colorbar(MB_CMAP, height=0.3, loc=(0.03, 0.65), vmax=0.2, vmin=-0.75, tick_right=True, label=DH_MWE_LABEL, rotate_ticks=False, labelpad=5)
    #inset.yaxis.set_label_position("right")
    #inset.set_ylabel("dHdt$^{-1}$ (" + DH_MWE_UNIT + ")", rotation=270, labelpad=14)
    #inset.yaxis.tick_right()

    legend_items = (
        # Add empty patch as a header
        {"Size (km²)": matplotlib.patches.Patch(facecolor="none", edgecolor="none")}  # Add an empty patch as a header
        | {
            f"{area_km2} km²": plt.scatter(0, 0, s=size_func(area_km2), c="white", edgecolor="k")
            for area_km2 in [10, 50, 400]
        }
    )
    legend_items_list = list(legend_items.items())
    #legend_items_list.insert(-1, ("", matplotlib.patches.Patch(facecolor="none", edgecolor="none")))
    legend_items = dict(legend_items_list)

    plt.ylim(61995, 302000)
    plt.xlim(480000, 838755)

    xticks = plt.gca().get_xticks()[[1, -2]]
    plt.xticks(xticks, (xticks + 2e6).astype(int))
    yticks = plt.gca().get_yticks()[[1, -3]]
    plt.yticks(yticks, (yticks + 1e6).astype(int), rotation=270, va="center")

    plt.ylabel("Northing (m)")
    plt.xlabel("Easting (m)")

    plt.legend(labels=legend_items.keys(), handles=legend_items.values(), borderpad=0.9, labelspacing=1.3)
    plt.tight_layout()

    plt.savefig("temp/figures/dh_bubbles.jpg", dpi=600)

    if show:
        plt.show()
    # print(data.select_dtypes(np.number).groupby("i").aggregate(lambda df: np.average(df, weights=data.loc[df.index, "start_area"], axis=0)))


def interpolation_before_and_after(show: bool = True):

    bounds = INTERPOLATION_BEFORE_AFTER_BOUNDS

    ddem_paths = {
        "gappy": terradem.files.TEMP_FILES["ddem_coreg_tcorr"],
        "interp": terradem.files.TEMP_FILES["ddem_coreg_tcorr_national-interp-extrap"],
    }

    ddems: dict[str, np.ndarray] = {}

    for key in ddem_paths:
        with rio.open(ddem_paths[key]) as raster:
            window = rio.windows.from_bounds(*bounds, raster.transform)
            ddems[key] = downsample_nans(raster.read(1, window=window, masked=True).filled(np.nan), 7)

    imshow_params = {
        "cmap": DH_CMAP.get_cmap(),
        "norm": DH_CMAP.norm,
        "interpolation": "bilinear",
        "extent": [bounds.left, bounds.right, bounds.bottom, bounds.top],
    }

    plt.figure(figsize=(8.3, 5), dpi=200)
    for i, key in enumerate(ddems, start=1):
        plt.subplot(1, 2, i)
        plt.imshow(np.ma.masked_array(ddems[key], mask=np.isnan(ddems[key])), **imshow_params)

        yticks = plt.gca().get_yticks()[[2, -2]]
        plt.yticks(yticks, (yticks + 1e6).astype(int) if i == 1 else [""] * len(yticks), rotation=270, va="center")
        xticks = plt.gca().get_xticks()[[2, -3]]
        plt.xticks(xticks, (xticks + 2e6).astype(int) if i == 1 else [""] * len(xticks))

        plt.text(0.03, 0.98, "A)" if i == 1 else "B)", ha="left", va="top", transform=plt.gca().transAxes, fontsize=12)
        if i == 1:
            inset = colorbar(label=DH_MWE_LABEL, tick_right=True, loc=(0.02, 0.7), vmax=1, rotate_ticks=False, labelpad=5)
            inset.set_yticks([-4, -1, 0, 1])
            inset.set_yticklabels(["$<-4$", "-1", "0", ">1"])
            plt.ylabel("Northing (m)")
            plt.xlabel("Easting (m)")

    plt.tight_layout()

    plt.subplots_adjust(top=0.99, bottom=0.09, left=0.095, right=0.982, hspace=0.2, wspace=0.0)
    plt.savefig("temp/figures/interpolation_before_and_after.jpg", dpi=900)

    if show:
        plt.show()


def mb_correlations(show: bool = True):
    glacier_wise_dh = pd.read_csv(terradem.files.TEMP_FILES["glacier_wise_dh"])

    debris_cover = gpd.read_file("data/external/shapefiles/SGI_2016_debriscover.shp").dissolve(by="sgi-id")
    sgi_2016 = gpd.read_file(terradem.files.INPUT_FILE_PATHS["sgi_2016"]).set_index("sgi-id")

    debris_cover = debris_cover[debris_cover.index.isin(sgi_2016.index)]

    sgi_2016["debris_m2"] = np.nan
    sgi_2016.loc[debris_cover.index, "debris_m2"] = debris_cover.area.values
    sgi_2016["debris_frac"] = np.clip(100 * sgi_2016["debris_m2"] / sgi_2016.area, 0, 100)

    glacier_wise_dh["sgi_2016"] = glacier_wise_dh["sgi_2016_ids"].str.split(",", expand=True).iloc[:, 0]
    glacier_wise_dh["n_samples"] = (1e5 * glacier_wise_dh["start_area"] / glacier_wise_dh["start_area"].max()).astype(
        int
    )
    glacier_wise_dh = glacier_wise_dh.merge(sgi_2016["debris_frac"], left_on="sgi_2016", right_index=True)

    glacier_wise_dh = glacier_wise_dh.select_dtypes(np.number)
    #glacier_wise_dh = glacier_wise_dh[
    #    glacier_wise_dh["dh_m_we"].abs() < (xdem.spatialstats.nmad(glacier_wise_dh["dh_m_we"]) * 3)
    #]

    glacier_wise_dh["area_log10"] = np.log10(glacier_wise_dh["start_area"] / 1e6)
    glacier_wise_dh["debris_log10"] = np.log10(glacier_wise_dh["debris_frac"] + 1e-4)

    weighted = pd.DataFrame(
        np.repeat(glacier_wise_dh.values, glacier_wise_dh["n_samples"].values, axis=0),
        columns=glacier_wise_dh.columns,
        dtype="float64",
    )

    labels = {
        "med_elev": "Median elevation (m a.s.l.)",
        "area_log10": "Area (km²)",
        "modern_slope_lower_10percent": "Lower 10% slope (°)",
        "debris_log10": "Debris cover (%)",
    }

    plt.figure(figsize=(8.3, 5))

    #nmad = xdem.spatialstats.nmad(glacier_wise_dh["dh_m_we"])
    percentile = 99
    lower = np.nanpercentile(weighted["dh_m_we"], (100 - percentile) / 2)
    upper = np.nanpercentile(weighted["dh_m_we"], percentile + (100 - percentile) / 2)

    for i, col in enumerate(
        ["med_elev", "area_log10", "modern_slope_lower_10percent", "debris_log10"],
        start=1,
    ):
        plt.subplot(2, 2, i)

        filtered = weighted.iloc[(weighted[col].values > np.nanpercentile(weighted[col].values, 2.5)) & (weighted[col].values <= np.nanpercentile(weighted[col].values, 97.5))]

        bins = np.percentile(filtered[col].values, np.linspace(0, 100, 10))
        bins[-1] += 1e-3  # Make sure the last bin includes the maximum largest numbers
        indices = np.digitize(filtered[col].values, bins=bins)

        bin_values = [filtered["dh_m_we"].iloc[indices == i] for i in np.unique(indices)]
        midpoints = (bins[1:] - np.diff(bins) / 2)[np.unique(indices) - 1]

        boxplot = plt.boxplot(
            bin_values,
            positions=midpoints,
            widths=np.diff(bins),
            manage_ticks=False,
            showfliers=False,
            notch=False,
            patch_artist=True,
            boxprops=dict(
                facecolor="gray",
            ),
            medianprops=dict(
                color="black",
                linestyle=":",
            ),
            zorder=3,
        )

        medians = np.array([np.median(vals) for vals in bin_values])
        for j, box in enumerate(boxplot["boxes"]):
            plt.setp(box, facecolor=MB_CMAP.to_rgba(medians[j]))
        warnings.simplefilter("error")
        if "log10" in col:
            min_log = np.floor(bins[0])
            max_log = np.ceil(bins[-1])

            xticks_log = np.arange(min_log, max_log + 1)
            minor_xticks = np.log10(np.ravel([[a * 10 ** b for a in np.arange(1, 11)] for b in xticks_log]))

            plt.xticks(xticks_log, [int(a) if int(a) == a else a for a in  10** xticks_log])
            plt.gca().set_xticks(minor_xticks, minor=True)
            plt.gca().tick_params(axis='x', which='minor', **(dict(top=True) if i <= 2 else dict(bottom=True)))
        elif col == "easting":
            xticks = np.round(np.linspace(bins[0], bins[-1], 3), -5)
            plt.xticks(xticks, labels=(xticks + 2e6).astype(int))
        elif col == "northing":
            xticks = np.round(np.linspace(bins[0], bins[-1], 3), -4)
            plt.xticks(xticks, labels=(xticks + 1e6).astype(int))

        plt.xlabel(labels[col])

        if i <= 2:
            plt.gca().xaxis.tick_top()
            plt.gca().xaxis.set_label_position("top")

        ylim = (-1.3, 0.2)

        yticks = (np.ceil(np.linspace(*ylim, 5) * 10) / 10)[:-1]
        plt.yticks(yticks, labels=([""] * yticks.size if i % 2 == 0 else None))
        if i % 2 == 0:
            yticks = plt.gca().get_yticks()
        #else:
        #    plt.ylabel(r"dHdt$^{-1}$ " + DH_MWE_UNIT)
        

        mask = (filtered["dh_m_we"] > lower) & (filtered["dh_m_we"] <= upper)
        corr = np.corrcoef(filtered.loc[mask, col], filtered.loc[mask, "dh_m_we"])[0, 1]
        #corr = np.corrcoef(midpoints, medians)[0, 1]
        plt.text(0.98, 0.98, f"r={corr:.2f}".replace("-", "–"), ha="right", va="top", transform=plt.gca().transAxes,bbox= dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.8))
        plt.text(0.01, 0.98, "ABCD"[i - 1] + ")", fontsize=12, ha="left", va="top", transform=plt.gca().transAxes,bbox= dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.8))
        plt.xlim(bins[0] - abs(bins[0]) * 0.05, bins[-1] * 1.05)
        plt.ylim(ylim)
        plt.grid(zorder=0)

        if i == 2:
            inset = colorbar(MB_CMAP, vmin=ylim[0], vmax=ylim[1], loc=(1.01, 0.1), tick_right=True, height=0.7, width=0.08)
            inset.set_ylabel(DH_MWE_LABEL, labelpad=8, fontsize=10)

    plt.text(0, 0.5, r"dHdt$^{-1}$ " + DH_MWE_UNIT, ha="left", va="center", rotation=90, transform=plt.gcf().transFigure)

    plt.subplots_adjust(left=0.083, bottom=0.092, right=0.905, top=0.886, wspace=0.013, hspace=0.015)
    plt.savefig("temp/figures/mb_correlations.jpg", dpi=600)
    if show:
        plt.show()


@numba.njit(parallel=True)
def downsample_nans(array: np.ndarray, downsample: int = 5):
    """
    Downsample a 2D array, ignoring nans.

    :param array: A 2D array to downsample.
    :param downsample: The downsampling level (2 means half the original width/height)

    :examples:
        >>> array = np.arange(16, dtype="float32").reshape(4, 4)
        >>> array[-1, -1] = np.nan
        >>> downsampled = downsample_nans(array, downsample=2)
        >>> downsampled.shape
        (2, 2)
        >>> downsampled
        array([[ 2.5,  4. ],
               [ 8.5, 10. ]])
    """

    new_array_shape = array.shape[0] // downsample, array.shape[1] // downsample

    new_array = np.zeros(new_array_shape[0] * new_array_shape[1], dtype="float32") + np.nan

    for i in numba.prange(new_array.size):
        min_row = min((i // new_array_shape[1]) * downsample, array.shape[0] - 1)
        min_col = min((i % new_array_shape[1]) * downsample, array.shape[1] - 1)

        max_col = min(min_col + downsample, array.shape[1] - 1)
        max_row = min(min_row + downsample, array.shape[0] - 1)

        values = array[min_row:max_row, min_col:max_col]

        if np.count_nonzero(np.isfinite(values)) == 0:
            continue
        new_array[i] = np.nanmean(values)

    return new_array.reshape(new_array_shape)


def render_all_figures():
    for func in tqdm(
        [
            overview,
            historic_images,
            error_ensemble,
            outline_error,
            regional_dh,
            interpolation_before_and_after,
            west_east_transect,
            mb_correlations
        ],
        desc="Rendering figures",
    ):
        func(show=False)

