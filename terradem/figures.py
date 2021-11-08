import json
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import shapely.geometry
import shapely.ops
import skimage.exposure

import terradem.climate
import terradem.error
import terradem.files
import terradem.massbalance
import terradem.utilities
import xdem


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
    plt.ylabel(r"dH dt$^{-1}$ (m a$^{-1})$")
    plt.xlabel("Area (m²)")
    plt.tight_layout()

    plt.savefig("temp/figures/dh_histograms.jpg", dpi=300)
    plt.show()


def overview():

    climate = terradem.climate.mean_climate_deviation(slice(1961, 1990))
    fig = plt.figure(figsize=(8, 4.3))

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

    lk50_outlines = gpd.read_file(terradem.files.INPUT_FILE_PATHS["lk50_outlines"])

    xlim = (1900, 2021)
    colors = {
        "precipitation": "royalblue",
        "s_temperature": "red",
        "temperature": "black",
        "w_temperature": "royalblue",
    }

    plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=3)
    plt.imshow(
        slope, cmap="Greys", extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], interpolation="bilinear"
    )
    lk50_outlines.plot(ax=plt.gca(), edgecolor="black", lw=0.1)
    outline.plot(ax=plt.gca(), facecolor="none", edgecolor="black")
    plt.xlim(lk50_outlines.total_bounds[[0, 2]])
    # plt.ylim(lk50_outlines.total_bounds[[1, 3]])
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
    plt.text(0.02, 0.98, "A)", transform=plt.gca().transAxes, fontsize=12, ha="left", va="top")

    xticks = np.arange(1900, 2040, 40)
    plt.subplot2grid((2, 4), (0, 3))
    for key in ["temperature", "s_temperature"]:
        plt.plot(climate[key].rolling(10, min_periods=1).mean(), color=colors[key])
        plt.scatter(climate.index, climate[key], c=colors[key], s=0.5)
    plt.ylabel(r"$\Delta$T ($\degree$C)")
    plt.xlim(xlim)
    plt.xticks(xticks, [""] * xticks.shape[0])
    plt.text(0.02, 0.97, "B)", transform=plt.gca().transAxes, fontsize=12, ha="left", va="top")
    plt.grid(zorder=0, alpha=0.4)

    plt.subplot2grid((2, 4), (1, 3))
    plt.plot(climate["precipitation"].rolling(10, min_periods=1).mean(), color=colors["precipitation"])
    plt.scatter(climate.index, climate["precipitation"], c=colors["precipitation"], s=0.5)
    plt.ylabel(r"$\Delta$P (mm)", labelpad=0)
    plt.xlim(xlim)
    plt.xticks(xticks)
    plt.yticks(rotation=45)
    plt.text(0.02, 0.97, "C)", transform=plt.gca().transAxes, fontsize=12, ha="left", va="top")
    plt.grid(zorder=0, alpha=0.4)

    plt.subplots_adjust(top=0.991, bottom=0.082, left=0.0, right=0.971, hspace=0.034, wspace=0.0)

    plt.savefig("temp/figures/overview.jpg", dpi=600)

    plt.show()


def error_ensemble():
    fig = plt.figure(figsize=(8, 5), dpi=200)
    glacier_wise_dh = pd.read_csv(terradem.files.TEMP_FILES["glacier_wise_dh"])

    grid = (19, 19)
    names = {
        "topo_err": "Stable-ground error",
        "time_err": "Temporal error",
        "area_err": "Area error",
        "interp_err": "Interpolation error",
    }
    plt.subplot2grid(grid, (0, 0), rowspan=grid[0] // 2, colspan=grid[0] // 2)
    terradem.error.glacier_outline_error(plot=14)
    plt.xlim(659300, 661000)
    plt.ylim(160200, 162000)
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.yticks(
        plt.gca().get_yticks()[[1, -1]],
        labels=(plt.gca().get_yticks()[[1, -1]] + 1e6).astype(int),
        rotation=90,
        ha="right",
        va="center",
    )
    plt.xticks(plt.gca().get_xticks()[[1, -1]], labels=(plt.gca().get_xticks()[[1, -1]] + 2e6).astype(int))
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position("top")
    plt.text(0.02, 0.97, "A)", transform=plt.gca().transAxes, fontsize=12, ha="left", va="top")

    plt.subplot2grid(grid, (0, 1 + grid[1] // 2), rowspan=grid[0] // 2, colspan=grid[0] // 2)
    plt.hist(glacier_wise_dh[list(names.keys())], bins=np.linspace(0, 0.4, 20), label=[names[col] for col in names])
    plt.ylabel("Count")
    plt.xlabel(r"dH dt$^{-1}$ (m a$^{-1})$")
    plt.yscale("log")
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position("top")
    plt.text(0.02, 0.97, "B)", transform=plt.gca().transAxes, fontsize=12, ha="left", va="top")
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
                        "C)" if i == 0 else "D)",
                        transform=plt.gca().transAxes,
                        fontsize=12,
                        ha="left",
                        va="top",
                    )

                if i == 1:
                    plt.gca().yaxis.tick_right()
                    plt.gca().yaxis.set_label_position("right")
    plt.subplots_adjust(left=0.065, bottom=0.09, right=0.92, top=0.905, wspace=0, hspace=0.2)
    plt.savefig("temp/figures/error_approach_ensemble.jpg", dpi=600)
    plt.show()


def historic_images():
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

    ylim = (1.1567e6, 1.168e6)
    xlim = (2.670e6, 2.676e6)
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
    plt.show()

