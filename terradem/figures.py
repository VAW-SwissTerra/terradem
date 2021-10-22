import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import terradem.files
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

    
    fig = plt.figure(figsize=(8, 5), dpi=300)

    for i, col in enumerate(filter(lambda s: "err" in s, data.columns), start=1):
        plt.subplot(3, 2, i)
        plt.title(col)
        plt.hist(data[col], bins=np.linspace(0, 0.5))

    plt.tight_layout()
    plt.savefig("temp/figures/error_histograms.jpg")


def topographic_error_variogram():
    vgm = pd.read_csv(terradem.files.TEMP_FILES["topographic_error_variogram"]).drop(columns=["bins.1"])
    vgm = vgm[vgm["bins"] < 2e5]

    vgm_model, params = xdem.spatialstats.fit_sum_model_variogram(["Sph"] * 2, vgm)

    xdem.spatialstats.plot_vgm(
        vgm,
        xscale_range_split=[100, 1000, 10000],
        list_fit_fun=[vgm_model],
        list_fit_fun_label=["Standardized double-range variogram"]
    )
    fig: plt.Figure = plt.gcf()

    fig.set_size_inches(8, 5)

    for axis in plt.gcf().get_axes():
        if axis.get_legend():
            axis.legend().remove()

    print(params)

    plt.savefig("temp/figures/topographic_error_variogram.jpg", dpi=300)


