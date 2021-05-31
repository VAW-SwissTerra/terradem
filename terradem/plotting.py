"""Plotting functions for different steps in the processing."""
import os

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd

import terradem.outlines
import terradem.files
import terradem.utilities


def plot_hypsometric_signals():
    """Plot the normalized hypsometric signals."""
    files = terradem.utilities.list_files(
        terradem.files.TEMP_SUBDIRS["hypsometric_signals"], r".*\.csv"
    )

    outlines = terradem.outlines.get_sgi_regions(level=1)


    shape = (7, 4)

    plt.figure(figsize=(10, 10))
    for i, filepath in enumerate(files, start=1):
        sgi_zone = os.path.basename(filepath).split("_")[1]

        max_area = outlines[sgi_zone].geometry.area.sum()

        signal = pd.read_csv(filepath)

        signal.index = pd.IntervalIndex.from_tuples(
            signal.iloc[:, 0].apply(
                lambda s: tuple(
                    map(float, s.replace("(", "").replace("]", "").split(","))
                )
            )
        )

        signal.drop(columns=signal.columns[0], inplace=True)
        signal = signal[~signal["median"].isna()]
        
        covered_area = signal["count"].sum() * 5 ** 2

        coverage_percentage = 100 * (covered_area / max_area)

        plt.subplot(*shape, i)

        plt.title(f"{sgi_zone}: {coverage_percentage:.1f}% cov.")

        plt.fill_between(
            signal.index.mid,
            signal["w_mean"] + signal["std"],
            signal["w_mean"] - signal["std"],
            alpha=0.3,
        )
        plt.plot(signal.index.mid, signal["w_mean"])
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()
