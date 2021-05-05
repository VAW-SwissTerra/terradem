"""DEM coregistration functions."""
import threading
from contextlib import contextmanager
from typing import Optional

import matplotlib.pyplot as plt
import rasterio as rio
import xdem

import terradem.files


def coregister_dem(filepath: str, base_dem_lock: Optional[threading.Lock] = None, stable_ground_lock: Optional[threading.Lock] = None):

    dem_ds = rio.open(filepath)

    base_dem_ds = rio.open(terradem.files.EXTERNAL_DATA_PATHS["base_dem"])
    stable_ground_ds = rio.open(terradem.files.EXTERNAL_DATA_PATHS["stable_ground_mask"])

    assert base_dem_ds.shape == stable_ground_ds.shape

    window = rio.windows.Window(
        *base_dem_ds.index(dem_ds.bounds.left, dem_ds.bounds.top),
        height=dem_ds.height,
        width=dem_ds.width
    )

    dem = dem_ds.read(1)
    # Read the base DEM with a threading lock (if given, otherwise make an empty context)
    with (base_dem_lock or contextmanager(lambda: iter([None]))()):
        base_dem = base_dem_ds.read(1, window=window)
    # Read the stable ground mask with a threading lock (if given, otherwise make an empty context)
    with (stable_ground_lock or contextmanager(lambda: iter([None]))()):
        stable_ground = stable_ground_ds.read(1, window=window).astype(bool)

    assert dem.shape == base_dem.shape == stable_ground.shape

    pipeline = xdem.coreg.BiasCorr() + xdem.coreg.ICP() + xdem.coreg.NuthKaab()

    pipeline.fit(
        reference_dem=base_dem,
        dem_to_be_aligned=dem,
        inlier_mask=stable_ground,
        transform=dem_ds.transform
    )

    dem_coreg = pipeline.apply(dem, transform=dem_ds.transform)

    plt.subplot(211)
    plt.imshow((base_dem - dem).squeeze(), cmap="coolwarm_r", vmin=-20, vmax=20)
    plt.subplot(212)
    plt.imshow((base_dem - dem_coreg).squeeze(), cmap="coolwarm_r", vmin=-20, vmax=20)
    plt.show()

    print(pipeline.to_matrix())
