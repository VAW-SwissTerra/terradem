"""Utility functions to handle input and output files."""
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile

import requests
from tqdm import tqdm

# URLs to the data
DATA_URLS = {
    "results": "https://schyttholmlund.com/share/terra/terra_results.tar.gz",
    "external": os.getenv("TERRADEM_EXTERNAL_URL") or "",
}


# Expected MD5 checksums for the downloaded files.
CHECKSUMS = {
    "terra_results.tar.gz": "89a705790195f6021a1e79a7c2edcf85",
    "terra_inputs.tar.gz": "4b9c1d5d4bc43cd0d97a303567ffe981",
}

BASE_DIRECTORY = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
TEMP_DIRECTORY = os.path.join(BASE_DIRECTORY, "temp/")

TEMP_SUBDIRS = {
    "dems_coreg": os.path.join(TEMP_DIRECTORY, "dems_coreg"),
    "coreg_matrices": os.path.join(TEMP_DIRECTORY, "coreg_matrices"),
    "orthos_coreg": os.path.join(TEMP_DIRECTORY, "orthos_coreg"),
    "ddems_coreg": os.path.join(TEMP_DIRECTORY, "ddems_coreg/"),
    "ddems_coreg_filtered": os.path.join(TEMP_DIRECTORY, "ddems_coreg_filtered"),
    "ddems_coreg_tcorr": os.path.join(TEMP_DIRECTORY, "ddems_coreg_tcorr/"),
    "ddems_non_coreg": os.path.join(TEMP_DIRECTORY, "ddems_non_coreg"),
    "tcorr_meta_coreg": os.path.join(TEMP_DIRECTORY, "tcorr_meta_coreg/"),
    "rasterized_sgi_zones": os.path.join(TEMP_DIRECTORY, "rasterized_sgi_zones"),
    "hypsometric_signals": os.path.join(TEMP_DIRECTORY, "hypsometric_signals"),
    "base_dem": os.path.join(TEMP_DIRECTORY, "base_dem"),
    "merged_ddems": os.path.join(TEMP_DIRECTORY, "merged_ddems"),
}
TEMP_FILES = {
    "ddem_stats": os.path.join(TEMP_DIRECTORY, "ddem_stats.csv"),
    "ddem_coreg": os.path.join(TEMP_SUBDIRS["merged_ddems"], "ddem_coreg.tif"),
    "ddem_coreg_tcorr": os.path.join(TEMP_SUBDIRS["merged_ddems"], "ddem_coreg_tcorr.tif"),
    "ddem_coreg_tcorr_interp_signal": os.path.join(TEMP_SUBDIRS["hypsometric_signals"], "SGI_national_normalized.csv"),
    "lk50_rasterized": os.path.join(TEMP_DIRECTORY, "lk50_rasterized.tif"),
    "ddem_vs_ideal_error": os.path.join(TEMP_DIRECTORY, "ddem_vs_ideal_error.csv"),
    "ddem_error": os.path.join(TEMP_DIRECTORY, "ddem_error.tif"),
}

TERRAIN_ATTRIBUTES = ["slope", "aspect", "curvature", "planform_curvature", "profile_curvature"]
for attr in TERRAIN_ATTRIBUTES:
    TEMP_FILES[f"base_dem_{attr}"] = os.path.join(TEMP_SUBDIRS["base_dem"], f"base_dem_{attr}.tif")

INTERPOLATION_SCALES = ["national", "subregion0", "subregion1"]

# Add keys for interpolation and extrapolation products from the corrected dDEM
for product in INTERPOLATION_SCALES:
    BASE_KEY = "ddem_coreg_tcorr_" + product

    for ext in ["-interp", "-interp-ideal", "-interp-extrap", "-interp-extrap-ideal"]:
        TEMP_FILES[BASE_KEY + ext] = os.path.join(TEMP_SUBDIRS["merged_ddems"], BASE_KEY + ext + ".tif")


for key in TEMP_SUBDIRS:
    os.makedirs(TEMP_SUBDIRS[key], exist_ok=True)

DIRECTORY_PATHS = {
    "data": os.path.join(BASE_DIRECTORY, "data/"),
    "external": os.path.join(BASE_DIRECTORY, "data/", "external/"),
    "results": os.path.join(BASE_DIRECTORY, "data/", "results/"),
    "dems": os.path.join(BASE_DIRECTORY, "data/results/dems/"),
    "orthos": os.path.join(BASE_DIRECTORY, "data/results/orthos"),
    "manual_input": os.path.join(BASE_DIRECTORY, "manual_input"),
}

INPUT_FILE_PATHS = {
    "base_dem": os.path.join(DIRECTORY_PATHS["external"], "rasters", "base_dem.tif"),
    "base_dem_years": os.path.join(DIRECTORY_PATHS["external"], "rasters", "base_dem_years.tif"),
    "stable_ground_mask": os.path.join(DIRECTORY_PATHS["results"], "masks", "stable_ground_mask.tif"),
    "swisstopo_metadata": os.path.join(DIRECTORY_PATHS["results"], "metadata", "image_meta.csv"),
    "massbalance_index": os.path.join(DIRECTORY_PATHS["external"], "mass_balance", "massbalance_index.dat"),
    "lk50_outlines": os.path.join(DIRECTORY_PATHS["results"], "outlines", "lk50_outlines.shp"),
    "lk50_centerlines": os.path.join(DIRECTORY_PATHS["results"], "outlines", "lk50_centrelines.shp"),
    "sgi_2016": os.path.join(DIRECTORY_PATHS["external"], "shapefiles", "SGI_2016_glaciers.shp"),
    "digitized_outlines": os.path.join(DIRECTORY_PATHS["manual_input"], "digitized_outlines.geojson"),
}


def _download_file(url: str, output_directory: str) -> None:
    """
    Download a file from a url and save it in a given directory.

    Mostly copied from: https://gist.github.com/ruxi/5d6803c116ec1130d484a4ab8c00c603

    :param url: The URL to the file.
    :param output_directory: The directory to save the file.
    """
    temp_dir = tempfile.TemporaryDirectory()

    # The header of the dl link has a Content-Length which is in bytes.
    # The bytes is in string hence has to convert to integer.
    filesize = int(requests.head(url).headers["Content-Length"])

    # os.path.basename returns python-3.8.5-macosx10.9.pkg,
    # without this module I will have to manually split the url by "/"
    # then get the last index with -1.
    # Example:
    # url.split("/")[-1]
    filename = os.path.basename(url)

    # The absolute path to download the python program to.
    dl_path = os.path.join(temp_dir.name, filename)
    chunk_size = 1024

    # Use the requests.get with stream enable, with iter_content by chunk size,
    # the contents will be written to the dl_path.
    # tqdm tracks the progress by progress.update(datasize)
    with requests.get(url, stream=True) as reader, open(dl_path, "wb") as outfile, tqdm(
        unit="B",  # unit string to be displayed.
        # let tqdm to determine the scale in kilo, mega..etc.
        unit_scale=True,
        unit_divisor=1024,  # is used when unit_scale is true
        total=filesize,  # the total iteration.
        # default goes to stderr, this is the display on console.
        file=sys.stdout,
        # prefix to be displayed on progress bar.
        desc=f"Downloading {filename}",
    ) as progress:
        for chunk in reader.iter_content(chunk_size=chunk_size):
            # download the file chunk by chunk
            datasize = outfile.write(chunk)
            # on each chunk update the progress bar.
            progress.update(datasize)

    shutil.move(dl_path, os.path.join(output_directory, filename))


def _verify_hash(filepath: str) -> bool:
    """Return True if the hash of a file matches the expected hash."""
    if not os.path.isfile(filepath):
        return False

    with open(filepath, "rb") as infile:
        md5 = hashlib.md5(infile.read()).hexdigest()

    return md5 == CHECKSUMS[os.path.basename(filepath)]


def _get_directory_size(directory: str) -> int:
    """Get the size of a directory in KiB."""
    if not os.path.isdir(directory):
        return 0
    size = sum(d.stat().st_size for d in os.scandir(directory) if d.is_file())

    return size


def get_data(overwrite: bool = False) -> None:
    """
    Download the data if necessary.

    Validates the checksums of the tarballs.
    If the data directories are not empty and the tarball is valid, nothing happens.
    Otherwise, it will download the file again.

    :param overwrite: Force a re-download of all data and extract them.

    :raises AssertionError: If the hash of a newly downloaded file is invalid.
    """
    for key in DATA_URLS:
        filepath = os.path.join(DIRECTORY_PATHS["data"], os.path.basename(DATA_URLS[key]))
        if overwrite or not _verify_hash(filepath):
            _download_file(DATA_URLS[key], DIRECTORY_PATHS["data"])
            if not _verify_hash(filepath):
                raise AssertionError("Downloaded file hash does not match the expected hash.")

        if not overwrite and _get_directory_size(DIRECTORY_PATHS[key]) > 1024:
            continue
        if os.path.isdir(DIRECTORY_PATHS[key]):
            shutil.rmtree(DIRECTORY_PATHS[key])
        os.makedirs(DIRECTORY_PATHS[key])
        print(f"Extracting {filepath} to {DIRECTORY_PATHS[key]}")
        tarfile.open(filepath).extractall(DIRECTORY_PATHS[key])
