"""Functions to handle image metadata."""
import pandas as pd

import terradem.files


def read_swisstopo_metadata():
    data = pd.read_csv(terradem.files.INPUT_FILE_PATHS["swisstopo_metadata"]).set_index(
        "Image file", drop=True
    )
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")

    return data


def get_stereo_pair_dates() -> pd.Series:
    data = read_swisstopo_metadata()
    data["stereo_pair"] = data["station_name"].str.replace("_L", "").replace("_R", "")

    dates = data.groupby("stereo_pair").first()["date"]

    return dates
