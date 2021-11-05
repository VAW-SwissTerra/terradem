"""Functions reading climate data (for visualization only)."""
import concurrent.futures
import io
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

METEOSWISS_STATIONS = [
    "ALT",
    "ANT",
    "RAG",
    "BAS",
    "BER",
    "CHM",
    "CHD",
    "GSB",
    "DAV",
    "ELM",
    "ENG",
    "GVE",
    "GRH",
    "GRC",
    "JUN",
    "CDF",
    "OTL",
    "LUG",
    "LUZ",
    "MER",
    "NEU",
    "PAY",
    "SBE",
    "SAM",
    "SAR",
    "SIA",
    "SIO",
    "STG",
    "SAE",
    "SMA",
]


def read_meteoswiss_data(url: str) -> pd.DataFrame:

    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError()

    text = ""
    header = True
    station = ""
    altitude: float = np.nan
    for line in response.text.splitlines():
        if "Station:" in line:
            station = line.replace("Station:", "").strip()
        elif "Altitude [m asl]:" in line:
            altitude = float(line.replace("Altitude [m asl]:", "").replace(" m", ""))
        if header and not all(s in line for s in ["Year", "Month"]):
            continue
        header = False
        text += line + "\n"

    data = pd.read_csv(io.StringIO(text), sep=r"\s+").astype(float)
    data["station"] = station
    data["altitude"] = altitude

    return data


def read_all_data() -> pd.DataFrame:

    url_constructor = (
        lambda s: "https://www.meteoswiss.admin.ch/product/output/climate-data/"
        f"homogenous-monthly-data-processing/data/homog_mo_{s}.txt"
    )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        data = pd.concat(list(executor.map(read_meteoswiss_data, map(url_constructor, METEOSWISS_STATIONS))))
    data.columns = map(lambda s: s.lower(), data.columns)
    return data


def climate_stats():
    data = read_all_data()

    summer = data[data["month"].isin([6, 7, 8])].groupby("year").mean()
    yearly = data.groupby("year").mean().drop(columns="month")
    yearly["precipitation"] *= 12

    yearly["s_temperature"] = summer["temperature"]

    #yearly = yearly.reset_index().set_index(["altitude", "year"])

    climate_intervals = {
        0: np.arange(1901, 1931),
        1: np.arange(1931, 1961),
        2: np.arange(1961, 1991),
        3: np.arange(1991, 2021)
    }
    for key in climate_intervals:
        yearly.loc[yearly.index.isin(climate_intervals[key]), "climate_interval"] = key


    trends = 100 * yearly.apply(lambda col: np.polyfit(col.index, col.values, 1)[0])


    print("1961–1990:")
    print(yearly[yearly["climate_interval"] == 2].mean(axis=0))

    print("Trends (per 100 years):")
    print(trends)
    #print(change.reset_index().corr())


def mean_climate_deviation(climate_interval: slice = slice(1961, 1990)) -> pd.DataFrame:

    data = read_all_data()
    summer = data[data["month"].isin([6, 7, 8])].groupby("year").mean()
    winter = data[data["month"].isin([12, 1, 2])].groupby("year").mean()

    means = data.groupby("year").mean().drop(columns="month")
    means["precipitation"] *= 12  # This should be the yearly sum

    means["s_temperature"] = summer["temperature"]
    means["w_temperature"] = winter["temperature"]

    all_except_elev = means.columns[means.columns != "altitude"]
    means[all_except_elev] -= means.loc[1991:2020, all_except_elev].mean(axis=0)

    return means
