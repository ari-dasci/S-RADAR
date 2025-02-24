import pandas as pd
import numpy as np
import urllib.request
import io
import zipfile

TIME_SERIES_DATASETS = {
    "nab_power_demand": "https://archive.ics.uci.edu/ml/machine-learning-databases/00592/NAB_power_demand.csv",
    "ecg_five_days": "https://archive.ics.uci.edu/ml/machine-learning-databases/mitdb/100.dat",
    "electricity_consumption": "https://archive.ics.uci.edu/ml/machine-learning-databases/00192/household_power_consumption.zip",
    "swat": "https://itrust.sutd.edu.sg/itrust-labs/datasets/2019_dataset.zip",
    "smap": "https://www.nasa.gov/sites/default/files/atoms/files/smap_data.zip",  # Nuevo dataset SMAP
}

def download_dataset(url):
    """Downloads a dataset into memory and returns its contents in bytes."""
    print(f"Downloads dataset from {url}...")
    response = urllib.request.urlopen(url)
    return response.read()

def load_nab_power_demand(format="pandas", **kwargs):
    data = download_dataset(TIME_SERIES_DATASETS["nab_power_demand"])
    df = pd.read_csv(io.BytesIO(data), **kwargs)
    return df.to_numpy() if format == "numpy" else df

def load_ecg_five_days(format="pandas", **kwargs):
    data = download_dataset(TIME_SERIES_DATASETS["ecg_five_days"])
    df = pd.read_csv(io.BytesIO(data), **kwargs)
    return df.to_numpy() if format == "numpy" else df

def load_electricity_consumption(format="pandas", **kwargs):
    data = download_dataset(TIME_SERIES_DATASETS["electricity_consumption"])
    with zipfile.ZipFile(io.BytesIO(data), "r") as zip_ref:
        file_list = zip_ref.namelist()
        main_file = file_list[0]  # Selecciona el primer archivo dentro del zip
        with zip_ref.open(main_file) as file:
            df = pd.read_csv(file, **kwargs)
    return df.to_numpy() if format == "numpy" else df

def load_swat(format="pandas", **kwargs):
    data = download_dataset(TIME_SERIES_DATASETS["swat"])
    with zipfile.ZipFile(io.BytesIO(data), "r") as zip_ref:
        file_list = zip_ref.namelist()
        main_file = file_list[0]  # Selecciona el archivo principal
        with zip_ref.open(main_file) as file:
            df = pd.read_csv(file, **kwargs)
    return df.to_numpy() if format == "numpy" else df

def load_smap(format="pandas", **kwargs):
    data = download_dataset(TIME_SERIES_DATASETS["smap"])
    with zipfile.ZipFile(io.BytesIO(data), "r") as zip_ref:
        file_list = zip_ref.namelist()
        main_file = file_list[0]  # Selecciona el archivo principal
        with zip_ref.open(main_file) as file:
            df = pd.read_csv(file, **kwargs)
    return df.to_numpy() if format == "numpy" else df

def load_dataset(name, format="pandas", **kwargs):
    """General function to load time series datasets."""
    loaders = {
        "nab_power_demand": load_nab_power_demand,
        "ecg_five_days": load_ecg_five_days,
        "electricity_consumption": load_electricity_consumption,
        "swat": load_swat,
        "smap": load_smap,  # Nuevo dataset SMAP
    }

    if name in loaders:
        return loaders[name](format=format, **kwargs)
    else:
        raise ValueError(f"Dataset {name} not supported. Options: {list(loaders.keys())}")
