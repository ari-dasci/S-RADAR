from ucimlrepo import fetch_ucirepo
from io import BytesIO
import requests
import pandas as pd
from zipfile import ZipFile


"""
Datasets used in anomaly detection (Source: UCI Machine Learning Repository)
Source: https://archive.ics.uci.edu/ml/index.php

Datasets used:
1. Individual Household Electric Power Consumption: Contains time series data for electric power consumption in individual households.
2. Power Consumption of Tetouan City: This dataset is related to power consumption of three different distribution networks of Tetouan city.
3. MetroPT-3: Provides Data from pressure, temperature, current and intake valve sensors in the air production unit of a metro train, useful for predictive maintenance and fault detection.
4. Metro Interstate Traffic Volume: Contains traffic volume data for highways in the United States, useful for detecting traffic-related anomalies.
5. Gas Sensor Temperature Modulation: Time series data from gas sensors that measure temperature modulation in the presence of various gases, useful for detecting sensor malfunctions or environmental changes.
6. AI4I 2020 Predictive Maintenance Dataset: Contains sensor data from industrial machines, useful for detecting anomalies in machine performance.
7. Online Retail II: Time series data from online retail sales, useful for detecting anomalies in e-commerce transactions and sales patterns.
8. Gas Sensor Array Drift at Different Concentrations: Contains time series data from gas sensors exposed to different concentrations, useful for detecting sensor drift or environmental anomalies.

"""

def global_load(name_dataset):
    """
    Loads a dataset using the corresponding loading method and parameters.

    Parameters:
    name_dataset (str): The name of the dataset to be loaded.

    Returns:
    The dataset loaded using the corresponding method.
    """
    method_load = datasets[name_dataset][0]
    kwargs = datasets[name_dataset][1]
    return method_load(**kwargs)
    
# fetch dataset
def load_from_id(id):
    """
    Fetches a dataset from the UCI repository using its ID.

    Parameters:
    id (int): The identifier of the dataset in the UCI repository.

    Returns:
    tuple: A tuple containing:
        - X (pd.DataFrame): The feature matrix.
        - y (pd.Series or np.array): The target variable.
    """
    dataset = fetch_ucirepo(id=id)
    X = dataset.data.features
    y = dataset.data.targets
    
    print("Metadata:",dataset.metadata)
    # variable information 
    print("Variable information:", dataset.variables)
    return X,y


def load_from_url(url, **kwargs):
    """
    Loads a dataset from a given URL.

    Parameters:
    url (str): The URL from which to fetch the dataset.
    **kwargs: Additional arguments to be passed to `pd.read_csv()`.

    Returns:
    pd.DataFrame: The dataset loaded from the URL.
    """
    
    data = requests.get(url).content
    dataset = pd.read_csv(BytesIO(data), **kwargs)
    return dataset

def load_from_zip_url(url, filename, **kwargs):
    """
    Loads a specific file from a ZIP archive hosted at a given URL.
    The function supports handling different file types inside the ZIP, including nested ZIP files.
    If the file inside the ZIP is another ZIP, it will extract that nested ZIP and load the file inside it.
    If the file is a CSV or an Excel file, it will load it into a Pandas DataFrame.

    Parameters:
    url (str): The URL from which to fetch the ZIP archive containing the dataset.
    filename (str): The name of the file inside the ZIP archive to be loaded. This file can be in CSV, Excel (.xlsx), or ZIP format.
    **kwargs: Additional arguments to be passed to `pd.read_csv()` or `pd.read_excel()`, depending on the file format. These arguments can include parameters like `sep`, `header`, `index_col`, etc., for CSV files or `sheet_name`, `dtype`, etc., for Excel files.

    Returns:
    DataFrame: A Pandas DataFrame containing the dataset loaded from the specified file.
    
    Behavior:
    - If the file inside the ZIP is another ZIP file, the function will open the nested ZIP and attempt to load its contents.
    - If the file is a CSV (identified by the `.csv` extension), it will be read using `pandas.read_csv()`.
    - If the file is an Excel file (identified by the `.xlsx` extension), it will be read using `pandas.read_excel()`.
    - If the file is of an unsupported format, the function raises a `ValueError`.
    """
    
   
    response = requests.get(url)
    with ZipFile(BytesIO(response.content)) as z:
        file_list = z.namelist()
        
        if filename not in file_list:
            raise ValueError(f"The file '{filename}' not found in the ZIP. Available files: {file_list}")

        with z.open(filename) as file:
            if filename.endswith('.zip'):
                return ZipFile(BytesIO(file.read()))
            else:
                
                if filename.endswith('.csv'):
                    dataset = pd.read_csv(file, **kwargs)
                elif filename.endswith('.xlsx'):
                    dataset = pd.read_excel(file, **kwargs)
                else:
                    raise ValueError(f"Unsupported file type: {filename}")
    
    return dataset


def load_gas_sensor_dataset(url, filename, **kwargs):
    """
    Loads multiple CSV files from a nested ZIP archive hosted at a given URL,
    and combines them into a single DataFrame, preserving the timestamp for each file.

    Parameters:
    url (str): The URL from which to fetch the ZIP dataset.
    filename (str): The name of the file inside the ZIP archive that might contain another ZIP.
    **kwargs: Additional arguments to be passed to `pd.read_csv()`.

    Returns:
    pd.DataFrame: A combined DataFrame with all the CSV files inside the nested ZIP archive.
    """
    # Use the load_from_zip_url function to get the contents of the zip file
    nested_zip = load_from_zip_url(url, filename, **kwargs)
    combined_data = []

    # Iterate over CSV files in the last ZIP file
    for file_name in sorted(nested_zip.namelist()):  # Sort files by their name (timestamp)
        if file_name.endswith('.csv'):  
            with nested_zip.open(file_name) as file:
                df = pd.read_csv(file, **kwargs)
                timestamp = file_name.split('.')[0]  # Get timestamp from filename
                df.insert(0, 'timestamp', timestamp)
                
                combined_data.append(df)

    combined_df = pd.concat(combined_data, ignore_index=True)
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], format='%Y%m%d_%H%M%S')
    
    return combined_df


datasets = {
    "individual_household_electric_power_consumption": [load_from_id, {"id": 235}],
    "power_consumption_of_tetouan_city": [load_from_id, {"id": 849}],
    "MetroPT-3": [
        load_from_zip_url,
        {
            "url": "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip",
            "filename": "MetroPT3(AirCompressor).csv",
            "index_col":0,
        },
    ],
    "metro_interstate_traffic_volume": [load_from_id, {"id": 492}],
    "gas_sensor_temperature_modulation": [
        load_gas_sensor_dataset,
        {
            "url": "https://archive.ics.uci.edu/static/public/487/gas+sensor+array+temperature+modulation.zip",
            "filename": "gas-sensor-array-temperature-modulation.zip",
        },
    ],    
    "ai4i_2020_predictive_maintenance_dataset": [load_from_id, {"id": 601}],
    "online_retail_II": [
        load_from_zip_url,
        {
            "url": "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
            "filename": "online_retail_II.xlsx",
        },
    ]
}    
    






























