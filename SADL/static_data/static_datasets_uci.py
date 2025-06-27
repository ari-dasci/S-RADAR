from ucimlrepo import fetch_ucirepo
from io import BytesIO
import requests
import pandas as pd
import zipfile
import numpy as np

"""
Datasets used in anomaly detection (Source: UCI Machine Learning Repository)
Source: https://archive.ics.uci.edu/ml/index.php

Datasets used:
 1. "shuttle": Dataset on event classifications in a space shuttle, useful for identifying anomalies in control systems.
 2. "kddcup99": Dataset with network traffic logs, used to detect computer attacks and anomalies in security systems.
 3. "spambase": Dataset of e-mail labeled as spam or non-spam, ideal for detecting anomalies in the classification of messages.
 4. "mammographic_mass": Data on mammary tumors, used to identify anomalies in the classification of benign or malignant mammary masses.
 5. "arrhythmia": Electrocardiogram data, useful for detecting heart rhythm abnormalities.
 6. "default_of_credit_card_clients": Dataset containing financial information, used to detect clients with high risk of non-payment.
 7. "Wine Quality": Dataset to detect anomalies in the quality of wines according to various chemical characteristics.
 8. "Detection of IoT Botnet Attacks (N-BaIoT)": Used to detect botnet attacks on IoT devices, focused on identifying anomalous patterns in networks.
 9. "Human Activity Recognition Using Smartphones": A dataset that measures human activities using smartphone sensors, useful for detecting anomalies in human behavior.
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
    return np.array(X),np.array(y)


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

def load_human_activity_recognition(url,**kwargs):
    response = requests.get(url)

    # Descomprimir el archivo ZIP
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        # Listar los archivos dentro del ZIP
        z.printdir()

        # Extraer los archivos necesarios (X_train, X_test, y_train, y_test)
        z.extract("UCI HAR Dataset/train/X_train.txt", "UCI_HAR")
        z.extract("UCI HAR Dataset/test/X_test.txt", "UCI_HAR")
        z.extract("UCI HAR Dataset/train/y_train.txt", "UCI_HAR")
        z.extract("UCI HAR Dataset/test/y_test.txt", "UCI_HAR")

    # Leer los archivos extraídos con pandas
    X_train = pd.read_csv("UCI_HAR/UCI HAR Dataset/train/X_train.txt", **kwargs)
    X_test = pd.read_csv("UCI_HAR/UCI HAR Dataset/test/X_test.txt", **kwargs)

    y_train = pd.read_csv("UCI_HAR/UCI HAR Dataset/train/y_train.txt",**kwargs)
    y_test = pd.read_csv("UCI_HAR/UCI HAR Dataset/test/y_test.txt", **kwargs)
    
    return np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)
    
def load_kddcup99(data_url,names_url,**kwargs):
    """Reads the KDD Cup 99 dataset and processes it."""

    print("Downloading dataset...")
    data = load_from_url(data_url, **kwargs)
    
    print("Downloading names file...")
    response = requests.get(names_url)
    response.raise_for_status()
    lines = response.text.splitlines()
    
    attack_types = lines[0].strip().split(',')
    variable_names = [line.split(':')[0] for line in lines[1:]] + ["attack_type"]
    data.columns = variable_names
    
    # Clean data
    data["attack_type"] = data["attack_type"].str.replace('\.', '', regex=True)
    
        
    attack_class = data.pop("attack_type").values
    attack_types[-1] = attack_types[-1].strip()
    
    return data, attack_types, attack_class    
    

datasets = {
    "shuttle": [load_from_id, {"id": 148}],
    "kddcup99": [
        load_kddcup99,
        {
            "data_url": "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz",
            "names_url":'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names',
            "header":None,
            "compression": "gzip",
        },
    ],
    "spambase": [load_from_id, {"id": 94}],
    "mammographic_mass": [load_from_id, {"id": 161}],
    "arrhythmia": [
        load_from_url,
        {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data",
        },
    ],
    "default_of_credit_card_clients": [load_from_id, {"id": 350}],
    "detection_of_IoT_botnet_attacks_N_BaIoT":[
        load_from_url,
        {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00442/Philips_B120N10_Baby_Monitor/benign_traffic.csv",
        },
    ],
    'wine_quality':[load_from_id, {"id": 186}],
    
    'human_activity_recognition':[load_human_activity_recognition, {"url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip", 'header':None, 'delim_whitespace':True}]
}
