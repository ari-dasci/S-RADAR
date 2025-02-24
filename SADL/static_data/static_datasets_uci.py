from ucimlrepo import fetch_ucirepo
from io import BytesIO
import requests
import pandas as pd


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



datasets = {
    "shuttle": [load_from_id, {"id": 148}],
    "kddcup99": [
        load_from_url,
        {
            "url": "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz",
            "compression": "gzip",
        },
    ],
    "spambase": [load_from_id, {"id": 94}],
    "mammographic_mass": [load_from_id, {"id": 161}],
    "arrhythmia": [
        load_from_url,
        {
            "url": "https://archive.ics.uci.edu/static/public/5/arrhythmia.zip",
            "compression": "zip",
        },
    ],
    
}






# DATASETS = {
#     "kddcup99": "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz",
#     "spambase": "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
#     "shuttle": 'https://archive.ics.uci.edu/dataset/148/statlog+shuttle',
   
# }

# def download_dataset(url, format="pandas", **kwargs):
#     """Download a dataset into memory and load it as Pandas or NumPy without saving it to disk."""
#     print(f"Downloading dataset from {url}...")
#     try:
#         data= requests.get(url)
#     except Exception as e:
#         raise ConnectionError(f"Failed to download dataset: {e}")
    
#     if format == "pandas":
#         return pd.read_csv(io.BytesIO(data.content), **kwargs)
#     elif format == "numpy":
#         return pd.read_csv(io.BytesIO(data.content), **kwargs).to_numpy()
#     else:
#         raise ValueError("Format not supported. Use ‘pandas’ or ‘numpy’.")

# def load_kddcup99(url,format="pandas", **kwargs):
#     data = download_dataset(DATASETS["kddcup99"], format=format, compression='gzip', **kwargs)
#     return data.to_numpy() if format == "numpy" else data

# def load_spambase(format="pandas", **kwargs):
#     return download_dataset(DATASETS["spambase"], format=format, header=None, **kwargs)

# def load_shuttle(format="pandas", **kwargs):
#     return download_dataset(DATASETS["shuttle"], format=format, delim_whitespace=True, **kwargs)


# def load_dataset(name, format="pandas", **kwargs):
#     """General function for loading static datasets."""
#     loaders = {
#         "kddcup99": load_kddcup99,
#         "spambase": load_spambase,
#         "shuttle": load_shuttle,
       
#     }

#     if name in loaders:
#         return loaders[name](format=format, **kwargs)
#     else:
#         raise ValueError(f"Dataset {name} not supported. Options: {list(loaders.keys())}")
