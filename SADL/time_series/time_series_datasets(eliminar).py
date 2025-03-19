import pandas as pd
import wget
import os
import numpy as np
import gzip
import shutil

def oneHotEncode(data, columns):
    """
    This function performs one hot encoding over a set of columns.

    Parameters
    ----------
    data : array-like
        Data for performing one-hot encoding.
    columns : str list
        Names of the columns to perform one-hot encoding.

    Returns
    -------
    data : array-like
        Dataset modified with one-hot encoding.
    """
    for col in columns:
        one_hot = pd.get_dummies(data[col], prefix=col)
        data = data.drop(col, axis=1)
        data = data.join(one_hot)
    return data


def readKDDCup99Dataset(output_directory=None):
    """
    This function reads the KDD Cup 99 dataset.

    Parameters
    ----------
    route : str
        Route of the dataset.
    route_names : str
        Route of the names of the variables.

    Returns
    -------
    data : array-like
        Array with the read data.
    attack_types : array-like
        Array with the possible attacks.
    attack_class : array-like
        Labels of the data.
    """
    if output_directory == None:
        output_directory = os.getcwd()
    url = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz'
    if not os.path.exists(os.path.join(output_directory, 'kddcup.data.gz')): 
        print("downloading")
        filename = wget.download(url, out=output_directory)
    
        with gzip.open('kddcup.data.gz', 'rb') as f_in:
            with open('kddcup.data', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    if not os.path.exists(os.path.join(output_directory, 'kddcup.names')): 
        print("downloading names")
        url = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names'
        filename = wget.download(url, out=output_directory)

    data = pd.read_csv('kddcup.data', sep=",")
    names_file = open('kddcup.names', "r")
    variable_names = []
    attack_types = []
    first_line = True
    for line in names_file:
        if first_line:
            attack_types = line[:-1].split(",")
            first_line = False
        else:
            variable_names.append(line.split(":")[0])
    variable_names.append("attack_type")
    data.columns = variable_names
    data["attack_type"] = data["attack_type"].replace('\.','', regex=True)
    data = data.astype({"duration": "float",
                        "protocol_type": "str",
                        "service": "str",
                        "flag": "str",
                        "src_bytes": "float",
                        "dst_bytes": "float",
                        "land": "str",
                        "wrong_fragment": "float",
                        "urgent": "float",
                        "hot": "float",
                        "num_failed_logins": "float",
                        "logged_in": "str",
                        "num_compromised": "float",
                        "root_shell": "float",
                        "su_attempted": "float",
                        "num_root": "float",
                        "num_file_creations": "float",
                        "num_shells": "float",
                        "num_access_files": "float",
                        "num_outbound_cmds": "float",
                        "is_host_login": "str",
                        "is_guest_login": "str",
                        "count": "float",
                        "srv_count": "float",
                        "serror_rate": "float",
                        "srv_serror_rate": "float",
                        "rerror_rate": "float",
                        "srv_rerror_rate": "float",
                        "same_srv_rate": "float",
                        "diff_srv_rate": "float",
                        "srv_diff_host_rate": "float",
                        "dst_host_count": "float",
                        "dst_host_srv_count": "float",
                        "dst_host_same_srv_rate": "float",
                        "dst_host_diff_srv_rate": "float",
                        "dst_host_same_src_port_rate": "float",
                        "dst_host_srv_diff_host_rate": "float",
                        "dst_host_serror_rate": "float",
                        "dst_host_srv_serror_rate": "float",
                        "dst_host_rerror_rate": "float",
                        "dst_host_srv_rerror_rate": "float",
                    })
    
    
    data = oneHotEncode(data, ["protocol_type", "service", "flag", "land", "logged_in", "is_host_login", "is_guest_login",])
    attack_class = np.array(data["attack_type"])
    data = data.drop("attack_type", axis=1)

    attack_types[-1] = attack_types[-1][:-1]
    return np.array(data), attack_types, attack_class
