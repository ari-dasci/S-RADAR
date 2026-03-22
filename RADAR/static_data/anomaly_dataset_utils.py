import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from RADAR.static_data.preprocessing.preprocessing_static import StandardScalerPreprocessing
from RADAR.static_data.static_datasets_uci import global_load, load_kddcup99, load_human_activity_recognition


def load_dataset_silently(dataset_name):
    """Load a dataset from the UCI static dataset registry without printing metadata."""

    with io.StringIO() as buffer, redirect_stdout(buffer):
        X, y = global_load(dataset_name)
    return np.asarray(X, dtype=float), np.asarray(y).astype(int).ravel()


def build_loaded_uci_anomaly_dataset(
    dataset_name,
    normal_label=1,
    target_test_contamination=0.1,
    random_state=42,
    max_train_normals=None,
    max_test_size=None,
    imputer_strategy="median",
    scaler_cls=StandardScalerPreprocessing,
):
    """Build an anomaly-detection benchmark from a loaded UCI dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset registered in ``static_datasets_uci``.
    normal_label : int, default=1
        Label considered normal. All other labels are mapped to anomaly ``1``.
    target_test_contamination : float, default=0.1
        Desired anomaly ratio in the evaluation split.
    random_state : int, default=42
        Random seed for reproducible splits and sampling.
    max_train_normals : int, optional
        Optional cap for the number of normal training samples.
    max_test_size : int, optional
        Optional cap for the number of evaluation samples.
    imputer_strategy : str, default="median"
        Strategy passed to ``SimpleImputer``.
    scaler_cls : type, default=StandardScalerPreprocessing
        Preprocessing class used to scale train and test splits.
    """

    X_raw, y_raw = load_dataset_silently(dataset_name)
    X = pd.DataFrame(X_raw).apply(pd.to_numeric, errors="coerce")
    y = (np.asarray(y_raw).ravel().astype(int) != normal_label).astype(int)

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    train_normals_mask = y_train_full == 0
    X_train_normals = X_train_full.loc[train_normals_mask].copy()

    if max_train_normals is not None and len(X_train_normals) > max_train_normals:
        X_train_normals = X_train_normals.sample(
            n=max_train_normals, random_state=random_state
        )

    imputer = SimpleImputer(strategy=imputer_strategy)
    X_train_normals_imputed = imputer.fit_transform(X_train_normals)
    X_test_full_imputed = imputer.transform(X_test_full)

    normal_test_indices = np.flatnonzero(y_test_full == 0)
    anomaly_test_indices = np.flatnonzero(y_test_full == 1)

    max_anomalies_for_target = int(
        len(normal_test_indices) * target_test_contamination / (1 - target_test_contamination)
    )

    rng = np.random.default_rng(random_state)
    if 0 < max_anomalies_for_target < len(anomaly_test_indices):
        sampled_anomaly_indices = rng.choice(
            anomaly_test_indices, size=max_anomalies_for_target, replace=False
        )
    else:
        sampled_anomaly_indices = anomaly_test_indices

    benchmark_test_indices = np.concatenate([normal_test_indices, sampled_anomaly_indices])

    if max_test_size is not None and len(benchmark_test_indices) > max_test_size:
        benchmark_test_indices = rng.choice(
            benchmark_test_indices, size=max_test_size, replace=False
        )

    rng.shuffle(benchmark_test_indices)

    X_test_benchmark = X_test_full_imputed[benchmark_test_indices]
    y_test_benchmark = y_test_full[benchmark_test_indices]

    scaler = scaler_cls()
    X_train_scaled = scaler.fit_transform(X_train_normals_imputed)
    X_test_scaled = scaler.transform(X_test_benchmark)

    return {
        "name": dataset_name,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_test": y_test_benchmark,
        "n_samples": len(y),
        "n_features": X.shape[1],
        "original_positive_ratio": float(np.mean(y == 1)),
        "benchmark_test_positive_ratio": float(np.mean(y_test_benchmark == 1)),
        "train_normals": int(X_train_normals.shape[0]),
        "test_normals": int(np.sum(y_test_benchmark == 0)),
        "test_anomalies": int(np.sum(y_test_benchmark == 1)),
    }


def build_kddcup99_anomaly_dataset(
    normal_label="normal",
    target_test_contamination=0.1,
    random_state=42,
    max_train_normals=None,
    max_test_size=None,
    scaler_cls=StandardScalerPreprocessing,
):
    """Build an anomaly-detection benchmark from the KDD Cup 99 dataset.

    ``load_kddcup99`` returns three objects ``(data, attack_types, attack_class)``
    instead of the standard ``(X, y)`` pair.  Labels are strings (e.g.
    ``'normal'``, ``'smurf'``, ``'neptune'``).  This helper converts them to
    binary ``0`` (normal) / ``1`` (anomaly) and applies the same benchmark
    pipeline used by ``build_loaded_uci_anomaly_dataset``.

    Parameters
    ----------
    normal_label : str, default="normal"
        String label that represents normal traffic.
    target_test_contamination : float, default=0.1
        Desired anomaly ratio in the evaluation split.
    random_state : int, default=42
        Random seed for reproducible splits and sampling.
    max_train_normals : int, optional
        Optional cap for the number of normal training samples.
    max_test_size : int, optional
        Optional cap for the number of evaluation samples.
    scaler_cls : type, default=StandardScalerPreprocessing
        Preprocessing class used to scale train and test splits.
    """

    with io.StringIO() as buffer, redirect_stdout(buffer):
        data, attack_types, attack_class = load_kddcup99()

    X = pd.DataFrame(data).apply(pd.to_numeric, errors="coerce")
    # Binary labels: 0 = normal, 1 = anomaly
    y = (np.array(attack_class) != normal_label).astype(int)

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    train_normals_mask = y_train_full == 0
    X_train_normals = X_train_full.loc[train_normals_mask].copy()

    if max_train_normals is not None and len(X_train_normals) > max_train_normals:
        X_train_normals = X_train_normals.sample(
            n=max_train_normals, random_state=random_state
        )

    # KDD Cup 99 is already numeric after one-hot encoding in load_kddcup99
    imputer = SimpleImputer(strategy="median")
    X_train_normals_imputed = imputer.fit_transform(X_train_normals)
    X_test_full_imputed = imputer.transform(X_test_full)

    normal_test_indices = np.flatnonzero(y_test_full == 0)
    anomaly_test_indices = np.flatnonzero(y_test_full == 1)

    max_anomalies_for_target = int(
        len(normal_test_indices) * target_test_contamination / (1 - target_test_contamination)
    )

    rng = np.random.default_rng(random_state)
    if 0 < max_anomalies_for_target < len(anomaly_test_indices):
        sampled_anomaly_indices = rng.choice(
            anomaly_test_indices, size=max_anomalies_for_target, replace=False
        )
    else:
        sampled_anomaly_indices = anomaly_test_indices

    benchmark_test_indices = np.concatenate([normal_test_indices, sampled_anomaly_indices])

    if max_test_size is not None and len(benchmark_test_indices) > max_test_size:
        benchmark_test_indices = rng.choice(
            benchmark_test_indices, size=max_test_size, replace=False
        )

    rng.shuffle(benchmark_test_indices)

    X_test_benchmark = X_test_full_imputed[benchmark_test_indices]
    y_test_benchmark = np.asarray(y_test_full)[benchmark_test_indices]

    scaler = scaler_cls()
    X_train_scaled = scaler.fit_transform(X_train_normals_imputed)
    X_test_scaled = scaler.transform(X_test_benchmark)

    return {
        "name": "kddcup99",
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_test": y_test_benchmark,
        "n_samples": len(y),
        "n_features": X.shape[1],
        "attack_types": attack_types,
        "original_positive_ratio": float(np.mean(y == 1)),
        "benchmark_test_positive_ratio": float(np.mean(y_test_benchmark == 1)),
        "train_normals": int(X_train_normals.shape[0]),
        "test_normals": int(np.sum(y_test_benchmark == 0)),
        "test_anomalies": int(np.sum(y_test_benchmark == 1)),
    }


def build_har_anomaly_dataset(
    normal_labels=None,
    target_test_contamination=0.1,
    random_state=42,
    max_train_normals=None,
    max_test_size=None,
    scaler_cls=StandardScalerPreprocessing,
):
    """Build an anomaly-detection benchmark from the Human Activity Recognition dataset.

    ``load_human_activity_recognition`` returns four objects
    ``(X_train, X_test, y_train, y_test)`` already pre-split.  This helper
    concatenates them back into a single ``(X, y)``, maps the 6 activity
    labels to binary normal/anomaly, and applies the standard benchmark
    pipeline.

    Activity labels: 1=WALKING, 2=WALKING_UPSTAIRS, 3=WALKING_DOWNSTAIRS,
    4=SITTING, 5=STANDING, 6=LAYING.

    Parameters
    ----------
    normal_labels : list of int, optional
        Activity labels considered normal.  Defaults to ``[1, 2, 3]``
        (walking activities).  All other labels become anomalies.
    target_test_contamination : float, default=0.1
        Desired anomaly ratio in the evaluation split.
    random_state : int, default=42
        Random seed for reproducible splits and sampling.
    max_train_normals : int, optional
        Optional cap for the number of normal training samples.
    max_test_size : int, optional
        Optional cap for the number of evaluation samples.
    scaler_cls : type, default=StandardScalerPreprocessing
        Preprocessing class used to scale train and test splits.
    """

    if normal_labels is None:
        normal_labels = [1, 2, 3]  # Walking activities are "normal"

    with io.StringIO() as buffer, redirect_stdout(buffer):
        X_tr, X_te, y_tr, y_te = load_human_activity_recognition(
            url="https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
            header=None,
            delim_whitespace=True,
        )

    # Concatenate pre-split data back into a single dataset
    X = pd.DataFrame(np.vstack([X_tr, X_te])).apply(pd.to_numeric, errors="coerce")
    y_raw = np.concatenate([y_tr.ravel(), y_te.ravel()]).astype(int)

    # Binary labels: 0 = normal (in normal_labels), 1 = anomaly
    y = (~np.isin(y_raw, normal_labels)).astype(int)

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    train_normals_mask = y_train_full == 0
    X_train_normals = X_train_full.loc[train_normals_mask].copy()

    if max_train_normals is not None and len(X_train_normals) > max_train_normals:
        X_train_normals = X_train_normals.sample(
            n=max_train_normals, random_state=random_state
        )

    imputer = SimpleImputer(strategy="median")
    X_train_normals_imputed = imputer.fit_transform(X_train_normals)
    X_test_full_imputed = imputer.transform(X_test_full)

    normal_test_indices = np.flatnonzero(y_test_full == 0)
    anomaly_test_indices = np.flatnonzero(y_test_full == 1)

    max_anomalies_for_target = int(
        len(normal_test_indices) * target_test_contamination / (1 - target_test_contamination)
    )

    rng = np.random.default_rng(random_state)
    if 0 < max_anomalies_for_target < len(anomaly_test_indices):
        sampled_anomaly_indices = rng.choice(
            anomaly_test_indices, size=max_anomalies_for_target, replace=False
        )
    else:
        sampled_anomaly_indices = anomaly_test_indices

    benchmark_test_indices = np.concatenate([normal_test_indices, sampled_anomaly_indices])

    if max_test_size is not None and len(benchmark_test_indices) > max_test_size:
        benchmark_test_indices = rng.choice(
            benchmark_test_indices, size=max_test_size, replace=False
        )

    rng.shuffle(benchmark_test_indices)

    X_test_benchmark = X_test_full_imputed[benchmark_test_indices]
    y_test_benchmark = np.asarray(y_test_full)[benchmark_test_indices]

    scaler = scaler_cls()
    X_train_scaled = scaler.fit_transform(X_train_normals_imputed)
    X_test_scaled = scaler.transform(X_test_benchmark)

    activity_names = {
        1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
        4: "SITTING", 5: "STANDING", 6: "LAYING",
    }

    return {
        "name": "human_activity_recognition",
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_test": y_test_benchmark,
        "n_samples": len(y),
        "n_features": X.shape[1],
        "normal_activities": [activity_names[l] for l in normal_labels],
        "anomaly_activities": [activity_names[l] for l in range(1, 7) if l not in normal_labels],
        "original_positive_ratio": float(np.mean(y == 1)),
        "benchmark_test_positive_ratio": float(np.mean(y_test_benchmark == 1)),
        "train_normals": int(X_train_normals.shape[0]),
        "test_normals": int(np.sum(y_test_benchmark == 0)),
        "test_anomalies": int(np.sum(y_test_benchmark == 1)),
    }