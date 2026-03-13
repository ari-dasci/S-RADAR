import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from RADAR.static_data.preprocessing.preprocessing_static import StandardScalerPreprocessing
from RADAR.static_data.static_datasets_uci import global_load


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