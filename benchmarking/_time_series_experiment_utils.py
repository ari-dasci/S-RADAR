from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_WINDOW_SIZE = 24
DEFAULT_STEP_SIZE = 1
DEFAULT_TEST_SIZE = 0.2
DEFAULT_METRO_LOW_Q = 0.05
DEFAULT_METRO_HIGH_Q = 0.95


def resolve_project_root(reference_path: Path | None = None) -> Path:
    candidate = (reference_path or Path.cwd()).resolve()
    locations = [candidate]
    locations.extend(candidate.parents)
    if candidate.is_file():
        locations.insert(1, candidate.parent)

    seen = set()
    for location in locations:
        if location in seen:
            continue
        seen.add(location)
        if (location / "RADAR").is_dir() and (location / "requirements.txt").exists():
            return location

    raise RuntimeError("Could not locate the project root containing 'RADAR/' and 'requirements.txt'.")


def ensure_project_root_on_path(project_root: Path) -> None:
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


def print_dataframe(dataframe: pd.DataFrame, title: str | None = None, max_rows: int | None = None) -> None:
    if title:
        print(f"\n{title}")
        print("-" * len(title))
    if dataframe.empty:
        print("<empty>")
        return
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", None, "display.width", 200):
        print(dataframe.to_string(index=False))


def chronological_split(X, y, test_size: float = DEFAULT_TEST_SIZE):
    split_idx = int(len(X) * (1 - test_size))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def aggregate_window_labels(y_windows):
    y_windows = np.asarray(y_windows)
    if y_windows.ndim == 1:
        return y_windows.astype(int)
    return (y_windows.reshape(y_windows.shape[0], -1).sum(axis=1) > 0).astype(int)


def _load_time_series_dependencies():
    from RADAR.time_series.preprocessing.preprocessing_ts import StandardScalerPreprocessing
    from RADAR.time_series.time_series_datasets_uci import global_load as load_time_series
    from RADAR.time_series.time_series_utils import TimeSeriesProcessor

    return StandardScalerPreprocessing, load_time_series, TimeSeriesProcessor


def prepare_ai4i_windowed_dataset(
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
):
    StandardScalerPreprocessing, load_time_series, TimeSeriesProcessor = _load_time_series_dependencies()

    X, y = load_time_series("ai4i_2020_predictive_maintenance_dataset")
    labels = y["Machine failure"].astype(int).to_numpy()
    X = X.drop(columns=["Type"], errors="ignore")

    scaler = StandardScalerPreprocessing()
    X_scaled = scaler.fit_transform(X)
    X_values = np.asarray(X_scaled, dtype=np.float32)

    X_train, X_test, y_train, y_test = chronological_split(X_values, labels, test_size=test_size)

    processor = TimeSeriesProcessor(window_size=window_size, step_size=step_size, future_prediction=False)
    X_train_windows, y_train_windows, X_test_windows, y_test_windows = processor.process_train_test(
        X_train, y_train, X_test, y_test
    )

    y_test_labels = aggregate_window_labels(y_test_windows)
    return {
        "dataset": "ai4i_2020_predictive_maintenance_dataset",
        "X_train_windows": np.asarray(X_train_windows, dtype=np.float32),
        "X_test_windows": np.asarray(X_test_windows, dtype=np.float32),
        "y_test_windows": np.asarray(y_test_windows),
        "y_test_labels": y_test_labels,
        "n_samples": len(X_values),
        "n_features": X_values.shape[1],
        "window_size": window_size,
        "train_windows": len(X_train_windows),
        "test_windows": len(X_test_windows),
        "positive_ratio_points": round(float(np.mean(labels)), 4),
        "positive_ratio_windows": round(float(np.mean(y_test_labels)), 4),
        "label_note": "Machine failure from UCI target",
    }


def prepare_metro_windowed_dataset(
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    low_q: float = DEFAULT_METRO_LOW_Q,
    high_q: float = DEFAULT_METRO_HIGH_Q,
):
    StandardScalerPreprocessing, load_time_series, TimeSeriesProcessor = _load_time_series_dependencies()

    X, y = load_time_series("metro_interstate_traffic_volume")
    traffic_volume = y["traffic_volume"].astype(float)
    low_threshold = float(traffic_volume.quantile(low_q))
    high_threshold = float(traffic_volume.quantile(high_q))
    labels = ((traffic_volume <= low_threshold) | (traffic_volume >= high_threshold)).astype(int).to_numpy()

    X = X.drop(columns=["date_time", "holiday", "weather_main", "weather_description"], errors="ignore")

    scaler = StandardScalerPreprocessing()
    X_scaled = scaler.fit_transform(X)
    X_values = np.asarray(X_scaled, dtype=np.float32)

    X_train, X_test, y_train, y_test = chronological_split(X_values, labels, test_size=test_size)

    processor = TimeSeriesProcessor(window_size=window_size, step_size=step_size, future_prediction=False)
    X_train_windows, y_train_windows, X_test_windows, y_test_windows = processor.process_train_test(
        X_train, y_train, X_test, y_test
    )

    y_test_labels = aggregate_window_labels(y_test_windows)
    return {
        "dataset": "metro_interstate_traffic_volume",
        "X_train_windows": np.asarray(X_train_windows, dtype=np.float32),
        "X_test_windows": np.asarray(X_test_windows, dtype=np.float32),
        "y_test_windows": np.asarray(y_test_windows),
        "y_test_labels": y_test_labels,
        "n_samples": len(X_values),
        "n_features": X_values.shape[1],
        "window_size": window_size,
        "train_windows": len(X_train_windows),
        "test_windows": len(X_test_windows),
        "positive_ratio_points": round(float(np.mean(labels)), 4),
        "positive_ratio_windows": round(float(np.mean(y_test_labels)), 4),
        "label_note": f"Extreme traffic volume: <= q{low_q:.2f} or >= q{high_q:.2f}",
        "low_threshold": round(low_threshold, 3),
        "high_threshold": round(high_threshold, 3),
    }


def build_windowed_dataset_configs(
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    low_q: float = DEFAULT_METRO_LOW_Q,
    high_q: float = DEFAULT_METRO_HIGH_Q,
    dataset_keys: Iterable[str] | None = None,
) -> dict[str, dict]:
    all_configs = {
        "ai4i": prepare_ai4i_windowed_dataset(window_size=window_size, step_size=step_size, test_size=test_size),
        "metro_interstate": prepare_metro_windowed_dataset(
            window_size=window_size,
            step_size=step_size,
            test_size=test_size,
            low_q=low_q,
            high_q=high_q,
        ),
    }
    if dataset_keys is None:
        return all_configs
    return {dataset_key: all_configs[dataset_key] for dataset_key in dataset_keys}


def build_windowed_dataset_summary(dataset_configs: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_key": dataset_key,
                "dataset_name": config["dataset"],
                "samples": config["n_samples"],
                "features": config["n_features"],
                "window_size": config["window_size"],
                "train_windows": config["train_windows"],
                "test_windows": config["test_windows"],
                "positive_ratio_points": config["positive_ratio_points"],
                "positive_ratio_windows": config["positive_ratio_windows"],
                "label_note": config["label_note"],
            }
            for dataset_key, config in dataset_configs.items()
        ]
    ).reset_index(drop=True)


def prepare_ai4i_raw_dataset(test_size: float = DEFAULT_TEST_SIZE):
    StandardScalerPreprocessing, load_time_series, _ = _load_time_series_dependencies()

    X, y = load_time_series("ai4i_2020_predictive_maintenance_dataset")
    labels = y["Machine failure"].astype(int).to_numpy()
    X = X.drop(columns=["Type"], errors="ignore")

    scaler = StandardScalerPreprocessing()
    X_scaled = scaler.fit_transform(X)
    X_values = np.asarray(X_scaled, dtype=np.float32)
    X_train, X_test, y_train, y_test = chronological_split(X_values, labels, test_size=test_size)

    return {
        "dataset": "ai4i_2020_predictive_maintenance_dataset",
        "X_train": np.asarray(X_train, dtype=np.float32),
        "X_test": np.asarray(X_test, dtype=np.float32),
        "y_train": np.asarray(y_train, dtype=int),
        "y_test": np.asarray(y_test, dtype=int),
        "n_samples": len(X_values),
        "n_features": X_values.shape[1],
        "positive_ratio_points": round(float(np.mean(labels)), 4),
        "label_note": "Machine failure from UCI target",
    }


def prepare_metro_raw_dataset(
    test_size: float = DEFAULT_TEST_SIZE,
    low_q: float = DEFAULT_METRO_LOW_Q,
    high_q: float = DEFAULT_METRO_HIGH_Q,
):
    StandardScalerPreprocessing, load_time_series, _ = _load_time_series_dependencies()

    X, y = load_time_series("metro_interstate_traffic_volume")
    traffic_volume = y["traffic_volume"].astype(float)
    low_threshold = float(traffic_volume.quantile(low_q))
    high_threshold = float(traffic_volume.quantile(high_q))
    labels = ((traffic_volume <= low_threshold) | (traffic_volume >= high_threshold)).astype(int).to_numpy()

    X = X.drop(columns=["date_time", "holiday", "weather_main", "weather_description"], errors="ignore")

    scaler = StandardScalerPreprocessing()
    X_scaled = scaler.fit_transform(X)
    X_values = np.asarray(X_scaled, dtype=np.float32)
    X_train, X_test, y_train, y_test = chronological_split(X_values, labels, test_size=test_size)

    return {
        "dataset": "metro_interstate_traffic_volume",
        "X_train": np.asarray(X_train, dtype=np.float32),
        "X_test": np.asarray(X_test, dtype=np.float32),
        "y_train": np.asarray(y_train, dtype=int),
        "y_test": np.asarray(y_test, dtype=int),
        "n_samples": len(X_values),
        "n_features": X_values.shape[1],
        "positive_ratio_points": round(float(np.mean(labels)), 4),
        "label_note": f"Extreme traffic volume: <= q{low_q:.2f} or >= q{high_q:.2f}",
        "low_threshold": round(low_threshold, 3),
        "high_threshold": round(high_threshold, 3),
    }


def build_raw_time_series_dataset_configs(
    test_size: float = DEFAULT_TEST_SIZE,
    low_q: float = DEFAULT_METRO_LOW_Q,
    high_q: float = DEFAULT_METRO_HIGH_Q,
    dataset_keys: Iterable[str] | None = None,
) -> dict[str, dict]:
    all_configs = {
        "ai4i": prepare_ai4i_raw_dataset(test_size=test_size),
        "metro_interstate": prepare_metro_raw_dataset(test_size=test_size, low_q=low_q, high_q=high_q),
    }
    if dataset_keys is None:
        return all_configs
    return {dataset_key: all_configs[dataset_key] for dataset_key in dataset_keys}


def build_raw_dataset_summary(dataset_configs: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_key": dataset_key,
                "dataset_name": config["dataset"],
                "samples": config["n_samples"],
                "features": config["n_features"],
                "positive_ratio_points": config["positive_ratio_points"],
                "label_note": config["label_note"],
            }
            for dataset_key, config in dataset_configs.items()
        ]
    ).reset_index(drop=True)


def build_autoencoder_windows(config: dict, window_size: int = DEFAULT_WINDOW_SIZE, step_size: int = DEFAULT_STEP_SIZE):
    _, _, TimeSeriesProcessor = _load_time_series_dependencies()
    processor = TimeSeriesProcessor(window_size=window_size, step_size=step_size, future_prediction=False)
    X_train_windows, y_train_windows, X_test_windows, y_test_windows = processor.process_train_test(
        config["X_train"], config["y_train"], config["X_test"], config["y_test"]
    )
    return X_train_windows, y_train_windows, X_test_windows, y_test_windows, aggregate_window_labels(y_test_windows)


def build_forecasting_windows(
    config: dict,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
    n_pred: int = 1,
):
    _, _, TimeSeriesProcessor = _load_time_series_dependencies()
    processor = TimeSeriesProcessor(window_size=window_size, step_size=step_size, future_prediction=True, n_pred=n_pred)
    X_train_windows, y_train_windows, X_test_windows, y_test_windows, label_test_windows = processor.process_train_test(
        config["X_train"],
        config["y_train"],
        config["X_test"],
        config["y_test"],
        l_test=config["y_test"],
    )
    return X_train_windows, y_train_windows, X_test_windows, y_test_windows, aggregate_window_labels(label_test_windows)
