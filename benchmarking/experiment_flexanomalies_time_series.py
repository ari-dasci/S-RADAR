from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _time_series_experiment_utils import (
    DEFAULT_METRO_HIGH_Q,
    DEFAULT_METRO_LOW_Q,
    DEFAULT_STEP_SIZE,
    DEFAULT_TEST_SIZE,
    DEFAULT_WINDOW_SIZE,
    aggregate_window_labels,
    build_autoencoder_windows,
    build_forecasting_windows,
    build_raw_dataset_summary,
    build_raw_time_series_dataset_configs,
    ensure_project_root_on_path,
    print_dataframe,
    resolve_project_root,
)


PROJECT_ROOT = resolve_project_root(Path(__file__))
ensure_project_root_on_path(PROJECT_ROOT)

from flex.pool import FlexPool
from flexanomalies.pool.aggregators_cl import aggregate_cl
from flexanomalies.pool.aggregators_favg import aggregate_ae
from flexanomalies.pool.aggregators_pca import aggregate_pca
from flexanomalies.pool.primitives_cluster import (
    build_server_model_cl,
    copy_model_to_clients_cl,
    get_clients_weights_cl,
    set_aggregated_weights_cl,
    train_cl,
)
from flexanomalies.pool.primitives_deepmodel import (
    build_server_model_ae,
    copy_model_to_clients_ae,
    set_aggregated_weights_ae,
    train_ae,
    weights_collector_ae,
)
from flexanomalies.pool.primitives_iforest import (
    aggregate_if,
    build_server_model_if,
    copy_model_to_clients_if,
    get_clients_weights_if,
    set_aggregated_weights_if,
    train_if,
)
from flexanomalies.pool.primitives_pca import (
    build_server_model_pca,
    copy_model_to_clients_pca,
    get_clients_weights_pca,
    set_aggregated_weights_pca,
    train_pca,
)
from flexanomalies.utils import AutoEncoder, ClusterAnomaly, DeepCNN_LSTM, IsolationForest, PCA_Anomaly
from flexanomalies.utils.load_data import federate_data

from RADAR.federated_data.algorithms import flexanomalies
import RADAR.metrics_module as metrics_module


def flatten_1d(values):
    return np.asarray(values).astype(float).ravel()


def binary_1d(values):
    return np.asarray(values).astype(int).ravel()


def summarize_mse(scores):
    scores = np.asarray(scores, dtype=float).ravel()
    return float(scores.mean()) if scores.size else np.nan


def align_binary_targets(targets, expected_len):
    targets = np.asarray(targets)
    window_targets = aggregate_window_labels(targets)
    flat_targets = binary_1d(targets)

    if len(window_targets) == expected_len:
        return window_targets
    if len(flat_targets) == expected_len:
        return flat_targets
    if len(window_targets) > expected_len:
        return window_targets[:expected_len]
    if len(flat_targets) > expected_len:
        return flat_targets[:expected_len]
    return np.resize(window_targets, expected_len).astype(int)


def align_scores(scores, expected_len):
    scores = flatten_1d(scores)
    if len(scores) >= expected_len:
        return scores[:expected_len]
    return np.resize(scores, expected_len)


def resolve_contamination_ratio(ratio, default=0.1, min_value=0.01, max_value=0.5):
    if ratio is None or not np.isfinite(ratio):
        return float(default)
    return float(np.clip(ratio, min_value, max_value - 1e-6))


def compute_time_series_metrics_row(y_true, y_pred, scores):
    y_true = binary_1d(y_true)
    y_pred = binary_1d(y_pred)
    scores = flatten_1d(scores)

    limit = min(len(y_true), len(y_pred), len(scores))
    y_true = y_true[:limit]
    y_pred = y_pred[:limit]
    scores = scores[:limit]

    finite_scores = bool(np.isfinite(scores).all())
    mse = summarize_mse(scores) if finite_scores else np.nan

    return {
        "accuracy": round(metrics_module.metric_accuracy(y_true, y_pred) / 100, 4),
        "precision": round(metrics_module.metric_precision(y_true, y_pred), 4),
        "recall": round(metrics_module.metric_recall(y_true, y_pred), 4),
        "mse": round(float(mse), 6) if np.isfinite(mse) else np.nan,
        "evaluated_samples": int(limit),
    }


direct_flex_model_classes = {
    "isolationForest": IsolationForest,
    "pcaAnomaly": PCA_Anomaly,
    "clusterAnomaly": ClusterAnomaly,
    "autoencoder": AutoEncoder,
    "deepCNN_LSTM": DeepCNN_LSTM,
}


direct_federated_ops = {
    "isolationForest": {
        "build_model": build_server_model_if,
        "copy": copy_model_to_clients_if,
        "train": train_if,
        "collect": get_clients_weights_if,
        "aggregate": aggregate_if,
        "set_weights": set_aggregated_weights_if,
    },
    "pcaAnomaly": {
        "build_model": build_server_model_pca,
        "copy": copy_model_to_clients_pca,
        "train": train_pca,
        "collect": get_clients_weights_pca,
        "aggregate": aggregate_pca,
        "set_weights": set_aggregated_weights_pca,
    },
    "clusterAnomaly": {
        "build_model": build_server_model_cl,
        "copy": copy_model_to_clients_cl,
        "train": train_cl,
        "collect": get_clients_weights_cl,
        "aggregate": aggregate_cl,
        "set_weights": set_aggregated_weights_cl,
    },
    "autoencoder": {
        "build_model": build_server_model_ae,
        "copy": copy_model_to_clients_ae,
        "train": train_ae,
        "collect": weights_collector_ae,
        "aggregate": aggregate_ae,
        "set_weights": set_aggregated_weights_ae,
    },
    "deepCNN_LSTM": {
        "build_model": build_server_model_ae,
        "copy": copy_model_to_clients_ae,
        "train": train_ae,
        "collect": weights_collector_ae,
        "aggregate": aggregate_ae,
        "set_weights": set_aggregated_weights_ae,
    },
}


def extract_prediction_labels(model_object, prediction_output):
    for candidate in (
        getattr(model_object, "labels_", None),
        getattr(getattr(model_object, "model", None), "labels_", None),
        getattr(prediction_output, "labels_", None),
    ):
        if candidate is not None:
            return binary_1d(candidate)
    return binary_1d(prediction_output)


def predict_and_score_model(model_object, X, y=None):
    prediction_output = model_object.predict(X, y) if y is not None else model_object.predict(X)
    labels = extract_prediction_labels(model_object, prediction_output)
    scores = model_object.decision_function(X, y) if y is not None else model_object.decision_function(X)
    return labels, flatten_1d(scores)


def build_direct_model_kwargs(model_kwargs):
    excluded_keys = {"algorithm_", "label_parser", "n_clients", "n_rounds"}
    return {key: value for key, value in model_kwargs.items() if key not in excluded_keys}


def train_platform_model(model_kwargs, X_train, y_train):
    model = flexanomalies.FlexAnomalyDetection(**model_kwargs)
    model.fit(X_train, y_train)
    return model


def train_direct_federated_model(model_kwargs, X_train, y_train):
    algorithm_name = model_kwargs["algorithm_"]
    direct_model_cls = direct_flex_model_classes[algorithm_name]
    direct_model = direct_model_cls(**build_direct_model_kwargs(model_kwargs))
    federated_ops = direct_federated_ops[algorithm_name]

    flex_dataset = federate_data(model_kwargs["n_clients"], X_train, y_train)
    pool = FlexPool.client_server_pool(
        fed_dataset=flex_dataset,
        server_id=f"{algorithm_name}_server",
        init_func=federated_ops["build_model"],
        model=direct_model,
    )

    for _ in range(model_kwargs["n_rounds"]):
        pool.servers.map(federated_ops["copy"], pool.clients)
        pool.clients.map(federated_ops["train"])
        pool.aggregators.map(federated_ops["collect"], pool.clients)
        if algorithm_name == "clusterAnomaly":
            pool.aggregators.map(federated_ops["aggregate"], model=direct_model)
        else:
            pool.aggregators.map(federated_ops["aggregate"])
        pool.aggregators.map(federated_ops["set_weights"], pool.servers)

    return pool.servers._models[f"{algorithm_name}_server"]["model"]


def measure_training_time(train_callable):
    start_time = time.perf_counter()
    trained_model = train_callable()
    return time.perf_counter() - start_time, trained_model


def build_model_configs(window_size: int, step_size: int, n_rounds: int) -> list[dict]:
    return [
        {
            "algorithm_": "autoencoder",
            "builder": lambda config: build_autoencoder_windows(
                config,
                window_size=window_size,
                step_size=step_size,
            ),
            "base_kwargs": {
                "epochs": 100,
                "batch_size": 16,
                "neurons": [32, 16, 32],
                "hidden_act": ["relu", "relu", "relu"],
                "preprocess": False,
                "w_size": window_size,
                "n_pred": 1,
                "n_clients": 3,
                "n_rounds": n_rounds,
            },
        },
        {
            "algorithm_": "deepCNN_LSTM",
            "builder": lambda config: build_forecasting_windows(
                config,
                window_size=window_size,
                step_size=step_size,
                n_pred=1,
            ),
            "base_kwargs": {
                "epochs": 100,
                "batch_size": 8,
                "filters_cnn": [8, 6],
                "units_lstm": [8, 6],
                "kernel_size": [4, 4],
                "hidden_act": ["relu", "relu"],
                "w_size": window_size,
                "n_pred": 1,
                "n_clients": 3,
                "n_rounds": n_rounds,
            },
        },
    ]


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_configs = build_raw_time_series_dataset_configs(
        test_size=args.test_size,
        low_q=args.metro_low_q,
        high_q=args.metro_high_q,
        dataset_keys=args.datasets,
    )
    dataset_summary = build_raw_dataset_summary(dataset_configs)
    print_dataframe(dataset_summary, title="Dataset summary")

    model_configs = [
        model
        for model in build_model_configs(args.window_size, args.step_size, args.n_rounds)
        if model["algorithm_"] in args.models
    ]
    time_series_results = []
    time_series_timing_results = []

    for dataset_key, config in dataset_configs.items():
        print(f"\nTime-series dataset: {config['dataset']}")

        for model_config in model_configs:
            X_train_windows, y_train_windows, X_test_windows, y_test_windows, y_eval_reference = model_config["builder"](config)
            contamination = resolve_contamination_ratio(config.get("positive_ratio_points"))

            model_kwargs = {
                "algorithm_": model_config["algorithm_"],
                "contamination": contamination,
                "label_parser": None,
                "input_dim": int(config["n_features"]),
                **model_config["base_kwargs"],
            }

            platform_time_s, platform_model = measure_training_time(
                lambda mk=model_kwargs, Xw=X_train_windows, yw=y_train_windows: train_platform_model(mk, Xw, yw)
            )

            if model_config["algorithm_"] == "deepCNN_LSTM":
                platform_predictions, platform_raw_scores = predict_and_score_model(platform_model, X_test_windows, y_test_windows)
            else:
                platform_predictions, platform_raw_scores = predict_and_score_model(platform_model, X_test_windows)

            platform_scores = align_scores(platform_raw_scores, len(platform_predictions))
            y_true_platform = align_binary_targets(y_eval_reference, len(platform_predictions))
            platform_metrics = compute_time_series_metrics_row(y_true_platform, platform_predictions, platform_scores)

            result_row = {
                "data_type": "time_series",
                "dataset": dataset_key,
                "dataset_name": config["dataset"],
                "algorithm": model_config["algorithm_"],
                "window_size": args.window_size,
                "contamination": round(contamination, 4),
                "train_windows": int(len(X_train_windows)),
                "test_windows": int(len(X_test_windows)),
                **platform_metrics,
            }
            time_series_results.append(result_row)
            print(result_row)

            if args.run_timing:
                base_time_s, base_model = measure_training_time(
                    lambda mk=model_kwargs, Xw=X_train_windows, yw=y_train_windows: train_direct_federated_model(mk, Xw, yw)
                )

                if model_config["algorithm_"] == "deepCNN_LSTM":
                    base_predictions, base_raw_scores = predict_and_score_model(base_model, X_test_windows, y_test_windows)
                else:
                    base_predictions, base_raw_scores = predict_and_score_model(base_model, X_test_windows)

                base_scores = align_scores(base_raw_scores, len(base_predictions))
                y_true_base = align_binary_targets(y_eval_reference, len(base_predictions))
                base_metrics = compute_time_series_metrics_row(y_true_base, base_predictions, base_scores)

                time_series_timing_results.append(
                    {
                        "dataset": dataset_key,
                        "dataset_name": config["dataset"],
                        "algorithm": model_config["algorithm_"],
                        "timing_repetitions": 5,
                        "platform_time_s": round(platform_time_s, 4),
                        "base_time_s": round(base_time_s, 4),
                        "overhead_s": round(platform_time_s - base_time_s, 4),
                        "speedup_base_over_platform": round(base_time_s / platform_time_s, 4) if platform_time_s > 0 else np.nan,
                        "platform_mse": platform_metrics["mse"],
                        "base_mse": base_metrics["mse"],
                        "mse_diff": round(platform_metrics["mse"] - base_metrics["mse"], 6)
                        if np.isfinite(platform_metrics["mse"]) and np.isfinite(base_metrics["mse"])
                        else np.nan,
                    }
                )

    time_series_results_df = pd.DataFrame(time_series_results)
    if not time_series_results_df.empty:
        time_series_results_df = time_series_results_df.sort_values(
            ["dataset", "mse"],
            ascending=[True, True],
            na_position="last",
        ).reset_index(drop=True)

    time_series_timing_results_df = pd.DataFrame(time_series_timing_results)
    if args.run_timing and not time_series_timing_results_df.empty:
        time_series_timing_results_df = time_series_timing_results_df.sort_values(
            ["dataset", "speedup_base_over_platform"],
            ascending=[True, False],
            na_position="last",
        ).reset_index(drop=True)

    return time_series_results_df, time_series_timing_results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the time-series section of the FlexAnomalies experiment notebook as a terminal script.",
    )
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["ai4i", "metro_interstate"],
        default=["ai4i", "metro_interstate"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["autoencoder", "deepCNN_LSTM"],
        default=["autoencoder", "deepCNN_LSTM"],
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--step-size", type=int, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--n-rounds", type=int, default=8)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--metro-low-q", type=float, default=DEFAULT_METRO_LOW_Q)
    parser.add_argument("--metro-high-q", type=float, default=DEFAULT_METRO_HIGH_Q)
    parser.add_argument("--run-timing", action="store_true", default=True)
    parser.add_argument("--skip-timing", action="store_false", dest="run_timing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    results_df, timing_df = run_experiment(args)
    print_dataframe(results_df, title="Time-series results")

    results_path = args.results_dir / "uci_flexanomalies_time_series_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved time-series results to: {results_path}")

    if args.run_timing:
        print_dataframe(timing_df, title="Time-series timing")
        timing_path = args.results_dir / "uci_flexanomalies_time_series_timing_results.csv"
        timing_df.to_csv(timing_path, index=False)
        print(f"Saved time-series timing results to: {timing_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
