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
    build_windowed_dataset_configs,
    build_windowed_dataset_summary,
    ensure_project_root_on_path,
    print_dataframe,
    resolve_project_root,
)


PROJECT_ROOT = resolve_project_root(Path(__file__))
ensure_project_root_on_path(PROJECT_ROOT)

from RADAR.time_series.algorithms import transformers


def tensor_to_numpy(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def summarize_mse(scores):
    scores = np.asarray(scores, dtype=float).ravel()
    return float(scores.mean()) if scores.size else np.nan


def build_model_configs(
    train_epochs: int, batch_size: int, learning_rate: float
) -> list[dict]:
    return [
        {
            "algorithm_": "transformer",
            "d_model": 64,
            "d_qk": 64,
            "d_v": 64,
            "n_layers": 2,
            "n_heads": 8,
            "ulayers_feedfwd": 128,
            "dropout_rate": 0.1,
            "attns_outs": False,
            "train_epochs": train_epochs,
            "batch_size": batch_size,
            "lr": learning_rate,
        },
        {
            "algorithm_": "informer",
            "d_model": 64,
            "n_heads": 8,
            "e_layers": 2,
            "d_layers": 1,
            "d_ff": 128,
            "factor": 5,
            "dropout": 0.1,
            "attn": "prob",
            "activation": "gelu",
            "output_attention": False,
            "distil": True,
            "mix": True,
            "train_epochs": train_epochs,
            "batch_size": batch_size,
            "lr": learning_rate,
        },
        {
            "algorithm_": "autoformer",
            "d_model": 64,
            "n_heads": 8,
            "e_layers": 2,
            "d_layers": 1,
            "d_ff": 128,
            "factor": 5,
            "moving_avg": 5,
            "dropout": 0.1,
            "activation": "gelu",
            "output_attention": False,
            "train_epochs": train_epochs,
            "batch_size": batch_size,
            "lr": learning_rate,
        },
    ]


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_configs = build_windowed_dataset_configs(
        window_size=args.window_size,
        step_size=args.step_size,
        test_size=args.test_size,
        low_q=args.metro_low_q,
        high_q=args.metro_high_q,
        dataset_keys=args.datasets,
    )
    dataset_summary = build_windowed_dataset_summary(dataset_configs)
    print_dataframe(dataset_summary, title="Dataset summary")

    transformer_results = []
    transformer_model_configs = build_model_configs(
        args.train_epochs, args.batch_size, args.learning_rate
    )

    for dataset_key, config in dataset_configs.items():
        input_dim = config["X_train_windows"].shape[2]
        seq_len = config["window_size"]

        print(f"\nDataset: {config['dataset']}")
        print(
            f"Features: {input_dim} | Train windows: {config['train_windows']} | "
            f"Test windows: {config['test_windows']}"
        )

        for model_template in transformer_model_configs:
            model_params = dict(model_template)
            algorithm_name = model_params["algorithm_"]

            if algorithm_name == "transformer":
                model_params.update(
                    {
                        "label_parser": None,
                        "size_enc_in": input_dim,
                        "size_dec_in": input_dim,
                        "seq_len": seq_len,
                    }
                )
            elif algorithm_name == "informer":
                model_params.update(
                    {
                        "label_parser": None,
                        "enc_in": input_dim,
                        "dec_in": input_dim,
                        "c_out": input_dim,
                        "seq_len": seq_len,
                        "label_len": seq_len,
                        "out_len": seq_len,
                    }
                )
            elif algorithm_name == "autoformer":
                model_params.update(
                    {
                        "label_parser": None,
                        "enc_in": input_dim,
                        "dec_in": input_dim,
                        "c_out": input_dim,
                        "seq_len": seq_len,
                        "label_len": seq_len,
                        "pred_len": seq_len,
                    }
                )

            model = transformers.TransformersAnomalyDetection(**model_params)

            train_start = time.time()
            model.fit(config["X_train_windows"])
            train_time = time.time() - train_start

            scores = tensor_to_numpy(
                model.decision_function(config["X_test_windows"])
            ).ravel()

            finite_scores = bool(np.isfinite(scores).all())
            mse = summarize_mse(scores) if finite_scores else np.nan

            print(f"  Model: {algorithm_name}")
            print(
                f"    MSE={mse:.6f}"
                if np.isfinite(mse)
                else "    MSE=nan (non-finite scores)"
            )

            transformer_results.append(
                {
                    "dataset_key": dataset_key,
                    "dataset_name": config["dataset"],
                    "algorithm": algorithm_name,
                    "window_size": seq_len,
                    "n_features": input_dim,
                    "train_windows": config["train_windows"],
                    "test_windows": config["test_windows"],
                    "platform_time_s": round(train_time, 4),
                    "mse": round(float(mse), 6) if np.isfinite(mse) else np.nan,
                }
            )

    transformer_results_df = (
        pd.DataFrame(transformer_results)
        .sort_values(
            ["dataset_name", "mse"],
            ascending=[True, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    transformer_summary_df = (
        transformer_results_df.groupby(["dataset_name", "algorithm"], as_index=False)
        .agg(
            {
                "mse": "min",
                "platform_time_s": "mean",
            }
        )
        .sort_values(
            ["dataset_name", "mse"], ascending=[True, True], na_position="last"
        )
        .reset_index(drop=True)
    )

    return transformer_results_df, transformer_summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the notebook experiment for Transformer time-series models as a terminal script.",
    )
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["ai4i", "metro_interstate"],
        default=["ai4i", "metro_interstate"],
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--step-size", type=int, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--metro-low-q", type=float, default=DEFAULT_METRO_LOW_Q)
    parser.add_argument("--metro-high-q", type=float, default=DEFAULT_METRO_HIGH_Q)
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    results_df, summary_df = run_experiment(args)
    print_dataframe(results_df, title="Detailed results")
    print_dataframe(summary_df, title="Per-dataset summary")

    results_main_path = args.results_dir / "uci_transformers_results.csv"
    results_summary_path = args.results_dir / "uci_transformers_summary.csv"
    results_df.to_csv(results_main_path, index=False)
    summary_df.to_csv(results_summary_path, index=False)

    print(f"\nSaved detailed results to: {results_main_path}")
    print(f"Saved summary results to: {results_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
