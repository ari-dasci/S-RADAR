from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _time_series_experiment_utils import (
    ensure_project_root_on_path,
    print_dataframe,
    resolve_project_root,
)


PROJECT_ROOT = resolve_project_root(Path(__file__))
ensure_project_root_on_path(PROJECT_ROOT)

from RADAR.static_data.algorithms import pyod
import RADAR.metrics_module as metrics_module
import RADAR.static_data.anomaly_dataset_utils as anomaly_dataset_utils

# ---------------------------------------------------------------------------
# PyOD direct imports (original models)
# ---------------------------------------------------------------------------
from pyod.models.abod import ABOD
from pyod.models.alad import ALAD
from pyod.models.anogan import AnoGAN
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.gmm import GMM
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.kde import KDE
from pyod.models.knn import KNN
from pyod.models.lmdd import LMDD
from pyod.models.lof import LOF
from pyod.models.lscp import LSCP
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA

# ---------------------------------------------------------------------------
# Classical models
# ---------------------------------------------------------------------------
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.cof import COF
from pyod.models.sod import SOD
from pyod.models.sos import SOS
from pyod.models.loda import LODA
# from pyod.models.loci import LOCI  # Skipped: extremely slow on large datasets
from pyod.models.kpca import KPCA
from pyod.models.rod import ROD
from pyod.models.qmcd import QMCD
from pyod.models.sampling import Sampling
from pyod.models.cd import CD
from pyod.models.rgraph import RGraph
from pyod.models.suod import SUOD

# ---------------------------------------------------------------------------
# Deep Learning models
# ---------------------------------------------------------------------------
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.ae1svm import AE1SVM
from pyod.models.dif import DIF
from pyod.models.lunar import LUNAR


# ===================================================================
# Dataset builder
# ===================================================================

def build_uci_dataset_configs(
    max_train_normals: int = 8000,
    max_test_size: int = 5000,
    target_test_contamination: float = 0.1,
) -> dict:
    """Build the UCI anomaly-detection benchmark datasets."""
    return {
        "shuttle": anomaly_dataset_utils.build_loaded_uci_anomaly_dataset(
            dataset_name="shuttle",
            normal_label=1,
            target_test_contamination=target_test_contamination,
            max_train_normals=max_train_normals,
            max_test_size=max_test_size,
        ),
        "arrhythmia": anomaly_dataset_utils.build_loaded_uci_anomaly_dataset(
            dataset_name="arrhythmia",
            normal_label=1,
            target_test_contamination=target_test_contamination,
        ),
    }


def build_dataset_summary(dataset_configs: dict) -> pd.DataFrame:
    rows = []
    for dataset_name, config in dataset_configs.items():
        rows.append(
            {
                "dataset": dataset_name,
                "samples": config["n_samples"],
                "features": config["n_features"],
                "original_anomaly_ratio": round(config["original_positive_ratio"], 4),
                "benchmark_test_contamination": round(
                    config["benchmark_test_positive_ratio"], 4
                ),
                "train_normals_used": config["train_normals"],
                "test_normals": config["test_normals"],
                "test_anomalies": config["test_anomalies"],
            }
        )
    return pd.DataFrame(rows)


# ===================================================================
# Model definitions
# ===================================================================

ORIGINAL_MODELS = [
    {"algorithm_": "abod"},
    {"algorithm_": "alad", "epochs": 30, "verbose": 0},
    {"algorithm_": "anogan", "epochs": 30, "verbose": 0},
    {"algorithm_": "cblof"},
    {"algorithm_": "feature_bagging"},
    {"algorithm_": "gmm"},
    {"algorithm_": "hbos"},
    {"algorithm_": "iforest", "random_state": 42},
    {"algorithm_": "inne", "random_state": 42},
    {"algorithm_": "kde"},
    {"algorithm_": "knn", "n_neighbors": 5},
    {"algorithm_": "lmdd"},
    {"algorithm_": "lof", "n_neighbors": 5},
    {"algorithm_": "lscp", "detector_list": [LOF(), LOF()]},
    {"algorithm_": "mcd"},
    {"algorithm_": "ocsvm"},
    {"algorithm_": "pca"},
]

DIRECT_PYOD_ORIGINAL = {
    "abod": ABOD,
    "alad": ALAD,
    "anogan": AnoGAN,
    "cblof": CBLOF,
    "feature_bagging": FeatureBagging,
    "gmm": GMM,
    "hbos": HBOS,
    "iforest": IForest,
    "inne": INNE,
    "kde": KDE,
    "knn": KNN,
    "lmdd": LMDD,
    "lof": LOF,
    "lscp": LSCP,
    "mcd": MCD,
    "ocsvm": OCSVM,
    "pca": PCA,
}

CLASSICAL_MODELS = [
    {"algorithm_": "copod"},
    {"algorithm_": "ecod"},
    {"algorithm_": "cof", "n_neighbors": 5},
    {"algorithm_": "sod", "n_neighbors": 5},
    {"algorithm_": "sos"},
    {"algorithm_": "loda"},
    # {"algorithm_": "loci"},  # Skipped: extremely slow on large datasets
    {"algorithm_": "kpca"},
    {"algorithm_": "rod"},
    {"algorithm_": "qmcd"},
    {"algorithm_": "sampling"},
    {"algorithm_": "cd"},
    {"algorithm_": "rgraph"},
    {"algorithm_": "suod"},
]

DIRECT_PYOD_CLASSICAL = {
    "copod": COPOD,
    "ecod": ECOD,
    "cof": COF,
    "sod": SOD,
    "sos": SOS,
    "loda": LODA,
    # "loci": LOCI,  # Skipped
    "kpca": KPCA,
    "rod": ROD,
    "qmcd": QMCD,
    "sampling": Sampling,
    "cd": CD,
    "rgraph": RGraph,
    "suod": SUOD,
}

DEEP_LEARNING_MODELS = [
    {"algorithm_": "auto_encoder", "epoch_num": 30, "verbose": 0},  # uses epoch_num
    {"algorithm_": "vae", "epoch_num": 30, "verbose": 0},            # uses epoch_num
    {"algorithm_": "deep_svdd", "epochs": 30, "verbose": 0},         # n_features injected at runtime
    {"algorithm_": "ae1svm", "epochs": 30},                           # no verbose support
    {"algorithm_": "dif"},
    {"algorithm_": "lunar"},
]

DIRECT_PYOD_DEEP = {
    "auto_encoder": AutoEncoder,
    "vae": VAE,
    "deep_svdd": DeepSVDD,
    "ae1svm": AE1SVM,
    "dif": DIF,
    "lunar": LUNAR,
}


# ===================================================================
# Generic experiment runner
# ===================================================================

def _evaluate_model_group(
    dataset_configs: dict,
    model_param_list: list[dict],
    direct_class_map: dict,
    category: str,
) -> list[dict]:
    """Run a group of models on every dataset and return a list of result dicts."""
    results: list[dict] = []

    for dataset_name, config in dataset_configs.items():
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name} - {category}")
        print(
            f"Training samples: {config['train_normals']} | "
            f"Test contamination: {config['benchmark_test_positive_ratio']:.3f}"
        )
        print("=" * 60)

        for model_params in model_param_list:
            algorithm_name = model_params["algorithm_"]
            shared_kwargs = {k: v for k, v in model_params.items() if k != "algorithm_"}

            try:
                model_kwargs = {
                    **model_params,
                    "contamination": config["benchmark_test_positive_ratio"],
                }

                # Inject dataset-dependent positional params
                # (e.g. DeepSVDD requires n_features at __init__ time)
                if algorithm_name == "deep_svdd":
                    n_feat = config["X_train"].shape[1]
                    model_kwargs["n_features"] = n_feat
                    shared_kwargs["n_features"] = n_feat

                print(f"\n  Training {algorithm_name}...", end=" ", flush=True)

                # --- RADAR Platform timing ---
                model = pyod.PyodAnomalyDetection(**model_kwargs)
                start_platform = time.time()
                model.fit(config["X_train"])
                platform_fit_time = time.time() - start_platform

                start_predict = time.time()
                predictions = np.asarray(model.predict(config["X_test"])).astype(int).ravel()
                scores = np.asarray(model.decision_function(config["X_test"])).ravel()
                platform_predict_time = time.time() - start_predict
                platform_total_time = platform_fit_time + platform_predict_time

                # --- Direct PyOD timing ---
                direct_cls = direct_class_map[algorithm_name]
                direct_model = direct_cls(
                    contamination=config["benchmark_test_positive_ratio"],
                    **shared_kwargs,
                )
                start_direct = time.time()
                direct_model.fit(config["X_train"])
                direct_fit_time = time.time() - start_direct

                start_direct_pred = time.time()
                _ = direct_model.predict(config["X_test"])
                _ = direct_model.decision_function(config["X_test"])
                direct_predict_time = time.time() - start_direct_pred
                direct_total_time = direct_fit_time + direct_predict_time

                # Overhead / speedup
                overhead = platform_total_time - direct_total_time
                speedup = (
                    direct_total_time / platform_total_time
                    if platform_total_time > 0
                    else np.nan
                )

                # Metrics
                accuracy = metrics_module.metric_accuracy(config["y_test"], predictions) / 100
                precision = metrics_module.metric_precision(config["y_test"], predictions)
                recall = metrics_module.metric_recall(config["y_test"], predictions)
                f1 = metrics_module.metric_F1score(config["y_test"], predictions)

                finite_scores = np.isfinite(scores)
                if finite_scores.all():
                    roc_auc = metrics_module.metric_AUC_ROC_scores(config["y_test"], scores)
                    pr_auc = metrics_module.metric_PR_AUC(config["y_test"], scores)
                else:
                    roc_auc = np.nan
                    pr_auc = np.nan

                print(
                    f"F1={f1:.3f}, ROC-AUC={roc_auc:.3f} | "
                    f"Platform: {platform_total_time:.4f}s, Direct: {direct_total_time:.4f}s, "
                    f"Overhead: {overhead:.4f}s"
                )

                results.append(
                    {
                        "dataset": dataset_name,
                        "algorithm": algorithm_name,
                        "category": category,
                        "contamination": round(config["benchmark_test_positive_ratio"], 4),
                        "accuracy": round(accuracy, 4),
                        "precision": round(precision, 4),
                        "recall": round(recall, 4),
                        "f1": round(f1, 4),
                        "roc_auc": round(float(roc_auc), 4) if np.isfinite(roc_auc) else np.nan,
                        "pr_auc": round(float(pr_auc), 4) if np.isfinite(pr_auc) else np.nan,
                        "platform_time_s": round(platform_total_time, 4),
                        "direct_time_s": round(direct_total_time, 4),
                        "overhead_s": round(overhead, 4),
                        "speedup": round(speedup, 4) if not np.isnan(speedup) else np.nan,
                    }
                )

            except Exception as e:
                print(f"Error - {str(e)[:80]}")
                results.append(
                    {
                        "dataset": dataset_name,
                        "algorithm": algorithm_name,
                        "category": category,
                        "contamination": round(config["benchmark_test_positive_ratio"], 4),
                        "accuracy": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "roc_auc": np.nan,
                        "pr_auc": np.nan,
                        "platform_time_s": np.nan,
                        "direct_time_s": np.nan,
                        "overhead_s": np.nan,
                        "speedup": np.nan,
                        "error": str(e)[:100],
                    }
                )

    return results


def _sort_results(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["dataset", "f1"], ascending=[True, False])
        .reset_index(drop=True)
    )


# ===================================================================
# Main experiment
# ===================================================================

# Mapping from group name to (model_param_list, direct_class_map, category_label, csv_filename)
GROUP_REGISTRY = {
    "original": (
        ORIGINAL_MODELS,
        DIRECT_PYOD_ORIGINAL,
        "original",
        "uci_pyod_original_results.csv",
    ),
    "classical": (
        CLASSICAL_MODELS,
        DIRECT_PYOD_CLASSICAL,
        "classical_new",
        "uci_pyod_new_classical_results.csv",
    ),
    "deep_learning": (
        DEEP_LEARNING_MODELS,
        DIRECT_PYOD_DEEP,
        "deep_learning",
        "uci_pyod_deep_learning_results.csv",
    ),
}

ALL_GROUPS = list(GROUP_REGISTRY.keys())  # ["original", "classical", "deep_learning"]


def run_experiment(
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    """Execute selected PyOD experiment group(s) and return result DataFrames.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain ``group`` (one of 'original', 'classical',
        'deep_learning', 'all') plus the usual dataset parameters.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping  group_name → sorted results DataFrame.
    """

    dataset_configs = build_uci_dataset_configs(
        max_train_normals=args.max_train_normals,
        max_test_size=args.max_test_size,
        target_test_contamination=args.target_contamination,
    )

    summary_df = build_dataset_summary(dataset_configs)
    print_dataframe(summary_df, title="Dataset summary")

    # Decide which groups to run
    groups_to_run = ALL_GROUPS if args.group == "all" else [args.group]

    group_dfs: dict[str, pd.DataFrame] = {}

    for group_name in groups_to_run:
        model_list, direct_map, category, _csv = GROUP_REGISTRY[group_name]
        results = _evaluate_model_group(
            dataset_configs, model_list, direct_map, category
        )
        df = _sort_results(pd.DataFrame(results))
        print_dataframe(df, title=f"Results – {group_name}")
        group_dfs[group_name] = df

    # --- Summary analysis (uses whatever groups were run) ---
    combined_df = _sort_results(
        pd.concat(group_dfs.values(), ignore_index=True)
    )

    print("\n" + "=" * 60)
    print("Top 10 Models by F1 Score (per dataset)")
    print("=" * 60)
    for dataset in combined_df["dataset"].unique():
        print(f"\n{dataset.upper()}")
        top = combined_df[combined_df["dataset"] == dataset].head(10)
        print_dataframe(
            top[["algorithm", "category", "f1", "roc_auc", "pr_auc",
                 "platform_time_s", "overhead_s", "speedup"]]
        )

    print("\n" + "=" * 60)
    print("Timing Analysis: RADAR Platform vs Direct PyOD")
    print("=" * 60)
    print("\n• speedup > 1: Direct PyOD is faster than RADAR platform")
    print("• speedup < 1: RADAR platform is faster than Direct PyOD")
    print("• speedup ≈ 1: Both have similar performance")
    print("• overhead > 0: RADAR adds overhead (slower)")
    print("• overhead < 0: RADAR is faster (negative overhead)\n")

    for dataset in combined_df["dataset"].unique():
        print(f"\n{dataset.upper()} - Overhead Analysis")
        dataset_df = combined_df[combined_df["dataset"] == dataset].copy()
        dataset_df = dataset_df.sort_values("overhead_s", ascending=True)
        print_dataframe(
            dataset_df[["algorithm", "category", "platform_time_s",
                         "direct_time_s", "overhead_s", "speedup", "f1"]]
        )

    return group_dfs


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PyOD anomaly-detection experiment as a terminal script.",
    )
    parser.add_argument(
        "--group",
        choices=["original", "classical", "deep_learning", "all"],
        default="all",
        help=(
            "Which model group to run. "
            "'original' = classic PyOD models (ABOD, ALAD, …), "
            "'classical' = newer classical models (COPOD, ECOD, …), "
            "'deep_learning' = neural-net models (AutoEncoder, VAE, …), "
            "'all' = run every group sequentially (default)."
        ),
    )
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "results",
        help="Directory where CSV results will be saved.",
    )
    parser.add_argument(
        "--max-train-normals", type=int, default=8000,
        help="Maximum number of normal training samples for shuttle.",
    )
    parser.add_argument(
        "--max-test-size", type=int, default=5000,
        help="Maximum test-set size for shuttle.",
    )
    parser.add_argument(
        "--target-contamination", type=float, default=0.1,
        help="Target test contamination ratio.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Running group(s): {args.group}")
    group_dfs = run_experiment(args)

    # Save one CSV per executed group
    for group_name, df in group_dfs.items():
        csv_name = GROUP_REGISTRY[group_name][3]  # filename from registry
        csv_path = args.results_dir / csv_name
        df.to_csv(csv_path, index=False)
        print(f"Saved {group_name} results to: {csv_path}")

    # If more than one group was run, also save a combined CSV
    if len(group_dfs) > 1:
        combined = _sort_results(
            pd.concat(group_dfs.values(), ignore_index=True)
        )
        all_path = args.results_dir / "uci_pyod_all_models_results.csv"
        combined.to_csv(all_path, index=False)
        print(f"Saved combined results to:    {all_path}")

    print(f"\nAll CSVs include:")
    print("   - Metrics: accuracy, precision, recall, f1, roc_auc, pr_auc")
    print("   - Timing: platform_time_s, direct_time_s, overhead_s, speedup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
